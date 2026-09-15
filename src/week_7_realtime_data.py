"""Build a fault-tolerant daily market snapshot for the Week 7 prototype.

The module requests the latest available JPM daily bar from Twelve Data, the
latest valid VIXCLS and DGS10 observations from FRED, and (when an Alpha
Vantage key is configured) a short JPM news-sentiment summary.  Every source
is collected independently: a failed or unconfigured source is represented by
an explicit status and ``null`` data instead of a fabricated zero.

Run from the project root with::

    python src/week_7_realtime_data.py
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import copy
import os
import re
import shutil
import socket
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:  # Environment variables still work without python-dotenv.
    load_dotenv = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "realtime"
    / "latest_market_snapshot.json"
)
USER_AGENT = "chooser-option-pricing-week-7/1.0"
FRED_LOOKBACK_DAYS = 90


class DataSourceError(RuntimeError):
    """An expected, safely reportable data-source failure."""


def utc_now() -> datetime:
    """Return an aware UTC timestamp; isolated to make testing straightforward."""
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    """Format an aware timestamp using an explicit UTC ``Z`` suffix."""
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _safe_error_message(error: BaseException) -> str:
    """Create a concise error message without exposing request URLs/API keys."""
    if isinstance(error, HTTPError):
        return f"HTTP {error.code}: {error.reason or 'request failed'}"
    if isinstance(error, URLError):
        return f"Network error: {error.reason}"
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "Request timed out"
    if isinstance(error, json.JSONDecodeError):
        return "Provider returned invalid JSON"
    if isinstance(error, UnicodeDecodeError):
        return "Provider returned undecodable text"
    message = str(error) or error.__class__.__name__
    message = re.sub(
        r"(?i)(api\s*key(?:\s+as)?\s+)[A-Za-z0-9_-]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)(apikey=)[^&\s]+",
        r"\1[REDACTED]",
        message,
    )
    return message


def request_bytes(url: str, timeout: float, retries: int) -> bytes:
    """Request a URL with a small exponential-backoff retry budget."""
    attempts = max(1, retries + 1)
    last_error: Optional[BaseException] = None

    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as error:
            last_error = error
            # Authentication and most client errors will not improve on retry.
            if 400 <= error.code < 500 and error.code != 429:
                break
        except (URLError, TimeoutError, socket.timeout, OSError) as error:
            last_error = error

        if attempt < attempts - 1:
            time.sleep(min(2.0**attempt, 4.0))

    assert last_error is not None
    raise DataSourceError(_safe_error_message(last_error)) from last_error


def request_json(url: str, timeout: float, retries: int) -> Dict[str, Any]:
    """Request and validate a JSON object."""
    try:
        payload = json.loads(request_bytes(url, timeout, retries).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DataSourceError(_safe_error_message(error)) from error
    if not isinstance(payload, dict):
        raise DataSourceError("Provider returned JSON that is not an object")
    return payload


def request_bytes_with_curl_fallback(
    url: str, timeout: float, retries: int
) -> tuple[bytes, str]:
    """Request an official URL with urllib, then the system curl client.

    This addresses networks that allow ``curl.exe`` while terminating Python's
    TLS connection.  Error messages intentionally omit the URL so credentials
    cannot leak if the helper is reused for a keyed endpoint.
    """
    # Treat ``retries`` as a total retry budget shared by the two transports.
    # With the Streamlit default (retries=1), this means one urllib attempt and
    # one curl attempt instead of four full timeout periods.
    try:
        return request_bytes(url, timeout, 0), "urllib"
    except DataSourceError as urllib_error:
        curl_path = shutil.which("curl.exe") or shutil.which("curl")
        if not curl_path:
            raise DataSourceError(
                f"urllib failed ({urllib_error}); system curl is unavailable"
            ) from urllib_error

        last_message = "curl request failed"
        curl_attempts = max(1, retries)
        for attempt in range(curl_attempts):
            command = [
                curl_path,
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--connect-timeout",
                str(max(1, int(timeout))),
                "--max-time",
                str(max(1, int(timeout))),
                "-A",
                USER_AGENT,
                url,
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=timeout + 5,
                )
                if completed.returncode == 0 and completed.stdout:
                    return completed.stdout, "curl"
                last_message = f"curl exited with code {completed.returncode}"
            except subprocess.TimeoutExpired:
                last_message = "curl request timed out"
            except OSError as error:
                last_message = _safe_error_message(error)
            if attempt < curl_attempts - 1:
                time.sleep(min(2.0**attempt, 4.0))

        raise DataSourceError(
            f"urllib failed ({urllib_error}); {last_message}"
        ) from urllib_error


def _to_float(value: Any) -> Optional[float]:
    """Convert a provider value to float while preserving missingness."""
    if value in (None, "", ".", "null", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    """Convert a numeric provider value to an integer when possible."""
    numeric = _to_float(value)
    return None if numeric is None else int(numeric)


def _base_result(provider: str, retrieved_at: str) -> Dict[str, Any]:
    return {
        "status": "ok",
        "provider": provider,
        "retrieved_at_utc": retrieved_at,
        "data": None,
        "message": None,
    }


def parse_fred_csv_latest(text: str, series_id: str) -> Dict[str, Any]:
    """Parse the latest valid observation from a FRED graph CSV payload."""
    rows: List[Dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(text)):
        date_value = row.get("observation_date") or row.get("DATE")
        numeric_value = _to_float(row.get(series_id))
        if date_value and numeric_value is not None:
            rows.append({"date": str(date_value)[:10], "value": numeric_value})
    if not rows:
        raise DataSourceError(f"FRED returned no valid {series_id} observations")
    return max(rows, key=lambda item: str(item["date"]))


def _normalise_date(value: str) -> Optional[str]:
    value = value.strip()[:19]
    for date_format in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, date_format).date().isoformat()
        except ValueError:
            continue
    return None


def parse_cboe_vix_csv_latest(text: str) -> Dict[str, Any]:
    """Parse the latest VIX close from Cboe's official history CSV."""
    observations: List[Dict[str, Any]] = []
    for raw_row in csv.DictReader(io.StringIO(text)):
        row = {str(key).strip().upper(): value for key, value in raw_row.items()}
        date_value = _normalise_date(str(row.get("DATE", "")))
        close_value = _to_float(row.get("CLOSE"))
        if date_value and close_value is not None:
            observations.append({"date": date_value, "value": close_value})
    if not observations:
        raise DataSourceError("Cboe returned no valid VIX observations")
    return max(observations, key=lambda item: str(item["date"]))


def parse_local_vix_csv_latest(path: Path) -> Dict[str, Any]:
    """Read the project's historical VIX archive as an explicitly stale backup."""
    if not path.exists():
        raise DataSourceError("Local historical VIX archive is unavailable")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise DataSourceError(_safe_error_message(error)) from error
    return parse_fred_csv_latest(text, "VIXCLS")


def parse_treasury_10y_xml_latest(text: str) -> Dict[str, Any]:
    """Parse NEW_DATE and BC_10YEAR pairs from official Treasury XML."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise DataSourceError("U.S. Treasury returned invalid XML") from error

    observations: List[Dict[str, Any]] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].upper() != "PROPERTIES":
            continue
        fields = {
            child.tag.rsplit("}", 1)[-1].upper(): (child.text or "").strip()
            for child in element
        }
        date_value = _normalise_date(fields.get("NEW_DATE", ""))
        rate_value = _to_float(fields.get("BC_10YEAR"))
        if date_value and rate_value is not None:
            observations.append({"date": date_value, "value": rate_value})
    if not observations:
        raise DataSourceError("U.S. Treasury returned no valid 10-year yields")
    return max(observations, key=lambda item: str(item["date"]))


def parse_fred_history_html_latest(text: str, series_id: str) -> Dict[str, Any]:
    """Parse visible ``YYYY-MM-DD: value`` pairs from a FRED history page."""
    visible = re.sub(r"(?is)<(?:script|style)\b.*?</(?:script|style)>", " ", text)
    visible = html.unescape(re.sub(r"(?s)<[^>]+>", " ", visible))
    visible = re.sub(r"\s+", " ", visible)
    pattern = re.compile(
        r"(?P<date>\d{4}-\d{2}-\d{2})\s*(?::|\||\s)\s*"
        r"(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    )
    rows = [
        {
            "date": match.group("date"),
            "value": float(match.group("value").replace(",", "")),
        }
        for match in pattern.finditer(visible)
    ]
    if not rows:
        raise DataSourceError(
            f"FRED history page returned no valid {series_id} observations"
        )
    return max(rows, key=lambda item: str(item["date"]))


def collect_official_series_fallback(
    series_id: str,
    reference: datetime,
    timeout: float,
    retries: int,
) -> tuple[Dict[str, Any], str, str, str]:
    """Use an independent official publisher, then FRED history as last resort."""
    candidates: List[tuple[str, str, Callable[[str], Dict[str, Any]], str]] = []
    if series_id == "VIXCLS":
        candidates.extend(
            [
                (
                    "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
                    "cboe_vix_history_csv",
                    parse_cboe_vix_csv_latest,
                    "Cboe Global Markets",
                ),
                (
                    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
                    "cboe_vix_history_csv_mirror",
                    parse_cboe_vix_csv_latest,
                    "Cboe Global Markets",
                ),
            ]
        )
    elif series_id == "DGS10":
        month = reference.astimezone(timezone.utc).strftime("%Y%m")
        year = reference.astimezone(timezone.utc).strftime("%Y")
        base = (
            "https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/pages/xml?data=daily_treasury_yield_curve&"
        )
        candidates.extend(
            [
                (
                    base + "field_tdr_date_value_month=" + month,
                    "treasury_yield_curve_xml_current_month",
                    parse_treasury_10y_xml_latest,
                    "U.S. Department of the Treasury",
                ),
                (
                    base + "field_tdr_date_value=" + year,
                    "treasury_yield_curve_xml_current_year",
                    parse_treasury_10y_xml_latest,
                    "U.S. Department of the Treasury",
                ),
            ]
        )

    errors: List[str] = []
    for url, endpoint, parser, provider in candidates:
        try:
            payload, method = request_bytes_with_curl_fallback(url, timeout, retries)
            return (
                parser(payload.decode("utf-8-sig", errors="replace")),
                endpoint,
                method,
                provider,
            )
        except DataSourceError as error:
            errors.append(_safe_error_message(error))

    history_url = f"https://fred.stlouisfed.org/series/{series_id}/history"
    try:
        payload, method = request_bytes_with_curl_fallback(
            history_url, timeout, retries
        )
        return (
            parse_fred_history_html_latest(
                payload.decode("utf-8", errors="replace"), series_id
            ),
            "fred_series_history_html",
            method,
            "Federal Reserve Bank of St. Louis (FRED)",
        )
    except DataSourceError as error:
        errors.append(_safe_error_message(error))
        raise DataSourceError(
            "All official fallback endpoints failed: " + "; ".join(errors)
        ) from error


def collect_twelve_data_jpm(
    api_key: Optional[str], timeout: float, retries: int, retrieved_at: str
) -> Dict[str, Any]:
    """Collect the latest complete JPM daily OHLCV observation."""
    result = _base_result("Twelve Data", retrieved_at)
    if not api_key:
        result.update(
            status="not_configured",
            message="TWELVE_DATA_API_KEY is not configured.",
        )
        return result

    params = {
        "symbol": "JPM",
        "interval": "1day",
        "outputsize": "10",
        "order": "DESC",
        "timezone": "America/New_York",
        "adjust": "none",
        "apikey": api_key,
    }
    url = "https://api.twelvedata.com/time_series?" + urlencode(params)
    payload = request_json(url, timeout, retries)

    if payload.get("status") == "error":
        raise DataSourceError(
            "Twelve Data error: " + str(payload.get("message", "unknown error"))
        )
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise DataSourceError("Twelve Data returned no JPM daily observations")

    for row in values:
        if not isinstance(row, dict):
            continue
        observation = {
            "date": row.get("datetime"),
            "open": _to_float(row.get("open")),
            "high": _to_float(row.get("high")),
            "low": _to_float(row.get("low")),
            "close": _to_float(row.get("close")),
            "volume": _to_int(row.get("volume")),
            "price_unit": "USD",
            "volume_unit": "shares",
        }
        required = ("date", "open", "high", "low", "close")
        if all(observation[field] is not None for field in required):
            result["data"] = observation
            result["symbol"] = "JPM"
            result["frequency"] = "daily"
            return result

    raise DataSourceError("Twelve Data returned no complete JPM daily bar")


def collect_fred_latest(
    series_id: str,
    display_name: str,
    unit: str,
    timeout: float,
    retries: int,
    retrieved_at: str,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Collect a series from a short FRED window with official fallbacks."""
    result = _base_result("Federal Reserve Bank of St. Louis (FRED)", retrieved_at)
    if not re.fullmatch(r"[A-Z0-9_-]+", series_id):
        raise ValueError("Invalid FRED series id")

    reference = as_of or utc_now()
    end_date = reference.astimezone(timezone.utc).date()
    start_date = end_date - timedelta(days=FRED_LOOKBACK_DAYS)
    params = {
        "id": series_id,
        "cosd": start_date.isoformat(),
        "coed": end_date.isoformat(),
    }
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode(params)
    try:
        payload, method = request_bytes_with_curl_fallback(url, timeout, retries)
        latest = parse_fred_csv_latest(
            payload.decode("utf-8-sig"), series_id
        )
        endpoint = "fredgraph_csv_recent_window"
        data_provider = "Federal Reserve Bank of St. Louis (FRED)"
        fallback_used = method != "urllib"
    except (DataSourceError, UnicodeDecodeError):
        latest, endpoint, method, data_provider = collect_official_series_fallback(
            series_id, reference, timeout, retries
        )
        fallback_used = True

    result.update(
        series_id=series_id,
        series_name=display_name,
        frequency="daily",
        endpoint=endpoint,
        request_method=method,
        fallback_used=fallback_used,
        data_provider=data_provider,
        requested_window={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "lookback_days": FRED_LOOKBACK_DAYS,
        },
        data={
            "date": latest["date"],
            "value": latest["value"],
            "unit": unit,
        },
    )
    return result


def collect_vix_with_local_archive(
    timeout: float,
    retries: int,
    retrieved_at: str,
    project_root: Path,
) -> Dict[str, Any]:
    """Collect current VIX data, falling back to the dated project archive.

    The local value is labelled ``stale`` and retains its observation date; it
    is never represented as a fresh market observation.
    """
    try:
        return collect_fred_latest(
            "VIXCLS",
            "CBOE Volatility Index: VIX",
            "index points",
            timeout,
            retries,
            retrieved_at,
        )
    except Exception as live_error:
        latest = parse_local_vix_csv_latest(
            project_root / "data" / "raw" / "vix_daily_2018_2024.csv"
        )
        result = _base_result("Project historical archive", retrieved_at)
        result.update(
            status="stale",
            series_id="VIXCLS",
            series_name="CBOE Volatility Index: VIX",
            frequency="daily",
            endpoint="data/raw/vix_daily_2018_2024.csv",
            request_method="local_file",
            fallback_used=True,
            data_provider="Local project archive (original source: FRED)",
            message=(
                "Current VIX sources were unavailable; displaying the dated "
                "project archive value. Live-source error: "
                + _safe_error_message(live_error)
            ),
            data={
                "date": latest["date"],
                "value": latest["value"],
                "unit": "index points",
            },
        )
        return result


def _ticker_sentiment(article: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    details = article.get("ticker_sentiment", [])
    if not isinstance(details, list):
        return None
    for detail in details:
        if isinstance(detail, dict) and str(detail.get("ticker", "")).upper() == ticker:
            return detail
    return None


def collect_alpha_vantage_news(
    api_key: Optional[str],
    lookback_days: int,
    timeout: float,
    retries: int,
    now: datetime,
    retrieved_at: str,
) -> Dict[str, Any]:
    """Collect a recent JPM news-sentiment summary when a key is available."""
    result = _base_result("Alpha Vantage", retrieved_at)
    result["optional_source"] = True
    if not api_key:
        result.update(
            status="not_configured",
            message="ALPHA_VANTAGE_API_KEY is not configured; news was skipped.",
        )
        return result

    start = now - timedelta(days=max(1, lookback_days))
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "JPM",
        "time_from": start.strftime("%Y%m%dT%H%M"),
        "time_to": now.strftime("%Y%m%dT%H%M"),
        "sort": "LATEST",
        "limit": "200",
        "apikey": api_key,
    }
    url = "https://www.alphavantage.co/query?" + urlencode(params)
    payload = request_json(url, timeout, retries)
    for key in ("Error Message", "Information", "Note"):
        if key in payload:
            provider_message = str(payload[key])
            if "rate limit" in provider_message.lower():
                raise DataSourceError(
                    "Alpha Vantage daily request limit reached; the last "
                    "successful same-day snapshot will be reused when available."
                )
            raise DataSourceError(
                "Alpha Vantage response: "
                + _safe_error_message(DataSourceError(provider_message))
            )

    feed = payload.get("feed")
    if not isinstance(feed, list):
        raise DataSourceError("Alpha Vantage returned no news feed")

    observations: List[Dict[str, Any]] = []
    for article in feed:
        if not isinstance(article, dict):
            continue
        detail = _ticker_sentiment(article, "JPM")
        if detail is None:
            continue
        score = _to_float(detail.get("ticker_sentiment_score"))
        relevance = _to_float(detail.get("relevance_score"))
        if score is None:
            continue
        observations.append(
            {
                "score": score,
                "relevance": relevance,
                "time_published": article.get("time_published"),
            }
        )

    window = {
        "from_utc": iso_utc(start),
        "to_utc": iso_utc(now),
        "lookback_days": max(1, lookback_days),
    }
    if not observations:
        result.update(
            status="no_data",
            message="No usable JPM sentiment observations were returned for the window.",
            frequency="event_based",
            data={
                "window": window,
                "article_count": 0,
                "mean_sentiment_score": None,
                "relevance_weighted_sentiment_score": None,
                "positive_share": None,
                "negative_share": None,
                "neutral_share": None,
                "latest_article_time_utc": None,
                "score_unit": "Alpha Vantage ticker sentiment score [-1, 1]",
            },
        )
        return result

    scores = [item["score"] for item in observations]
    weighted_items = [
        item
        for item in observations
        if item["relevance"] is not None and item["relevance"] > 0
    ]
    total_relevance = sum(item["relevance"] for item in weighted_items)
    weighted_score = (
        sum(item["score"] * item["relevance"] for item in weighted_items)
        / total_relevance
        if total_relevance > 0
        else None
    )
    count = len(scores)
    # Alpha Vantage's ticker score labels use +/-0.15 as the neutral interval.
    positive = sum(score > 0.15 for score in scores)
    negative = sum(score < -0.15 for score in scores)
    neutral = count - positive - negative
    published = [
        str(item["time_published"])
        for item in observations
        if item["time_published"]
    ]

    result.update(
        frequency="event_based",
        data={
            "window": window,
            "article_count": count,
            "mean_sentiment_score": sum(scores) / count,
            "relevance_weighted_sentiment_score": weighted_score,
            "positive_share": positive / count,
            "negative_share": negative / count,
            "neutral_share": neutral / count,
            "latest_article_time_utc": max(published) if published else None,
            "score_unit": "Alpha Vantage ticker sentiment score [-1, 1]",
        },
    )
    return result


def run_source(
    provider: str, retrieved_at: str, collector: Callable[[], Dict[str, Any]]
) -> Dict[str, Any]:
    """Protect the overall snapshot from one provider's failure."""
    try:
        return collector()
    except Exception as error:  # Boundary intentionally isolates provider failures.
        return {
            "status": "error",
            "provider": provider,
            "retrieved_at_utc": retrieved_at,
            "data": None,
            "message": _safe_error_message(error),
        }


def build_snapshot(
    timeout: float,
    retries: int,
    news_lookback_days: int,
    project_root: Path | None = None,
    cached_news: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect every configured source and return a serializable snapshot."""
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if retries < 0:
        raise ValueError("retries cannot be negative")

    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    now = utc_now()
    retrieved_at = iso_utc(now)
    project = project_root or PROJECT_ROOT
    twelve_key = os.getenv("TWELVE_DATA_API_KEY")
    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    sources = {
        "jpm_daily": run_source(
            "Twelve Data",
            retrieved_at,
            lambda: collect_twelve_data_jpm(
                twelve_key, timeout, retries, retrieved_at
            ),
        ),
        "vix": run_source(
            "Federal Reserve Bank of St. Louis (FRED)",
            retrieved_at,
            lambda: collect_vix_with_local_archive(
                timeout,
                retries,
                retrieved_at,
                project,
            ),
        ),
        "treasury_10y": run_source(
            "Federal Reserve Bank of St. Louis (FRED)",
            retrieved_at,
            lambda: collect_fred_latest(
                "DGS10",
                "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
                "percent per annum",
                timeout,
                retries,
                retrieved_at,
            ),
        ),
        "jpm_news_sentiment": cached_news or run_source(
            "Alpha Vantage",
            retrieved_at,
            lambda: collect_alpha_vantage_news(
                alpha_key,
                news_lookback_days,
                timeout,
                retries,
                now,
                retrieved_at,
            ),
        ),
    }

    statuses = {name: item["status"] for name, item in sources.items()}
    return {
        "schema_version": "1.0",
        "generated_at_utc": retrieved_at,
        "snapshot_frequency": "daily",
        "sources": sources,
        "quality_summary": {
            "source_statuses": statuses,
            "successful_source_count": sum(
                status == "ok" for status in statuses.values()
            ),
            "failed_source_count": sum(
                status == "error" for status in statuses.values()
            ),
            "not_configured_source_count": sum(
                status == "not_configured" for status in statuses.values()
            ),
            "cached_source_count": sum(
                status == "cached" for status in statuses.values()
            ),
            "stale_source_count": sum(
                status == "stale" for status in statuses.values()
            ),
            "no_data_source_count": sum(
                status == "no_data" for status in statuses.values()
            ),
        },
        "disclosures": [
            "This is a latest-available daily snapshot, not tick-level real-time data.",
            "Provider publication times differ, so observation dates may not match.",
            "Missing or failed source values remain null and are never replaced with zero.",
            "News sentiment is descriptive and must not be interpreted as a causal signal.",
        ],
    }


def save_snapshot(snapshot: Dict[str, Any], output_path: Path) -> None:
    """Atomically replace the latest snapshot JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(output_path)


def _same_day_cached_news(
    previous: Optional[Dict[str, Any]], now: datetime
) -> Optional[Dict[str, Any]]:
    """Reuse one successful Alpha Vantage result per UTC day.

    This prevents a dashboard refresh from repeatedly spending the provider's
    small free daily request allowance.
    """
    if not previous or not isinstance(previous.get("sources"), dict):
        return None
    old = previous["sources"].get("jpm_news_sentiment")
    if not isinstance(old, dict):
        return None

    if old.get("status") in {"ok", "cached"} and old.get("data") is not None:
        data = old["data"]
        retrieved_at = old.get("retrieved_at_utc") or old.get("cached_from_utc")
    else:
        data = old.get("last_successful_data")
        retrieved_at = old.get("last_successful_retrieved_at_utc")
    if data is None or not retrieved_at:
        return None
    try:
        retrieved = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if retrieved.astimezone(timezone.utc).date() != now.astimezone(timezone.utc).date():
        return None

    return {
        "status": "cached",
        "provider": "Alpha Vantage",
        "data_provider": "Alpha Vantage (same-day cached result)",
        "retrieved_at_utc": iso_utc(now),
        "cached_from_utc": iso_utc(retrieved),
        "endpoint": "same_day_snapshot_cache",
        "request_method": "local_file",
        "data": copy.deepcopy(data),
        "message": "Reused today's successful news snapshot to protect the daily API quota.",
        "frequency": "event_based",
        "optional_source": True,
    }


def collect_realtime_snapshot(
    project_root: str | Path | None = None,
    save: bool = True,
    timeout: float = 30.0,
    retries: int = 2,
    news_lookback_days: int = 7,
) -> Dict[str, Any]:
    """Application-friendly wrapper used by the Streamlit refresh button.

    The word ``realtime`` is retained in the public function name for project
    consistency, while the returned metadata explicitly labels the data as a
    latest-available *daily* snapshot rather than tick-level real-time data.
    """
    project = Path(project_root).resolve() if project_root else PROJECT_ROOT
    if load_dotenv is not None:
        load_dotenv(project / ".env")
    output_path = (
        project
        / "data"
        / "external"
        / "realtime"
        / "latest_market_snapshot.json"
    )
    previous: Optional[Dict[str, Any]] = None
    if output_path.exists():
        try:
            loaded = json.loads(output_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            previous = None

    collection_time = utc_now()
    cached_news = _same_day_cached_news(previous, collection_time)
    snapshot = build_snapshot(
        timeout,
        retries,
        news_lookback_days,
        project_root=project,
        cached_news=cached_news,
    )
    # Preserve the last successful payload separately when a provider is
    # temporarily unavailable.  ``data`` remains null, so stale data is never
    # presented as a fresh observation.
    if previous and isinstance(previous.get("sources"), dict):
        for name, current in snapshot["sources"].items():
            old = previous["sources"].get(name, {})
            if current.get("status") in {"ok", "cached"} or not isinstance(old, dict):
                continue
            if old.get("status") in {"ok", "cached"} and old.get("data") is not None:
                last_data = old["data"]
                last_time = old.get("retrieved_at_utc") or old.get("cached_from_utc")
            else:
                last_data = old.get("last_successful_data")
                last_time = old.get("last_successful_retrieved_at_utc")
            if last_data is not None:
                current["last_successful_data"] = copy.deepcopy(last_data)
                current["last_successful_retrieved_at_utc"] = last_time
    if save:
        save_snapshot(snapshot, output_path)
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect the Week 7 latest-available daily market snapshot."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Snapshot JSON output path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each request (default: 30).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries after the initial request (default: 2).",
    )
    parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=7,
        help="Alpha Vantage news lookback window in days (default: 7).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = build_snapshot(args.timeout, args.retries, args.news_lookback_days)
    save_snapshot(snapshot, args.output.resolve())

    print("Week 7 latest-available daily snapshot completed.")
    for name, result in snapshot["sources"].items():
        message = f" | {result['message']}" if result.get("message") else ""
        print(f"{name}: {result['status']}{message}")
    print(f"Snapshot saved to: {args.output.resolve()}")
    print("Missing or failed source values remain null; they are not replaced with zero.")


if __name__ == "__main__":
    main()
