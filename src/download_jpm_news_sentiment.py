"""Download and aggregate JPM news sentiment for 2018-2024."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "data" / "external" / "news_sentiment"
RAW_DIR = BASE_DIR / "raw"
ARTICLES_PATH = BASE_DIR / "jpm_news_articles_2018_2024.csv"
DAILY_PATH = BASE_DIR / "jpm_news_sentiment_daily_2018_2024.csv"
REQUEST_DELAY_SECONDS = 13


def build_windows() -> list[dict[str, str]]:
    """Create 14 non-overlapping half-year request windows."""
    windows = []
    for year in range(2018, 2025):
        windows.extend(
            [
                {
                    "name": f"{year}_h1",
                    "time_from": f"{year}0101T0000",
                    "time_to": f"{year}0630T2359",
                },
                {
                    "name": f"{year}_h2",
                    "time_from": f"{year}0701T0000",
                    "time_to": f"{year}1231T2359",
                },
            ]
        )
    return windows


def request_window(api_key: str, window: dict[str, str]) -> dict:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "JPM",
        "time_from": window["time_from"],
        "time_to": window["time_to"],
        "sort": "EARLIEST",
        "limit": "1000",
        "apikey": api_key,
    }
    url = "https://www.alphavantage.co/query?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "chooser-option-pricing/1.0"})
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"Alpha Vantage returned HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to Alpha Vantage: {error.reason}") from error

    for key in ("Error Message", "Information", "Note"):
        if key in payload:
            raise RuntimeError(f"Alpha Vantage response: {payload[key]}")
    if not isinstance(payload.get("feed"), list):
        raise RuntimeError(f"No feed in response. Returned keys: {sorted(payload)}")
    return payload


def extract_article(item: dict, window_name: str) -> dict:
    jpm_detail = next(
        (
            detail
            for detail in item.get("ticker_sentiment", [])
            if detail.get("ticker") == "JPM"
        ),
        {},
    )
    return {
        "request_window": window_name,
        "time_published": item.get("time_published", ""),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": item.get("source", ""),
        "overall_sentiment_score": item.get("overall_sentiment_score"),
        "overall_sentiment_label": item.get("overall_sentiment_label", ""),
        "jpm_relevance_score": jpm_detail.get("relevance_score"),
        "jpm_sentiment_score": jpm_detail.get("ticker_sentiment_score"),
        "jpm_sentiment_label": jpm_detail.get("ticker_sentiment_label", ""),
    }


def aggregate_daily(articles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level JPM sentiment to UTC calendar dates."""
    data = articles.copy()
    data["timestamp_utc"] = pd.to_datetime(
        data["time_published"], format="%Y%m%dT%H%M%S", utc=True, errors="coerce"
    )
    data["date"] = data["timestamp_utc"].dt.date
    data["jpm_sentiment_score"] = pd.to_numeric(
        data["jpm_sentiment_score"], errors="coerce"
    )
    data["jpm_relevance_score"] = pd.to_numeric(
        data["jpm_relevance_score"], errors="coerce"
    )
    data = data.dropna(subset=["date", "jpm_sentiment_score"])

    label = data["jpm_sentiment_label"].fillna("").str.lower()
    data["positive"] = label.str.contains("bullish", regex=False).astype(float)
    data["negative"] = label.str.contains("bearish", regex=False).astype(float)
    data["neutral"] = label.str.contains("neutral", regex=False).astype(float)
    data["weighted_score"] = (
        data["jpm_sentiment_score"] * data["jpm_relevance_score"].fillna(0.0)
    )

    daily = (
        data.groupby("date", as_index=False)
        .agg(
            article_count=("jpm_sentiment_score", "size"),
            sentiment_mean=("jpm_sentiment_score", "mean"),
            sentiment_median=("jpm_sentiment_score", "median"),
            sentiment_std=("jpm_sentiment_score", "std"),
            relevance_mean=("jpm_relevance_score", "mean"),
            weighted_score_sum=("weighted_score", "sum"),
            relevance_sum=("jpm_relevance_score", "sum"),
            positive_share=("positive", "mean"),
            negative_share=("negative", "mean"),
            neutral_share=("neutral", "mean"),
        )
        .sort_values("date")
    )
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    daily["sentiment_weighted_mean"] = np.divide(
        daily["weighted_score_sum"],
        daily["relevance_sum"],
        out=np.zeros(len(daily), dtype=float),
        where=daily["relevance_sum"].to_numpy() != 0,
    )
    return daily.drop(columns=["weighted_score_sum", "relevance_sum"])


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY was not found in the .env file.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    windows = build_windows()
    article_records: list[dict] = []
    limit_windows: list[str] = []
    actual_requests = 0

    for index, window in enumerate(windows, start=1):
        raw_path = RAW_DIR / f"jpm_news_{window['name']}.json"
        if raw_path.is_file():
            print(f"[{index:02d}/{len(windows)}] {window['name']}: cached")
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            if actual_requests:
                time.sleep(REQUEST_DELAY_SECONDS)
            print(f"[{index:02d}/{len(windows)}] {window['name']}: requesting")
            payload = request_window(api_key, window)
            raw_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            actual_requests += 1

        feed = payload.get("feed", [])
        print(f"             articles: {len(feed)}")
        if len(feed) >= 1000:
            limit_windows.append(window["name"])
        article_records.extend(extract_article(item, window["name"]) for item in feed)

    articles = pd.DataFrame(article_records)
    if articles.empty:
        raise RuntimeError("No JPM news articles were returned for 2018-2024.")

    articles = (
        articles.drop_duplicates(subset=["url", "time_published", "title"])
        .sort_values("time_published")
        .reset_index(drop=True)
    )
    daily = aggregate_daily(articles)
    articles.to_csv(ARTICLES_PATH, index=False, encoding="utf-8-sig")
    daily.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")

    print()
    print("JPM news-sentiment collection completed.")
    print(f"API requests made in this run: {actual_requests}")
    print(f"Unique articles: {len(articles)}")
    print(f"Daily sentiment rows: {len(daily)}")
    if not daily.empty:
        print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"Article-level CSV: {ARTICLES_PATH}")
    print(f"Daily CSV: {DAILY_PATH}")
    if limit_windows:
        print("WARNING: these windows reached 1000 articles and may be truncated:")
        for name in limit_windows:
            print(f"  - {name}")
        print("Do not treat the affected windows as complete until they are subdivided.")
    else:
        print("No half-year window reached the 1000-article limit.")
    print("Daily dates use UTC. A one-trading-day lag will be added before modeling.")


if __name__ == "__main__":
    main()
