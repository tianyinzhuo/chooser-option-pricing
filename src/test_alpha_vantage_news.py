"""Test the Alpha Vantage JPM news-sentiment endpoint with one request."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "news_sentiment"
OUTPUT_PATH = OUTPUT_DIR / "jpm_news_sentiment_test_2024.json"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY was not found in the .env file.")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "JPM",
        "time_from": "20240101T0000",
        "time_to": "20241231T2359",
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

    feed = payload.get("feed")
    if not isinstance(feed, list):
        raise RuntimeError(
            "The response did not contain a news feed. "
            f"Returned keys: {sorted(payload)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Alpha Vantage NEWS_SENTIMENT test completed.")
    print(f"Requested ticker: JPM")
    print(f"Requested period: 2024-01-01 to 2024-12-31")
    print(f"Articles returned: {len(feed)}")
    if feed:
        first = feed[0]
        jpm_detail = next(
            (
                item
                for item in first.get("ticker_sentiment", [])
                if item.get("ticker") == "JPM"
            ),
            {},
        )
        print(f"First article time: {first.get('time_published', '')}")
        print(f"First article title: {first.get('title', '')}")
        print(f"JPM relevance score: {jpm_detail.get('relevance_score', '')}")
        print(f"JPM sentiment score: {jpm_detail.get('ticker_sentiment_score', '')}")
    if len(feed) == 1000:
        print("Warning: the result reached the 1000-article limit; use smaller date windows.")
    print(f"Raw JSON saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
