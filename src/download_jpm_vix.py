from pathlib import Path
import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
JPM_END_DATE = "2025-01-01"

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("TWELVE_DATA_API_KEY")
if not api_key:
    raise ValueError(
        "TWELVE_DATA_API_KEY is missing. Check your .env file or GitHub secret."
    )


def fetch_json(url: str) -> dict:
    """Fetch and parse a JSON response without relying on system curl."""
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Download an unmodified FRED daily series and limit it to the project period."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    data = pd.read_csv(url, parse_dates=["observation_date"])

    return data[
        (data["observation_date"] >= START_DATE)
        & (data["observation_date"] <= END_DATE)
    ]


def collect_jpm() -> pd.DataFrame:
    """Download raw JPM daily OHLCV data from Twelve Data."""
    params = {
        "symbol": "JPM",
        "interval": "1day",
        "start_date": START_DATE,
        "end_date": JPM_END_DATE,
        "adjust": "none",
        "apikey": api_key,
    }

    url = "https://api.twelvedata.com/time_series?" + urlencode(params)
    response = fetch_json(url)

    if response.get("status") == "error" or "values" not in response:
        message = response.get("message", "Unknown Twelve Data response.")
        raise RuntimeError(f"Twelve Data request failed: {message}")

    data = pd.DataFrame(response["values"])
    data = data.rename(columns={"datetime": "date"})

    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date")


def main() -> None:
    jpm = collect_jpm()
    vix = fetch_fred_series("VIXCLS")
    treasury = fetch_fred_series("DGS10")

    jpm.to_csv(RAW_DATA_DIR / "jpm_daily_2018_2024.csv", index=False)
    vix.to_csv(RAW_DATA_DIR / "vix_daily_2018_2024.csv", index=False)
    treasury.to_csv(
        RAW_DATA_DIR / "treasury_10y_daily_2018_2024.csv",
        index=False,
    )

    print(f"JPM rows: {len(jpm)}")
    print(f"VIX rows: {len(vix)}")
    print(f"Treasury rows: {len(treasury)}")
    print(f"Raw data saved to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()