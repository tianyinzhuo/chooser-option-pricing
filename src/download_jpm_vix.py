from pathlib import Path
import json
import os
import subprocess
import shutil
from urllib.parse import urlencode

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CURL_BINARY = "CURL_BINARY," if shutil.which("curl.exe") else "curl"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

api_key = os.getenv("TWELVE_DATA_API_KEY")
if not api_key:
    raise ValueError("未找到 TWELVE_DATA_API_KEY，请检查 .env 文件。")


def download_file(url, output_path):
    """下载公开 CSV 文件。"""
    subprocess.run(
        [
            "curl.exe", "-L", "--fail", "--silent", "--show-error",
            "--max-time", "60", url, "-o", str(output_path),
        ],
        check=True,
    )


def get_json(url):
    """下载并解析 JSON 数据。"""
    result = subprocess.run(
        [
            "curl.exe", "-L", "--fail", "--silent", "--show-error",
            "--max-time", "60", url,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


# 1. JPM 每日 OHLCV：Twelve Data
jpm_params = {
    "symbol": "JPM",
    "interval": "1day",
    "start_date": START_DATE,
    "end_date": "2025-01-01",
    "adjust": "none",
    "apikey": api_key,
}
jpm_url = "https://api.twelvedata.com/time_series?" + urlencode(jpm_params)
jpm_json = get_json(jpm_url)

if jpm_json.get("status") == "error":
    raise RuntimeError(jpm_json.get("message", "Twelve Data API 返回错误。"))

jpm = pd.DataFrame(jpm_json["values"])
jpm = jpm.rename(columns={"datetime": "date"})

for column in ["open", "high", "low", "close", "volume"]:
    jpm[column] = pd.to_numeric(jpm[column], errors="coerce")

jpm["date"] = pd.to_datetime(jpm["date"])
jpm = jpm.sort_values("date")
jpm.to_csv(RAW_DATA_DIR / "jpm_daily_2018_2024.csv", index=False)

# 2. VIX：FRED
vix_raw_path = RAW_DATA_DIR / "_vix_fred_full_history.csv"
download_file(
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS",
    vix_raw_path,
)
vix = pd.read_csv(vix_raw_path, parse_dates=["observation_date"])
vix = vix[
    (vix["observation_date"] >= START_DATE)
    & (vix["observation_date"] <= END_DATE)
]
vix.to_csv(RAW_DATA_DIR / "vix_daily_2018_2024.csv", index=False)
vix_raw_path.unlink()

# 3. 10 年期美国国债利率：FRED
treasury_raw_path = RAW_DATA_DIR / "_treasury_10y_fred_full_history.csv"
download_file(
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
    treasury_raw_path,
)
treasury = pd.read_csv(treasury_raw_path, parse_dates=["observation_date"])
treasury = treasury[
    (treasury["observation_date"] >= START_DATE)
    & (treasury["observation_date"] <= END_DATE)
]
treasury.to_csv(
    RAW_DATA_DIR / "treasury_10y_daily_2018_2024.csv",
    index=False,
)
treasury_raw_path.unlink()

print(f"JPM: {len(jpm)} 行")
print(f"VIX: {len(vix)} 行")
print(f"10年期美国国债利率: {len(treasury)} 行")
print(f"已保存至：{RAW_DATA_DIR}")