from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

files = {
    "JPM 股价": ("jpm_daily_2018_2024.csv", "date"),
    "VIX": ("vix_daily_2018_2024.csv", "observation_date"),
    "10年期美国国债利率": (
        "treasury_10y_daily_2018_2024.csv",
        "observation_date",
    ),
}

for name, (filename, date_column) in files.items():
    data = pd.read_csv(RAW_DATA_DIR / filename)
    data[date_column] = pd.to_datetime(data[date_column])

    print(f"\n{name}")
    print(f"  行数：{len(data)}")
    print(f"  日期范围：{data[date_column].min().date()} 至 "
          f"{data[date_column].max().date()}")
    print(f"  字段：{', '.join(data.columns)}")
    print("  各字段缺失值：")
    print(data.isna().sum().to_string().replace("\n", "\n    "))