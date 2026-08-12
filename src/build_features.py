from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

input_path = PROCESSED_DIR / "aligned_clean_data.csv"
data = pd.read_csv(input_path, parse_dates=["date"])
data = data.sort_values("date").reset_index(drop=True)

# 1. 价格、收益率与成交量特征
data["daily_return"] = data["close"].pct_change()
data["log_return"] = np.log(data["close"] / data["close"].shift(1))
data["intraday_return"] = (data["close"] - data["open"]) / data["open"]
data["high_low_range"] = (data["high"] - data["low"]) / data["close"]
data["volume_change"] = data["volume"].pct_change()
data["volume_ma_20"] = data["volume"].rolling(20).mean()

# 2. 趋势与历史波动率特征
data["close_ma_20"] = data["close"].rolling(20).mean()
data["close_ma_60"] = data["close"].rolling(60).mean()
data["rolling_vol_20"] = data["daily_return"].rolling(20).std() * np.sqrt(252)
data["rolling_vol_60"] = data["daily_return"].rolling(60).std() * np.sqrt(252)

# 3. VIX 与利率特征
data["vix_change"] = data["vix"].pct_change()
data["vix_ma_20"] = data["vix"].rolling(20).mean()
data["treasury_10y_change"] = data["treasury_10y"].diff()
data["treasury_10y_momentum_20"] = (
    data["treasury_10y"] - data["treasury_10y"].shift(20)
)

# 4. JPM 收益率与 VIX 变化的滚动相关性
data["jpm_vix_corr_20"] = (
    data["daily_return"]
    .rolling(20)
    .corr(data["vix_change"])
)

# 5. 删除滚动窗口尚未形成的记录，不使用未来数据回填
feature_columns = [
    "daily_return",
    "log_return",
    "intraday_return",
    "high_low_range",
    "volume_change",
    "volume_ma_20",
    "close_ma_20",
    "close_ma_60",
    "rolling_vol_20",
    "rolling_vol_60",
    "vix_change",
    "vix_ma_20",
    "treasury_10y_change",
    "treasury_10y_momentum_20",
    "jpm_vix_corr_20",
]

feature_data = data.dropna(subset=feature_columns).copy()

# 6. 输出结构化数据集
csv_path = PROCESSED_DIR / "feature_dataset_2018_2024.csv"
parquet_path = PROCESSED_DIR / "feature_dataset_2018_2024.parquet"

feature_data.to_csv(csv_path, index=False)
feature_data.to_parquet(parquet_path, index=False)

print(f"输入行数：{len(data)}")
print(f"特征完成后行数：{len(feature_data)}")
print(
    f"特征数据日期范围："
    f"{feature_data['date'].min().date()} 至 "
    f"{feature_data['date'].max().date()}"
)
print(f"特征数量：{len(feature_columns)}")
print("特征列表：")
for column in feature_columns:
    print(f"  - {column}")
print(f"CSV 已保存：{csv_path}")
print(f"Parquet 已保存：{parquet_path}")