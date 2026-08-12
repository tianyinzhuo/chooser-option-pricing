from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def iqr_outlier_flag(series):
    """使用 IQR 规则标记异常值，不直接删除金融市场中的真实极端行情。"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return ((series < lower_bound) | (series > upper_bound)).astype(int)


# 1. 读取三份原始数据
jpm = pd.read_csv(RAW_DIR / "jpm_daily_2018_2024.csv", parse_dates=["date"])
vix = pd.read_csv(
    RAW_DIR / "vix_daily_2018_2024.csv",
    parse_dates=["observation_date"],
)
treasury = pd.read_csv(
    RAW_DIR / "treasury_10y_daily_2018_2024.csv",
    parse_dates=["observation_date"],
)

# 2. 统一日期字段和变量名
vix = vix.rename(
    columns={
        "observation_date": "date",
        "VIXCLS": "vix",
    }
)
treasury = treasury.rename(
    columns={
        "observation_date": "date",
        "DGS10": "treasury_10y",
    }
)

# 3. 清除重复日期，并以 JPM 交易日为主日历合并
jpm = jpm.drop_duplicates(subset="date").sort_values("date")
vix = vix.drop_duplicates(subset="date").sort_values("date")
treasury = treasury.drop_duplicates(subset="date").sort_values("date")

data = jpm.merge(vix[["date", "vix"]], on="date", how="left")
data = data.merge(
    treasury[["date", "treasury_10y"]],
    on="date",
    how="left",
)

# 4. 填补因周末、节假日造成的宏观和波动率缺失
missing_before = data[["vix", "treasury_10y"]].isna().sum()

data[["vix", "treasury_10y"]] = (
    data[["vix", "treasury_10y"]]
    .ffill()
    .bfill()
)

missing_after = data[["vix", "treasury_10y"]].isna().sum()

# 5. IQR 异常值标记：保留原始值，仅供后续审查
data["vix_iqr_outlier"] = iqr_outlier_flag(data["vix"])
data["treasury_10y_iqr_outlier"] = iqr_outlier_flag(data["treasury_10y"])

# 6. 输出对齐后的清洗数据
output_path = PROCESSED_DIR / "aligned_clean_data.csv"
data.to_csv(output_path, index=False)

print(f"合并后行数：{len(data)}")
print(f"日期范围：{data['date'].min().date()} 至 {data['date'].max().date()}")
print("填补前缺失值：")
print(missing_before.to_string())
print("填补后缺失值：")
print(missing_after.to_string())
print(f"VIX IQR 异常值标记数：{data['vix_iqr_outlier'].sum()}")
print(
    "10年期国债利率 IQR 异常值标记数："
    f"{data['treasury_10y_iqr_outlier'].sum()}"
)
print(f"已保存：{output_path}")