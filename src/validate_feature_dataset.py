from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_dataset_2018_2024.csv"
)

data = pd.read_csv(DATA_PATH, parse_dates=["date"])

print(f"行数：{len(data)}")
print(f"列数：{len(data.columns)}")
print(
    f"日期范围：{data['date'].min().date()} 至 "
    f"{data['date'].max().date()}"
)
print(f"重复日期数：{data['date'].duplicated().sum()}")
print(f"总缺失值：{data.isna().sum().sum()}")

if data.isna().sum().sum() == 0:
    print("结果：通过——特征数据集不存在缺失值。")
else:
    print("结果：需检查——仍存在缺失值：")
    print(data.isna().sum()[data.isna().sum() > 0].to_string())

print("\n数据集前五行：")
print(data.head().to_string(index=False))