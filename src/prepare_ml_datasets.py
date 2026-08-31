import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_dataset_2018_2024.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_splits"
CONFIG_DIR = PROJECT_ROOT / "config"

TARGET_NAME = "future_realized_vol_20"

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vix",
    "treasury_10y",
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


def create_forward_volatility_target(data: pd.DataFrame) -> pd.Series:
    """
    At date t, calculate realized annualized volatility from returns
    t+1 through t+20. This is a future target, never an input feature.
    """
    future_returns = data["log_return"].shift(-1)
    reversed_returns = future_returns.iloc[::-1]

    target = (
        reversed_returns.rolling(window=20, min_periods=20).std().iloc[::-1]
        * np.sqrt(252)
    )
    return target


def save_split(data: pd.DataFrame, split_name: str) -> None:
    output_path = OUTPUT_DIR / f"{split_name}.csv"
    data.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)

    data[TARGET_NAME] = create_forward_volatility_target(data)
    dataset = data[["date", *FEATURE_COLUMNS, TARGET_NAME]].dropna().copy()

    total_rows = len(dataset)
    train_end = int(total_rows * 0.70)
    validation_end = train_end + int(total_rows * 0.15)

    train_data = dataset.iloc[:train_end].copy()
    validation_data = dataset.iloc[train_end:validation_end].copy()
    test_data = dataset.iloc[validation_end:].copy()

    save_split(train_data, "week_5_train")
    save_split(validation_data, "week_5_validation")
    save_split(test_data, "week_5_test")

    metadata = {
        "target": TARGET_NAME,
        "target_description": (
            "Annualized realized volatility calculated from the next "
            "20 trading days of log returns."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "split_method": "chronological_70_15_15",
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "test_rows": len(test_data),
        "train_date_range": [
            train_data["date"].min().date().isoformat(),
            train_data["date"].max().date().isoformat(),
        ],
        "validation_date_range": [
            validation_data["date"].min().date().isoformat(),
            validation_data["date"].max().date().isoformat(),
        ],
        "test_date_range": [
            test_data["date"].min().date().isoformat(),
            test_data["date"].max().date().isoformat(),
        ],
        "look_ahead_bias_control": (
            "Rows are sorted chronologically and never randomly shuffled. "
            "The target uses future returns only as a label, not as a feature."
        ),
    }

    metadata_path = CONFIG_DIR / "week_5_ml_data_config.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Input rows: {len(data)}")
    print(f"Usable rows after forward target construction: {total_rows}")
    print()
    print("Chronological split")
    print(
        f"Train:      {len(train_data)} rows | "
        f"{metadata['train_date_range'][0]} to {metadata['train_date_range'][1]}"
    )
    print(
        f"Validation: {len(validation_data)} rows | "
        f"{metadata['validation_date_range'][0]} to "
        f"{metadata['validation_date_range'][1]}"
    )
    print(
        f"Test:       {len(test_data)} rows | "
        f"{metadata['test_date_range'][0]} to {metadata['test_date_range'][1]}"
    )
    print()
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print(f"Target: {TARGET_NAME}")
    print(f"Saved splits to: {OUTPUT_DIR}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()