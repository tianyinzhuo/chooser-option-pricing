import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bsm_chooser import simple_chooser_price
from heston_chooser import heston_simple_chooser_price


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_dataset_2018_2024.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_synthetic_price_splits"
CONFIG_PATH = PROJECT_ROOT / "config" / "week_5_end_to_end_config.json"

MARKET_FEATURES = [
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

CONTRACT_FEATURES = [
    "strike",
    "moneyness",
    "choice_time_years",
    "maturity_years",
]

TARGET_COLUMN = "heston_synthetic_price"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)

    # One date every 30 trading days gives a manageable initial ML panel.
    sampled_dates = data.iloc[::30].copy()

    if sampled_dates["date"].iloc[-1] != data["date"].iloc[-1]:
        sampled_dates = pd.concat(
            [sampled_dates, data.tail(1)],
            ignore_index=True,
        )

    moneyness_values = [0.90, 1.00, 1.10]
    choice_time_values = [0.25, 0.50]

    records = []
    total_contracts = len(sampled_dates) * len(moneyness_values) * len(choice_time_values)

    print(f"Sampled market dates: {len(sampled_dates)}")
    print(f"Synthetic contracts to price: {total_contracts}")

    contract_number = 0

    for _, row in sampled_dates.iterrows():
        spot_price = float(row["close"])
        risk_free_rate = float(row["treasury_10y"]) / 100.0
        volatility = float(row["rolling_vol_60"])

        for moneyness in moneyness_values:
            for choice_time_years in choice_time_values:
                contract_number += 1
                strike = spot_price * moneyness

                bsm_result = simple_chooser_price(
                    spot_price=spot_price,
                    strike=strike,
                    choice_time_years=choice_time_years,
                    maturity_years=1.0,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    dividend_yield=0.0,
                )

                heston_result = heston_simple_chooser_price(
                    spot=spot_price,
                    strike=strike,
                    choice_time_years=choice_time_years,
                    maturity_years=1.0,
                    risk_free_rate=risk_free_rate,
                    initial_variance=volatility**2,
                    kappa=2.0,
                    long_run_variance=0.04,
                    vol_of_vol=0.5,
                    rho=-0.6,
                    dividend_yield=0.0,
                    num_paths=2000,
                    trading_days_per_year=252,
                    seed=1000 + contract_number,
                )

                record = {
                    "date": row["date"].date().isoformat(),
                    **{feature: row[feature] for feature in MARKET_FEATURES},
                    "strike": strike,
                    "moneyness": moneyness,
                    "choice_time_years": choice_time_years,
                    "maturity_years": 1.0,
                    "bsm_price": bsm_result["chooser_price"],
                    TARGET_COLUMN: heston_result["price"],
                    "heston_mc_standard_error": heston_result["standard_error"],
                }

                records.append(record)

                if contract_number % 30 == 0 or contract_number == total_contracts:
                    print(
                        f"Priced {contract_number}/{total_contracts} synthetic contracts"
                    )

    panel = pd.DataFrame(records)
    panel["date"] = pd.to_datetime(panel["date"])

    unique_dates = panel["date"].drop_duplicates().sort_values().tolist()
    train_end = int(len(unique_dates) * 0.70)
    validation_end = train_end + int(len(unique_dates) * 0.15)

    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end:validation_end]
    test_dates = unique_dates[validation_end:]

    train_data = panel[panel["date"].isin(train_dates)].copy()
    validation_data = panel[panel["date"].isin(validation_dates)].copy()
    test_data = panel[panel["date"].isin(test_dates)].copy()

    train_data.to_csv(
        OUTPUT_DIR / "week_5_e2e_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_data.to_csv(
        OUTPUT_DIR / "week_5_e2e_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_data.to_csv(
        OUTPUT_DIR / "week_5_e2e_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "label_type": "Heston Monte Carlo synthetic reference price",
        "disclosure": (
            "This target is a simulated reference price, not an actual OTC "
            "transaction price."
        ),
        "market_features": MARKET_FEATURES,
        "contract_features": CONTRACT_FEATURES,
        "target_column": TARGET_COLUMN,
        "label_simulation_paths": 2000,
        "date_sampling_step_days": 30,
        "moneyness_values": moneyness_values,
        "choice_time_values": choice_time_values,
        "maturity_years": 1.0,
        "split_method": "chronological_70_15_15_by_unique_date",
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
    }

    CONFIG_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print("Chronological end-to-end pricing splits")
    print(f"Train:      {len(train_data)} contracts")
    print(f"Validation: {len(validation_data)} contracts")
    print(f"Test:       {len(test_data)} contracts")
    print(f"Saved synthetic pricing splits to: {OUTPUT_DIR}")
    print(f"Saved metadata to: {CONFIG_PATH}")
    print("Disclosure: targets are Heston MC synthetic reference prices.")


if __name__ == "__main__":
    main()