import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_synthetic_price_splits"
CONFIG_PATH = PROJECT_ROOT / "config" / "week_5_end_to_end_config.json"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_5"

RANDOM_SEED = 42
NORMALIZED_TARGET = "heston_price_to_spot"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLIT_DIR / f"week_5_e2e_{name}.csv", parse_dates=["date"])


def add_stationary_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create scale-free features to improve time-series generalization."""
    result = data.copy()

    result["close_to_ma20_ratio"] = result["close"] / result["close_ma_20"]
    result["close_to_ma60_ratio"] = result["close"] / result["close_ma_60"]
    result["volume_to_ma20_ratio"] = result["volume"] / result["volume_ma_20"]
    result[NORMALIZED_TARGET] = result["heston_synthetic_price"] / result["close"]

    return result


def calculate_metrics(actual, predicted) -> dict:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    feature_columns = [
        "daily_return",
        "log_return",
        "intraday_return",
        "high_low_range",
        "volume_change",
        "volume_to_ma20_ratio",
        "close_to_ma20_ratio",
        "close_to_ma60_ratio",
        "rolling_vol_20",
        "rolling_vol_60",
        "vix",
        "vix_change",
        "vix_ma_20",
        "treasury_10y",
        "treasury_10y_change",
        "treasury_10y_momentum_20",
        "jpm_vix_corr_20",
        "moneyness",
        "choice_time_years",
        "maturity_years",
    ]

    train_data = add_stationary_features(load_split("train"))
    validation_data = add_stationary_features(load_split("validation"))
    test_data = add_stationary_features(load_split("test"))

    candidate_settings = [
        {
            "model_name": "normalized_gbdt_conservative",
            "n_estimators": 250,
            "learning_rate": 0.04,
            "max_depth": 2,
            "min_samples_leaf": 5,
        },
        {
            "model_name": "normalized_gbdt_balanced",
            "n_estimators": 350,
            "learning_rate": 0.04,
            "max_depth": 2,
            "min_samples_leaf": 3,
        },
    ]

    validation_results = [
        {
            "model_name": "bsm_closed_form_baseline",
            **calculate_metrics(
                validation_data["heston_synthetic_price"],
                validation_data["bsm_price"],
            ),
        }
    ]

    for settings in candidate_settings:
        model = GradientBoostingRegressor(
            n_estimators=settings["n_estimators"],
            learning_rate=settings["learning_rate"],
            max_depth=settings["max_depth"],
            min_samples_leaf=settings["min_samples_leaf"],
            random_state=RANDOM_SEED,
            loss="huber",
        )

        model.fit(
            train_data[feature_columns],
            train_data[NORMALIZED_TARGET],
        )

        predicted_ratio = model.predict(validation_data[feature_columns])
        predicted_price = predicted_ratio * validation_data["close"]

        validation_results.append(
            {
                "model_name": settings["model_name"],
                **calculate_metrics(
                    validation_data["heston_synthetic_price"],
                    predicted_price,
                ),
            }
        )

    validation_results_df = pd.DataFrame(validation_results).sort_values("rmse")

    best_ml_name = validation_results_df[
        ~validation_results_df["model_name"].str.startswith("bsm_")
    ].iloc[0]["model_name"]

    selected_settings = next(
        item
        for item in candidate_settings
        if item["model_name"] == best_ml_name
    )

    train_validation_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    final_model = GradientBoostingRegressor(
        n_estimators=selected_settings["n_estimators"],
        learning_rate=selected_settings["learning_rate"],
        max_depth=selected_settings["max_depth"],
        min_samples_leaf=selected_settings["min_samples_leaf"],
        random_state=RANDOM_SEED,
        loss="huber",
    )

    final_model.fit(
        train_validation_data[feature_columns],
        train_validation_data[NORMALIZED_TARGET],
    )

    test_predicted_ratio = final_model.predict(test_data[feature_columns])
    test_predicted_price = test_predicted_ratio * test_data["close"]

    test_results = pd.DataFrame(
        [
            {
                "model_name": "bsm_closed_form_baseline",
                **calculate_metrics(
                    test_data["heston_synthetic_price"],
                    test_data["bsm_price"],
                ),
            },
            {
                "model_name": best_ml_name,
                **calculate_metrics(
                    test_data["heston_synthetic_price"],
                    test_predicted_price,
                ),
            },
        ]
    )

    predictions = test_data[
        [
            "date",
            "close",
            "strike",
            "moneyness",
            "choice_time_years",
            "bsm_price",
            "heston_synthetic_price",
            "heston_mc_standard_error",
        ]
    ].copy()

    predictions["normalized_gbdt_predicted_price"] = test_predicted_price
    predictions["normalized_gbdt_predicted_ratio"] = test_predicted_ratio
    predictions["bsm_minus_heston"] = (
        predictions["bsm_price"] - predictions["heston_synthetic_price"]
    )
    predictions["normalized_gbdt_minus_heston"] = (
        predictions["normalized_gbdt_predicted_price"]
        - predictions["heston_synthetic_price"]
    )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    model_path = MODEL_DIR / "week_5_normalized_pricing_gbdt.joblib"
    validation_path = OUTPUT_DIR / "normalized_pricing_validation_results.csv"
    test_path = OUTPUT_DIR / "normalized_pricing_test_metrics.csv"
    predictions_path = OUTPUT_DIR / "normalized_pricing_test_predictions.csv"
    importance_path = OUTPUT_DIR / "normalized_pricing_feature_importance.csv"

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "target_column": NORMALIZED_TARGET,
            "selected_settings": selected_settings,
            "target_disclosure": metadata["disclosure"],
            "normalization": "Heston synthetic price divided by current spot price.",
        },
        model_path,
    )

    validation_results_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
    test_results.to_csv(test_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")

    print("Validation model selection")
    print(
        validation_results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Selected ML model: {best_ml_name}")
    print()
    print("Final test metrics")
    print(
        test_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print("Top 10 feature importances")
    print(
        importance.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Model saved to: {model_path}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("Disclosure: target is a Heston MC synthetic reference price.")


if __name__ == "__main__":
    main()