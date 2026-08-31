import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_splits"
CONFIG_PATH = PROJECT_ROOT / "config" / "week_5_ml_data_config.json"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_5"

RANDOM_SEED = 42


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Calculate comparable regression metrics."""
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
    }


def load_split(split_name: str) -> pd.DataFrame:
    """Load one chronologically ordered ML split."""
    return pd.read_csv(SPLIT_DIR / f"{split_name}.csv", parse_dates=["date"])


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target"]

    train_data = load_split("week_5_train")
    validation_data = load_split("week_5_validation")
    test_data = load_split("week_5_test")

    x_train = train_data[feature_columns]
    y_train = train_data[target_column]

    x_validation = validation_data[feature_columns]
    y_validation = validation_data[target_column]

    candidate_settings = [
        {
            "model_name": "rf_conservative",
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 5,
        },
        {
            "model_name": "rf_balanced",
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_leaf": 3,
        },
    ]

    validation_results = []
    trained_models = {}

    baseline_metrics = calculate_metrics(
        y_validation,
        validation_data["rolling_vol_60"],
    )
    validation_results.append(
        {
            "model_name": "persistence_baseline_rolling_vol_60",
            **baseline_metrics,
        }
    )

    for settings in candidate_settings:
        model = RandomForestRegressor(
            n_estimators=settings["n_estimators"],
            max_depth=settings["max_depth"],
            min_samples_leaf=settings["min_samples_leaf"],
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        model.fit(x_train, y_train)
        validation_prediction = model.predict(x_validation)
        metrics = calculate_metrics(y_validation, validation_prediction)

        validation_results.append(
            {
                "model_name": settings["model_name"],
                **metrics,
            }
        )
        trained_models[settings["model_name"]] = model

    validation_results_df = pd.DataFrame(validation_results).sort_values("rmse")
    selected_name = validation_results_df.iloc[0]["model_name"]

    if selected_name.startswith("persistence_baseline"):
        print(
            "Warning: the persistence baseline won on validation data. "
            "The best Random Forest will still be trained and evaluated on the test set."
        )

        selected_name = validation_results_df[
            ~validation_results_df["model_name"].str.startswith(
                "persistence_baseline"
            )
        ].iloc[0]["model_name"]
    selected_settings = next(
        settings
        for settings in candidate_settings
        if settings["model_name"] == selected_name
    )

    # Refit only after choosing settings using validation data.
    train_validation_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    final_model = RandomForestRegressor(
        n_estimators=selected_settings["n_estimators"],
        max_depth=selected_settings["max_depth"],
        min_samples_leaf=selected_settings["min_samples_leaf"],
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    final_model.fit(
        train_validation_data[feature_columns],
        train_validation_data[target_column],
    )

    test_prediction = final_model.predict(test_data[feature_columns])

    test_results = pd.DataFrame(
        [
            {
                "model_name": "persistence_baseline_rolling_vol_60",
                **calculate_metrics(
                    test_data[target_column],
                    test_data["rolling_vol_60"],
                ),
            },
            {
                "model_name": selected_name,
                **calculate_metrics(
                    test_data[target_column],
                    test_prediction,
                ),
            },
        ]
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    model_path = MODEL_DIR / "week_5_volatility_random_forest.joblib"
    validation_path = OUTPUT_DIR / "volatility_rf_validation_results.csv"
    test_path = OUTPUT_DIR / "volatility_rf_test_metrics.csv"
    importance_path = OUTPUT_DIR / "volatility_rf_feature_importance.csv"

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "selected_settings": selected_settings,
            "split_method": metadata["split_method"],
        },
        model_path,
    )

    validation_results_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
    test_results.to_csv(test_path, index=False, encoding="utf-8-sig")
    feature_importance.to_csv(importance_path, index=False, encoding="utf-8-sig")

    print("Validation model selection")
    print(validation_results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"Selected model: {selected_name}")
    print()
    print("Final test metrics")
    print(test_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("Top 10 feature importances")
    print(feature_importance.head(10).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"Model saved to: {model_path}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()