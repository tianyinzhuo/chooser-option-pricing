import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_synthetic_price_splits"
CONFIG_PATH = PROJECT_ROOT / "config" / "week_5_end_to_end_config.json"
HYPERPARAMETER_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "week_5_model_hyperparameters.json"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_5"



def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLIT_DIR / f"week_5_e2e_{name}.csv", parse_dates=["date"])


def calculate_metrics(actual: pd.Series, predicted) -> dict:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    hyperparameters = json.loads(
        HYPERPARAMETER_CONFIG_PATH.read_text(encoding="utf-8-sig")
    )
    training_defaults = hyperparameters["training_defaults"]
    selection_metric = hyperparameters["raw_dollar_gbdt"]["selection_metric"]
    if selection_metric not in {"mae", "rmse"}:
        raise ValueError("selection_metric must be 'mae' or 'rmse'.")

    feature_columns = (
        metadata["market_features"]
        + metadata["contract_features"]
    )
    target_column = metadata["target_column"]

    train_data = load_split("train")
    validation_data = load_split("validation")
    test_data = load_split("test")

    if hyperparameters["raw_dollar_gbdt"]["target"] != target_column:
        raise ValueError("Configured target does not match this training script.")
    candidate_settings = hyperparameters["raw_dollar_gbdt"]["candidates"]
    if not candidate_settings or len({c["model_name"] for c in candidate_settings}) != len(candidate_settings):
        raise ValueError("Candidates must be non-empty with unique model names.")
    supported_keys = {"model_name", "n_estimators", "learning_rate", "max_depth", "min_samples_leaf"}
    for settings in candidate_settings:
        if set(settings) != supported_keys:
            raise ValueError(f"Candidate keys must contain exactly: {sorted(supported_keys)}")

    validation_results = [
        {
            "model_name": "bsm_closed_form_baseline",
            **calculate_metrics(
                validation_data[target_column],
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
            random_state=training_defaults["random_seed"],
            loss=training_defaults["gbdt_loss"],
        )

        model.fit(
            train_data[feature_columns],
            train_data[target_column],
        )

        validation_prediction = model.predict(
            validation_data[feature_columns]
        )

        validation_results.append(
            {
                "model_name": settings["model_name"],
                **calculate_metrics(
                    validation_data[target_column],
                    validation_prediction,
                ),
            }
        )

    validation_results_df = pd.DataFrame(validation_results).sort_values(selection_metric, kind="stable")

    best_gbdt_name = validation_results_df[
        ~validation_results_df["model_name"].str.startswith("bsm_")
    ].iloc[0]["model_name"]

    selected_settings = next(
        item
        for item in candidate_settings
        if item["model_name"] == best_gbdt_name
    )

    # Refit the selected ML specification on train + validation only.
    train_validation_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    final_model = GradientBoostingRegressor(
        n_estimators=selected_settings["n_estimators"],
        learning_rate=selected_settings["learning_rate"],
        max_depth=selected_settings["max_depth"],
        min_samples_leaf=selected_settings["min_samples_leaf"],
        random_state=training_defaults["random_seed"],
        loss=training_defaults["gbdt_loss"],
    )

    final_model.fit(
        train_validation_data[feature_columns],
        train_validation_data[target_column],
    )

    test_prediction = final_model.predict(test_data[feature_columns])

    test_results = pd.DataFrame(
        [
            {
                "model_name": "bsm_closed_form_baseline",
                **calculate_metrics(
                    test_data[target_column],
                    test_data["bsm_price"],
                ),
            },
            {
                "model_name": best_gbdt_name,
                **calculate_metrics(
                    test_data[target_column],
                    test_prediction,
                ),
            },
        ]
    )

    prediction_results = test_data[
        [
            "date",
            "close",
            "strike",
            "moneyness",
            "choice_time_years",
            "bsm_price",
            target_column,
            "heston_mc_standard_error",
        ]
    ].copy()

    prediction_results["gbdt_predicted_price"] = test_prediction
    prediction_results["bsm_minus_heston"] = (
        prediction_results["bsm_price"] - prediction_results[target_column]
    )
    prediction_results["gbdt_minus_heston"] = (
        prediction_results["gbdt_predicted_price"]
        - prediction_results[target_column]
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    model_path = MODEL_DIR / "week_5_end_to_end_gbdt.joblib"
    validation_path = OUTPUT_DIR / "e2e_pricing_validation_results.csv"
    test_path = OUTPUT_DIR / "e2e_pricing_test_metrics.csv"
    predictions_path = OUTPUT_DIR / "e2e_pricing_test_predictions.csv"
    importance_path = OUTPUT_DIR / "e2e_pricing_feature_importance.csv"

    selected_record = {
        "model_name": best_gbdt_name,
        "selection_metric": selection_metric,
        "selected_parameters": selected_settings,
        "resolved_parameters": final_model.get_params(deep=False),
        "sklearn_version": sklearn.__version__,
        "configuration_snapshot": hyperparameters,
    }
    (OUTPUT_DIR / "e2e_pricing_selected_hyperparameters.json").write_text(
        json.dumps(selected_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "selected_settings": selected_settings,
            "hyperparameter_config": hyperparameters,
            "sklearn_version": sklearn.__version__,
            "target_disclosure": metadata["disclosure"],
        },
        model_path,
    )

    validation_results_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
    test_results.to_csv(test_path, index=False, encoding="utf-8-sig")
    prediction_results.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    feature_importance.to_csv(importance_path, index=False, encoding="utf-8-sig")

    print("Validation model selection")
    print(
        validation_results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Selected ML model: {best_gbdt_name}")
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
        feature_importance.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Model saved to: {model_path}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("Disclosure: evaluation target is a Heston MC synthetic reference price.")


if __name__ == "__main__":
    main()
