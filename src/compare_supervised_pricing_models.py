import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_synthetic_price_splits"
CONFIG_PATH = PROJECT_ROOT / "config" / "week_5_end_to_end_config.json"
HYPERPARAMETER_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "week_5_model_hyperparameters.json"
)

MODEL_DIR = PROJECT_ROOT / "models" / "week_5_supervised_comparison"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_5"

TARGET_COLUMN = "heston_price_to_spot"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(
        SPLIT_DIR / f"week_5_e2e_{name}.csv",
        parse_dates=["date"],
    )


def add_stationary_features(data: pd.DataFrame) -> pd.DataFrame:
    """Use scale-free targets and features for cross-period comparison."""
    result = data.copy()

    result["close_to_ma20_ratio"] = result["close"] / result["close_ma_20"]
    result["close_to_ma60_ratio"] = result["close"] / result["close_ma_60"]
    result["volume_to_ma20_ratio"] = result["volume"] / result["volume_ma_20"]
    result[TARGET_COLUMN] = (
        result["heston_synthetic_price"] / result["close"]
    )

    return result


def calculate_metrics(actual, predicted) -> dict:
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
    }


def create_model(family: str, parameters: dict, training_defaults: dict):
    """Create one supervised-learning candidate."""
    supported = {
        "ridge": {"alpha"},
        "random_forest": {"n_estimators", "max_depth", "min_samples_leaf"},
        "gbdt": {"n_estimators", "learning_rate", "max_depth", "min_samples_leaf"},
        "mlp": {"hidden_layer_sizes", "alpha", "learning_rate_init", "max_iter", "early_stopping"},
    }
    if family not in supported:
        raise ValueError(f"Unknown model family: {family}")
    if set(parameters) != supported[family]:
        raise ValueError(f"{family} parameters must contain exactly: {sorted(supported[family])}")
    if family == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=parameters["alpha"])),
            ]
        )

    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            random_state=training_defaults["random_seed"],
            n_jobs=training_defaults["n_jobs"],
        )

    if family == "gbdt":
        return GradientBoostingRegressor(
            n_estimators=parameters["n_estimators"],
            learning_rate=parameters["learning_rate"],
            max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"],
            random_state=training_defaults["random_seed"],
            loss=training_defaults["gbdt_loss"],
        )

    if family == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=tuple(parameters["hidden_layer_sizes"]),
                        alpha=parameters["alpha"],
                        learning_rate_init=parameters["learning_rate_init"],
                        max_iter=parameters["max_iter"],
                        early_stopping=parameters["early_stopping"],
                        random_state=training_defaults["random_seed"],
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model family: {family}")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    hyperparameters = json.loads(
        HYPERPARAMETER_CONFIG_PATH.read_text(encoding="utf-8-sig")
    )
    training_defaults = hyperparameters["training_defaults"]
    selection_metric = hyperparameters["supervised_pricing_comparison"]["selection_metric"]
    if selection_metric not in {"mae", "rmse"}:
        raise ValueError("selection_metric must be 'mae' or 'rmse'.")

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

    if hyperparameters["supervised_pricing_comparison"]["target"] != TARGET_COLUMN:
        raise ValueError("Configured target does not match this training script.")
    candidates = hyperparameters["supervised_pricing_comparison"]["candidates"]
    if not candidates or len({c["model_name"] for c in candidates}) != len(candidates):
        raise ValueError("Candidates must be non-empty with unique model names.")
    # Validate every candidate before any training starts.
    for candidate in candidates:
        if set(candidate) != {"model_name", "family", "parameters"}:
            raise ValueError("Candidates require model_name, family and parameters only.")
        create_model(candidate["family"], candidate["parameters"], training_defaults)

    x_train = train_data[feature_columns]
    y_train = train_data[TARGET_COLUMN]

    x_validation = validation_data[feature_columns]
    y_validation = validation_data[TARGET_COLUMN]

    validation_results = [
        {
            "model_name": "bsm_closed_form_baseline",
            "family": "bsm",
            **calculate_metrics(
                validation_data["heston_synthetic_price"],
                validation_data["bsm_price"],
            ),
        }
    ]

    for candidate in candidates:
        model = create_model(candidate["family"], candidate["parameters"], training_defaults)
        model.fit(x_train, y_train)

        predicted_ratio = model.predict(x_validation)
        predicted_price = predicted_ratio * validation_data["close"]

        validation_results.append(
            {
                "model_name": candidate["model_name"],
                "family": candidate["family"],
                **calculate_metrics(
                    validation_data["heston_synthetic_price"],
                    predicted_price,
                ),
            }
        )

    validation_results_df = pd.DataFrame(validation_results).sort_values(selection_metric, kind="stable")

    selected_candidates = []

    for family in dict.fromkeys(candidate["family"] for candidate in candidates):
        best_name = validation_results_df[
            validation_results_df["family"] == family
        ].iloc[0]["model_name"]

        selected_candidates.append(
            next(
                item for item in candidates
                if item["model_name"] == best_name
            )
        )

    train_validation_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )

    test_results = [
        {
            "model_name": "bsm_closed_form_baseline",
            "family": "bsm",
            **calculate_metrics(
                test_data["heston_synthetic_price"],
                test_data["bsm_price"],
            ),
        }
    ]

    prediction_results = test_data[
        [
            "date",
            "close",
            "strike",
            "moneyness",
            "choice_time_years",
            "bsm_price",
            "heston_synthetic_price",
        ]
    ].copy()

    for candidate in selected_candidates:
        model = create_model(candidate["family"], candidate["parameters"], training_defaults)

        model.fit(
            train_validation_data[feature_columns],
            train_validation_data[TARGET_COLUMN],
        )

        predicted_ratio = model.predict(test_data[feature_columns])
        predicted_price = predicted_ratio * test_data["close"]

        prediction_column = f"{candidate['family']}_predicted_price"
        prediction_results[prediction_column] = predicted_price

        test_results.append(
            {
                "model_name": candidate["model_name"],
                "family": candidate["family"],
                **calculate_metrics(
                    test_data["heston_synthetic_price"],
                    predicted_price,
                ),
            }
        )

        selected_estimator = (
            model.named_steps["model"] if isinstance(model, Pipeline) else model
        )
        selected_record = {
            "model_name": candidate["model_name"],
            "family": candidate["family"],
            "selection_metric": selection_metric,
            "selected_parameters": candidate["parameters"],
            "resolved_parameters": selected_estimator.get_params(deep=False),
            "preprocessing": "StandardScaler" if isinstance(model, Pipeline) else "none",
            "sklearn_version": sklearn.__version__,
            "configuration_snapshot": hyperparameters,
        }
        (OUTPUT_DIR / f"supervised_{candidate['family']}_selected_hyperparameters.json").write_text(
            json.dumps(selected_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        joblib.dump(
            {
                "model": model,
                "family": candidate["family"],
                "feature_columns": feature_columns,
                "target_column": TARGET_COLUMN,
                "parameters": candidate["parameters"],
                "model_name": candidate["model_name"],
                "selection_metric": selection_metric,
                "hyperparameter_config": hyperparameters,
                "sklearn_version": sklearn.__version__,
                "target_disclosure": metadata["disclosure"],
            },
            MODEL_DIR / f"{candidate['family']}_best_model.joblib",
        )

    test_results_df = pd.DataFrame(test_results).sort_values(selection_metric, kind="stable")

    validation_results_df.to_csv(
        OUTPUT_DIR / "supervised_model_validation_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_results_df.to_csv(
        OUTPUT_DIR / "supervised_model_test_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_results.to_csv(
        OUTPUT_DIR / "supervised_model_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Validation comparison: all candidates")
    print(
        validation_results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print("Final test comparison: best validation candidate per family")
    print(
        test_results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print("Selected candidates by family")
    for candidate in selected_candidates:
        print(f"- {candidate['family']}: {candidate['model_name']}")
    print()
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Disclosure: all metrics use Heston MC synthetic reference prices.")


if __name__ == "__main__":
    main()
