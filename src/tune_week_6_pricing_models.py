"""Week 6: tune normalized supervised chooser-pricing models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.week_6_cv import make_grouped_date_cv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_synthetic_price_splits"
SEARCH_CONFIG_PATH = PROJECT_ROOT / "config" / "week_6_model_search.json"
MODEL_DIR = PROJECT_ROOT / "models" / "week_6"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_6"

TARGET_COLUMN = "heston_price_to_spot"

FEATURE_COLUMNS = [
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


def load_split(name: str) -> pd.DataFrame:
    """Load one chronological synthetic-pricing split."""
    path = SPLIT_DIR / f"week_5_e2e_{name}.csv"
    return pd.read_csv(path, parse_dates=["date"])


def add_stationary_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the scale-free features and normalized pricing target."""
    result = data.copy()
    result["close_to_ma20_ratio"] = result["close"] / result["close_ma_20"]
    result["close_to_ma60_ratio"] = result["close"] / result["close_ma_60"]
    result["volume_to_ma20_ratio"] = result["volume"] / result["volume_ma_20"]
    result[TARGET_COLUMN] = result["heston_synthetic_price"] / result["close"]
    return result


def calculate_metrics(actual, predicted) -> dict:
    """Calculate dollar-price regression metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def create_model(family: str, parameters: dict, random_seed: int, n_jobs: int):
    """Create one estimator from a model family and parameter dictionary."""
    if family == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=parameters["alpha"])),
            ]
        )

    if family == "random_forest":
        return RandomForestRegressor(
            **parameters,
            random_state=random_seed,
            n_jobs=n_jobs,
        )

    if family == "gbdt":
        return GradientBoostingRegressor(
            **parameters,
            random_state=random_seed,
        )

    if family == "mlp":
        mlp_parameters = parameters.copy()
        mlp_parameters["hidden_layer_sizes"] = tuple(
            mlp_parameters["hidden_layer_sizes"]
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        **mlp_parameters,
                        random_state=random_seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model family: {family}")


def get_parameter_candidates(family: str, family_config: dict, seed: int) -> list:
    """Expand a grid or reproducibly sample a parameter distribution."""
    search_type = family_config["search_type"]

    if search_type == "grid":
        return list(ParameterGrid(family_config["parameter_grid"]))

    if search_type == "randomized":
        return list(
            ParameterSampler(
                family_config["parameter_distributions"],
                n_iter=family_config["n_iter"],
                random_state=seed,
            )
        )

    raise ValueError(f"Unsupported search type for {family}: {search_type}")


def mean_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Average metrics across folds so every fold has equal weight."""
    return {
        metric: float(np.mean([item[metric] for item in fold_metrics]))
        for metric in ["mae", "rmse", "r2"]
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads(SEARCH_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    pricing_config = config["pricing_route"]
    cv_config = config["cross_validation"]["pricing"]

    train_data = add_stationary_features(load_split("train"))
    validation_data = add_stationary_features(load_split("validation"))
    test_data = add_stationary_features(load_split("test"))

    pretest_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    ).sort_values(["date", "strike", "choice_time_years"]).reset_index(drop=True)

    test_data = test_data.sort_values(
        ["date", "strike", "choice_time_years"]
    ).reset_index(drop=True)

    if pretest_data["date"].max() >= test_data["date"].min():
        raise ValueError("Pre-test dates must end before the test period.")

    required_columns = [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        "heston_synthetic_price",
        "bsm_price",
        "close",
    ]

    if pretest_data[required_columns].isna().any().any():
        raise ValueError("Pre-test pricing data contains missing values.")

    if test_data[required_columns].isna().any().any():
        raise ValueError("Test pricing data contains missing values.")

    cv_folds = make_grouped_date_cv(
        dates=pretest_data[cv_config["group_column"]],
        n_splits=cv_config["n_splits"],
        gap_groups=cv_config["gap_groups"],
    )

    fold_summary_records = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        cv_folds,
        start=1,
    ):
        fold_summary_records.append(
            {
                "fold": fold_number,
                "train_contracts": len(train_indices),
                "validation_contracts": len(validation_indices),
                "train_unique_dates": pretest_data.iloc[train_indices]["date"].nunique(),
                "validation_unique_dates": pretest_data.iloc[validation_indices]["date"].nunique(),
                "train_start": pretest_data.iloc[train_indices]["date"].min(),
                "train_end": pretest_data.iloc[train_indices]["date"].max(),
                "validation_start": pretest_data.iloc[validation_indices]["date"].min(),
                "validation_end": pretest_data.iloc[validation_indices]["date"].max(),
            }
        )

    fold_summary = pd.DataFrame(fold_summary_records)
    x_pretest = pretest_data[FEATURE_COLUMNS]
    y_pretest = pretest_data[TARGET_COLUMN]

    candidate_records = []
    selected_by_family = {}

    print("Starting Week 6 grouped-date pricing model search...")
    print(
        f"Pre-test contracts: {len(pretest_data)} | "
        f"unique dates: {pretest_data['date'].nunique()}"
    )
    print(
        f"Test contracts kept separate: {len(test_data)} | "
        f"unique dates: {test_data['date'].nunique()}"
    )
    print()

    # Evaluate the closed-form BSM baseline with the same CV folds.
    bsm_fold_metrics = []
    for _, validation_indices in cv_folds:
        fold = pretest_data.iloc[validation_indices]
        bsm_fold_metrics.append(
            calculate_metrics(
                fold["heston_synthetic_price"],
                fold["bsm_price"],
            )
        )

    bsm_cv_metrics = mean_fold_metrics(bsm_fold_metrics)

    family_seed_offsets = {
        "ridge": 0,
        "random_forest": 1,
        "gbdt": 2,
        "mlp": 3,
    }

    for family, family_config in pricing_config["models"].items():
        candidates = get_parameter_candidates(
            family,
            family_config,
            config["random_seed"] + family_seed_offsets[family],
        )

        print(f"Searching {family}: {len(candidates)} candidates")
        family_records = []

        for candidate_number, parameters in enumerate(candidates, start=1):
            fold_metrics = []

            for train_indices, validation_indices in cv_folds:
                model = create_model(
                    family,
                    parameters,
                    random_seed=config["random_seed"],
                    n_jobs=config["n_jobs"],
                )

                model.fit(
                    x_pretest.iloc[train_indices],
                    y_pretest.iloc[train_indices],
                )

                predicted_ratio = model.predict(
                    x_pretest.iloc[validation_indices]
                )
                predicted_price = (
                    predicted_ratio
                    * pretest_data.iloc[validation_indices]["close"].to_numpy()
                )

                fold_metrics.append(
                    calculate_metrics(
                        pretest_data.iloc[validation_indices][
                            "heston_synthetic_price"
                        ],
                        predicted_price,
                    )
                )

            averaged = mean_fold_metrics(fold_metrics)
            record = {
                "family": family,
                "candidate_number": candidate_number,
                **averaged,
                "parameters": json.dumps(
                    parameters,
                    ensure_ascii=False,
                ),
            }
            candidate_records.append(record)
            family_records.append((averaged["rmse"], parameters, averaged))

            if candidate_number % 10 == 0 or candidate_number == len(candidates):
                print(f"  completed {candidate_number}/{len(candidates)}")

        family_records.sort(key=lambda item: item[0])
        selected_by_family[family] = {
            "parameters": family_records[0][1],
            "cv_metrics": family_records[0][2],
        }

    candidate_results = pd.DataFrame(candidate_records).sort_values(
        ["rmse", "mae"],
        kind="stable",
    )

    cv_comparison_records = [
        {
            "model_name": "bsm_closed_form_baseline",
            "family": "bsm",
            **bsm_cv_metrics,
        }
    ]

    for family, selected in selected_by_family.items():
        cv_comparison_records.append(
            {
                "model_name": f"tuned_{family}",
                "family": family,
                **selected["cv_metrics"],
            }
        )

    cv_comparison = pd.DataFrame(cv_comparison_records).sort_values(
        "rmse",
        kind="stable",
    )

    prediction_results = test_data[
        [
            "date",
            "close",
            "strike",
            "moneyness",
            "choice_time_years",
            "maturity_years",
            "heston_synthetic_price",
            "bsm_price",
        ]
    ].copy()

    test_comparison_records = [
        {
            "model_name": "bsm_closed_form_baseline",
            "family": "bsm",
            **calculate_metrics(
                test_data["heston_synthetic_price"],
                test_data["bsm_price"],
            ),
        }
    ]

    for family, selected in selected_by_family.items():
        final_model = create_model(
            family,
            selected["parameters"],
            random_seed=config["random_seed"],
            n_jobs=config["n_jobs"],
        )
        final_model.fit(x_pretest, y_pretest)

        predicted_ratio = final_model.predict(test_data[FEATURE_COLUMNS])
        predicted_price = predicted_ratio * test_data["close"].to_numpy()
        prediction_results[f"{family}_predicted_price"] = predicted_price

        test_comparison_records.append(
            {
                "model_name": f"tuned_{family}",
                "family": family,
                **calculate_metrics(
                    test_data["heston_synthetic_price"],
                    predicted_price,
                ),
            }
        )

        model_bundle = {
            "model": final_model,
            "model_name": f"week_6_tuned_{family}",
            "family": family,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "target_definition": pricing_config["normalized_target_definition"],
            "restore_to_dollars": pricing_config["restore_to_dollars"],
            "best_parameters": selected["parameters"],
            "cv_metrics": selected["cv_metrics"],
            "cv_method": cv_config,
            "sklearn_version": sklearn.__version__,
            "disclosures": config["disclosures"],
        }
        joblib.dump(
            model_bundle,
            MODEL_DIR / f"week_6_tuned_{family}_pricing.joblib",
        )

    test_comparison = pd.DataFrame(test_comparison_records).sort_values(
        "rmse",
        kind="stable",
    )

    selected_record = {
        "selection_metric": config["selection_metric"],
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "cv_method": cv_config,
        "selected_by_family": selected_by_family,
        "sklearn_version": sklearn.__version__,
        "disclosures": config["disclosures"],
    }

    fold_summary.to_csv(
        OUTPUT_DIR / "pricing_cv_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_results.to_csv(
        OUTPUT_DIR / "pricing_cv_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_comparison.to_csv(
        OUTPUT_DIR / "pricing_cv_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_comparison.to_csv(
        OUTPUT_DIR / "pricing_test_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_results.to_csv(
        OUTPUT_DIR / "pricing_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "pricing_selected_hyperparameters.json").write_text(
        json.dumps(selected_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Selected hyperparameters by family")
    for family, selected in selected_by_family.items():
        print(f"- {family}: {json.dumps(selected['parameters'])}")

    print()
    print("Grouped-date cross-validation comparison")
    print(
        cv_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("Final chronological test comparison")
    print(
        test_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("Disclosure: targets are Heston MC synthetic reference prices.")


if __name__ == "__main__":
    main()
