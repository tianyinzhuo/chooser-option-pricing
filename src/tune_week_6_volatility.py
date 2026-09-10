"""Week 6: tune the volatility Random Forest with time-series CV."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV

from src.week_6_cv import make_volatility_cv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_splits"
DATA_CONFIG_PATH = PROJECT_ROOT / "config" / "week_5_ml_data_config.json"
SEARCH_CONFIG_PATH = PROJECT_ROOT / "config" / "week_6_model_search.json"

MODEL_DIR = PROJECT_ROOT / "models" / "week_6"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_6"


def load_split(name: str) -> pd.DataFrame:
    """Load one Week 5 chronological split."""
    path = SPLIT_DIR / f"{name}.csv"
    return pd.read_csv(path, parse_dates=["date"])


def calculate_metrics(actual, predicted) -> dict:
    """Calculate MAE, RMSE and R-squared."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_config = json.loads(
        DATA_CONFIG_PATH.read_text(encoding="utf-8")
    )
    search_config = json.loads(
        SEARCH_CONFIG_PATH.read_text(encoding="utf-8-sig")
    )

    feature_columns = data_config["feature_columns"]
    target_column = data_config["target"]

    volatility_config = search_config["volatility_route"]
    cv_config = search_config["cross_validation"]["volatility"]

    if target_column != volatility_config["target_column"]:
        raise ValueError(
            "The Week 5 target does not match the Week 6 configuration."
        )

    if search_config["selection_metric"] != "rmse":
        raise ValueError(
            "This script expects RMSE as the selection metric."
        )

    train_data = load_split("week_5_train")
    validation_data = load_split("week_5_validation")
    test_data = load_split("week_5_test")

    pretest_data = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)

    if pretest_data["date"].max() >= test_data["date"].min():
        raise ValueError(
            "Pre-test data must end before the test period starts."
        )

    if pretest_data[feature_columns + [target_column]].isna().any().any():
        raise ValueError("Pre-test data contains missing values.")

    if test_data[feature_columns + [target_column]].isna().any().any():
        raise ValueError("Test data contains missing values.")

    cv_folds = make_volatility_cv(
        n_rows=len(pretest_data),
        n_splits=cv_config["n_splits"],
        gap_rows=cv_config["gap_rows"],
    )

    fold_records = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cv_folds,
        start=1,
    ):
        actual_gap = (
            validation_indices.min()
            - train_indices.max()
            - 1
        )

        fold_records.append(
            {
                "fold": fold_number,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "gap_rows": actual_gap,
                "train_start": pretest_data.iloc[
                    train_indices
                ]["date"].min(),
                "train_end": pretest_data.iloc[
                    train_indices
                ]["date"].max(),
                "validation_start": pretest_data.iloc[
                    validation_indices
                ]["date"].min(),
                "validation_end": pretest_data.iloc[
                    validation_indices
                ]["date"].max(),
            }
        )

    fold_summary = pd.DataFrame(fold_records)

    x_pretest = pretest_data[feature_columns]
    y_pretest = pretest_data[target_column]

    parameter_distributions = (
        volatility_config["parameter_distributions"]
    )

    # During search, parallelism is controlled by RandomizedSearchCV.
    # Each individual forest therefore uses one worker.
    search_estimator = RandomForestRegressor(
        random_state=search_config["random_seed"],
        n_jobs=1,
    )

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    search = RandomizedSearchCV(
        estimator=search_estimator,
        param_distributions=parameter_distributions,
        n_iter=volatility_config["n_iter"],
        scoring=scoring,
        refit="rmse",
        cv=cv_folds,
        random_state=search_config["random_seed"],
        n_jobs=search_config["n_jobs"],
        return_train_score=False,
        verbose=1,
    )

    print("Starting Week 6 volatility hyperparameter search...")
    print(
        f"Pre-test rows: {len(pretest_data)} | "
        f"Test rows kept separate: {len(test_data)}"
    )
    print(
        f"Candidates: {volatility_config['n_iter']} | "
        f"CV folds: {len(cv_folds)}"
    )
    print()

    search.fit(x_pretest, y_pretest)

    raw_results = search.cv_results_

    search_results = pd.DataFrame(
        {
            "rank_rmse": raw_results["rank_test_rmse"],
            "mean_cv_mae": -raw_results["mean_test_mae"],
            "std_cv_mae": raw_results["std_test_mae"],
            "mean_cv_rmse": -raw_results["mean_test_rmse"],
            "std_cv_rmse": raw_results["std_test_rmse"],
            "mean_cv_r2": raw_results["mean_test_r2"],
            "std_cv_r2": raw_results["std_test_r2"],
            "parameters": [
                json.dumps(item, ensure_ascii=False)
                for item in raw_results["params"]
            ],
        }
    ).sort_values(
        ["rank_rmse", "mean_cv_rmse"],
        kind="stable",
    )

    best_index = search.best_index_

    baseline_column = volatility_config[
        "persistence_baseline_column"
    ]

    # Calculate the baseline separately in every validation fold.
    # This matches the averaging convention used by RandomizedSearchCV.
    baseline_fold_metrics = []

    for _, validation_indices in cv_folds:
        fold_actual = y_pretest.iloc[validation_indices]
        fold_prediction = pretest_data.iloc[
            validation_indices
        ][baseline_column]

        baseline_fold_metrics.append(
            calculate_metrics(
                fold_actual,
                fold_prediction,
            )
        )

    baseline_cv_metrics = {
        metric_name: float(
            np.mean(
                [
                    fold_metrics[metric_name]
                    for fold_metrics in baseline_fold_metrics
                ]
            )
        )
        for metric_name in ["mae", "rmse", "r2"]
    }

    best_cv_metrics = {
        "mae": float(-raw_results["mean_test_mae"][best_index]),
        "rmse": float(-raw_results["mean_test_rmse"][best_index]),
        "r2": float(raw_results["mean_test_r2"][best_index]),
    }

    cv_comparison = pd.DataFrame(
        [
            {
                "model_name": (
                    "persistence_baseline_rolling_vol_60"
                ),
                **baseline_cv_metrics,
            },
            {
                "model_name": "tuned_random_forest",
                **best_cv_metrics,
            },
        ]
    ).sort_values("rmse")

    # Refit the selected settings on all pre-test data.
    final_model = RandomForestRegressor(
        **search.best_params_,
        random_state=search_config["random_seed"],
        n_jobs=search_config["n_jobs"],
    )

    final_model.fit(x_pretest, y_pretest)

    x_test = test_data[feature_columns]
    y_test = test_data[target_column]

    rf_test_prediction = final_model.predict(x_test)
    baseline_test_prediction = test_data[baseline_column]

    test_comparison = pd.DataFrame(
        [
            {
                "model_name": (
                    "persistence_baseline_rolling_vol_60"
                ),
                **calculate_metrics(
                    y_test,
                    baseline_test_prediction,
                ),
            },
            {
                "model_name": "tuned_random_forest",
                **calculate_metrics(
                    y_test,
                    rf_test_prediction,
                ),
            },
        ]
    ).sort_values("rmse")

    test_predictions = pd.DataFrame(
        {
            "date": test_data["date"],
            "actual_future_realized_vol_20": y_test,
            "persistence_prediction": baseline_test_prediction,
            "tuned_rf_prediction": rf_test_prediction,
        }
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": final_model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    selected_record = {
        "model_name": "week_6_tuned_volatility_random_forest",
        "route_name": volatility_config["route_name"],
        "selection_metric": "rmse",
        "best_parameters": search.best_params_,
        "best_cv_metrics": best_cv_metrics,
        "test_metrics": calculate_metrics(
            y_test,
            rf_test_prediction,
        ),
        "cv_method": cv_config,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "pretest_rows": len(pretest_data),
        "test_rows": len(test_data),
        "pretest_date_range": [
            pretest_data["date"].min().date().isoformat(),
            pretest_data["date"].max().date().isoformat(),
        ],
        "test_date_range": [
            test_data["date"].min().date().isoformat(),
            test_data["date"].max().date().isoformat(),
        ],
        "sklearn_version": sklearn.__version__,
        "disclosures": search_config["disclosures"],
    }

    model_bundle = {
        "model": final_model,
        "model_name": selected_record["model_name"],
        "route_name": volatility_config["route_name"],
        "feature_columns": feature_columns,
        "target_column": target_column,
        "best_parameters": search.best_params_,
        "cv_method": cv_config,
        "sklearn_version": sklearn.__version__,
        "disclosures": search_config["disclosures"],
    }

    model_path = (
        MODEL_DIR
        / "week_6_tuned_volatility_random_forest.joblib"
    )

    joblib.dump(model_bundle, model_path)

    fold_summary.to_csv(
        OUTPUT_DIR / "volatility_cv_fold_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    search_results.to_csv(
        OUTPUT_DIR / "volatility_cv_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_comparison.to_csv(
        OUTPUT_DIR / "volatility_cv_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_comparison.to_csv(
        OUTPUT_DIR / "volatility_test_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_predictions.to_csv(
        OUTPUT_DIR / "volatility_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_importance.to_csv(
        OUTPUT_DIR / "volatility_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        OUTPUT_DIR
        / "volatility_selected_hyperparameters.json"
    ).write_text(
        json.dumps(
            selected_record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Best parameters")
    print(
        json.dumps(
            search.best_params_,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("Cross-validation comparison")
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


if __name__ == "__main__":
    main()
