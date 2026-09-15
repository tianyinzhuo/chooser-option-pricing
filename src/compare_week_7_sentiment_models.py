"""Compare the Week 6 GBDT with and without leakage-safe news features."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from src.week_6_cv import make_grouped_date_cv
except ModuleNotFoundError:
    from week_6_cv import make_grouped_date_cv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("CHOOSER_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["CHOOSER_PROJECT_ROOT"]).resolve()

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_synthetic_price_splits"
SENTIMENT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "week_7_feature_dataset_with_sentiment.csv"
)
WEEK6_SELECTION_PATH = (
    PROJECT_ROOT / "outputs" / "week_6" / "pricing_selected_hyperparameters.json"
)
SENTIMENT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "week_7" / "sentiment_feature_config.json"
)
OUTPUT_DIR = Path(
    os.environ.get("WEEK7_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "week_7"))
).resolve()
MODEL_DIR = Path(
    os.environ.get("WEEK7_MODEL_DIR", str(PROJECT_ROOT / "models" / "week_7"))
).resolve()

RANDOM_SEED = 42


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def load_pricing_split(name: str) -> pd.DataFrame:
    path = SPLIT_DIR / f"week_5_e2e_{name}.csv"
    data = pd.read_csv(path, parse_dates=["date"])
    data["close_to_ma20_ratio"] = data["close"] / data["close_ma_20"]
    data["close_to_ma60_ratio"] = data["close"] / data["close_ma_60"]
    data["volume_to_ma20_ratio"] = data["volume"] / data["volume_ma_20"]
    data["heston_price_to_spot"] = data["heston_synthetic_price"] / data["close"]
    return data


def attach_sentiment(data: pd.DataFrame, sentiment_features: list[str]) -> pd.DataFrame:
    sentiment = pd.read_csv(
        SENTIMENT_PATH,
        parse_dates=["date"],
        usecols=["date", *sentiment_features],
    )
    if sentiment["date"].duplicated().any():
        raise ValueError("Sentiment feature dates must be unique.")
    merged = data.merge(sentiment, on="date", how="left", validate="many_to_one")
    if merged[sentiment_features].isna().any().any():
        raise RuntimeError("Missing sentiment values remain after joining pricing data.")
    return merged


def make_model(parameters: dict) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(**parameters, random_state=RANDOM_SEED)


def cross_validate(
    data: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
) -> tuple[dict[str, float], list[dict]]:
    fold_records = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        train = data.iloc[train_indices]
        validation = data.iloc[validation_indices]
        model = make_model(parameters)
        model.fit(train[feature_columns], train["heston_price_to_spot"])
        predicted = model.predict(validation[feature_columns]) * validation["close"].to_numpy()
        metrics = calculate_metrics(
            validation["heston_synthetic_price"].to_numpy(), predicted
        )
        fold_records.append(
            {
                "model_name": model_name,
                "fold": fold_number,
                "train_contracts": len(train),
                "validation_contracts": len(validation),
                "train_end": train["date"].max().date().isoformat(),
                "validation_start": validation["date"].min().date().isoformat(),
                "validation_end": validation["date"].max().date().isoformat(),
                **metrics,
            }
        )
    frame = pd.DataFrame(fold_records)
    summary = {
        "mae": float(frame["mae"].mean()),
        "rmse": float(frame["rmse"].mean()),
        "r2": float(frame["r2"].mean()),
    }
    return summary, fold_records


def fit_and_test(
    pretest: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    parameters: dict,
) -> tuple[GradientBoostingRegressor, np.ndarray, dict[str, float]]:
    model = make_model(parameters)
    model.fit(pretest[feature_columns], pretest["heston_price_to_spot"])
    predicted = model.predict(test[feature_columns]) * test["close"].to_numpy()
    metrics = calculate_metrics(test["heston_synthetic_price"].to_numpy(), predicted)
    return model, predicted, metrics


def main() -> None:
    week6_selection = json.loads(WEEK6_SELECTION_PATH.read_text(encoding="utf-8"))
    sentiment_config = json.loads(SENTIMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    base_features = week6_selection["feature_columns"]
    sentiment_features = sentiment_config["sentiment_features"]
    extended_features = [*base_features, *sentiment_features]
    parameters = week6_selection["selected_by_family"]["gbdt"]["parameters"]

    train = attach_sentiment(load_pricing_split("train"), sentiment_features)
    validation = attach_sentiment(load_pricing_split("validation"), sentiment_features)
    test = attach_sentiment(load_pricing_split("test"), sentiment_features)
    pretest = pd.concat([train, validation], ignore_index=True).sort_values(
        ["date", "moneyness", "choice_time_years"], kind="stable"
    ).reset_index(drop=True)
    test = test.sort_values(
        ["date", "moneyness", "choice_time_years"], kind="stable"
    ).reset_index(drop=True)

    folds = make_grouped_date_cv(pretest["date"], n_splits=4, gap_groups=0)
    model_specs = {
        "gbdt_without_sentiment": base_features,
        "gbdt_with_sentiment": extended_features,
    }

    cv_records = []
    cv_summaries = []
    test_records = []
    predictions = pd.DataFrame(
        {
            "date": test["date"],
            "close": test["close"],
            "strike": test["strike"],
            "moneyness": test["moneyness"],
            "choice_time_years": test["choice_time_years"],
            "actual_heston_synthetic_price": test["heston_synthetic_price"],
        }
    )

    for model_name, feature_columns in model_specs.items():
        cv_summary, fold_records = cross_validate(
            pretest, feature_columns, parameters, folds, model_name
        )
        cv_records.extend(fold_records)
        cv_summaries.append(
            {
                "model_name": model_name,
                "feature_count": len(feature_columns),
                **cv_summary,
            }
        )

        model, predicted, metrics = fit_and_test(
            pretest, test, feature_columns, parameters
        )
        predictions[f"{model_name}_predicted_price"] = predicted
        predictions[f"{model_name}_absolute_error"] = np.abs(
            test["heston_synthetic_price"].to_numpy() - predicted
        )
        test_records.append(
            {
                "model_name": model_name,
                "feature_count": len(feature_columns),
                **metrics,
            }
        )

        bundle = {
            "model": model,
            "model_name": model_name,
            "family": "gbdt",
            "feature_columns": feature_columns,
            "target_column": "heston_price_to_spot",
            "best_parameters": parameters,
            "sklearn_version": sklearn.__version__,
            "disclosures": [
                "The pricing label is a Heston Monte Carlo synthetic reference price.",
                "Sentiment features are lagged to the next trading day.",
                "The test period has been inspected and remains exploratory.",
            ],
        }
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / f"week_7_{model_name}.joblib")
        joblib.dump(bundle, MODEL_DIR / f"week_7_{model_name}_bundle.joblib")

        if model_name == "gbdt_with_sentiment":
            importance = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance": model.feature_importances_,
                    "feature_group": [
                        "sentiment" if feature in sentiment_features else "market"
                        for feature in feature_columns
                    ],
                }
            ).sort_values("importance", ascending=False)

    cv_comparison = pd.DataFrame(cv_summaries).sort_values("rmse")
    cv_folds = pd.DataFrame(cv_records)
    test_comparison = pd.DataFrame(test_records).sort_values("rmse")

    base_test = next(item for item in test_records if item["model_name"] == "gbdt_without_sentiment")
    extended_test = next(item for item in test_records if item["model_name"] == "gbdt_with_sentiment")
    base_cv = next(item for item in cv_summaries if item["model_name"] == "gbdt_without_sentiment")
    extended_cv = next(item for item in cv_summaries if item["model_name"] == "gbdt_with_sentiment")
    incremental = {
        "comparison": "same GBDT parameters, with versus without lagged sentiment features",
        "base_feature_count": len(base_features),
        "sentiment_feature_count": len(sentiment_features),
        "extended_feature_count": len(extended_features),
        "cv_rmse_improvement_percent": float(
            (base_cv["rmse"] - extended_cv["rmse"]) / base_cv["rmse"] * 100
        ),
        "test_rmse_improvement_percent": float(
            (base_test["rmse"] - extended_test["rmse"]) / base_test["rmse"] * 100
        ),
        "test_mae_improvement_percent": float(
            (base_test["mae"] - extended_test["mae"]) / base_test["mae"] * 100
        ),
        "sentiment_improved_cv_rmse": bool(extended_cv["rmse"] < base_cv["rmse"]),
        "sentiment_improved_test_rmse": bool(extended_test["rmse"] < base_test["rmse"]),
        "disclosures": [
            "The target is a Heston Monte Carlo synthetic reference price, not an OTC premium.",
            "The same fixed Week 6 GBDT hyperparameters are used for a controlled comparison.",
            "The existing test period has already been inspected, so results are exploratory.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv_comparison.to_csv(
        OUTPUT_DIR / "week_7_sentiment_model_cv_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_folds.to_csv(
        OUTPUT_DIR / "week_7_sentiment_model_cv_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_comparison.to_csv(
        OUTPUT_DIR / "week_7_sentiment_model_test_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        OUTPUT_DIR / "week_7_sentiment_model_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    importance.to_csv(
        OUTPUT_DIR / "week_7_sentiment_model_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "week_7_sentiment_incremental_value.json").write_text(
        json.dumps(incremental, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Week 7 controlled sentiment-feature comparison completed.")
    print(f"Pre-test contracts: {len(pretest)} | dates: {pretest['date'].nunique()}")
    print(f"Test contracts: {len(test)} | dates: {test['date'].nunique()}")
    print(f"Base features: {len(base_features)}")
    print(f"Added sentiment features: {len(sentiment_features)}")
    print()
    print("Grouped-date cross-validation comparison")
    print(cv_comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("Chronological test comparison")
    print(test_comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(
        "CV RMSE improvement from sentiment: "
        f"{incremental['cv_rmse_improvement_percent']:.2f}%"
    )
    print(
        "Test RMSE improvement from sentiment: "
        f"{incremental['test_rmse_improvement_percent']:.2f}%"
    )
    print()
    print("Top 10 extended-model feature importances")
    print(importance.head(10).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("Disclosure: targets are Heston MC synthetic reference prices.")


if __name__ == "__main__":
    main()
