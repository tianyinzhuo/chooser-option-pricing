"""Week 6: evaluate ML-predicted volatility inserted into BSM."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.bsm_chooser import simple_chooser_price


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_synthetic_price_splits"
    / "week_5_e2e_test.csv"
)
VOLATILITY_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "week_6"
    / "week_6_tuned_volatility_random_forest.joblib"
)
PRICING_COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "week_6"
    / "pricing_test_model_comparison.csv"
)
PRICING_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "week_6"
    / "pricing_test_predictions.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_6"


def calculate_metrics(actual, predicted) -> dict:
    """Calculate comparable dollar-price regression metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_data = pd.read_csv(TEST_PATH, parse_dates=["date"])
    volatility_bundle = joblib.load(VOLATILITY_MODEL_PATH)
    volatility_model = volatility_bundle["model"]
    volatility_features = volatility_bundle["feature_columns"]

    if test_data[volatility_features].isna().any().any():
        raise ValueError("Route A input features contain missing values.")

    predicted_volatility = volatility_model.predict(
        test_data[volatility_features]
    )

    if (predicted_volatility <= 0).any():
        raise ValueError("Predicted volatility must be strictly positive.")

    route_a_prices = []

    for row_number, row in test_data.iterrows():
        result = simple_chooser_price(
            spot_price=float(row["close"]),
            strike=float(row["strike"]),
            choice_time_years=float(row["choice_time_years"]),
            maturity_years=float(row["maturity_years"]),
            risk_free_rate=float(row["treasury_10y"]) / 100.0,
            volatility=float(predicted_volatility[row_number]),
            dividend_yield=0.0,
        )
        route_a_prices.append(result["chooser_price"])

    route_a_metrics = calculate_metrics(
        test_data["heston_synthetic_price"],
        route_a_prices,
    )

    route_a_metric_table = pd.DataFrame(
        [
            {
                "model_name": "route_a_ml_volatility_plus_bsm",
                "family": "hybrid_ml_bsm",
                **route_a_metrics,
            }
        ]
    )

    route_a_predictions = test_data[
        [
            "date",
            "close",
            "strike",
            "moneyness",
            "choice_time_years",
            "maturity_years",
            "rolling_vol_60",
            "bsm_price",
            "heston_synthetic_price",
        ]
    ].copy()
    route_a_predictions["ml_predicted_future_vol_20"] = predicted_volatility
    route_a_predictions["route_a_predicted_price"] = route_a_prices
    route_a_predictions["route_a_error"] = (
        route_a_predictions["route_a_predicted_price"]
        - route_a_predictions["heston_synthetic_price"]
    )

    pricing_comparison = pd.read_csv(PRICING_COMPARISON_PATH)
    final_comparison = pd.concat(
        [pricing_comparison, route_a_metric_table],
        ignore_index=True,
    ).sort_values("rmse", kind="stable")

    pricing_predictions = pd.read_csv(
        PRICING_PREDICTIONS_PATH,
        parse_dates=["date"],
    )

    merge_keys = [
        "date",
        "close",
        "strike",
        "moneyness",
        "choice_time_years",
        "maturity_years",
    ]

    final_predictions = pricing_predictions.merge(
        route_a_predictions[
            [
                *merge_keys,
                "ml_predicted_future_vol_20",
                "route_a_predicted_price",
            ]
        ],
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    if final_predictions["route_a_predicted_price"].isna().any():
        raise ValueError("Route A predictions did not merge completely.")

    bsm_metrics = final_comparison.loc[
        final_comparison["model_name"] == "bsm_closed_form_baseline"
    ].iloc[0]

    summary = {
        "route_name": "route_a_ml_volatility_plus_bsm",
        "volatility_model": volatility_bundle["model_name"],
        "volatility_target": volatility_bundle["target_column"],
        "test_contracts": len(test_data),
        "test_unique_dates": int(test_data["date"].nunique()),
        "route_a_metrics": route_a_metrics,
        "bsm_baseline_metrics": {
            "mae": float(bsm_metrics["mae"]),
            "rmse": float(bsm_metrics["rmse"]),
            "r2": float(bsm_metrics["r2"]),
        },
        "rmse_improvement_over_bsm_percent": float(
            (bsm_metrics["rmse"] - route_a_metrics["rmse"])
            / bsm_metrics["rmse"]
            * 100.0
        ),
        "methodological_caveat": (
            "The ML model predicts annualized volatility over the next 20 trading "
            "days, while the chooser contracts have a one-year maturity. Route A "
            "therefore treats the short-horizon forecast as a constant annualized "
            "BSM volatility and should be interpreted as an exploratory hybrid."
        ),
        "target_disclosure": (
            "Heston Monte Carlo prices are synthetic reference labels, not real "
            "OTC transaction premiums."
        ),
    }

    route_a_metric_table.to_csv(
        OUTPUT_DIR / "route_a_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    route_a_predictions.to_csv(
        OUTPUT_DIR / "route_a_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_comparison.to_csv(
        OUTPUT_DIR / "final_pricing_approach_test_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_predictions.to_csv(
        OUTPUT_DIR / "final_pricing_approach_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "route_a_evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Route A: ML volatility plus BSM")
    print(route_a_metric_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print("Final test-set pricing approach comparison")
    print(final_comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(
        "Route A RMSE improvement over original BSM: "
        f"{summary['rmse_improvement_over_bsm_percent']:.2f}%"
    )
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("Caveat: the volatility forecast horizon is 20 trading days, while T2 is one year.")
    print("Disclosure: targets are Heston MC synthetic reference prices.")


if __name__ == "__main__":
    main()
