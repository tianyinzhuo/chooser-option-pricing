"""Create Week 6 comparison charts for the final report."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_6"

DISPLAY_NAMES = {
    "bsm_closed_form_baseline": "BSM baseline",
    "route_a_ml_volatility_plus_bsm": "ML volatility + BSM",
    "tuned_ridge": "Ridge",
    "tuned_random_forest": "Random Forest",
    "tuned_gbdt": "GBDT",
    "tuned_mlp": "MLP",
}


def add_value_labels(axis, bars, decimals=2) -> None:
    """Add readable values above vertical bars."""
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_test_metrics(test_results: pd.DataFrame) -> Path:
    """Compare final test MAE and RMSE across all pricing approaches."""
    ordered = test_results.sort_values("rmse").copy()
    names = ordered["model_name"].map(DISPLAY_NAMES)
    positions = np.arange(len(ordered))
    width = 0.36

    figure, axis = plt.subplots(figsize=(12, 7))
    mae_bars = axis.bar(
        positions - width / 2,
        ordered["mae"],
        width,
        label="MAE",
        color="#4C78A8",
    )
    rmse_bars = axis.bar(
        positions + width / 2,
        ordered["rmse"],
        width,
        label="RMSE",
        color="#F58518",
    )
    axis.set_title("Week 6 final test pricing-error comparison")
    axis.set_ylabel("Error in USD (lower is better)")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=22, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_ylim(0, ordered["rmse"].max() * 1.23)
    add_value_labels(axis, mae_bars)
    add_value_labels(axis, rmse_bars)

    figure.tight_layout()
    path = OUTPUT_DIR / "week_6_final_test_metric_comparison.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_cv_vs_test(
    cv_results: pd.DataFrame,
    test_results: pd.DataFrame,
) -> Path:
    """Compare CV RMSE with test RMSE for approaches available in both."""
    comparable_names = [
        "tuned_ridge",
        "tuned_mlp",
        "tuned_gbdt",
        "tuned_random_forest",
        "bsm_closed_form_baseline",
    ]

    cv_lookup = cv_results.set_index("model_name")
    test_lookup = test_results.set_index("model_name")

    cv_values = [cv_lookup.loc[name, "rmse"] for name in comparable_names]
    test_values = [test_lookup.loc[name, "rmse"] for name in comparable_names]
    labels = [DISPLAY_NAMES[name] for name in comparable_names]

    positions = np.arange(len(comparable_names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(11, 7))
    cv_bars = axis.bar(
        positions - width / 2,
        cv_values,
        width,
        label="Grouped-date CV RMSE",
        color="#72B7B2",
    )
    test_bars = axis.bar(
        positions + width / 2,
        test_values,
        width,
        label="Chronological test RMSE",
        color="#E45756",
    )

    axis.set_title("Model stability: cross-validation versus test RMSE")
    axis.set_ylabel("RMSE in USD (lower is better)")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_ylim(0, max(max(cv_values), max(test_values)) * 1.25)
    add_value_labels(axis, cv_bars)
    add_value_labels(axis, test_bars)

    figure.tight_layout()
    path = OUTPUT_DIR / "week_6_cv_vs_test_rmse.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_gbdt_predictions(predictions: pd.DataFrame) -> Path:
    """Plot GBDT predicted prices against synthetic reference prices."""
    actual = predictions["heston_synthetic_price"]
    predicted = predictions["gbdt_predicted_price"]

    lower = min(actual.min(), predicted.min()) - 1
    upper = max(actual.max(), predicted.max()) + 1

    figure, axis = plt.subplots(figsize=(8, 8))

    for moneyness, group in predictions.groupby("moneyness"):
        axis.scatter(
            group["heston_synthetic_price"],
            group["gbdt_predicted_price"],
            s=58,
            alpha=0.78,
            label=f"K/S = {moneyness:.2f}",
        )

    axis.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
        linewidth=1.3,
        label="Perfect prediction",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("GBDT test predictions versus Heston synthetic prices")
    axis.set_xlabel("Heston MC synthetic reference price (USD)")
    axis.set_ylabel("GBDT predicted price (USD)")
    axis.grid(alpha=0.22)
    axis.legend()

    figure.tight_layout()
    path = OUTPUT_DIR / "week_6_gbdt_test_scatter.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_volatility_stability(
    cv_results: pd.DataFrame,
    test_results: pd.DataFrame,
) -> Path:
    """Show the volatility model's CV and test RMSE side by side."""
    model_order = [
        "persistence_baseline_rolling_vol_60",
        "tuned_random_forest",
    ]
    labels = ["60-day persistence", "Tuned Random Forest"]
    cv_lookup = cv_results.set_index("model_name")
    test_lookup = test_results.set_index("model_name")
    cv_values = [cv_lookup.loc[name, "rmse"] for name in model_order]
    test_values = [test_lookup.loc[name, "rmse"] for name in model_order]

    positions = np.arange(2)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 6))
    cv_bars = axis.bar(
        positions - width / 2,
        cv_values,
        width,
        label="Time-series CV RMSE",
        color="#54A24B",
    )
    test_bars = axis.bar(
        positions + width / 2,
        test_values,
        width,
        label="Chronological test RMSE",
        color="#B279A2",
    )

    axis.set_title("Volatility forecast stability comparison")
    axis.set_ylabel("Annualized volatility RMSE")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_ylim(0, max(max(cv_values), max(test_values)) * 1.25)
    add_value_labels(axis, cv_bars, decimals=3)
    add_value_labels(axis, test_bars, decimals=3)

    figure.tight_layout()
    path = OUTPUT_DIR / "week_6_volatility_cv_vs_test_rmse.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_results = pd.read_csv(
        OUTPUT_DIR / "final_pricing_approach_test_comparison.csv"
    )
    cv_results = pd.read_csv(
        OUTPUT_DIR / "pricing_cv_model_comparison.csv"
    )
    predictions = pd.read_csv(
        OUTPUT_DIR / "final_pricing_approach_test_predictions.csv"
    )
    volatility_cv = pd.read_csv(
        OUTPUT_DIR / "volatility_cv_model_comparison.csv"
    )
    volatility_test = pd.read_csv(
        OUTPUT_DIR / "volatility_test_model_comparison.csv"
    )

    required_prediction_columns = {
        "heston_synthetic_price",
        "gbdt_predicted_price",
        "moneyness",
    }
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError("Required GBDT prediction columns are missing.")

    paths = [
        plot_test_metrics(test_results),
        plot_cv_vs_test(cv_results, test_results),
        plot_gbdt_predictions(predictions),
        plot_volatility_stability(volatility_cv, volatility_test),
    ]

    print("Week 6 comparison charts saved:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
