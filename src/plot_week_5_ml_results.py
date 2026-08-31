from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_5"

metrics = pd.read_csv(OUTPUT_DIR / "normalized_pricing_test_metrics.csv")
importance = pd.read_csv(
    OUTPUT_DIR / "normalized_pricing_feature_importance.csv"
)
predictions = pd.read_csv(
    OUTPUT_DIR / "normalized_pricing_test_predictions.csv"
)

# Chart 1: BSM versus normalized end-to-end GBDT test metrics.
plt.figure(figsize=(8, 5))

x_positions = range(len(metrics))
bar_width = 0.35

plt.bar(
    [x - bar_width / 2 for x in x_positions],
    metrics["mae"],
    width=bar_width,
    color="#4C78A8",
    label="MAE",
)
plt.bar(
    [x + bar_width / 2 for x in x_positions],
    metrics["rmse"],
    width=bar_width,
    color="#F2A541",
    label="RMSE",
)

labels = ["BSM baseline", "Normalized GBDT"]
plt.xticks(list(x_positions), labels)
plt.ylabel("Difference from Heston MC synthetic reference")
plt.title("Week 5 End-to-End Pricing Test Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "e2e_test_metric_comparison.png", dpi=200)
plt.close()

# Chart 2: Top feature importances.
top_features = importance.head(10).sort_values("importance")

plt.figure(figsize=(8, 5.5))
plt.barh(
    top_features["feature"],
    top_features["importance"],
    color="#59A14F",
)
plt.xlabel("Feature importance")
plt.title("Normalized GBDT: Top 10 Feature Importances")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "normalized_pricing_feature_importance.png",
    dpi=200,
)
plt.close()

# Chart 3: Predicted price versus Heston MC synthetic reference.
actual = predictions["heston_synthetic_price"]
predicted = predictions["normalized_gbdt_predicted_price"]

lower_limit = min(actual.min(), predicted.min()) - 1
upper_limit = max(actual.max(), predicted.max()) + 1

plt.figure(figsize=(7, 6))
plt.scatter(
    actual,
    predicted,
    color="#E15759",
    alpha=0.75,
    s=50,
)
plt.plot(
    [lower_limit, upper_limit],
    [lower_limit, upper_limit],
    linestyle="--",
    color="black",
    label="Equal-price line",
)

plt.xlim(lower_limit, upper_limit)
plt.ylim(lower_limit, upper_limit)
plt.xlabel("Heston MC synthetic reference price")
plt.ylabel("Normalized GBDT predicted price")
plt.title("Normalized GBDT Test Predictions")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "normalized_pricing_test_scatter.png",
    dpi=200,
)
plt.close()

print("Charts saved to:")
print(OUTPUT_DIR / "e2e_test_metric_comparison.png")
print(OUTPUT_DIR / "normalized_pricing_feature_importance.png")
print(OUTPUT_DIR / "normalized_pricing_test_scatter.png")