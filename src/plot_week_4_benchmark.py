from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_4"

results = pd.read_csv(OUTPUT_DIR / "bsm_heston_scenario_results.csv")
metrics = pd.read_csv(OUTPUT_DIR / "bsm_heston_benchmark_metrics.csv")

# Figure 1: BSM price versus Heston Monte Carlo reference.
plt.figure(figsize=(8, 6))

for regime, color, label in [
    ("normal_or_low", "#4C78A8", "Normal / low volatility"),
    ("high", "#E45756", "High volatility"),
]:
    subset = results[results["volatility_regime"] == regime]
    plt.scatter(
        subset["heston_price"],
        subset["bsm_price"],
        color=color,
        s=60,
        alpha=0.8,
        label=label,
    )

lower_limit = min(results["heston_price"].min(), results["bsm_price"].min()) - 1
upper_limit = max(results["heston_price"].max(), results["bsm_price"].max()) + 1

plt.plot(
    [lower_limit, upper_limit],
    [lower_limit, upper_limit],
    linestyle="--",
    color="black",
    label="Equal-price line",
)

plt.xlim(lower_limit, upper_limit)
plt.ylim(lower_limit, upper_limit)
plt.xlabel("Heston Monte Carlo synthetic reference price")
plt.ylabel("BSM simple chooser price")
plt.title("BSM versus Heston Synthetic Benchmark")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bsm_vs_heston_scatter.png", dpi=200)
plt.close()

# Figure 2: MAE and RMSE by volatility regime.
plot_metrics = metrics[metrics["group"] != "all_scenarios"].copy()
plot_metrics["group"] = plot_metrics["group"].replace(
    {
        "normal_or_low_volatility": "Normal / low",
        "high_volatility": "High",
    }
)

x_positions = range(len(plot_metrics))
bar_width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(
    [position - bar_width / 2 for position in x_positions],
    plot_metrics["mae"],
    width=bar_width,
    label="MAE",
    color="#4C78A8",
)
plt.bar(
    [position + bar_width / 2 for position in x_positions],
    plot_metrics["rmse"],
    width=bar_width,
    label="RMSE",
    color="#F2A541",
)

plt.xticks(list(x_positions), plot_metrics["group"])
plt.ylabel("Difference from Heston MC reference")
plt.title("BSM Benchmark Difference by Volatility Regime")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_metrics_by_volatility_regime.png", dpi=200)
plt.close()

print("Charts saved to:")
print(OUTPUT_DIR / "bsm_vs_heston_scatter.png")
print(OUTPUT_DIR / "error_metrics_by_volatility_regime.png")