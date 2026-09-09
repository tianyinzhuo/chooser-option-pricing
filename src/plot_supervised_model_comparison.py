from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_5"

metrics = pd.read_csv(
    OUTPUT_DIR / "supervised_model_test_comparison.csv"
)

display_names = {
    "bsm_closed_form_baseline": "BSM",
    "ridge_alpha_0_1": "Ridge",
    "random_forest_conservative": "Random Forest",
    "gbdt_balanced": "Normalized GBDT",
    "mlp_medium": "MLP",
}

order = [
    "bsm_closed_form_baseline",
    "ridge_alpha_0_1",
    "random_forest_conservative",
    "gbdt_balanced",
    "mlp_medium",
]

metrics["display_name"] = metrics["model_name"].map(display_names)
metrics["sort_order"] = metrics["model_name"].map(
    {name: index for index, name in enumerate(order)}
)
metrics = metrics.sort_values("sort_order")

x_positions = range(len(metrics))
bar_width = 0.35

plt.figure(figsize=(10, 5.5))

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

plt.xticks(list(x_positions), metrics["display_name"], rotation=12)
plt.ylabel("Difference from Heston MC synthetic reference")
plt.title("Week 5: Supervised Pricing Model Test Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

output_path = OUTPUT_DIR / "supervised_model_test_comparison.png"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Chart saved to: {output_path}")