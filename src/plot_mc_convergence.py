from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_4"

data = pd.read_csv(OUTPUT_DIR / "mc_convergence_results.csv")

reference_price = data.loc[
    data["num_paths"] == 50000,
    "heston_price",
].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# Left: price convergence.
axes[0].plot(
    data["num_paths"],
    data["heston_price"],
    marker="o",
    color="#4C78A8",
    linewidth=2,
)
axes[0].axhline(
    reference_price,
    linestyle="--",
    color="#E45756",
    label="50,000-path reference",
)
axes[0].set_xscale("log")
axes[0].set_xlabel("Number of Monte Carlo paths (log scale)")
axes[0].set_ylabel("Heston chooser price")
axes[0].set_title("Monte Carlo Price Convergence")
axes[0].grid(alpha=0.25)
axes[0].legend()

# Right: standard error convergence.
axes[1].plot(
    data["num_paths"],
    data["mc_standard_error"],
    marker="o",
    color="#F2A541",
    linewidth=2,
)
axes[1].set_xscale("log")
axes[1].set_xlabel("Number of Monte Carlo paths (log scale)")
axes[1].set_ylabel("Monte Carlo standard error")
axes[1].set_title("Monte Carlo Standard Error Decline")
axes[1].grid(alpha=0.25)

plt.tight_layout()

output_path = OUTPUT_DIR / "mc_convergence_analysis.png"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Convergence chart saved to: {output_path}")