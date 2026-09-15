"""Create Week 7 extreme-scenario comparison charts."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("CHOOSER_PROJECT_ROOT", DEFAULT_ROOT)).resolve()
OUTPUT_DIR = Path(
    os.environ.get("WEEK7_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "week_7"))
).resolve()

SUMMARY_PATH = OUTPUT_DIR / "week_7_scenario_summary.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "week_7_scenario_contract_predictions.csv"

SCENARIO_ORDER = [
    "baseline",
    "underlying_volatility_spike_50pct",
    "vix_spike_50pct",
    "rate_hike_2_percentage_points",
    "combined_stress",
]

DISPLAY_NAMES = {
    "baseline": "基准",
    "underlying_volatility_spike_50pct": "历史波动率\n提高50%",
    "vix_spike_50pct": "VIX\n提高50%",
    "rate_hike_2_percentage_points": "利率提高\n2个百分点",
    "combined_stress": "组合压力\n情景",
}

COLORS = ["#6B7280", "#2F6B9A", "#E69F00", "#4C956C", "#C44E52"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    result = summary.copy()
    result["scenario_order"] = result["scenario"].map(order)
    if result["scenario_order"].isna().any():
        unknown = result.loc[result["scenario_order"].isna(), "scenario"].tolist()
        raise ValueError(f"Unknown scenarios in summary: {unknown}")
    return result.sort_values("scenario_order")


def label_bars(ax, bars, decimals: int = 2, suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.{decimals}f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_mean_prices(summary: pd.DataFrame) -> Path:
    labels = [DISPLAY_NAMES[name] for name in summary["scenario"]]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(labels, summary["mean_price"], color=COLORS, width=0.68)
    label_bars(ax, bars)
    ax.axhline(
        summary.iloc[0]["mean_price"],
        color="#374151",
        linestyle="--",
        linewidth=1.2,
        label="基准平均价格",
    )
    ax.set_title("第七周极端情景下的平均选择权价格", fontsize=15, pad=14)
    ax.set_ylabel("GBDT预测价格（美元）")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_scenario_mean_price_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_percentage_changes(summary: pd.DataFrame) -> Path:
    stressed = summary[summary["scenario"] != "baseline"].copy()
    labels = [DISPLAY_NAMES[name] for name in stressed["scenario"]]
    colors = [COLORS[SCENARIO_ORDER.index(name)] for name in stressed["scenario"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    bars = ax.bar(labels, stressed["mean_percentage_change"], color=colors, width=0.65)
    label_bars(ax, bars, suffix="%")
    ax.axhline(0, color="#374151", linewidth=1)
    ax.set_title("各压力情景相对基准价格的平均变化", fontsize=15, pad=14)
    ax.set_ylabel("平均价格变化（%）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_scenario_mean_percentage_change.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_change_distribution(predictions: pd.DataFrame) -> Path:
    stressed_order = SCENARIO_ORDER[1:]
    values = [
        predictions.loc[predictions["scenario"] == name, "price_change"].to_numpy()
        for name in stressed_order
    ]
    if any(len(item) == 0 for item in values):
        raise ValueError("At least one stress scenario has no contract predictions.")

    labels = [DISPLAY_NAMES[name] for name in stressed_order]
    fig, ax = plt.subplots(figsize=(10, 5.9))
    box = ax.boxplot(values, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], COLORS[1:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for median in box["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.6)
    for mean in box["means"]:
        mean.set_marker("D")
        mean.set_markerfacecolor("white")
        mean.set_markeredgecolor("#111827")
    ax.axhline(0, color="#374151", linewidth=1, linestyle="--")
    ax.set_title("60份测试合约在各压力情景下的价格变化分布", fontsize=15, pad=14)
    ax.set_ylabel("相对基准的价格变化（美元）")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_scenario_price_change_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    if not SUMMARY_PATH.is_file() or not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Run src/run_week_7_scenarios.py before creating the charts."
        )

    configure_style()
    summary = ordered_summary(pd.read_csv(SUMMARY_PATH))
    predictions = pd.read_csv(PREDICTIONS_PATH)

    expected_rows = len(SCENARIO_ORDER) * 60
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} contract-scenario rows, got {len(predictions)}."
        )

    paths = [
        plot_mean_prices(summary),
        plot_percentage_changes(summary),
        plot_change_distribution(predictions),
    ]
    print("Week 7 scenario charts completed:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
