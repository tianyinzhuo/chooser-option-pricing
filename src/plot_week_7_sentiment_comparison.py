"""Plot the controlled Week 7 GBDT sentiment-feature comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_7"

CV_PATH = OUTPUT_DIR / "week_7_sentiment_model_cv_comparison.csv"
TEST_PATH = OUTPUT_DIR / "week_7_sentiment_model_test_comparison.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "week_7_sentiment_model_feature_importance.csv"

MODEL_ORDER = ["gbdt_without_sentiment", "gbdt_with_sentiment"]
MODEL_LABELS = ["GBDT无情绪特征", "GBDT加入情绪特征"]
COLORS = ["#2F6B9A", "#C44E52"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("model_name")
    missing = [name for name in MODEL_ORDER if name not in indexed.index]
    if missing:
        raise ValueError(f"Missing comparison models: {missing}")
    return indexed.loc[MODEL_ORDER].reset_index()


def annotate(ax, bars, decimals: int = 3) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )


def plot_cv_test_rmse(cv: pd.DataFrame, test: pd.DataFrame) -> Path:
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    cv_bars = ax.bar(x - width / 2, cv["rmse"], width, label="按日期交叉验证", color="#4472C4")
    test_bars = ax.bar(x + width / 2, test["rmse"], width, label="时间测试集", color="#ED7D31")
    annotate(ax, cv_bars)
    annotate(ax, test_bars)
    ax.set_xticks(x, MODEL_LABELS)
    ax.set_ylabel("RMSE（美元）")
    ax.set_title("加入新闻情绪前后的GBDT定价误差")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_sentiment_cv_test_rmse_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_test_metrics(test: pd.DataFrame) -> Path:
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    mae_bars = ax.bar(x - width / 2, test["mae"], width, label="MAE", color="#70AD47")
    rmse_bars = ax.bar(x + width / 2, test["rmse"], width, label="RMSE", color="#5B9BD5")
    annotate(ax, mae_bars)
    annotate(ax, rmse_bars)
    ax.set_xticks(x, MODEL_LABELS)
    ax.set_ylabel("测试误差（美元，越低越好）")
    ax.set_title("情绪特征控制实验的测试集指标")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_sentiment_test_metric_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(importance: pd.DataFrame) -> Path:
    top = importance.nlargest(15, "importance").sort_values("importance")
    colors = np.where(top["feature_group"].eq("sentiment"), "#C44E52", "#2F6B9A")
    fig, ax = plt.subplots(figsize=(10, 7.0))
    bars = ax.barh(top["feature"], top["importance"], color=colors)
    for bar in bars:
        ax.text(
            bar.get_width() + top["importance"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("GBDT内置特征重要性")
    ax.set_title("加入新闻情绪后的GBDT前15项特征")
    ax.grid(axis="x", alpha=0.22)
    ax.text(
        0.99,
        0.02,
        "蓝色：市场特征    红色：新闻情绪特征",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_sentiment_feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    configure_style()
    cv = ordered(pd.read_csv(CV_PATH))
    test = ordered(pd.read_csv(TEST_PATH))
    importance = pd.read_csv(IMPORTANCE_PATH)
    paths = [
        plot_cv_test_rmse(cv, test),
        plot_test_metrics(test),
        plot_feature_importance(importance),
    ]
    print("Week 7 sentiment-comparison charts completed:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
