"""Explain the Week 7 sentiment-extended GBDT with audited SHAP outputs.

The fitted model predicts a normalized price (option price divided by spot).
SHAP values are retained in normalized units and also converted row by row to
US dollars. Multiplying the complete additive decomposition by each contract's
spot price preserves SHAP additivity in dollar units.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT / "models" / "week_7" / "week_7_gbdt_with_sentiment_bundle.joblib"
)
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_synthetic_price_splits"
SENTIMENT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "week_7_feature_dataset_with_sentiment.csv"
)
SENTIMENT_CONFIG_PATH = PROJECT_ROOT / "config" / "week_7" / "sentiment_feature_config.json"
PREDICTION_PATH = (
    PROJECT_ROOT / "outputs" / "week_7" / "week_7_sentiment_model_test_predictions.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_7" / "shap"

SPLIT_FILES = {
    "train": "week_5_e2e_train.csv",
    "validation": "week_5_e2e_validation.csv",
    "test": "week_5_e2e_test.csv",
}
VIX_CORE_FEATURES = ["vix", "vix_change", "vix_ma_20"]
VIX_INTERACTION_FEATURES = ["jpm_vix_corr_20"]
CONTRACT_FEATURES = ["moneyness", "choice_time_years", "maturity_years"]
HISTORICAL_VOLATILITY_FEATURES = ["rolling_vol_20", "rolling_vol_60"]
DEPENDENCE_FEATURES = [
    "vix",
    "vix_ma_20",
    "news_article_count_ma_5",
    "news_sentiment_ma_20",
]


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


def load_split(split_name: str, sentiment_features: list[str]) -> pd.DataFrame:
    """Load one pricing split and attach strictly lagged sentiment features."""
    data = pd.read_csv(SPLIT_DIR / SPLIT_FILES[split_name], parse_dates=["date"])
    data["close_to_ma20_ratio"] = data["close"] / data["close_ma_20"]
    data["close_to_ma60_ratio"] = data["close"] / data["close_ma_60"]
    data["volume_to_ma20_ratio"] = data["volume"] / data["volume_ma_20"]
    sentiment = pd.read_csv(
        SENTIMENT_PATH,
        parse_dates=["date"],
        usecols=["date", *sentiment_features],
    )
    result = data.merge(sentiment, on="date", how="left", validate="many_to_one")
    result = result.sort_values(["date", "moneyness", "choice_time_years"]).reset_index(
        drop=True
    )
    if result[sentiment_features].isna().any().any():
        raise RuntimeError(f"Missing lagged sentiment values in {split_name} data.")
    return result


def feature_group(feature: str, sentiment_features: list[str]) -> str:
    if feature in sentiment_features:
        return "news_sentiment"
    if feature in VIX_CORE_FEATURES:
        return "vix_core"
    if feature in VIX_INTERACTION_FEATURES:
        return "vix_interaction"
    if feature in CONTRACT_FEATURES:
        return "contract_terms"
    if feature in HISTORICAL_VOLATILITY_FEATURES:
        return "historical_volatility"
    return "other_market"


def as_vector(values: object, row_count: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 1:
        vector = np.repeat(vector.item(), row_count)
    if vector.size != row_count:
        raise ValueError("Unexpected SHAP base-value shape.")
    return vector


def validate_feature_groups(
    feature_columns: list[str], groups: list[str], sentiment_features: list[str]
) -> None:
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Feature list contains duplicates.")
    missing_sentiment = sorted(set(sentiment_features) - set(feature_columns))
    if missing_sentiment:
        raise ValueError(f"Bundle is missing sentiment features: {missing_sentiment}")
    if len(groups) != len(feature_columns) or any(not group for group in groups):
        raise ValueError("Every model feature must belong to exactly one SHAP group.")


def plot_global_importance(global_importance: pd.DataFrame) -> Path:
    top = global_importance.head(15).sort_values("mean_abs_shap_usd")
    palette = {
        "news_sentiment": "#C44E52",
        "vix_core": "#E69F00",
        "vix_interaction": "#F4B183",
        "contract_terms": "#70AD47",
        "historical_volatility": "#8064A2",
        "other_market": "#2F6B9A",
    }
    fig, ax = plt.subplots(figsize=(10.4, 7.3))
    bars = ax.barh(
        top["feature"], top["mean_abs_shap_usd"], color=top["feature_group"].map(palette)
    )
    offset = max(float(global_importance["mean_abs_shap_usd"].max()), 1e-9) * 0.008
    for bar in bars:
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("平均绝对SHAP值（美元）")
    ax.set_title("加入新闻情绪后的GBDT全局SHAP重要性")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(
        handles=[
            Patch(color=palette["other_market"], label="其他市场变量"),
            Patch(color=palette["historical_volatility"], label="JPM历史波动率"),
            Patch(color=palette["contract_terms"], label="合约条款"),
            Patch(color=palette["vix_core"], label="VIX核心变量"),
            Patch(color=palette["vix_interaction"], label="JPM-VIX相关性"),
            Patch(color=palette["news_sentiment"], label="新闻情绪"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_shap_global_importance_usd.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_focus_importance(focus: pd.DataFrame) -> Path:
    ordered = focus.sort_values("mean_abs_shap_usd")
    palette = {
        "news_sentiment": "#C44E52",
        "vix_core": "#E69F00",
        "vix_interaction": "#F4B183",
    }
    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    bars = ax.barh(
        ordered["feature"],
        ordered["mean_abs_shap_usd"],
        color=ordered["feature_group"].map(palette),
    )
    offset = max(float(ordered["mean_abs_shap_usd"].max()), 1e-9) * 0.01
    for bar in bars:
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("平均绝对SHAP值（美元）")
    ax.set_title("VIX与新闻情绪特征的SHAP影响幅度")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(
        handles=[
            Patch(color=palette["vix_core"], label="VIX核心变量"),
            Patch(color=palette["vix_interaction"], label="JPM-VIX相关性"),
            Patch(color=palette["news_sentiment"], label="新闻情绪"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_shap_vix_sentiment_focus_usd.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_group_importance(group_importance: pd.DataFrame) -> Path:
    labels = {
        "news_sentiment": "新闻情绪",
        "vix_core": "VIX核心变量",
        "vix_interaction": "JPM-VIX相关性",
        "contract_terms": "合约条款",
        "historical_volatility": "JPM历史波动率",
        "other_market": "其他市场变量",
    }
    ordered = group_importance.sort_values("mean_group_absolute_shap_usd")
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bars = ax.barh(
        ordered["feature_group"].map(labels),
        ordered["mean_group_absolute_shap_usd"],
        color="#4472C4",
    )
    offset = max(float(ordered["mean_group_absolute_shap_usd"].max()), 1e-9) * 0.012
    for bar in bars:
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("组内绝对SHAP值之和的样本均值（美元）")
    ax.set_title("GBDT特征组平均绝对SHAP幅度对比（非精度贡献率）")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    path = OUTPUT_DIR / "week_7_shap_group_importance_usd.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dependence(
    x_explain: pd.DataFrame,
    dollar_values: np.ndarray,
    feature_columns: list[str],
) -> Path:
    missing = [feature for feature in DEPENDENCE_FEATURES if feature not in feature_columns]
    if missing:
        raise ValueError(f"Dependence-chart features are missing: {missing}")
    # Reserve a dedicated narrow column for the colour bar.  Using
    # ``fig.colorbar(..., ax=axes)`` followed by ``bbox_inches='tight'`` can
    # place the bar on top of the right-hand panels on some Matplotlib builds.
    fig = plt.figure(figsize=(13.6, 9.0))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.045],
        wspace=0.32,
        hspace=0.34,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    colorbar_axis = fig.add_subplot(grid[:, 2])
    moneyness = x_explain["moneyness"].to_numpy()
    scatter = None
    for ax, feature in zip(axes.ravel(), DEPENDENCE_FEATURES):
        position = feature_columns.index(feature)
        scatter = ax.scatter(
            x_explain[feature],
            dollar_values[:, position],
            c=moneyness,
            cmap="viridis",
            s=40,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.axhline(0, color="#374151", linestyle="--", linewidth=1)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP价格贡献（美元）")
        ax.set_title(feature)
        ax.grid(alpha=0.2)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, cax=colorbar_axis)
        colorbar.set_label("价内外程度 K/S")
    fig.suptitle("VIX与情绪特征取值—模型贡献关系（描述性而非因果）", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.91)
    path = OUTPUT_DIR / "week_7_shap_vix_sentiment_dependence_usd.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_native_shap_plots(
    dollar_explanation: shap.Explanation,
    absolute_errors: np.ndarray,
) -> tuple[Path, Path, int]:
    plt.figure(figsize=(10.5, 7.8))
    shap.plots.beeswarm(dollar_explanation, max_display=20, show=False)
    plt.xlabel("SHAP价格贡献（美元）")
    plt.title("测试集合约SHAP贡献分布")
    beeswarm_path = OUTPUT_DIR / "week_7_shap_beeswarm_usd.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=220)
    plt.close()

    representative_position = int(np.argsort(absolute_errors)[len(absolute_errors) // 2])
    plt.figure(figsize=(10.5, 7.8))
    shap.plots.waterfall(
        dollar_explanation[representative_position], max_display=15, show=False
    )
    plt.title("代表性合约的SHAP价格贡献（美元）")
    waterfall_path = OUTPUT_DIR / "week_7_shap_waterfall_representative_usd.png"
    plt.tight_layout()
    plt.savefig(waterfall_path, bbox_inches="tight", dpi=220)
    plt.close()
    return beeswarm_path, waterfall_path, representative_position


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sentiment_config = json.loads(SENTIMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    sentiment_features = list(sentiment_config["sentiment_features"])
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Expected a Week 7 model bundle, not a bare estimator.")
    if bundle.get("family") != "gbdt":
        raise ValueError("The Week 7 SHAP model must be a GBDT bundle.")
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    if len(feature_columns) != 20 + len(sentiment_features):
        raise ValueError("Expected 20 base features plus all Week 7 sentiment features.")

    train = load_split("train", sentiment_features)
    validation = load_split("validation", sentiment_features)
    test = load_split("test", sentiment_features)
    pretest = pd.concat([train, validation], ignore_index=True)
    if set(pretest["date"]).intersection(set(test["date"])):
        raise RuntimeError("Pre-test and test dates overlap.")
    if len(test) != 60 or test["date"].nunique() != 10:
        raise RuntimeError("Expected 60 contracts across 10 chronological test dates.")

    x_pretest = pretest[feature_columns]
    x_test = test[feature_columns]
    if x_pretest.isna().any().any() or x_test.isna().any().any():
        raise ValueError("SHAP input contains missing values.")
    if not np.isfinite(x_test.to_numpy(dtype=float)).all():
        raise ValueError("SHAP input contains non-finite values.")

    groups = [feature_group(feature, sentiment_features) for feature in feature_columns]
    validate_feature_groups(feature_columns, groups, sentiment_features)

    # Use the exact tree-path-dependent decomposition as the primary result.
    # This is also the method used successfully in Week 6.  For this sklearn
    # Huber-GBDT, SHAP 0.49.1's interventional background calculation can leave
    # a small base-value residual, so it is retained only as a ranking
    # robustness check below and never used for local dollar explanations.
    explainer = shap.TreeExplainer(model)
    ratio_explanation = explainer(x_test)
    ratio_values = np.asarray(ratio_explanation.values, dtype=float)
    if ratio_values.shape != (len(test), len(feature_columns)):
        raise RuntimeError(f"Unexpected SHAP matrix shape: {ratio_values.shape}")
    if not np.isfinite(ratio_values).all():
        raise RuntimeError("SHAP matrix contains non-finite values.")

    ratio_base = as_vector(ratio_explanation.base_values, len(test))
    predicted_ratio = np.asarray(model.predict(x_test), dtype=float)
    reconstructed_ratio = ratio_base + ratio_values.sum(axis=1)
    ratio_reconstruction_error = float(
        np.max(np.abs(reconstructed_ratio - predicted_ratio))
    )
    if not np.allclose(reconstructed_ratio, predicted_ratio, rtol=1e-7, atol=1e-7):
        raise RuntimeError("Normalized SHAP values do not reconstruct predictions.")

    close_values = test["close"].to_numpy(dtype=float)
    dollar_values = ratio_values * close_values[:, np.newaxis]
    dollar_base = ratio_base * close_values
    predicted_usd = predicted_ratio * close_values
    reconstructed_usd = dollar_base + dollar_values.sum(axis=1)
    dollar_reconstruction_error = float(
        np.max(np.abs(reconstructed_usd - predicted_usd))
    )
    if not np.allclose(reconstructed_usd, predicted_usd, rtol=1e-7, atol=1e-7):
        raise RuntimeError("Dollar SHAP values do not reconstruct predictions.")

    # Regression guard: explanations must use the same fitted model as the saved comparison.
    saved_predictions = pd.read_csv(PREDICTION_PATH, parse_dates=["date"])
    saved_predictions = saved_predictions.sort_values(
        ["date", "moneyness", "choice_time_years"]
    ).reset_index(drop=True)
    saved_column = "gbdt_with_sentiment_predicted_price"
    if len(saved_predictions) != len(test) or saved_column not in saved_predictions:
        raise RuntimeError("Saved Week 7 sentiment predictions are incomplete.")
    prediction_match_error = float(
        np.max(np.abs(saved_predictions[saved_column].to_numpy() - predicted_usd))
    )
    if not np.allclose(
        saved_predictions[saved_column].to_numpy(), predicted_usd, rtol=1e-8, atol=1e-8
    ):
        raise RuntimeError("SHAP bundle predictions do not match saved Week 7 predictions.")

    actual_usd = test["heston_synthetic_price"].to_numpy(dtype=float)
    test_rmse = float(mean_squared_error(actual_usd, predicted_usd) ** 0.5)

    primary_abs_ratio = np.abs(ratio_values).mean(axis=0)
    background = x_pretest.sample(n=min(100, len(x_pretest)), random_state=42)
    secondary_error = ""
    secondary_reconstruction_error = float("nan")
    try:
        secondary_explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
        )
        secondary_explanation = secondary_explainer(
            x_test,
            check_additivity=False,
        )
        secondary_values = np.asarray(secondary_explanation.values, dtype=float)
        secondary_base = as_vector(secondary_explanation.base_values, len(test))
        secondary_reconstructed = secondary_base + secondary_values.sum(axis=1)
        secondary_reconstruction_error = float(
            np.max(np.abs(secondary_reconstructed - predicted_ratio))
        )
        secondary_abs_ratio = np.abs(secondary_values).mean(axis=0)
        rank_robustness = float(
            pd.Series(primary_abs_ratio)
            .rank()
            .corr(pd.Series(secondary_abs_ratio).rank())
        )
    except Exception as exc:  # optional robustness check must not block outputs
        secondary_abs_ratio = np.full(len(feature_columns), np.nan)
        rank_robustness = float("nan")
        secondary_error = f"{type(exc).__name__}: {exc}"

    global_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_group": groups,
            "mean_abs_shap_ratio": primary_abs_ratio,
            "mean_abs_shap_usd": np.abs(dollar_values).mean(axis=0),
            "mean_signed_shap_ratio": ratio_values.mean(axis=0),
            "mean_signed_shap_usd": dollar_values.mean(axis=0),
            "secondary_mean_abs_shap_ratio": secondary_abs_ratio,
        }
    )
    total_feature_impact = float(global_importance["mean_abs_shap_usd"].sum())
    global_importance["share_of_total_mean_abs"] = (
        global_importance["mean_abs_shap_usd"] / total_feature_impact
    )
    global_importance = global_importance.sort_values(
        "mean_abs_shap_usd", ascending=False
    ).reset_index(drop=True)
    global_importance.insert(0, "rank", np.arange(1, len(global_importance) + 1))

    group_records: list[dict[str, float | str | int]] = []
    contract_records = pd.DataFrame(
        {
            "row_position": np.arange(len(test)),
            "date": test["date"].dt.date.astype(str),
            "spot_price": close_values,
            "strike": test["strike"].to_numpy(dtype=float),
            "moneyness": test["moneyness"].to_numpy(dtype=float),
            "choice_time_years": test["choice_time_years"].to_numpy(dtype=float),
            "actual_heston_synthetic_price": actual_usd,
            "predicted_price_usd": predicted_usd,
            "absolute_error_usd": np.abs(actual_usd - predicted_usd),
            "base_value_ratio": ratio_base,
            "base_value_usd": dollar_base,
            "reconstructed_price_usd": reconstructed_usd,
        }
    )
    group_order = [
        "contract_terms",
        "historical_volatility",
        "vix_core",
        "vix_interaction",
        "news_sentiment",
        "other_market",
    ]
    for group in group_order:
        positions = [index for index, value in enumerate(groups) if value == group]
        signed_ratio = ratio_values[:, positions].sum(axis=1)
        absolute_ratio = np.abs(ratio_values[:, positions]).sum(axis=1)
        signed_usd = dollar_values[:, positions].sum(axis=1)
        absolute_usd = np.abs(dollar_values[:, positions]).sum(axis=1)
        contract_records[f"{group}_signed_contribution_usd"] = signed_usd
        contract_records[f"{group}_absolute_contribution_usd"] = absolute_usd
        group_records.append(
            {
                "feature_group": group,
                "feature_count": len(positions),
                "mean_group_absolute_shap_ratio": float(absolute_ratio.mean()),
                "mean_group_absolute_shap_usd": float(absolute_usd.mean()),
                "mean_group_signed_shap_ratio": float(signed_ratio.mean()),
                "mean_group_signed_shap_usd": float(signed_usd.mean()),
            }
        )
    group_importance = pd.DataFrame(group_records)
    group_importance["share_of_total_group_amplitude"] = (
        group_importance["mean_group_absolute_shap_usd"]
        / group_importance["mean_group_absolute_shap_usd"].sum()
    )
    group_importance = group_importance.sort_values(
        "mean_group_absolute_shap_usd", ascending=False
    ).reset_index(drop=True)

    date_columns = [
        "actual_heston_synthetic_price",
        "predicted_price_usd",
        "absolute_error_usd",
        *[
            f"{group}_{kind}_contribution_usd"
            for group in group_order
            for kind in ("signed", "absolute")
        ],
    ]
    date_group_effects = contract_records.groupby("date", as_index=False)[date_columns].mean()
    date_counts = contract_records.groupby("date").size().rename("contracts")
    date_group_effects.insert(
        1, "contracts", date_group_effects["date"].map(date_counts).astype(int)
    )

    focus = global_importance[
        global_importance["feature_group"].isin(
            ["vix_core", "vix_interaction", "news_sentiment"]
        )
    ].copy()
    long_values = pd.DataFrame(
        {
            "row_position": np.repeat(np.arange(len(test)), len(feature_columns)),
            "date": np.repeat(test["date"].dt.date.astype(str).to_numpy(), len(feature_columns)),
            "feature": np.tile(feature_columns, len(test)),
            "feature_group": np.tile(groups, len(test)),
            "feature_value": x_test.to_numpy(dtype=float).reshape(-1),
            "shap_contribution_ratio": ratio_values.reshape(-1),
            "shap_contribution_usd": dollar_values.reshape(-1),
        }
    )

    dollar_explanation = shap.Explanation(
        values=dollar_values,
        base_values=dollar_base,
        data=x_test.to_numpy(dtype=float),
        feature_names=feature_columns,
    )
    beeswarm_path, waterfall_path, representative_position = save_native_shap_plots(
        dollar_explanation,
        contract_records["absolute_error_usd"].to_numpy(),
    )
    representative = contract_records.iloc[[representative_position]].copy()

    chart_paths = [
        plot_global_importance(global_importance),
        plot_focus_importance(focus),
        plot_group_importance(group_importance),
        plot_dependence(x_test, dollar_values, feature_columns),
        beeswarm_path,
        waterfall_path,
    ]

    global_importance.to_csv(
        OUTPUT_DIR / "week_7_shap_global_importance.csv", index=False, encoding="utf-8-sig"
    )
    focus.to_csv(
        OUTPUT_DIR / "week_7_shap_vix_sentiment_focus.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_importance.to_csv(
        OUTPUT_DIR / "week_7_shap_group_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contract_records.to_csv(
        OUTPUT_DIR / "week_7_shap_contract_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    date_group_effects.to_csv(
        OUTPUT_DIR / "week_7_shap_date_group_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    representative.to_csv(
        OUTPUT_DIR / "week_7_shap_representative_contract.csv",
        index=False,
        encoding="utf-8-sig",
    )
    long_values.to_csv(
        OUTPUT_DIR / "week_7_shap_test_values_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    group_map = group_importance.set_index("feature_group")[
        "mean_group_absolute_shap_usd"
    ].to_dict()
    summary = {
        "model_name": bundle["model_name"],
        "explainer_primary": "TreeExplainer tree_path_dependent exact additive decomposition",
        "explainer_robustness": "TreeExplainer interventional with 100 pre-test background rows",
        "explained_contracts": int(len(test)),
        "unique_test_dates": int(test["date"].nunique()),
        "feature_count": int(len(feature_columns)),
        "sentiment_feature_count": int(len(sentiment_features)),
        "test_rmse_usd": test_rmse,
        "maximum_ratio_reconstruction_error": ratio_reconstruction_error,
        "maximum_dollar_reconstruction_error": dollar_reconstruction_error,
        "maximum_saved_prediction_match_error_usd": prediction_match_error,
        "primary_secondary_global_rank_correlation": rank_robustness,
        "interventional_maximum_ratio_reconstruction_residual": (
            secondary_reconstruction_error
        ),
        "interventional_robustness_warning": secondary_error,
        "vix_core_mean_group_absolute_shap_usd": float(group_map["vix_core"]),
        "historical_volatility_mean_group_absolute_shap_usd": float(
            group_map["historical_volatility"]
        ),
        "vix_interaction_mean_group_absolute_shap_usd": float(
            group_map["vix_interaction"]
        ),
        "news_sentiment_mean_group_absolute_shap_usd": float(
            group_map["news_sentiment"]
        ),
        "representative_contract": representative.iloc[0].to_dict(),
        "disclosures": [
            "The supervised target is a Heston Monte Carlo synthetic reference price, not a real OTC transaction premium.",
            "SHAP describes fitted-model attributions and does not establish causal effects.",
            "Non-zero sentiment SHAP values do not mean sentiment improved predictive accuracy; the controlled comparison determines that question.",
            "The test set has 60 contracts but only 10 independent market dates, with six contract variants per date.",
            "Group shares are shares of the mean absolute SHAP amplitude, not shares of predictive accuracy or model-performance improvement.",
            "Correlated features, including volatility levels, moving averages, and VIX-related variables, can divide attribution between one another.",
            "Zero-filled sentiment on no-news days means no observed prior news, not neutral investor sentiment.",
        ],
    }
    (OUTPUT_DIR / "week_7_shap_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("Week 7 VIX and sentiment SHAP analysis completed.")
    print(f"Explained contracts: {len(test)} | unique dates: {test['date'].nunique()}")
    print(f"Test RMSE reproduced: {test_rmse:.6f} USD")
    print(f"Maximum dollar reconstruction error: {dollar_reconstruction_error:.10f}")
    print(f"Maximum saved-prediction match error: {prediction_match_error:.10f}")
    print(f"Path-dependent/interventional rank correlation: {rank_robustness:.4f}")
    print(
        "Interventional maximum ratio residual "
        f"(diagnostic only): {secondary_reconstruction_error:.10f}"
    )
    if secondary_error:
        print(f"Optional interventional robustness check warning: {secondary_error}")
    print()
    print("Grouped SHAP impact")
    print(group_importance.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(
        "Note: group shares measure mean absolute SHAP amplitude, not predictive-accuracy contribution; "
        "correlated features can split attribution."
    )
    print()
    print("Top VIX and sentiment features")
    print(
        focus.head(12).to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print()
    print("Charts saved:")
    for path in chart_paths:
        print(path)
    print("Warning: SHAP attribution is descriptive, not causal.")


if __name__ == "__main__":
    main()
