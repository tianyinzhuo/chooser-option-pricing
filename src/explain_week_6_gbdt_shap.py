"""Week 6: SHAP interpretation for the tuned normalized-pricing GBDT."""

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "week_6"
    / "week_6_tuned_gbdt_pricing.joblib"
)
TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_synthetic_price_splits"
    / "week_5_e2e_test.csv"
)
CONFIG_PATH = PROJECT_ROOT / "config" / "week_6_model_search.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_6" / "shap"


def add_stationary_features(data: pd.DataFrame) -> pd.DataFrame:
    """Recreate the scale-free inputs used during model training."""
    result = data.copy()
    result["close_to_ma20_ratio"] = result["close"] / result["close_ma_20"]
    result["close_to_ma60_ratio"] = result["close"] / result["close_ma_60"]
    result["volume_to_ma20_ratio"] = result["volume"] / result["volume_ma_20"]
    return result


def save_current_figure(path: Path, width: float, height: float) -> None:
    """Apply a consistent layout and save the active SHAP figure."""
    figure = plt.gcf()
    figure.set_size_inches(width, height)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    if model_bundle["family"] != "gbdt":
        raise ValueError("The selected SHAP file is not a GBDT model.")

    test_data = add_stationary_features(
        pd.read_csv(TEST_PATH, parse_dates=["date"])
    )

    maximum_samples = config["interpretability"][
        "maximum_explanation_samples"
    ]
    explanation_data = test_data.head(maximum_samples).copy()
    x_explain = explanation_data[feature_columns]

    if x_explain.isna().any().any():
        raise ValueError("SHAP input data contains missing values.")

    explainer = shap.TreeExplainer(model)
    ratio_explanation = explainer(x_explain)

    close_values = explanation_data["close"].to_numpy(dtype=float)
    ratio_base_values = np.asarray(ratio_explanation.base_values)

    if ratio_base_values.ndim == 0:
        ratio_base_values = np.repeat(
            float(ratio_base_values),
            len(explanation_data),
        )

    # The model predicts price / spot. Multiplying every additive SHAP
    # component by spot converts the explanation into US-dollar units.
    dollar_explanation = shap.Explanation(
        values=(
            np.asarray(ratio_explanation.values)
            * close_values[:, np.newaxis]
        ),
        base_values=ratio_base_values * close_values,
        data=x_explain.to_numpy(),
        feature_names=feature_columns,
    )

    predicted_ratio = model.predict(x_explain)
    predicted_price = predicted_ratio * close_values
    reconstructed_price = (
        np.asarray(dollar_explanation.base_values)
        + np.asarray(dollar_explanation.values).sum(axis=1)
    )

    if not np.allclose(
        predicted_price,
        reconstructed_price,
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ValueError("Dollar SHAP contributions do not reconstruct predictions.")

    actual_price = explanation_data["heston_synthetic_price"].to_numpy()
    absolute_error = np.abs(predicted_price - actual_price)
    representative_position = int(np.argsort(absolute_error)[len(absolute_error) // 2])

    global_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_absolute_shap_usd": np.abs(
                dollar_explanation.values
            ).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap_usd", ascending=False)

    representative_values = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_value": x_explain.iloc[
                representative_position
            ].to_numpy(),
            "shap_contribution_usd": dollar_explanation.values[
                representative_position
            ],
        }
    ).sort_values(
        "shap_contribution_usd",
        key=lambda series: series.abs(),
        ascending=False,
    )

    representative_contract = {
        "row_position": representative_position,
        "date": explanation_data.iloc[representative_position][
            "date"
        ].date().isoformat(),
        "spot_price": float(close_values[representative_position]),
        "strike": float(
            explanation_data.iloc[representative_position]["strike"]
        ),
        "moneyness": float(
            explanation_data.iloc[representative_position]["moneyness"]
        ),
        "choice_time_years": float(
            explanation_data.iloc[representative_position][
                "choice_time_years"
            ]
        ),
        "actual_heston_synthetic_price": float(
            actual_price[representative_position]
        ),
        "gbdt_predicted_price": float(
            predicted_price[representative_position]
        ),
        "absolute_error": float(absolute_error[representative_position]),
        "shap_base_value_usd": float(
            dollar_explanation.base_values[representative_position]
        ),
    }

    bar_path = OUTPUT_DIR / "gbdt_shap_summary_bar_usd.png"
    beeswarm_path = OUTPUT_DIR / "gbdt_shap_beeswarm_usd.png"
    waterfall_path = OUTPUT_DIR / "gbdt_shap_waterfall_representative_usd.png"

    shap.plots.bar(
        dollar_explanation,
        max_display=15,
        show=False,
    )
    plt.title("GBDT global feature importance (mean absolute SHAP, USD)")
    save_current_figure(bar_path, width=10, height=7)

    shap.plots.beeswarm(
        dollar_explanation,
        max_display=15,
        show=False,
    )
    plt.title("GBDT feature effects on chooser price (SHAP, USD)")
    save_current_figure(beeswarm_path, width=11, height=8)

    shap.plots.waterfall(
        dollar_explanation[representative_position],
        max_display=15,
        show=False,
    )
    plt.title("Representative GBDT chooser-price decision (USD)")
    save_current_figure(waterfall_path, width=11, height=8)

    global_importance.to_csv(
        OUTPUT_DIR / "gbdt_shap_global_importance_usd.csv",
        index=False,
        encoding="utf-8-sig",
    )
    representative_values.to_csv(
        OUTPUT_DIR / "gbdt_shap_representative_contributions_usd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "model_name": model_bundle["model_name"],
        "model_family": model_bundle["family"],
        "explainer": "SHAP TreeExplainer",
        "explained_target": (
            "Normalized price predictions converted back to US dollars by "
            "multiplying SHAP components by each contract's spot price."
        ),
        "samples_explained": len(explanation_data),
        "top_ten_global_features": global_importance.head(10).to_dict(
            orient="records"
        ),
        "representative_contract": representative_contract,
        "interpretation_warning": (
            "SHAP values explain the fitted model's prediction behavior. "
            "They are not estimates of causal effects."
        ),
        "target_disclosure": (
            "The model target is a Heston Monte Carlo synthetic reference "
            "price, not a real OTC transaction premium."
        ),
    }

    (OUTPUT_DIR / "gbdt_shap_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Week 6 GBDT SHAP analysis completed.")
    print(f"Explained contracts: {len(explanation_data)}")
    print()
    print("Top 10 global SHAP features (USD)")
    print(
        global_importance.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print("Representative contract")
    print(json.dumps(representative_contract, ensure_ascii=False, indent=2))
    print()
    print(f"SHAP bar chart:      {bar_path}")
    print(f"SHAP beeswarm chart: {beeswarm_path}")
    print(f"SHAP waterfall:      {waterfall_path}")
    print("Warning: SHAP attribution is descriptive, not causal.")


if __name__ == "__main__":
    main()
