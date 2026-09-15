"""Run Week 7 extreme scenarios with the tuned Week 6 GBDT pricing model."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("CHOOSER_PROJECT_ROOT", DEFAULT_ROOT)).resolve()
CONFIG_PATH = PROJECT_ROOT / "config" / "week_7" / "scenario_config.json"

VOLATILITY_COLUMNS = ["rolling_vol_20", "rolling_vol_60"]
VIX_COLUMNS = ["vix", "vix_ma_20"]
RATE_LEVEL_COLUMNS = ["treasury_10y"]


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a configured path and reject paths outside the project."""
    path = (PROJECT_ROOT / relative_path).resolve()
    if path != PROJECT_ROOT and PROJECT_ROOT not in path.parents:
        raise ValueError(f"Configured path leaves the project directory: {relative_path}")
    return path


def add_stationary_features(data: pd.DataFrame) -> pd.DataFrame:
    """Recreate the scale-free columns used during Week 6 training."""
    result = data.copy()
    result["close_to_ma20_ratio"] = result["close"] / result["close_ma_20"]
    result["close_to_ma60_ratio"] = result["close"] / result["close_ma_60"]
    result["volume_to_ma20_ratio"] = result["volume"] / result["volume_ma_20"]
    return result


def load_pretest_feature_ranges(
    config: dict,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build feature minima and maxima from Week 5 train + validation only."""
    range_config = config.get("feature_range_check", {})
    reference_paths = range_config.get("reference_data_paths", [])
    if not reference_paths:
        raise ValueError(
            "feature_range_check.reference_data_paths must contain the Week 5 "
            "train and validation data paths."
        )

    reference_frames: list[pd.DataFrame] = []
    for relative_path in reference_paths:
        reference_path = resolve_project_path(relative_path)
        reference = pd.read_csv(reference_path)
        reference_frames.append(add_stationary_features(reference))

    pretest_data = pd.concat(reference_frames, ignore_index=True)
    missing = sorted(set(feature_columns) - set(pretest_data.columns))
    if missing:
        raise ValueError(f"Pre-test range reference is missing model features: {missing}")

    records: list[dict] = []
    for feature in feature_columns:
        values = pd.to_numeric(pretest_data[feature], errors="coerce")
        finite_values = values[np.isfinite(values)]
        if finite_values.empty:
            raise ValueError(
                f"Pre-test range reference has no finite values for feature: {feature}"
            )
        records.append(
            {
                "feature": feature,
                "pretest_min": float(finite_values.min()),
                "pretest_max": float(finite_values.max()),
                "pretest_observations": int(finite_values.size),
            }
        )

    return pd.DataFrame(records)


def feature_range_mask(
    data: pd.DataFrame,
    feature_ranges: pd.DataFrame,
) -> pd.DataFrame:
    """Flag model-feature values outside the Week 5 pre-test range."""
    features = feature_ranges["feature"].tolist()
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Scenario data is missing range-check features: {missing}")

    values = data[features].apply(pd.to_numeric, errors="coerce")
    lower = feature_ranges.set_index("feature").loc[features, "pretest_min"]
    upper = feature_ranges.set_index("feature").loc[features, "pretest_max"]
    outside = values.lt(lower, axis="columns") | values.gt(upper, axis="columns")
    outside |= ~np.isfinite(values)
    return outside


def joined_flagged_features(mask: pd.DataFrame) -> pd.Series:
    """Convert a boolean feature mask to a stable semicolon-separated list."""
    return mask.apply(
        lambda row: ";".join(mask.columns[row.to_numpy(dtype=bool)]),
        axis=1,
    )


def apply_scenario(
    baseline: pd.DataFrame,
    scenario_name: str,
    scenario: dict,
) -> pd.DataFrame:
    """Return a shocked copy without changing the baseline data."""
    shocked = baseline.copy()

    if scenario_name == "baseline":
        return shocked

    if scenario_name in {
        "underlying_volatility_spike_50pct",
        "vix_spike_50pct",
    }:
        shocked[scenario["columns"]] *= float(scenario["multiplier"])
        return shocked

    if scenario_name == "rate_hike_2_percentage_points":
        columns = scenario.get("columns", RATE_LEVEL_COLUMNS)
        if columns != RATE_LEVEL_COLUMNS:
            raise ValueError(
                "The rate-hike scenario must shift treasury_10y only; "
                "daily change and momentum must remain unchanged."
            )
        shocked[columns] += float(scenario["additive_shock"])
        return shocked

    if scenario_name == "combined_stress":
        multiplier = float(scenario["volatility_multiplier"])
        shocked[VOLATILITY_COLUMNS + VIX_COLUMNS] *= multiplier
        rate_columns = scenario.get("rate_level_columns", RATE_LEVEL_COLUMNS)
        if rate_columns != RATE_LEVEL_COLUMNS:
            raise ValueError(
                "The combined scenario must shift treasury_10y only; "
                "daily change and momentum must remain unchanged."
            )
        shocked[rate_columns] += float(scenario["rate_additive_shock"])
        return shocked

    raise ValueError(f"Unsupported scenario: {scenario_name}")


def predict_prices(model_bundle: dict, data: pd.DataFrame) -> np.ndarray:
    """Predict normalized premium and restore it to US dollars."""
    feature_columns = model_bundle["feature_columns"]
    missing = sorted(set(feature_columns) - set(data.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    predicted_ratio = model_bundle["model"].predict(data[feature_columns])
    predicted_price = predicted_ratio * data["close"].to_numpy()
    if not np.isfinite(predicted_price).all():
        raise RuntimeError("Scenario prediction contains non-finite values.")
    return predicted_price


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_path = resolve_project_path(config["test_data_path"])
    model_path = resolve_project_path(config["model_path"])
    configured_output = resolve_project_path(config["output_directory"])
    output_dir = Path(
        os.environ.get("WEEK7_OUTPUT_DIR", str(configured_output))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_data = pd.read_csv(data_path, parse_dates=["date"])
    baseline_data = add_stationary_features(baseline_data)
    baseline_data["contract_id"] = np.arange(1, len(baseline_data) + 1)

    model_bundle = joblib.load(model_path)
    if model_bundle.get("family") != "gbdt":
        raise ValueError("The configured model is not the Week 6 GBDT bundle.")

    feature_columns = list(model_bundle["feature_columns"])
    feature_ranges = load_pretest_feature_ranges(config, feature_columns)
    baseline_range_mask = feature_range_mask(baseline_data, feature_ranges)
    baseline_outside_count = baseline_range_mask.sum(axis=1).astype(int)
    baseline_outside_features = joined_flagged_features(baseline_range_mask)

    baseline_prices = predict_prices(model_bundle, baseline_data)
    prediction_frames: list[pd.DataFrame] = []
    summary_records: list[dict] = []

    for scenario_name, scenario in config["scenarios"].items():
        scenario_data = apply_scenario(baseline_data, scenario_name, scenario)
        scenario_range_mask = feature_range_mask(scenario_data, feature_ranges)
        scenario_outside_count = scenario_range_mask.sum(axis=1).astype(int)
        scenario_outside_features = joined_flagged_features(scenario_range_mask)
        newly_outside_mask = scenario_range_mask & ~baseline_range_mask
        newly_outside_count = newly_outside_mask.sum(axis=1).astype(int)
        newly_outside_features = joined_flagged_features(newly_outside_mask)

        scenario_prices = predict_prices(model_bundle, scenario_data)
        price_change = scenario_prices - baseline_prices
        percentage_change = np.divide(
            price_change * 100.0,
            baseline_prices,
            out=np.zeros_like(price_change),
            where=baseline_prices != 0,
        )

        details = pd.DataFrame(
            {
                "contract_id": baseline_data["contract_id"],
                "date": baseline_data["date"],
                "moneyness": baseline_data["moneyness"],
                "choice_time_years": baseline_data["choice_time_years"],
                "scenario": scenario_name,
                "baseline_ml_price": baseline_prices,
                "scenario_ml_price": scenario_prices,
                "price_change": price_change,
                "percentage_change": percentage_change,
                "heston_synthetic_price": baseline_data["heston_synthetic_price"],
                "baseline_any_feature_outside_pretest_range": (
                    baseline_outside_count > 0
                ).astype(int),
                "baseline_outside_pretest_range_feature_count": (
                    baseline_outside_count
                ),
                "baseline_outside_pretest_range_features": (
                    baseline_outside_features
                ),
                "any_feature_outside_pretest_range": (
                    scenario_outside_count > 0
                ).astype(int),
                "outside_pretest_range_feature_count": scenario_outside_count,
                "outside_pretest_range_features": scenario_outside_features,
                "newly_outside_pretest_range_due_to_scenario": (
                    newly_outside_count > 0
                ).astype(int),
                "newly_outside_pretest_range_feature_count": (
                    newly_outside_count
                ),
                "newly_outside_pretest_range_features": newly_outside_features,
            }
        )
        prediction_frames.append(details)

        outside_features = [
            feature
            for feature in scenario_range_mask.columns
            if bool(scenario_range_mask[feature].any())
        ]
        newly_outside_feature_names = [
            feature
            for feature in newly_outside_mask.columns
            if bool(newly_outside_mask[feature].any())
        ]

        summary_records.append(
            {
                "scenario": scenario_name,
                "description": scenario["description"],
                "contracts": len(details),
                "mean_price": float(np.mean(scenario_prices)),
                "median_price": float(np.median(scenario_prices)),
                "mean_price_change": float(np.mean(price_change)),
                "mean_percentage_change": float(np.mean(percentage_change)),
                "maximum_absolute_price_change": float(
                    np.max(np.abs(price_change))
                ),
                "contracts_outside_pretest_range": int(
                    (scenario_outside_count > 0).sum()
                ),
                "outside_pretest_range_contract_pct": float(
                    (scenario_outside_count > 0).mean() * 100.0
                ),
                "outside_pretest_range_feature_values": int(
                    scenario_outside_count.sum()
                ),
                "outside_pretest_range_features": ";".join(outside_features),
                "contracts_newly_outside_pretest_range_due_to_scenario": int(
                    (newly_outside_count > 0).sum()
                ),
                "newly_outside_pretest_range_contract_pct": float(
                    (newly_outside_count > 0).mean() * 100.0
                ),
                "newly_outside_pretest_range_feature_values": int(
                    newly_outside_count.sum()
                ),
                "newly_outside_pretest_range_features": ";".join(
                    newly_outside_feature_names
                ),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summary_records)

    predictions_path = output_dir / "week_7_scenario_contract_predictions.csv"
    summary_path = output_dir / "week_7_scenario_summary.csv"
    feature_ranges_path = output_dir / "week_7_pretest_feature_ranges.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    feature_ranges.to_csv(feature_ranges_path, index=False, encoding="utf-8-sig")

    print("Week 7 extreme-scenario analysis completed.")
    print(f"Test contracts: {len(baseline_data)}")
    print(f"Scenarios: {len(summary)}")
    print()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"Predictions saved to: {predictions_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Pre-test feature ranges saved to: {feature_ranges_path}")
    print(
        "Range-check reference: Week 5 train + validation contracts only "
        "(the chronological test set is excluded)."
    )
    print("Disclosure: results describe model responses to synthetic shocks, not causal forecasts.")


if __name__ == "__main__":
    main()
