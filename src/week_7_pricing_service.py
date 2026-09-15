"""Reusable pricing helpers for the Week 7 Streamlit prototype.

This module deliberately keeps the user-facing application separate from model
loading and feature preparation.  Both fitted GBDT models predict the chooser
premium divided by spot, so every prediction is converted back to US dollars by
multiplying by the selected day's JPM close.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.bsm_chooser import simple_chooser_price


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_SPECS = {
    "week6_gbdt": {
        "label": "Week 6 GBDT（推荐，无情绪）",
        "path": "models/week_6/week_6_tuned_gbdt_pricing.joblib",
        "test_mae": 1.224687,
        "test_rmse": 1.437762,
        "experimental": False,
    },
    "week7_sentiment_gbdt": {
        "label": "Week 7 GBDT + 新闻情绪（实验性）",
        "path": "models/week_7/week_7_gbdt_with_sentiment_bundle.joblib",
        "test_mae": 1.214470,
        "test_rmse": 1.453125,
        "experimental": True,
    },
}

SCENARIOS = {
    "baseline": "基准情景",
    "underlying_volatility_spike_50pct": "JPM历史波动率状态提高50%",
    "vix_spike_50pct": "VIX持续高位状态提高50%",
    "rate_hike_2_percentage_points": "10年期美债利率水平平行上移2个百分点",
    "combined_stress": "波动率与VIX提高50%，利率水平上移2个百分点",
}


def _root(project_root: str | Path | None) -> Path:
    return Path(project_root).resolve() if project_root else DEFAULT_PROJECT_ROOT


def add_stationary_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the three scale-stable ratios used by the normalized pricing model."""
    result = data.copy()
    result["close_to_ma20_ratio"] = result["close"] / result["close_ma_20"]
    result["close_to_ma60_ratio"] = result["close"] / result["close_ma_60"]
    result["volume_to_ma20_ratio"] = result["volume"] / result["volume_ma_20"]
    return result


def load_market_history(project_root: str | Path | None = None) -> pd.DataFrame:
    """Load the Week 7 feature table, falling back to the Week 2 market table."""
    project = _root(project_root)
    sentiment_path = (
        project / "data" / "processed" / "week_7_feature_dataset_with_sentiment.csv"
    )
    base_path = project / "data" / "processed" / "feature_dataset_2018_2024.csv"
    path = sentiment_path if sentiment_path.exists() else base_path
    data = pd.read_csv(path, parse_dates=["date"])
    data = add_stationary_features(data)
    if data["date"].duplicated().any():
        raise ValueError("Historical feature table contains duplicate dates.")
    return data.sort_values("date").reset_index(drop=True)


def load_model_bundle(
    model_key: str, project_root: str | Path | None = None
) -> dict[str, Any]:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key: {model_key}")
    project = _root(project_root)
    path = project / MODEL_SPECS[model_key]["path"]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"model", "feature_columns"}.issubset(bundle):
        raise ValueError(f"Expected a model bundle with metadata: {path}")
    return bundle


def load_pricing_training_ranges(
    bundle: dict[str, Any], project_root: str | Path | None = None
) -> dict[str, tuple[float, float]]:
    """Return Week 6 pre-test feature ranges used to fit the pricing models.

    The Week 6 search used the Week 5 training and validation contracts as its
    pre-test sample.  The sentiment bundle additionally needs date-level news
    features, which are joined from the leakage-safe Week 7 feature table.
    """
    project = _root(project_root)
    split_dir = project / "data" / "processed" / "ml_synthetic_price_splits"
    frames = [
        pd.read_csv(split_dir / name, parse_dates=["date"])
        for name in ("week_5_e2e_train.csv", "week_5_e2e_validation.csv")
    ]
    training = add_stationary_features(pd.concat(frames, ignore_index=True))

    feature_columns = list(bundle["feature_columns"])
    missing = [name for name in feature_columns if name not in training.columns]
    if missing:
        sentiment_path = (
            project
            / "data"
            / "processed"
            / "week_7_feature_dataset_with_sentiment.csv"
        )
        sentiment = pd.read_csv(sentiment_path, parse_dates=["date"])
        available = [name for name in missing if name in sentiment.columns]
        if available:
            date_features = sentiment[["date", *available]].drop_duplicates("date")
            training = training.merge(
                date_features,
                on="date",
                how="left",
                validate="many_to_one",
            )

    ranges: dict[str, tuple[float, float]] = {}
    for name in feature_columns:
        if name not in training.columns:
            continue
        values = pd.to_numeric(training[name], errors="coerce").dropna()
        if not values.empty:
            ranges[name] = (float(values.min()), float(values.max()))
    return ranges


def feature_range_violations(
    values: pd.Series | dict[str, Any],
    feature_ranges: dict[str, tuple[float, float]],
) -> list[dict[str, float | str]]:
    """Describe model inputs that fall outside the Week 6 pre-test range."""
    row = dict(values)
    violations: list[dict[str, float | str]] = []
    for name, (lower, upper) in feature_ranges.items():
        if name not in row:
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        tolerance = max(1.0, abs(lower), abs(upper)) * 1e-12
        if value < lower - tolerance or value > upper + tolerance:
            violations.append(
                {
                    "feature": name,
                    "value": value,
                    "training_min": lower,
                    "training_max": upper,
                }
            )
    return violations


def validate_contract(
    moneyness: float,
    choice_time_years: float,
    maturity_years: float,
) -> list[str]:
    """Validate mathematical constraints and return non-blocking range warnings."""
    if moneyness <= 0:
        raise ValueError("价内外程度 K/S 必须大于0。")
    if not 0 < choice_time_years < maturity_years:
        raise ValueError("选择时点必须满足 0 < T1 < T2。")
    warnings: list[str] = []
    if not 0.9 <= moneyness <= 1.1:
        warnings.append("K/S 超出训练时使用的 0.9—1.1 范围，属于外推。")
    if choice_time_years not in {0.25, 0.5}:
        warnings.append("T1 超出训练时使用的 0.25 或 0.50 年，属于外推。")
    if abs(maturity_years - 1.0) > 1e-12:
        warnings.append("T2 与训练时固定的1年不同，属于外推。")
    return warnings


def _one_row_frame(
    bundle: dict[str, Any],
    market_row: pd.Series | dict[str, Any],
    moneyness: float,
    choice_time_years: float,
    maturity_years: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = dict(market_row)
    values["moneyness"] = float(moneyness)
    values["choice_time_years"] = float(choice_time_years)
    values["maturity_years"] = float(maturity_years)
    missing = [name for name in bundle["feature_columns"] if name not in values]
    if missing:
        raise ValueError(f"定价行缺少模型特征：{missing}")
    frame = pd.DataFrame(
        [[values[name] for name in bundle["feature_columns"]]],
        columns=bundle["feature_columns"],
    )
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("模型输入包含缺失值或无穷值。")
    return frame, values


def price_contract(
    bundle: dict[str, Any],
    market_row: pd.Series | dict[str, Any],
    moneyness: float,
    choice_time_years: float,
    maturity_years: float = 1.0,
) -> dict[str, Any]:
    warnings = validate_contract(moneyness, choice_time_years, maturity_years)
    frame, values = _one_row_frame(
        bundle,
        market_row,
        moneyness,
        choice_time_years,
        maturity_years,
    )
    spot = float(values["close"])
    strike = spot * float(moneyness)
    volatility = float(values["rolling_vol_60"])
    rate_decimal = float(values["treasury_10y"]) / 100.0
    if min(spot, strike, volatility) <= 0:
        raise ValueError("现价、执行价与波动率必须为正数。")

    normalized_price = float(bundle["model"].predict(frame)[0])
    ml_price = normalized_price * spot
    bsm = simple_chooser_price(
        spot_price=spot,
        strike=strike,
        choice_time_years=float(choice_time_years),
        maturity_years=float(maturity_years),
        risk_free_rate=rate_decimal,
        volatility=volatility,
        dividend_yield=0.0,
    )["chooser_price"]
    if ml_price < 0:
        warnings.append("模型产生了负权利金，说明输入已超出可靠范围。")
    return {
        "date": pd.Timestamp(values["date"]).date().isoformat(),
        "spot": spot,
        "strike": strike,
        "moneyness": float(moneyness),
        "choice_time_years": float(choice_time_years),
        "maturity_years": float(maturity_years),
        "volatility": volatility,
        "rate_decimal": rate_decimal,
        "normalized_price": normalized_price,
        "ml_price_usd": ml_price,
        "bsm_price_usd": float(bsm),
        "difference_usd": ml_price - float(bsm),
        "warnings": warnings,
    }


def shock_market_row(
    market_row: pd.Series | dict[str, Any], scenario_name: str
) -> dict[str, Any]:
    if scenario_name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario_name}")
    shocked = dict(market_row)
    volatility_columns = ["rolling_vol_20", "rolling_vol_60"]
    vix_columns = ["vix", "vix_ma_20"]
    if scenario_name == "baseline":
        return shocked
    if scenario_name == "underlying_volatility_spike_50pct":
        for name in volatility_columns:
            shocked[name] = float(shocked[name]) * 1.5
    elif scenario_name == "vix_spike_50pct":
        for name in vix_columns:
            shocked[name] = float(shocked[name]) * 1.5
    elif scenario_name == "rate_hike_2_percentage_points":
        # Interpret the shock as a parallel level shift.  A parallel shift does
        # not change daily differences or 20-day momentum, so those derived
        # features are deliberately held fixed instead of receiving +2.0.
        shocked["treasury_10y"] = float(shocked["treasury_10y"]) + 2.0
    elif scenario_name == "combined_stress":
        for name in volatility_columns + vix_columns:
            shocked[name] = float(shocked[name]) * 1.5
        shocked["treasury_10y"] = float(shocked["treasury_10y"]) + 2.0
    return shocked


def run_contract_scenarios(
    bundle: dict[str, Any],
    market_row: pd.Series | dict[str, Any],
    moneyness: float,
    choice_time_years: float,
    maturity_years: float = 1.0,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    baseline_price: float | None = None
    for scenario_name, label in SCENARIOS.items():
        shocked_row = shock_market_row(market_row, scenario_name)
        result = price_contract(
            bundle,
            shocked_row,
            moneyness,
            choice_time_years,
            maturity_years,
        )
        if baseline_price is None:
            baseline_price = float(result["ml_price_usd"])
        difference = float(result["ml_price_usd"]) - baseline_price
        range_row = dict(shocked_row)
        range_row.update(
            {
                "moneyness": float(moneyness),
                "choice_time_years": float(choice_time_years),
                "maturity_years": float(maturity_years),
            }
        )
        violations = (
            feature_range_violations(range_row, feature_ranges)
            if feature_ranges
            else []
        )
        records.append(
            {
                "scenario": scenario_name,
                "情景": label,
                "预测权利金（美元）": float(result["ml_price_usd"]),
                "相对基准变化（美元）": difference,
                "相对基准变化率（%）": difference / baseline_price * 100,
                "训练范围检查": "⚠ 超出" if violations else "✓ 范围内",
                "越界特征": "、".join(
                    str(item["feature"]) for item in violations
                )
                or "—",
            }
        )
    return pd.DataFrame(records)


def volatility_sensitivity(
    bundle: dict[str, Any],
    market_row: pd.Series | dict[str, Any],
    moneyness: float,
    choice_time_years: float,
    maturity_years: float = 1.0,
    multipliers: np.ndarray | None = None,
) -> pd.DataFrame:
    if multipliers is None:
        multipliers = np.linspace(0.5, 1.75, 26)
    records = []
    for multiplier in multipliers:
        shocked = dict(market_row)
        shocked["rolling_vol_20"] = float(shocked["rolling_vol_20"]) * float(
            multiplier
        )
        shocked["rolling_vol_60"] = float(shocked["rolling_vol_60"]) * float(
            multiplier
        )
        result = price_contract(
            bundle,
            shocked,
            moneyness,
            choice_time_years,
            maturity_years,
        )
        records.append(
            {
                "volatility_multiplier": float(multiplier),
                "historical_volatility": float(result["volatility"]),
                "ml_price_usd": float(result["ml_price_usd"]),
                "bsm_price_usd": float(result["bsm_price_usd"]),
            }
        )
    return pd.DataFrame(records)
