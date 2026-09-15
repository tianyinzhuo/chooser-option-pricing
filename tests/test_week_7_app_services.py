"""Unit tests for the Week 7 pricing-tool service layer."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.week_7_pricing_service import (
    feature_range_violations,
    price_contract,
    run_contract_scenarios,
    shock_market_row,
    validate_contract,
)


class DeterministicRatioModel:
    """Small test double whose normalized price reacts to key inputs."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            0.08
            + 0.20 * frame["rolling_vol_60"]
            + 0.01 * (frame["moneyness"] - 1.0)
            + 0.001 * frame["vix"]
            + 0.0002 * frame["treasury_10y"],
            dtype=float,
        )


FEATURES = [
    "rolling_vol_20",
    "rolling_vol_60",
    "vix",
    "vix_ma_20",
    "treasury_10y",
    "treasury_10y_change",
    "treasury_10y_momentum_20",
    "moneyness",
    "choice_time_years",
    "maturity_years",
]


def sample_market_row() -> dict[str, float | str]:
    return {
        "date": "2024-12-31",
        "close": 200.0,
        "rolling_vol_20": 0.25,
        "rolling_vol_60": 0.30,
        "vix": 18.0,
        "vix_ma_20": 17.0,
        "treasury_10y": 4.5,
        "treasury_10y_change": 0.02,
        "treasury_10y_momentum_20": 0.25,
    }


class TestWeek7PricingService(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = {
            "model": DeterministicRatioModel(),
            "feature_columns": FEATURES,
        }

    def test_normalized_prediction_is_restored_to_dollars(self) -> None:
        result = price_contract(self.bundle, sample_market_row(), 1.0, 0.5)
        expected_ratio = 0.08 + 0.20 * 0.30 + 0.001 * 18.0 + 0.0002 * 4.5
        self.assertAlmostEqual(result["normalized_price"], expected_ratio)
        self.assertAlmostEqual(result["ml_price_usd"], expected_ratio * 200.0)
        self.assertTrue(np.isfinite(result["bsm_price_usd"]))

    def test_training_range_warnings_do_not_hide_math_errors(self) -> None:
        warnings = validate_contract(1.2, 0.4, 1.0)
        self.assertEqual(len(warnings), 2)
        with self.assertRaises(ValueError):
            validate_contract(1.0, 1.0, 1.0)

    def test_volatility_stress_changes_the_same_contract(self) -> None:
        results = run_contract_scenarios(
            self.bundle,
            sample_market_row(),
            moneyness=1.0,
            choice_time_years=0.5,
        ).set_index("scenario")
        baseline = results.loc["baseline", "预测权利金（美元）"]
        stressed = results.loc[
            "underlying_volatility_spike_50pct", "预测权利金（美元）"
        ]
        self.assertGreater(stressed, baseline)
        self.assertAlmostEqual(
            results.loc["baseline", "相对基准变化（美元）"], 0.0
        )

    def test_rate_parallel_shift_changes_only_the_level(self) -> None:
        baseline = sample_market_row()
        shocked = shock_market_row(baseline, "rate_hike_2_percentage_points")
        self.assertAlmostEqual(shocked["treasury_10y"], 6.5)
        self.assertAlmostEqual(
            shocked["treasury_10y_change"], baseline["treasury_10y_change"]
        )
        self.assertAlmostEqual(
            shocked["treasury_10y_momentum_20"],
            baseline["treasury_10y_momentum_20"],
        )

    def test_feature_range_check_identifies_extrapolation(self) -> None:
        violations = feature_range_violations(
            {"treasury_10y": 6.5, "rolling_vol_60": 0.3},
            {"treasury_10y": (1.0, 5.0), "rolling_vol_60": (0.1, 0.5)},
        )
        self.assertEqual([item["feature"] for item in violations], ["treasury_10y"])


if __name__ == "__main__":
    unittest.main()
