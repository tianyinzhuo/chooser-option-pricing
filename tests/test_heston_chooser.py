import math
import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heston_chooser import heston_simple_chooser_price


class TestHestonChooserModel(unittest.TestCase):
    """Basic validation tests for the Week 4 Heston MC benchmark."""

    def setUp(self):
        self.parameters = {
            "spot": 150.0,
            "strike": 150.0,
            "choice_time_years": 0.5,
            "maturity_years": 1.0,
            "risk_free_rate": 0.05,
            "initial_variance": 0.09,
            "kappa": 2.0,
            "long_run_variance": 0.04,
            "vol_of_vol": 0.5,
            "rho": -0.6,
            "num_paths": 1000,
            "trading_days_per_year": 252,
            "seed": 42,
        }

    def test_result_is_positive_and_finite(self):
        """The simulated price and standard error should be valid numbers."""
        result = heston_simple_chooser_price(**self.parameters)

        self.assertGreater(result["price"], 0.0)
        self.assertGreater(result["standard_error"], 0.0)
        self.assertTrue(math.isfinite(result["price"]))
        self.assertTrue(math.isfinite(result["standard_error"]))

    def test_same_seed_produces_same_result(self):
        """A fixed random seed makes the benchmark reproducible."""
        first_result = heston_simple_chooser_price(**self.parameters)
        second_result = heston_simple_chooser_price(**self.parameters)

        self.assertEqual(first_result["price"], second_result["price"])
        self.assertEqual(
            first_result["standard_error"],
            second_result["standard_error"],
        )

    def test_invalid_choice_date_raises_error(self):
        """The choice date must be strictly between today and maturity."""
        invalid_parameters = self.parameters.copy()
        invalid_parameters["choice_time_years"] = 1.0

        with self.assertRaises(ValueError):
            heston_simple_chooser_price(**invalid_parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)