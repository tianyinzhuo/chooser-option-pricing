import json
from pathlib import Path
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOLATILITY_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_splits"
PRICING_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "ml_synthetic_price_splits"


class TestWeek5MLDataSplits(unittest.TestCase):
    """Validate chronological ML data splits and target completeness."""

    @staticmethod
    def load_csv(directory: Path, filename: str) -> pd.DataFrame:
        return pd.read_csv(directory / filename, parse_dates=["date"])

    def test_volatility_splits_are_chronological_and_non_overlapping(self):
        train = self.load_csv(VOLATILITY_SPLIT_DIR, "week_5_train.csv")
        validation = self.load_csv(VOLATILITY_SPLIT_DIR, "week_5_validation.csv")
        test = self.load_csv(VOLATILITY_SPLIT_DIR, "week_5_test.csv")

        self.assertLess(train["date"].max(), validation["date"].min())
        self.assertLess(validation["date"].max(), test["date"].min())

        self.assertTrue(
            set(train["date"]).isdisjoint(set(validation["date"]))
        )
        self.assertTrue(
            set(validation["date"]).isdisjoint(set(test["date"]))
        )

    def test_volatility_target_has_no_missing_values(self):
        for filename in [
            "week_5_train.csv",
            "week_5_validation.csv",
            "week_5_test.csv",
        ]:
            data = self.load_csv(VOLATILITY_SPLIT_DIR, filename)
            self.assertEqual(
                int(data["future_realized_vol_20"].isna().sum()),
                0,
            )

    def test_synthetic_pricing_dates_do_not_cross_splits(self):
        train = self.load_csv(PRICING_SPLIT_DIR, "week_5_e2e_train.csv")
        validation = self.load_csv(
            PRICING_SPLIT_DIR,
            "week_5_e2e_validation.csv",
        )
        test = self.load_csv(PRICING_SPLIT_DIR, "week_5_e2e_test.csv")

        self.assertLess(train["date"].max(), validation["date"].min())
        self.assertLess(validation["date"].max(), test["date"].min())

        self.assertTrue(
            set(train["date"]).isdisjoint(set(validation["date"]))
        )
        self.assertTrue(
            set(validation["date"]).isdisjoint(set(test["date"]))
        )

    def test_end_to_end_config_discloses_synthetic_label(self):
        config_path = PROJECT_ROOT / "config" / "week_5_end_to_end_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["label_type"],
            "Heston Monte Carlo synthetic reference price",
        )
        self.assertIn("not an actual OTC", config["disclosure"])


if __name__ == "__main__":
    unittest.main(verbosity=2)