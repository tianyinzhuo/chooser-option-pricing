"""Tests for the Week 6 time-aware cross-validation utilities."""

import unittest

import numpy as np
import pandas as pd

from src.week_6_cv import make_grouped_date_cv, make_volatility_cv


class TestWeek6CrossValidation(unittest.TestCase):

    def test_volatility_cv_preserves_twenty_row_gap(self):
        folds = make_volatility_cv(
            n_rows=180,
            n_splits=5,
            gap_rows=20,
        )

        self.assertEqual(len(folds), 5)

        for train_indices, validation_indices in folds:
            self.assertLess(
                train_indices.max(),
                validation_indices.min(),
            )

            actual_gap = (
                validation_indices.min()
                - train_indices.max()
                - 1
            )

            self.assertGreaterEqual(actual_gap, 20)

    def test_pricing_cv_keeps_each_date_together(self):
        unique_dates = pd.date_range(
            start="2020-01-01",
            periods=20,
            freq="D",
        )

        # 模拟每个市场日期包含6份不同参数的选择权合约。
        dates = np.repeat(unique_dates, 6)

        folds = make_grouped_date_cv(
            dates=dates,
            n_splits=4,
        )

        self.assertEqual(len(folds), 4)

        date_series = pd.Series(dates)

        for train_indices, validation_indices in folds:
            train_dates = set(date_series.iloc[train_indices])
            validation_dates = set(date_series.iloc[validation_indices])

            self.assertTrue(train_dates.isdisjoint(validation_dates))
            self.assertLess(max(train_dates), min(validation_dates))

            for validation_date in validation_dates:
                all_rows_for_date = set(
                    np.flatnonzero(
                        date_series.eq(validation_date).to_numpy()
                    )
                )
                validation_rows = set(validation_indices)

                self.assertTrue(
                    all_rows_for_date.issubset(validation_rows)
                )

    def test_unsorted_dates_are_rejected(self):
        unsorted_dates = [
            "2024-01-02",
            "2024-01-04",
            "2024-01-03",
        ]

        with self.assertRaises(ValueError):
            make_grouped_date_cv(
                dates=unsorted_dates,
                n_splits=2,
            )


if __name__ == "__main__":
    unittest.main()