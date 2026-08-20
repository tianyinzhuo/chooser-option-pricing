import math
import unittest

from src.bsm_chooser import (
    bsm_call_price,
    bsm_put_price,
    simple_chooser_price,
)


class TestBSMChooserModel(unittest.TestCase):
    """Tests for the Week 3 BSM simple chooser model."""

    def setUp(self):
        """Use the paper baseline case for every test."""
        self.spot_price = 150.0
        self.strike = 150.0
        self.choice_time_years = 0.5
        self.maturity_years = 1.0
        self.risk_free_rate = 0.05
        self.volatility = 0.30

    def test_put_call_parity(self):
        """
        Check put-call parity:
        Call - Put = S - K * exp(-rT)
        """
        call_price = bsm_call_price(
            self.spot_price,
            self.strike,
            self.maturity_years,
            self.risk_free_rate,
            self.volatility,
        )

        put_price = bsm_put_price(
            self.spot_price,
            self.strike,
            self.maturity_years,
            self.risk_free_rate,
            self.volatility,
        )

        expected_difference = self.spot_price - self.strike * math.exp(
            -self.risk_free_rate * self.maturity_years
        )

        self.assertAlmostEqual(
            call_price - put_price,
            expected_difference,
            places=8,
        )

    def test_chooser_is_not_cheaper_than_call_or_put(self):
        """
        A chooser gives extra flexibility, so it cannot be cheaper
        than an ordinary call or ordinary put with the same K and T2.
        """
        call_price = bsm_call_price(
            self.spot_price,
            self.strike,
            self.maturity_years,
            self.risk_free_rate,
            self.volatility,
        )

        put_price = bsm_put_price(
            self.spot_price,
            self.strike,
            self.maturity_years,
            self.risk_free_rate,
            self.volatility,
        )

        result = simple_chooser_price(
            self.spot_price,
            self.strike,
            self.choice_time_years,
            self.maturity_years,
            self.risk_free_rate,
            self.volatility,
        )

        self.assertGreaterEqual(result["chooser_price"], call_price)
        self.assertGreaterEqual(result["chooser_price"], put_price)

    def test_paper_baseline_result(self):
        """Check that the baseline result remains stable."""
        result = simple_chooser_price(
            self.spot_price,
            self.strike,
            self.choice_time_years,
            self.maturity_years,
            self.risk_free_rate,
            self.volatility,
        )

        self.assertAlmostEqual(
            result["chooser_price"],
            30.3910,
            places=4,
        )

    def test_invalid_choice_date_raises_error(self):
        """The choice date must be strictly between today and maturity."""
        with self.assertRaises(ValueError):
            simple_chooser_price(
                self.spot_price,
                self.strike,
                choice_time_years=1.0,
                maturity_years=1.0,
                risk_free_rate=self.risk_free_rate,
                volatility=self.volatility,
            )


if __name__ == "__main__":
    unittest.main()