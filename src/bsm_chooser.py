"""
Week 3: Black-Scholes-Merton simple chooser option pricing model.

This version prices a European simple chooser option:
- The holder chooses call or put at T1.
- Both options share strike K and final maturity T2.
- The exact closed-form chooser formula used here assumes q = 0.
"""

import argparse
import json
import math
from pathlib import Path


def normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution function N(x)."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def validate_bsm_inputs(
    spot_price: float,
    strike: float,
    maturity_years: float,
    volatility: float,
) -> None:
    """Check that BSM inputs are economically and mathematically valid."""
    if spot_price <= 0:
        raise ValueError("spot_price must be greater than 0.")
    if strike <= 0:
        raise ValueError("strike must be greater than 0.")
    if maturity_years <= 0:
        raise ValueError("maturity_years must be greater than 0.")
    if volatility <= 0:
        raise ValueError("volatility must be greater than 0.")


def bsm_d1_d2(
    spot_price: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Calculate d1 and d2 used by the BSM model."""
    validate_bsm_inputs(spot_price, strike, maturity_years, volatility)

    volatility_sqrt_time = volatility * math.sqrt(maturity_years)

    d1 = (
        math.log(spot_price / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2)
        * maturity_years
    ) / volatility_sqrt_time

    d2 = d1 - volatility_sqrt_time
    return d1, d2


def bsm_call_price(
    spot_price: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European call option using Black-Scholes-Merton."""
    d1, d2 = bsm_d1_d2(
        spot_price,
        strike,
        maturity_years,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    return (
        spot_price
        * math.exp(-dividend_yield * maturity_years)
        * normal_cdf(d1)
        - strike
        * math.exp(-risk_free_rate * maturity_years)
        * normal_cdf(d2)
    )


def bsm_put_price(
    spot_price: float,
    strike: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European put option using Black-Scholes-Merton."""
    d1, d2 = bsm_d1_d2(
        spot_price,
        strike,
        maturity_years,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    return (
        strike
        * math.exp(-risk_free_rate * maturity_years)
        * normal_cdf(-d2)
        - spot_price
        * math.exp(-dividend_yield * maturity_years)
        * normal_cdf(-d1)
    )


def simple_chooser_price(
    spot_price: float,
    strike: float,
    choice_time_years: float,
    maturity_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    """
    Price a European simple chooser option.

    Exact closed-form relation for q = 0:
    chooser = Call(S, K, T2) + Put(S, K * exp[-r * (T2-T1)], T1)
    """
    validate_bsm_inputs(spot_price, strike, maturity_years, volatility)

    if not 0 < choice_time_years < maturity_years:
        raise ValueError(
            "choice_time_years must satisfy 0 < choice_time_years < maturity_years."
        )

    if abs(dividend_yield) > 1e-12:
        raise ValueError(
            "This Week 3 simple chooser formula assumes dividend_yield = 0. "
            "Dividend adjustment will be added in a later extension."
        )

    call_component = bsm_call_price(
        spot_price,
        strike,
        maturity_years,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    adjusted_put_strike = strike * math.exp(
        -risk_free_rate * (maturity_years - choice_time_years)
    )

    choice_put_component = bsm_put_price(
        spot_price,
        adjusted_put_strike,
        choice_time_years,
        risk_free_rate,
        volatility,
        dividend_yield,
    )

    chooser_price = call_component + choice_put_component

    return {
        "call_component": call_component,
        "adjusted_put_strike": adjusted_put_strike,
        "choice_put_component": choice_put_component,
        "chooser_price": chooser_price,
    }


def load_case(config_path: Path, case_name: str) -> dict:
    """Load one pricing case from the JSON configuration file."""
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if case_name not in config:
        raise KeyError(f"Case '{case_name}' was not found in {config_path}.")

    return config[case_name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Price a European simple chooser option with BSM."
    )
    parser.add_argument(
        "--config",
        default="config/chooser_bsm_config.json",
        help="Path to the JSON parameter configuration file.",
    )
    parser.add_argument(
        "--case",
        default="paper_baseline",
        choices=["paper_baseline", "data_driven_case"],
        help="Pricing scenario to run.",
    )
    args = parser.parse_args()

    parameters = load_case(Path(args.config), args.case)
    result = simple_chooser_price(
        spot_price=parameters["spot_price"],
        strike=parameters["strike"],
        choice_time_years=parameters["choice_time_years"],
        maturity_years=parameters["maturity_years"],
        risk_free_rate=parameters["risk_free_rate"],
        volatility=parameters["volatility"],
        dividend_yield=parameters["dividend_yield"],
    )

    print(f"Pricing case: {args.case}")
    print(f"Call component:       {result['call_component']:.4f}")
    print(f"Adjusted put strike:  {result['adjusted_put_strike']:.4f}")
    print(f"Choice put component: {result['choice_put_component']:.4f}")
    print(f"Simple chooser price: {result['chooser_price']:.4f}")


if __name__ == "__main__":
    main()