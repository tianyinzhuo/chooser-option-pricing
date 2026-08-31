import math

import numpy as np


def validate_inputs(
    spot,
    strike,
    choice_time_years,
    maturity_years,
    risk_free_rate,
    initial_variance,
    kappa,
    long_run_variance,
    vol_of_vol,
    rho,
    dividend_yield,
    num_paths,
    trading_days_per_year,
):
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot price and strike must be positive.")

    if not 0 < choice_time_years < maturity_years:
        raise ValueError("Choice time must satisfy 0 < T1 < T2.")

    if initial_variance < 0 or long_run_variance < 0:
        raise ValueError("Variance values must be non-negative.")

    if kappa < 0 or vol_of_vol < 0:
        raise ValueError("kappa and vol_of_vol must be non-negative.")

    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1.")

    if num_paths <= 0 or trading_days_per_year <= 0:
        raise ValueError("num_paths and trading_days_per_year must be positive.")


def heston_simple_chooser_price(
    spot,
    strike,
    choice_time_years,
    maturity_years,
    risk_free_rate,
    initial_variance,
    kappa,
    long_run_variance,
    vol_of_vol,
    rho,
    dividend_yield=0.0,
    num_paths=5000,
    trading_days_per_year=252,
    seed=42,
):
    """
    Price a European simple chooser option with Heston stochastic volatility.

    At the choice date T1, the holder selects a call or put with the
    same strike K and maturity T2. The selection threshold follows
    put-call parity.
    """
    validate_inputs(
        spot,
        strike,
        choice_time_years,
        maturity_years,
        risk_free_rate,
        initial_variance,
        kappa,
        long_run_variance,
        vol_of_vol,
        rho,
        dividend_yield,
        num_paths,
        trading_days_per_year,
    )

    total_steps = math.ceil(maturity_years * trading_days_per_year)
    choice_step = round(choice_time_years * trading_days_per_year)
    dt = maturity_years / total_steps
    sqrt_dt = math.sqrt(dt)

    rng = np.random.default_rng(seed)

    stock_paths = np.full(num_paths, spot, dtype=float)
    variance_paths = np.full(num_paths, initial_variance, dtype=float)

    # At T1, choose call when call value is at least put value.
    time_after_choice = maturity_years - choice_time_years
    choice_threshold = strike * math.exp(
        -(risk_free_rate - dividend_yield) * time_after_choice
    )
    choose_call = np.zeros(num_paths, dtype=bool)

    for step in range(1, total_steps + 1):
        z_stock = rng.standard_normal(num_paths)
        z_independent = rng.standard_normal(num_paths)
        z_variance = rho * z_stock + math.sqrt(1 - rho**2) * z_independent

        non_negative_variance = np.maximum(variance_paths, 0.0)

        stock_paths *= np.exp(
            (
                risk_free_rate
                - dividend_yield
                - 0.5 * non_negative_variance
            )
            * dt
            + np.sqrt(non_negative_variance) * sqrt_dt * z_stock
        )

        variance_paths += (
            kappa * (long_run_variance - non_negative_variance) * dt
            + vol_of_vol
            * np.sqrt(non_negative_variance)
            * sqrt_dt
            * z_variance
        )
        variance_paths = np.maximum(variance_paths, 0.0)

        if step == choice_step:
            choose_call = stock_paths >= choice_threshold

    call_payoff = np.maximum(stock_paths - strike, 0.0)
    put_payoff = np.maximum(strike - stock_paths, 0.0)
    chooser_payoff = np.where(choose_call, call_payoff, put_payoff)

    discount_factor = math.exp(-risk_free_rate * maturity_years)
    discounted_payoffs = discount_factor * chooser_payoff

    price = float(np.mean(discounted_payoffs))
    standard_error = float(
        np.std(discounted_payoffs, ddof=1) / math.sqrt(num_paths)
    )

    return {
        "price": price,
        "standard_error": standard_error,
        "choice_threshold": choice_threshold,
        "total_steps": total_steps,
        "choice_step": choice_step,
    }