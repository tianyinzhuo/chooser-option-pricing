from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bsm_chooser import simple_chooser_price
from heston_chooser import heston_simple_chooser_price


# Same baseline contract as Week 3.
spot_price = 150.0
strike = 150.0
choice_time_years = 0.5
maturity_years = 1.0
risk_free_rate = 0.05
volatility = 0.30

# BSM closed-form benchmark.
bsm_result = simple_chooser_price(
    spot_price=spot_price,
    strike=strike,
    choice_time_years=choice_time_years,
    maturity_years=maturity_years,
    risk_free_rate=risk_free_rate,
    volatility=volatility,
    dividend_yield=0.0,
)

# Heston Monte Carlo synthetic reference benchmark.
heston_result = heston_simple_chooser_price(
    spot=spot_price,
    strike=strike,
    choice_time_years=choice_time_years,
    maturity_years=maturity_years,
    risk_free_rate=risk_free_rate,
    initial_variance=volatility**2,
    kappa=2.0,
    long_run_variance=0.04,
    vol_of_vol=0.5,
    rho=-0.6,
    dividend_yield=0.0,
    num_paths=5000,
    trading_days_per_year=252,
    seed=42,
)

difference = bsm_result["chooser_price"] - heston_result["price"]

print("Week 4 baseline comparison")
print(f"BSM simple chooser price: {bsm_result['chooser_price']:.4f}")
print(f"Heston MC reference:      {heston_result['price']:.4f}")
print(f"MC standard error:        {heston_result['standard_error']:.4f}")
print(f"Difference (BSM - MC):    {difference:.4f}")
print(f"Choice threshold at T1:   {heston_result['choice_threshold']:.4f}")
print()
print("Note: Heston MC is a synthetic reference benchmark, not an OTC transaction price.")