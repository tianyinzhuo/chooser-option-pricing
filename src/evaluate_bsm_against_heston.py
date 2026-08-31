import json
import math
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bsm_chooser import simple_chooser_price
from heston_chooser import heston_simple_chooser_price


def calculate_metrics(data: pd.DataFrame, group_name: str) -> dict:
    """Calculate BSM divergence from the Heston MC synthetic benchmark."""
    errors = data["bsm_minus_heston"]

    return {
        "group": group_name,
        "scenarios": len(data),
        "mae": errors.abs().mean(),
        "rmse": math.sqrt((errors**2).mean()),
        "mean_signed_difference": errors.mean(),
        "mean_bsm_price": data["bsm_price"].mean(),
        "mean_heston_price": data["heston_price"].mean(),
    }


def main() -> None:
    feature_path = PROJECT_ROOT / "data" / "processed" / "feature_dataset_2018_2024.csv"
    config_path = PROJECT_ROOT / "config" / "week_4_heston_benchmark.json"
    output_dir = PROJECT_ROOT / "outputs" / "week_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    data = pd.read_csv(feature_path, parse_dates=["date"])
    data = data.dropna(
        subset=["date", "close", "rolling_vol_60", "treasury_10y"]
    ).copy()

    sampling_step = config["simulation"]["sampling_step_days"]
    scenarios = data.iloc[::sampling_step].copy()

    # Keep the final available observation even when it is not on a sampling date.
    if scenarios["date"].iloc[-1] != data["date"].iloc[-1]:
        scenarios = pd.concat([scenarios, data.tail(1)], ignore_index=True)

    high_volatility_threshold = data["rolling_vol_60"].quantile(
        config["evaluation"]["high_volatility_quantile"]
    )

    option_config = config["option_contract"]
    heston_config = config["heston_parameters"]
    simulation_config = config["simulation"]

    results = []

    print(f"Total feature observations: {len(data)}")
    print(f"Evaluation scenarios: {len(scenarios)}")
    print(f"High-volatility threshold: {high_volatility_threshold:.4f}")
    print()

    for scenario_number, (_, row) in enumerate(scenarios.iterrows(), start=1):
        spot_price = float(row["close"])
        volatility = float(row["rolling_vol_60"])
        risk_free_rate = float(row["treasury_10y"]) / 100.0
        strike = spot_price  # At-the-money contract for comparable scenarios.

        bsm_result = simple_chooser_price(
            spot_price=spot_price,
            strike=strike,
            choice_time_years=option_config["choice_time_years"],
            maturity_years=option_config["maturity_years"],
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=option_config["dividend_yield"],
        )

        heston_result = heston_simple_chooser_price(
            spot=spot_price,
            strike=strike,
            choice_time_years=option_config["choice_time_years"],
            maturity_years=option_config["maturity_years"],
            risk_free_rate=risk_free_rate,
            initial_variance=volatility**2,
            kappa=heston_config["kappa"],
            long_run_variance=heston_config["long_run_variance"],
            vol_of_vol=heston_config["vol_of_vol"],
            rho=heston_config["rho"],
            dividend_yield=option_config["dividend_yield"],
            num_paths=simulation_config["num_paths"],
            trading_days_per_year=simulation_config["trading_days_per_year"],
            seed=simulation_config["random_seed"] + scenario_number,
        )

        bsm_price = bsm_result["chooser_price"]
        heston_price = heston_result["price"]

        results.append(
            {
                "scenario": scenario_number,
                "date": row["date"].date().isoformat(),
                "spot_price": spot_price,
                "strike": strike,
                "risk_free_rate": risk_free_rate,
                "rolling_vol_60": volatility,
                "volatility_regime": (
                    "high"
                    if volatility >= high_volatility_threshold
                    else "normal_or_low"
                ),
                "bsm_price": bsm_price,
                "heston_price": heston_price,
                "heston_mc_standard_error": heston_result["standard_error"],
                "bsm_minus_heston": bsm_price - heston_price,
                "absolute_difference": abs(bsm_price - heston_price),
            }
        )

        print(
            f"[{scenario_number:02d}/{len(scenarios):02d}] "
            f"{row['date'].date()}  "
            f"BSM={bsm_price:.4f}  Heston={heston_price:.4f}"
        )

    results_df = pd.DataFrame(results)
    metrics_df = pd.DataFrame(
        [
            calculate_metrics(results_df, "all_scenarios"),
            calculate_metrics(
                results_df[results_df["volatility_regime"] == "normal_or_low"],
                "normal_or_low_volatility",
            ),
            calculate_metrics(
                results_df[results_df["volatility_regime"] == "high"],
                "high_volatility",
            ),
        ]
    )

    results_path = output_dir / "bsm_heston_scenario_results.csv"
    metrics_path = output_dir / "bsm_heston_benchmark_metrics.csv"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print()
    print("Synthetic benchmark metrics")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Scenario-level results saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("Reminder: Heston values are synthetic reference prices, not OTC transactions.")


if __name__ == "__main__":
    main()