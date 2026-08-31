from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heston_chooser import heston_simple_chooser_price


def run_simulation(num_paths: int) -> dict:
    """Run one reproducible Heston MC pricing experiment."""
    start_time = time.perf_counter()

    result = heston_simple_chooser_price(
        spot=150.0,
        strike=150.0,
        choice_time_years=0.5,
        maturity_years=1.0,
        risk_free_rate=0.05,
        initial_variance=0.30**2,
        kappa=2.0,
        long_run_variance=0.04,
        vol_of_vol=0.5,
        rho=-0.6,
        dividend_yield=0.0,
        num_paths=num_paths,
        trading_days_per_year=252,
        seed=42,
    )

    elapsed_seconds = time.perf_counter() - start_time

    return {
        "num_paths": num_paths,
        "heston_price": result["price"],
        "mc_standard_error": result["standard_error"],
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "week_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    path_counts = [1000, 5000, 10000, 20000, 50000]
    results = []

    for num_paths in path_counts:
        result = run_simulation(num_paths)
        results.append(result)

        print(
            f"Paths={num_paths:>6,} | "
            f"Price={result['heston_price']:.4f} | "
            f"SE={result['mc_standard_error']:.4f} | "
            f"Time={result['elapsed_seconds']:.2f}s"
        )

    results_df = pd.DataFrame(results)

    reference_price = results_df.loc[
        results_df["num_paths"] == 50000,
        "heston_price",
    ].iloc[0]

    results_df["difference_from_50000_path_reference"] = (
        results_df["heston_price"] - reference_price
    ).abs()

    output_path = output_dir / "mc_convergence_results.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print()
    print("Convergence summary")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"Saved to: {output_path}")
    print("Note: The 50,000-path value is an internal MC reference, not a market price.")


if __name__ == "__main__":
    main()