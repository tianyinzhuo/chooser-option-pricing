"""Integration checks for the completed Week 6 modelling pipeline."""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "week_6"
MODEL_DIR = PROJECT_ROOT / "models" / "week_6"
CONFIG_PATH = PROJECT_ROOT / "config" / "week_6_model_search.json"


class TestWeek6Pipeline(unittest.TestCase):
    """Verify that the Week 6 deliverables are complete and consistent."""

    def test_required_model_files_exist(self):
        required_models = [
            "week_6_tuned_volatility_random_forest.joblib",
            "week_6_tuned_ridge_pricing.joblib",
            "week_6_tuned_random_forest_pricing.joblib",
            "week_6_tuned_gbdt_pricing.joblib",
            "week_6_tuned_mlp_pricing.joblib",
        ]

        for filename in required_models:
            path = MODEL_DIR / filename
            self.assertTrue(path.is_file(), f"Missing model: {path}")
            self.assertGreater(path.stat().st_size, 0)

    def test_final_comparison_contains_all_approaches(self):
        comparison = pd.read_csv(
            OUTPUT_DIR / "final_pricing_approach_test_comparison.csv"
        )

        expected_models = {
            "bsm_closed_form_baseline",
            "route_a_ml_volatility_plus_bsm",
            "tuned_ridge",
            "tuned_random_forest",
            "tuned_gbdt",
            "tuned_mlp",
        }

        self.assertEqual(set(comparison["model_name"]), expected_models)
        self.assertEqual(len(comparison), 6)

        metric_values = comparison[["mae", "rmse", "r2"]].to_numpy()
        self.assertTrue(np.isfinite(metric_values).all())
        self.assertTrue((comparison[["mae", "rmse"]] >= 0).all().all())

    def test_route_a_improves_on_original_bsm(self):
        comparison = pd.read_csv(
            OUTPUT_DIR / "final_pricing_approach_test_comparison.csv"
        ).set_index("model_name")

        route_a = comparison.loc["route_a_ml_volatility_plus_bsm"]
        bsm = comparison.loc["bsm_closed_form_baseline"]

        self.assertLess(route_a["mae"], bsm["mae"])
        self.assertLess(route_a["rmse"], bsm["rmse"])
        self.assertGreater(route_a["r2"], bsm["r2"])

    def test_test_predictions_are_complete(self):
        predictions = pd.read_csv(
            OUTPUT_DIR / "final_pricing_approach_test_predictions.csv"
        )

        prediction_columns = [
            "heston_synthetic_price",
            "bsm_price",
            "route_a_predicted_price",
            "ridge_predicted_price",
            "random_forest_predicted_price",
            "gbdt_predicted_price",
            "mlp_predicted_price",
        ]

        self.assertEqual(len(predictions), 60)
        self.assertEqual(pd.to_datetime(predictions["date"]).nunique(), 10)
        self.assertFalse(predictions[prediction_columns].isna().any().any())
        self.assertTrue(
            np.isfinite(predictions[prediction_columns].to_numpy()).all()
        )

    def test_cv_and_test_selection_are_reported_separately(self):
        cv_results = pd.read_csv(
            OUTPUT_DIR / "pricing_cv_model_comparison.csv"
        ).sort_values("rmse")
        test_results = pd.read_csv(
            OUTPUT_DIR / "final_pricing_approach_test_comparison.csv"
        ).sort_values("rmse")

        self.assertEqual(cv_results.iloc[0]["model_name"], "tuned_ridge")
        self.assertEqual(test_results.iloc[0]["model_name"], "tuned_gbdt")

    def test_shap_deliverables_exist_and_disclose_limits(self):
        shap_dir = OUTPUT_DIR / "shap"
        required_images = [
            "gbdt_shap_summary_bar_usd.png",
            "gbdt_shap_beeswarm_usd.png",
            "gbdt_shap_waterfall_representative_usd.png",
        ]

        for filename in required_images:
            path = shap_dir / filename
            self.assertTrue(path.is_file(), f"Missing SHAP chart: {path}")
            self.assertGreater(path.stat().st_size, 10000)

        summary = json.loads(
            (shap_dir / "gbdt_shap_analysis_summary.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(summary["explainer"], "SHAP TreeExplainer")
        self.assertEqual(summary["samples_explained"], 60)
        self.assertIn("not", summary["interpretation_warning"].lower())
        self.assertIn("synthetic", summary["target_disclosure"].lower())

    def test_configuration_requires_all_report_metrics(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))

        self.assertEqual(config["selection_metric"], "rmse")
        self.assertEqual(
            set(config["report_metrics"]),
            {"mae", "rmse", "r2"},
        )
        self.assertTrue(config["disclosures"])


if __name__ == "__main__":
    unittest.main()
