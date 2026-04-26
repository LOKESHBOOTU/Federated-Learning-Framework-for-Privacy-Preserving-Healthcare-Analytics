from __future__ import annotations

import unittest

import numpy as np

from src.data.registry import SUPPORTED_DATASETS, load_dataset_split
from src.experiments.baseline import run_baseline_experiment
from src.experiments.compare import run_benchmark_comparison
from src.experiments.federated import run_federated_experiment
from src.fl.strategies import FederatedConfig, run_federated_training


class DatasetAdapterTests(unittest.TestCase):
    def test_all_dataset_loaders_return_binary_targets_and_no_nan_features(self) -> None:
        for dataset_name in sorted(SUPPORTED_DATASETS):
            with self.subTest(dataset=dataset_name):
                split = load_dataset_split(dataset_name, seed=42)
                self.assertGreater(len(split.feature_names), 0)
                self.assertTrue(set(np.unique(split.y_train)).issubset({0, 1}))
                self.assertTrue(set(np.unique(split.y_test)).issubset({0, 1}))
                self.assertFalse(np.isnan(split.X_train).any())
                self.assertFalse(np.isnan(split.X_test).any())

    def test_diabetes_zero_as_missing_summary_matches_dataset_profile(self) -> None:
        split = load_dataset_split("diabetes", seed=42)
        self.assertEqual(
            split.preprocessing_summary["zero_as_missing_counts"],
            {
                "Glucose": 5,
                "BloodPressure": 35,
                "SkinThickness": 227,
                "Insulin": 374,
                "BMI": 11,
            },
        )
        self.assertEqual(
            split.preprocessing_summary["median_imputation_columns"],
            ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        )

    def test_liver_mapping_summary_is_stable(self) -> None:
        split = load_dataset_split("liver", seed=42)
        self.assertEqual(split.preprocessing_summary["gender_mapping"], {"Female": 0, "Male": 1})
        self.assertEqual(split.preprocessing_summary["target_mapping"], {"1": 1, "2": 0})

    def test_kidney_cleanup_summary_drops_id_and_tracks_whitespace_fix(self) -> None:
        split = load_dataset_split("kidney", seed=42)
        self.assertNotIn("id", split.feature_names)
        self.assertEqual(split.preprocessing_summary["target_mapping"], {"ckd": 1, "notckd": 0})
        self.assertIn("classification", split.preprocessing_summary["trimmed_string_columns"])


class StrategyAndComparisonTests(unittest.TestCase):
    def test_fedprox_zero_matches_fedavg(self) -> None:
        split = load_dataset_split("heart", seed=42)
        base_kwargs = {
            "num_rounds": 3,
            "local_epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.05,
            "l2_reg": 0.001,
            "num_clients": 5,
            "fraction_fit": 1.0,
            "partition_mode": "iid",
            "seed": 42,
        }
        fedavg_payload = run_federated_training(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            split.feature_names,
            FederatedConfig(strategy="fedavg", **base_kwargs),
        )
        fedprox_payload = run_federated_training(
            split.X_train,
            split.y_train,
            split.X_test,
            split.y_test,
            split.feature_names,
            FederatedConfig(strategy="fedprox", proximal_mu=0.0, **base_kwargs),
        )

        self.assertAlmostEqual(
            fedavg_payload["final_test_metrics"]["accuracy"],
            fedprox_payload["final_test_metrics"]["accuracy"],
            places=12,
        )
        self.assertAlmostEqual(
            fedavg_payload["final_test_metrics"]["f1"],
            fedprox_payload["final_test_metrics"]["f1"],
            places=12,
        )

    def test_fedprox_positive_runs_and_compare_outputs_summary(self) -> None:
        run_baseline_experiment("diabetes", seed=42, epochs=5, batch_size=32)
        run_federated_experiment(
            "diabetes",
            config=FederatedConfig(
                num_rounds=2,
                local_epochs=2,
                batch_size=32,
                learning_rate=0.05,
                l2_reg=0.001,
                num_clients=5,
                fraction_fit=1.0,
                partition_mode="iid",
                dirichlet_alpha=0.5,
                seed=42,
                strategy="fedprox",
                proximal_mu=0.01,
            ),
        )
        payload = run_benchmark_comparison()
        self.assertGreaterEqual(payload["run_count"], 1)


if __name__ == "__main__":
    unittest.main()
