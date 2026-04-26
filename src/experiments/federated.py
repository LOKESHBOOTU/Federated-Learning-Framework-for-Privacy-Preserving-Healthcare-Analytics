from __future__ import annotations

from pathlib import Path

from src.core.artifacts import plot_history, write_json
from src.core.paths import METRICS_DIR, PLOTS_DIR, ensure_project_dirs
from src.data.registry import load_dataset_split
from src.fl.strategies import FederatedConfig, run_federated_training


def run_federated_experiment(
    dataset_name: str,
    dataset_path: Path | None = None,
    config: FederatedConfig | None = None,
) -> dict:
    ensure_project_dirs()
    active_config = config or FederatedConfig()
    split = load_dataset_split(
        dataset_name,
        dataset_path=dataset_path,
        seed=active_config.seed,
    )

    payload = run_federated_training(
        X_train=split.X_train,
        y_train=split.y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        feature_names=split.feature_names,
        config=active_config,
    )
    artifact_stem = f"{split.dataset_name}_{payload['strategy']}"
    payload["experiment"] = artifact_stem
    payload["mode"] = "federated"
    payload["dataset"] = split.dataset_name
    payload["source_path"] = split.source_path
    payload["normalization"] = split.normalization
    payload["preprocessing_summary"] = split.preprocessing_summary
    payload["split_sizes"] = {
        "train": int(len(split.y_train)),
        "test": int(len(split.y_test)),
    }

    write_json(METRICS_DIR / f"{artifact_stem}_metrics.json", payload)
    plot_history(
        rounds=[step["round"] for step in payload["history"]],
        values=[step["test_accuracy"] for step in payload["history"]],
        title=f"{split.dataset_name.title()} {payload['algorithm']} Test Accuracy",
        ylabel="Accuracy",
        output_path=PLOTS_DIR / f"{artifact_stem}_test_accuracy.png",
    )
    plot_history(
        rounds=[step["round"] for step in payload["history"]],
        values=[step["test_loss"] for step in payload["history"]],
        title=f"{split.dataset_name.title()} {payload['algorithm']} Test Loss",
        ylabel="Loss",
        output_path=PLOTS_DIR / f"{artifact_stem}_test_loss.png",
    )

    return payload

