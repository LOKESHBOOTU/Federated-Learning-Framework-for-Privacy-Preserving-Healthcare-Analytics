from __future__ import annotations

from pathlib import Path

from src.core.artifacts import plot_epoch_history, write_json
from src.core.metrics import binary_classification_metrics
from src.core.paths import METRICS_DIR, PLOTS_DIR, ensure_project_dirs
from src.data.registry import load_dataset_split
from src.models.logistic_regression import NumpyLogisticRegression


def run_baseline_experiment(
    dataset_name: str,
    dataset_path: Path | None = None,
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    l2_reg: float = 0.001,
) -> dict:
    ensure_project_dirs()
    split = load_dataset_split(dataset_name, dataset_path=dataset_path, seed=seed)

    model = NumpyLogisticRegression(
        n_features=split.X_train.shape[1],
        learning_rate=learning_rate,
        l2_reg=l2_reg,
        seed=seed,
    )
    history = model.fit(
        split.X_train,
        split.y_train,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    train_probabilities = model.predict_proba(split.X_train)
    test_probabilities = model.predict_proba(split.X_test)
    train_metrics = binary_classification_metrics(split.y_train, train_probabilities)
    test_metrics = binary_classification_metrics(split.y_test, test_probabilities)

    artifact_stem = f"{split.dataset_name}_baseline"
    payload = {
        "experiment": artifact_stem,
        "mode": "baseline",
        "strategy": "baseline",
        "dataset": split.dataset_name,
        "source_path": split.source_path,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "l2_reg": l2_reg,
            "seed": seed,
        },
        "feature_names": split.feature_names,
        "normalization": split.normalization,
        "preprocessing_summary": split.preprocessing_summary,
        "history": history,
        "split_sizes": {
            "train": int(len(split.y_train)),
            "test": int(len(split.y_test)),
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    write_json(METRICS_DIR / f"{artifact_stem}_metrics.json", payload)
    plot_epoch_history(
        [step["loss"] for step in history],
        title=f"{split.dataset_name.title()} Baseline Training Loss",
        ylabel="Loss",
        output_path=PLOTS_DIR / f"{artifact_stem}_loss.png",
    )
    plot_epoch_history(
        [step["accuracy"] for step in history],
        title=f"{split.dataset_name.title()} Baseline Training Accuracy",
        ylabel="Accuracy",
        output_path=PLOTS_DIR / f"{artifact_stem}_accuracy.png",
    )

    return payload

