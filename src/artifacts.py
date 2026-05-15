from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import train_baselines
from src.config import METRICS_DIR, RESULTS_DIR
from src.data import PreparedData
from src.federated import FederatedConfig, run_federated_training
from src.model import MLPConfig, NumpyMLP


FIXED_CDC_CONFIG = FederatedConfig(
    num_clients=5,
    rounds=15,
    local_epochs=2,
    partition="iid",
    use_dp=True,
    noise_multiplier=0.05,
    learning_rate=0.03,
    random_state=77,
)

FIXED_HEART_CONFIG = FederatedConfig(
    num_clients=3,
    rounds=40,
    local_epochs=3,
    partition="iid",
    use_dp=True,
    noise_multiplier=0.01,
    learning_rate=0.02,
)

DEFAULT_ARTIFACT_KEY = "diabetes"


def fixed_config_for_artifact(artifact_key: str = DEFAULT_ARTIFACT_KEY) -> FederatedConfig:
    if artifact_key == "heart_disease":
        return FIXED_HEART_CONFIG
    return FIXED_CDC_CONFIG


def artifact_paths(artifact_key: str = DEFAULT_ARTIFACT_KEY) -> dict[str, Path]:
    directory = RESULTS_DIR / f"saved_{artifact_key}_run"
    return {
        "dir": directory,
        "model": directory / "federated_model.npz",
        "history": directory / "federated_history.csv",
        "clients": directory / "client_summary.csv",
        "baselines": directory / "baseline_metrics.json",
        "metadata": directory / "metadata.json",
    }


def make_client_summary(client_parts) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Client": [f"Hospital {i + 1}" for i in range(len(client_parts))],
            "Records": [len(part[1]) for part in client_parts],
            "Positive Rate": [float(part[1].mean()) for part in client_parts],
        }
    )


def save_federated_model(model: NumpyMLP, path: Path) -> None:
    params = model.get_params()
    np.savez_compressed(path, **{f"param_{idx}": param for idx, param in enumerate(params)})


def load_federated_model(path: Path, input_dim: int, config: FederatedConfig) -> NumpyMLP:
    model = NumpyMLP(
        MLPConfig(
            input_dim=input_dim,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
        )
    )
    saved = np.load(path)
    params = [saved[f"param_{idx}"] for idx in range(6)]
    model.set_params(params)
    return model


def _metadata_matches(data: PreparedData, artifact_key: str = DEFAULT_ARTIFACT_KEY) -> bool:
    paths = artifact_paths(artifact_key)
    if not paths["metadata"].exists():
        return False
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    fixed_config = fixed_config_for_artifact(artifact_key)
    return (
        metadata.get("source_path") == str(data.source_path)
        and metadata.get("feature_names") == data.feature_names
        and metadata.get("target_name") == data.target_name
        and metadata.get("artifact_key") == artifact_key
        and metadata.get("fixed_config") == asdict(fixed_config)
    )


def saved_artifacts_exist(data: PreparedData, artifact_key: str = DEFAULT_ARTIFACT_KEY) -> bool:
    paths = artifact_paths(artifact_key)
    required = [
        paths["model"],
        paths["history"],
        paths["clients"],
        paths["baselines"],
        paths["metadata"],
    ]
    return all(path.exists() for path in required) and _metadata_matches(data, artifact_key)


def train_and_save_fixed_cdc_results(
    data: PreparedData,
    artifact_key: str = DEFAULT_ARTIFACT_KEY,
) -> tuple[pd.DataFrame, pd.DataFrame, NumpyMLP, dict]:
    paths = artifact_paths(artifact_key)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fixed_config = fixed_config_for_artifact(artifact_key)

    baseline_results, _ = train_baselines(data.X_train, data.y_train, data.X_test, data.y_test)
    model, history, client_parts = run_federated_training(
        data.X_train,
        data.y_train,
        data.X_test,
        data.y_test,
        fixed_config,
    )

    history_df = pd.DataFrame(history)
    client_summary = make_client_summary(client_parts)
    metadata = {
        "dataset_name": data.name,
        "source_path": str(data.source_path),
        "target_name": data.target_name,
        "feature_names": data.feature_names,
        "artifact_key": artifact_key,
        "fixed_config": asdict(fixed_config),
        "prediction_source": "saved fixed federated model",
    }

    save_federated_model(model, paths["model"])
    history_df.to_csv(paths["history"], index=False)
    client_summary.to_csv(paths["clients"], index=False)
    paths["baselines"].write_text(json.dumps(baseline_results, indent=2), encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return history_df, client_summary, model, baseline_results


def load_fixed_cdc_results(
    data: PreparedData,
    artifact_key: str = DEFAULT_ARTIFACT_KEY,
) -> tuple[pd.DataFrame, pd.DataFrame, NumpyMLP, dict]:
    paths = artifact_paths(artifact_key)
    history_df = pd.read_csv(paths["history"])
    client_summary = pd.read_csv(paths["clients"])
    baseline_results = json.loads(paths["baselines"].read_text(encoding="utf-8"))
    model = load_federated_model(
        paths["model"],
        input_dim=len(data.feature_names),
        config=fixed_config_for_artifact(artifact_key),
    )
    return history_df, client_summary, model, baseline_results


def get_or_create_fixed_cdc_results(
    data: PreparedData,
    artifact_key: str = DEFAULT_ARTIFACT_KEY,
) -> tuple[pd.DataFrame, pd.DataFrame, NumpyMLP, dict, bool]:
    if saved_artifacts_exist(data, artifact_key):
        history_df, client_summary, model, baseline_results = load_fixed_cdc_results(data, artifact_key)
        return history_df, client_summary, model, baseline_results, True

    history_df, client_summary, model, baseline_results = train_and_save_fixed_cdc_results(data, artifact_key)
    return history_df, client_summary, model, baseline_results, False
