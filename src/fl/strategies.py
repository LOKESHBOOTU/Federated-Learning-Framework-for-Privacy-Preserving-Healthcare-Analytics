from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.core.metrics import binary_classification_metrics
from src.data.partition import (
    partition_indices_dirichlet,
    partition_indices_iid,
    summarize_client_labels,
)
from src.models.logistic_regression import NumpyLogisticRegression


SUPPORTED_STRATEGIES = {"fedavg", "fedprox"}


@dataclass
class FederatedConfig:
    num_rounds: int = 25
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.05
    l2_reg: float = 0.001
    num_clients: int = 5
    fraction_fit: float = 1.0
    partition_mode: str = "iid"
    dirichlet_alpha: float = 0.5
    seed: int = 42
    strategy: str = "fedavg"
    proximal_mu: float = 0.0


def _validate_strategy(strategy: str) -> str:
    if strategy not in SUPPORTED_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(f"Unsupported strategy '{strategy}'. Supported strategies: {supported}")
    return strategy


def _average_states(
    states: list[dict[str, np.ndarray | float]],
    sample_counts: list[int],
) -> dict[str, np.ndarray | float]:
    total_samples = float(sum(sample_counts))
    averaged_weights = np.zeros_like(states[0]["weights"], dtype=float)
    averaged_bias = 0.0

    for state, count in zip(states, sample_counts):
        weight = count / total_samples
        averaged_weights += np.array(state["weights"], dtype=float) * weight
        averaged_bias += float(state["bias"]) * weight

    return {
        "weights": averaged_weights,
        "bias": averaged_bias,
    }


def _select_clients(
    num_clients: int,
    fraction_fit: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_size = max(1, math.ceil(num_clients * fraction_fit))
    return np.sort(rng.choice(num_clients, size=sample_size, replace=False))


def _build_partitions(
    y_train: np.ndarray,
    config: FederatedConfig,
) -> list[np.ndarray]:
    if config.partition_mode == "iid":
        return partition_indices_iid(y_train, num_clients=config.num_clients, seed=config.seed)
    if config.partition_mode == "dirichlet":
        return partition_indices_dirichlet(
            y_train,
            num_clients=config.num_clients,
            alpha=config.dirichlet_alpha,
            seed=config.seed,
        )
    raise ValueError(f"Unsupported partition mode: {config.partition_mode}")


def run_federated_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    config: FederatedConfig,
) -> dict:
    strategy = _validate_strategy(config.strategy)
    partitions = _build_partitions(y_train, config)
    rng = np.random.default_rng(config.seed)

    global_model = NumpyLogisticRegression(
        n_features=X_train.shape[1],
        learning_rate=config.learning_rate,
        l2_reg=config.l2_reg,
        seed=config.seed,
    )
    algorithm = "FedProx" if strategy == "fedprox" else "FedAvg"
    proximal_mu = config.proximal_mu if strategy == "fedprox" else 0.0

    history: list[dict] = []

    for round_number in range(1, config.num_rounds + 1):
        selected_clients = _select_clients(
            num_clients=config.num_clients,
            fraction_fit=config.fraction_fit,
            rng=rng,
        )
        local_states: list[dict[str, np.ndarray | float]] = []
        local_counts: list[int] = []
        local_losses: list[float] = []
        local_accuracies: list[float] = []
        global_state = global_model.get_state()

        for client_id in selected_clients:
            client_indices = partitions[int(client_id)]
            local_model = NumpyLogisticRegression(
                n_features=X_train.shape[1],
                learning_rate=config.learning_rate,
                l2_reg=config.l2_reg,
                seed=config.seed + int(client_id),
            )
            local_model.set_state(global_state)

            local_history = local_model.fit(
                X_train[client_indices],
                y_train[client_indices],
                epochs=config.local_epochs,
                batch_size=config.batch_size,
                seed=config.seed + round_number * 100 + int(client_id),
                reference_state=(global_state if strategy == "fedprox" else None),
                proximal_mu=proximal_mu,
            )
            last_epoch = local_history[-1]
            local_states.append(local_model.get_state())
            local_counts.append(len(client_indices))
            local_losses.append(float(last_epoch["loss"]))
            local_accuracies.append(float(last_epoch["accuracy"]))

        global_model.set_state(_average_states(local_states, local_counts))

        train_probabilities = global_model.predict_proba(X_train)
        test_probabilities = global_model.predict_proba(X_test)
        train_metrics = binary_classification_metrics(y_train, train_probabilities)
        test_metrics = binary_classification_metrics(y_test, test_probabilities)

        history.append(
            {
                "round": round_number,
                "selected_clients": [int(client_id) for client_id in selected_clients.tolist()],
                "mean_client_loss": float(np.mean(local_losses)),
                "mean_client_accuracy": float(np.mean(local_accuracies)),
                "train_accuracy": train_metrics["accuracy"],
                "train_loss": train_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
            }
        )

    final_test_probabilities = global_model.predict_proba(X_test)
    final_train_probabilities = global_model.predict_proba(X_train)

    return {
        "algorithm": algorithm,
        "strategy": strategy,
        "feature_names": feature_names,
        "config": asdict(config),
        "client_sizes": {
            f"client_{client_id}": int(len(indices))
            for client_id, indices in enumerate(partitions)
        },
        "client_label_distribution": summarize_client_labels(y_train, partitions),
        "history": history,
        "final_train_metrics": binary_classification_metrics(y_train, final_train_probabilities),
        "final_test_metrics": binary_classification_metrics(y_test, final_test_probabilities),
        "proximal_mu": proximal_mu,
    }

