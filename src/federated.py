from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation import classification_metrics
from src.model import MLPConfig, NumpyMLP


@dataclass
class FederatedConfig:
    num_clients: int = 5
    rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.03
    hidden1: int = 64
    hidden2: int = 32
    partition: str = "iid"
    use_dp: bool = True
    clip_norm: float = 1.0
    noise_multiplier: float = 0.05
    random_state: int = 42


def make_partitions(X: np.ndarray, y: np.ndarray, num_clients: int, mode: str, random_state: int):
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))

    if mode.lower() == "non-iid":
        indices = indices[np.argsort(y)]
        shards = np.array_split(indices, num_clients * 2)
        rng.shuffle(shards)
        client_indices = [np.concatenate(shards[i::num_clients]) for i in range(num_clients)]
    else:
        rng.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)

    return [(X[idx], y[idx], idx) for idx in client_indices]


def aggregate_params(global_params, client_params, client_sizes):
    total = float(sum(client_sizes))
    new_params = []
    for layer_idx, base in enumerate(global_params):
        layer = np.zeros_like(base)
        for params, size in zip(client_params, client_sizes):
            layer += params[layer_idx] * (size / total)
        new_params.append(layer)
    return new_params


def privatize_update(update, clip_norm: float, noise_multiplier: float, rng: np.random.Generator):
    flat_norm = np.sqrt(sum(float(np.sum(delta**2)) for delta in update))
    scale = min(1.0, clip_norm / (flat_norm + 1e-12))
    private = []
    for delta in update:
        clipped = delta * scale
        noise = rng.normal(0, noise_multiplier * clip_norm, size=delta.shape)
        private.append(clipped + noise)
    return private


def run_federated_training(X_train, y_train, X_test, y_test, config: FederatedConfig):
    model = NumpyMLP(
        MLPConfig(
            input_dim=X_train.shape[1],
            hidden1=config.hidden1,
            hidden2=config.hidden2,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
        )
    )
    clients = make_partitions(X_train, y_train, config.num_clients, config.partition, config.random_state)
    rng = np.random.default_rng(config.random_state)
    history = []

    for round_id in range(1, config.rounds + 1):
        global_params = model.get_params()
        uploaded_params = []
        client_sizes = []
        client_losses = []

        for client_id, (client_x, client_y, _) in enumerate(clients, start=1):
            local_model = model.clone()
            losses = [
                local_model.train_epoch(client_x, client_y, batch_size=config.batch_size)
                for _ in range(config.local_epochs)
            ]
            local_params = local_model.get_params()

            if config.use_dp:
                update = [local - global_ for local, global_ in zip(local_params, global_params)]
                private_update = privatize_update(update, config.clip_norm, config.noise_multiplier, rng)
                local_params = [global_ + delta for global_, delta in zip(global_params, private_update)]

            uploaded_params.append(local_params)
            client_sizes.append(len(client_x))
            client_losses.append(float(np.mean(losses)))

        model.set_params(aggregate_params(global_params, uploaded_params, client_sizes))
        y_prob = model.predict_proba(X_test)
        metrics = classification_metrics(y_test, y_prob)
        metrics.update(
            {
                "round": round_id,
                "loss": float(np.mean(client_losses)),
                "clients": config.num_clients,
                "partition": config.partition,
                "dp_enabled": bool(config.use_dp),
            }
        )
        history.append(metrics)

    return model, history, clients
