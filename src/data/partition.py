from __future__ import annotations

from collections import Counter

import numpy as np


def stratified_train_test_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0].copy()
        rng.shuffle(label_indices)
        label_test_size = max(1, int(round(len(label_indices) * test_size)))
        label_test_size = min(label_test_size, len(label_indices) - 1)

        test_parts.append(label_indices[:label_test_size])
        train_parts.append(label_indices[label_test_size:])

    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def partition_indices_iid(
    y: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for label in np.unique(y):
        label_indices = np.where(y == label)[0].copy()
        rng.shuffle(label_indices)
        splits = np.array_split(label_indices, num_clients)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return [np.array(indices, dtype=int) for indices in client_indices]


def partition_indices_dirichlet(
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_size: int = 20,
    max_attempts: int = 100,
) -> list[np.ndarray]:
    if alpha <= 0:
        raise ValueError("alpha must be positive for Dirichlet partitioning.")

    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]

        for label in np.unique(y):
            label_indices = np.where(y == label)[0].copy()
            rng.shuffle(label_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            boundaries = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
            splits = np.split(label_indices, boundaries)
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_size:
            return [np.array(indices, dtype=int) for indices in client_indices]

    raise ValueError(
        "Could not create a stable Dirichlet partition. Try a larger alpha value."
    )


def summarize_client_labels(y: np.ndarray, partitions: list[np.ndarray]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for client_id, indices in enumerate(partitions):
        counts = Counter(y[indices].tolist())
        summary[f"client_{client_id}"] = {str(label): int(count) for label, count in counts.items()}
    return summary

