from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


@dataclass
class MLPConfig:
    input_dim: int
    hidden1: int = 64
    hidden2: int = 32
    learning_rate: float = 0.03
    random_state: int = 42


class NumpyMLP:
    """Small binary classifier MLP used to make FedAvg transparent and lightweight."""

    def __init__(self, config: MLPConfig):
        self.config = config
        rng = np.random.default_rng(config.random_state)
        self.rng = np.random.default_rng(config.random_state)
        self.W1 = rng.normal(0, np.sqrt(2 / config.input_dim), (config.input_dim, config.hidden1))
        self.b1 = np.zeros(config.hidden1)
        self.W2 = rng.normal(0, np.sqrt(2 / config.hidden1), (config.hidden1, config.hidden2))
        self.b2 = np.zeros(config.hidden2)
        self.W3 = rng.normal(0, np.sqrt(2 / config.hidden2), (config.hidden2, 1))
        self.b3 = np.zeros(1)

    def clone(self) -> "NumpyMLP":
        model = NumpyMLP(self.config)
        model.set_params(self.get_params())
        return model

    def get_params(self) -> list[np.ndarray]:
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), self.W3.copy(), self.b3.copy()]

    def set_params(self, params: list[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [p.copy() for p in params]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)
        y_prob = sigmoid(a2 @ self.W3 + self.b3).reshape(-1)
        return y_prob, (a1, a2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_prob, _ = self.forward(X)
        return y_prob

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> float:
        n = len(X)
        order = self.rng.permutation(n)
        losses = []

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = X[idx]
            yb = y[idx].reshape(-1, 1)
            y_prob, (a1, a2) = self.forward(xb)
            pred = y_prob.reshape(-1, 1)

            eps = 1e-8
            loss = -np.mean(yb * np.log(pred + eps) + (1 - yb) * np.log(1 - pred + eps))
            losses.append(loss)

            dz3 = (pred - yb) / len(xb)
            dW3 = a2.T @ dz3
            db3 = dz3.sum(axis=0)

            da2 = dz3 @ self.W3.T
            dz2 = da2 * (a2 > 0)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (a1 > 0)
            dW1 = xb.T @ dz1
            db1 = dz1.sum(axis=0)

            lr = self.config.learning_rate
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W3 -= lr * dW3
            self.b3 -= lr * db3

        return float(np.mean(losses))
