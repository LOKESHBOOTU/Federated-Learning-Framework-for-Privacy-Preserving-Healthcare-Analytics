from __future__ import annotations

import numpy as np


class NumpyLogisticRegression:
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.05,
        l2_reg: float = 0.0,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, size=n_features)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def get_state(self) -> dict[str, np.ndarray | float]:
        return {
            "weights": self.weights.copy(),
            "bias": float(self.bias),
        }

    def set_state(self, state: dict[str, np.ndarray | float]) -> None:
        self.weights = np.array(state["weights"], dtype=float).copy()
        self.bias = float(state["bias"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.compute_objective(X, y)

    def compute_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reference_state: dict[str, np.ndarray | float] | None = None,
        proximal_mu: float = 0.0,
    ) -> float:
        probabilities = self.predict_proba(X)
        clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
        data_loss = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean()
        regularization = 0.5 * self.l2_reg * np.sum(self.weights ** 2)
        proximal_penalty = 0.0

        if reference_state is not None and proximal_mu > 0.0:
            reference_weights = np.array(reference_state["weights"], dtype=float)
            reference_bias = float(reference_state["bias"])
            weight_gap = self.weights - reference_weights
            bias_gap = self.bias - reference_bias
            proximal_penalty = 0.5 * proximal_mu * (
                float(np.sum(weight_gap ** 2)) + bias_gap ** 2
            )

        return float(data_loss + regularization + proximal_penalty)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        seed: int = 42,
        reference_state: dict[str, np.ndarray | float] | None = None,
        proximal_mu: float = 0.0,
    ) -> list[dict[str, float]]:
        rng = np.random.default_rng(seed)
        history: list[dict[str, float]] = []
        indices = np.arange(len(X))

        for epoch in range(epochs):
            rng.shuffle(indices)
            shuffled_X = X[indices]
            shuffled_y = y[indices]

            for start in range(0, len(shuffled_X), batch_size):
                X_batch = shuffled_X[start : start + batch_size]
                y_batch = shuffled_y[start : start + batch_size]

                probabilities = self.predict_proba(X_batch)
                error = probabilities - y_batch

                grad_w = (X_batch.T @ error) / len(X_batch)
                grad_w += self.l2_reg * self.weights
                grad_b = float(np.mean(error))

                if reference_state is not None and proximal_mu > 0.0:
                    reference_weights = np.array(reference_state["weights"], dtype=float)
                    reference_bias = float(reference_state["bias"])
                    grad_w += proximal_mu * (self.weights - reference_weights)
                    grad_b += proximal_mu * (self.bias - reference_bias)

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            predictions = self.predict(X)
            accuracy = float(np.mean(predictions == y))
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": self.compute_objective(
                        X,
                        y,
                        reference_state=reference_state,
                        proximal_mu=proximal_mu,
                    ),
                    "accuracy": accuracy,
                }
            )

        return history
