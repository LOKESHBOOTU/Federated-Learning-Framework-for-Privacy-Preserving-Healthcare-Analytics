from __future__ import annotations

from typing import Any

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    clipped = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
    losses = -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
    return float(np.mean(losses))


def binary_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(-y_prob)
    sorted_y = y_true[order]

    true_positive = np.cumsum(sorted_y == 1)
    false_positive = np.cumsum(sorted_y == 0)

    tpr = np.concatenate(([0.0], true_positive / positives))
    fpr = np.concatenate(([0.0], false_positive / negatives))
    return float(np.trapezoid(tpr, fpr))


def binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = _safe_divide(true_positive + true_negative, len(y_true))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": binary_roc_auc(y_true, y_prob),
        "loss": binary_cross_entropy(y_true, y_prob),
        "confusion_matrix": {
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
    }
