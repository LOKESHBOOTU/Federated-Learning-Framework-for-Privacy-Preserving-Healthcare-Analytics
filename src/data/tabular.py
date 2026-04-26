from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.partition import stratified_train_test_indices


@dataclass
class TabularSplit:
    dataset_name: str
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    normalization: dict[str, list[float]]
    source_path: str
    preprocessing_summary: dict[str, Any]


def split_feature_frame(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    y = target.to_numpy(dtype=int)
    train_indices, test_indices = stratified_train_test_indices(y, test_size=test_size, seed=seed)
    X_train = features.iloc[train_indices].reset_index(drop=True).copy()
    X_test = features.iloc[test_indices].reset_index(drop=True).copy()
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test


def impute_from_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    median_columns: list[str] | None = None,
    mode_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    fill_values: dict[str, float | int | str] = {}

    for column in median_columns or []:
        value = float(X_train[column].median())
        X_train[column] = X_train[column].fillna(value)
        X_test[column] = X_test[column].fillna(value)
        fill_values[column] = value

    for column in mode_columns or []:
        modes = X_train[column].mode(dropna=True)
        if modes.empty:
            raise ValueError(f"Could not compute mode for column '{column}'.")
        value = modes.iloc[0]
        X_train[column] = X_train[column].fillna(value)
        X_test[column] = X_test[column].fillna(value)
        fill_values[column] = value

    return X_train, X_test, fill_values


def standardize_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train_array = X_train.to_numpy(dtype=float)
    X_test_array = X_test.to_numpy(dtype=float)
    mean = X_train_array.mean(axis=0)
    std = X_train_array.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (X_train_array - mean) / std, (X_test_array - mean) / std, mean, std


def finalize_tabular_split(
    dataset_name: str,
    feature_names: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    source_path: Path,
    preprocessing_summary: dict[str, Any],
) -> TabularSplit:
    X_train_array, X_test_array, mean, std = standardize_train_test(X_train, X_test)
    summary = dict(preprocessing_summary)
    summary["train_size"] = int(len(y_train))
    summary["test_size"] = int(len(y_test))

    return TabularSplit(
        dataset_name=dataset_name,
        feature_names=feature_names,
        X_train=X_train_array,
        y_train=y_train.astype(int),
        X_test=X_test_array,
        y_test=y_test.astype(int),
        normalization={
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        source_path=str(source_path),
        preprocessing_summary=summary,
    )

