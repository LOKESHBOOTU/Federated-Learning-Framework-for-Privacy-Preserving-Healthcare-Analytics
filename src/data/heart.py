from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.paths import resolve_dataset_path
from src.data.tabular import TabularSplit, finalize_tabular_split, split_feature_frame


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"


def load_heart_dataframe(dataset_path: Path | None = None) -> pd.DataFrame:
    path = dataset_path or resolve_dataset_path("heart.csv")
    dataframe = pd.read_csv(path)

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"heart.csv is missing columns: {missing_columns}")

    dataframe = dataframe[expected_columns].copy()
    dataframe = dataframe.dropna()
    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)
    return dataframe


def load_heart_split(
    dataset_path: Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> TabularSplit:
    source_path = dataset_path or resolve_dataset_path("heart.csv")
    dataframe = load_heart_dataframe(source_path)
    features = dataframe[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="raise").astype(int)
    X_train, X_test, y_train, y_test = split_feature_frame(
        features,
        target,
        test_size=test_size,
        seed=seed,
    )

    return finalize_tabular_split(
        dataset_name="heart",
        feature_names=FEATURE_COLUMNS,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        source_path=source_path,
        preprocessing_summary={
            "source_file": source_path.name,
            "target_mapping": {"0": 0, "1": 1},
            "dropped_rows_with_missing": 0,
            "median_imputation_columns": [],
            "mode_imputation_columns": [],
        },
    )

