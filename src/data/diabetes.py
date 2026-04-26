from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.paths import resolve_dataset_path
from src.data.tabular import (
    TabularSplit,
    finalize_tabular_split,
    impute_from_train,
    split_feature_frame,
)


FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_diabetes_split(
    dataset_path: Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> TabularSplit:
    source_path = dataset_path or resolve_dataset_path("diabetes.csv")
    dataframe = pd.read_csv(source_path)

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"diabetes.csv is missing columns: {missing_columns}")

    dataframe = dataframe[expected_columns].copy()
    dataframe[FEATURE_COLUMNS] = dataframe[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    dataframe[TARGET_COLUMN] = pd.to_numeric(dataframe[TARGET_COLUMN], errors="raise").astype(int)

    zero_as_missing_counts = {
        column: int((dataframe[column] == 0).sum()) for column in ZERO_AS_MISSING_COLUMNS
    }
    dataframe[ZERO_AS_MISSING_COLUMNS] = dataframe[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)

    features = dataframe[FEATURE_COLUMNS]
    target = dataframe[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = split_feature_frame(
        features,
        target,
        test_size=test_size,
        seed=seed,
    )
    X_train, X_test, fill_values = impute_from_train(
        X_train,
        X_test,
        median_columns=ZERO_AS_MISSING_COLUMNS,
    )

    return finalize_tabular_split(
        dataset_name="diabetes",
        feature_names=FEATURE_COLUMNS,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        source_path=source_path,
        preprocessing_summary={
            "source_file": source_path.name,
            "target_mapping": {"0": 0, "1": 1},
            "zero_as_missing_columns": ZERO_AS_MISSING_COLUMNS,
            "zero_as_missing_counts": zero_as_missing_counts,
            "median_imputation_columns": list(fill_values.keys()),
            "mode_imputation_columns": [],
            "imputation_values": fill_values,
        },
    )

