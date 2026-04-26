from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.paths import resolve_dataset_path
from src.data.tabular import (
    TabularSplit,
    finalize_tabular_split,
    impute_from_train,
    split_feature_frame,
)


FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio",
]
TARGET_COLUMN = "Dataset"
GENDER_MAPPING = {"Female": 0.0, "Male": 1.0}
TARGET_MAPPING = {1: 1, 2: 0}


def load_liver_split(
    dataset_path: Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> TabularSplit:
    source_path = dataset_path or resolve_dataset_path("indian_liver_patient.csv")
    dataframe = pd.read_csv(source_path)

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"indian_liver_patient.csv is missing columns: {missing_columns}")

    dataframe = dataframe[expected_columns].copy()
    dataframe["Gender"] = dataframe["Gender"].map(GENDER_MAPPING)
    if dataframe["Gender"].isna().any():
        raise ValueError("Found unknown values in liver Gender column.")

    numeric_columns = [column for column in FEATURE_COLUMNS if column != "Gender"]
    dataframe[numeric_columns] = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce")

    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="raise").map(TARGET_MAPPING)
    if target.isna().any():
        raise ValueError("Found unknown values in liver Dataset target column.")

    features = dataframe[FEATURE_COLUMNS]
    X_train, X_test, y_train, y_test = split_feature_frame(
        features,
        target,
        test_size=test_size,
        seed=seed,
    )
    X_train, X_test, fill_values = impute_from_train(
        X_train,
        X_test,
        median_columns=["Albumin_and_Globulin_Ratio"],
    )

    return finalize_tabular_split(
        dataset_name="liver",
        feature_names=FEATURE_COLUMNS,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        source_path=source_path,
        preprocessing_summary={
            "source_file": source_path.name,
            "gender_mapping": {"Female": 0, "Male": 1},
            "target_mapping": {"1": 1, "2": 0},
            "median_imputation_columns": list(fill_values.keys()),
            "mode_imputation_columns": [],
            "imputation_values": fill_values,
        },
    )

