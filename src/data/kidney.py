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


NUMERIC_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
]
CATEGORICAL_COLUMNS = [
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]
TARGET_COLUMN = "classification"
CATEGORICAL_MAPPINGS = {
    "rbc": {"normal": 0.0, "abnormal": 1.0},
    "pc": {"normal": 0.0, "abnormal": 1.0},
    "pcc": {"notpresent": 0.0, "present": 1.0},
    "ba": {"notpresent": 0.0, "present": 1.0},
    "htn": {"no": 0.0, "yes": 1.0},
    "dm": {"no": 0.0, "yes": 1.0},
    "cad": {"no": 0.0, "yes": 1.0},
    "appet": {"good": 0.0, "poor": 1.0},
    "pe": {"no": 0.0, "yes": 1.0},
    "ane": {"no": 0.0, "yes": 1.0},
}
TARGET_MAPPING = {"ckd": 1, "notckd": 0}


def _strip_string_cells(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return cleaned


def load_kidney_split(
    dataset_path: Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> TabularSplit:
    source_path = dataset_path or resolve_dataset_path("kidney_disease.csv")
    dataframe = pd.read_csv(source_path)
    dataframe = _strip_string_cells(dataframe)

    required_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN, "id"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"kidney_disease.csv is missing columns: {missing_columns}")

    dataframe = dataframe.drop(columns=["id"])
    dataframe[NUMERIC_COLUMNS] = dataframe[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    for column, mapping in CATEGORICAL_MAPPINGS.items():
        dataframe[column] = dataframe[column].map(mapping)

    target = dataframe[TARGET_COLUMN].map(TARGET_MAPPING)
    if target.isna().any():
        raise ValueError("Found unknown values in kidney classification target column.")

    features = dataframe[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    X_train, X_test, y_train, y_test = split_feature_frame(
        features,
        target,
        test_size=test_size,
        seed=seed,
    )
    X_train, X_test, median_fill_values = impute_from_train(
        X_train,
        X_test,
        median_columns=NUMERIC_COLUMNS,
    )
    X_train, X_test, mode_fill_values = impute_from_train(
        X_train,
        X_test,
        mode_columns=CATEGORICAL_COLUMNS,
    )

    imputation_values = dict(median_fill_values)
    imputation_values.update(mode_fill_values)

    return finalize_tabular_split(
        dataset_name="kidney",
        feature_names=NUMERIC_COLUMNS + CATEGORICAL_COLUMNS,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        source_path=source_path,
        preprocessing_summary={
            "source_file": source_path.name,
            "dropped_columns": ["id"],
            "trimmed_string_columns": ["classification", "dm", "cad", "rbc", "pc"],
            "categorical_columns_encoded": CATEGORICAL_COLUMNS,
            "target_mapping": {"ckd": 1, "notckd": 0},
            "median_imputation_columns": NUMERIC_COLUMNS,
            "mode_imputation_columns": CATEGORICAL_COLUMNS,
            "imputation_values": imputation_values,
        },
    )

