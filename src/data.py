from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DATASET_CANDIDATES, RAW_DATA_DIR, DEFAULT_RANDOM_STATE


@dataclass
class PreparedData:
    name: str
    source_path: Path
    frame: pd.DataFrame
    feature_names: list[str]
    target_name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


CDC_FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


def find_main_dataset() -> tuple[Path, str]:
    for path, dataset_name in DATASET_CANDIDATES:
        if path.exists():
            return path, dataset_name

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        path = csv_files[0]
        return path, f"Original Healthcare Dataset ({path.stem})"

    raise FileNotFoundError(
        "No original dataset CSV found. Add your dataset under data/raw/ as diabetes.csv, "
        "heart.csv, kidney_disease.csv, liver.csv, or any .csv file. You can still run with "
        "synthetic_demo=True for a generated demo dataset."
    )


def make_synthetic_cdc_data(rows: int = 5000, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Create CDC-like demo data for UI/testing when the real dataset has not been added."""
    rng = np.random.default_rng(random_state)
    frame = pd.DataFrame(
        {
            "HighBP": rng.binomial(1, 0.42, rows),
            "HighChol": rng.binomial(1, 0.40, rows),
            "CholCheck": rng.binomial(1, 0.96, rows),
            "BMI": np.clip(rng.normal(28.5, 6.5, rows), 12, 60).round().astype(int),
            "Smoker": rng.binomial(1, 0.44, rows),
            "HeartDiseaseorAttack": rng.binomial(1, 0.09, rows),
            "PhysActivity": rng.binomial(1, 0.75, rows),
            "Fruits": rng.binomial(1, 0.63, rows),
            "Veggies": rng.binomial(1, 0.81, rows),
            "HvyAlcoholConsump": rng.binomial(1, 0.06, rows),
            "AnyHealthcare": rng.binomial(1, 0.95, rows),
            "NoDocbcCost": rng.binomial(1, 0.08, rows),
            "GenHlth": rng.integers(1, 6, rows),
            "MentHlth": rng.integers(0, 31, rows),
            "PhysHlth": rng.integers(0, 31, rows),
            "DiffWalk": rng.binomial(1, 0.17, rows),
            "Sex": rng.binomial(1, 0.48, rows),
            "Age": rng.integers(1, 14, rows),
            "Education": rng.integers(1, 7, rows),
            "Income": rng.integers(1, 9, rows),
        }
    )
    logits = (
        -5.2
        + 0.8 * frame["HighBP"]
        + 0.65 * frame["HighChol"]
        + 0.08 * (frame["BMI"] - 25)
        + 0.55 * frame["HeartDiseaseorAttack"]
        + 0.22 * frame["GenHlth"]
        + 0.12 * frame["DiffWalk"]
        + 0.13 * frame["Age"]
        - 0.12 * frame["PhysActivity"]
        - 0.08 * frame["Income"]
    )
    prob = 1 / (1 + np.exp(-logits))
    frame["Diabetes_binary"] = rng.binomial(1, prob)
    return frame


def load_diabetes_frame(path: Path | None = None, synthetic_demo: bool = False) -> tuple[pd.DataFrame, str, Path]:
    if synthetic_demo:
        return make_synthetic_cdc_data(), "Synthetic CDC-like Diabetes Demo", Path("in-memory")
    if path is None:
        path, dataset_name = find_main_dataset()
    else:
        path = Path(path)
        dataset_name = "Custom Diabetes Dataset"

    frame = pd.read_csv(path)
    frame.columns = [col.strip() for col in frame.columns]
    return frame, dataset_name, path


def infer_target_column(frame: pd.DataFrame) -> str:
    candidates = [
        "Diabetes_binary",
        "Diabetes_012",
        "Outcome",
        "target",
        "label",
        "diabetes",
        "diagnosis",
        "classification",
        "dataset",
        "disease",
        "result",
        "status",
        "class",
    ]
    lowered = {col.lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return frame.columns[-1]


def normalize_binary_target(target: pd.Series, target_name: str) -> pd.Series:
    values = target.copy()
    if target_name.lower() == "diabetes_012":
        return (pd.to_numeric(values, errors="coerce") > 0).astype(int)

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        numeric = numeric.astype(int)
        unique = sorted(numeric.dropna().unique().tolist())
        if len(unique) == 2:
            return numeric.map({unique[0]: 0, unique[1]: 1}).astype(int)
        return (numeric > 0).astype(int)

    text = values.astype(str).str.strip().str.lower()
    positive_tokens = {"1", "yes", "y", "true", "positive", "present", "ckd", "disease", "diabetes"}
    negative_tokens = {"0", "no", "n", "false", "negative", "absent", "notckd", "normal", "healthy"}
    mapped = text.map(lambda value: 1 if value in positive_tokens else 0 if value in negative_tokens else np.nan)
    if mapped.notna().all():
        return mapped.astype(int)

    codes, _ = pd.factorize(text)
    return (codes > 0).astype(int)


def preprocess_features(
    frame: pd.DataFrame,
    target_name: str,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    clean = frame.replace({"?": np.nan, "": np.nan, " ": np.nan}).copy()
    y = normalize_binary_target(clean[target_name], target_name)
    columns_to_drop = [target_name]
    if drop_columns:
        columns_to_drop.extend([col for col in drop_columns if col in clean.columns and col != target_name])
    X = clean.drop(columns=columns_to_drop)

    for col in X.columns:
        numeric = pd.to_numeric(X[col], errors="coerce")
        if numeric.notna().mean() >= 0.9:
            X[col] = numeric

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    if len(numeric_cols):
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    for col in categorical_cols:
        mode = X[col].mode(dropna=True)
        X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "Unknown").astype(str).str.strip()

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    prepared = pd.concat([X, y.rename(target_name)], axis=1).dropna(axis=0).reset_index(drop=True)
    return prepared.drop(columns=[target_name]), prepared[target_name].astype(int)


def prepare_diabetes_data(
    path: Path | None = None,
    synthetic_demo: bool = False,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> PreparedData:
    return prepare_healthcare_data(
        path=path,
        synthetic_demo=synthetic_demo,
        target_name=None,
        dataset_label=None,
        test_size=test_size,
        random_state=random_state,
    )


def prepare_healthcare_data(
    path: Path | None = None,
    synthetic_demo: bool = False,
    target_name: str | None = None,
    dataset_label: str | None = None,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> PreparedData:
    frame, dataset_name, source_path = load_diabetes_frame(path, synthetic_demo=synthetic_demo)
    if target_name is None:
        target_name = infer_target_column(frame)
    if target_name not in frame.columns:
        raise ValueError(f"Target column {target_name!r} was not found in the dataset.")

    outcome_columns = ["Diabetes_binary", "Diabetes_012"]
    drop_columns = [col for col in outcome_columns if col != target_name]
    X, y = preprocess_features(frame, target_name, drop_columns=drop_columns)
    feature_names = X.columns.tolist()
    clean = pd.concat([X, y.rename(target_name)], axis=1)

    stratify = y.values.astype(int) if y.nunique() == 2 and y.value_counts().min() >= 2 else None

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X.values.astype(float),
        y.values.astype(int),
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return PreparedData(
        name=dataset_label or dataset_name,
        source_path=source_path,
        frame=clean,
        feature_names=feature_names,
        target_name=target_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        scaler=scaler,
    )
