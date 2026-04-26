from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.data.diabetes import load_diabetes_split
from src.data.heart import load_heart_split
from src.data.kidney import load_kidney_split
from src.data.liver import load_liver_split
from src.data.tabular import TabularSplit


DatasetLoader = Callable[[Path | None, float, int], TabularSplit]


DATASET_LOADERS: dict[str, DatasetLoader] = {
    "heart": load_heart_split,
    "diabetes": load_diabetes_split,
    "liver": load_liver_split,
    "kidney": load_kidney_split,
}
SUPPORTED_DATASETS = set(DATASET_LOADERS.keys())


def load_dataset_split(
    dataset_name: str,
    dataset_path: Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> TabularSplit:
    try:
        loader = DATASET_LOADERS[dataset_name]
    except KeyError as error:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise ValueError(
            f"Dataset '{dataset_name}' is not implemented. Supported datasets: {supported}"
        ) from error
    return loader(dataset_path=dataset_path, test_size=test_size, seed=seed)

