from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        DOCS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path(filename: str) -> Path:
    candidates = (
        RAW_DATA_DIR / filename,
        PROJECT_ROOT / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked: {searched}")

