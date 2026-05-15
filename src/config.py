from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLIENT_DATA_DIR = DATA_DIR / "clients"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

DATASET_CANDIDATES = [
    (RAW_DATA_DIR / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv", "CDC Diabetes Health Indicators 50/50 Balanced"),
    (RAW_DATA_DIR / "cdc_diabetes_health_indicators.csv", "CDC Diabetes Health Indicators"),
    (RAW_DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv", "CDC Diabetes Health Indicators"),
    (RAW_DATA_DIR / "diabetes_012_health_indicators_BRFSS2015.csv", "CDC Diabetes Health Indicators"),
    (RAW_DATA_DIR / "diabetes.csv", "Original Diabetes Dataset"),
    (RAW_DATA_DIR / "pima_indians_diabetes.csv", "Original PIMA Diabetes Dataset"),
    (RAW_DATA_DIR / "pima-indians-diabetes.csv", "Original PIMA Diabetes Dataset"),
    (RAW_DATA_DIR / "heart.csv", "Original Heart Disease Dataset"),
    (RAW_DATA_DIR / "kidney_disease.csv", "Original Kidney Disease Dataset"),
    (RAW_DATA_DIR / "liver.csv", "Original Liver Disease Dataset"),
]

HEART_DISEASE_DATASET_CANDIDATES = [
    (ROOT / "heart_cleveland_upload.csv", "Heart Disease Cleveland UCI"),
    (RAW_DATA_DIR / "heart_cleveland_upload.csv", "Heart Disease Cleveland UCI"),
    (RAW_DATA_DIR / "heart.csv", "Heart Disease Cleveland UCI"),
    (ROOT / "heart.csv", "Heart Disease Dataset"),
]

DEFAULT_RANDOM_STATE = 42
