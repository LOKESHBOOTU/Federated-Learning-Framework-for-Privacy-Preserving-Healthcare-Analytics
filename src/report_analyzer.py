from __future__ import annotations


REFERENCE_RANGES = {
    "glucose": ("Fasting glucose", "mg/dL", 70, 99),
    "bmi": ("BMI", "", 18.5, 24.9),
    "cholesterol": ("Total cholesterol", "mg/dL", 0, 199),
    "hdl": ("HDL cholesterol", "mg/dL", 40, None),
    "ldl": ("LDL cholesterol", "mg/dL", 0, 99),
    "triglycerides": ("Triglycerides", "mg/dL", 0, 149),
    "systolic_bp": ("Systolic blood pressure", "mmHg", 90, 120),
    "diastolic_bp": ("Diastolic blood pressure", "mmHg", 60, 80),
}


def evaluate_value(name: str, value: float) -> str:
    if name not in REFERENCE_RANGES:
        return "Extracted"

    _, _, low, high = REFERENCE_RANGES[name]
    if low is not None and value < low:
        return "Below reference"
    if high is not None and value > high:
        return "Above reference"
    return "Within reference"


def analyze_features(values: dict[str, float]) -> list[dict]:
    rows = []
    for key, value in values.items():
        if key not in REFERENCE_RANGES:
            continue
        label, unit, low, high = REFERENCE_RANGES[key]
        reference = (
            f"{low:g}-{high:g}" if low is not None and high is not None
            else f">= {low:g}" if low is not None
            else f"<= {high:g}"
        )
        rows.append({
            "Parameter": label,
            "Value": value,
            "Unit": unit,
            "Reference (generic)": reference,
            "Status": evaluate_value(key, value),
        })
    return rows
