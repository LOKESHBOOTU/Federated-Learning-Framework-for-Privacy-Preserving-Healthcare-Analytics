from __future__ import annotations

from typing import Any


REFERENCE_RANGES = {
    "fasting_glucose": ("Fasting glucose", "mg/dL", 70, 99),
    "postprandial_glucose": ("Post-meal glucose", "mg/dL", 70, 140),
    "bmi": ("BMI", "", 18.5, 24.9),
    "cholesterol": ("Total cholesterol", "mg/dL", 0, 199),
    "hdl": ("HDL cholesterol", "mg/dL", 40, None),
    "ldl": ("LDL cholesterol", "mg/dL", 0, 99),
    "triglycerides": ("Triglycerides", "mg/dL", 0, 149),
    "systolic_bp": ("Systolic blood pressure", "mmHg", 90, 120),
    "diastolic_bp": ("Diastolic blood pressure", "mmHg", 60, 80),
}


def _status(value: float, low: float | None, high: float | None, report_specific: bool) -> str:
    if low is not None and value < low:
        return "Below report range" if report_specific else "Below generic range"
    if high is not None and value > high:
        return "Above report range" if report_specific else "Above generic range"
    return "Within report range" if report_specific else "Within generic range"


def analyze_features(
    values: dict[str, Any],
    report_references: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """Analyze explicit measurements without converting them into unrelated model features."""
    report_references = report_references or {}
    rows = []

    for key, value in values.items():
        if key not in REFERENCE_RANGES:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        label, unit, generic_low, generic_high = REFERENCE_RANGES[key]
        low, high = report_references.get(key, (generic_low, generic_high))
        report_specific = key in report_references

        if low is not None and high is not None:
            reference = f"{low:g}-{high:g}"
        elif low is not None:
            reference = f">= {low:g}"
        else:
            reference = f"<= {high:g}"

        rows.append(
            {
                "Parameter": label,
                "Value": numeric_value,
                "Unit": unit,
                "Reference": reference,
                "Reference source": "Uploaded report" if report_specific else "Generic fallback",
                "Status": _status(numeric_value, low, high, report_specific),
            }
        )

    return rows
