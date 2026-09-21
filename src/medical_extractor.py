from __future__ import annotations

import re
from typing import Any


PATTERNS = {
    "glucose": [
        r"(?:fasting\s+)?(?:blood\s+)?glucose\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?",
        r"fbs\s*[:=\-]?\s*(\d+(?:\.\d+)?)",
    ],
    "bmi": [
        r"(?:body\s+mass\s+index|bmi)\s*[:=\-]?\s*(\d+(?:\.\d+)?)",
    ],
    "cholesterol": [
        r"(?:total\s+)?cholesterol\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:mg/dl)?",
    ],
    "hdl": [r"(?:hdl(?:\s+cholesterol)?)\s*[:=\-]?\s*(\d+(?:\.\d+)?)"],
    "ldl": [r"(?:ldl(?:\s+cholesterol)?)\s*[:=\-]?\s*(\d+(?:\.\d+)?)"],
    "triglycerides": [r"(?:triglycerides|tg)\s*[:=\-]?\s*(\d+(?:\.\d+)?)"],
    "systolic_bp": [
        r"(?:blood\s+pressure|bp)\s*[:=\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
    ],
    "age": [r"\bage\s*[:=\-]?\s*(\d{1,3})\s*(?:years?|yrs?)?\b"],
}


def _first_number(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_medical_features(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"[ \t]+", " ", text)
    values: dict[str, Any] = {}

    for name, patterns in PATTERNS.items():
        if name == "systolic_bp":
            match = re.search(patterns[0], cleaned, flags=re.IGNORECASE)
            if match:
                values["systolic_bp"] = float(match.group(1))
                values["diastolic_bp"] = float(match.group(2))
        else:
            value = _first_number(cleaned, patterns)
            if value is not None:
                values[name] = value

    # Sex is only extracted when explicitly written; it is never inferred.
    sex = re.search(r"\b(sex|gender)\s*[:=\-]?\s*(male|female)\b", cleaned, re.I)
    if sex:
        values["sex"] = 1 if sex.group(2).lower() == "male" else 0

    return values


def extract_prescription_lines(text: str) -> list[dict[str, str]]:
    """Lightweight structure extraction; handwritten prescriptions require verification."""
    rows = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if re.search(r"\b(tab(?:let)?|cap(?:sule)?|syrup|inj(?:ection)?|mg|ml)\b", line, re.I):
            rows.append({
                "text": line,
                "medicine": line,
            })
    return rows
