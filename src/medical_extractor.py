from __future__ import annotations

import re
from typing import Any


_NUMBER = r"(\d+(?:\.\d+)?)"


def _clean(text: str) -> str:
    # Preserve newlines because report tables are easier to parse line-by-line.
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _number_after_label(line: str, label_pattern: str) -> float | None:
    match = re.search(label_pattern + r"\s*[:=\-]?\s*" + _NUMBER, line, re.I)
    return float(match.group(1)) if match else None


def _range_after_number(line: str, value: float | None) -> tuple[float, float] | None:
    if value is None:
        return None
    # Prefer a reference range appearing after the result value.
    match = re.search(
        _NUMBER + r"\s*(?:mg/dl|mg/dL|mmhg|%)?\s+" +
        r"(" + _NUMBER + r")\s*(?:-|–|—|to)\s*(" + _NUMBER + r")",
        line,
        re.I,
    )
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        return (low, high)
    return None


def _search_lines(lines: list[str], patterns: list[str]) -> tuple[float | None, tuple[float, float] | None]:
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.I)
            if match:
                value = float(match.group(1))
                low_high = None
                if match.lastindex and match.lastindex >= 3:
                    low_high = (float(match.group(2)), float(match.group(3)))
                return value, low_high
    return None, None


def extract_medical_features(text: str) -> dict[str, Any]:
    """Extract explicit report values only; never invent missing patient features."""
    cleaned = _clean(text)
    lines = cleaned.splitlines()
    values: dict[str, Any] = {}

    # Patient demographics.
    age = None
    for line in lines:
        age = _number_after_label(line, r"\bage\b")
        if age is not None and 0 < age < 130:
            values["age"] = int(age)
            break

    sex = re.search(r"\b(?:sex|gender)\s*[:=\-]?\s*(male|female)\b", cleaned, re.I)
    if sex:
        values["sex"] = 1 if sex.group(1).lower() == "male" else 0

    # Diabetes/lab report patterns. These are kept separate so fasting and PP
    # results cannot be confused with one another.
    fasting_patterns = [
        r"glucose\s+fasting[^\d]*" + _NUMBER + r"\s*(?:mg/dl)?\s+" + _NUMBER + r"\s*(?:-|–|—|to)\s*" + _NUMBER,
        r"fasting\s+(?:plasma\s+)?glucose[^\d]*" + _NUMBER + r"\s*(?:mg/dl)?",
        r"fbs\s*[:=\-]?\s*" + _NUMBER,
    ]
    pp_patterns = [
        r"glucose\s*\(\s*pp\s*\)[^\d]*" + _NUMBER + r"\s*(?:mg/dl)?\s+" + _NUMBER + r"\s*(?:-|–|—|to)\s*" + _NUMBER,
        r"glucose\s+pp[^\d]*" + _NUMBER + r"\s*(?:mg/dl)?\s+" + _NUMBER + r"\s*(?:-|–|—|to)\s*" + _NUMBER,
        r"post[- ]?meal\s+(?:plasma\s+)?glucose[^\d]*" + _NUMBER + r"\s*(?:mg/dl)?",
    ]

    fasting, fasting_ref = _search_lines(lines, fasting_patterns)
    pp, pp_ref = _search_lines(lines, pp_patterns)

    if fasting is not None:
        values["fasting_glucose"] = fasting
    if pp is not None:
        values["postprandial_glucose"] = pp

    # Other common measurements.
    simple_patterns = {
        "bmi": [r"(?:body\s+mass\s+index|bmi)\s*[:=\-]?\s*" + _NUMBER],
        "cholesterol": [r"(?:total\s+)?cholesterol\s*[:=\-]?\s*" + _NUMBER],
        "hdl": [r"hdl(?:\s+cholesterol)?\s*[:=\-]?\s*" + _NUMBER],
        "ldl": [r"ldl(?:\s+cholesterol)?\s*[:=\-]?\s*" + _NUMBER],
        "triglycerides": [r"(?:triglycerides|tg)\s*[:=\-]?\s*" + _NUMBER],
    }
    for name, patterns in simple_patterns.items():
        for line in lines:
            value = _number_after_label(line, patterns[0].split(r"\s*[:=\-]?")[0])
            if value is not None:
                values[name] = value
                break

    bp = re.search(r"(?:blood\s+pressure|\bbp\b)\s*[:=\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})", cleaned, re.I)
    if bp:
        values["systolic_bp"] = float(bp.group(1))
        values["diastolic_bp"] = float(bp.group(2))

    # Store report-specific reference intervals separately.
    references = {}
    if fasting_ref:
        references["fasting_glucose"] = fasting_ref
    if pp_ref:
        references["postprandial_glucose"] = pp_ref
    values["_report_references"] = references

    return values


def extract_prescription_lines(text: str) -> list[dict[str, str]]:
    """Lightweight structure extraction; handwritten prescriptions require verification."""
    rows = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if re.search(r"\b(tab(?:let)?|cap(?:sule)?|syrup|inj(?:ection)?|mg|ml)\b", line, re.I):
            rows.append({"text": line, "medicine": line})
    return rows
