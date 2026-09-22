from __future__ import annotations

import pandas as pd
import streamlit as st

from src.document_processor import process_document
from src.medical_extractor import extract_medical_features, extract_prescription_lines
from src.report_analyzer import analyze_features


st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="",
    layout="wide",
)

st.title("Medical Report & Prescription Analyzer")
st.caption("Upload a report or prescription and extract information locally before reviewing it.")

st.warning(
    "This is a research/demo decision-support feature, not a medical diagnosis. "
    "OCR can misread values. Verify every extracted value against the original document. "
    "The app never fills missing model inputs with dataset medians or zeros."
)

file_type = st.radio(
    "Document type",
    ["Medical test report", "Prescription"],
    horizontal=True,
)

uploaded = st.file_uploader(
    "Upload PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
)

if uploaded:
    try:
        file_bytes = uploaded.getvalue()
        text, method = process_document(file_bytes, uploaded.name)

        st.success(f"Processed locally using: {method}")

        with st.expander("Extracted report text", expanded=False):
            st.text_area("OCR / PDF text", text, height=320, label_visibility="collapsed")

        if not text.strip():
            st.error("No readable text was found. Try a clearer scan or image.")
            st.stop()

        if file_type == "Prescription":
            rows = extract_prescription_lines(text)
            st.subheader("Extracted prescription entries")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.info(
                    "Medicine names, strengths, frequency, and duration from OCR must be manually "
                    "verified, especially for handwritten prescriptions."
                )
            else:
                st.info("No prescription-like lines were detected. Review the extracted text above.")

        else:
            raw_values = extract_medical_features(text)
            report_references = raw_values.pop("_report_references", {})

            st.subheader("1. Verify extracted patient information")

            patient_rows = []
            if "age" in raw_values:
                patient_rows.append({"Field": "Age", "Extracted value": raw_values["age"], "Unit": "years"})
            if "sex" in raw_values:
                patient_rows.append({
                    "Field": "Gender",
                    "Extracted value": "Male" if raw_values["sex"] == 1 else "Female",
                    "Unit": "",
                })

            if patient_rows:
                patient_df = pd.DataFrame(patient_rows)
                edited_patient = st.data_editor(
                    patient_df,
                    hide_index=True,
                    use_container_width=True,
                    disabled=["Field", "Unit"],
                    key="report_patient_verification",
                )
                for _, row in edited_patient.iterrows():
                    if row["Field"] == "Age":
                        try:
                            raw_values["age"] = int(float(row["Extracted value"]))
                        except (TypeError, ValueError):
                            raw_values.pop("age", None)
                    elif row["Field"] == "Gender":
                        gender = str(row["Extracted value"]).strip().lower()
                        if gender in {"male", "female"}:
                            raw_values["sex"] = 1 if gender == "male" else 0
            else:
                st.info("No explicit age or gender was detected.")

            st.subheader("2. Verify extracted laboratory results")

            lab_keys = [
                "fasting_glucose",
                "postprandial_glucose",
                "bmi",
                "cholesterol",
                "hdl",
                "ldl",
                "triglycerides",
                "systolic_bp",
                "diastolic_bp",
            ]
            labels = {
                "fasting_glucose": "Fasting glucose",
                "postprandial_glucose": "Post-meal (PP) glucose",
                "bmi": "BMI",
                "cholesterol": "Total cholesterol",
                "hdl": "HDL cholesterol",
                "ldl": "LDL cholesterol",
                "triglycerides": "Triglycerides",
                "systolic_bp": "Systolic blood pressure",
                "diastolic_bp": "Diastolic blood pressure",
            }
            units = {
                "fasting_glucose": "mg/dL",
                "postprandial_glucose": "mg/dL",
                "bmi": "",
                "cholesterol": "mg/dL",
                "hdl": "mg/dL",
                "ldl": "mg/dL",
                "triglycerides": "mg/dL",
                "systolic_bp": "mmHg",
                "diastolic_bp": "mmHg",
            }

            lab_rows = [
                {
                    "Parameter": labels[key],
                    "Value": float(raw_values[key]),
                    "Unit": units[key],
                    "Detected": "Yes",
                }
                for key in lab_keys
                if key in raw_values
            ]

            if lab_rows:
                lab_df = pd.DataFrame(lab_rows)
                edited_lab = st.data_editor(
                    lab_df,
                    hide_index=True,
                    use_container_width=True,
                    disabled=["Parameter", "Unit", "Detected"],
                    key="report_lab_verification",
                )
                for _, row in edited_lab.iterrows():
                    parameter = str(row["Parameter"])
                    key = next((k for k, v in labels.items() if v == parameter), None)
                    if key:
                        try:
                            raw_values[key] = float(row["Value"])
                        except (TypeError, ValueError):
                            raw_values.pop(key, None)

                st.caption("Edit a value if OCR read it incorrectly, then review the analysis below.")
            else:
                st.info("No supported laboratory measurements were detected.")

            st.subheader("3. Report-based analysis")
            analysis_rows = analyze_features(raw_values, report_references)

            if analysis_rows:
                st.dataframe(
                    pd.DataFrame(analysis_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                for row in analysis_rows:
                    if row["Status"] == "Below report range":
                        st.info(
                            f"{row['Parameter']}: {row['Value']} {row['Unit']} is below the "
                            f"reference interval shown on the report ({row['Reference']})."
                        )
                    elif row["Status"] == "Above report range":
                        st.warning(
                            f"{row['Parameter']}: {row['Value']} {row['Unit']} is above the "
                            f"reference interval shown on the report ({row['Reference']})."
                        )
            else:
                st.info("No laboratory result with a usable reference interval was detected.")

            st.subheader("4. Federated-model compatibility")

            diabetes_required = [
                "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
                "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
            ]
            heart_required = [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal",
            ]

            diabetes_direct = []
            if "age" in raw_values:
                diabetes_direct.append("Age")
            if "bmi" in raw_values:
                diabetes_direct.append("BMI")
            if "sex" in raw_values:
                diabetes_direct.append("Sex")

            heart_direct = []
            if "age" in raw_values:
                heart_direct.append("age")
            if "sex" in raw_values:
                heart_direct.append("sex")
            if "systolic_bp" in raw_values:
                heart_direct.append("trestbps")
            if "cholesterol" in raw_values:
                heart_direct.append("chol")

            st.write("**Diabetes model — directly supported by report:**", diabetes_direct or "None")
            st.write("**Heart model — directly supported by report:**", heart_direct or "None")

            st.warning(
                "No ML prediction is produced from this report alone when required model features "
                "are missing. Missing fields are NOT replaced with dataset medians, zeros, or "
                "values inferred from unrelated laboratory tests."
            )

            with st.expander("Why no automatic diabetes prediction?"):
                st.write(
                    "The current CDC diabetes model expects survey-derived variables such as BMI, "
                    "blood-pressure history, cholesterol history, lifestyle, health-status days, "
                    "and demographic categories. A glucose result alone cannot legitimately supply "
                    "those missing variables."
                )

        st.divider()
        st.caption(
            "Privacy note: this page processes the uploaded file in memory and does not add the "
            "patient document to the federated training dataset."
        )

    except Exception as exc:
        st.error(f"Could not process this document: {exc}")
        st.info(
            "Tesseract OCR is required for image/scanned-PDF input. The app also checks the common "
            "Windows installation path C:\\Program Files\\Tesseract-OCR\\tesseract.exe."
        )
else:
    st.info("Choose a document type and upload a PDF or image to begin.")
