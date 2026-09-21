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
    "Use the laboratory's own reference ranges when they are shown."
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
        with st.expander("Extracted text", expanded=False):
            st.text_area("OCR / PDF text", text, height=280, label_visibility="collapsed")

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
            values = extract_medical_features(text)
            st.subheader("Extracted medical parameters")

            if values:
                st.dataframe(
                    pd.DataFrame(
                        [{"Parameter": k, "Extracted value": v} for k, v in values.items()]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                analysis_rows = analyze_features(values)
                if analysis_rows:
                    st.subheader("Reference-range review")
                    st.dataframe(
                        pd.DataFrame(analysis_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.subheader("Model compatibility")
                st.info(
                    "The current federated models use dataset-specific features. "
                    "Laboratory values such as glucose or cholesterol are not automatically "
                    "converted into survey/clinical variables such as HighBP or HighChol."
                )

                diabetes_features = {
                    "Age": "age",
                    "BMI": "bmi",
                }
                heart_features = {
                    "age": "age",
                    "trestbps": "systolic_bp",
                    "chol": "cholesterol",
                    "sex": "sex",
                }

                def present(feature_map):
                    return [
                        model_feature
                        for model_feature, extracted_name in feature_map.items()
                        if extracted_name in values
                    ]

                st.write("Potentially usable for Diabetes model:", present(diabetes_features) or "None")
                st.write("Potentially usable for Heart Disease model:", present(heart_features) or "None")
                st.caption(
                    "Missing model inputs remain manual inputs. No feature is invented from an "
                    "unrelated laboratory measurement."
                )
            else:
                st.info("No supported medical measurements were detected. Review the extracted text and try a clearer file.")

        st.divider()
        st.caption(
            "Privacy note: this page processes the uploaded file in memory and does not add the "
            "patient document to the federated training dataset."
        )

    except Exception as exc:
        st.error(f"Could not process this document: {exc}")
        st.info(
            "For image/PDF OCR on Windows, install Tesseract OCR separately and ensure "
            "tesseract.exe is available on PATH."
        )
else:
    st.info("Choose a document type and upload a PDF or image to begin.")
