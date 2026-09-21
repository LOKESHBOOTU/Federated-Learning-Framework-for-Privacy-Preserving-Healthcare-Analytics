from __future__ import annotations

from io import BytesIO

import fitz
from PIL import Image
import pytesseract


def _ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract embedded PDF text. OCR is handled separately for scanned pages."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text("text") for page in doc).strip()
    finally:
        doc.close()


def extract_text_from_scanned_pdf(file_bytes: bytes, dpi: int = 180) -> str:
    """Render PDF pages and run local Tesseract OCR."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = []
    try:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            texts.append(_ocr_image(image))
    finally:
        doc.close()
    return "\n".join(texts).strip()


def extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    return _ocr_image(image).strip()


def process_document(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Return (text, method). Files are processed in memory and not persisted."""
    suffix = filename.lower().rsplit(".", 1)[-1]

    if suffix == "pdf":
        text = extract_text_from_pdf(file_bytes)
        if len(" ".join(text.split())) >= 30:
            return text, "PDF text extraction"
        return extract_text_from_scanned_pdf(file_bytes), "Local Tesseract OCR"

    if suffix in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
        return extract_text_from_image(file_bytes), "Local Tesseract OCR"

    raise ValueError("Supported files: PDF, PNG, JPG, JPEG, WEBP, BMP, or TIFF.")
