from __future__ import annotations

import os
from io import BytesIO

import fitz
from PIL import Image, ImageEnhance, ImageOps
import pytesseract


_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _configure_tesseract() -> None:
    """Use the normal PATH first, then common Windows installation paths."""
    configured = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    if configured != "tesseract" and os.path.isfile(configured):
        return

    for candidate in _TESSERACT_CANDIDATES:
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


def _prepare_for_ocr(image: Image.Image) -> Image.Image:
    """Improve contrast and resolution for lab-report OCR."""
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    return image


def _ocr_image(image: Image.Image) -> str:
    _configure_tesseract()
    prepared = _prepare_for_ocr(image)

    # PSM 6 works well for structured reports; PSM 11 helps sparse layouts.
    candidates = [
        pytesseract.image_to_string(prepared, config="--psm 6"),
        pytesseract.image_to_string(prepared, config="--psm 11"),
    ]
    return max(candidates, key=lambda value: len(" ".join(value.split()))).strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract embedded PDF text. OCR is handled separately for scanned pages."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text("text") for page in doc).strip()
    finally:
        doc.close()


def extract_text_from_scanned_pdf(file_bytes: bytes, dpi: int = 240) -> str:
    """Render PDF pages at high resolution and run local Tesseract OCR."""
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
    return _ocr_image(image)


def process_document(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Return (text, method). Files are processed in memory and not persisted."""
    suffix = filename.lower().rsplit(".", 1)[-1]

    if suffix == "pdf":
        text = extract_text_from_pdf(file_bytes)
        compact = " ".join(text.split())
        if len(compact) >= 30:
            return text, "PDF text extraction"
        return extract_text_from_scanned_pdf(file_bytes), "Local Tesseract OCR"

    if suffix in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
        return extract_text_from_image(file_bytes), "Local Tesseract OCR"

    raise ValueError("Supported files: PDF, PNG, JPG, JPEG, WEBP, BMP, or TIFF.")
