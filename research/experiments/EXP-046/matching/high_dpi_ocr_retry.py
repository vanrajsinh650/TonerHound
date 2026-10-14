"""Stage 22: High-DPI OCR Retry.

For regions where text extraction yielded zero text:
- Crop region from PDF page rendered at 300 DPI
- Apply CLAHE contrast enhancement
- Run Tesseract with PSM 11 (sparse text)
- Fallback to PSM 6 (uniform block)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import fitz  # PyMuPDF
import numpy as np
import cv2
import pytesseract
from PIL import Image

from tonerhound.geometry.coordinates import BBox


def high_dpi_ocr_crop(
    pdf_path: str | Path,
    page_num: int,  # 1-indexed
    bbox: BBox | tuple[float, float, float, float] | list[float],
    target_dpi: int = 300,
) -> tuple[str, float]:
    """Renders the given bbox at 300 DPI, applies CLAHE, and runs Tesseract OCR.
    
    Returns:
        (recognized_text, confidence)
    """
    if isinstance(bbox, (list, tuple)):
        b = BBox(bbox[0], bbox[1], bbox[2], bbox[3])
    else:
        b = bbox

    if b.width <= 0 or b.height <= 0:
        return "", 0.0

    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return "", 0.0

        page = doc[page_num - 1]
        zoom = target_dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # Expand bbox slightly for context
        pad_x = max(2.0, b.width * 0.05)
        pad_y = max(2.0, b.height * 0.05)
        clip_rect = fitz.Rect(
            max(0.0, b.x0 - pad_x),
            max(0.0, b.y0 - pad_y),
            min(page.rect.width, b.x1 + pad_x),
            min(page.rect.height, b.y1 + pad_y),
        )

        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        doc.close()

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Try PSM 11 (Sparse text)
        txt = pytesseract.image_to_string(enhanced, config="--psm 11").strip()

        # Fallback to PSM 6 (Uniform text block) if empty
        if not txt:
            txt = pytesseract.image_to_string(enhanced, config="--psm 6").strip()

        conf = 0.85 if txt else 0.0
        return txt, conf

    except Exception:
        return "", 0.0
