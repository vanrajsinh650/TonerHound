"""EXP-038 Fix 3: High-DPI Adaptive Binarization OCR Retry.

Addresses unresolved zero-text fields on scanned pages:
- Problem: Low-contrast scans and degraded text fail Tesseract default binarization at 200 DPI.
- Solution: Render at 300 DPI, apply adaptive Otsu / CLAHE contrast enhancement, and run Tesseract PSM 6 / 11.
- Safety:
  * Only activated on pages with < 5 native digital tokens.
  * Strictly bounded to 1 retry attempt per page.
  * If page remains empty of text, flags page as VISUAL_NON_TEXT for checkbox provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytesseract

from tonerhound.document.index import _cluster_tokens_into_lines, _parse_ocr_data
from tonerhound.models.types import DocumentPage, DocumentToken


class HighDPIOCRRetrier:
    """Retries OCR on stubborn zero-token pages at 300 DPI with adaptive binarization."""

    def __init__(self, target_dpi: int = 300) -> None:
        self.scale = target_dpi / 72.0
        self._retried_pages: set[tuple[str, int]] = set()

    def retry_page(
        self,
        pdf_path: Path | str,
        page_num: int,
    ) -> list[DocumentToken]:
        key = (str(pdf_path), page_num)
        if key in self._retried_pages:
            return []
        self._retried_pages.add(key)

        try:
            doc = pdfium.PdfDocument(str(pdf_path))
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return []

            pdf_page = doc[page_num - 1]
            bitmap = pdf_page.render(scale=self.scale)
            img_pil = bitmap.to_pil()
            img_np = np.array(img_pil)
            doc.close()

            # Convert to grayscale
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_np

            img_h, img_w = gray.shape

            # Adaptive preprocessing: CLAHE + Otsu binarization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Tesseract OCR with PSM 6 (uniform block of text) and PSM 11 (sparse text)
            data6 = pytesseract.image_to_data(binary, config="--psm 6", output_type=pytesseract.Output.DICT)
            tokens = _parse_ocr_data(data6, img_w, img_h, page_num)

            if len(tokens) < 30:
                data11 = pytesseract.image_to_data(binary, config="--psm 11", output_type=pytesseract.Output.DICT)
                tokens11 = _parse_ocr_data(data11, img_w, img_h, page_num)
                if len(tokens11) > len(tokens):
                    tokens = tokens11

            return tokens

        except Exception:
            return []
