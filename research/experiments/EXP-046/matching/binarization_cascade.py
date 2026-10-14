"""Stage 23: Adaptive Binarization Cascade.

For degraded or low-contrast document regions:
- Niblack binarization
- Sauvola binarization
- Otsu global threshold
- Adaptive Gaussian threshold
- Votes on OCR output across all four methods.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence
import cv2
import numpy as np
import pytesseract
from skimage.filters import threshold_niblack, threshold_sauvola, threshold_otsu


def binarize_niblack(gray_img: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    thresh = threshold_niblack(gray_img, window_size=window_size, k=k)
    return (gray_img > thresh).astype(np.uint8) * 255


def binarize_sauvola(gray_img: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    thresh = threshold_sauvola(gray_img, window_size=window_size, k=k)
    return (gray_img > thresh).astype(np.uint8) * 255


def binarize_otsu(gray_img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive_gaussian(gray_img: np.ndarray, block_size: int = 15, c: float = 2.0) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def ocr_binarization_cascade(gray_crop: np.ndarray) -> tuple[str, str]:
    """Applies the 4 binarizations, runs OCR, and votes on majority text.
    
    Returns:
        (voted_text, winning_method)
    """
    if gray_crop is None or gray_crop.size == 0:
        return "", "none"

    methods = {
        "sauvola": lambda img: binarize_sauvola(img),
        "niblack": lambda img: binarize_niblack(img),
        "otsu": lambda img: binarize_otsu(img),
        "adaptive_gaussian": lambda img: binarize_adaptive_gaussian(img),
    }

    results: list[tuple[str, str]] = []

    for name, binarizer in methods.items():
        try:
            bin_img = binarizer(gray_crop)
            txt = pytesseract.image_to_string(bin_img, config="--psm 6").strip()
            if txt:
                results.append((txt, name))
        except Exception:
            continue

    if not results:
        return "", "none"

    texts = [r[0] for r in results]
    counts = Counter(texts)
    winner, _ = counts.most_common(1)[0]

    # Find method that produced winner
    for txt, name in results:
        if txt == winner:
            return winner, name

    return results[0][0], results[0][1]
