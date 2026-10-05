"""EXP-043 Phase B: Niblack + Multi-Pass OCR Voting.

Target: NO_TEXT_AT_GOLD_REGION (14,952 fields).
Applies adaptive local thresholding (Niblack, Sauvola, CLAHE) and multi-PSM OCR execution,
aggregating character-level majority consensus across passes to rescue missing text on degraded pages.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import pytesseract


def niblack_binarize(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = -0.2,
) -> np.ndarray:
    """Niblack local thresholding: T = mean + k * std."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (window_size, window_size))
    sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
    threshold = mean + k * std
    return (gray_f > threshold).astype(np.uint8) * 255


def sauvola_binarize(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    R: float = 128,
) -> np.ndarray:
    """Sauvola local thresholding: T = mean * (1 + k * (std/R - 1))."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (window_size, window_size))
    sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))
    threshold = mean * (1.0 + k * (std / R - 1.0))
    return (gray_f > threshold).astype(np.uint8) * 255


def preprocess_variants(page_image: np.ndarray) -> dict[str, np.ndarray]:
    """Generate multiple binarization variants."""
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY) if len(page_image.shape) == 3 else page_image

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return {
        "niblack": niblack_binarize(enhanced),
        "sauvola": sauvola_binarize(enhanced),
        "otsu": cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        "adaptive": cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
        ),
    }


def multi_pass_ocr(page_image: np.ndarray, psms: list[int] | None = None) -> list[dict[str, Any]]:
    """Run OCR with multiple preprocessing + PSM combinations."""
    if page_image is None or page_image.size == 0:
        return []

    selected_psms = psms or [6, 11]
    variants = preprocess_variants(page_image)
    results = []

    for mode_name, preprocessed in variants.items():
        for psm in selected_psms:
            try:
                data = pytesseract.image_to_data(
                    preprocessed,
                    config=f"--psm {psm} --oem 3",
                    output_type=pytesseract.Output.DICT,
                )
                results.append({"mode": mode_name, "psm": psm, "data": data})
            except Exception:
                continue

    return results


def vote_tokens(ocr_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Character-level majority voting across OCR passes."""
    position_groups = defaultdict(list)

    for result in ocr_results:
        data = result.get("data", {})
        texts = data.get("text", [])
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])
        confs = data.get("conf", [])

        for i, text in enumerate(texts):
            clean = str(text).strip()
            if not clean:
                continue
            x = lefts[i]
            y = tops[i]
            w = widths[i]
            h = heights[i]
            conf = int(confs[i]) if str(confs[i]).isdigit() or isinstance(confs[i], (int, float)) else 50
            key = (round(x / 15) * 15, round(y / 10) * 10)
            position_groups[key].append({
                "text": clean,
                "bbox": (x, y, w, h),
                "confidence": conf,
            })

    voted = []
    for key, candidates in position_groups.items():
        text_counts: dict[str, int] = defaultdict(int)
        for c in candidates:
            text_counts[c["text"]] += 1

        best_text = max(
            text_counts.keys(),
            key=lambda t: (text_counts[t], max(c["confidence"] for c in candidates if c["text"] == t)),
        )
        best_candidate = max(
            [c for c in candidates if c["text"] == best_text],
            key=lambda c: c["confidence"],
        )
        voted.append(best_candidate)

    return voted
