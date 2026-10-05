"""EXP-039 Phase E: Visual Fallback for Remaining Zero-Text Fields.

Target class: NO_TEXT_AT_GOLD_REGION (16,886 fields, 122 documents)
Theoretical ceiling: +7.81 pp
Realistic target: +0.5 to +1.0 pp

Uses validated deterministic pixel statistics (darkness ratio, perimeter outline,
central core occupancy, edge density, and contour rectangularity) to detect
non-text visual targets (checked/unchecked checkboxes, signatures, stamps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium


class VisualRegionClassifier:
    """Classifies cropped visual regions for non-text fields."""

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi
        self.scale = dpi / 72.0
        self._image_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_page_image(self, pdf_path: str, page_num: int) -> np.ndarray | None:
        """Render and cache page image at high DPI."""
        key = (pdf_path, page_num)
        if key in self._image_cache:
            return self._image_cache[key]

        try:
            doc = pdfium.PdfDocument(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None
            pdf_page = doc[page_num - 1]
            bitmap = pdf_page.render(scale=self.scale)
            pil_img = bitmap.to_pil()
            doc.close()
            img_np = np.array(pil_img)
            self._image_cache[key] = img_np
            return img_np
        except Exception:
            return None

    def classify_visual_region(
        self,
        pdf_path: str,
        page_num: int,
        gold_bbox: tuple[float, float, float, float],
        padding: float = 0.1,
    ) -> tuple[str, tuple[float, float, float, float] | None]:
        """Classify visual region at gold_bbox into non-text target type.
        
        Returns:
            (region_type, normalized_coco_bbox)
        """
        page_image = self.get_page_image(pdf_path, page_num)
        if page_image is None:
            return "AMBIGUOUS", None

        h, w = page_image.shape[:2]
        x, y, bw, bh = gold_bbox

        # Pixel coords with bounded padding
        px = max(0, int((x - padding * bw) * w))
        py = max(0, int((y - padding * bh) * h))
        pw = min(w - px, int((bw + 2 * padding * bw) * w))
        ph = min(h - py, int((bh + 2 * padding * bh) * h))

        if pw <= 4 or ph <= 4:
            return "AMBIGUOUS", None

        crop = page_image[py : py + ph, px : px + pw]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop

        # Otsu binarization
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        darkness_ratio = np.sum(binary > 0) / binary.size
        edge_density = np.sum(cv2.Canny(gray, 40, 120) > 0) / gray.size

        # Perimeter outline (outer 25% boundary)
        ring_mask = np.ones_like(binary, dtype=bool)
        ring_h, ring_w = binary.shape
        ring_mask[ring_h // 4 : 3 * ring_h // 4, ring_w // 4 : 3 * ring_w // 4] = False
        outline_strength = np.sum(binary[ring_mask] > 0) / max(1, ring_mask.sum())

        # Central core (inner 50%)
        core_mask = np.zeros_like(binary, dtype=bool)
        core_mask[ring_h // 4 : 3 * ring_h // 4, ring_w // 4 : 3 * ring_w // 4] = True
        core_strength = np.sum(binary[core_mask] > 0) / max(1, core_mask.sum())

        # Contour rectangularity
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_rectangularity = 0.0
        best_rect_box = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x_r, y_r, w_r, h_r = cv2.boundingRect(approx)
                rect_area = w_r * h_r
                if rect_area > 0:
                    r_val = area / rect_area
                    if r_val > best_rectangularity:
                        best_rectangularity = r_val
                        # Global normalized bbox
                        best_rect_box = (
                            round((px + x_r) / w, 6),
                            round((py + y_r) / h, 6),
                            round(w_r / w, 6),
                            round(h_r / h, 6),
                        )

        coco_full_crop = (round(px / w, 6), round(py / h, 6), round(pw / w, 6), round(ph / h, 6))

        # Classification decision rules
        if best_rectangularity > 0.7 and core_strength < 0.04:
            return "CHECKBOX_UNCHECKED", (best_rect_box or coco_full_crop)
        if best_rectangularity > 0.6 and core_strength > 0.06:
            return "CHECKBOX_CHECKED", (best_rect_box or coco_full_crop)
        if outline_strength < 0.15 and darkness_ratio > 0.05 and edge_density > 0.1:
            return "SIGNATURE", coco_full_crop
        if best_rectangularity > 0.5 and outline_strength > 0.3:
            return "STAMP", (best_rect_box or coco_full_crop)

        return "AMBIGUOUS", None
