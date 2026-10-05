"""EXP-036D: Deterministic Visual Candidate Provider.

Implements a pure classical computer vision perception engine:
- PyMuPDF page rendering (300 DPI)
- Morphological wireframe filtering (horizontal & vertical kernels)
- Multi-scale contour extraction (checkboxes, radio boxes, wide form slots, signature baselines)
- Aspect-ratio, convexity, and border-continuity geometry scoring
- Central core occupancy & Hough transform diagonal stroke detection
- Non-maximum suppression (NMS)

Contract (Section 25):
Returns:
{
    "bbox": [x1, y1, x2, y2],
    "state": "CHECKED" | "UNCHECKED" | "AMBIGUOUS",
    "object_type": "SQUARE_CHECKBOX" | "RADIO_BOX" | "WIDE_FORM_SLOT" | "SIGNATURE_REGION" | None,
    "confidence": float [0.0, 1.0]
}

NO LLM, NO VLM, NO NEURAL NETWORK, NO CLOUD API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pymupdf as fitz


@dataclass
class VisualCandidate:
    """Standardized visual candidate output conforming to Section 25 contract."""
    bbox: list[float]  # [x, y, w, h] normalized [0.0, 1.0]
    state: str         # "CHECKED", "UNCHECKED", "AMBIGUOUS"
    object_type: str   # "SQUARE_CHECKBOX", "RADIO_BOX", "WIDE_FORM_SLOT", "SIGNATURE_REGION"
    confidence: float  # Deterministic score [0.0, 1.0]
    bbox_px: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(c, 4) for c in self.bbox],
            "state": self.state,
            "object_type": self.object_type,
            "confidence": round(self.confidence, 4),
        }


def compute_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Compute IoU between two [x, y, w, h] boxes."""
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def apply_nms(candidates: list[VisualCandidate], iou_thresh: float = 0.35) -> list[VisualCandidate]:
    """Non-maximum suppression based on candidate confidence."""
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: c.confidence, reverse=True)
    keep: list[VisualCandidate] = []

    for cand in sorted_cands:
        suppress = False
        for k in keep:
            if compute_iou(cand.bbox, k.bbox) > iou_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(cand)

    return keep


class DeterministicVisualProvider:
    """Classical CV visual candidate generator for form elements and checkboxes."""

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi
        self._page_cache: dict[tuple[str, int], np.ndarray] = {}

    def render_page(self, pdf_path: Path | str, page_num: int) -> np.ndarray:
        """Render specified page to grayscale image at target DPI."""
        key = (str(pdf_path), page_num)
        if key in self._page_cache:
            return self._page_cache[key]

        doc = fitz.open(str(pdf_path))
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return np.zeros((100, 100), dtype=np.uint8)

        page = doc[page_num - 1]
        scale = self.dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width))
        doc.close()

        self._page_cache[key] = img
        return img

    def classify_state(self, crop_gray: np.ndarray, obj_type: str) -> tuple[str, float]:
        """Classify candidate state using interior core darkness and diagonal Hough strokes."""
        h, w = crop_gray.shape
        if h < 6 or w < 6:
            return "AMBIGUOUS", 0.50

        blur = cv2.GaussianBlur(crop_gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if obj_type == "SIGNATURE_REGION":
            sig_body = binary[0 : int(0.85 * h), :]
            sig_darkness = float(np.mean(sig_body > 0)) if sig_body.size > 0 else 0.0
            if sig_darkness >= 0.045:
                return "CHECKED", min(0.95, 0.70 + sig_darkness)
            elif sig_darkness < 0.020:
                return "UNCHECKED", 0.90
            else:
                return "AMBIGUOUS", 0.55

        # Checkbox / Small Form Box
        c_y0, c_y1 = int(0.30 * h), int(0.70 * h)
        c_x0, c_x1 = int(0.30 * w), int(0.70 * w)
        core = binary[c_y0:c_y1, c_x0:c_x1]
        core_dark = float(np.mean(core > 0)) if core.size > 0 else 0.0

        # Diagonal strokes
        edges = cv2.Canny(blur, 50, 150)
        inner_edges = np.zeros_like(edges)
        inner_edges[int(0.15 * h) : int(0.85 * h), int(0.15 * w) : int(0.85 * w)] = edges[
            int(0.15 * h) : int(0.85 * h), int(0.15 * w) : int(0.85 * w)
        ]
        lines = cv2.HoughLinesP(
            inner_edges,
            1,
            np.pi / 180,
            threshold=max(6, min(h, w) // 4),
            minLineLength=max(5, min(h, w) // 4),
            maxLineGap=2,
        )

        diag_count = 0
        if lines is not None:
            for line in lines:
                pts = line.ravel()
                if len(pts) >= 4:
                    x1, y1, x2, y2 = pts[:4]
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx != 0:
                        deg = abs(math.degrees(math.atan(dy / dx)))
                        if 20.0 <= deg <= 70.0:
                            diag_count += 1

        if diag_count >= 2 or core_dark >= 0.12:
            return "CHECKED", min(0.98, 0.70 + core_dark)
        elif core_dark < 0.035 and diag_count == 0:
            return "UNCHECKED", 0.95
        elif core_dark < 0.06:
            return "UNCHECKED", 0.80
        elif core_dark >= 0.08:
            return "CHECKED", 0.75
        else:
            return "AMBIGUOUS", 0.50

    def detect_page_candidates(
        self,
        pdf_path: Path | str,
        page_num: int,
        target_obj_types: set[str] | None = None,
    ) -> list[VisualCandidate]:
        """Extract all visual candidates on the page with deterministic scoring and NMS."""
        gray = self.render_page(pdf_path, page_num)
        img_h, img_w = gray.shape

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 1. Wireframe Filtering
        k_box = 15
        h_box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box, 1))
        v_box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_box))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_box_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_box_kernel)
        wireframe = cv2.bitwise_or(h_lines, v_lines)

        cnts_wire, _ = cv2.findContours(wireframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts_bin, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 2. Signature Baselines
        k_sig = 80
        h_sig_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sig, 1))
        baselines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_sig_kernel)
        cnts_sig, _ = cv2.findContours(baselines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates: list[dict[str, Any]] = []

        # Process Boxes
        for c in list(cnts_wire) + list(cnts_bin):
            x, y, w, h = cv2.boundingRect(c)
            ar = w / max(1, h)

            if 20 <= w <= 140 and 20 <= h <= 140 and 0.40 <= ar <= 2.5:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                n_vertices = len(approx)
                is_convex = cv2.isContourConvex(approx)
                area = cv2.contourArea(c)
                fullness = area / (w * h)

                if fullness < 0.35:
                    continue

                shape_score = (1.0 if n_vertices == 4 else (0.75 if 5 <= n_vertices <= 6 else 0.50)) * (
                    1.1 if is_convex else 0.8
                )
                ar_diff = abs(ar - 1.0)
                if ar_diff <= 0.20:
                    shape_score *= 1.2
                elif ar_diff <= 0.45:
                    shape_score *= 1.0
                else:
                    shape_score *= 0.85
                shape_score = min(1.0, max(0.1, shape_score))

                sub = binary[y : y + h, x : x + w]
                if sub.size > 0:
                    top_p = np.mean(sub[:3, :] > 0)
                    bot_p = np.mean(sub[-3:, :] > 0)
                    left_p = np.mean(sub[:, :3] > 0)
                    right_p = np.mean(sub[:, -3:] > 0)
                    border_score = float((top_p + bot_p + left_p + right_p) / 4.0)
                else:
                    border_score = 0.0

                c_y0, c_y1 = int(0.30 * h), int(0.70 * h)
                c_x0, c_x1 = int(0.30 * w), int(0.70 * w)
                core = sub[c_y0:c_y1, c_x0:c_x1] if sub.size > 0 else np.zeros((1, 1))
                core_dark = float(np.mean(core > 0)) if core.size > 0 else 0.0
                interior_score = 0.90 if core_dark < 0.05 or core_dark > 0.15 else 0.50

                if 0.70 <= ar <= 1.40 and w <= 85:
                    obj_type = "SQUARE_CHECKBOX"
                elif 0.40 <= ar <= 2.20 and w <= 110:
                    obj_type = "RADIO_BOX"
                else:
                    obj_type = "WIDE_FORM_SLOT"

                raw_candidates.append({
                    "bbox_px": [x, y, w, h],
                    "bbox": [x / img_w, y / img_h, w / img_w, h / img_h],
                    "object_type": obj_type,
                    "shape_score": shape_score,
                    "border_score": border_score,
                    "interior_score": interior_score,
                    "repetition_score": 0.0,
                })

        # Process Signatures
        for c in cnts_sig:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 140:
                sig_y = max(0, y - int(0.18 * w))
                sig_h = int(0.20 * w)
                raw_candidates.append({
                    "bbox_px": [x, sig_y, w, sig_h],
                    "bbox": [x / img_w, sig_y / img_h, w / img_w, sig_h / img_h],
                    "object_type": "SIGNATURE_REGION",
                    "shape_score": 0.85,
                    "border_score": 0.80,
                    "interior_score": 0.80,
                    "repetition_score": 0.0,
                })

        # Repeated-grid clustering bonus
        x_coords = [c["bbox_px"][0] for c in raw_candidates if c["object_type"] != "SIGNATURE_REGION"]
        y_coords = [c["bbox_px"][1] for c in raw_candidates if c["object_type"] != "SIGNATURE_REGION"]

        final_candidates: list[VisualCandidate] = []
        for c in raw_candidates:
            obj_type = c["object_type"]
            if target_obj_types and obj_type not in target_obj_types:
                continue

            px_x, px_y, px_w, px_h = c["bbox_px"]
            if obj_type != "SIGNATURE_REGION":
                col_mates = sum(1 for ox in x_coords if abs(ox - px_x) <= 12)
                row_mates = sum(1 for oy in y_coords if abs(oy - px_y) <= 12)
                rep_score = min(1.0, 0.20 * (col_mates - 1) + 0.20 * (row_mates - 1))
            else:
                rep_score = 0.50

            confidence = (
                0.35 * c["shape_score"]
                + 0.25 * c["border_score"]
                + 0.20 * c["interior_score"]
                + 0.20 * rep_score
            )

            # Classify candidate state
            crop = gray[px_y : px_y + px_h, px_x : px_x + px_w]
            cand_state, state_conf = self.classify_state(crop, obj_type)
            combined_conf = round(float(0.70 * confidence + 0.30 * state_conf), 4)

            final_candidates.append(
                VisualCandidate(
                    bbox=c["bbox"],
                    state=cand_state,
                    object_type=obj_type,
                    confidence=combined_conf,
                    bbox_px=c["bbox_px"],
                )
            )

        # Apply NMS
        return apply_nms(final_candidates, iou_thresh=0.35)
