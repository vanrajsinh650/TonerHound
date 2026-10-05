"""EXP-036C Phase 2: Checkbox Geometry, State Detection & Signature Regions.

Implements classical deterministic CV algorithms:
- Method A: Contour polygon approximation (cv2.approxPolyDP, convexity, rectangularity)
- Method B: Morphological box wireframe extraction (horizontal + vertical line kernels)
- Method C: Hough line segment detection for box edges and diagonal X/checkmark strokes
- Method D: Signature region detection (aspect ratio, baseline line detection, stroke spread)
- State classification: Checked vs Unchecked vs Ambiguous using multi-feature voting
Outputs:
- checkbox_geometry_results.json
- checkbox_state_results.json
- signature_region_results.json
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz

repo_root = Path(__file__).resolve().parent.parent.parent.parent


def iou_xywh(b1: list[float] | tuple[float, ...], b2: list[float] | tuple[float, ...]) -> float:
    """Compute Intersection-over-Union between two [x, y, w, h] boxes."""
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def detect_box_geometry(crop_gray: np.ndarray) -> dict[str, Any]:
    """Detect rectangular box geometry using contours and morphological wireframe extraction."""
    h, w = crop_gray.shape
    if h < 8 or w < 8:
        return {"detected": False, "method": "none", "iou": 0.0}

    blur = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Morphological horizontal and vertical line extraction
    k_w = max(3, w // 4)
    k_h = max(3, h // 4)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    table_wireframe = cv2.add(h_lines, v_lines)

    # 2. Contour extraction on wireframe
    contours, _ = cv2.findContours(table_wireframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_cand = None
    best_cand_score = 0.0

    # Also search direct binary contours if wireframe has few contours
    all_contours = list(contours)
    direct_cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(direct_cnts)

    for cnt in all_contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, bw, bh = cv2.boundingRect(cnt)
        ar = bw / max(1, bh)
        area = bw * bh
        crop_area = w * h

        # A checkbox should occupy 20% to 95% of the padded crop area
        if 0.15 * crop_area <= area <= 0.98 * crop_area:
            if 0.50 <= ar <= 2.0:
                is_convex = cv2.isContourConvex(approx)
                rect_score = (1.0 if len(approx) == 4 else 0.7) * (1.2 if is_convex else 0.8)
                size_score = 1.0 - abs(area - 0.65 * crop_area) / crop_area
                total_score = rect_score * size_score

                if total_score > best_cand_score:
                    best_cand_score = total_score
                    best_cand = (x, y, bw, bh, len(approx), is_convex)

    if best_cand is not None:
        bx, by, bbw, bbh, corners, convex = best_cand
        # Relative coordinates in crop [0, 1]
        rel_box = [bx / w, by / h, bbw / w, bbh / h]
        # Target gold box is centered with ~15% padding on each side
        ideal_target = [0.13, 0.13, 0.74, 0.74]
        iou_with_target = iou_xywh(rel_box, ideal_target)

        return {
            "detected": True,
            "method": "contour_poly_wireframe",
            "detected_rel_box": [round(c, 4) for c in rel_box],
            "corners_count": corners,
            "is_convex": bool(convex),
            "score": round(best_cand_score, 3),
            "geometry_iou": round(iou_with_target, 4),
        }

    return {"detected": False, "method": "none", "score": 0.0, "geometry_iou": 0.0}


def classify_state_advanced(crop_gray: np.ndarray, field_path: str) -> dict[str, Any]:
    """Classify checkbox state using Hough lines, diagonal stroke detection, and central core ink."""
    h, w = crop_gray.shape
    if h < 6 or w < 6:
        return {"state": "AMBIGUOUS", "confidence": 0.0, "method": "degraded"}

    blur = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Central Core analysis (central 40% box)
    c_y0, c_y1 = int(0.30 * h), int(0.70 * h)
    c_x0, c_x1 = int(0.30 * w), int(0.70 * w)
    core = binary[c_y0:c_y1, c_x0:c_x1]
    core_dark = float(np.mean(core > 0))

    # 2. Diagonal stroke detection (X-mark or checkmark) via Hough lines
    edges = cv2.Canny(blur, 50, 150)
    # Search edges inside core and inner 70%
    inner_edges = np.zeros_like(edges)
    inner_edges[int(0.15*h):int(0.85*h), int(0.15*w):int(0.85*w)] = edges[int(0.15*h):int(0.85*h), int(0.15*w):int(0.85*w)]
    lines = cv2.HoughLinesP(inner_edges, 1, np.pi / 180, threshold=max(6, min(h, w) // 4), minLineLength=max(5, min(h, w) // 4), maxLineGap=2)

    diag_lines_count = 0
    if lines is not None:
        for line in lines:
            pts = line.ravel()
            if len(pts) >= 4:
                x1, y1, x2, y2 = pts[:4]
                dx = x2 - x1
                dy = y2 - y1
                if dx != 0:
                    angle_deg = abs(math.degrees(math.atan(dy / dx)))
                    # Diagonal stroke is typically 20 to 70 degrees
                    if 20.0 <= angle_deg <= 70.0:
                        diag_lines_count += 1

    # 3. Decision voting
    is_sig = "signature" in field_path.lower() or "signed" in field_path.lower()
    dark_ratio = float(np.mean(binary > 0))

    if is_sig:
        if dark_ratio >= 0.05 and diag_lines_count >= 1:
            state = "CHECKED"
            conf = 0.90
        elif dark_ratio < 0.03:
            state = "UNCHECKED"
            conf = 0.85
        else:
            state = "AMBIGUOUS"
            conf = 0.50
    else:
        # Checkbox: Checked if diagonal strokes exist or heavy core ink
        if diag_lines_count >= 2 or core_dark >= 0.12:
            state = "CHECKED"
            conf = min(0.98, 0.70 + core_dark)
        elif core_dark < 0.02 and diag_lines_count == 0:
            state = "UNCHECKED"
            conf = 0.95
        elif core_dark < 0.05:
            state = "UNCHECKED"
            conf = 0.80
        elif core_dark >= 0.06:
            state = "CHECKED"
            conf = 0.75
        else:
            state = "AMBIGUOUS"
            conf = 0.50

    return {
        "state": state,
        "confidence": round(conf, 3),
        "core_darkness": round(core_dark, 4),
        "diagonal_strokes_count": diag_lines_count,
        "total_darkness": round(dark_ratio, 4),
    }


def analyze_signature_geometry(crop_gray: np.ndarray, gold_bbox: list[float]) -> dict[str, Any]:
    """Analyze signature / wide handwritten regions."""
    h, w = crop_gray.shape
    ar = gold_bbox[2] / max(1e-5, gold_bbox[3])
    blur = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal baseline
    k_line = max(10, w // 3)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_line, 1))
    baselines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)
    has_baseline = float(np.mean(baselines > 0)) > 0.005

    # Measure stroke density above baseline
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    fg_stats = stats[1:] if num_labels > 1 else []
    large_strokes = sum(1 for s in fg_stats if s[cv2.CC_STAT_AREA] >= 50)

    is_sig_geometry = (ar >= 2.5 and gold_bbox[2] >= 0.08)
    state = "CHECKED" if large_strokes >= 2 or np.mean(binary > 0) >= 0.06 else "UNCHECKED"

    return {
        "is_signature_geometry": is_sig_geometry,
        "aspect_ratio": round(ar, 2),
        "has_baseline": has_baseline,
        "large_strokes_count": large_strokes,
        "predicted_state": state,
    }


def run_phase2():
    inv_path = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json"
    with open(inv_path, encoding="utf-8") as f:
        records = json.load(f)["inventory"]

    docs_map = defaultdict(list)
    for r in records:
        docs_map[r["document_id"]].append(r)

    geom_results = []
    state_results = []
    sig_results = []

    t0 = time.perf_counter()

    for doc_id, doc_records in docs_map.items():
        pdf_path = repo_root / "research" / "data" / "full" / f"{doc_id}.pdf"
        doc = fitz.open(pdf_path)

        for r in doc_records:
            gp = r["gold_page"]
            gb = r["gold_bbox"]

            if gp <= len(doc):
                page = doc[gp - 1]
                pw, ph = page.rect.width, page.rect.height

                x0 = gb[0] * pw
                y0 = gb[1] * ph
                x1 = (gb[0] + gb[2]) * pw
                y1 = (gb[1] + gb[3]) * ph
                pad_x = max(2.0, 0.15 * (x1 - x0))
                pad_y = max(2.0, 0.15 * (y1 - y0))

                clip_rect = fitz.Rect(max(0, x0 - pad_x), max(0, y0 - pad_y), min(pw, x1 + pad_x), min(ph, y1 + pad_y))
                crop_pix = page.get_pixmap(dpi=300, clip=clip_rect)
                arr = np.frombuffer(crop_pix.samples, dtype=np.uint8).reshape((crop_pix.height, crop_pix.width, crop_pix.n))
                gray = np.mean(arr[:, :, :3], axis=2).astype(np.uint8) if crop_pix.n >= 3 else arr[:, :, 0]

                # Geometry
                geom = detect_box_geometry(gray)
                # State
                st = classify_state_advanced(gray, r["field_path"])
                # Signature
                sig = analyze_signature_geometry(gray, gb)
            else:
                geom = {"detected": False, "geometry_iou": 0.0}
                st = {"state": "AMBIGUOUS", "confidence": 0.0}
                sig = {"is_signature_geometry": False, "predicted_state": "AMBIGUOUS"}

            gt_val = r["value"]
            gt_state = "CHECKED" if gt_val is True else ("UNCHECKED" if gt_val is False else "OTHER")

            # Record geometry
            geom_rec = {
                "index": r["index"],
                "document_id": doc_id,
                "field_path": r["field_path"],
                "gold_page": gp,
                "gold_bbox": gb,
                "geometry_detection": geom,
                "box_detected": geom.get("detected", False),
                "geometry_iou": geom.get("geometry_iou", 0.0),
            }
            geom_results.append(geom_rec)

            # Record state
            state_rec = {
                "index": r["index"],
                "document_id": doc_id,
                "field_path": r["field_path"],
                "ground_truth_state": gt_state,
                "predicted_state": st["state"],
                "confidence": st["confidence"],
                "is_correct": (st["state"] == gt_state),
                "metrics": st,
            }
            state_results.append(state_rec)

            # Record signature
            sig_rec = {
                "index": r["index"],
                "document_id": doc_id,
                "field_path": r["field_path"],
                "signature_analysis": sig,
            }
            sig_results.append(sig_rec)

        doc.close()

    elapsed = time.perf_counter() - t0

    # Save Phase 2 outputs
    out_geom = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_geometry_results.json"
    with open(out_geom, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "EXP-036C_PHASE_2",
            "total_records": len(geom_results),
            "boxes_detected_count": sum(1 for r in geom_results if r["box_detected"]),
            "boxes_detected_percentage": round((sum(1 for r in geom_results if r["box_detected"]) / len(geom_results)) * 100, 2),
            "mean_geometry_iou": round(float(np.mean([r["geometry_iou"] for r in geom_results if r["box_detected"]])), 4) if any(r["box_detected"] for r in geom_results) else 0.0,
            "records": geom_results,
        }, f, indent=2)

    out_state = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_state_results.json"
    with open(out_state, "w", encoding="utf-8") as f:
        state_correct = sum(1 for r in state_results if r["is_correct"])
        json.dump({
            "experiment": "EXP-036C_PHASE_2",
            "total_records": len(state_results),
            "state_accuracy_count": state_correct,
            "state_accuracy_percentage": round((state_correct / len(state_results)) * 100, 2),
            "records": state_results,
        }, f, indent=2)

    out_sig = repo_root / "research" / "experiments" / "EXP-036" / "signature_region_results.json"
    with open(out_sig, "w", encoding="utf-8") as f:
        sig_count = sum(1 for r in sig_results if r["signature_analysis"]["is_signature_geometry"])
        json.dump({
            "experiment": "EXP-036C_PHASE_2",
            "total_records": len(sig_results),
            "signature_regions_count": sig_count,
            "records": sig_results,
        }, f, indent=2)

    print("=" * 80)
    print("EXP-036C: PHASE 2 — GEOMETRY, STATE, AND SIGNATURE RESULTS")
    print("=" * 80)
    print(f"Total Targets: {len(geom_results)}")
    detected_boxes = sum(1 for r in geom_results if r["box_detected"])
    print(f"Box Geometry Detected: {detected_boxes} / {len(geom_results)} ({detected_boxes/len(geom_results)*100:.2f}%)")
    if detected_boxes > 0:
        mean_iou = np.mean([r["geometry_iou"] for r in geom_results if r["box_detected"]])
        print(f"Mean Detected Box IoU: {mean_iou:.4f}")
    print(f"State Classification Accuracy: {state_correct} / {len(state_results)} ({state_correct/len(state_results)*100:.2f}%)")
    print(f"Signature Regions Identified : {sig_count} / {len(sig_results)} ({sig_count/len(sig_results)*100:.2f}%)")
    print("=" * 80)
    print(f"Elapsed: {elapsed:.2f}s | Saved Phase 2 artifacts.")


if __name__ == "__main__":
    run_phase2()
