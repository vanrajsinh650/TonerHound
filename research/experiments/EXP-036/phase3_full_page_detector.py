"""EXP-036C Phase 3: Full-Page End-to-End Detector.

Pure classical deterministic computer vision pipeline:
- Morphological wireframe filtering (horizontal & vertical kernels)
- Multi-scale contour extraction (checkboxes, small form boxes, signature baselines)
- Non-maximum suppression (NMS)
- Deterministic confidence scoring:
    confidence = (
        0.35 * shape_score
        + 0.25 * border_score
        + 0.20 * interior_score
        + 0.20 * repetition_score
    )
- Candidate state classification (checked, unchecked, ambiguous)
- Target association & evaluation against the 286-field inventory

Outputs:
- research/experiments/EXP-036/full_page_detection.json
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


def compute_iou(b1: list[float] | tuple[float, ...], b2: list[float] | tuple[float, ...]) -> float:
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


def apply_nms(candidates: list[dict[str, Any]], iou_thresh: float = 0.35) -> list[dict[str, Any]]:
    """Non-maximum suppression based on candidate confidence."""
    if not candidates:
        return []

    # Sort descending by confidence then area
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c["confidence"], c["bbox_px"][2] * c["bbox_px"][3]),
        reverse=True,
    )

    keep = []
    for cand in sorted_cands:
        b1 = cand["bbox_px"]
        suppress = False
        for k in keep:
            b2 = k["bbox_px"]
            # If same general type and high overlap, suppress
            if compute_iou(b1, b2) > iou_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(cand)

    return keep


def classify_candidate_state(crop_gray: np.ndarray, obj_type: str) -> tuple[str, float]:
    """Classify candidate state using central core darkness and Hough diagonal strokes."""
    h, w = crop_gray.shape
    if h < 6 or w < 6:
        return "AMBIGUOUS", 0.50

    blur = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if obj_type == "SIGNATURE_OR_HANDWRITTEN_MARK":
        # Signature: check for handwritten strokes across upper 70% of crop
        sig_body = binary[0 : int(0.85 * h), :]
        sig_darkness = float(np.mean(sig_body > 0)) if sig_body.size > 0 else 0.0
        if sig_darkness >= 0.045:
            return "CHECKED", min(0.95, 0.70 + sig_darkness)
        elif sig_darkness < 0.020:
            return "UNCHECKED", 0.90
        else:
            return "AMBIGUOUS", 0.55

    # Checkbox / Form Box
    c_y0, c_y1 = int(0.30 * h), int(0.70 * h)
    c_x0, c_x1 = int(0.30 * w), int(0.70 * w)
    core = binary[c_y0:c_y1, c_x0:c_x1]
    core_dark = float(np.mean(core > 0)) if core.size > 0 else 0.0

    # Diagonal strokes inside box
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


def detect_page_candidates(gray: np.ndarray) -> list[dict[str, Any]]:
    """Detect visual form objects across a full page image."""
    img_h, img_w = gray.shape

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Morphological Wireframe Extraction for Box-shaped Objects
    k_box = 15
    h_box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box, 1))
    v_box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_box))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_box_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_box_kernel)
    wireframe = cv2.bitwise_or(h_lines, v_lines)

    cnts_wire, _ = cv2.findContours(wireframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_bin, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Horizontal Baseline Rules for Signatures
    k_sig = 80
    h_sig_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sig, 1))
    baselines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_sig_kernel)
    cnts_sig, _ = cv2.findContours(baselines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_candidates: list[dict[str, Any]] = []

    # Process Box Contours
    for c in list(cnts_wire) + list(cnts_bin):
        x, y, w, h = cv2.boundingRect(c)
        ar = w / max(1, h)

        # Square checkbox: 20 to 95 px; Small form box: 20 to 120 px
        if 20 <= w <= 125 and 20 <= h <= 125 and 0.45 <= ar <= 2.2:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            n_vertices = len(approx)
            is_convex = cv2.isContourConvex(approx)
            area = cv2.contourArea(c)
            fullness = area / (w * h)

            if fullness < 0.35:
                continue

            # Deterministic Shape Score (0.0 to 1.0)
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

            # Border continuity score: perimeter darkness
            sub = binary[y : y + h, x : x + w]
            if sub.size > 0:
                top_p = np.mean(sub[:3, :] > 0)
                bot_p = np.mean(sub[-3:, :] > 0)
                left_p = np.mean(sub[:, :3] > 0)
                right_p = np.mean(sub[:, -3:] > 0)
                border_score = float((top_p + bot_p + left_p + right_p) / 4.0)
            else:
                border_score = 0.0

            # Interior score: contrast between border and center
            c_y0, c_y1 = int(0.30 * h), int(0.70 * h)
            c_x0, c_x1 = int(0.30 * w), int(0.70 * w)
            core = sub[c_y0:c_y1, c_x0:c_x1] if sub.size > 0 else np.zeros((1, 1))
            core_dark = float(np.mean(core > 0)) if core.size > 0 else 0.0
            interior_score = 0.90 if core_dark < 0.05 or core_dark > 0.15 else 0.50

            obj_type = "SQUARE_CHECKBOX" if 0.70 <= ar <= 1.40 and w <= 80 else "RADIO_OR_SMALL_FORM_BOX"

            raw_candidates.append({
                "bbox_px": [x, y, w, h],
                "bbox": [round(x / img_w, 4), round(y / img_h, 4), round(w / img_w, 4), round(h / img_h, 4)],
                "object_type": obj_type,
                "shape_score": round(shape_score, 3),
                "border_score": round(border_score, 3),
                "interior_score": round(interior_score, 3),
                "repetition_score": 0.0,  # Computed in post-clustering
            })

    # Process Signature Contours
    for c in cnts_sig:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 140:
            sig_y = max(0, y - int(0.18 * w))
            sig_h = int(0.20 * w)
            raw_candidates.append({
                "bbox_px": [x, sig_y, w, sig_h],
                "bbox": [round(x / img_w, 4), round(sig_y / img_h, 4), round(w / img_w, 4), round(sig_h / img_h, 4)],
                "object_type": "SIGNATURE_OR_HANDWRITTEN_MARK",
                "shape_score": 0.85,
                "border_score": 0.80,
                "interior_score": 0.80,
                "repetition_score": 0.0,
            })

    # 3. Repeated-Form Alignment Scoring
    # Check if candidate shares column (similar X) or row (similar Y) with other candidates
    x_coords = [c["bbox_px"][0] for c in raw_candidates if c["object_type"] != "SIGNATURE_OR_HANDWRITTEN_MARK"]
    y_coords = [c["bbox_px"][1] for c in raw_candidates if c["object_type"] != "SIGNATURE_OR_HANDWRITTEN_MARK"]

    for c in raw_candidates:
        if c["object_type"] != "SIGNATURE_OR_HANDWRITTEN_MARK":
            cx, cy = c["bbox_px"][0], c["bbox_px"][1]
            aligned_x = sum(1 for ox in x_coords if abs(cx - ox) <= 12 and cx != ox)
            aligned_y = sum(1 for oy in y_coords if abs(cy - oy) <= 12 and cy != oy)
            if aligned_x >= 2 or aligned_y >= 2:
                c["repetition_score"] = 0.95
            elif aligned_x >= 1 or aligned_y >= 1:
                c["repetition_score"] = 0.70
            else:
                c["repetition_score"] = 0.30
        else:
            c["repetition_score"] = 0.50

        # Deterministic Confidence Formula
        conf = (
            0.35 * c["shape_score"]
            + 0.25 * c["border_score"]
            + 0.20 * c["interior_score"]
            + 0.20 * c["repetition_score"]
        )
        c["confidence"] = round(conf, 3)

    # 4. Non-Maximum Suppression
    dedup_candidates = apply_nms(raw_candidates, iou_thresh=0.35)

    # 5. State Classification for Deduped Candidates
    for c in dedup_candidates:
        x, y, w, h = c["bbox_px"]
        crop = gray[max(0, y) : min(img_h, y + h), max(0, x) : min(img_w, x + w)]
        st, st_conf = classify_candidate_state(crop, c["object_type"])
        c["state"] = st
        c["state_confidence"] = round(st_conf, 3)

    return dedup_candidates


def run_phase3():
    inv_path = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json"
    with open(inv_path, encoding="utf-8") as f:
        records = json.load(f)["inventory"]

    # Group by (document_id, gold_page)
    pages_map = defaultdict(list)
    for r in records:
        pages_map[(r["document_id"], r["gold_page"])].append(r)

    total_pages = len(pages_map)
    total_targets = len(records)

    page_results = []
    target_evaluations = []

    total_render_sec = 0.0
    total_detect_sec = 0.0
    all_candidate_counts = []

    print("=" * 80)
    print(f"EXP-036C PHASE 3: RUNNING FULL-PAGE DETECTOR ON {total_pages} PAGES ({total_targets} TARGETS)")
    print("=" * 80)

    t0_overall = time.perf_counter()

    for (doc_id, page_num), doc_records in pages_map.items():
        pdf_path = repo_root / "research" / "data" / "full" / f"{doc_id}.pdf"
        doc = fitz.open(pdf_path)

        t0_render = time.perf_counter()
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=300)
        img_w, img_h = pix.width, pix.height
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((img_h, img_w, pix.n))
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if pix.n >= 3 else arr[:, :, 0]
        render_sec = time.perf_counter() - t0_render
        total_render_sec += render_sec

        t0_detect = time.perf_counter()
        candidates = detect_page_candidates(gray)
        detect_sec = time.perf_counter() - t0_detect
        total_detect_sec += detect_sec

        all_candidate_counts.append(len(candidates))

        page_record = {
            "document_id": doc_id,
            "page": page_num,
            "image_dimensions": [img_w, img_h],
            "render_ms": round(render_sec * 1000, 2),
            "detect_ms": round(detect_sec * 1000, 2),
            "candidates_count": len(candidates),
            "candidate_regions": [
                {
                    "bbox": c["bbox"],
                    "object_type": c["object_type"],
                    "state": c["state"],
                    "confidence": c["confidence"],
                }
                for c in candidates
            ],
        }
        page_results.append(page_record)

        # Evaluate against targets on this page
        for r in doc_records:
            gb = r["gold_bbox"]
            gb_px = [gb[0] * img_w, gb[1] * img_h, gb[2] * img_w, gb[3] * img_h]
            gt_val = r["value"]
            gt_state = "CHECKED" if gt_val is True else ("UNCHECKED" if gt_val is False else "OTHER")

            best_iou = 0.0
            best_cand = None

            for c in candidates:
                iou = compute_iou(c["bbox_px"], gb_px)
                if iou > best_iou:
                    best_iou = iou
                    best_cand = c

            is_localized_50 = best_iou >= 0.50
            is_localized_30 = best_iou >= 0.30
            cand_state = best_cand["state"] if best_cand else "NONE"
            state_match = (cand_state == gt_state)
            grounding_success = is_localized_50 and state_match

            target_eval = {
                "index": r["index"],
                "document_id": doc_id,
                "field_path": r["field_path"],
                "gold_page": page_num,
                "gold_bbox": gb,
                "ground_truth_state": gt_state,
                "best_iou": round(best_iou, 4),
                "is_localized_50": is_localized_50,
                "is_localized_30": is_localized_30,
                "matched_candidate_type": best_cand["object_type"] if best_cand else None,
                "matched_candidate_state": cand_state,
                "matched_candidate_confidence": best_cand["confidence"] if best_cand else 0.0,
                "state_match": state_match,
                "grounding_success": grounding_success,
            }
            target_evaluations.append(target_eval)

        doc.close()

    total_time_sec = time.perf_counter() - t0_overall

    # Summary Statistics
    localized_50_count = sum(1 for t in target_evaluations if t["is_localized_50"])
    localized_30_count = sum(1 for t in target_evaluations if t["is_localized_30"])
    state_match_count = sum(1 for t in target_evaluations if t["state_match"])
    grounding_success_count = sum(1 for t in target_evaluations if t["grounding_success"])
    all_ious = [t["best_iou"] for t in target_evaluations]

    # Breakdown by document length class / family
    family_stats = defaultdict(lambda: {"total": 0, "loc50": 0, "state_match": 0, "success": 0})
    for t in target_evaluations:
        fam = "medium" if t["document_id"].startswith("medium/") else "short"
        family_stats[fam]["total"] += 1
        if t["is_localized_50"]:
            family_stats[fam]["loc50"] += 1
        if t["state_match"]:
            family_stats[fam]["state_match"] += 1
        if t["grounding_success"]:
            family_stats[fam]["success"] += 1

    summary = {
        "experiment": "EXP-036C_PHASE_3",
        "total_targets": total_targets,
        "total_pages": total_pages,
        "overall_runtime_seconds": round(total_time_sec, 2),
        "mean_render_ms_per_page": round((total_render_sec / total_pages) * 1000, 2),
        "mean_detect_ms_per_page": round((total_detect_sec / total_pages) * 1000, 2),
        "mean_candidates_per_page": round(float(np.mean(all_candidate_counts)), 2),
        "target_localization_iou_50_count": localized_50_count,
        "target_localization_iou_50_pct": round((localized_50_count / total_targets) * 100, 2),
        "target_localization_iou_30_count": localized_30_count,
        "target_localization_iou_30_pct": round((localized_30_count / total_targets) * 100, 2),
        "mean_target_iou": round(float(np.mean(all_ious)), 4),
        "median_target_iou": round(float(np.median(all_ious)), 4),
        "state_accuracy_count": state_match_count,
        "state_accuracy_pct": round((state_match_count / total_targets) * 100, 2),
        "grounding_success_count": grounding_success_count,
        "grounding_success_pct": round((grounding_success_count / total_targets) * 100, 2),
        "family_breakdown": dict(family_stats),
    }

    out_json = repo_root / "research" / "experiments" / "EXP-036" / "full_page_detection.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "target_evaluations": target_evaluations,
            "pages": page_results,
        }, f, indent=2)

    print("=" * 80)
    print("PHASE 3 FULL-PAGE DETECTOR SUMMARY:")
    print("=" * 80)
    print(f"Total Targets: {total_targets} across {total_pages} pages")
    print(f"Average Speed: {summary['mean_render_ms_per_page']} ms render + {summary['mean_detect_ms_per_page']} ms detect per page")
    print(f"Candidates per Page: {summary['mean_candidates_per_page']}")
    print(f"Localization (IoU >= 0.50): {localized_50_count} / {total_targets} ({summary['target_localization_iou_50_pct']}%)")
    print(f"Localization (IoU >= 0.30): {localized_30_count} / {total_targets} ({summary['target_localization_iou_30_pct']}%)")
    print(f"Mean Target IoU: {summary['mean_target_iou']} | Median Target IoU: {summary['median_target_iou']}")
    print(f"State Accuracy: {state_match_count} / {total_targets} ({summary['state_accuracy_pct']}%)")
    print(f"Grounding Success (IoU >= 0.50 & State Correct): {grounding_success_count} / {total_targets} ({summary['grounding_success_pct']}%)")
    print("-" * 80)
    for fam, stats in family_stats.items():
        print(f"  Family '{fam}': {stats['success']} / {stats['total']} grounding success ({stats['success']/stats['total']*100:.1f}%), {stats['loc50']} / {stats['total']} localized ({stats['loc50']/stats['total']*100:.1f}%)")
    print("=" * 80)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    run_phase3()
