"""EXP-036C Phase 4: Systematic Ablation Study.

Compares 6 distinct deterministic configurations as mandated by Directive Section 14:
A. Darkness-only: Global thresholding + connected component ink density
B. Geometry-only: Wireframe morphology + aspect-ratio/size filtering, naive overall state
C. Geometry + interior occupancy: Wireframe geometry + central core (30%-70%) darkness
D. Geometry + mark detection: Wireframe geometry + Hough diagonal strokes
E. Geometry + repeated-form structure: Wireframe geometry + column/row alignment
F. Full deterministic system: Full multi-feature pipeline (geometry + core + Hough + baselines + repetition + NMS)

Outputs:
- research/experiments/EXP-036/ablation_results.json
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
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda c: (c.get("confidence", 0.5), c["bbox_px"][2] * c["bbox_px"][3]), reverse=True)
    keep = []
    for cand in sorted_cands:
        b1 = cand["bbox_px"]
        suppress = False
        for k in keep:
            if compute_iou(b1, k["bbox_px"]) > iou_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(cand)
    return keep


def run_configuration(config_name: str, pages_data: dict[tuple[str, int], tuple[np.ndarray, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Execute a single ablation configuration across all pages."""
    t0 = time.perf_counter()
    all_ious = []
    loc50_count = 0
    loc30_count = 0
    state_match_count = 0
    grounding_success_count = 0
    total_candidates = 0
    total_targets = 0

    for (doc_id, page_num), (gray, targets) in pages_data.items():
        img_h, img_w = gray.shape
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        candidates: list[dict[str, Any]] = []

        if config_name == "A_DARKNESS_ONLY":
            # Connected components on binary image without wireframe filtering
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                if 20 <= w <= 120 and 20 <= h <= 120 and area >= 50:
                    crop = binary[y : y + h, x : x + w]
                    darkness = float(np.mean(crop > 0)) if crop.size > 0 else 0.0
                    state = "CHECKED" if darkness >= 0.20 else "UNCHECKED"
                    candidates.append({"bbox_px": [x, y, w, h], "state": state, "confidence": round(darkness, 3)})

        else:
            # Wireframe line morphology
            k_box = 15
            h_box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box, 1))
            v_box_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_box))
            h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_box_kernel)
            v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_box_kernel)
            wireframe = cv2.bitwise_or(h_lines, v_lines)

            cnts_wire, _ = cv2.findContours(wireframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts_bin, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            raw_cands = []
            for c in list(cnts_wire) + list(cnts_bin):
                x, y, w, h = cv2.boundingRect(c)
                ar = w / max(1, h)
                if 20 <= w <= 125 and 20 <= h <= 125 and 0.45 <= ar <= 2.2:
                    fullness = cv2.contourArea(c) / (w * h)
                    if fullness >= 0.35:
                        raw_cands.append({"bbox_px": [x, y, w, h], "ar": ar, "fullness": fullness})

            # Baseline for signatures if Config F
            if config_name == "F_FULL_SYSTEM":
                k_sig = 80
                h_sig_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sig, 1))
                baselines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_sig_kernel)
                cnts_sig, _ = cv2.findContours(baselines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts_sig:
                    x, y, w, h = cv2.boundingRect(c)
                    if w >= 140:
                        raw_cands.append({"bbox_px": [x, max(0, y - int(0.18 * w)), w, int(0.20 * w)], "ar": w / (0.20 * w), "is_sig": True})

            # Alignment scoring if E or F
            if config_name in ("E_GEOMETRY_REPEATED_STRUCTURE", "F_FULL_SYSTEM"):
                x_coords = [c["bbox_px"][0] for c in raw_cands if not c.get("is_sig")]
                y_coords = [c["bbox_px"][1] for c in raw_cands if not c.get("is_sig")]
                for c in raw_cands:
                    cx, cy = c["bbox_px"][0], c["bbox_px"][1]
                    al_x = sum(1 for ox in x_coords if abs(cx - ox) <= 12 and cx != ox)
                    al_y = sum(1 for oy in y_coords if abs(cy - oy) <= 12 and cy != oy)
                    c["rep_score"] = 0.95 if (al_x >= 2 or al_y >= 2) else (0.70 if (al_x >= 1 or al_y >= 1) else 0.30)

            # State determination according to config
            for c in raw_cands:
                x, y, w, h = c["bbox_px"]
                crop = binary[y : y + h, x : x + w]
                if crop.size == 0:
                    c["state"] = "AMBIGUOUS"
                    c["confidence"] = 0.0
                    continue

                if config_name == "B_GEOMETRY_ONLY":
                    # Naive overall darkness
                    dark = float(np.mean(crop > 0))
                    c["state"] = "CHECKED" if dark >= 0.15 else "UNCHECKED"
                    c["confidence"] = 0.60

                elif config_name in ("C_GEOMETRY_INTERIOR_OCCUPANCY", "E_GEOMETRY_REPEATED_STRUCTURE"):
                    # Central core darkness only
                    c_y0, c_y1 = int(0.30 * h), int(0.70 * h)
                    c_x0, c_x1 = int(0.30 * w), int(0.70 * w)
                    core = crop[c_y0:c_y1, c_x0:c_x1]
                    core_dark = float(np.mean(core > 0)) if core.size > 0 else 0.0
                    c["state"] = "CHECKED" if core_dark >= 0.07 else "UNCHECKED"
                    c["confidence"] = round(0.50 + c.get("rep_score", 0.20), 3)

                elif config_name == "D_GEOMETRY_MARK_DETECTION":
                    # Core + Hough diagonal strokes
                    c_y0, c_y1 = int(0.30 * h), int(0.70 * h)
                    c_x0, c_x1 = int(0.30 * w), int(0.70 * w)
                    core = crop[c_y0:c_y1, c_x0:c_x1]
                    core_dark = float(np.mean(core > 0)) if core.size > 0 else 0.0

                    edges = cv2.Canny(crop, 50, 150)
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(6, min(h, w) // 4), minLineLength=max(5, min(h, w) // 4), maxLineGap=2)
                    diag_count = 0
                    if lines is not None:
                        for line in lines:
                            pts = line.ravel()
                            if len(pts) >= 4 and pts[2] - pts[0] != 0:
                                deg = abs(math.degrees(math.atan((pts[3] - pts[1]) / (pts[2] - pts[0]))))
                                if 20.0 <= deg <= 70.0:
                                    diag_count += 1
                    c["state"] = "CHECKED" if (diag_count >= 2 or core_dark >= 0.10) else ("UNCHECKED" if core_dark < 0.05 else "AMBIGUOUS")
                    c["confidence"] = 0.75

                elif config_name == "F_FULL_SYSTEM":
                    # Full multi-feature logic
                    if c.get("is_sig"):
                        sig_body = crop[0 : int(0.85 * h), :]
                        sig_dark = float(np.mean(sig_body > 0)) if sig_body.size > 0 else 0.0
                        c["state"] = "CHECKED" if sig_dark >= 0.045 else "UNCHECKED"
                        c["confidence"] = 0.85
                    else:
                        c_y0, c_y1 = int(0.30 * h), int(0.70 * h)
                        c_x0, c_x1 = int(0.30 * w), int(0.70 * w)
                        core = crop[c_y0:c_y1, c_x0:c_x1]
                        core_dark = float(np.mean(core > 0)) if core.size > 0 else 0.0
                        edges = cv2.Canny(crop, 50, 150)
                        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(6, min(h, w) // 4), minLineLength=max(5, min(h, w) // 4), maxLineGap=2)
                        diag_count = 0
                        if lines is not None:
                            for line in lines:
                                pts = line.ravel()
                                if len(pts) >= 4 and pts[2] - pts[0] != 0:
                                    deg = abs(math.degrees(math.atan((pts[3] - pts[1]) / (pts[2] - pts[0]))))
                                    if 20.0 <= deg <= 70.0:
                                        diag_count += 1
                        c["state"] = "CHECKED" if (diag_count >= 2 or core_dark >= 0.08) else ("UNCHECKED" if core_dark < 0.05 else "AMBIGUOUS")
                        c["confidence"] = round(0.40 + 0.30 * c.get("rep_score", 0.50) + min(0.30, core_dark), 3)

            candidates = apply_nms(raw_cands, iou_thresh=0.35)

        total_candidates += len(candidates)
        total_targets += len(targets)

        # Evaluate targets
        for r in targets:
            gb = r["gold_bbox"]
            gb_px = [gb[0] * img_w, gb[1] * img_h, gb[2] * img_w, gb[3] * img_h]
            gt_state = "CHECKED" if r["value"] is True else ("UNCHECKED" if r["value"] is False else "OTHER")

            best_iou = 0.0
            best_c = None
            for c in candidates:
                iou = compute_iou(c["bbox_px"], gb_px)
                if iou > best_iou:
                    best_iou = iou
                    best_c = c

            all_ious.append(best_iou)
            if best_iou >= 0.50:
                loc50_count += 1
            if best_iou >= 0.30:
                loc30_count += 1
            if best_c and best_c["state"] == gt_state:
                state_match_count += 1
                if best_iou >= 0.50:
                    grounding_success_count += 1

    elapsed = time.perf_counter() - t0
    num_pages = len(pages_data)

    return {
        "configuration": config_name,
        "runtime_seconds": round(elapsed, 2),
        "mean_ms_per_page": round((elapsed / num_pages) * 1000, 2),
        "mean_candidates_per_page": round(total_candidates / num_pages, 2),
        "total_targets": total_targets,
        "localized_iou_50_count": loc50_count,
        "localized_iou_50_pct": round((loc50_count / total_targets) * 100, 2),
        "localized_iou_30_count": loc30_count,
        "localized_iou_30_pct": round((loc30_count / total_targets) * 100, 2),
        "mean_iou": round(float(np.mean(all_ious)), 4),
        "median_iou": round(float(np.median(all_ious)), 4),
        "state_accuracy_count": state_match_count,
        "state_accuracy_pct": round((state_match_count / total_targets) * 100, 2),
        "grounding_success_count": grounding_success_count,
        "grounding_success_pct": round((grounding_success_count / total_targets) * 100, 2),
    }


def run_phase4():
    inv_path = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json"
    with open(inv_path, encoding="utf-8") as f:
        records = json.load(f)["inventory"]

    # Pre-render pages once so all configurations run on identical cached image data
    pages_map = defaultdict(list)
    for r in records:
        pages_map[(r["document_id"], r["gold_page"])].append(r)

    print(f"Pre-rendering {len(pages_map)} pages at 300 DPI for fair ablation benchmarking...")
    pages_data = {}
    for (doc_id, page_num), doc_records in pages_map.items():
        pdf_path = repo_root / "research" / "data" / "full" / f"{doc_id}.pdf"
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=300)
        img_w, img_h = pix.width, pix.height
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((img_h, img_w, pix.n))
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if pix.n >= 3 else arr[:, :, 0]
        pages_data[(doc_id, page_num)] = (gray, doc_records)
        doc.close()

    print("Pre-rendering complete. Commencing systematic ablations...")
    configs = [
        "A_DARKNESS_ONLY",
        "B_GEOMETRY_ONLY",
        "C_GEOMETRY_INTERIOR_OCCUPANCY",
        "D_GEOMETRY_MARK_DETECTION",
        "E_GEOMETRY_REPEATED_STRUCTURE",
        "F_FULL_SYSTEM",
    ]

    results = []
    print("=" * 95)
    print(f"{'Config':<32} | {'Cands/Pg':<8} | {'Loc@50':<8} | {'MeanIoU':<8} | {'StateAcc':<9} | {'GroundSuccess':<13}")
    print("=" * 95)

    for cfg in configs:
        res = run_configuration(cfg, pages_data)
        results.append(res)
        print(
            f"{res['configuration']:<32} | "
            f"{res['mean_candidates_per_page']:<8.1f} | "
            f"{res['localized_iou_50_pct']:<7.1f}% | "
            f"{res['mean_iou']:<8.4f} | "
            f"{res['state_accuracy_pct']:<8.1f}% | "
            f"{res['grounding_success_count']:>3d} ({res['grounding_success_pct']:<4.1f}%)"
        )

    print("=" * 95)

    out_path = repo_root / "research" / "experiments" / "EXP-036" / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "EXP-036C_PHASE_4_ABLATIONS",
            "total_targets": len(records),
            "total_pages": len(pages_map),
            "configurations": results,
        }, f, indent=2)

    print(f"Ablation results successfully saved to {out_path}")


if __name__ == "__main__":
    run_phase4()
