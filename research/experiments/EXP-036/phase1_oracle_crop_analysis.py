"""EXP-036C Phase 1: Oracle-Crop Classification & Feature Statistics.

Exhaustively extracts classical CV features from oracle crops of all 286 target regions:
- Global Otsu & Adaptive Gaussian thresholding
- Morphological opening & closing
- Connected component analysis (cv2.connectedComponentsWithStats)
- Contour analysis (cv2.findContours, rectangularity, area, aspect ratio)
- Inner vs outer ink distribution (core vs perimeter)
- Horizontal & vertical projection profile statistics
- Evaluates oracle-crop classification accuracy against ground truth.
Outputs:
- oracle_crop_analysis.json
- feature_statistics.json
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz

repo_root = Path(__file__).resolve().parent.parent.parent.parent


def extract_crop_cv_features(gray: np.ndarray, field_path: str) -> dict[str, Any]:
    """Extract rigorous deterministic computer vision features from grayscale crop."""
    h, w = gray.shape
    total_pixels = h * w
    if total_pixels == 0:
        return {}

    # 1. Global Otsu thresholding
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Adaptive Gaussian thresholding
    block_size = max(5, (min(h, w) // 4) * 2 + 1)
    if block_size % 2 == 0:
        block_size += 1
    block_size = min(block_size, 31)
    adaptive_inv = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 3
    )

    # 3. Morphological operations on thresholded binary mask
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_open = cv2.morphologyEx(otsu_inv, cv2.MORPH_OPEN, kernel_2)
    morph_close = cv2.morphologyEx(otsu_inv, cv2.MORPH_CLOSE, kernel_3)

    # 4. Darkness & Ink ratio
    darkness_ratio = float(np.mean(otsu_inv > 0))
    adaptive_dark_ratio = float(np.mean(adaptive_inv > 0))

    # 5. Canny edge density
    edges = cv2.Canny(blur, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # 6. Connected Components Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu_inv, connectivity=8)
    # Ignore background label 0
    fg_stats = stats[1:] if num_labels > 1 else []
    cc_count = len(fg_stats)
    max_cc_area = float(np.max(fg_stats[:, cv2.CC_STAT_AREA])) if cc_count > 0 else 0.0
    mean_cc_area = float(np.mean(fg_stats[:, cv2.CC_STAT_AREA])) if cc_count > 0 else 0.0

    # 7. Contours & Shape Rectangularity
    contours, _ = cv2.findContours(otsu_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    max_contour_area = 0.0
    best_rectangularity = 0.0
    best_aspect_ratio = 0.0

    for cnt in contours:
        c_area = cv2.contourArea(cnt)
        if c_area > max_contour_area:
            max_contour_area = float(c_area)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            best_aspect_ratio = float(bw / max(1, bh))
            box_area = bw * bh
            best_rectangularity = float(c_area / max(1, box_area))

    # 8. Inner core (central 50%) vs Perimeter (outer 25%) ink distribution
    m_y = max(1, int(0.25 * h))
    m_x = max(1, int(0.25 * w))
    perimeter_mask = np.ones((h, w), dtype=bool)
    if h > 2 * m_y and w > 2 * m_x:
        perimeter_mask[m_y : h - m_y, m_x : w - m_x] = False
        core_mask = ~perimeter_mask
    else:
        core_mask = perimeter_mask

    perimeter_density = float(np.mean(otsu_inv[perimeter_mask] > 0)) if np.any(perimeter_mask) else 0.0
    core_density = float(np.mean(otsu_inv[core_mask] > 0)) if np.any(core_mask) else 0.0

    # 9. Border continuity: fraction of 4 boundary lines with dark pixels
    top_border = np.mean(otsu_inv[:max(1, int(0.15 * h)), :] > 0)
    bottom_border = np.mean(otsu_inv[-max(1, int(0.15 * h)):, :] > 0)
    left_border = np.mean(otsu_inv[:, :max(1, int(0.15 * w))] > 0)
    right_border = np.mean(otsu_inv[:, -max(1, int(0.15 * w)):] > 0)
    border_continuity = float(np.mean([top_border > 0.05, bottom_border > 0.05, left_border > 0.05, right_border > 0.05]))

    # 10. Horizontal and Vertical Projections
    proj_h = np.mean(otsu_inv > 0, axis=1)  # row profile
    proj_v = np.mean(otsu_inv > 0, axis=0)  # col profile
    proj_h_var = float(np.var(proj_h))
    proj_v_var = float(np.var(proj_v))

    # 11. Deterministic Oracle State Prediction
    is_sig = "signature" in field_path.lower() or "signed" in field_path.lower()
    if is_sig:
        if darkness_ratio >= 0.06 and cc_count >= 1:
            predicted_state = "CHECKED"
            conf = min(0.95, 0.50 + darkness_ratio * 2.0)
        else:
            predicted_state = "UNCHECKED"
            conf = 0.85
    else:
        # Checkbox classification rule:
        # Checked if central core has ink, or morph_close fills core substantially
        if core_density >= 0.06 or (perimeter_density >= 0.04 and core_density >= 0.04):
            predicted_state = "CHECKED"
            conf = min(0.98, 0.60 + core_density * 2.0)
        elif perimeter_density >= 0.04 and core_density < 0.04:
            predicted_state = "UNCHECKED"
            conf = min(0.98, 0.70 + (0.04 - core_density) * 5.0)
        elif darkness_ratio < 0.02:
            predicted_state = "UNCHECKED"
            conf = 0.80
        else:
            predicted_state = "AMBIGUOUS"
            conf = 0.40

    return {
        "width_px": w,
        "height_px": h,
        "aspect_ratio": round(w / max(1, h), 3),
        "darkness_ratio": round(darkness_ratio, 4),
        "adaptive_dark_ratio": round(adaptive_dark_ratio, 4),
        "edge_density": round(edge_density, 4),
        "connected_components_count": cc_count,
        "max_cc_area": round(max_cc_area, 1),
        "mean_cc_area": round(mean_cc_area, 1),
        "contour_count": contour_count,
        "max_contour_area": round(max_contour_area, 1),
        "best_rectangularity": round(best_rectangularity, 3),
        "best_contour_aspect_ratio": round(best_aspect_ratio, 3),
        "perimeter_density": round(perimeter_density, 4),
        "core_density": round(core_density, 4),
        "border_continuity": round(border_continuity, 2),
        "proj_h_var": round(proj_h_var, 5),
        "proj_v_var": round(proj_v_var, 5),
        "predicted_state": predicted_state,
        "confidence": round(conf, 3),
    }


def run_phase1_analysis():
    inv_path = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json"
    with open(inv_path, encoding="utf-8") as f:
        records = json.load(f)["inventory"]

    docs_map = defaultdict(list)
    for r in records:
        docs_map[r["document_id"]].append(r)

    results = []
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

                clip_rect = fitz.Rect(
                    max(0.0, x0 - pad_x),
                    max(0.0, y0 - pad_y),
                    min(pw, x1 + pad_x),
                    min(ph, y1 + pad_y),
                )
                crop_pix = page.get_pixmap(dpi=300, clip=clip_rect)
                arr = np.frombuffer(crop_pix.samples, dtype=np.uint8).reshape((crop_pix.height, crop_pix.width, crop_pix.n))
                gray = np.mean(arr[:, :, :3], axis=2).astype(np.uint8) if crop_pix.n >= 3 else arr[:, :, 0]

                features = extract_crop_cv_features(gray, r["field_path"])
            else:
                features = {"predicted_state": "AMBIGUOUS", "confidence": 0.0}

            gt_val = r["value"]
            gt_state = "CHECKED" if gt_val is True else ("UNCHECKED" if gt_val is False else "OTHER")
            pred_state = features.get("predicted_state", "AMBIGUOUS")
            is_correct = (pred_state == gt_state)

            rec_out = {
                "index": r["index"],
                "document_id": doc_id,
                "field_path": r["field_path"],
                "ground_truth_value": gt_val,
                "ground_truth_state": gt_state,
                "predicted_state": pred_state,
                "is_correct": is_correct,
                "confidence": features.get("confidence", 0.0),
                "features": features,
            }
            results.append(rec_out)

        doc.close()

    elapsed = time.perf_counter() - t0

    # Aggregate Statistics
    correct_count = sum(1 for r in results if r["is_correct"])
    accuracy = (correct_count / len(results)) * 100

    gt_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        gt = r["ground_truth_state"]
        pr = r["predicted_state"]
        gt_counts[gt] += 1
        pred_counts[pr] += 1
        confusion[gt][pr] += 1

    # Feature statistical summaries across checked vs unchecked
    feat_stats = defaultdict(lambda: {"checked": [], "unchecked": []})
    for r in results:
        st = r["ground_truth_state"]
        if st in ("CHECKED", "UNCHECKED"):
            f = r["features"]
            for k in ("darkness_ratio", "adaptive_dark_ratio", "edge_density", "core_density", "perimeter_density", "best_rectangularity"):
                if k in f:
                    feat_stats[k][st.lower()].append(f[k])

    feature_summary = {}
    for feat_name, vals in feat_stats.items():
        feature_summary[feat_name] = {
            "checked_mean": round(float(np.mean(vals["checked"])), 4) if vals["checked"] else 0.0,
            "checked_std": round(float(np.std(vals["checked"])), 4) if vals["checked"] else 0.0,
            "unchecked_mean": round(float(np.mean(vals["unchecked"])), 4) if vals["unchecked"] else 0.0,
            "unchecked_std": round(float(np.std(vals["unchecked"])), 4) if vals["unchecked"] else 0.0,
        }

    out_crops = repo_root / "research" / "experiments" / "EXP-036" / "oracle_crop_analysis.json"
    with open(out_crops, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "EXP-036C_PHASE_1",
            "total_records": len(results),
            "oracle_accuracy_percentage": round(accuracy, 2),
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
            "records": results,
        }, f, indent=2)

    out_stats = repo_root / "research" / "experiments" / "EXP-036" / "feature_statistics.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "EXP-036C_PHASE_1",
            "features_summary": feature_summary,
            "ground_truth_distribution": dict(gt_counts),
            "predicted_distribution": dict(pred_counts),
        }, f, indent=2)

    print("=" * 80)
    print("EXP-036C: PHASE 1 — ORACLE-CROP CLASSIFICATION & FEATURE ANALYSIS")
    print("=" * 80)
    print(f"Total Targets Evaluated: {len(results)}")
    print(f"Oracle-Crop Accuracy   : {correct_count} / {len(results)} ({accuracy:.2f}%)")
    print("\nConfusion Matrix (Ground Truth -> Predicted):")
    for gt_k in ("CHECKED", "UNCHECKED", "OTHER"):
        if gt_k in confusion:
            print(f"  {gt_k:<10} -> {dict(confusion[gt_k])}")
    print("\nFeature Separability (Mean ± Std):")
    for feat_name, s in feature_summary.items():
        print(f"  {feat_name:<22}: Checked={s['checked_mean']:.3f}±{s['checked_std']:.3f} | Unchecked={s['unchecked_mean']:.3f}±{s['unchecked_std']:.3f}")
    print("=" * 80)
    print(f"Elapsed: {elapsed:.2f}s | Saved to {out_crops} and {out_stats}")


if __name__ == "__main__":
    run_phase1_analysis()
