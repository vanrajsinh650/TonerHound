"""EXP-036 Phase 0: Step 3 - Raster / Non-Text Diagnosis (Calibrated).

Renders each of the 286 boolean fields at 300 DPI, crops the gold bounding box region,
measures empirical image properties (histogram, darkness ratio, edge density, outline strength,
interior core strength), and classifies into:
- RASTER_CHECKED: Visible mark / check / signature exists
- RASTER_UNCHECKED: Box outline present with empty central core
- RASTER_EMPTY: No meaningful checkbox / box outline or mark in crop
- RASTER_UNKNOWN: Ambiguous or unresolvable representation
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf as fitz

repo_root = Path(__file__).resolve().parent.parent.parent.parent


def analyze_crop(crop_pix: fitz.Pixmap, field_path: str) -> dict[str, Any]:
    """Measure empirical image properties on cropped raster region."""
    arr = np.frombuffer(crop_pix.samples, dtype=np.uint8).reshape((crop_pix.height, crop_pix.width, crop_pix.n))
    if crop_pix.n >= 3:
        gray = np.mean(arr[:, :, :3], axis=2)
    else:
        gray = arr[:, :, 0].astype(float)

    h, w = gray.shape
    total_pixels = h * w
    if total_pixels == 0:
        return {
            "darkness_ratio": 0.0,
            "edge_density": 0.0,
            "outline_strength": 0.0,
            "interior_strength": 0.0,
            "histogram": [0] * 10,
            "classification": "RASTER_EMPTY",
            "reason": "zero dimension crop",
        }

    # Darkness ratio (ink pixels < 180 on 0-255 scale)
    is_dark = (gray < 180)
    darkness_ratio = float(np.mean(is_dark))

    # 10-bin histogram (0 to 255)
    hist, _ = np.histogram(gray, bins=10, range=(0, 256))
    hist_norm = [round(float(count / total_pixels), 4) for count in hist]

    # Edge density via gradient magnitude threshold (> 40)
    gy, gx = np.gradient(gray)
    grad_mag = np.hypot(gx, gy)
    edge_density = float(np.mean(grad_mag > 40))

    # Perimeter (outer 25%) vs Central Core (inner 50%)
    m_y = max(1, int(0.25 * h))
    m_x = max(1, int(0.25 * w))

    perimeter_mask = np.ones((h, w), dtype=bool)
    if h > 2 * m_y and w > 2 * m_x:
        perimeter_mask[m_y : h - m_y, m_x : w - m_x] = False
        core_mask = ~perimeter_mask
    else:
        core_mask = perimeter_mask

    outline_strength = float(np.mean(is_dark[perimeter_mask])) if np.any(perimeter_mask) else 0.0
    interior_strength = float(np.mean(is_dark[core_mask])) if np.any(core_mask) else 0.0

    is_sig = "signature" in field_path.lower() or "signed" in field_path.lower()

    # Documented classification thresholds:
    # 1. RASTER_EMPTY: virtually no ink anywhere in the crop
    if darkness_ratio < 0.02 and edge_density < 0.015:
        classification = "RASTER_EMPTY"
        reason = f"minimal ink: dark_ratio={darkness_ratio:.4f} < 0.02, edge_density={edge_density:.4f} < 0.015"

    # 2. Signature fields: handwriting strokes in signature box
    elif is_sig:
        if darkness_ratio >= 0.05 and edge_density >= 0.025:
            classification = "RASTER_CHECKED"
            reason = f"signature mark present: dark_ratio={darkness_ratio:.4f} >= 0.05, edge_density={edge_density:.4f}"
        elif darkness_ratio < 0.03:
            classification = "RASTER_UNCHECKED"
            reason = f"empty signature line: dark_ratio={darkness_ratio:.4f} < 0.03"
        else:
            classification = "RASTER_UNKNOWN"
            reason = f"ambiguous signature region: dark_ratio={darkness_ratio:.4f}"

    # 3. Standard checkboxes:
    # A. RASTER_UNCHECKED: perimeter outline is present, but central core is empty
    elif outline_strength >= 0.04 and interior_strength < 0.04:
        classification = "RASTER_UNCHECKED"
        reason = f"box outline present (outline={outline_strength:.4f} >= 0.04) with clear core (interior={interior_strength:.4f} < 0.04)"

    # B. RASTER_CHECKED: central core contains visible mark/X/check
    elif interior_strength >= 0.06:
        classification = "RASTER_CHECKED"
        reason = f"visible interior mark (interior={interior_strength:.4f} >= 0.06)"

    # C. Ambiguous / Partial
    elif outline_strength >= 0.03 or darkness_ratio >= 0.05:
        classification = "RASTER_UNKNOWN"
        reason = f"intermediate density (outline={outline_strength:.4f}, interior={interior_strength:.4f}, dark={darkness_ratio:.4f})"

    else:
        classification = "RASTER_EMPTY"
        reason = f"insufficient ink for outline or mark (darkness={darkness_ratio:.4f} < 0.03)"

    return {
        "width_px": w,
        "height_px": h,
        "darkness_ratio": round(darkness_ratio, 4),
        "edge_density": round(edge_density, 4),
        "outline_strength": round(outline_strength, 4),
        "interior_strength": round(interior_strength, 4),
        "histogram_10bin": hist_norm,
        "classification": classification,
        "reason": reason,
    }


def run_raster_diagnosis():
    inv_path = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json"
    with open(inv_path, encoding="utf-8") as f:
        inv_data = json.load(f)

    records = inv_data["inventory"]
    print(f"Loaded {len(records)} records from checkbox inventory.")

    docs_map = defaultdict(list)
    for r in records:
        docs_map[r["document_id"]].append(r)

    results = []
    class_counts = defaultdict(int)

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
                pad_x = max(2.0, 0.10 * (x1 - x0))
                pad_y = max(2.0, 0.10 * (y1 - y0))

                clip_rect = fitz.Rect(
                    max(0.0, x0 - pad_x),
                    max(0.0, y0 - pad_y),
                    min(pw, x1 + pad_x),
                    min(ph, y1 + pad_y),
                )
                crop_pix = page.get_pixmap(dpi=300, clip=clip_rect)
                metrics = analyze_crop(crop_pix, r["field_path"])
            else:
                metrics = {
                    "width_px": 0,
                    "height_px": 0,
                    "darkness_ratio": 0.0,
                    "edge_density": 0.0,
                    "outline_strength": 0.0,
                    "interior_strength": 0.0,
                    "histogram_10bin": [0] * 10,
                    "classification": "RASTER_UNKNOWN",
                    "reason": "page number exceeds document length",
                }

            rec_out = {
                "index": r["index"],
                "document_id": doc_id,
                "field_path": r["field_path"],
                "value": r["value"],
                "gold_page": gp,
                "gold_bbox": gb,
                "metrics": metrics,
                "raster_classification": metrics["classification"],
            }
            results.append(rec_out)
            class_counts[metrics["classification"]] += 1

        doc.close()

    elapsed = time.perf_counter() - t0

    out_json = repo_root / "research" / "experiments" / "EXP-036" / "raster_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "EXP-036_PHASE_0",
            "total_records": len(results),
            "summary": {
                k: {
                    "count": class_counts[k],
                    "percentage": round((class_counts[k] / len(results)) * 100, 2),
                }
                for k in ("RASTER_CHECKED", "RASTER_UNCHECKED", "RASTER_EMPTY", "RASTER_UNKNOWN")
            },
            "records": results,
        }, f, indent=2)

    print("=" * 70)
    print("EXP-036 PHASE 0: STEP 3 — RASTER / NON-TEXT DIAGNOSIS RESULTS")
    print("=" * 70)
    for k in ("RASTER_CHECKED", "RASTER_UNCHECKED", "RASTER_EMPTY", "RASTER_UNKNOWN"):
        count = class_counts[k]
        pct = (count / len(results)) * 100
        print(f"  {k:<20}: {count:>4} / {len(results)} ({pct:>6.2f}%)")
    print(f"Elapsed: {elapsed:.2f}s | Saved to {out_json}")


if __name__ == "__main__":
    run_raster_diagnosis()
