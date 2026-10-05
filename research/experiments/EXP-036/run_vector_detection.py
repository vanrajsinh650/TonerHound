"""EXP-036 Phase 0: Step 1 - Vector Path Detection.

Analyzes vector paths from page.get_drawings() for all 286 boolean checkbox fields.
Computes geometric IoU against gold bounding boxes and classifies into:
- VECTOR_BOX_DETECTED (IoU >= 0.50)
- VECTOR_PARTIAL (0.25 <= IoU < 0.50)
- NO_VECTOR_DETECTION (IoU < 0.25)
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pymupdf as fitz

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from tonerhound.geometry.coordinates import union_bbox_list


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


def run_vector_detection():
    inv_path = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json"
    with open(inv_path, encoding="utf-8") as f:
        inv_data = json.load(f)

    records = inv_data["inventory"]
    print(f"Loaded {len(records)} records from checkbox inventory.")

    # Group by document to load each PDF only once
    docs_map = defaultdict(list)
    for r in records:
        docs_map[r["document_id"]].append(r)

    results = []
    category_counts = defaultdict(int)

    t0 = time.perf_counter()

    for doc_id, doc_records in docs_map.items():
        pdf_path = repo_root / "research" / "data" / "full" / f"{doc_id}.pdf"
        doc = fitz.open(pdf_path)

        # Cache page drawings
        page_drawings_cache = {}

        for r in doc_records:
            gp = r["gold_page"]
            gb = r["gold_bbox"]

            if gp not in page_drawings_cache:
                if gp <= len(doc):
                    page = doc[gp - 1]
                    pw, ph = page.rect.width, page.rect.height
                    drawings = page.get_drawings()
                    page_drawings_cache[gp] = (pw, ph, drawings)
                else:
                    page_drawings_cache[gp] = (0.0, 0.0, [])

            pw, ph, drawings = page_drawings_cache[gp]

            candidates = []
            max_iou = 0.0

            if pw > 0 and ph > 0 and drawings:
                for d in drawings:
                    dr = d["rect"]
                    # Normalize drawing rect to 0.0 - 1.0
                    nx = dr.x0 / pw
                    ny = dr.y0 / ph
                    nw = (dr.x1 - dr.x0) / pw
                    nh = (dr.y1 - dr.y0) / ph
                    cand_box = [round(nx, 6), round(ny, 6), round(nw, 6), round(nh, 6)]

                    iou = iou_xywh(cand_box, gb)
                    if iou > max_iou:
                        max_iou = iou

                    # Record candidates with plausible proximity/overlap (IoU >= 0.10 or spatial intersection)
                    if iou >= 0.10 or (
                        nx < gb[0] + gb[2] and nx + nw > gb[0] and
                        ny < gb[1] + gb[3] and ny + nh > gb[1]
                    ):
                        candidates.append({
                            "bbox": cand_box,
                            "iou": round(iou, 4),
                            "items_count": len(d.get("items", [])),
                            "fill": str(d.get("fill")),
                            "color": str(d.get("color")),
                            "width": round(d["width"], 3) if d.get("width") is not None else 0.0,
                        })

            # Classification per specification
            if max_iou >= 0.50:
                det_result = "VECTOR_BOX_DETECTED"
            elif max_iou >= 0.25:
                det_result = "VECTOR_PARTIAL"
            else:
                det_result = "NO_VECTOR_DETECTION"

            record_out = {
                "index": r["index"],
                "document_id": doc_id,
                "field_path": r["field_path"],
                "value": r["value"],
                "gold_page": gp,
                "gold_bbox": gb,
                "detection_result": det_result,
                "max_iou": round(max_iou, 4),
                "candidates_count": len(candidates),
                "candidates": sorted(candidates, key=lambda c: c["iou"], reverse=True)[:5],
            }
            results.append(record_out)
            category_counts[det_result] += 1

        doc.close()

    elapsed = time.perf_counter() - t0

    out_json = repo_root / "research" / "experiments" / "EXP-036" / "vector_detection.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "EXP-036_PHASE_0",
            "total_records": len(results),
            "summary": {
                k: {
                    "count": category_counts[k],
                    "percentage": round((category_counts[k] / len(results)) * 100, 2),
                }
                for k in ("VECTOR_BOX_DETECTED", "VECTOR_PARTIAL", "NO_VECTOR_DETECTION")
            },
            "records": results,
        }, f, indent=2)

    print("=" * 70)
    print("EXP-036 PHASE 0: STEP 1 — VECTOR PATH DETECTION RESULTS")
    print("=" * 70)
    for k in ("VECTOR_BOX_DETECTED", "VECTOR_PARTIAL", "NO_VECTOR_DETECTION"):
        count = category_counts[k]
        pct = (count / len(results)) * 100
        print(f"  {k:<24}: {count:>4} / {len(results)} ({pct:>6.2f}%)")
    print(f"Elapsed: {elapsed:.2f}s | Saved to {out_json}")


if __name__ == "__main__":
    run_vector_detection()
