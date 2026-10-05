"""EXP-036 Phase 0: Step 4 - Reconciled Final Classification Summary.

Combines Vector Path Detection, Text-Layer Glyph Detection, and Raster Diagnosis
under strict operational precedence:
VECTOR_BOX_DETECTED > GLYPH_DETECTED > VECTOR_PARTIAL > RASTER_CHECKED > RASTER_UNCHECKED > RASTER_EMPTY > RASTER_UNKNOWN
Reconciles to exactly 286 records.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent

with open(repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json") as f:
    inv_records = json.load(f)["inventory"]

with open(repo_root / "research" / "experiments" / "EXP-036" / "vector_detection.json") as f:
    v_records = {r["index"]: r for r in json.load(f)["records"]}

with open(repo_root / "research" / "experiments" / "EXP-036" / "glyph_detection.json") as f:
    g_records = {r["index"]: r for r in json.load(f)["records"]}

with open(repo_root / "research" / "experiments" / "EXP-036" / "raster_analysis.json") as f:
    r_records = {r["index"]: r for r in json.load(f)["records"]}

assert len(inv_records) == 286

reconciled_records = []
category_counts = defaultdict(int)
category_docs = defaultdict(set)

for r in inv_records:
    idx = r["index"]
    v = v_records[idx]
    g = g_records[idx]
    rst = r_records[idx]

    # Precedence rule:
    # 1. Full vector box detection (IoU >= 0.50)
    if v["detection_result"] == "VECTOR_BOX_DETECTED":
        primary_class = "VECTOR_BOX_DETECTED"
        detail = f"Vector drawing candidate matched with IoU={v['max_iou']:.4f} >= 0.50"
    # 2. Text-layer glyph detection (IoU >= 0.50)
    elif g["detection_result"] == "GLYPH_DETECTED":
        primary_class = "GLYPH_DETECTED"
        detail = f"Text glyph candidate matched with IoU={g['max_iou']:.4f} >= 0.50"
    # 3. Partial vector box (0.25 <= IoU < 0.50)
    elif v["detection_result"] == "VECTOR_PARTIAL":
        primary_class = "VECTOR_PARTIAL"
        detail = f"Partial vector drawing overlap with IoU={v['max_iou']:.4f}"
    # 4. Raster classification
    else:
        primary_class = rst["raster_classification"]
        detail = rst["metrics"]["reason"]

    rec_entry = {
        "index": idx,
        "document_id": r["document_id"],
        "field_path": r["field_path"],
        "value": r["value"],
        "gold_page": r["gold_page"],
        "gold_bbox": r["gold_bbox"],
        "primary_classification": primary_class,
        "classification_detail": detail,
        "vector_max_iou": v["max_iou"],
        "glyph_max_iou": g["max_iou"],
        "raster_darkness_ratio": rst["metrics"]["darkness_ratio"],
        "raster_outline_strength": rst["metrics"]["outline_strength"],
        "raster_interior_strength": rst["metrics"]["interior_strength"],
    }
    reconciled_records.append(rec_entry)
    category_counts[primary_class] += 1
    category_docs[primary_class].add(r["document_id"])

# Verify exact reconciliation
total_classified = sum(category_counts.values())
assert total_classified == 286, f"Reconciliation error: expected 286, got {total_classified}"

ordered_categories = [
    "VECTOR_BOX_DETECTED",
    "VECTOR_PARTIAL",
    "GLYPH_DETECTED",
    "RASTER_CHECKED",
    "RASTER_UNCHECKED",
    "RASTER_EMPTY",
    "RASTER_UNKNOWN",
]

summary_table = []
for rank, cat in enumerate(ordered_categories, 1):
    cnt = category_counts[cat]
    pct = round((cnt / 286) * 100, 2)
    summary_table.append({
        "rank": rank,
        "category": cat,
        "count": cnt,
        "documents_count": len(category_docs[cat]),
        "percentage": pct,
    })

out_json = repo_root / "research" / "experiments" / "EXP-036" / "classification_summary.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "experiment": "EXP-036_PHASE_0",
        "description": "Exhaustive Reconciled Physical Representation Summary for 286 Checkbox Fields",
        "total_fields": 286,
        "total_documents": len(set(r["document_id"] for r in inv_records)),
        "precedence_rule": "VECTOR_BOX_DETECTED > GLYPH_DETECTED > VECTOR_PARTIAL > RASTER_CHECKED > RASTER_UNCHECKED > RASTER_EMPTY > RASTER_UNKNOWN",
        "summary": summary_table,
        "deterministic_total": {
            "count": category_counts["VECTOR_BOX_DETECTED"] + category_counts["GLYPH_DETECTED"],
            "percentage": round(((category_counts["VECTOR_BOX_DETECTED"] + category_counts["GLYPH_DETECTED"]) / 286) * 100, 2),
        },
        "records": reconciled_records,
    }, f, indent=2)

print("=" * 80)
print("EXP-036 PHASE 0: FINAL RECONCILED REPRESENTATION DISTRIBUTION")
print("=" * 80)
print(f"{'Category':<24} | {'Count':<8} | {'Docs':<6} | {'Percentage':<12}")
print("-" * 80)
for row in summary_table:
    print(f"{row['category']:<24} | {row['count']:>8} | {row['documents_count']:<6} | {row['percentage']:>10.2f}%")
print("=" * 80)
print(f"Total Reconciled: {total_classified} / 286 (100.00%)")
print(f"Deterministic (Vector + Glyph): {category_counts['VECTOR_BOX_DETECTED'] + category_counts['GLYPH_DETECTED']} / 286 ({((category_counts['VECTOR_BOX_DETECTED'] + category_counts['GLYPH_DETECTED']) / 286) * 100:.2f}%)")
print(f"Saved to {out_json}")
