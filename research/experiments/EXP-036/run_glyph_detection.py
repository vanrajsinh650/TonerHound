"""EXP-036 Phase 0: Step 2 - Text-Layer Glyph Detection.

Inspects page.get_text("rawdict") for unicode box glyphs and bracket-like forms:
- Unicode box glyphs: ☐, ☑, ☒, ❑, ❒, ■, □, ▢, ✓, ✔, ✗, ✘
- Bracket pairs: [ ], [X], [x], [ X ], ( ), (X), (x)
Computes geometric IoU against gold bounding boxes and classifies into:
- GLYPH_DETECTED (IoU >= 0.50)
- NO_GLYPH_DETECTION (IoU < 0.50)
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pymupdf as fitz

repo_root = Path(__file__).resolve().parent.parent.parent.parent

CHECKBOX_UNICODE_GLYPHS = frozenset({
    "\u2610", "\u2611", "\u2612",  # ☐, ☑, ☒
    "\u2751", "\u2752",            # ❑, ❒
    "\u25a0", "\u25a1", "\u25a2",  # ■, □, ▢
    "\u2713", "\u2714",            # ✓, ✔
    "\u2717", "\u2718",            # ✗, ✘
})

BRACKET_REGEX = re.compile(r"(\[\s*[xX\u2713\u2714]?\s*\]|\(\s*[xX\u2713\u2714]?\s*\))")


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


def extract_page_glyph_candidates(page: fitz.Page) -> list[dict[str, Any]]:
    """Extract all candidate checkbox glyphs and bracket pairs from a page."""
    pw, ph = page.rect.width, page.rect.height
    if pw <= 0 or ph <= 0:
        return []

    candidates: list[dict[str, Any]] = []
    raw = page.get_text("rawdict")

    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0 is text block
            continue

        for line in block.get("lines", []):
            line_bbox = line.get("bbox")
            line_h = (line_bbox[3] - line_bbox[1]) if line_bbox else 10.0

            # 1. Search individual characters for unicode box glyphs
            line_chars = []
            for span in line.get("spans", []):
                span_text = span.get("text", "")

                # Check span-level regex for bracket pairs
                for m in BRACKET_REGEX.finditer(span_text):
                    # Find characters corresponding to match
                    start_char_idx = m.start()
                    end_char_idx = m.end()
                    chars_in_span = span.get("chars", [])
                    if chars_in_span and end_char_idx <= len(chars_in_span):
                        matched_chars = chars_in_span[start_char_idx:end_char_idx]
                        if matched_chars:
                            min_x = min(c["bbox"][0] for c in matched_chars)
                            min_y = min(c["bbox"][1] for c in matched_chars)
                            max_x = max(c["bbox"][2] for c in matched_chars)
                            max_y = max(c["bbox"][3] for c in matched_chars)
                            candidates.append({
                                "text": m.group(0),
                                "type": "bracket_pair",
                                "bbox": [min_x / pw, min_y / ph, (max_x - min_x) / pw, (max_y - min_y) / ph],
                            })

                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    cb = ch.get("bbox")
                    if not cb:
                        continue
                    line_chars.append(ch)

                    if c in CHECKBOX_UNICODE_GLYPHS:
                        candidates.append({
                            "text": c,
                            "type": "unicode_glyph",
                            "bbox": [cb[0] / pw, cb[1] / ph, (cb[2] - cb[0]) / pw, (cb[3] - cb[1]) / ph],
                        })

            # 2. Check for bracket pairs formed across span boundaries on the same line
            for i, ch1 in enumerate(line_chars):
                if ch1.get("c") in ("[", "("):
                    open_c = ch1["c"]
                    close_c = "]" if open_c == "[" else ")"
                    for j in range(i + 1, min(i + 5, len(line_chars))):
                        ch2 = line_chars[j]
                        if ch2.get("c") == close_c:
                            # Verify spatial distance
                            dist = ch2["bbox"][0] - ch1["bbox"][2]
                            if 0 <= dist <= max(15.0, line_h * 2.0):
                                min_x = ch1["bbox"][0]
                                min_y = min(ch1["bbox"][1], ch2["bbox"][1])
                                max_x = ch2["bbox"][2]
                                max_y = max(ch1["bbox"][3], ch2["bbox"][3])
                                mid_chars = "".join(line_chars[k]["c"] for k in range(i, j + 1))
                                candidates.append({
                                    "text": mid_chars,
                                    "type": "bracket_pair_cross_span",
                                    "bbox": [min_x / pw, min_y / ph, (max_x - min_x) / pw, (max_y - min_y) / ph],
                                })
                            break

    return candidates


def run_glyph_detection():
    inv_path = repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json"
    with open(inv_path, encoding="utf-8") as f:
        inv_data = json.load(f)

    records = inv_data["inventory"]
    print(f"Loaded {len(records)} records from checkbox inventory.")

    docs_map = defaultdict(list)
    for r in records:
        docs_map[r["document_id"]].append(r)

    results = []
    category_counts = defaultdict(int)

    t0 = time.perf_counter()

    for doc_id, doc_records in docs_map.items():
        pdf_path = repo_root / "research" / "data" / "full" / f"{doc_id}.pdf"
        doc = fitz.open(pdf_path)

        page_glyphs_cache = {}

        for r in doc_records:
            gp = r["gold_page"]
            gb = r["gold_bbox"]

            if gp not in page_glyphs_cache:
                if gp <= len(doc):
                    page = doc[gp - 1]
                    page_glyphs_cache[gp] = extract_page_glyph_candidates(page)
                else:
                    page_glyphs_cache[gp] = []

            page_candidates = page_glyphs_cache[gp]

            matching_cands = []
            max_iou = 0.0

            for cand in page_candidates:
                iou = iou_xywh(cand["bbox"], gb)
                if iou > max_iou:
                    max_iou = iou

                if iou >= 0.10 or (
                    cand["bbox"][0] < gb[0] + gb[2] and cand["bbox"][0] + cand["bbox"][2] > gb[0] and
                    cand["bbox"][1] < gb[1] + gb[3] and cand["bbox"][1] + cand["bbox"][3] > gb[1]
                ):
                    matching_cands.append({
                        "text": cand["text"],
                        "type": cand["type"],
                        "bbox": [round(x, 6) for x in cand["bbox"]],
                        "iou": round(iou, 4),
                    })

            if max_iou >= 0.50:
                det_result = "GLYPH_DETECTED"
            elif max_iou >= 0.25:
                det_result = "GLYPH_PARTIAL"
            else:
                det_result = "NO_GLYPH_DETECTION"

            record_out = {
                "index": r["index"],
                "document_id": doc_id,
                "field_path": r["field_path"],
                "value": r["value"],
                "gold_page": gp,
                "gold_bbox": gb,
                "detection_result": det_result,
                "max_iou": round(max_iou, 4),
                "candidates_count": len(matching_cands),
                "candidates": sorted(matching_cands, key=lambda c: c["iou"], reverse=True)[:5],
            }
            results.append(record_out)
            category_counts[det_result] += 1

        doc.close()

    elapsed = time.perf_counter() - t0

    out_json = repo_root / "research" / "experiments" / "EXP-036" / "glyph_detection.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "EXP-036_PHASE_0",
            "total_records": len(results),
            "summary": {
                k: {
                    "count": category_counts[k],
                    "percentage": round((category_counts[k] / len(results)) * 100, 2),
                }
                for k in ("GLYPH_DETECTED", "GLYPH_PARTIAL", "NO_GLYPH_DETECTION")
            },
            "records": results,
        }, f, indent=2)

    print("=" * 70)
    print("EXP-036 PHASE 0: STEP 2 — TEXT-LAYER GLYPH DETECTION RESULTS")
    print("=" * 70)
    for k in ("GLYPH_DETECTED", "GLYPH_PARTIAL", "NO_GLYPH_DETECTION"):
        count = category_counts[k]
        pct = (count / len(results)) * 100
        print(f"  {k:<24}: {count:>4} / {len(results)} ({pct:>6.2f}%)")
    print(f"Elapsed: {elapsed:.2f}s | Saved to {out_json}")


if __name__ == "__main__":
    run_glyph_detection()
