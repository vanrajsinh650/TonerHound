"""EXP-035: Forensic extraction of the 571 alleged INDEXING_MISS cases.

Reproduces the exact gap classification from EXP-034R and extracts every individual
field classified as INDEXING_MISS with full metadata.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from extract_bench.test_cases.loader import load_test_case
def classify_gap_field(
    fpath: str,
    val: Any,
    gp: int,
    gb: list[float],
    cur_cit: dict[str, Any] | None,
    exp_cit: dict[str, Any],
    doc_id: str,
) -> str:
    """Classify why current pool oracle failed to achieve IoU >= 0.50."""
    is_tbl = ("[" in fpath and "]" in fpath)

    # 1. OCR Coverage / Geometry in corrupted documents
    if "corrupted" in doc_id:
        if exp_cit.get("source") == "true_text_oracle" or "ocr" in str(exp_cit.get("source", "")).lower():
            return "OCR_COVERAGE"
        return "OCR_GEOMETRY"

    # 2. No citation at all in current pool
    if not cur_cit or not cur_cit.get("bbox") or cur_cit.get("page") is None:
        if "\n" in str(val):
            return "MULTI_LINE_SPLIT"
        if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(r"\d", val)):
            if is_tbl:
                return "TOP_K_TRUNCATION"
            return "INDEXING_MISS"
        if is_tbl:
            return "DEDUPLICATION_COLLAPSE"
        return "GLOBAL_ROUTING"

    cp = cur_cit["page"]
    cb = cur_cit["bbox"]

    # 3. Wrong page
    if cp != gp:
        if abs(cp - gp) <= 1:
            return "PAGE_PRUNING"
        return "WRONG_PAGE"

    # 4. Same page, check spatial alignment
    cur_iou = iou_xywh(cb, gb)

    if is_tbl:
        cy_cur = cb[1] + cb[3] / 2.0
        cy_gold = gb[1] + gb[3] / 2.0
        if abs(cy_cur - cy_gold) > gb[3] * 0.8:
            return "WRONG_ROW"

        cx_cur = cb[0] + cb[2] / 2.0
        cx_gold = gb[0] + gb[2] / 2.0
        if abs(cx_cur - cx_gold) > gb[2] * 1.5:
            return "WRONG_COLUMN"

    w_ratio = cb[2] / max(1e-4, gb[2])
    h_ratio = cb[3] / max(1e-4, gb[3])

    if w_ratio > 1.5 or h_ratio > 1.5:
        return "BBOX_TOO_WIDE"
    if w_ratio < 0.65 or h_ratio < 0.65:
        if isinstance(val, str) and any(ch in val for ch in ("$", "%", "'", "\"", "-", ",")):
            return "TOKEN_SLICING"
        return "BBOX_TOO_NARROW"

    if "\n" in str(val):
        return "MULTI_LINE_SPLIT"

    if isinstance(val, str) and any(ch in val for ch in ("$", "%", "'", "\"", "-", ",")):
        return "TOKEN_SLICING"

    return "OTHER"


def extract_alleged_indexing_misses():
    data_dir = repo_root / "research" / "data" / "full"
    pred_dir_current = repo_root / "research" / "experiments" / "EXP-032" / "oracle_predictions" / "mode_a" / "tonerhound"
    pred_dir_expanded = repo_root / "research" / "experiments" / "EXP-034R" / "oracle_predictions" / "mode_a"

    pred_files_current = sorted(list(pred_dir_current.rglob("*.result.json")))
    print(f"Comparing {len(pred_files_current)} documents...")

    alleged_cases = []
    docs_with_miss = set()
    fields_per_doc = defaultdict(int)

    for idx, rf in enumerate(pred_files_current, 1):
        rel = rf.relative_to(pred_dir_current)
        tid = rel.as_posix().removesuffix(".result.json")
        pdf_path = data_dir / f"{tid}.pdf"
        rf_exp = pred_dir_expanded / rel

        if not rf_exp.exists() or not pdf_path.exists():
            continue

        with open(rf, encoding="utf-8") as f:
            data_cur = json.load(f)
        with open(rf_exp, encoding="utf-8") as f:
            data_exp = json.load(f)

        cits_cur = {c["field_path"]: c for c in data_cur.get("output", {}).get("field_citations", []) if c.get("field_path")}
        cits_exp = {c["field_path"]: c for c in data_exp.get("output", {}).get("field_citations", []) if c.get("field_path")}

        tc = load_test_case(pdf_path)
        rules = tc.get_extract_field_rules()

        for r in rules:
            if not r.evidence or r.evidence[0].page is None or r.evidence[0].bbox is None:
                continue

            fpath = r.field_path
            gp = r.evidence[0].page
            gb = r.evidence[0].bbox
            val = r.evidence[0].value

            c_cur = cits_cur.get(fpath)
            c_exp = cits_exp.get(fpath)

            cur_pass = bool(c_cur and c_cur.get("page") == gp and c_cur.get("bbox") and iou_xywh(c_cur["bbox"], gb) >= 0.50)
            exp_pass = bool(c_exp and c_exp.get("page") == gp and c_exp.get("bbox") and iou_xywh(c_exp["bbox"], gb) >= 0.50)

            if not cur_pass and exp_pass:
                f_class = classify_gap_field(fpath, val, gp, gb, c_cur, c_exp, tid)
                if f_class == "INDEXING_MISS":
                    cur_iou = iou_xywh(c_cur["bbox"], gb) if c_cur and c_cur.get("bbox") else 0.0
                    exp_iou = iou_xywh(c_exp["bbox"], gb) if c_exp and c_exp.get("bbox") else 0.0
                    
                    record = {
                        "document_id": tid,
                        "field_path": fpath,
                        "value": val,
                        "gold_page": gp,
                        "gold_bbox": [round(x, 4) for x in gb],
                        "gold_text": r.evidence[0].text if hasattr(r.evidence[0], "text") else None,
                        "cur_citation": c_cur,
                        "exp_citation": c_exp,
                        "cur_iou": round(cur_iou, 4),
                        "exp_iou": round(exp_iou, 4),
                    }
                    alleged_cases.append(record)
                    docs_with_miss.add(tid)
                    fields_per_doc[tid] += 1

    out_file = repo_root / "research" / "experiments" / "EXP-035" / "alleged_indexing_miss_fields.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_alleged_fields": len(alleged_cases),
            "unique_alleged_documents": len(docs_with_miss),
            "distribution": dict(sorted(fields_per_doc.items(), key=lambda x: x[1], reverse=True)),
            "cases": alleged_cases,
        }, f, indent=2)

    print(f"Extracted {len(alleged_cases)} alleged INDEXING_MISS fields across {len(docs_with_miss)} documents.")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    extract_alleged_indexing_misses()
