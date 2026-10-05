"""EXP-034R: Gap Characterization and Macro-Weighted Contribution Analysis.

Analyzes every field where Current-Pool Oracle FAILs but Expanded Text/OCR Oracle PASSes.
Quantifies the exact failure class distribution and computes the official macro-weighted
Word Grounding F1 contribution for each failure class.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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
        # Check vertical / row alignment
        cy_cur = cb[1] + cb[3] / 2.0
        cy_gold = gb[1] + gb[3] / 2.0
        if abs(cy_cur - cy_gold) > gb[3] * 0.8:
            return "WRONG_ROW"

        # Check horizontal / column alignment
        cx_cur = cb[0] + cb[2] / 2.0
        cx_gold = gb[0] + gb[2] / 2.0
        if abs(cx_cur - cx_gold) > gb[2] * 1.5:
            return "WRONG_COLUMN"

    # Bbox aspect ratio and boundary issues
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


def run_gap_characterization() -> dict[str, Any]:
    print("=" * 80)
    print("EXP-034R: GAP CHARACTERIZATION & MACRO-WEIGHTED CONTRIBUTION")
    print("=" * 80)

    t0 = time.perf_counter()

    data_dir = repo_root / "research" / "data" / "full"
    pred_dir_current = repo_root / "research" / "experiments" / "EXP-032" / "oracle_predictions" / "mode_a" / "tonerhound"
    pred_dir_expanded = repo_root / "research" / "experiments" / "EXP-034R" / "oracle_predictions" / "mode_a"

    out_macro_json = repo_root / "research" / "experiments" / "EXP-034R" / "macro_weighted_contribution.json"
    out_samples_json = repo_root / "research" / "experiments" / "EXP-034R" / "gap_samples.json"

    pred_files_current = sorted(list(pred_dir_current.rglob("*.result.json")))
    print(f"Loaded {len(pred_files_current)} prediction files for comparison.")

    # Tracking data structures
    all_gap_fields: list[dict[str, Any]] = []
    class_field_counts: dict[str, int] = defaultdict(int)
    class_doc_counts: dict[str, set[str]] = defaultdict(set)

    # Document-level gain tracking: doc_id -> class -> delta_correct
    doc_class_gains: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    doc_total_gap: dict[str, int] = defaultdict(int)
    doc_eval_stats: dict[str, dict[str, float]] = {}

    # Load baseline evaluation caches for denominators (g_claims + g_expected)
    eval_cache_base = repo_root / "research" / "experiments" / "EXP-032" / "eval_cache" / "baseline"

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

        # Cache file for counts
        cache_f = eval_cache_base / f"{tid}.eval.json"
        g_claims = 0
        g_expected = 0
        if cache_f.exists():
            try:
                with open(cache_f, encoding="utf-8") as f:
                    cdata = json.load(f)
                if "word_f1" in cdata:
                    f1_val = cdata["word_f1"]
                else:
                    raw_m = {m["metric_name"]: m["value"] for m in cdata.get("metrics", [])}
                    f1_val = raw_m.get("extract_unified_grounded_f1")
                # If ungradeable, skip
                if f1_val is None:
                    continue
            except Exception:
                pass

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
                # GAP FIELD DETECTED
                f_class = classify_gap_field(fpath, val, gp, gb, c_cur, c_exp, tid)
                cur_iou = iou_xywh(c_cur["bbox"], gb) if c_cur and c_cur.get("bbox") else 0.0
                exp_iou = iou_xywh(c_exp["bbox"], gb) if c_exp and c_exp.get("bbox") else 0.0

                record = {
                    "document_id": tid,
                    "field_path": fpath,
                    "value": str(val)[:60] if val is not None else "",
                    "gold_page": gp,
                    "gold_bbox": [round(x, 4) for x in gb],
                    "cur_page": c_cur.get("page") if c_cur else None,
                    "cur_bbox": [round(x, 4) for x in c_cur["bbox"]] if c_cur and c_cur.get("bbox") else None,
                    "exp_page": c_exp.get("page") if c_exp else None,
                    "exp_bbox": [round(x, 4) for x in c_exp["bbox"]] if c_exp and c_exp.get("bbox") else None,
                    "cur_iou": round(cur_iou, 4),
                    "exp_iou": round(exp_iou, 4),
                    "failure_class": f_class,
                }
                all_gap_fields.append(record)
                class_field_counts[f_class] += 1
                class_doc_counts[f_class].add(tid)
                doc_class_gains[tid][f_class] += 1
                doc_total_gap[tid] += 1

        if idx % 50 == 0 or idx == len(pred_files_current):
            print(f"[{idx}/{len(pred_files_current)}] Processed (cumulative gap fields: {len(all_gap_fields):,})...")

    print(f"\nTotal GAP fields identified across benchmark: {len(all_gap_fields):,}")

    # Load evaluated oracle results for exact document F1 deltas
    eval_cache_cur = repo_root / "research" / "experiments" / "EXP-032" / "eval_cache" / "mode_a"
    eval_cache_exp = repo_root / "research" / "experiments" / "EXP-028B0" / "eval_cache"

    doc_f1_deltas: dict[str, float] = {}
    grounded_doc_count = 0

    for rf in pred_files_current:
        rel = rf.relative_to(pred_dir_current)
        tid = rel.as_posix().removesuffix(".result.json")
        f_cur = eval_cache_cur / f"{tid}.eval.json"
        f_exp = eval_cache_exp / f"{tid}.eval.json"

        if f_cur.exists() and f_exp.exists():
            with open(f_cur, encoding="utf-8") as fp:
                dc = json.load(fp)
            with open(f_exp, encoding="utf-8") as fp:
                de = json.load(fp)

            if "word_f1" in dc:
                f1_c = dc["word_f1"]
            else:
                mc = {m["metric_name"]: m["value"] for m in dc.get("metrics", [])}
                f1_c = mc.get("extract_unified_grounded_f1")

            if "word_f1" in de:
                f1_e = de["word_f1"]
            else:
                me = {m["metric_name"]: m["value"] for m in de.get("metrics", [])}
                f1_e = me.get("extract_unified_grounded_f1")

            if f1_c is not None and f1_e is not None:
                grounded_doc_count += 1
                doc_f1_deltas[tid] = max(0.0, (f1_e - f1_c) * 100)

    print(f"Grounded benchmark documents with F1 delta: {len(doc_f1_deltas)} (expected 236).")

    # Compute macro-weighted contribution for each class
    total_gap_pp = 75.1243 - 60.9591  # 14.1652 pp
    class_macro_pp: dict[str, float] = defaultdict(float)

    for tid, delta_pp in doc_f1_deltas.items():
        tot_gap_doc = doc_total_gap.get(tid, 0)
        if tot_gap_doc <= 0:
            continue
        # Allocate document's F1 delta proportionally to the failure classes in that document
        for f_class, cnt in doc_class_gains[tid].items():
            class_share_in_doc = cnt / tot_gap_doc
            class_doc_contribution = (delta_pp * class_share_in_doc) / grounded_doc_count
            class_macro_pp[f_class] += class_doc_contribution

    # Compile ranked results
    ranked_classes = sorted(class_macro_pp.keys(), key=lambda c: class_macro_pp[c], reverse=True)

    summary_table = []
    for rank, fc in enumerate(ranked_classes, 1):
        pp = round(class_macro_pp[fc], 4)
        pct_explained = round((pp / total_gap_pp) * 100, 2)
        summary_table.append({
            "rank": rank,
            "failure_class": fc,
            "fields_affected": class_field_counts[fc],
            "documents_affected": len(class_doc_counts[fc]),
            "macro_weighted_f1_contribution_pp": pp,
            "percentage_of_gap_explained": pct_explained,
        })

    # Sample 200 fields deterministically (seed = 42)
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_gap_fields), size=min(200, len(all_gap_fields)), replace=False)
    sample_records = [all_gap_fields[i] for i in sorted(sample_indices)]

    # Save outputs
    out_macro_json.parent.mkdir(parents=True, exist_ok=True)
    final_output = {
        "experiment": "EXP-034R",
        "description": "Macro-Weighted Contribution Analysis of the 14.1652 pp Candidate-Quality Gap",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_gap_fields": len(all_gap_fields),
        "total_candidate_quality_gap_pp": round(total_gap_pp, 4),
        "grounded_benchmark_documents": grounded_doc_count,
        "ranked_failure_classes": summary_table,
    }

    with open(out_macro_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    with open(out_samples_json, "w", encoding="utf-8") as f:
        json.dump(sample_records, f, indent=2)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 95)
    print(f"{'Rank':<5} | {'Failure Class':<24} | {'Fields':<8} | {'Docs':<6} | {'Macro F1 (pp)':<14} | {'% of Gap':<10}")
    print("-" * 95)
    for row in summary_table:
        print(f"{row['rank']:<5} | {row['failure_class']:<24} | {row['fields_affected']:>8,} | {row['documents_affected']:<6} | +{row['macro_weighted_f1_contribution_pp']:<13.4f} | {row['percentage_of_gap_explained']:>6.2f}%")
    print("=" * 95)
    print(f"Elapsed: {elapsed:.2f}s | Saved to {out_macro_json} and {out_samples_json}")

    return final_output


if __name__ == "__main__":
    run_gap_characterization()
