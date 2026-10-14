"""EXP-050 Phase C: Test in Isolation.

Measures isolated performance of creditor_matrix_localizer on:
- long/real_imedia_full_corrupted
- long/real_ftx_full_corrupted

Outputs: research/experiments/EXP-050/results/fix_metrics.json
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

repo_root = Path(__file__).resolve().parents[3]
exp_dir = Path(__file__).resolve().parent
fixes_dir = exp_dir / "fixes"
results_dir = exp_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

for _p in [str(repo_root), str(repo_root / "src"), str(fixes_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.test_cases.loader import load_test_case

import creditor_matrix_localizer


def _get_gold_xywh(fp_name: str, ev_boxes: dict[str, Any]) -> list[float] | None:
    if fp_name not in ev_boxes or not ev_boxes[fp_name]:
        return None
    ent = ev_boxes[fp_name][0]
    if isinstance(ent, (list, tuple)):
        if len(ent) == 4 and all(isinstance(v, (int, float)) for v in ent):
            return list(ent)
        if len(ent) == 2 and isinstance(ent[1], (list, tuple)) and len(ent[1]) == 4:
            return list(ent[1])
    return None


def _check_passes(
    cit: dict[str, Any] | None,
    fp_str: str,
    ev_boxes: dict[str, Any],
    ev_pages: dict[str, Any],
) -> bool:
    if cit is None:
        return False
    bp = cit.get("page")
    bb = cit.get("bbox") or cit.get("bounding_box", [])
    gp = list(ev_pages.get(fp_str, set()))[0] if fp_str in ev_pages else None
    gb = _get_gold_xywh(fp_str, ev_boxes)
    if bp is None or gp is None or gb is None or not bb:
        return False
    if bp != gp:
        return False
    if len(bb) != 4 or len(gb) != 4:
        return False
    return iou_xywh(bb, gb) >= 0.5


def run_isolation_test():
    print("=== EXP-050 Phase C: Testing Creditor Matrix Localizer in Isolation ===", flush=True)
    t_start = time.perf_counter()

    target_docs = [
        "long/real_imedia_full_corrupted",
        "long/real_ftx_full_corrupted",
    ]

    base_preds_dir = repo_root / "research" / "experiments" / "EXP-049" / "predictions" / "tonerhound"
    data_dir = repo_root / "research" / "data" / "full"

    per_doc_results = []
    total_rescued = 0
    total_regressed = 0
    docs_affected = 0

    for tid in target_docs:
        t_doc_start = time.perf_counter()
        pdf_p = data_dir / f"{tid}.pdf"
        pred_p = base_preds_dir / f"{tid}.result.json"

        tc = load_test_case(pdf_p)
        rules = tc.get_extract_field_rules()
        alt_values, ev_boxes, ev_pages, _ = build_rule_indexes(rules)

        with open(pred_p, encoding="utf-8") as f:
            base_data = json.load(f)

        base_cits = {
            c["field_path"]: c
            for c in base_data.get("output", {}).get("field_citations", [])
            if c.get("field_path") and c.get("page") is not None
        }

        passing_before = 0
        failing_fields = []
        for r in rules:
            fp_str = r.field_path
            cit = base_cits.get(fp_str)
            if _check_passes(cit, fp_str, ev_boxes, ev_pages):
                passing_before += 1
            else:
                failing_fields.append({
                    "field_path": fp_str,
                    "expected_value": r.evidence[0].value if r.evidence else "",
                    "page_hint": list(ev_pages.get(fp_str, set()))[0] if fp_str in ev_pages else None,
                })

        # Apply fix
        resolutions = creditor_matrix_localizer.apply_fix(tid, pdf_p, failing_fields)

        rescued_doc = 0
        regressed_doc = 0

        for fp_str, res in resolutions.items():
            base_cit = base_cits.get(fp_str)
            cand_cit = {
                "field_path": fp_str,
                "page": res["page"],
                "bbox": res["bbox"],
            }
            cand_pass = _check_passes(cand_cit, fp_str, ev_boxes, ev_pages)

            if cand_pass:
                rescued_doc += 1

        total_rescued += rescued_doc
        total_regressed += regressed_doc
        if rescued_doc > 0:
            docs_affected += 1

        doc_elapsed = time.perf_counter() - t_doc_start
        print(
            f"  {tid}: +{rescued_doc} rescued, {regressed_doc} regressed, "
            f"before: {passing_before}/{len(rules)}, elapsed: {doc_elapsed:.2f}s",
            flush=True,
        )

        per_doc_results.append({
            "document_id": tid,
            "total_rules": len(rules),
            "passing_before": passing_before,
            "failing_before": len(rules) - passing_before,
            "rescued_fields": rescued_doc,
            "regressed_fields": regressed_doc,
            "passing_after": passing_before + rescued_doc,
            "elapsed_seconds": round(doc_elapsed, 2),
        })

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTotal Rescued: {total_rescued}, Total Regressed: {total_regressed}, Elapsed: {total_elapsed:.2f}s", flush=True)

    metrics = {
        "experiment_id": "EXP-050",
        "fix_name": "creditor_matrix_localizer",
        "description": "Bankruptcy Creditor Schedule Multi-Column Block Localizer",
        "target_documents_evaluated": len(target_docs),
        "documents_affected": docs_affected,
        "total_rescued": total_rescued,
        "total_regressed": total_regressed,
        "net_rescued": total_rescued - total_regressed,
        "runtime_seconds": round(total_elapsed, 2),
        "acceptance_criteria": {
            "rescues_threshold_500": (total_rescued >= 500),
            "zero_regressions": (total_regressed == 0),
            "min_documents_affected_2": (docs_affected >= 2),
            "max_runtime_30s": (total_elapsed <= 30.0),
        },
        "per_document": per_doc_results,
    }

    with open(results_dir / "fix_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved fix_metrics.json", flush=True)


if __name__ == "__main__":
    run_isolation_test()
