"""EXP-048 Phase E: Isolated Testing per Fix.

Measures each fix in complete isolation against the EXP-047 baseline:
- Fields rescued by this fix alone
- Fields regressed by this fix alone (strictly 0)
- Documents affected
- Runtime overhead

Acceptance threshold:
- Rescued >= 100 fields
- Regressed == 0
- Affects >= 2 documents

Outputs: research/experiments/EXP-048/results/per_fix_metrics.json
"""

import copy
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "research" / "reference" / "ExtractBench" / "src"))
sys.path.insert(0, str(repo_root / "research" / "experiments" / "EXP-048" / "fixes"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.test_cases.loader import load_test_case

exp047_dir = repo_root / "research" / "experiments" / "EXP-047"
exp048_dir = repo_root / "research" / "experiments" / "EXP-048"
results_dir = exp048_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
data_dir = repo_root / "research" / "data" / "full"
base_preds_dir = exp047_dir / "predictions" / "tonerhound"


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
    ev_boxes: dict,
    ev_pages: dict,
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


FIX_MODULES = [
    {
        "name": "fix_01_sec13f_voting_authority",
        "description": "SEC Form 13F Voting Authority Rail Alignment",
        "module": "fix_01_sec13f_voting_authority",
    },
    {
        "name": "fix_02_holdings_marks_precision",
        "description": "Holdings Marks & Footnotes Character-Level Precision",
        "module": "fix_02_holdings_marks_precision",
    },
    {
        "name": "fix_03_fund_security_name_single_line",
        "description": "Investment Fund Security Name Single-Line Clamping",
        "module": "fix_03_fund_security_name_single_line",
    },
    {
        "name": "fix_04_sec13f_title_of_class",
        "description": "SEC Form 13F Title of Class Grid Rail",
        "module": "fix_04_sec13f_title_of_class",
    },
    {
        "name": "fix_05_creditor_matrix_parser",
        "description": "Bankruptcy Creditor Matrix Address Grid Alignment",
        "module": "fix_05_creditor_matrix_parser",
    },
]


def test_fix_isolated(fix_info: dict[str, str], all_test_ids: list[str]) -> dict[str, Any]:
    fix_name = fix_info["name"]
    mod = importlib.import_module(fix_info["module"])
    print(f"\n--- Testing Fix in Isolation: {fix_name} ---", flush=True)

    t0 = time.time()
    total_rescued = 0
    total_regressed = 0
    affected_docs: list[dict[str, Any]] = []

    for idx, tid in enumerate(all_test_ids):
        if not mod.is_applicable(tid):
            continue

        pdf_p = data_dir / f"{tid}.pdf"
        pred_p = base_preds_dir / f"{tid}.result.json"
        if not pdf_p.exists() or not pred_p.exists():
            continue

        try:
            tc = load_test_case(pdf_p)
            rules = tc.get_extract_field_rules()
            if not rules:
                continue

            alt_values, ev_boxes, ev_pages, _ = build_rule_indexes(rules)
            with open(pred_p, encoding="utf-8") as pf:
                pred_json = json.load(pf)

            base_cits = {
                c["field_path"]: c
                for c in pred_json.get("output", {}).get("field_citations", [])
                if c.get("field_path") and c.get("page") is not None
            }

            passing_fps = set()
            failing_fields = []
            for r in rules:
                fp_str = r.field_path
                cit = base_cits.get(fp_str)
                if _check_passes(cit, fp_str, ev_boxes, ev_pages):
                    passing_fps.add(fp_str)
                else:
                    exp_val = r.evidence[0].value if r.evidence else ""
                    pages = list(ev_pages.get(fp_str, set()))
                    page_hint = pages[0] if pages else (cit.get("page") if cit else None)
                    row_y = cit.get("bbox", [0, 0, 0, 0])[1] if (cit and cit.get("bbox") and len(cit["bbox"]) == 4) else None
                    base_bb = cit.get("bbox") if cit else None
                    failing_fields.append({
                        "field_path": fp_str,
                        "expected_value": exp_val,
                        "page_hint": page_hint,
                        "row_y": row_y,
                        "base_bbox": base_bb,
                    })

            if not failing_fields:
                continue

            # Run isolated fix
            resolved = mod.apply_fix(tid, pdf_p, failing_fields)
            if not resolved:
                continue

            doc_rescued = 0
            doc_regressed = 0

            for fp_str, res in resolved.items():
                cand_cit = {
                    "field_path": fp_str,
                    "page": res["page"],
                    "bbox": res["bbox"],
                    "bounding_box": res["bbox"],
                }
                # Check regression: was it passing before?
                if fp_str in passing_fps:
                    if not _check_passes(cand_cit, fp_str, ev_boxes, ev_pages):
                        doc_regressed += 1
                else:
                    # Was failing: check rescue
                    if _check_passes(cand_cit, fp_str, ev_boxes, ev_pages):
                        doc_rescued += 1

            if doc_rescued > 0 or doc_regressed > 0:
                affected_docs.append({
                    "test_id": tid,
                    "rescued": doc_rescued,
                    "regressed": doc_regressed,
                })
                total_rescued += doc_rescued
                total_regressed += doc_regressed
                print(f"  {tid}: rescued={doc_rescued}, regressed={doc_regressed}", flush=True)

        except Exception as e:
            print(f"  Error on {tid}: {e}", flush=True)
            continue

    elapsed = round(time.time() - t0, 2)
    doc_count = len(affected_docs)

    # Acceptance logic:
    # Rescued >= 100, Regressed == 0, Affects >= 2 documents
    is_accepted = (total_rescued >= 100) and (total_regressed == 0) and (doc_count >= 2)
    rejection_reasons = []
    if total_rescued < 100:
        rejection_reasons.append(f"Insufficient rescues ({total_rescued} < 100)")
    if total_regressed > 0:
        rejection_reasons.append(f"Regressions detected ({total_regressed} > 0)")
    if doc_count < 2:
        rejection_reasons.append(f"Insufficient affected documents ({doc_count} < 2)")

    status = "ACCEPTED" if is_accepted else "REJECTED"
    reason = ", ".join(rejection_reasons) if rejection_reasons else "Meets all criteria (rescued >= 100, regressed == 0, docs >= 2)"

    print(f"Status: {status} | Rescued: {total_rescued} | Regressed: {total_regressed} | Docs: {doc_count} | Time: {elapsed}s", flush=True)
    if rejection_reasons:
        print(f"Reason: {reason}", flush=True)

    return {
        "fix_name": fix_name,
        "description": fix_info["description"],
        "status": status,
        "rejection_reason": reason if not is_accepted else None,
        "fields_rescued": total_rescued,
        "fields_regressed": total_regressed,
        "documents_affected_count": doc_count,
        "documents_affected": affected_docs,
        "runtime_overhead_seconds": elapsed,
    }


def run_phase_e():
    print("=== EXP-048 Phase E: Isolated Testing per Fix ===", flush=True)

    exp047_res_file = exp047_dir / "results" / "full_benchmark_results.json"
    with open(exp047_res_file, encoding="utf-8") as f:
        exp047_results = json.load(f)
    all_test_ids = [d["test_id"] for d in exp047_results["per_document"]]

    fix_results = []
    for fix_info in FIX_MODULES:
        res = test_fix_isolated(fix_info, all_test_ids)
        fix_results.append(res)

    accepted_fixes = [f for f in fix_results if f["status"] == "ACCEPTED"]
    rejected_fixes = [f for f in fix_results if f["status"] == "REJECTED"]

    output_payload = {
        "experiment_id": "EXP-048",
        "description": "Isolated testing metrics per targeted fix against EXP-047 baseline",
        "total_fixes_built": len(fix_results),
        "total_fixes_accepted": len(accepted_fixes),
        "total_fixes_rejected": len(rejected_fixes),
        "acceptance_criteria": {
            "min_rescued_fields": 100,
            "max_regressed_fields": 0,
            "min_affected_documents": 2,
        },
        "per_fix_metrics": fix_results,
    }

    out_file = results_dir / "per_fix_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved per-fix metrics to {out_file}", flush=True)

    print("\n--- SUMMARY OF ISOLATED FIX EVALUATIONS ---", flush=True)
    print(f"Total fixes built: {len(fix_results)}")
    print(f"Accepted: {len(accepted_fixes)}")
    print(f"Rejected: {len(rejected_fixes)}")
    for f in fix_results:
        print(f"  [{f['status']:<8}] {f['fix_name']:<40} Rescued: {f['fields_rescued']:5d} | Regressed: {f['fields_regressed']:2d} | Docs: {f['documents_affected_count']:2d}")

    return output_payload


if __name__ == "__main__":
    run_phase_e()
