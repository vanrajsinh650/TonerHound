"""EXP-049 Phase C: Test Fix in Isolation.

Measures the IRS line grid localizer in complete isolation against EXP-048 baseline:
- Fields rescued by this fix alone
- Fields regressed by this fix alone (strictly 0)
- Documents affected
- Runtime overhead

Acceptance criteria:
- >= 300 total rescues across the family (or report actual and evaluate)
- 0 regressions
- Affects >= 5 documents
- Overhead <= 5 seconds across the corpus

Outputs: research/experiments/EXP-049/results/fix_metrics.json
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

sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "research" / "reference" / "ExtractBench" / "src"))
sys.path.insert(0, str(fixes_dir))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.test_cases.loader import load_test_case
from tonerhound.document.index import DocumentIndex

import irs_line_grid


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


def run_phase_c():
    print("=== EXP-049 Phase C: Testing Fix in Isolation ===", flush=True)
    t_start = time.perf_counter()

    data_dir = repo_root / "research" / "data" / "full"
    base_preds_dir = repo_root / "research" / "experiments" / "EXP-048" / "predictions" / "tonerhound"

    with open(exp_dir / "audit" / "target_documents.json", encoding="utf-8") as f:
        target_info = json.load(f)
    target_docs = target_info["documents"]

    total_rescued = 0
    total_regressed = 0
    per_doc_metrics = []

    for item in target_docs:
        doc_id = item["document_id"]
        pdf_path = data_dir / f"{doc_id}.pdf"
        pred_path = base_preds_dir / f"{doc_id}.result.json"

        if not pdf_path.exists() or not pred_path.exists():
            continue

        try:
            tc = load_test_case(pdf_path)
            rules = tc.get_extract_field_rules()
        except Exception:
            continue

        alt_values, ev_boxes, ev_pages, _ = build_rule_indexes(rules)
        with open(pred_path, encoding="utf-8") as pf:
            base_data = json.load(pf)

        base_cits = {
            c["field_path"]: c
            for c in base_data.get("output", {}).get("field_citations", [])
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

        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="pdfium")
        resolved = irs_line_grid.apply_fix(doc_id, pdf_path, failing_fields, doc_index=doc_index)

        doc_rescued = 0
        doc_regressed = 0

        for fp_str, res in resolved.items():
            cand_cit = {
                "field_path": fp_str,
                "page": res["page"],
                "bbox": res["bbox"],
                "bounding_box": res["bbox"],
            }
            cand_passes = _check_passes(cand_cit, fp_str, ev_boxes, ev_pages)

            if fp_str in passing_fps:
                if not cand_passes:
                    doc_regressed += 1
            else:
                if cand_passes:
                    doc_rescued += 1

        total_rescued += doc_rescued
        total_regressed += doc_regressed

        per_doc_metrics.append({
            "document_id": doc_id,
            "word_f1_baseline": item["word_f1"],
            "page_f1_baseline": item["page_f1"],
            "gradeable_fields": item["gradeable_fields"],
            "passing_baseline": item["passing_fields"],
            "failing_baseline": item["failing_fields"],
            "rescued_fields": doc_rescued,
            "regressed_fields": doc_regressed,
            "net_rescued": doc_rescued - doc_regressed,
        })
        print(f"  {doc_id:25}: +{doc_rescued} rescued, -{doc_regressed} regressed", flush=True)

    runtime_seconds = round(time.perf_counter() - t_start, 2)
    docs_affected = [m for m in per_doc_metrics if m["rescued_fields"] > 0]

    fix_metrics = {
        "fix_name": "irs_line_grid",
        "description": "IRS Form Line-Item Grid Localizer",
        "target_documents_evaluated": len(target_docs),
        "documents_affected": len(docs_affected),
        "total_rescued": total_rescued,
        "total_regressed": total_regressed,
        "net_rescued": total_rescued - total_regressed,
        "runtime_seconds": runtime_seconds,
        "acceptance_criteria": {
            "rescues_threshold_300": total_rescued >= 300,
            "zero_regressions": total_regressed == 0,
            "min_documents_affected_5": len(docs_affected) >= 5,
            "max_runtime_5s": runtime_seconds <= 5.0,
        },
        "per_document": per_doc_metrics,
    }

    with open(results_dir / "fix_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fix_metrics, f, indent=2)

    print("\nPhase C Complete.")
    print(f"Total Rescued: {total_rescued}")
    print(f"Total Regressed: {total_regressed}")
    print(f"Documents Affected: {len(docs_affected)} / {len(target_docs)}")
    print(f"Runtime: {runtime_seconds}s")
    print(f"Saved {results_dir / 'fix_metrics.json'}")


if __name__ == "__main__":
    run_phase_c()
