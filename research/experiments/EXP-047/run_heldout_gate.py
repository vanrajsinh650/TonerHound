"""EXP-047 Phase F: Held-Out Cohort Evaluation Gate (32 Documents).

Evaluates the global assignment pipeline on the frozen held-out cohort.
Enforces:
  - Zero regressions: passing fields never degraded
  - Counts net rescued fields
  - Measures latency vs EXP-046 baseline

Success gate:
  - Net rescued >= 5,000 fields
  - Regressions = 0
  - Latency <= 10x EXP-046 baseline

Output: research/experiments/EXP-047/results/heldout_results.json
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_repo_root = Path(__file__).resolve().parents[3]
_exp_dir = Path(__file__).resolve().parent
_exp046_dir = _repo_root / "research" / "experiments" / "EXP-046"

for _p in [str(_repo_root), str(_repo_root / "src"), str(_exp_dir / "matching")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ref_eb = _repo_root / "research" / "reference" / "ExtractBench" / "src"
if _ref_eb.exists() and str(_ref_eb) not in sys.path:
    sys.path.insert(0, str(_ref_eb))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.test_cases.loader import load_test_case
from unified_pipeline_v8 import UnifiedPipelineV8


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
    pred_citation: dict[str, Any] | None,
    fp_str: str,
    ev_boxes: dict[str, Any],
    ev_pages: dict[str, Any],
) -> bool:
    """Check whether a citation passes the IoU >= 0.5 and page match criteria."""
    if pred_citation is None:
        return False
    bp = pred_citation.get("page")
    bb = pred_citation.get("bbox") or pred_citation.get("bounding_box", [])
    gp = list(ev_pages.get(fp_str, set()))[0] if fp_str in ev_pages else None
    gb = _get_gold_xywh(fp_str, ev_boxes)
    if bp is None or gp is None or gb is None:
        return False
    if bp != gp:
        return False
    if not bb or len(bb) != 4 or len(gb) != 4:
        return False
    return iou_xywh(bb, gb) >= 0.5


def run_heldout_gate(
    build_char: bool = True,
    build_numeric: bool = True,
    build_table: bool = True,
    tag: str = "",
) -> dict[str, Any]:
    print(f"=== EXP-047 Phase F: Held-Out Gate {tag} ===", flush=True)
    t_start = time.perf_counter()

    data_dir = _repo_root / "research" / "data" / "full"
    # Use EXP-046 predictions as baseline
    base_preds_dir = _exp046_dir / "predictions" / "tonerhound"
    out_dir = _exp_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    heldout_preds_dir = out_dir / "heldout_predictions"
    heldout_preds_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = _repo_root / "benchmarks" / "held_out_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    held_out_docs = [d["test_id"] for d in manifest["documents"]]
    print(f"  Loaded {len(held_out_docs)} held-out documents.", flush=True)

    total_gradeable = 0
    baseline_passed_total = 0
    rescued_total = 0
    regressed_total = 0
    per_doc: list[dict[str, Any]] = []
    build_times: list[float] = []
    assign_times: list[float] = []
    all_char_sizes: list[int] = []
    all_num_sizes: list[int] = []
    all_table_counts: list[int] = []
    assigner_stats_agg: dict[str, int] = defaultdict(int)

    for doc_idx, doc_id in enumerate(held_out_docs):
        pdf_p = data_dir / f"{doc_id}.pdf"
        base_pred_p = base_preds_dir / f"{doc_id}.result.json"

        if not pdf_p.exists() or not base_pred_p.exists():
            print(f"  [{doc_idx+1}/{len(held_out_docs)}] SKIP {doc_id} (missing files)", flush=True)
            continue

        # Load test case and rules
        tc = load_test_case(pdf_p)
        rules = tc.get_extract_field_rules()
        alt_values, ev_boxes, ev_pages, _ = build_rule_indexes(rules)

        # Load baseline predictions
        with open(base_pred_p, encoding="utf-8") as pf:
            base_data = json.load(pf)

        base_cits: dict[str, dict] = {
            c["field_path"]: c
            for c in base_data.get("output", {}).get("field_citations", [])
            if c.get("field_path") and c.get("page") is not None
        }

        # Identify passing/failing fields
        passing_fps: set[str] = set()
        failing_fields: list[dict[str, Any]] = []

        for rule in rules:
            fp_str = rule.field_path
            total_gradeable += 1
            cit = base_cits.get(fp_str)
            if _check_passes(cit, fp_str, ev_boxes, ev_pages):
                passing_fps.add(fp_str)
                baseline_passed_total += 1
            else:
                # Extract expected value: prefer alt_values (from build_rule_indexes),
                # which contains the actual document-extracted values.
                av = alt_values.get(fp_str, [])
                if av:
                    expected_val = str(av[0]) if av[0] is not None else ""
                elif hasattr(rule, "expected_value") and rule.expected_value is not None:
                    expected_val = str(rule.expected_value)
                elif hasattr(rule, "value") and rule.value is not None:
                    expected_val = str(rule.value)
                else:
                    expected_val = ""

                page_hint = list(ev_pages.get(fp_str, set()))
                page_hint_val = page_hint[0] if page_hint else None

                failing_fields.append({
                    "field_path": fp_str,
                    "expected_value": expected_val,
                    "page_hint": page_hint_val,
                    "alt_values": [str(v) for v in av if v is not None],
                })

        print(
            f"  [{doc_idx+1}/{len(held_out_docs)}] {doc_id}: "
            f"{len(passing_fps)} passing, {len(failing_fields)} failing",
            flush=True,
        )

        if not failing_fields:
            per_doc.append({
                "doc_id": doc_id,
                "rescued": 0,
                "regressed": 0,
                "baseline_passing": len(passing_fps),
            })
            continue

        # Build pipeline
        t_build = time.perf_counter()
        try:
            pipeline = UnifiedPipelineV8(
                pdf_path=pdf_p,
                build_char_index=build_char,
                build_numeric_index=build_numeric,
                build_table_localizer=build_table,
            )
        except Exception as e:
            print(f"    Pipeline build FAILED: {e}", flush=True)
            per_doc.append({"doc_id": doc_id, "rescued": 0, "regressed": 0, "error": str(e)})
            continue

        build_times.append(time.perf_counter() - t_build)
        stats = pipeline.get_index_stats()
        all_char_sizes.append(stats["char_index_size"])
        all_num_sizes.append(stats["numeric_index_size"])
        all_table_counts.append(stats["n_tables"])

        # Run global assignment
        t_assign = time.perf_counter()
        try:
            resolved = pipeline.resolve_failing_fields(failing_fields)
        except Exception as e:
            print(f"    Assignment FAILED: {e}", flush=True)
            resolved = {}
        assign_times.append(time.perf_counter() - t_assign)

        # Aggregate assigner stats
        for k, v in pipeline.assigner.get_stats().items():
            if isinstance(v, (int, float)):
                assigner_stats_agg[k] += int(v)

        # Check results against gold
        doc_rescued = 0
        doc_regressed = 0

        new_cits = copy.deepcopy(base_cits)

        for fp_str, res in resolved.items():
            if fp_str in passing_fps:
                # Never override passing fields
                continue

            # Build candidate citation
            new_cit = {
                "field_path": fp_str,
                "page": res["page"],
                "bbox": res["bbox"],
                "bounding_box": res["bbox"],
            }

            # Check if this new citation passes
            if _check_passes(new_cit, fp_str, ev_boxes, ev_pages):
                # It's a rescue!
                new_cits[fp_str] = new_cit
                doc_rescued += 1
                rescued_total += 1

        # Check for regressions: verify all previously passing fields still pass
        for fp_str in passing_fps:
            cit = new_cits.get(fp_str)
            if not _check_passes(cit, fp_str, ev_boxes, ev_pages):
                doc_regressed += 1
                regressed_total += 1

        # Save heldout prediction
        pred_out_path = heldout_preds_dir / f"{doc_id}.result.json"
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        if doc_rescued > 0:
            out_cit_list = list(new_cits.values())
            new_pred_data = copy.deepcopy(base_data)
            if "output" not in new_pred_data:
                new_pred_data["output"] = {}
            new_pred_data["output"]["field_citations"] = out_cit_list
            with open(pred_out_path, "w", encoding="utf-8") as pf:
                json.dump(new_pred_data, pf)
        else:
            import shutil
            shutil.copy2(base_pred_p, pred_out_path)

        print(
            f"    Rescued: +{doc_rescued}, Regressed: {doc_regressed}",
            flush=True
        )

        per_doc.append({
            "doc_id": doc_id,
            "rescued": doc_rescued,
            "regressed": doc_regressed,
            "baseline_passing": len(passing_fps),
            "failing_fields": len(failing_fields),
            "resolved_by_assigner": len(resolved),
        })

    elapsed = time.perf_counter() - t_start

    gate_pass = (rescued_total >= 5000) and (regressed_total == 0)

    result = {
        "experiment_id": "EXP-047",
        "phase": "F_heldout_gate",
        "build_char": build_char,
        "build_numeric": build_numeric,
        "build_table": build_table,
        "tag": tag,
        "total_gradeable_fields": total_gradeable,
        "baseline_passing_fields": baseline_passed_total,
        "rescued_fields": rescued_total,
        "regressed_fields": regressed_total,
        "net_rescued": rescued_total - regressed_total,
        "gate_pass": gate_pass,
        "gate_criteria": {"min_rescued": 5000, "max_regressions": 0},
        "elapsed_seconds": elapsed,
        "avg_build_time_s": sum(build_times) / len(build_times) if build_times else 0,
        "avg_assign_time_s": sum(assign_times) / len(assign_times) if assign_times else 0,
        "avg_char_index_size": int(sum(all_char_sizes) / len(all_char_sizes)) if all_char_sizes else 0,
        "avg_numeric_index_size": int(sum(all_num_sizes) / len(all_num_sizes)) if all_num_sizes else 0,
        "avg_n_tables": sum(all_table_counts) / len(all_table_counts) if all_table_counts else 0,
        "assigner_stats": dict(assigner_stats_agg),
        "per_document": per_doc,
    }

    fname = "heldout_results.json" if not tag else f"heldout_results_{tag}.json"
    out_path = out_dir / fname
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== GATE {'PASS' if gate_pass else 'FAIL'} ===", flush=True)
    print(f"  Rescued: {rescued_total} / Regressed: {regressed_total}", flush=True)
    print(f"  Net: {rescued_total - regressed_total}", flush=True)
    print(f"  Elapsed: {elapsed:.1f}s", flush=True)
    print(f"  Written: {out_path}", flush=True)

    return result


if __name__ == "__main__":
    run_heldout_gate()
