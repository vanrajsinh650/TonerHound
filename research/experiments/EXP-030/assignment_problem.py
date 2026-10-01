"""EXP-030: Global Structured Assignment Engine for Table-Level Evidence Grounding.

Implements joint multi-field record resolution governed by Constraints A through F:
- Constraint A: Row Coherence (Intra-record vertical corridor binding)
- Constraint B: Row Monotonicity (Inter-record top-to-bottom reading order)
- Constraint C: Column Rail Consistency (Horizontal corridor alignment)
- Constraint D: Sibling Proximity (Line co-linearity within record)
- Constraint E: Table Membership (Page boundary containment)
- Constraint F: Abstention & Conservatism Gating (Fall back to baseline when uncertain)

Evaluates on Held-Out Cohort B (32 documents) using both:
1. Field-level candidate Hit@1 and selection gap metrics.
2. Official ExtractBench ExtractEvaluator Word Grounding F1, Precision, Recall, and False Grounding.

Saves:
- research/experiments/EXP-030/heldout_results.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

curr_dir = Path(__file__).resolve().parent
repo_root = curr_dir.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.loader import load_test_case


def _eval_doc_worker(task: dict[str, str]) -> dict[str, Any]:
    test_id = task["test_id"]
    pdf_path = Path(task["pdf_path"])
    res_path = Path(task["res_path"])
    try:
        with open(res_path, encoding="utf-8") as f:
            inf = InferenceResult.model_validate(json.load(f))
        tc = load_test_case(pdf_path)
        evaluator = ExtractEvaluator()
        res = evaluator.evaluate(inf, tc)
        m_map = {m.metric_name: m.value for m in res.metrics}
        return {
            "test_id": test_id,
            "success": True,
            "word_f1": m_map.get("extract_unified_grounded_f1", 0.0),
            "word_precision": m_map.get("extract_unified_grounded_precision", 0.0),
            "word_recall": m_map.get("extract_unified_grounded_recall", 0.0),
            "page_f1": m_map.get("extract_unified_page_f1", 0.0),
            "false_grounding": m_map.get("extract_unified_false_grounding_rate", 0.0),
            "abstention_rate": m_map.get("extract_unified_abstention_rate", 0.0),
        }
    except Exception as exc:
        return {"test_id": test_id, "success": False, "error": str(exc)}


class GlobalStructuredAssigner:
    """Solves joint record-level assignment for table evidence grounding."""

    def __init__(
        self,
        row_tolerance: float = 0.025,
        lambda_row: float = 100.0,
        lambda_col: float = 20.0,
        max_anchor_candidates: int = 5,
        enable_monotonicity: bool = False,
        enable_column_consistency: bool = True,
    ) -> None:
        self.row_tolerance = row_tolerance
        self.lambda_row = lambda_row
        self.lambda_col = lambda_col
        self.max_anchor_candidates = max_anchor_candidates
        self.enable_monotonicity = enable_monotonicity
        self.enable_column_consistency = enable_column_consistency

    def assign_table_records(
        self,
        doc_id: str,
        table_name: str,
        rows_map: dict[int, list[dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        """Assign all fields across records in a table to coherent row/column candidates."""
        assigned: dict[str, dict[str, Any]] = {}

        # 1. Estimate column corridors per subfield (Constraint C)
        col_medians: dict[str, float] = {}
        if self.enable_column_consistency:
            col_xs: dict[str, list[float]] = defaultdict(list)
            for ridx, flds in rows_map.items():
                for f in flds:
                    subf = f["subfield"]
                    cands = f["candidates"]
                    if cands:
                        col_xs[subf].append(cands[0]["bbox"][0])
            for subf, xs in col_xs.items():
                if len(xs) >= 3:
                    col_medians[subf] = float(np.median(xs))

        # 2. Discover distinctive record anchors (Constraint A & D)
        row_anchors: dict[int, tuple[int, float]] = {}
        anchor_fields: dict[int, str] = {}

        for ridx in sorted(rows_map.keys()):
            flds = rows_map[ridx]
            best_f = None
            min_c = 999
            for f in flds:
                c_cnt = len(f["candidates"])
                if 1 <= c_cnt < min_c:
                    min_c = c_cnt
                    best_f = f

            if best_f and min_c <= self.max_anchor_candidates:
                c0 = best_f["candidates"][0]
                row_anchors[ridx] = (int(c0["page"]), float(c0["bbox"][1] + c0["bbox"][3] / 2.0))
                anchor_fields[ridx] = best_f["field_path"]

        # 3. Optional Monotonicity filter (Constraint B)
        if self.enable_monotonicity:
            page_rows: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for ridx, (p, y) in row_anchors.items():
                page_rows[p].append((ridx, y))
            filtered_anchors: dict[int, tuple[int, float]] = {}
            for p, r_list in page_rows.items():
                r_list.sort(key=lambda it: it[0])
                last_y = -1.0
                for ridx, y in r_list:
                    if y >= last_y - 0.008:
                        filtered_anchors[ridx] = (p, y)
                        last_y = max(last_y, y)
            row_anchors = filtered_anchors

        # 4. Resolve every field within the record using joint row & column scoring
        for ridx, flds in rows_map.items():
            anc = row_anchors.get(ridx)
            anc_fpath = anchor_fields.get(ridx)

            for f in flds:
                fpath = f["field_path"]
                cands = f["candidates"]
                if not cands:
                    continue

                # If this field was the anchor, use its baseline candidate
                if anc and fpath == anc_fpath:
                    assigned[fpath] = cands[0]
                    continue

                if not anc:
                    # Constraint F: Fall back to baseline when no anchor exists
                    assigned[fpath] = cands[0]
                    continue

                anc_p, anc_y = anc
                same_page = [c for c in cands if c.get("page") == anc_p]
                if not same_page:
                    assigned[fpath] = cands[0]
                    continue

                # Score candidates by vertical proximity to anchor + column alignment
                subf = f["subfield"]
                col_med = col_medians.get(subf)

                def _score_c(c: dict[str, Any]) -> float:
                    cy = float(c["bbox"][1] + c["bbox"][3] / 2.0)
                    dy = abs(cy - anc_y)
                    s = -dy * self.lambda_row
                    if col_med is not None:
                        dx = abs(float(c["bbox"][0]) - col_med)
                        s += -dx * self.lambda_col
                    return s

                best_c = max(same_page, key=_score_c)
                dy = abs(float(best_c["bbox"][1] + best_c["bbox"][3] / 2.0) - anc_y)

                # Row corridor constraint: if outside tolerance, retain baseline
                if dy <= self.row_tolerance:
                    assigned[fpath] = best_c
                else:
                    assigned[fpath] = cands[0]

        return assigned


def run_cohort_b_evaluation() -> dict[str, Any]:
    print("=== EXP-030: HELD-OUT COHORT B EVALUATION ===")
    t0 = time.time()

    with open(repo_root / "benchmarks" / "held_out_manifest.json") as f:
        held_docs = json.load(f)["documents"]
    held_ids = [d["test_id"] for d in held_docs]
    held_id_set = set(held_ids)

    parquet_path = repo_root / "research" / "observer" / "field_records.parquet"
    dataset = ds.dataset(str(parquet_path), format="parquet")
    filter_expr = ds.field("document_id").isin(held_id_set) & (ds.field("candidate_count") >= 2)
    table = dataset.to_table(
        filter=filter_expr,
        columns=[
            "document_id",
            "field_path",
            "gold_value",
            "candidate_pool",
            "candidate_hit_at_1",
            "best_candidate_iou",
        ],
    )

    print(f"Loaded {len(table)} multi-candidate fields on Cohort B in {time.time()-t0:.2f}s.")

    # Group by (doc_id, table_name) -> ridx -> list[field]
    doc_tables: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    all_multi_fields: list[dict[str, Any]] = []

    for i in range(len(table)):
        doc_id = table["document_id"][i].as_py()
        fpath = table["field_path"][i].as_py()
        cands = json.loads(table["candidate_pool"][i].as_py())
        hit_1 = bool(table["candidate_hit_at_1"][i].as_py())
        best_iou = float(table["best_candidate_iou"][i].as_py())

        f_data = {
            "document_id": doc_id,
            "field_path": fpath,
            "candidates": cands,
            "baseline_hit": hit_1,
            "best_iou": best_iou,
        }
        all_multi_fields.append(f_data)

        m = re.match(r"^(.*?)\[(\d+)\]\.(.*)$", fpath)
        if m:
            tname = m.group(1)
            ridx = int(m.group(2))
            subf = m.group(3)
            f_data["subfield"] = subf
            doc_tables[doc_id][tname][ridx].append(f_data)

    tot_multi = len(all_multi_fields)
    base_hits = sum(1 for f in all_multi_fields if f["baseline_hit"])
    pool_recall = sum(1 for f in all_multi_fields if f["best_iou"] >= 0.50)
    selection_gap = pool_recall - base_hits

    print(f"Total Multi-Candidate Fields: {tot_multi}")
    print(f"Baseline Hit@1: {base_hits} / {tot_multi} ({base_hits/tot_multi*100:.2f}%)")
    print(f"Candidate Pool Recall: {pool_recall} / {tot_multi} ({pool_recall/tot_multi*100:.2f}%)")
    print(f"Initial Selection Gap: {selection_gap} fields ({selection_gap/tot_multi*100:.2f}pp)")

    # Execute Structured Assigner across all tables in Cohort B
    assigner = GlobalStructuredAssigner(
        row_tolerance=0.025,
        lambda_row=100.0,
        lambda_col=20.0,
        max_anchor_candidates=5,
        enable_column_consistency=True,
        enable_monotonicity=False,
    )

    all_assigned_candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for doc_id, tables in doc_tables.items():
        for tname, rows_map in tables.items():
            t_assigned = assigner.assign_table_records(doc_id, tname, rows_map)
            for fpath, cand in t_assigned.items():
                all_assigned_candidates[(doc_id, fpath)] = cand

    # Evaluate Selection Metrics on Held-Out Cohort B
    exp030_hits = 0
    beneficial_flips = 0
    harmful_flips = 0
    unchanged_hits = 0

    for f in all_multi_fields:
        doc_id = f["document_id"]
        fpath = f["field_path"]
        b_hit = f["baseline_hit"]

        assigned_c = all_assigned_candidates.get((doc_id, fpath))
        if assigned_c is not None:
            r_hit = bool(float(assigned_c.get("best_iou", 0.0)) >= 0.50)
            is_flip = (assigned_c != f["candidates"][0])
        else:
            r_hit = b_hit
            is_flip = False

        if r_hit:
            exp030_hits += 1

        if is_flip:
            if not b_hit and r_hit:
                beneficial_flips += 1
            elif b_hit and not r_hit:
                harmful_flips += 1
        else:
            if b_hit and r_hit:
                unchanged_hits += 1

    net_flips = beneficial_flips - harmful_flips
    hit1_gain_pp = (exp030_hits - base_hits) / tot_multi * 100
    gap_reduction_pct = (exp030_hits - base_hits) / max(1, selection_gap) * 100

    print("\n--- HELD-OUT COHORT B SELECTION ACCURACY RESULTS ---")
    print(f"Baseline Multi-Cand Hit@1: {base_hits} / {tot_multi} ({base_hits/tot_multi*100:.2f}%)")
    print(f"EXP-030 Multi-Cand Hit@1:  {exp030_hits} / {tot_multi} ({exp030_hits/tot_multi*100:.2f}%)")
    print(f"Hit@1 Gain (pp):           {hit1_gain_pp:+.2f}pp")
    print(f"Beneficial Flips (+):      {beneficial_flips}")
    print(f"Harmful Flips (-):         {harmful_flips}")
    print(f"Net Flips:                 {net_flips:+d}")
    print(f"Selection Gap Reduction:   {gap_reduction_pct:.2f}% of total selection gap")

    # 5. Generate Official Prediction Files for Cohort B and Run ExtractEvaluator
    print("\n--- Running Official ExtractBench Benchmark on Cohort B (32 Docs) ---")
    base_pred_dir = repo_root / "research" / "experiments" / "EXP-028E" / "predictions" / "tonerhound"
    exp030_pred_dir = curr_dir / "predictions_exp030" / "tonerhound"
    exp030_pred_dir.mkdir(parents=True, exist_ok=True)
    data_dir = repo_root / "research" / "data" / "full"

    eval_tasks = []
    base_eval_tasks = []

    for tid in held_ids:
        src_file = base_pred_dir / f"{tid}.result.json"
        dst_file = exp030_pred_dir / f"{tid}.result.json"
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        with open(src_file, encoding="utf-8") as fp:
            data = json.load(fp)

        # Update citations if modified by structured assigner
        cits = data.get("output", {}).get("field_citations", [])
        for c in cits:
            k = (tid, c.get("field_path"))
            if k in all_assigned_candidates:
                new_c = all_assigned_candidates[k]
                c["page"] = new_c["page"]
                c["bbox"] = new_c["bbox"]

        with open(dst_file, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

        pdf_path = data_dir / f"{tid}.pdf"
        eval_tasks.append({
            "test_id": tid,
            "pdf_path": str(pdf_path),
            "res_path": str(dst_file),
        })
        base_eval_tasks.append({
            "test_id": tid,
            "pdf_path": str(pdf_path),
            "res_path": str(src_file),
        })

    # Run evaluations with 4 parallel workers
    t_ev = time.time()
    exp030_doc_metrics = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_eval_doc_worker, t): t["test_id"] for t in eval_tasks}
        for fut in as_completed(futures):
            exp030_doc_metrics.append(fut.result())
    eval_time = time.time() - t_ev

    # Macro aggregation
    wf1s = [d["word_f1"] for d in exp030_doc_metrics if d.get("success")]
    wprecs = [d["word_precision"] for d in exp030_doc_metrics if d.get("success")]
    wrecs = [d["word_recall"] for d in exp030_doc_metrics if d.get("success")]
    pf1s = [d["page_f1"] for d in exp030_doc_metrics if d.get("success")]
    fgs = [d["false_grounding"] for d in exp030_doc_metrics if d.get("success")]
    absts = [d["abstention_rate"] for d in exp030_doc_metrics if d.get("success")]

    exp030_macro = {
        "word_f1": float(np.mean(wf1s) * 100),
        "word_precision": float(np.mean(wprecs) * 100),
        "word_recall": float(np.mean(wrecs) * 100),
        "page_f1": float(np.mean(pf1s) * 100),
        "false_grounding": float(np.mean(fgs) * 100),
        "abstention_rate": float(np.mean(absts) * 100),
    }

    # Baseline metrics from EXP-028E
    baseline_macro = {
        "word_f1": 59.3588,
        "word_precision": 63.7774,
        "word_recall": 56.5164,
        "page_f1": 85.4408,
        "false_grounding": 0.0,
        "abstention_rate": 0.0,
    }

    delta_f1 = exp030_macro["word_f1"] - baseline_macro["word_f1"]
    delta_prec = exp030_macro["word_precision"] - baseline_macro["word_precision"]
    delta_rec = exp030_macro["word_recall"] - baseline_macro["word_recall"]
    delta_page = exp030_macro["page_f1"] - baseline_macro["page_f1"]

    print("\n=======================================================")
    print("OFFICIAL EXTRACTBENCH HELD-OUT COHORT B RESULTS")
    print("=======================================================")
    print(f"Metric                  Baseline     EXP-030      Delta")
    print(f"Word Grounding F1:      {baseline_macro['word_f1']:6.2f}%     {exp030_macro['word_f1']:6.2f}%     {delta_f1:+6.2f}pp")
    print(f"Word Precision:         {baseline_macro['word_precision']:6.2f}%     {exp030_macro['word_precision']:6.2f}%     {delta_prec:+6.2f}pp")
    print(f"Word Recall:            {baseline_macro['word_recall']:6.2f}%     {exp030_macro['word_recall']:6.2f}%     {delta_rec:+6.2f}pp")
    print(f"Page Grounding F1:      {baseline_macro['page_f1']:6.2f}%     {exp030_macro['page_f1']:6.2f}%     {delta_page:+6.2f}pp")
    print(f"Evaluation Runtime:     {eval_time:.1f}s")
    print("=======================================================")

    # Decision Gate Check
    if hit1_gain_pp >= 5.0 and delta_f1 >= 0.0:
        gate_status = "GATE_A_PASSED"
        gate_msg = "Gate A Passed: Hit@1 gain >= 5pp and non-regressive Word F1."
    elif hit1_gain_pp >= 2.0 and delta_f1 >= 0.0:
        gate_status = "GATE_B_PASSED"
        gate_msg = "Gate B Passed: Hit@1 gain between 2pp and 5pp and non-regressive Word F1."
    else:
        gate_status = "GATE_C_STOP"
        gate_msg = "Gate C Triggered: Hit@1 gain < 2pp or Word F1 regressed. STOP."

    print(f"\nDECISION GATE STATUS: {gate_status}")
    print(f"DECISION MESSAGE:     {gate_msg}")

    results = {
        "experiment": "EXP-030",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Global Structured Assignment for Table-Level Evidence Grounding",
        "held_out_cohort": "Cohort B (32 documents)",
        "total_multi_candidate_fields": tot_multi,
        "selection_metrics": {
            "baseline_hit_count": base_hits,
            "baseline_hit_rate": round(base_hits / tot_multi * 100, 2),
            "exp030_hit_count": exp030_hits,
            "exp030_hit_rate": round(exp030_hits / tot_multi * 100, 2),
            "hit1_gain_pp": round(hit1_gain_pp, 2),
            "candidate_pool_recall": round(pool_recall / tot_multi * 100, 2),
            "selection_gap_pp": round(selection_gap / tot_multi * 100, 2),
            "gap_reduction_pct": round(gap_reduction_pct, 2),
            "beneficial_flips": beneficial_flips,
            "harmful_flips": harmful_flips,
            "net_flips": net_flips,
        },
        "official_extractbench_metrics": {
            "baseline": baseline_macro,
            "exp030": exp030_macro,
            "delta_pp": {
                "word_f1": round(delta_f1, 2),
                "word_precision": round(delta_prec, 2),
                "word_recall": round(delta_rec, 2),
                "page_f1": round(delta_page, 2),
            },
        },
        "decision_gate": {
            "status": gate_status,
            "message": gate_msg,
            "hit1_gain_pp": round(hit1_gain_pp, 2),
            "word_f1_delta_pp": round(delta_f1, 2),
        },
    }

    out_file = curr_dir / "heldout_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Saved {out_file}")

    return results


if __name__ == "__main__":
    run_cohort_b_evaluation()
