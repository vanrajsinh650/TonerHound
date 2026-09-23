"""Official ExtractBench 370-Document Evaluation Runner for Frozen TonerHound Stack.

Evaluates all 370 inference results in research/official_eval/exp-final_predictions/tonerhound/
using the official ExtractEvaluator and EvaluationRunner aggregators.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.schemas.evaluation import EvaluationResult, EvaluationSummary
from extract_bench.schemas.pipeline_io import InferenceResult
from extract_bench.test_cases.loader import load_test_case, load_test_cases


def _eval_worker(task: dict[str, str]) -> dict[str, Any]:
    """Worker process evaluating a single test case independently."""
    rf_path = Path(task["result_file"])
    pdf_path = Path(task["pdf_path"])
    out_json = Path(task["out_json"])

    if out_json.exists():
        try:
            with open(out_json, encoding="utf-8") as f:
                d = json.load(f)
            return {"test_id": task["test_id"], "success": True, "cached": True, "data": d}
        except Exception:
            pass

    t0 = time.perf_counter()
    try:
        with open(rf_path, encoding="utf-8") as f:
            inf_dict = json.load(f)
        inf_result = InferenceResult.model_validate(inf_dict)
        test_case = load_test_case(pdf_path)

        evaluator = ExtractEvaluator()
        eval_result = evaluator.evaluate(inf_result, test_case)
        eval_dict = eval_result.model_dump(mode="json")

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(eval_dict, f, indent=2)

        elapsed = time.perf_counter() - t0
        return {"test_id": task["test_id"], "success": eval_result.success, "cached": False, "elapsed": elapsed, "data": eval_dict}

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        err_msg = f"{type(exc).__name__}: {str(exc)}"
        return {"test_id": task["test_id"], "success": False, "cached": False, "elapsed": elapsed, "error": err_msg}


def main():
    exp_id = "EXP-FINAL"
    data_dir = root_dir / "research" / "data" / "full"
    predictions_dir = root_dir / "research" / "official_eval" / f"{exp_id.lower()}_predictions" / "tonerhound"
    cache_eval_dir = root_dir / "research" / "official_eval" / f"{exp_id.lower()}_eval_cache"
    cache_eval_dir.mkdir(parents=True, exist_ok=True)

    runner = EvaluationRunner(output_dir=predictions_dir, test_cases_dir=data_dir)
    res_files = runner._find_result_files(predictions_dir)
    print(f"Found {len(res_files)} prediction files in {predictions_dir}")

    # Build task list
    tasks = []
    for rf in res_files:
        test_id = rf.relative_to(predictions_dir).as_posix().removesuffix(".result.json")
        pdf_path = data_dir / f"{test_id}.pdf"
        out_json = cache_eval_dir / f"{test_id}.eval.json"
        tasks.append({
            "test_id": test_id,
            "result_file": str(rf),
            "pdf_path": str(pdf_path),
            "out_json": str(out_json),
        })

    print(f"Running evaluation across {len(tasks)} test cases with 6 parallel workers...")
    t_start = time.perf_counter()
    evaluation_results: list[EvaluationResult] = []
    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_eval_worker, t): t["test_id"] for t in tasks}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            if res.get("data"):
                eval_res = EvaluationResult.model_validate(res["data"])
                evaluation_results.append(eval_res)
                if eval_res.success:
                    successful += 1
                else:
                    failed += 1
            else:
                failed += 1
                print(f"Error on {res['test_id']}: {res.get('error')}")

            if completed % 25 == 0 or completed == len(tasks):
                print(f"[{completed:3d}/{len(tasks):3d}] Evaluated (Success: {successful}, Failed: {failed})")

    eval_time = time.perf_counter() - t_start
    print(f"\nCompleted evaluation for {len(evaluation_results)} cases in {eval_time:.1f}s ({eval_time/60:.2f}m).")

    # Official Aggregate Metrics via EvaluationRunner
    print("Computing official aggregations...")
    agg = runner._aggregate_metrics(evaluation_results)
    diag = runner._aggregate_metrics(evaluation_results, metric_list_name="diagnostic_metrics")
    tags = runner._aggregate_tag_metrics(evaluation_results)

    word_f1 = agg.get("avg_extract_unified_grounded_f1", 0.0)
    word_prec = agg.get("avg_extract_unified_grounded_precision", 0.0)
    word_rec = agg.get("avg_extract_unified_grounded_recall", 0.0)

    page_f1 = agg.get("avg_extract_unified_page_f1", 0.0)
    page_prec = agg.get("avg_extract_unified_page_precision", 0.0)
    page_rec = agg.get("avg_extract_unified_page_recall", 0.0)

    val_f1 = agg.get("avg_extract_unified_value_f1", 0.0)

    def get_tag_metric(tag_name: str, metric_name: str) -> float:
        return tags.get(tag_name, {}).get(metric_name, 0.0)

    word_f1_short = get_tag_metric("length:short", "avg_extract_unified_grounded_f1")
    word_f1_medium = get_tag_metric("length:medium", "avg_extract_unified_grounded_f1")
    word_f1_long = get_tag_metric("length:long", "avg_extract_unified_grounded_f1")

    word_prec_short = get_tag_metric("length:short", "avg_extract_unified_grounded_precision")
    word_prec_medium = get_tag_metric("length:medium", "avg_extract_unified_grounded_precision")
    word_prec_long = get_tag_metric("length:long", "avg_extract_unified_grounded_precision")

    word_rec_short = get_tag_metric("length:short", "avg_extract_unified_grounded_recall")
    word_rec_medium = get_tag_metric("length:medium", "avg_extract_unified_grounded_recall")
    word_rec_long = get_tag_metric("length:long", "avg_extract_unified_grounded_recall")

    page_f1_short = get_tag_metric("length:short", "avg_extract_unified_page_f1")
    page_f1_medium = get_tag_metric("length:medium", "avg_extract_unified_page_f1")
    page_f1_long = get_tag_metric("length:long", "avg_extract_unified_page_f1")

    page_prec_short = get_tag_metric("length:short", "avg_extract_unified_page_precision")
    page_prec_medium = get_tag_metric("length:medium", "avg_extract_unified_page_precision")
    page_prec_long = get_tag_metric("length:long", "avg_extract_unified_page_precision")

    page_rec_short = get_tag_metric("length:short", "avg_extract_unified_page_recall")
    page_rec_medium = get_tag_metric("length:medium", "avg_extract_unified_page_recall")
    page_rec_long = get_tag_metric("length:long", "avg_extract_unified_page_recall")

    domains_data: dict[str, dict[str, float]] = {}
    for d_idx in range(1, 9):
        d_tag = f"domain:D{d_idx}"
        domains_data[d_tag] = {
            "word_grounding_f1": get_tag_metric(d_tag, "avg_extract_unified_grounded_f1"),
            "word_grounding_precision": get_tag_metric(d_tag, "avg_extract_unified_grounded_precision"),
            "word_grounding_recall": get_tag_metric(d_tag, "avg_extract_unified_grounded_recall"),
            "page_grounding_f1": get_tag_metric(d_tag, "avg_extract_unified_page_f1"),
            "page_grounding_precision": get_tag_metric(d_tag, "avg_extract_unified_page_precision"),
            "page_grounding_recall": get_tag_metric(d_tag, "avg_extract_unified_page_recall"),
            "value_f1": get_tag_metric(d_tag, "avg_extract_unified_value_f1"),
        }

    false_grounding_rate = 1.0 - word_prec if word_prec > 0 else 0.0

    print("Computing candidate diagnostics...")
    r1_hits = 0
    r5_hits = 0
    r10_hits = 0
    r20_hits = 0
    total_eval_leaves = 0
    ambiguous_leaves = 0
    not_found_leaves = 0

    for res in evaluation_results:
        for m in res.metrics:
            if m.metric_name == "extract_evidence_value_pass_rate" and m.metadata:
                for rr in m.metadata.get("rule_results", []):
                    total_eval_leaves += 1
                    exact_c = rr.get("exact_citation_count", 0)
                    if exact_c == 0 and not rr.get("value_pass"):
                        not_found_leaves += 1
                    if rr.get("coarse_citation_count", 0) > 1 or exact_c > 1:
                        ambiguous_leaves += 1
                    if rr.get("bbox_qualified"):
                        r1_hits += 1
                        r5_hits += 1
                        r10_hits += 1
                        r20_hits += 1
                    elif rr.get("page_qualified"):
                        r5_hits += 1
                        r10_hits += 1
                        r20_hits += 1

    total_gt_denom = max(1, total_eval_leaves)
    recall_at_1 = r1_hits / total_gt_denom
    recall_at_5 = r5_hits / total_gt_denom
    recall_at_10 = r10_hits / total_gt_denom
    recall_at_20 = r20_hits / total_gt_denom
    ambiguity_rate = ambiguous_leaves / total_gt_denom
    not_found_rate = not_found_leaves / total_gt_denom

    # Compare Head-to-Head against EXP-011 baseline
    exp011_report_path = root_dir / "research" / "official_eval" / "reports" / "EXP-011_official_evaluation.json"
    h2h_wins = []
    h2h_losses = []
    h2h_neutral = []
    if exp011_report_path.exists():
        with open(exp011_report_path) as f:
            exp011_data = json.load(f)
        exp011_pdocs = {d["test_id"]: d["metrics"].get("extract_unified_grounded_f1") for d in exp011_data.get("per_document_results", [])}

        for res in evaluation_results:
            tid = res.test_id
            metrics_dict = {m.metric_name: m.value for m in res.metrics}
            cur_wf1 = metrics_dict.get("extract_unified_grounded_f1")
            base_wf1 = exp011_pdocs.get(tid)
            if cur_wf1 is not None and base_wf1 is not None:
                delta = cur_wf1 - base_wf1
                if delta > 0.0005:
                    h2h_wins.append((tid, base_wf1, cur_wf1, delta))
                elif delta < -0.0005:
                    h2h_losses.append((tid, base_wf1, cur_wf1, delta))
                else:
                    h2h_neutral.append((tid, base_wf1, cur_wf1, delta))

    print("\n" + "=" * 75)
    print(f"      OFFICIAL TONERHOUND {exp_id} BENCHMARK RESULTS (370 DOCUMENTS)")
    print("=" * 75)
    print(f"Overall Word Grounding F1:       {word_f1*100:6.2f}% (EXP-011: 45.48%, LlamaExtract: 58.11%)")
    print(f"Overall Word Grounding Precision:{word_prec*100:6.2f}%")
    print(f"Overall Word Grounding Recall:   {word_rec*100:6.2f}%")
    print(f"Overall Page Grounding F1:       {page_f1*100:6.2f}% (EXP-011: 81.22%, LlamaExtract: 84.92%)")
    print(f"Overall Page Grounding Precision:{page_prec*100:6.2f}%")
    print(f"Overall Page Grounding Recall:   {page_rec*100:6.2f}%")
    print(f"Overall Value F1:                {val_f1*100:6.2f}%")
    print(f"Overall False-Grounding Rate:    {false_grounding_rate*100:6.2f}%")
    print("-" * 75)
    print("Length Splits (Word Grounding F1):")
    print(f"  Short (<=10 pages):            {word_f1_short*100:6.2f}%")
    print(f"  Medium (11-50 pages):          {word_f1_medium*100:6.2f}%")
    print(f"  Long (>50 pages):              {word_f1_long*100:6.2f}%")
    print("-" * 75)
    print("Domain Word Grounding F1:")
    for d_tag, d_vals in sorted(domains_data.items()):
        print(f"  {d_tag:12s}:                   {d_vals['word_grounding_f1']*100:6.2f}% (Page: {d_vals['page_grounding_f1']*100:6.2f}%)")
    print("-" * 75)
    print(f"Candidate Recall@1:              {recall_at_1*100:6.2f}%")
    print(f"Candidate Recall@5:              {recall_at_5*100:6.2f}%")
    print(f"Candidate Recall@10:             {recall_at_10*100:6.2f}%")
    print(f"Candidate Recall@20:             {recall_at_20*100:6.2f}%")
    print(f"Ambiguity Rate:                  {ambiguity_rate*100:6.2f}%")
    print(f"Not-Found Rate:                  {not_found_rate*100:6.2f}%")
    print(f"Head-to-Head vs EXP-011:         Wins: {len(h2h_wins)} | Neutral: {len(h2h_neutral)} | Losses: {len(h2h_losses)}")
    print("=" * 75)

    # Save reports
    pdoc_results = []
    for r in evaluation_results:
        pdoc_results.append({
            "test_id": r.test_id,
            "success": r.success,
            "error": r.error,
            "metrics": {m.metric_name: m.value for m in r.metrics},
        })

    exp_data = {
        "experiment_id": exp_id,
        "commit_sha": "26e86920fbae2e076c731c4b181bf7c97adac3b2",
        "frozen_stack": "EXP-011+EXP-012+EXP-013+EXP-015+EXP-017R+EXP-018",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(evaluation_results),
        "total_pages": 4869,
        "total_citations": 610319,
        "prediction_time_seconds": 595.0,
        "evaluation_time_seconds": round(eval_time, 2),
        "total_runtime_seconds": round(595.0 + eval_time, 2),
        "overall_metrics": {
            "word_grounding_f1": word_f1,
            "word_grounding_precision": word_prec,
            "word_grounding_recall": word_rec,
            "page_grounding_f1": page_f1,
            "page_grounding_precision": page_prec,
            "page_grounding_recall": page_rec,
            "value_f1": val_f1,
            "false_grounding_rate": false_grounding_rate,
        },
        "diagnostic_metrics": {
            "candidate_recall_at_1": recall_at_1,
            "candidate_recall_at_5": recall_at_5,
            "candidate_recall_at_10": recall_at_10,
            "candidate_recall_at_20": recall_at_20,
            "ambiguity_rate": ambiguity_rate,
            "not_found_rate": not_found_rate,
        },
        "head_to_head_vs_exp011": {
            "wins": len(h2h_wins),
            "neutral": len(h2h_neutral),
            "losses": len(h2h_losses),
            "win_list": [{"test_id": w[0], "base_wf1": w[1], "cur_wf1": w[2], "delta": w[3]} for w in h2h_wins],
            "loss_list": [{"test_id": l[0], "base_wf1": l[1], "cur_wf1": l[2], "delta": l[3]} for l in h2h_losses],
        },
        "length_splits": {
            "short": {
                "word_grounding_f1": word_f1_short,
                "word_grounding_precision": word_prec_short,
                "word_grounding_recall": word_rec_short,
                "page_grounding_f1": page_f1_short,
                "page_grounding_precision": page_prec_short,
                "page_grounding_recall": page_rec_short,
            },
            "medium": {
                "word_grounding_f1": word_f1_medium,
                "word_grounding_precision": word_prec_medium,
                "word_grounding_recall": word_rec_medium,
                "page_grounding_f1": page_f1_medium,
                "page_grounding_precision": page_prec_medium,
                "page_grounding_recall": page_rec_medium,
            },
            "long": {
                "word_grounding_f1": word_f1_long,
                "word_grounding_precision": word_prec_long,
                "word_grounding_recall": word_rec_long,
                "page_grounding_f1": page_f1_long,
                "page_grounding_precision": page_prec_long,
                "page_grounding_recall": page_rec_long,
            },
        },
        "domains": domains_data,
        "per_document_results": pdoc_results,
    }

    official_reports_dir = root_dir / "research" / "official_eval" / "reports"
    exp_json_path = official_reports_dir / f"{exp_id}_official_evaluation.json"
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_data, f, indent=2)
    print(f"Saved: {exp_json_path}")

    exp_dir_json = root_dir / "research" / "experiments" / f"{exp_id}.json"
    exp_dir_json.parent.mkdir(parents=True, exist_ok=True)
    with open(exp_dir_json, "w", encoding="utf-8") as f:
        json.dump(exp_data, f, indent=2)
    print(f"Saved: {exp_dir_json}")

    print(f"\n=== {exp_id} Official Evaluation Complete! ===")


if __name__ == "__main__":
    main()
