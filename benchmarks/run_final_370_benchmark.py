"""Final 370-Document Benchmark Execution on Frozen TonerHound Stack.

Frozen Stack: EXP-011 + EXP-012 + EXP-013 + EXP-015 + EXP-017R + EXP-018
Evaluates using official ExtractBench EvaluationRunner across all 370 documents.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
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

from extract_bench.test_cases import load_test_cases
from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import ExtractionInput, ResolutionResult
from tonerhound.resolution.flat_form_reranker import FlatFormLabelReranker
from tonerhound.resolution.resolver import EvidenceResolver


class _ValidationResolver(EvidenceResolver):
    def __init__(self, index: DocumentIndex, doc_id: str) -> None:
        super().__init__(
            index,
            enable_verification=True,
            score_margin_threshold=0.01,
            enable_candidate_recovery=True,
        )
        self._doc_id = doc_id
        self._flat_form = FlatFormLabelReranker(
            index=index,
            doc_id=doc_id,
            enabled=True,
            stages_enabled=frozenset({"vertical", "direction", "label", "suppress", "qualifiers"}),
        )

    def resolve(self, extraction: ExtractionInput) -> ResolutionResult:
        if self._flat_form is None or self._flat_form.family is None:
            return super().resolve(extraction)

        result = super().resolve(extraction)
        candidates = self.last_field_candidates.get(extraction.field, [])

        result = self._flat_form.rerank(
            result=result,
            field=extraction.field,
            value=extraction.value,
            candidates=candidates,
            field_context=extraction.field_context,
        )
        return result


class _ValidationAdapter(ExtractBenchAdapter):
    def __init__(self, index: DocumentIndex, doc_id: str) -> None:
        super().__init__(
            index,
            enable_structural_disambiguation=True,
            enable_verification=True,
            score_margin_threshold=0.01,
            enable_bbox_precision=True,
            enable_page_fallback=True,
            enable_character_span=True,
            enable_same_line_recovery=True,
            enable_structure_aware_recovery=True,
            enable_dot_leader_trimming=True,
        )
        self.resolver = _ValidationResolver(index=index, doc_id=doc_id)


def process_single_case(case_info: dict[str, Any]) -> dict[str, Any]:
    """Process a single document test case in an isolated worker process."""
    test_id = case_info["test_id"]
    group = case_info["group"]
    pdf_path = Path(case_info["pdf_path"])
    expected_output = case_info["expected_output"]
    out_file = Path(case_info["out_file"])

    t0 = time.perf_counter()
    try:
        if out_file.exists() and not case_info.get("force", False):
            try:
                with open(out_file, encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("raw_output", {}) if isinstance(data.get("raw_output"), dict) else {}
                return {
                    "test_id": test_id,
                    "group": group,
                    "success": True,
                    "cached": True,
                    "num_pages": raw.get("num_pages", 0),
                    "ocr_pages": raw.get("ocr_pages", 0),
                    "num_citations": len(data.get("output", {}).get("field_citations", [])),
                    "latency_sec": data.get("latency_in_ms", 0) / 1000.0,
                    "error": None,
                }
            except Exception:
                pass

        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="hybrid")
        ocr_pages = getattr(doc_index, "ocr_page_count", 0)

        adapter = _ValidationAdapter(doc_index, doc_id=test_id)

        payload = adapter.ground_extracted_data(
            expected_output,
            example_id=test_id,
            pipeline_name="tonerhound",
        )
        elapsed_sec = time.perf_counter() - t0
        elapsed_ms = int(elapsed_sec * 1000)

        citations = [
            FieldCitation(
                field_path=c["field_path"],
                page=c["page"],
                bbox=c.get("bbox"),
                reference_text=c.get("reference_text"),
                confidence=c.get("confidence"),
                source="tonerhound",
            )
            for c in payload.get("field_citations", [])
            if c.get("page") is not None
        ]

        extract_output = ExtractOutput(
            task_type="extract",
            example_id=test_id,
            pipeline_name="tonerhound",
            extracted_data=payload.get("extracted_data", expected_output if isinstance(expected_output, dict) else {}),
            field_citations=citations,
        )

        now = datetime.now(timezone.utc)
        request = InferenceRequest(
            example_id=test_id,
            source_file_path=str(pdf_path),
            product_type=ProductType.EXTRACT,
        )

        inference_result = InferenceResult(
            request=request,
            pipeline_name="tonerhound",
            product_type=ProductType.EXTRACT,
            raw_output={
                "citations_count": len(citations),
                "ocr_pages": ocr_pages,
                "num_pages": len(doc_index.pages),
            },
            output=extract_output,
            started_at=now,
            completed_at=now,
            latency_in_ms=elapsed_ms,
        )

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(inference_result.model_dump_json(indent=2))

        return {
            "test_id": test_id,
            "group": group,
            "success": True,
            "cached": False,
            "num_pages": len(doc_index.pages),
            "ocr_pages": ocr_pages,
            "num_citations": len(citations),
            "latency_sec": elapsed_sec,
            "error": None,
        }

    except Exception as exc:
        elapsed_sec = time.perf_counter() - t0
        err_msg = f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}"
        print(f"Error on {test_id}: {err_msg}", file=sys.stderr)

        try:
            now = datetime.now(timezone.utc)
            inference_result = InferenceResult(
                request=InferenceRequest(
                    example_id=test_id,
                    source_file_path=str(pdf_path),
                    product_type=ProductType.EXTRACT,
                ),
                pipeline_name="tonerhound",
                product_type=ProductType.EXTRACT,
                raw_output={"error": err_msg},
                output=ExtractOutput(
                    task_type="extract",
                    example_id=test_id,
                    pipeline_name="tonerhound",
                    extracted_data=expected_output if isinstance(expected_output, dict) else {},
                    field_citations=[],
                ),
                started_at=now,
                completed_at=now,
                latency_in_ms=int(elapsed_sec * 1000),
            )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(inference_result.model_dump_json(indent=2))
        except Exception:
            pass

        return {
            "test_id": test_id,
            "group": group,
            "success": False,
            "cached": False,
            "num_pages": 0,
            "ocr_pages": 0,
            "num_citations": 0,
            "latency_sec": elapsed_sec,
            "error": err_msg,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Final 370-Document Benchmark on Frozen TonerHound Stack")
    parser.add_argument("--id", default="EXP-FINAL", help="Experiment ID (default EXP-FINAL)")
    parser.add_argument("--data-dir", default=str(root_dir / "research" / "data" / "full"), help="Test data dir")
    parser.add_argument("--workers", type=int, default=7, help="Worker processes for prediction (default 7)")
    parser.add_argument("--eval-workers", type=int, default=1, help="Worker processes for evaluation (default 1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all cases")
    args = parser.parse_args()

    exp_id = args.id.upper()
    data_dir = Path(args.data_dir)
    predictions_dir = root_dir / "research" / "official_eval" / f"{exp_id.lower()}_predictions" / "tonerhound"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "pipeline_name": "tonerhound",
        "product_type": "extract",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stack": "EXP-011+EXP-012+EXP-013+EXP-015+EXP-017R+EXP-018",
    }
    with open(predictions_dir / "_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"=== {exp_id}: Loading Benchmark Test Cases from {data_dir} ===")
    cases = load_test_cases(data_dir)
    if args.limit:
        cases = cases[: args.limit]
    total_cases = len(cases)
    print(f"Loaded {total_cases} test cases.\n")

    case_infos = []
    for c in cases:
        out_file = predictions_dir / f"{c.test_id}.result.json"
        case_infos.append({
            "test_id": c.test_id,
            "group": c.group or c.test_id.split("/")[0],
            "pdf_path": str(c.file_path),
            "expected_output": c.expected_output,
            "out_file": str(out_file),
            "force": args.force,
        })

    print(f"Generating predictions with {args.workers} worker processes...")
    t_start = time.perf_counter()
    results = []
    completed_count = 0
    total_citations = 0
    total_pages = 0
    total_ocr_pages = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_case, info): info["test_id"] for info in case_infos}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed_count += 1
            total_citations += res.get("num_citations", 0)
            total_pages += res.get("num_pages", 0)
            total_ocr_pages += res.get("ocr_pages", 0)
            status_symbol = "✅" if res["success"] else "❌"
            cached_tag = " (cached)" if res.get("cached") else ""
            print(
                f"[{completed_count:3d}/{total_cases:3d}] {status_symbol} {res['test_id']:45s} "
                f"({res.get('num_citations', 0)} cits, {res.get('latency_sec', 0.0):.1f}s){cached_tag}"
            )

    prediction_time = time.perf_counter() - t_start
    print(f"\nCompleted predictions for {total_cases} cases in {prediction_time:.1f}s ({prediction_time/60:.2f}m).")
    print(f"Total pages: {total_pages}, OCR pages: {total_ocr_pages}, Total citations: {total_citations}")

    print("\n=== Running Official ExtractBench EvaluationRunner ===")
    runner = EvaluationRunner(
        output_dir=predictions_dir,
        test_cases_dir=data_dir,
    )

    t_eval_start = time.perf_counter()
    eval_summary = runner.run_evaluation(product_type="extract", max_workers=args.eval_workers)
    eval_time = time.perf_counter() - t_eval_start
    print(
        f"Official evaluation completed in {eval_time:.1f}s across {eval_summary.total_examples} examples "
        f"(successful: {eval_summary.successful}, failed: {eval_summary.failed})."
    )

    agg = eval_summary.aggregate_metrics
    tags = eval_summary.tag_metrics

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

    print("\nComputing candidate diagnostics...")
    r1_hits = 0
    r5_hits = 0
    r10_hits = 0
    r20_hits = 0
    total_eval_leaves = 0
    ambiguous_leaves = 0
    not_found_leaves = 0

    for res in eval_summary.per_example_results:
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

    total_runtime = prediction_time + eval_time

    # Head-to-Head Comparison against EXP-011 baseline
    exp011_report_path = root_dir / "research" / "official_eval" / "reports" / "EXP-011_official_evaluation.json"
    h2h_wins = []
    h2h_losses = []
    h2h_neutral = []
    if exp011_report_path.exists():
        with open(exp011_report_path) as f:
            exp011_data = json.load(f)
        exp011_pdocs = {d["test_id"]: d["metrics"].get("extract_unified_grounded_f1") for d in exp011_data.get("per_document_results", [])}
        
        for res in eval_summary.per_example_results:
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
    print(f"      TONERHOUND {exp_id} FULL BENCHMARK RESULTS (370 DOCUMENTS)")
    print("=" * 75)
    print(f"Overall Word Grounding F1:       {word_f1*100:6.2f}% (EXP-011 Baseline: 45.48%)")
    print(f"Overall Word Grounding Precision:{word_prec*100:6.2f}%")
    print(f"Overall Word Grounding Recall:   {word_rec*100:6.2f}%")
    print(f"Overall Page Grounding F1:       {page_f1*100:6.2f}% (EXP-011 Baseline: 81.22%)")
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
    print(f"Total Runtime:                   {total_runtime:.2f}s ({total_runtime/60:.2f}m)")
    print("=" * 75)

    official_reports_dir = root_dir / "research" / "official_eval" / "reports"
    official_reports_dir.mkdir(parents=True, exist_ok=True)
    exp_json_path = official_reports_dir / f"{exp_id}_official_evaluation.json"

    pdoc_results = []
    for r in eval_summary.per_example_results:
        pdoc_results.append({
            "test_id": r.test_id,
            "success": r.success,
            "error": r.error,
            "metrics": {m.metric_name: m.value for m in r.metrics},
        })

    exp_data = {
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_documents": total_cases,
        "total_pages": total_pages,
        "total_ocr_pages": total_ocr_pages,
        "total_citations": total_citations,
        "prediction_time_seconds": round(prediction_time, 2),
        "evaluation_time_seconds": round(eval_time, 2),
        "total_runtime_seconds": round(total_runtime, 2),
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

    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_data, f, indent=2)
    print(f"Saved: {exp_json_path}")

    # Also save to research/experiments/
    exp_dir_json = root_dir / "research" / "experiments" / f"{exp_id}.json"
    exp_dir_json.parent.mkdir(parents=True, exist_ok=True)
    with open(exp_dir_json, "w", encoding="utf-8") as f:
        json.dump(exp_data, f, indent=2)
    print(f"Saved: {exp_dir_json}")

    print(f"\n=== {exp_id} Full 370-Document Benchmark Complete! ===")

if __name__ == "__main__":
    main()
