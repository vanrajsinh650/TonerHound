"""Local Benchmark Runner for EXP-005.

Executes the frozen 32-document local benchmark (benchmarks/exp005_local_manifest.json)
using ExtractBench's official evaluator, measuring Word Grounding F1, Page F1,
Candidate Recall@1/5/10/20, False Grounding Rate, and high-resolution stage timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure reference ExtractBench src is in Python path for official evaluation
_REF_EXTRACTBENCH = (
    Path(__file__).resolve().parent.parent / "research" / "reference" / "ExtractBench" / "src"
)
if _REF_EXTRACTBENCH.exists() and str(_REF_EXTRACTBENCH) not in sys.path:
    sys.path.insert(0, str(_REF_EXTRACTBENCH))

from extract_bench.test_cases.loader import load_test_case

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.document.index import DocumentIndex

try:
    from scratch.benchmark_hardest_backends import compute_candidate_recall_and_failures
    _HAS_CAND_RECALL = True
except ImportError:
    _HAS_CAND_RECALL = False


@dataclass
class LocalDocResult:
    test_id: str
    split: str
    length_class: str
    domain: str
    num_pages: int
    num_citations: int
    word_grounding_f1: float
    word_grounding_precision: float
    word_grounding_recall: float
    page_grounding_f1: float
    page_grounding_precision: float
    page_grounding_recall: float
    value_f1: float
    false_grounding_rate: float
    indexing_time_sec: float
    grounding_time_sec: float
    eval_time_sec: float
    total_time_sec: float
    candidate_recall_at_20: float | None = None
    exp004_baseline_f1: float | None = None
    delta_vs_exp004: float | None = None


def run_local_benchmark(
    manifest_path: Path | str = "benchmarks/exp005_local_manifest.json",
    split_filter: str = "all",
    experiment_id: str = "EXP-005-baseline",
    backend: str = "pdfium",
    enable_ocr: bool = False,
    enable_structural_disambiguation: bool = True,
    enable_verification: bool = True,
    score_margin_threshold: float = 0.01,
    enable_bbox_precision: bool = True,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    all_docs = manifest["documents"]
    if split_filter != "all":
        docs = [d for d in all_docs if d["split"] == split_filter]
    else:
        docs = all_docs

    if limit is not None:
        docs = docs[:limit]

    data_dir = Path("research/data/full")
    doc_results: list[LocalDocResult] = []

    if verbose:
        print("\n=======================================================")
        print(f"RUNNING EXP-005 LOCAL BENCHMARK: {experiment_id}")
        print(f"Split Filter: {split_filter} | Total Documents: {len(docs)}")
        print("=======================================================\n")

    suite_start = time.perf_counter()

    for i, item in enumerate(docs, start=1):
        tid = item["test_id"]
        split = item["split"]
        length_class = item["length_class"]
        domain = item["domain"]
        pdf_path = data_dir / f"{tid}.pdf"
        exp004_base = item.get("exp004_baseline", {}).get("word_grounding_f1")

        test_case = load_test_case(pdf_path)
        if test_case is None:
            if verbose:
                print(f"[{i}/{len(docs)}] ERROR: Could not load test case for {pdf_path}")
            continue

        # 1. Index document
        t0 = time.perf_counter()
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=enable_ocr, backend=backend)
        t_index = time.perf_counter() - t0

        # 2. Ground extractions
        t1 = time.perf_counter()
        adapter = ExtractBenchAdapter(
            doc_index,
            enable_structural_disambiguation=enable_structural_disambiguation,
            enable_verification=enable_verification,
            score_margin_threshold=score_margin_threshold,
            enable_bbox_precision=enable_bbox_precision,
        )
        payload = adapter.ground_extracted_data(
            test_case.expected_output,
            example_id=tid,
        )
        citations = payload["field_citations"]
        t_ground = time.perf_counter() - t1

        # 3. Official ExtractBench Evaluation
        t2 = time.perf_counter()
        metrics = evaluate_prediction(
            expected_output=test_case.expected_output,
            extracted_data=test_case.expected_output,
            field_rules=test_case.test_rules,
            field_citations=citations,
            data_schema=test_case.data_schema,
        )
        t_eval = time.perf_counter() - t2

        w_f1 = metrics.get("word_grounding_f1") or 0.0
        w_prec = metrics.get("word_grounding_precision") or 0.0
        w_rec = metrics.get("word_grounding_recall") or 0.0
        p_f1 = metrics.get("page_grounding_f1") or 0.0
        p_prec = metrics.get("page_grounding_precision") or 0.0
        p_rec = metrics.get("page_grounding_recall") or 0.0
        v_f1 = metrics.get("value_f1") or 1.0

        false_grounding = (1.0 - w_prec) if w_prec > 0.0 else 0.0
        delta = (w_f1 - exp004_base) if exp004_base is not None else None

        cand_rec_20: float | None = None
        if _HAS_CAND_RECALL:
            try:
                cand_rec_20, _, _ = compute_candidate_recall_and_failures(
                    test_case=test_case,
                    adapter=adapter,
                    citations=citations,
                    doc_index=doc_index,
                )
            except Exception:
                cand_rec_20 = None

        res = LocalDocResult(
            test_id=tid,
            split=split,
            length_class=length_class,
            domain=domain,
            num_pages=len(doc_index.pages),
            num_citations=len(citations),
            word_grounding_f1=w_f1,
            word_grounding_precision=w_prec,
            word_grounding_recall=w_rec,
            page_grounding_f1=p_f1,
            page_grounding_precision=p_prec,
            page_grounding_recall=p_rec,
            value_f1=v_f1,
            false_grounding_rate=false_grounding,
            indexing_time_sec=t_index,
            grounding_time_sec=t_ground,
            eval_time_sec=t_eval,
            total_time_sec=t_index + t_ground + t_eval,
            candidate_recall_at_20=cand_rec_20,
            exp004_baseline_f1=exp004_base,
            delta_vs_exp004=delta,
        )
        doc_results.append(res)

        if verbose:
            delta_str = f"({delta*100:+5.1f}%)" if delta is not None else ""
            cr_str = f"CR@20: {cand_rec_20*100:5.1f}% | " if cand_rec_20 is not None else ""
            print(
                f"[{i:2d}/{len(docs):2d}] {tid:48s} | "
                f"WF1: {w_f1*100:5.1f}% {delta_str:7s} | "
                f"PF1: {p_f1*100:5.1f}% | "
                f"{cr_str}"
                f"Time: {res.total_time_sec:5.2f}s (G: {t_ground:5.2f}s)"
            )

    suite_total_time = time.perf_counter() - suite_start

    # Compute summary aggregates
    def calc_agg(subset: list[LocalDocResult]) -> dict[str, float]:
        if not subset:
            return {}
        cands = [d.candidate_recall_at_20 for d in subset if d.candidate_recall_at_20 is not None]
        mean_cr20 = sum(cands) / len(cands) if cands else 0.0
        return {
            "count": len(subset),
            "total_pages": sum(d.num_pages for d in subset),
            "total_citations": sum(d.num_citations for d in subset),
            "mean_word_grounding_f1": sum(d.word_grounding_f1 for d in subset) / len(subset),
            "mean_word_precision": sum(d.word_grounding_precision for d in subset) / len(subset),
            "mean_word_recall": sum(d.word_grounding_recall for d in subset) / len(subset),
            "mean_page_grounding_f1": sum(d.page_grounding_f1 for d in subset) / len(subset),
            "mean_page_precision": sum(d.page_grounding_precision for d in subset) / len(subset),
            "mean_page_recall": sum(d.page_grounding_recall for d in subset) / len(subset),
            "mean_candidate_recall_at_20": mean_cr20,
            "mean_false_grounding_rate": sum(d.false_grounding_rate for d in subset) / len(subset),
            "total_index_sec": sum(d.indexing_time_sec for d in subset),
            "total_ground_sec": sum(d.grounding_time_sec for d in subset),
            "total_eval_sec": sum(d.eval_time_sec for d in subset),
            "total_latency_sec": sum(d.total_time_sec for d in subset),
        }

    train_dev_res = [d for d in doc_results if d.split == "train_dev"]
    local_val_res = [d for d in doc_results if d.split == "local_validation"]
    overall_agg = calc_agg(doc_results)
    train_dev_agg = calc_agg(train_dev_res)
    local_val_agg = calc_agg(local_val_res)

    short_agg = calc_agg([d for d in doc_results if d.length_class == "short"])
    medium_agg = calc_agg([d for d in doc_results if d.length_class == "medium"])
    long_agg = calc_agg([d for d in doc_results if d.length_class == "long"])

    summary = {
        "experiment_id": experiment_id,
        "backend": backend,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suite_total_time_sec": suite_total_time,
        "overall": overall_agg,
        "train_dev": train_dev_agg,
        "local_validation": local_val_agg,
        "slices": {
            "short": short_agg,
            "medium": medium_agg,
            "long": long_agg,
        },
        "document_results": [asdict(r) for r in doc_results],
    }

    if verbose:
        print("\n" + "=" * 60)
        print("EXP-005 LOCAL BENCHMARK SUMMARY RESULTS")
        print("=" * 60)
        print(f"Total Documents Evaluated: {len(doc_results)}")
        print(f"Total Runtime: {suite_total_time:.2f}s ({suite_total_time/60:.2f} min)")
        print(f"Overall Word Grounding F1:   {overall_agg.get('mean_word_grounding_f1', 0)*100:.2f}%")
        print(f"Overall Word Precision:      {overall_agg.get('mean_word_precision', 0)*100:.2f}%")
        print(f"Overall Word Recall:         {overall_agg.get('mean_word_recall', 0)*100:.2f}%")
        print(f"Overall Page Grounding F1:   {overall_agg.get('mean_page_grounding_f1', 0)*100:.2f}%")
        print(f"Overall False Grounding Rate:{overall_agg.get('mean_false_grounding_rate', 0)*100:.2f}%\n")
        print("--- By Split ---")
        print(f"Train/Dev F1 (20 docs):      {train_dev_agg.get('mean_word_grounding_f1', 0)*100:.2f}% (EXP-004 base: 37.87%)")
        print(f"Local Val F1 (12 docs):      {local_val_agg.get('mean_word_grounding_f1', 0)*100:.2f}% (EXP-004 base: 44.30%)\n")
        print("--- By Length Slice ---")
        print(f"Short Documents F1:          {short_agg.get('mean_word_grounding_f1', 0)*100:.2f}%")
        print(f"Medium Documents F1:         {medium_agg.get('mean_word_grounding_f1', 0)*100:.2f}%")
        print(f"Long Documents F1:           {long_agg.get('mean_word_grounding_f1', 0)*100:.2f}%\n")

    return summary


def update_leaderboard(
    summary: dict[str, Any],
    leaderboard_csv: Path | str = "experiments/EXP-005-leaderboard.csv",
) -> None:
    leaderboard_csv = Path(leaderboard_csv)
    file_exists = leaderboard_csv.exists()

    overall = summary.get("overall", {})
    train_dev = summary.get("train_dev", {})
    local_val = summary.get("local_validation", {})
    slices = summary.get("slices", {})

    row = {
        "experiment_id": summary["experiment_id"],
        "timestamp": summary["timestamp"],
        "overall_word_f1": f"{overall.get('mean_word_grounding_f1', 0)*100:.2f}%",
        "overall_word_precision": f"{overall.get('mean_word_precision', 0)*100:.2f}%",
        "overall_word_recall": f"{overall.get('mean_word_recall', 0)*100:.2f}%",
        "overall_page_f1": f"{overall.get('mean_page_grounding_f1', 0)*100:.2f}%",
        "train_dev_f1": f"{train_dev.get('mean_word_grounding_f1', 0)*100:.2f}%",
        "local_val_f1": f"{local_val.get('mean_word_grounding_f1', 0)*100:.2f}%",
        "short_f1": f"{slices.get('short', {}).get('mean_word_grounding_f1', 0)*100:.2f}%",
        "medium_f1": f"{slices.get('medium', {}).get('mean_word_grounding_f1', 0)*100:.2f}%",
        "long_f1": f"{slices.get('long', {}).get('mean_word_grounding_f1', 0)*100:.2f}%",
        "false_grounding_rate": f"{overall.get('mean_false_grounding_rate', 0)*100:.2f}%",
        "total_runtime_sec": f"{summary.get('suite_total_time_sec', 0):.2f}",
    }

    fieldnames = list(row.keys())
    with open(leaderboard_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EXP-005 local benchmark")
    parser.add_argument("--manifest", default="benchmarks/exp005_local_manifest.json")
    parser.add_argument("--split", default="all", choices=["all", "train_dev", "local_validation"])
    parser.add_argument("--id", default="EXP-005-baseline")
    parser.add_argument("--score-margin", type=float, default=0.01, help="Score margin threshold for verifier")
    parser.add_argument("--backend", default="pdfium", choices=["pdfium", "liteparse", "hybrid"], help="Document parsing backend")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save summary JSON to experiments/")
    args = parser.parse_args()

    summary = run_local_benchmark(
        manifest_path=args.manifest,
        split_filter=args.split,
        experiment_id=args.id,
        backend=args.backend,
        enable_ocr=not args.no_ocr,
        score_margin_threshold=args.score_margin,
        limit=args.limit,
    )

    if args.save:
        out_json = Path(f"experiments/{args.id}.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {out_json}")
        update_leaderboard(summary)
        print("Updated leaderboard: experiments/EXP-005-leaderboard.csv")
