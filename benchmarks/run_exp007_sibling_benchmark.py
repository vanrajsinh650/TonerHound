"""32-Document Benchmark Runner for EXP-007 Sibling Co-occurrence.

Runs the frozen 32-document local benchmark (benchmarks/exp005_local_manifest.json)
using SiblingAdapter with the verified hybrid backend, saving results to
experiments/EXP-007-sibling-32doc.json and updating the leaderboard.
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
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF_EXTRACTBENCH = _REPO_ROOT / "research" / "reference" / "ExtractBench" / "src"
if _REF_EXTRACTBENCH.exists() and str(_REF_EXTRACTBENCH) not in sys.path:
    sys.path.insert(0, str(_REF_EXTRACTBENCH))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from extract_bench.test_cases.loader import load_test_case
from extract_bench.test_cases.schema import iter_rule_evidence

from tonerhound.benchmark.adapter import (
    ExtractBenchAdapter as SiblingAdapter,
    _flatten_leaves_with_context,
)
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.document.index import DocumentIndex
from scratch.benchmark_hardest_backends import generate_raw_candidates_for_field


def compute_candidate_recalls_and_ambiguity(
    test_case: Any,
    adapter: Any,
    citations: list[dict[str, Any]],
    doc_index: DocumentIndex,
) -> tuple[float, float, float, float]:
    """Compute Candidate Recall@1, Recall@5, Recall@20, and Ambiguity Rate."""
    cit_by_field = {c["field_path"]: c for c in citations}
    resolver = adapter.resolver
    matcher = resolver.matcher

    leaves = _flatten_leaves_with_context(test_case.expected_output)
    doc_offset = adapter._detect_page_offset(leaves) if hasattr(adapter, "_detect_page_offset") else 0
    leaf_info: dict[str, dict[str, Any]] = {}
    for path, val, p_hint, ctx, rec_path in leaves:
        eff_hint = (p_hint + doc_offset) if p_hint is not None else None
        leaf_info[path] = {
            "value": val,
            "page_hint": eff_hint,
            "context": ctx,
            "record_path": rec_path,
        }

    rules_with_bbox = []
    for r in test_case.test_rules:
        ev_list = list(iter_rule_evidence(r))
        ev_with_box = [e for e in ev_list if e.bbox is not None and e.page is not None]
        if ev_with_box:
            rules_with_bbox.append((r, ev_with_box[0]))

    if not rules_with_bbox:
        return 1.0, 1.0, 1.0, 0.0

    r1_hits = 0
    r5_hits = 0
    r20_hits = 0
    ambig_count = 0

    for rule, ev in rules_with_bbox:
        field_path = rule.field_path
        gt_page = ev.page
        gt_bbox = ev.bbox
        gt_val = ev.value

        l_data = leaf_info.get(field_path, {})
        val = l_data.get("value", gt_val)
        ctx = l_data.get("context", "")

        pred_cit = cit_by_field.get(field_path)
        is_pred_hit = False
        if pred_cit and pred_cit.get("page") == gt_page and pred_cit.get("bbox"):
            if iou_xywh(pred_cit["bbox"], gt_bbox) >= 0.50:
                is_pred_hit = True

        raw_cands = generate_raw_candidates_for_field(matcher, val, page_hint=gt_page)
        if len(raw_cands) > 1:
            ambig_count += 1

        if is_pred_hit:
            r1_hits += 1
            r5_hits += 1
            r20_hits += 1
        else:
            if raw_cands:
                scored_cands = resolver._score_candidates_with_context(raw_cands, ctx or "")
                scored_cands.sort(key=lambda item: item[1], reverse=True)
                cands = [sc[0] for sc in scored_cands]
            else:
                cands = []

            if cands and cands[0].page == gt_page and iou_xywh(cands[0].bbox.to_coco(), gt_bbox) >= 0.50:
                r1_hits += 1
                r5_hits += 1
                r20_hits += 1
            elif any(c.page == gt_page and iou_xywh(c.bbox.to_coco(), gt_bbox) >= 0.50 for c in cands[:5]):
                r5_hits += 1
                r20_hits += 1
            elif any(c.page == gt_page and iou_xywh(c.bbox.to_coco(), gt_bbox) >= 0.50 for c in cands[:20]):
                r20_hits += 1

    total = len(rules_with_bbox)
    return r1_hits / total, r5_hits / total, r20_hits / total, ambig_count / total


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
    candidate_recall_at_1: float | None = None
    candidate_recall_at_5: float | None = None
    candidate_recall_at_20: float | None = None
    ambiguity_rate: float | None = None
    exp004_baseline_f1: float | None = None
    exp010_baseline_f1: float | None = None
    delta_vs_exp004: float | None = None
    delta_vs_exp010: float | None = None
    delta_vs_exp006: float | None = None


def run_benchmark(
    manifest_path: Path | str = "benchmarks/exp005_local_manifest.json",
    split_filter: str = "all",
    experiment_id: str = "EXP-007-sibling-32doc",
    backend: str = "hybrid",
    enable_ocr: bool = True,
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

    # Load frozen EXP-010 sprint wave3 baseline for direct delta comparison
    exp010_path = Path("experiments/EXP-010-sprint-wave3.json")
    exp010_scores: dict[str, float] = {}
    if exp010_path.exists():
        with open(exp010_path, encoding="utf-8") as f:
            exp010_data = json.load(f)
            for r in exp010_data.get("document_results", []):
                exp010_scores[r["test_id"]] = r.get("word_grounding_f1", 0.0)

    # Load frozen EXP-006 step 3 baseline
    exp006_path = Path("experiments/EXP-006-step3-hybrid-recovery.json")
    exp006_scores: dict[str, float] = {}
    if exp006_path.exists():
        with open(exp006_path, encoding="utf-8") as f:
            exp006_data = json.load(f)
            for r in exp006_data.get("document_results", []):
                exp006_scores[r["test_id"]] = r.get("word_grounding_f1", 0.0)

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
        print(f"RUNNING EXP-007 SIBLING 32-DOCUMENT BENCHMARK: {experiment_id}")
        print(f"Split Filter: {split_filter} | Total Documents: {len(docs)} | Backend: {backend}")
        print("Frozen Baseline: EXP-010 sprint-wave3 = 59.04% Word F1 (EXP-006: 57.10%)")
        print("=======================================================\n")

    suite_start = time.perf_counter()

    for i, item in enumerate(docs, start=1):
        tid = item["test_id"]
        split = item["split"]
        length_class = item["length_class"]
        domain = item["domain"]
        pdf_path = data_dir / f"{tid}.pdf"
        exp004_base = item.get("exp004_baseline", {}).get("word_grounding_f1")
        exp006_base = exp006_scores.get(tid)
        exp010_base = exp010_scores.get(tid)

        test_case = load_test_case(pdf_path)
        if test_case is None:
            if verbose:
                print(f"[{i}/{len(docs)}] ERROR: Could not load test case for {pdf_path}")
            continue

        # 1. Index document with hybrid backend
        t0 = time.perf_counter()
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=enable_ocr, backend=backend)
        t_index = time.perf_counter() - t0

        # 2. Ground extractions using SiblingAdapter
        t1 = time.perf_counter()
        adapter = SiblingAdapter(
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
        delta_004 = (w_f1 - exp004_base) if exp004_base is not None else None
        delta_006 = (w_f1 - exp006_base) if exp006_base is not None else None
        delta_010 = (w_f1 - exp010_base) if exp010_base is not None else None

        cand_rec_1: float | None = None
        cand_rec_5: float | None = None
        cand_rec_20: float | None = None
        ambig_rate: float | None = None
        try:
            cand_rec_1, cand_rec_5, cand_rec_20, ambig_rate = compute_candidate_recalls_and_ambiguity(
                test_case=test_case,
                adapter=adapter,
                citations=citations,
                doc_index=doc_index,
            )
        except Exception:
            pass

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
            candidate_recall_at_1=cand_rec_1,
            candidate_recall_at_5=cand_rec_5,
            candidate_recall_at_20=cand_rec_20,
            ambiguity_rate=ambig_rate,
            exp004_baseline_f1=exp004_base,
            exp010_baseline_f1=exp010_base,
            delta_vs_exp004=delta_004,
            delta_vs_exp010=delta_010,
            delta_vs_exp006=delta_006,
        )
        doc_results.append(res)

        if verbose:
            d010_str = f"({delta_010*100:+5.1f}pp)" if delta_010 is not None else ""
            cr1_str = f"R@1:{cand_rec_1*100:4.1f}% | " if cand_rec_1 is not None else ""
            cr5_str = f"R@5:{cand_rec_5*100:4.1f}% | " if cand_rec_5 is not None else ""
            cr20_str = f"R@20:{cand_rec_20*100:4.1f}% | " if cand_rec_20 is not None else ""
            amb_str = f"Amb:{ambig_rate*100:4.1f}% | " if ambig_rate is not None else ""
            print(
                f"[{i:2d}/{len(docs):2d}] {tid:48s} | "
                f"WF1: {w_f1*100:5.1f}% {d010_str:9s} | "
                f"PF1: {p_f1*100:5.1f}% | "
                f"{cr1_str}"
                f"{cr5_str}"
                f"{cr20_str}"
                f"{amb_str}"
                f"Time: {res.total_time_sec:5.2f}s"
            )

    suite_total_time = time.perf_counter() - suite_start

    def calc_agg(subset: list[LocalDocResult]) -> dict[str, float]:
        if not subset:
            return {}
        cr1_list = [d.candidate_recall_at_1 for d in subset if d.candidate_recall_at_1 is not None]
        cr5_list = [d.candidate_recall_at_5 for d in subset if d.candidate_recall_at_5 is not None]
        cands = [d.candidate_recall_at_20 for d in subset if d.candidate_recall_at_20 is not None]
        ambig_list = [d.ambiguity_rate for d in subset if d.ambiguity_rate is not None]
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
            "mean_candidate_recall_at_1": sum(cr1_list) / len(cr1_list) if cr1_list else 0.0,
            "mean_candidate_recall_at_5": sum(cr5_list) / len(cr5_list) if cr5_list else 0.0,
            "mean_candidate_recall_at_20": sum(cands) / len(cands) if cands else 0.0,
            "mean_ambiguity_rate": sum(ambig_list) / len(ambig_list) if ambig_list else 0.0,
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
        print("EXP-007 SIBLING 32-DOCUMENT BENCHMARK SUMMARY RESULTS")
        print("=" * 60)
        print(f"Total Documents Evaluated: {len(doc_results)}")
        print(f"Total Runtime: {suite_total_time:.2f}s ({suite_total_time/60:.2f} min)")
        print(f"Overall Word Grounding F1:   {overall_agg.get('mean_word_grounding_f1', 0)*100:.2f}% (EXP-010: 59.04%)")
        print(f"Overall Word Precision:      {overall_agg.get('mean_word_precision', 0)*100:.2f}% (EXP-010: 61.23%)")
        print(f"Overall Word Recall:         {overall_agg.get('mean_word_recall', 0)*100:.2f}% (EXP-010: 57.53%)")
        print(f"Overall Page Grounding F1:   {overall_agg.get('mean_page_grounding_f1', 0)*100:.2f}% (EXP-010: 92.59%)")
        print(f"Overall Candidate Recall@1:  {overall_agg.get('mean_candidate_recall_at_1', 0)*100:.2f}%")
        print(f"Overall Candidate Recall@5:  {overall_agg.get('mean_candidate_recall_at_5', 0)*100:.2f}%")
        print(f"Overall Candidate Recall@20: {overall_agg.get('mean_candidate_recall_at_20', 0)*100:.2f}% (EXP-010: 67.82%)")
        print(f"Overall Ambiguity Rate:      {overall_agg.get('mean_ambiguity_rate', 0)*100:.2f}%")
        print(f"Overall False Grounding Rate:{overall_agg.get('mean_false_grounding_rate', 0)*100:.2f}% (EXP-010: 38.77%)\n")
        print("--- By Split ---")
        print(f"Train/Dev F1 (20 docs):      {train_dev_agg.get('mean_word_grounding_f1', 0)*100:.2f}% (EXP-010: 57.43%)")
        print(f"Local Val F1 (12 docs):      {local_val_agg.get('mean_word_grounding_f1', 0)*100:.2f}% (EXP-010: 61.72%)\n")
        print("--- By Length Slice ---")
        print(f"Short Documents F1:          {short_agg.get('mean_word_grounding_f1', 0)*100:.2f}% (EXP-010: 65.48%)")
        print(f"Medium Documents F1:         {medium_agg.get('mean_word_grounding_f1', 0)*100:.2f}% (EXP-010: 45.63%)")
        print(f"Long Documents F1:           {long_agg.get('mean_word_grounding_f1', 0)*100:.2f}% (EXP-010: 70.81%)\n")

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


def main():
    parser = argparse.ArgumentParser(description="Run EXP-007 32-document benchmark")
    parser.add_argument("--manifest", default="benchmarks/exp005_local_manifest.json")
    parser.add_argument("--split", default="all", choices=["all", "train_dev", "local_validation"])
    parser.add_argument("--id", default="EXP-007-sibling-32doc")
    parser.add_argument("--backend", default="hybrid")
    parser.add_argument("--score-margin", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    summary = run_benchmark(
        manifest_path=args.manifest,
        split_filter=args.split,
        experiment_id=args.id,
        backend=args.backend,
        enable_ocr=True,
        score_margin_threshold=args.score_margin,
        limit=args.limit,
    )

    if args.save:
        out_json = Path(f"experiments/{args.id}.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {out_json}")
        update_leaderboard(summary)
        print("Updated leaderboard: experiments/EXP-005-leaderboard.csv")


if __name__ == "__main__":
    main()
