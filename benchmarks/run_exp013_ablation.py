"""EXP-013 Candidate Generation Recovery — Ablation & Benchmark Runner.

Measures:
- Candidate Recall@1
- Candidate Recall@5 (Primary Metric, reference ~96.36%)
- Candidate Recall@20
- Word Grounding Precision, Recall, F1
- False Grounding Rate
- Ambiguity Rate
- Runtime (indexing + grounding + evaluation)

Compares:
  A: EXP-011 baseline (Recovery OFF, FlatForm OFF)
  B: EXP-011 + EXP-012 (Recovery OFF, FlatForm ON)
  C: EXP-011 + EXP-012 + EXP-013 (Recovery ON, FlatForm ON)
  D: EXP-011 + EXP-013 (Recovery ON, FlatForm OFF)

Usage::
    python benchmarks/run_exp013_ablation.py --manifest benchmarks/exp012_flat_form_manifest.json --limit 5
    python benchmarks/run_exp013_ablation.py --manifest benchmarks/exp012_flat_form_manifest.json
    python benchmarks/run_exp013_ablation.py --manifest benchmarks/exp005_local_manifest.json --suite 32doc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Any

# Path setup
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

from tonerhound.benchmark.adapter import ExtractBenchAdapter, _flatten_leaves_with_context
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.document.index import DocumentIndex
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import ExtractionInput, ResolutionResult
from tonerhound.resolution.flat_form_reranker import FlatFormLabelReranker
from tonerhound.resolution.resolver import EvidenceResolver

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
CONFIGS = [
    ("A_exp011", {"recovery": False, "flatform": False}),
    ("B_exp011_exp012", {"recovery": False, "flatform": True}),
    ("C_exp011_exp012_exp013", {"recovery": True, "flatform": True}),
    ("D_exp011_exp013", {"recovery": True, "flatform": False}),
]


# ---------------------------------------------------------------------------
# EXP-013 Configurable Resolver
# ---------------------------------------------------------------------------
class _EXP013Resolver(EvidenceResolver):
    """EvidenceResolver configured with optional candidate recovery and FlatFormLabelReranker."""

    def __init__(
        self,
        index: DocumentIndex,
        doc_id: str,
        enable_candidate_recovery: bool = True,
        enable_flatform_reranker: bool = True,
        enable_verification: bool = True,
        score_margin_threshold: float = 0.01,
    ) -> None:
        super().__init__(
            index,
            enable_verification=enable_verification,
            score_margin_threshold=score_margin_threshold,
            enable_candidate_recovery=enable_candidate_recovery,
        )
        self._doc_id = doc_id
        if enable_flatform_reranker:
            self._flat_form = FlatFormLabelReranker(
                index=index,
                doc_id=doc_id,
                enabled=True,
                stages_enabled=frozenset({"vertical", "direction", "label", "suppress", "qualifiers"}),
            )
        else:
            self._flat_form = None

    def resolve(self, extraction: ExtractionInput) -> ResolutionResult:
        if self._flat_form is None or self._flat_form.family is None:
            return super().resolve(extraction)

        result = super().resolve(extraction)
        candidates = self.last_field_candidates.get(extraction.field, [])

        # Apply FlatFormLabelReranker post-selection
        result = self._flat_form.rerank(
            result=result,
            field=extraction.field,
            value=extraction.value,
            candidates=candidates,
            field_context=extraction.field_context,
        )
        return result


class _EXP013Adapter(ExtractBenchAdapter):
    def __init__(
        self,
        index: DocumentIndex,
        doc_id: str,
        enable_candidate_recovery: bool = True,
        enable_flatform_reranker: bool = True,
        enable_verification: bool = True,
        score_margin_threshold: float = 0.01,
    ) -> None:
        super().__init__(
            index,
            enable_structural_disambiguation=True,
            enable_verification=enable_verification,
            score_margin_threshold=score_margin_threshold,
            enable_bbox_precision=True,
            enable_page_fallback=True,
        )
        self.resolver = _EXP013Resolver(
            index=index,
            doc_id=doc_id,
            enable_candidate_recovery=enable_candidate_recovery,
            enable_flatform_reranker=enable_flatform_reranker,
            enable_verification=enable_verification,
            score_margin_threshold=score_margin_threshold,
        )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics_for_doc(
    test_case: Any,
    adapter: _EXP013Adapter,
    citations: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    """Compute Candidate Recall@1, Recall@5, Recall@20, and Ambiguity Rate."""
    cit_by_field = {c["field_path"]: c for c in citations}
    resolver = adapter.resolver
    candidates_by_field = getattr(resolver, "last_field_candidates", {})

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

        pred_cit = cit_by_field.get(field_path)
        is_pred_hit = False
        if pred_cit and pred_cit.get("page") == gt_page and pred_cit.get("bbox"):
            if iou_xywh(pred_cit["bbox"], gt_bbox) >= 0.50:
                is_pred_hit = True

        raw_cands = candidates_by_field.get(field_path, [])
        if len(raw_cands) > 1:
            ambig_count += 1

        if is_pred_hit:
            r1_hits += 1
            r5_hits += 1
            r20_hits += 1
        else:
            if raw_cands and raw_cands[0].page == gt_page and iou_xywh(raw_cands[0].bbox.to_coco(), gt_bbox) >= 0.50:
                r1_hits += 1
                r5_hits += 1
                r20_hits += 1
            elif any(c.page == gt_page and iou_xywh(c.bbox.to_coco(), gt_bbox) >= 0.50 for c in raw_cands[:5]):
                r5_hits += 1
                r20_hits += 1
            elif any(c.page == gt_page and iou_xywh(c.bbox.to_coco(), gt_bbox) >= 0.50 for c in raw_cands[:20]):
                r20_hits += 1

    total = len(rules_with_bbox)
    return r1_hits / total, r5_hits / total, r20_hits / total, ambig_count / total


@dataclass
class DocBenchmarkResult:
    doc_id: str
    config_name: str
    word_grounding_f1: float
    word_grounding_precision: float
    word_grounding_recall: float
    page_grounding_f1: float
    candidate_recall_at_1: float
    candidate_recall_at_5: float
    candidate_recall_at_20: float
    false_grounding_rate: float
    ambiguity_rate: float
    runtime_sec: float


def run_benchmark(
    manifest_path: str,
    configs: list[tuple[str, dict[str, bool]]] | None = None,
    data_dir: str = "research/data/full",
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    if configs is None:
        configs = CONFIGS

    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)

    docs = manifest.get("documents", [])
    if limit is not None:
        docs = docs[:limit]

    data_dir = Path(data_dir)
    results: dict[str, list[DocBenchmarkResult]] = {name: [] for name, _ in configs}
    all_diagnostics: list[dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("EXP-013 ABLATION & BENCHMARK RUNNER")
    print(f"Manifest: {manifest_path} | Docs: {len(docs)}")
    print(f"Configs: {[n for n, _ in configs]}")
    print("=" * 70)

    for i, item in enumerate(docs, 1):
        tid = item.get("id") or item.get("test_id", "")
        if "/" in tid:
            length_dir, doc_name = tid.split("/", 1)
            pdf_path = data_dir / length_dir / f"{doc_name}.pdf"
        else:
            pdf_path = data_dir / f"{tid}.pdf"

        if not pdf_path.exists():
            continue

        test_case = load_test_case(pdf_path)
        if test_case is None:
            continue

        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="hybrid")

        if verbose:
            print(f"\n[{i:2d}/{len(docs):2d}] {tid}")

        for config_name, opts in configs:
            t0 = time.perf_counter()
            adapter = _EXP013Adapter(
                index=doc_index,
                doc_id=tid,
                enable_candidate_recovery=opts["recovery"],
                enable_flatform_reranker=opts["flatform"],
                enable_verification=True,
                score_margin_threshold=0.01,
            )
            payload = adapter.ground_extracted_data(
                test_case.expected_output,
                example_id=tid,
            )
            citations = payload["field_citations"]
            t_run = time.perf_counter() - t0

            # Evaluate with ExtractBench official metric
            metrics = evaluate_prediction(
                expected_output=test_case.expected_output,
                extracted_data=test_case.expected_output,
                field_rules=test_case.test_rules,
                field_citations=citations,
                data_schema=test_case.data_schema,
            )

            w_f1 = metrics.get("word_grounding_f1") or 0.0
            w_prec = metrics.get("word_grounding_precision") or 0.0
            w_rec = metrics.get("word_grounding_recall") or 0.0
            p_f1 = metrics.get("page_grounding_f1") or 0.0
            fg_rate = metrics.get("false_grounding_rate") or 0.0

            r1, r5, r20, ambig = compute_metrics_for_doc(test_case, adapter, citations)

            res = DocBenchmarkResult(
                doc_id=tid,
                config_name=config_name,
                word_grounding_f1=w_f1,
                word_grounding_precision=w_prec,
                word_grounding_recall=w_rec,
                page_grounding_f1=p_f1,
                candidate_recall_at_1=r1,
                candidate_recall_at_5=r5,
                candidate_recall_at_20=r20,
                false_grounding_rate=fg_rate,
                ambiguity_rate=ambig,
                runtime_sec=t_run,
            )
            results[config_name].append(res)

            if verbose:
                print(f"  {config_name:25s} | R@1: {r1*100:5.2f}% | R@5: {r5*100:5.2f}% | R@20: {r20*100:5.2f}% | WF1: {w_f1*100:5.2f}% | Time: {t_run:5.2f}s")

            # Collect diagnostics from recovery engine if enabled
            if opts["recovery"] and adapter.resolver.recovery_engine:
                for d in adapter.resolver.recovery_engine.diagnostics:
                    all_diagnostics.append({
                        "doc_id": tid,
                        "config": config_name,
                        "original_value": str(d.original_value),
                        "raw_document_tokens": list(d.raw_document_tokens),
                        "reconstructed_candidate": d.reconstructed_candidate,
                        "candidate_bbox": [round(v, 5) for v in d.candidate_bbox],
                        "normalization_path": d.normalization_path,
                        "why_previously_missed": d.why_previously_missed,
                        "page": d.page,
                        "mechanism": d.mechanism,
                    })

    # Aggregate summaries
    summary = {}
    print("\n" + "=" * 70)
    print("EXP-013 AGGREGATE SUMMARY")
    print("=" * 70)
    for name, r_list in results.items():
        if not r_list:
            continue
        n = len(r_list)
        avg_r1 = sum(r.candidate_recall_at_1 for r in r_list) / n * 100
        avg_r5 = sum(r.candidate_recall_at_5 for r in r_list) / n * 100
        avg_r20 = sum(r.candidate_recall_at_20 for r in r_list) / n * 100
        avg_wf1 = sum(r.word_grounding_f1 for r in r_list) / n * 100
        avg_wp = sum(r.word_grounding_precision for r in r_list) / n * 100
        avg_wr = sum(r.word_grounding_recall for r in r_list) / n * 100
        avg_pf1 = sum(r.page_grounding_f1 for r in r_list) / n * 100
        avg_fg = sum(r.false_grounding_rate for r in r_list) / n * 100
        avg_amb = sum(r.ambiguity_rate for r in r_list) / n * 100
        tot_time = sum(r.runtime_sec for r in r_list)

        summary[name] = {
            "num_docs": n,
            "candidate_recall_at_1": round(avg_r1, 2),
            "candidate_recall_at_5": round(avg_r5, 2),
            "candidate_recall_at_20": round(avg_r20, 2),
            "word_grounding_f1": round(avg_wf1, 2),
            "word_grounding_precision": round(avg_wp, 2),
            "word_grounding_recall": round(avg_wr, 2),
            "page_grounding_f1": round(avg_pf1, 2),
            "false_grounding_rate": round(avg_fg, 2),
            "ambiguity_rate": round(avg_amb, 2),
            "total_runtime_sec": round(tot_time, 2),
        }
        print(f"Config: {name}")
        print(f"  Candidate Recall@1 : {avg_r1:6.2f}%")
        print(f"  Candidate Recall@5 : {avg_r5:6.2f}% (PRIMARY METRIC)")
        print(f"  Candidate Recall@20: {avg_r20:6.2f}%")
        print(f"  Word Grounding F1  : {avg_wf1:6.2f}% (Prec: {avg_wp:5.2f}%, Rec: {avg_wr:5.2f}%)")
        print(f"  Page Grounding F1  : {avg_pf1:6.2f}%")
        print(f"  False Grounding    : {avg_fg:6.2f}%")
        print(f"  Ambiguity Rate     : {avg_amb:6.2f}%")
        print(f"  Runtime (s)        : {tot_time:6.2f}s")
        print("-" * 70)

    return {
        "summary": summary,
        "diagnostics": all_diagnostics,
        "detailed_results": {k: [asdict(r) for r in v] for k, v in results.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/exp012_flat_form_manifest.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="scratch/exp013_ablation_results.json")
    args = parser.parse_args()

    out_data = run_benchmark(args.manifest, limit=args.limit)
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")
