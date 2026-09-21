"""EXP-017 Validation Benchmark Runner on the 32-Document Frozen Regression Suite.

Compares:
  EXP-015: EXP-011 + EXP-012 + EXP-013 + EXP-015 Safe Character-Span
  EXP-017: EXP-015 + EXP-017 Same-Line Multi-Token Geometry Recovery

Measures:
- Word Grounding F1, Precision, Recall
- Page Grounding F1
- Candidate Recall@1, Recall@5
- Ambiguity Rate
- False Grounding Rate
- Document Wins / Losses / Ties
- Regressions (Requires zero material regressions)

Usage::
    python benchmarks/run_exp017_validation.py --manifest benchmarks/exp005_local_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
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

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput, ResolutionResult
from tonerhound.resolution.flat_form_reranker import FlatFormLabelReranker
from tonerhound.resolution.resolver import EvidenceResolver

CONFIGS = [
    ("EXP-015", {"char_span": True, "same_line": False, "structure_aware": False}),
    ("EXP-017", {"char_span": True, "same_line": True, "structure_aware": False}),
    ("EXP-017R", {"char_span": True, "same_line": True, "structure_aware": True}),
]


class _ValidationResolver(EvidenceResolver):
    def __init__(
        self,
        index: DocumentIndex,
        doc_id: str,
    ) -> None:
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
    def __init__(
        self,
        index: DocumentIndex,
        doc_id: str,
        enable_character_span: bool = False,
        enable_same_line_recovery: bool = False,
        enable_structure_aware_recovery: bool = True,
    ) -> None:
        super().__init__(
            index,
            enable_structural_disambiguation=True,
            enable_verification=True,
            score_margin_threshold=0.01,
            enable_bbox_precision=True,
            enable_page_fallback=True,
            enable_character_span=enable_character_span,
            enable_same_line_recovery=enable_same_line_recovery,
            enable_structure_aware_recovery=enable_structure_aware_recovery,
        )
        self.resolver = _ValidationResolver(index=index, doc_id=doc_id)


def compute_metrics_for_doc(
    test_case: Any,
    adapter: _ValidationAdapter,
    citations: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
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
class DocResult:
    doc_id: str
    config_name: str
    word_grounding_f1: float
    word_grounding_precision: float
    word_grounding_recall: float
    page_grounding_f1: float
    candidate_recall_at_1: float
    candidate_recall_at_5: float
    false_grounding_rate: float
    ambiguity_rate: float
    runtime_sec: float


def run_validation(
    manifest_path: str = "benchmarks/exp005_local_manifest.json",
    data_dir: str = "research/data/full",
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    with open(manifest_path) as f:
        manifest = json.load(f)

    docs = manifest.get("documents", [])
    if limit is not None:
        docs = docs[:limit]

    data_dir = Path(data_dir)
    results: dict[str, list[DocResult]] = {name: [] for name, _ in CONFIGS}

    print("\n" + "=" * 70)
    print("EXP-017 VALIDATION RUNNER: EXP-015 vs EXP-017 (32-DOC SUITE)")
    print(f"Manifest: {manifest_path} | Docs: {len(docs)}")
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

        # Shared document indexing
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="hybrid")

        if verbose:
            print(f"\n[{i:2d}/{len(docs):2d}] {tid}")

        for config_name, opts in CONFIGS:
            t0 = time.perf_counter()
            adapter = _ValidationAdapter(
                index=doc_index,
                doc_id=tid,
                enable_character_span=opts["char_span"],
                enable_same_line_recovery=opts["same_line"],
                enable_structure_aware_recovery=opts.get("structure_aware", True),
            )
            payload = adapter.ground_extracted_data(
                test_case.expected_output,
                example_id=tid,
            )
            citations = payload["field_citations"]
            t_run = time.perf_counter() - t0

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

            r1, r5, _, ambig = compute_metrics_for_doc(test_case, adapter, citations)

            res = DocResult(
                doc_id=tid,
                config_name=config_name,
                word_grounding_f1=w_f1,
                word_grounding_precision=w_prec,
                word_grounding_recall=w_rec,
                page_grounding_f1=p_f1,
                candidate_recall_at_1=r1,
                candidate_recall_at_5=r5,
                false_grounding_rate=fg_rate,
                ambiguity_rate=ambig,
                runtime_sec=t_run,
            )
            results[config_name].append(res)

            if verbose:
                print(f"  {config_name:8s} | WF1: {w_f1*100:5.2f}% | Prec: {w_prec*100:5.2f}% | Rec: {w_rec*100:5.2f}% | PF1: {p_f1*100:5.2f}% | Time: {t_run:4.2f}s")

    # Aggregate summaries
    summary: dict[str, dict[str, Any]] = {}
    print("\n" + "=" * 70)
    print("32-DOCUMENT VALIDATION SUMMARY")
    print("=" * 70)
    for name, r_list in results.items():
        if not r_list:
            continue
        n = len(r_list)
        avg_wf1 = sum(r.word_grounding_f1 for r in r_list) / n * 100
        avg_wp = sum(r.word_grounding_precision for r in r_list) / n * 100
        avg_wr = sum(r.word_grounding_recall for r in r_list) / n * 100
        avg_pf1 = sum(r.page_grounding_f1 for r in r_list) / n * 100
        avg_r1 = sum(r.candidate_recall_at_1 for r in r_list) / n * 100
        avg_r5 = sum(r.candidate_recall_at_5 for r in r_list) / n * 100
        avg_fg = sum(r.false_grounding_rate for r in r_list) / n * 100
        avg_amb = sum(r.ambiguity_rate for r in r_list) / n * 100
        tot_time = sum(r.runtime_sec for r in r_list)

        summary[name] = {
            "num_docs": n,
            "word_grounding_f1": round(avg_wf1, 2),
            "word_grounding_precision": round(avg_wp, 2),
            "word_grounding_recall": round(avg_wr, 2),
            "page_grounding_f1": round(avg_pf1, 2),
            "candidate_recall_at_1": round(avg_r1, 2),
            "candidate_recall_at_5": round(avg_r5, 2),
            "false_grounding_rate": round(avg_fg, 2),
            "ambiguity_rate": round(avg_amb, 2),
            "total_runtime_sec": round(tot_time, 2),
        }
        print(f"Config: {name}")
        print(f"  Word Grounding F1 : {avg_wf1:6.2f}% (Prec: {avg_wp:5.2f}%, Rec: {avg_wr:5.2f}%)")
        print(f"  Page Grounding F1 : {avg_pf1:6.2f}%")
        print(f"  Recall@1          : {avg_r1:6.2f}%")
        print(f"  Recall@5          : {avg_r5:6.2f}%")
        print(f"  False Grounding   : {avg_fg:6.2f}%")
        print(f"  Ambiguity Rate    : {avg_amb:6.2f}%")
        print(f"  Runtime (s)       : {tot_time:6.2f}s")
        print("-" * 70)

    # Document-level head-to-head comparisons
    exp015_map = {r.doc_id: r for r in results["EXP-015"]}
    exp017_map = {r.doc_id: r for r in results.get("EXP-017", [])}
    exp017r_map = {r.doc_id: r for r in results.get("EXP-017R", [])}

    def _compare(cand_map: dict[str, Any], label: str) -> dict[str, Any]:
        wins, losses, ties = [], [], []
        for doc_id, r15 in exp015_map.items():
            rcand = cand_map.get(doc_id)
            if not rcand:
                continue
            diff = (rcand.word_grounding_f1 - r15.word_grounding_f1) * 100
            if diff > 0.05:
                wins.append((doc_id, diff, r15.word_grounding_f1 * 100, rcand.word_grounding_f1 * 100))
            elif diff < -0.05:
                losses.append((doc_id, diff, r15.word_grounding_f1 * 100, rcand.word_grounding_f1 * 100))
            else:
                ties.append((doc_id, diff))
        print(f"\nHEAD-TO-HEAD COMPARISON ({label} vs EXP-015):")
        print(f"  Wins   : {len(wins)}")
        for w in wins:
            print(f"    + {w[0]}: {w[2]:.2f}% -> {w[3]:.2f}% ({w[1]:+.2f} pp)")
        print(f"  Losses : {len(losses)}")
        for l in losses:
            print(f"    - {l[0]}: {l[2]:.2f}% -> {l[3]:.2f}% ({l[1]:+.2f} pp)")
        print(f"  Ties   : {len(ties)}")
        return {"wins": wins, "losses": losses, "ties": len(ties)}

    comp_017 = _compare(exp017_map, "EXP-017") if exp017_map else {}
    comp_017r = _compare(exp017r_map, "EXP-017R") if exp017r_map else {}

    return {
        "summary": summary,
        "exp017_vs_exp015": comp_017,
        "exp017r_vs_exp015": comp_017r,
        "detailed_results": {k: [asdict(r) for r in v] for k, v in results.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/exp005_local_manifest.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="scratch/exp017_validation_results.json")
    args = parser.parse_args()

    out_data = run_validation(args.manifest, limit=args.limit)
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")
