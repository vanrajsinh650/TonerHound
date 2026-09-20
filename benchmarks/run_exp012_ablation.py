"""EXP-012 Flat-Form Label-Grounded Reranker — Ablation Runner.

Measures word grounding F1, page F1, Candidate Recall@1, and wrong-occurrence
count for each of 6 pipeline configurations (A-F) on the 34-doc flat-form
evaluation cohort, then runs the 32-doc regression gate.

Ablation configurations:
  A: EXP-011 baseline (FlatFormLabelReranker disabled)
  B: + vertical family-gated filter only
  C: + directional relationship scoring
  D: + label proximity re-selection
  E: + competing-label suppression
  F: Full EXP-012 (identical to E — future tuning target)

Usage::

    python benchmarks/run_exp012_ablation.py
    python benchmarks/run_exp012_ablation.py --limit 5 --verbose
    python benchmarks/run_exp012_ablation.py --config A B --save
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — must happen before any tonerhound imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_REF_EXTRACTBENCH = _REPO_ROOT / "research" / "reference" / "ExtractBench" / "src"
if _REF_EXTRACTBENCH.exists() and str(_REF_EXTRACTBENCH) not in sys.path:
    sys.path.insert(0, str(_REF_EXTRACTBENCH))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# ExtractBench imports
# ---------------------------------------------------------------------------
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from extract_bench.test_cases.loader import load_test_case
from extract_bench.test_cases.schema import iter_rule_evidence

# ---------------------------------------------------------------------------
# TonerHound imports
# ---------------------------------------------------------------------------
from tonerhound.benchmark.adapter import ExtractBenchAdapter, _flatten_leaves_with_context
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.document.index import DocumentIndex
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import ExtractionInput, ResolutionResult
from tonerhound.resolution.resolver import EvidenceResolver

# ---------------------------------------------------------------------------
# FlatFormLabelReranker — graceful import
# ---------------------------------------------------------------------------
try:
    from tonerhound.resolution.flat_form_reranker import FlatFormLabelReranker
    HAS_FLAT_FORM = True
except ImportError as _e:
    HAS_FLAT_FORM = False
    print(f"[WARNING] FlatFormLabelReranker not available: {_e}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
#   Name          stages_enabled (empty = EXP-011 baseline, no reranker)
CONFIGS: list[tuple[str, set[str]]] = [
    ("A_baseline",  set()),                                              # EXP-011 baseline
    ("B_vertical",  {"vertical"}),                                       # + vertical filter
    ("C_direction", {"vertical", "direction"}),                          # + directional
    ("D_label",     {"vertical", "direction", "label"}),                 # + label proximity
    ("E_suppress",  {"vertical", "direction", "label", "suppress"}),     # + competing suppression
    ("F_full",      {"vertical", "direction", "label", "suppress", "qualifiers"}),  # Full EXP-012
]

# ---------------------------------------------------------------------------
# FlatForm-aware Resolver — patches EvidenceResolver.resolve() to capture candidates
# ---------------------------------------------------------------------------

class _FlatFormResolver(EvidenceResolver):
    """Subclasses EvidenceResolver to intercept resolve() and apply FlatFormLabelReranker.

    The FlatFormLabelReranker is a *post-selection* filter: it receives the
    ResolutionResult chosen by the upstream reranker together with the full
    raw candidate list, and may redirect to a better-grounded candidate.

    We capture raw candidates by monkey-patching the matcher's find methods.
    """

    def __init__(
        self,
        index: DocumentIndex,
        doc_id: str,
        stages_enabled: set[str],
        enable_verification: bool = True,
        score_margin_threshold: float = 0.01,
    ) -> None:
        super().__init__(
            index,
            enable_verification=enable_verification,
            score_margin_threshold=score_margin_threshold,
        )
        self._doc_id = doc_id
        self._stages_enabled = stages_enabled
        if HAS_FLAT_FORM and stages_enabled:
            self._flat_form = FlatFormLabelReranker(
                index=index,
                doc_id=doc_id,
                enabled=True,
                stages_enabled=frozenset(stages_enabled),
            )
        else:
            self._flat_form = None

    def resolve(self, extraction: ExtractionInput) -> ResolutionResult:
        """Resolve and optionally apply FlatFormLabelReranker post-selection."""
        if self._flat_form is None or self._flat_form.family is None:
            # Config A: straight EXP-011 baseline — no flat-form reranking
            return super().resolve(extraction)

        # Collect raw candidates (mirror the candidate generation logic from parent)
        candidates = self._collect_candidates(extraction)

        # Run standard resolution (EXP-011)
        result = super().resolve(extraction)

        # Post-selection: apply FlatFormLabelReranker
        result = self._flat_form.rerank(
            result=result,
            field=extraction.field,
            value=extraction.value,
            candidates=candidates,
            field_context=extraction.field_context,
        )
        return result

    def _collect_candidates(self, extraction: ExtractionInput) -> list[MatchCandidate]:
        """Re-run candidate generation to give the reranker the full candidate list."""
        return self.collect_candidates(extraction)


# ---------------------------------------------------------------------------
# FlatFormAwareAdapter — wraps ExtractBenchAdapter with patched resolver
# ---------------------------------------------------------------------------

class FlatFormAwareAdapter(ExtractBenchAdapter):
    """Subclass of ExtractBenchAdapter that injects FlatFormLabelReranker post-selection.

    This replaces the adapter's default EvidenceResolver with our patched
    _FlatFormResolver, which applies the reranker after each field resolution.
    """

    def __init__(
        self,
        index: DocumentIndex,
        doc_id: str,
        stages_enabled: set[str],
        enable_structural_disambiguation: bool = True,
        enable_verification: bool = True,
        score_margin_threshold: float = 0.01,
        enable_bbox_precision: bool = True,
        enable_page_fallback: bool = True,
    ) -> None:
        super().__init__(
            index,
            enable_structural_disambiguation=enable_structural_disambiguation,
            enable_verification=enable_verification,
            score_margin_threshold=score_margin_threshold,
            enable_bbox_precision=enable_bbox_precision,
            enable_page_fallback=enable_page_fallback,
        )
        # Replace the resolver with our FlatForm-aware version
        self.resolver = _FlatFormResolver(
            index=index,
            doc_id=doc_id,
            stages_enabled=stages_enabled,
            enable_verification=enable_verification,
            score_margin_threshold=score_margin_threshold,
        )


# ---------------------------------------------------------------------------
# Candidate Recall@1 computation (adapted from run_exp007_sibling_benchmark.py)
# ---------------------------------------------------------------------------

def compute_candidate_recall_at_1(
    test_case: Any,
    citations: list[dict[str, Any]],
) -> tuple[float, int]:
    """Compute Candidate Recall@1 for the given predictions.

    Returns (recall_at_1, wrong_occurrence_count).
    wrong_occurrence_count = number of fields where a citation was produced
    but it doesn't IoU-match the GT bbox (predicted wrong occurrence).
    """
    rules_with_bbox = []
    for r in test_case.test_rules:
        ev_list = list(iter_rule_evidence(r))
        ev_with_box = [e for e in ev_list if e.bbox is not None and e.page is not None]
        if ev_with_box:
            rules_with_bbox.append((r, ev_with_box[0]))

    if not rules_with_bbox:
        return 1.0, 0

    cit_by_field = {c["field_path"]: c for c in citations}
    r1_hits = 0
    wrong_occur = 0

    for rule, ev in rules_with_bbox:
        field_path = rule.field_path
        gt_page = ev.page
        gt_bbox = ev.bbox

        pred_cit = cit_by_field.get(field_path)
        if pred_cit and pred_cit.get("page") == gt_page and pred_cit.get("bbox"):
            if iou_xywh(pred_cit["bbox"], gt_bbox) >= 0.50:
                r1_hits += 1
            else:
                wrong_occur += 1

    total = len(rules_with_bbox)
    return r1_hits / total, wrong_occur


# ---------------------------------------------------------------------------
# Per-doc result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AblationDocResult:
    test_id: str
    config_name: str
    length_class: str
    domain: str
    family: str
    num_pages: int
    word_grounding_f1: float
    word_grounding_precision: float
    word_grounding_recall: float
    page_grounding_f1: float
    candidate_recall_at_1: float
    wrong_occur_count: int
    grounding_time_sec: float


# ---------------------------------------------------------------------------
# Core ablation benchmark
# ---------------------------------------------------------------------------

def run_ablation_benchmark(
    manifest_path: str = "benchmarks/exp012_flat_form_manifest.json",
    configs: list[tuple[str, set[str]]] | None = None,
    data_dir: str = "research/data/full",
    backend: str = "hybrid",
    enable_ocr: bool = True,
    limit: int | None = None,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run ablation benchmark for all specified configurations.

    Returns a dict of config_name -> aggregate metrics dict.
    """
    if configs is None:
        configs = CONFIGS

    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    docs = manifest["documents"]
    if limit is not None:
        docs = docs[:limit]

    data_dir = Path(data_dir)
    results: dict[str, list[AblationDocResult]] = {name: [] for name, _ in configs}

    if verbose:
        print("\n" + "=" * 65)
        print("EXP-012 ABLATION RUNNER")
        print(f"Manifest: {manifest_path} | Docs: {len(docs)}")
        print(f"Backend: {backend} | Configs: {[n for n, _ in configs]}")
        print("=" * 65)

    suite_start = time.perf_counter()

    for i, item in enumerate(docs, start=1):
        # Support both 'id' (exp012 manifest) and 'test_id' (exp005 manifest)
        tid = item.get("id") or item.get("test_id", "")
        length_class = item.get("length_class", "short")
        domain = item.get("domain", "unknown")
        family = item.get("family", "unknown")

        # Resolve PDF path — tid may be "short/docname" or just "docname"
        if "/" in tid:
            length_dir, doc_name = tid.split("/", 1)
            pdf_path = data_dir / length_dir / f"{doc_name}.pdf"
            test_json_path = data_dir / length_dir / f"{doc_name}.test.json"
        else:
            pdf_path = data_dir / f"{tid}.pdf"
            test_json_path = data_dir / f"{tid}.test.json"

        if verbose:
            print(f"\n[{i:2d}/{len(docs):2d}] {tid}")

        test_case = load_test_case(pdf_path)
        if test_case is None:
            if verbose:
                print(f"  ERROR: Could not load test case for {pdf_path}")
            continue

        # Index document once per doc (shared across configs)
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=enable_ocr, backend=backend)

        for config_name, stages in configs:
            t0 = time.perf_counter()

            adapter = FlatFormAwareAdapter(
                index=doc_index,
                doc_id=tid,
                stages_enabled=stages,
                enable_structural_disambiguation=True,
                enable_verification=True,
                score_margin_threshold=0.01,
                enable_bbox_precision=True,
            )
            payload = adapter.ground_extracted_data(
                test_case.expected_output,
                example_id=tid,
            )
            citations = payload["field_citations"]
            t_ground = time.perf_counter() - t0

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

            r1, wrong_occur = compute_candidate_recall_at_1(test_case, citations)

            res = AblationDocResult(
                test_id=tid,
                config_name=config_name,
                length_class=length_class,
                domain=domain,
                family=family,
                num_pages=len(doc_index.pages),
                word_grounding_f1=w_f1,
                word_grounding_precision=w_prec,
                word_grounding_recall=w_rec,
                page_grounding_f1=p_f1,
                candidate_recall_at_1=r1,
                wrong_occur_count=wrong_occur,
                grounding_time_sec=t_ground,
            )
            results[config_name].append(res)

            if verbose:
                print(
                    f"  [{config_name:12s}] "
                    f"WF1: {w_f1*100:5.1f}% | "
                    f"PF1: {p_f1*100:5.1f}% | "
                    f"R@1: {r1*100:5.1f}% | "
                    f"WrongOccur: {wrong_occur:4d} | "
                    f"Time: {t_ground:.2f}s"
                )

    suite_time = time.perf_counter() - suite_start

    # -----------------------------------------------------------------------
    # Aggregate metrics per config
    # -----------------------------------------------------------------------
    aggregates: dict[str, dict[str, Any]] = {}
    baseline_r1: float | None = None

    for config_name, _ in configs:
        doc_results = results[config_name]
        if not doc_results:
            aggregates[config_name] = {}
            continue

        n = len(doc_results)
        agg = {
            "config": config_name,
            "num_docs": n,
            "mean_word_grounding_f1": sum(r.word_grounding_f1 for r in doc_results) / n,
            "mean_page_grounding_f1": sum(r.page_grounding_f1 for r in doc_results) / n,
            "mean_candidate_recall_at_1": sum(r.candidate_recall_at_1 for r in doc_results) / n,
            "total_wrong_occur": sum(r.wrong_occur_count for r in doc_results),
            "total_grounding_sec": sum(r.grounding_time_sec for r in doc_results),
            "document_results": [asdict(r) for r in doc_results],
        }

        if config_name == "A_baseline":
            baseline_r1 = agg["mean_candidate_recall_at_1"]

        agg["delta_r1_vs_A"] = (
            agg["mean_candidate_recall_at_1"] - baseline_r1
            if baseline_r1 is not None
            else None
        )

        # Compute win/neutral/regression vs baseline (per doc WF1)
        if config_name != "A_baseline" and "A_baseline" in aggregates:
            a_docs = {r["test_id"]: r["word_grounding_f1"] for r in aggregates["A_baseline"]["document_results"]}
            wins = neutrals = regressions = 0
            for r in doc_results:
                a_wf1 = a_docs.get(r.test_id, r.word_grounding_f1)
                diff = r.word_grounding_f1 - a_wf1
                if diff > 0.001:
                    wins += 1
                elif diff < -0.001:
                    regressions += 1
                else:
                    neutrals += 1
            agg["wins_vs_A"] = wins
            agg["neutrals_vs_A"] = neutrals
            agg["regressions_vs_A"] = regressions

        aggregates[config_name] = agg

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n\nTotal suite time: {suite_time:.1f}s ({suite_time/60:.1f} min)")
        _print_comparison_table(aggregates, configs)

    return aggregates


def _print_comparison_table(
    aggregates: dict[str, dict[str, Any]],
    configs: list[tuple[str, set[str]]],
) -> None:
    """Print a formatted comparison table of ablation results."""
    print("\n" + "=" * 75)
    print("====== EXP-012 ABLATION RESULTS ======")
    print("=" * 75)
    header = (
        f"{'Config':16s} | {'Docs':4s} | {'WF1%':6s} | {'PF1%':6s} | "
        f"{'R@1%':6s} | {'WrongOccur':11s} | {'Δ R@1 vs A':10s}"
    )
    print(header)
    print("-" * 75)

    baseline_r1 = None
    for config_name, _ in configs:
        agg = aggregates.get(config_name, {})
        if not agg:
            print(f"{config_name:16s} | -- no data --")
            continue

        n = agg.get("num_docs", 0)
        wf1 = agg.get("mean_word_grounding_f1", 0.0) * 100
        pf1 = agg.get("mean_page_grounding_f1", 0.0) * 100
        r1 = agg.get("mean_candidate_recall_at_1", 0.0) * 100
        wrong = agg.get("total_wrong_occur", 0)
        delta_r1 = agg.get("delta_r1_vs_A")

        if config_name == "A_baseline":
            baseline_r1 = agg.get("mean_candidate_recall_at_1", 0.0)
            delta_str = "   ---"
        else:
            delta_str = f"{delta_r1*100:+6.1f}pp" if delta_r1 is not None else "   n/a"

        print(
            f"{config_name:16s} | {n:4d} | {wf1:6.1f}% | {pf1:6.1f}% | "
            f"{r1:6.1f}% | {wrong:11d} | {delta_str:10s}"
        )

    print("=" * 75)

    # Per-family breakdown for the last config (F_full)
    last_config = configs[-1][0]
    last_agg = aggregates.get(last_config, {})
    if last_agg.get("document_results"):
        print(f"\n--- Family breakdown ({last_config}) ---")
        family_map: dict[str, list] = {}
        for r in last_agg["document_results"]:
            fam = r.get("family", "unknown")
            family_map.setdefault(fam, []).append(r)
        for fam, docs in sorted(family_map.items()):
            n = len(docs)
            mean_r1 = sum(d["candidate_recall_at_1"] for d in docs) / n * 100
            mean_wf1 = sum(d["word_grounding_f1"] for d in docs) / n * 100
            total_wo = sum(d["wrong_occur_count"] for d in docs)
            print(f"  {fam:12s}: {n:2d} docs | WF1: {mean_wf1:5.1f}% | R@1: {mean_r1:5.1f}% | WrongOccur: {total_wo:4d}")


# ---------------------------------------------------------------------------
# 32-doc Regression Gate
# ---------------------------------------------------------------------------

def run_regression_gate(
    manifest_path: str = "benchmarks/exp005_local_manifest.json",
    frozen_baseline_path: str = "benchmarks/exp011_frozen_baseline.json",
    tolerance: float = 0.01,
    backend: str = "hybrid",
    enable_ocr: bool = True,
    verbose: bool = True,
) -> bool:
    """Run the 32-doc regression gate to ensure no regressions vs EXP-011 frozen baseline.

    Each doc's WF1 must be within -1.0pp of the frozen EXP-011 baseline.

    Returns True iff all docs pass the tolerance gate.
    """
    if verbose:
        print("\n" + "=" * 65)
        print("32-DOC REGRESSION GATE (EXP-011 frozen baseline)")
        print("=" * 65)

    # Load frozen baseline
    with open(frozen_baseline_path, encoding="utf-8") as f:
        frozen = json.load(f)
    per_doc_baseline: dict[str, float] = {
        k: v["word_grounding_f1"]
        for k, v in frozen.get("per_document_baseline", {}).items()
    }

    # Load manifest
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    docs = manifest["documents"]

    data_dir = Path("research/data/full")
    failures: list[str] = []
    all_pass = True

    for i, item in enumerate(docs, start=1):
        tid = item.get("test_id", item.get("id", ""))
        pdf_path = data_dir / f"{tid}.pdf"

        test_case = load_test_case(pdf_path)
        if test_case is None:
            if verbose:
                print(f"[{i:2d}/{len(docs)}] SKIP (no test case): {tid}")
            continue

        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=enable_ocr, backend=backend)

        # Use full EXP-012 (config F)
        adapter = FlatFormAwareAdapter(
            index=doc_index,
            doc_id=tid,
            stages_enabled={"vertical", "direction", "label", "suppress", "qualifiers"},
            enable_structural_disambiguation=True,
            enable_verification=True,
            score_margin_threshold=0.01,
            enable_bbox_precision=True,
        )
        payload = adapter.ground_extracted_data(test_case.expected_output, example_id=tid)
        citations = payload["field_citations"]

        metrics = evaluate_prediction(
            expected_output=test_case.expected_output,
            extracted_data=test_case.expected_output,
            field_rules=test_case.test_rules,
            field_citations=citations,
            data_schema=test_case.data_schema,
        )
        w_f1 = metrics.get("word_grounding_f1") or 0.0

        # Lookup frozen baseline (key may be "short/docname" or just "docname")
        base_f1 = per_doc_baseline.get(tid)
        if base_f1 is None:
            # Try stripping length prefix
            bare = tid.split("/")[-1] if "/" in tid else tid
            for k, v in per_doc_baseline.items():
                if bare == k.split("/")[-1]:
                    base_f1 = v
                    break

        delta = (w_f1 - base_f1) if base_f1 is not None else None
        passed = delta is None or delta >= -tolerance

        if not passed:
            all_pass = False
            failures.append(tid)

        if verbose:
            base_str = f"{base_f1*100:5.1f}%" if base_f1 is not None else "  N/A "
            delta_str = f"{delta*100:+5.1f}pp" if delta is not None else "   N/A"
            status = "✓" if passed else "✗ FAIL"
            print(
                f"[{i:2d}/{len(docs):2d}] {tid:50s} | "
                f"WF1: {w_f1*100:5.1f}% (base: {base_str}) {delta_str} {status}"
            )

    if verbose:
        print("\n" + "=" * 65)
        if all_pass:
            print("✓ REGRESSION GATE PASSED — all docs within tolerance")
        else:
            print(f"✗ REGRESSION GATE FAILED — {len(failures)} doc(s) regressed:")
            for f in failures:
                print(f"  - {f}")
        print("=" * 65)

    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-012 ablation runner")
    parser.add_argument(
        "--manifest", default="benchmarks/exp012_flat_form_manifest.json",
        help="Path to the 34-doc flat-form evaluation manifest",
    )
    parser.add_argument(
        "--config", nargs="+",
        choices=[n for n, _ in CONFIGS],
        default=None,
        help="Which configurations to run (default: all A-F)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of docs")
    parser.add_argument("--backend", default="hybrid", choices=["hybrid", "pdfium", "liteparse"])
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-regression-gate", action="store_true",
                        help="Skip the 32-doc regression gate")
    parser.add_argument("--save", action="store_true",
                        help="Save results to scratch/exp012_ablation_results.json")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Filter configs if requested
    selected_configs: list[tuple[str, set[str]]] = CONFIGS
    if args.config:
        selected_configs = [(n, s) for n, s in CONFIGS if n in args.config]

    # Run ablation
    aggregates = run_ablation_benchmark(
        manifest_path=args.manifest,
        configs=selected_configs,
        backend=args.backend,
        enable_ocr=not args.no_ocr,
        limit=args.limit,
        verbose=verbose,
    )

    # Save results
    if args.save:
        out_path = Path("scratch/exp012_ablation_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Strip large document_results for top-level save to keep file manageable
        save_data = {}
        for config_name, agg in aggregates.items():
            save_data[config_name] = {k: v for k, v in agg.items() if k != "document_results"}
            save_data[config_name]["document_results"] = agg.get("document_results", [])

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved ablation results to {out_path}")

    # Run regression gate
    if not args.no_regression_gate:
        gate_passed = run_regression_gate(
            backend=args.backend,
            enable_ocr=not args.no_ocr,
            verbose=verbose,
        )
        if not gate_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
