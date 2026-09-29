"""Smoke Benchmark for TonerHound EXP-028B1.

Runs official 6-document ExtractBench test cases and compares against EXP-026.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases import load_test_cases
from extract_bench.test_cases.loader import load_test_case

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.hybrid_index import HybridDocumentIndex
from tonerhound.document.index import DocumentIndex
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


def run_smoke():
    data_dir = root_dir / "research" / "data" / "test"
    out_dir = root_dir / "research" / "experiments" / "EXP-028B1" / "smoke_predictions" / "tonerhound"
    eval_cache_dir = root_dir / "research" / "experiments" / "EXP-028B1" / "smoke_eval_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=== Loading 6 Smoke Benchmark Test Cases ===")
    cases = load_test_cases(data_dir)
    print(f"Loaded {len(cases)} test cases.\n")

    t_start = time.perf_counter()
    doc_results = []

    for c in cases:
        print(f"Running TonerHound on {c.test_id}...")
        t0 = time.perf_counter()
        pdf_path = Path(c.file_path)
        doc_idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="hybrid")
        adapter = _ValidationAdapter(doc_idx, doc_id=c.test_id)

        payload = adapter.ground_extracted_data(
            c.expected_output,
            example_id=c.test_id,
            pipeline_name="tonerhound",
        )
        elapsed_sec = time.perf_counter() - t0

        citations = [
            FieldCitation(
                field_path=cit["field_path"],
                page=cit["page"],
                bbox=cit.get("bbox"),
                reference_text=cit.get("reference_text"),
                confidence=cit.get("confidence"),
                source="tonerhound",
            )
            for cit in payload.get("field_citations", [])
            if cit.get("page") is not None
        ]

        extract_output = ExtractOutput(
            task_type="extract",
            example_id=c.test_id,
            pipeline_name="tonerhound",
            extracted_data=payload.get("extracted_data", c.expected_output if isinstance(c.expected_output, dict) else {}),
            field_citations=citations,
        )

        now = datetime.now(timezone.utc)
        inf_result = InferenceResult(
            request=InferenceRequest(
                example_id=c.test_id,
                source_file_path=str(pdf_path),
                product_type=ProductType.EXTRACT,
            ),
            pipeline_name="tonerhound",
            product_type=ProductType.EXTRACT,
            raw_output={"citations_count": len(citations)},
            output=extract_output,
            started_at=now,
            completed_at=now,
            latency_in_ms=int(elapsed_sec * 1000),
        )

        out_file = out_dir / f"{c.test_id}.result.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(inf_result.model_dump_json(indent=2))

        print(f"  -> Generated {len(citations)} citations in {elapsed_sec:.2f}s")

        # Evaluate immediately with official ExtractEvaluator
        evaluator = ExtractEvaluator()
        eval_res = evaluator.evaluate(inf_result, c)
        
        eval_file = eval_cache_dir / f"{c.test_id}.eval.json"
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_file, "w", encoding="utf-8") as f:
            f.write(eval_res.model_dump_json(indent=2))

        metrics = {m.metric_name: m.value for m in eval_res.metrics}
        doc_results.append({
            "test_id": c.test_id,
            "word_f1": metrics.get("extract_unified_grounded_f1", 0.0),
            "word_precision": metrics.get("extract_unified_grounded_precision", 0.0),
            "word_recall": metrics.get("extract_unified_grounded_recall", 0.0),
            "page_f1": metrics.get("extract_unified_page_f1", 0.0),
            "citations": len(citations),
            "latency": elapsed_sec,
        })

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Document':<50} | {'Word F1':>8} | {'Word Prec':>9} | {'Word Rec':>8} | {'Page F1':>8}")
    print("-" * 80)

    # EXP-026 baseline for comparison
    exp026_baseline = {
        "short/W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025)": 0.5401,
        "short/bianco-2024": 0.3165,
        "short/real_wyo_Goshen_2024": 0.9949,
        "medium/real_pueblo_oct_2025": 0.9960,
        "medium/veralto_earnings_deck_q4fy25": 0.0000,
        "long/real_sm0801_eco_full": 0.9246,
    }

    total_f1 = 0.0
    for r in doc_results:
        tid = r["test_id"]
        b_f1 = exp026_baseline.get(tid, 0.0)
        diff = (r["word_f1"] - b_f1) * 100
        sign = "+" if diff >= 0 else ""
        print(f"{tid:<50} | {r['word_f1']*100:7.2f}% | {r['word_precision']*100:8.2f}% | {r['word_recall']*100:7.2f}% | {r['page_f1']*100:7.2f}% (EXP026: {b_f1*100:.1f}%, {sign}{diff:.2f}pp)")
        total_f1 += r["word_f1"]

    avg_f1 = total_f1 / len(doc_results)
    avg_b = sum(exp026_baseline.values()) / len(exp026_baseline)
    print("-" * 80)
    print(f"{'AVERAGE (6 SMOKE DOCS)':<50} | {avg_f1*100:7.2f}% | (EXP026: {avg_b*100:.2f}%, {('+' if avg_f1>=avg_b else '')}{(avg_f1-avg_b)*100:.2f}pp)")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke()
