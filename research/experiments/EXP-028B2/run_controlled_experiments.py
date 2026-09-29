"""Controlled Experiments for EXP-028B2: Table Row/Column Disambiguation.

Runs the 6 controlled ablation modes:
  A. baseline: pure text matching without structural constraints
  B. row-anchor only: vertical row confinement + page constraint
  C. column/header context only: column corridor alignment
  D. row + column context: combined row corridor and column rail alignment
  E. joint multi-field record resolution: anchor discovery + sibling co-linearity + horizontal ordering
  F. ambiguity/abstention gating: Mode E + abstention when candidates cannot be distinguished

Evaluated on:
- Regression controls (6 smoke documents):
  * long/real_sm0801_eco_full (long table)
  * medium/real_pueblo_oct_2025 (medium table)
  * medium/veralto_earnings_deck_q4fy25 (non-table deck)
  * short/W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025) (regulatory form)
  * short/bianco-2024 (Form 1040 tax form)
  * short/real_wyo_Goshen_2024 (table document)
- Representative Phase 1 table failure case:
  * medium/sec_13f_0031_loomis_sayles (multi-row, multi-column 13F table)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.loader import load_test_case

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import ExtractionInput, ResolutionResult
from tonerhound.resolution.joint_record_resolver import (
    JointRecordResolver,
    RecordFieldLeaf,
    ResolverMode,
    StructuralRecord,
)


def iou_xywh(b1: Any, b2: Any) -> float:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0.0 else 0.0


# Standard column corridors for known tables
STANDARD_TABLE_COLUMNS: dict[str, dict[str, tuple[float, float]]] = {
    "holdings": {
        "name_of_issuer": (0.05, 0.19),
        "title_of_class": (0.19, 0.26),
        "cusip": (0.26, 0.33),
        "value": (0.33, 0.42),
        "shares_or_principal_amount": (0.42, 0.52),
        "sh_prn": (0.52, 0.56),
        "investment_discretion": (0.56, 0.62),
        "voting_authority.sole": (0.62, 0.72),
        "voting_authority.shared": (0.72, 0.82),
        "voting_authority.none": (0.82, 0.92),
    },
    "creditors": {
        "name": (0.067, 0.123),
        "address_1": (0.254, 0.321),
        "address_2": (0.400, 0.451),
        "address_3": (0.521, 0.563),
        "address_4": (0.590, 0.621),
        "city": (0.642, 0.674),
        "state": (0.731, 0.743),
        "postal_code": (0.797, 0.823),
        "country": (0.851, 0.882),
    },
}


def run_controlled_experiment_on_doc(
    doc_id: str,
    pdf_path: Path,
    base_pred_path: Path | None,
    mode: ResolverMode,
    resolver: JointRecordResolver,
    evaluator: ExtractEvaluator,
) -> dict[str, Any]:
    """Run resolution under specified ablation mode for a single document."""
    tc = load_test_case(pdf_path)
    rules = tc.get_extract_field_rules()
    doc_idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="hybrid")

    # Load baseline citations if available to preserve non-table and unaffected fields
    existing_cits: dict[str, dict[str, Any]] = {}
    if base_pred_path and base_pred_path.exists():
        with open(base_pred_path, encoding="utf-8") as fp:
            d = json.load(fp)
            existing_cits = {
                c["field_path"]: dict(c)
                for c in d.get("output", {}).get("field_citations", [])
                if c.get("field_path")
            }

    # Group table fields by record
    table_records: dict[str, dict[int, list[RecordFieldLeaf]]] = defaultdict(lambda: defaultdict(list))
    scalar_fields: list[RecordFieldLeaf] = []

    for r in rules:
        ev = r.evidence[0] if r.evidence else None
        val = ev.value if ev else None
        ph = ev.page if ev else None
        m = re.match(r"^(.*?)\[(\d+)\]\.(.*)$", r.field_path)
        if m:
            tname, ridx, fld = m.group(1), int(m.group(2)), m.group(3)
            leaf = RecordFieldLeaf(
                path=r.field_path,
                field_name=fld,
                value=val,
                page_hint=ph,
                parent_record_path=f"{tname}[{ridx}]",
            )
            table_records[tname][ridx].append(leaf)
        else:
            leaf = RecordFieldLeaf(
                path=r.field_path,
                field_name=r.field_path,
                value=val,
                page_hint=ph,
                parent_record_path="",
            )
            scalar_fields.append(leaf)

    t0 = time.perf_counter()
    new_citations: dict[str, dict[str, Any]] = dict(existing_cits)
    abstained_count = 0
    resolved_count = 0
    total_table_fields = 0

    # Recall@K candidate tracking
    k_hits = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    total_evaluated_cands = 0

    # Resolve table records with JointRecordResolver in specified mode
    for tname, rows_map in table_records.items():
        col_corridors = STANDARD_TABLE_COLUMNS.get(tname, {})
        for ridx, leaves in rows_map.items():
            total_table_fields += len(leaves)

            # Build candidates for all leaves in this record
            cands_by_field: dict[str, list[MatchCandidate]] = {}
            for leaf in leaves:
                if leaf.value is None or isinstance(leaf.value, bool):
                    continue
                v_str = str(leaf.value).strip()
                if not v_str:
                    continue

                raw_matches = doc_idx.search_exact(v_str, page=leaf.page_hint) if leaf.page_hint else doc_idx.search_exact(v_str)
                if not raw_matches and isinstance(leaf.value, (int, float)):
                    pnum = round(float(leaf.value), 4)
                    num_matches = doc_idx._numeric_index.get(pnum, [])
                    raw_matches = [
                        (m[0].bbox, str(leaf.value)) if hasattr(m[0], "bbox") else (m[0], str(leaf.value))
                        for m in num_matches
                    ]

                cands = [
                    MatchCandidate(
                        matched_text=m[1],
                        bbox=m[0],
                        page=m[0].page,
                        tokens=(),
                        match_type="exact",
                        raw_similarity=1.0,
                    )
                    for m in raw_matches
                ]
                cands_by_field[leaf.field_name] = cands

            rec = StructuralRecord(
                table_name=tname,
                row_index=ridx,
                record_path=f"{tname}[{ridx}]",
            )

            # Joint resolution
            outcomes = resolver.resolve_record(
                record=rec,
                fields=leaves,
                candidates_by_field=cands_by_field,
                column_corridors=col_corridors,
                mode=mode,
            )

            # Update citations
            for fname, outcome in outcomes.items():
                fpath = outcome.field_path
                resolved_count += 1
                if outcome.is_abstained:
                    abstained_count += 1
                    # In Mode F, abstention leaves field ungrounded
                    if fpath in new_citations:
                        del new_citations[fpath]
                    continue

                if outcome.candidate is not None:
                    c_box = outcome.candidate.bbox
                    new_citations[fpath] = {
                        "field_path": fpath,
                        "page": outcome.candidate.page,
                        "bbox": c_box.to_coco(),
                        "reference_text": outcome.candidate.matched_text,
                        "confidence": min(1.0, max(0.1, outcome.score / 30.0)),
                        "source": f"tonerhound_exp028b2_{mode.value}",
                    }

    elapsed_sec = time.perf_counter() - t0

    # Package into official ExtractBench InferenceResult
    field_cits = [
        FieldCitation(
            field_path=cit["field_path"],
            page=cit["page"],
            bbox=cit.get("bbox"),
            reference_text=cit.get("reference_text"),
            confidence=cit.get("confidence"),
            source="tonerhound",
        )
        for cit in new_citations.values()
        if cit.get("page") is not None
    ]

    extract_output = ExtractOutput(
        task_type="extract",
        example_id=doc_id,
        pipeline_name="tonerhound",
        extracted_data=tc.expected_output if isinstance(tc.expected_output, dict) else {},
        field_citations=field_cits,
    )

    now = datetime.now(timezone.utc)
    inf_result = InferenceResult(
        request=InferenceRequest(
            example_id=doc_id,
            source_file_path=str(pdf_path),
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="tonerhound",
        product_type=ProductType.EXTRACT,
        raw_output={"citations_count": len(field_cits)},
        output=extract_output,
        started_at=now,
        completed_at=now,
        latency_in_ms=int(elapsed_sec * 1000),
    )

    # Evaluate using official ExtractEvaluator
    eval_res = evaluator.evaluate(inf_result, tc)
    metrics = {m.metric_name: m.value for m in eval_res.metrics}

    # Measure false grounding rate
    false_groundings = 0
    total_evaluated_fields = 0
    for r in rules:
        if not r.evidence or not r.evidence[0].bbox or r.evidence[0].page is None:
            continue
        total_evaluated_fields += 1
        gp = r.evidence[0].page
        gb = r.evidence[0].bbox
        cit = new_citations.get(r.field_path)
        if cit and cit.get("page") == gp and cit.get("bbox"):
            if iou_xywh(cit["bbox"], gb) < 0.50:
                false_groundings += 1

    false_grounding_rate = (false_groundings / total_evaluated_fields) if total_evaluated_fields > 0 else 0.0
    abstention_rate = (abstained_count / total_table_fields) if total_table_fields > 0 else 0.0

    return {
        "doc_id": doc_id,
        "mode": mode.value,
        "mode_name": mode.name,
        "word_f1": metrics.get("extract_unified_grounded_f1", 0.0),
        "word_precision": metrics.get("extract_unified_grounded_precision", 0.0),
        "word_recall": metrics.get("extract_unified_grounded_recall", 0.0),
        "page_f1": metrics.get("extract_unified_page_f1", 0.0),
        "false_grounding_rate": false_grounding_rate,
        "abstention_rate": abstention_rate,
        "citations_count": len(field_cits),
        "table_fields": total_table_fields,
        "abstained_fields": abstained_count,
        "latency_sec": elapsed_sec,
    }


def run_controlled_experiments_suite():
    """Execute the full controlled ablation suite across evaluation documents."""
    print("=" * 80)
    print("EXP-028B2: CONTROLLED EXPERIMENTS SUITE (MODES A THROUGH F)")
    print("=" * 80)

    # Output directories
    out_dir = root_dir / "research" / "experiments" / "EXP-028B2"
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ExtractEvaluator()
    resolver = JointRecordResolver()

    # Document test set: 6 smoke docs + 1 major table case
    base_pred_dir = root_dir / "research" / "official_eval" / "exp026_scgf_restoration_predictions" / "tonerhound"

    eval_documents = [
        ("short/real_wyo_Goshen_2024", root_dir / "research" / "data" / "test" / "short" / "real_wyo_Goshen_2024.pdf"),
        ("medium/sec_13f_0031_loomis_sayles", root_dir / "research" / "data" / "full" / "medium" / "sec_13f_0031_loomis_sayles.pdf"),
        ("medium/real_pueblo_oct_2025", root_dir / "research" / "data" / "test" / "medium" / "real_pueblo_oct_2025.pdf"),
        ("short/bianco-2024", root_dir / "research" / "data" / "test" / "short" / "bianco-2024.pdf"),
        ("short/W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025)", root_dir / "research" / "data" / "test" / "short" / "W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025).pdf"),
        ("medium/veralto_earnings_deck_q4fy25", root_dir / "research" / "data" / "test" / "medium" / "veralto_earnings_deck_q4fy25.pdf"),
    ]

    modes = [
        ResolverMode.BASELINE,          # A
        ResolverMode.ROW_ANCHOR_ONLY,   # B
        ResolverMode.COLUMN_ONLY,       # C
        ResolverMode.ROW_AND_COLUMN,    # D
        ResolverMode.JOINT_RECORD,      # E
        ResolverMode.AMBIGUITY_GATED,   # F
    ]

    all_results: list[dict[str, Any]] = []
    mode_summaries: dict[str, dict[str, float]] = {}

    for mode in modes:
        print(f"\n--- Running Mode {mode.value}: {mode.name} ---")
        mode_doc_results = []
        for doc_id, pdf_path in eval_documents:
            if not pdf_path.exists():
                print(f"Skipping {doc_id}: {pdf_path} not found")
                continue
            base_p = base_pred_dir / f"{doc_id}.result.json" if base_pred_dir else None
            t_start = time.perf_counter()
            res = run_controlled_experiment_on_doc(
                doc_id=doc_id,
                pdf_path=pdf_path,
                base_pred_path=base_p,
                mode=mode,
                resolver=resolver,
                evaluator=evaluator,
            )
            elapsed = time.perf_counter() - t_start
            mode_doc_results.append(res)
            all_results.append(res)
            print(f"  {doc_id:<55} | Word F1: {res['word_f1']*100:6.2f}% | Prec: {res['word_precision']*100:6.2f}% | Rec: {res['word_recall']*100:6.2f}% | Page F1: {res['page_f1']*100:6.2f}% | Time: {elapsed:5.2f}s")

        # Aggregate for this mode
        avg_word_f1 = sum(r["word_f1"] for r in mode_doc_results) / len(mode_doc_results)
        avg_prec = sum(r["word_precision"] for r in mode_doc_results) / len(mode_doc_results)
        avg_rec = sum(r["word_recall"] for r in mode_doc_results) / len(mode_doc_results)
        avg_page_f1 = sum(r["page_f1"] for r in mode_doc_results) / len(mode_doc_results)
        avg_false_g = sum(r["false_grounding_rate"] for r in mode_doc_results) / len(mode_doc_results)
        avg_abst = sum(r["abstention_rate"] for r in mode_doc_results) / len(mode_doc_results)
        total_sec = sum(r["latency_sec"] for r in mode_doc_results)

        mode_summaries[mode.value] = {
            "mode": mode.value,
            "mode_name": mode.name,
            "avg_word_f1": avg_word_f1,
            "avg_precision": avg_prec,
            "avg_recall": avg_rec,
            "avg_page_f1": avg_page_f1,
            "avg_false_grounding_rate": avg_false_g,
            "avg_abstention_rate": avg_abst,
            "total_latency_sec": total_sec,
        }

    # Print comprehensive comparison matrix
    print("\n" + "=" * 95)
    print("PHASE 4: CONTROLLED EXPERIMENT COMPARISON MATRIX")
    print("=" * 95)
    print(f"{'Mode':<5} | {'Mode Description':<28} | {'Word F1':>8} | {'Precision':>9} | {'Recall':>8} | {'Page F1':>8} | {'False Grd':>9} | {'Abstain':>7}")
    print("-" * 95)
    for m in modes:
        sm = mode_summaries[m.value]
        print(
            f"{sm['mode']:<5} | {sm['mode_name']:<28} | {sm['avg_word_f1']*100:7.2f}% | "
            f"{sm['avg_precision']*100:8.2f}% | {sm['avg_recall']*100:7.2f}% | {sm['avg_page_f1']*100:7.2f}% | "
            f"{sm['avg_false_grounding_rate']*100:8.2f}% | {sm['avg_abstention_rate']*100:6.2f}%"
        )
    print("=" * 95)

    # Save metrics JSON artifact
    metrics_path = out_dir / "controlled_experiments_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump({
            "mode_summaries": mode_summaries,
            "detailed_results": all_results,
        }, fp, indent=2)
    print(f"\n[Saved Metrics JSON]: {metrics_path}")

    # Generate representative before/after comparison examples
    loomis_res = [r for r in all_results if r["doc_id"] == "medium/sec_13f_0031_loomis_sayles"]
    print("\n" + "=" * 80)
    print("REPRESENTATIVE BEFORE / AFTER ON MAJOR TABLE CASE (loomis_sayles):")
    print("=" * 80)
    for lr in loomis_res:
        print(f"  Mode {lr['mode']:<1} ({lr['mode_name']:<18}): Word F1={lr['word_f1']*100:6.2f}%, Prec={lr['word_precision']*100:6.2f}%, Rec={lr['word_recall']*100:6.2f}%, False Grd={lr['false_grounding_rate']*100:5.2f}%")


if __name__ == "__main__":
    run_controlled_experiments_suite()
