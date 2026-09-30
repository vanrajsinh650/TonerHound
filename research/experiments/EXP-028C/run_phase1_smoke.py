"""EXP-028C Phase 1: Quick Sanity Check Smoke Benchmark.

Runs 6-document smoke benchmark with EXP-028B2 production resolver enabled.
Measures:
- Word Grounding F1
- Precision
- Recall
- Page Grounding F1
- Runtime
- False Grounding Rate
- Abstention Rate
Compares directly against EXP-028B1.
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from collections import defaultdict
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

from tonerhound.document.index import DocumentIndex
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.resolution.joint_record_resolver import (
    JointRecordResolver,
    RecordFieldLeaf,
    ResolverMode,
    StructuralRecord,
)

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


def run_doc_phase1(
    doc_id: str,
    pdf_path: Path,
    base_pred_path: Path,
    out_pred_path: Path,
    eval_cache_path: Path,
    mode: ResolverMode = ResolverMode.JOINT_RECORD,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    tc = load_test_case(pdf_path)
    rules = tc.get_extract_field_rules()
    doc_idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="hybrid")

    with open(base_pred_path, encoding="utf-8") as fp:
        base_data = json.load(fp)

    existing_cits = {
        c["field_path"]: dict(c)
        for c in base_data.get("output", {}).get("field_citations", [])
        if c.get("field_path")
    }

    table_records: dict[str, dict[int, list[RecordFieldLeaf]]] = defaultdict(lambda: defaultdict(list))
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

    resolver = JointRecordResolver(mode=mode)
    new_citations = dict(existing_cits)
    abstained_count = 0
    total_table_fields = 0
    resolved_count = 0

    for tname, rows_map in table_records.items():
        col_corridors = STANDARD_TABLE_COLUMNS.get(tname, {})
        for ridx, leaves in rows_map.items():
            total_table_fields += len(leaves)
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

            outcomes = resolver.resolve_record(
                record=rec,
                fields=leaves,
                candidates_by_field=cands_by_field,
                column_corridors=col_corridors,
                mode=mode,
            )

            for fname, outcome in outcomes.items():
                fpath = outcome.field_path
                resolved_count += 1
                if outcome.is_abstained:
                    abstained_count += 1
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
                        "source": "tonerhound_exp028c",
                    }

    elapsed_sec = time.perf_counter() - t0

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
        extracted_data=base_data.get("output", {}).get("extracted_data", {}),
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

    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pred_path, "w", encoding="utf-8") as f:
        f.write(inf_result.model_dump_json(indent=2))

    evaluator = ExtractEvaluator()
    eval_res = evaluator.evaluate(inf_result, tc)

    eval_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_cache_path, "w", encoding="utf-8") as f:
        f.write(eval_res.model_dump_json(indent=2))

    metrics = {m.metric_name: m.value for m in eval_res.metrics}

    # False grounding calculation
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

    del doc_idx, tc, rules, base_data, existing_cits, new_citations, inf_result, eval_res
    gc.collect()

    return {
        "doc_id": doc_id,
        "word_f1": metrics.get("extract_unified_grounded_f1", 0.0),
        "word_precision": metrics.get("extract_unified_grounded_precision", 0.0),
        "word_recall": metrics.get("extract_unified_grounded_recall", 0.0),
        "page_f1": metrics.get("extract_unified_page_f1", 0.0),
        "runtime_sec": elapsed_sec,
        "false_grounding_rate": false_grounding_rate,
        "abstention_rate": abstention_rate,
        "citations": len(field_cits),
        "table_fields": total_table_fields,
    }


def main():
    b1_smoke_metrics = {
        "short/W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025)": {
            "word_f1": 0.5401, "word_precision": 0.6852, "word_recall": 0.4458, "page_f1": 0.6109
        },
        "short/bianco-2024": {
            "word_f1": 0.3144, "word_precision": 0.3264, "word_recall": 0.3032, "page_f1": 0.6957
        },
        "short/real_wyo_Goshen_2024": {
            "word_f1": 0.9949, "word_precision": 1.0000, "word_recall": 0.9898, "page_f1": 0.9965
        },
        "medium/real_pueblo_oct_2025": {
            "word_f1": 0.9960, "word_precision": 0.9963, "word_recall": 0.9957, "page_f1": 0.9984
        },
        "medium/veralto_earnings_deck_q4fy25": {
            "word_f1": 0.0000, "word_precision": 0.0000, "word_recall": 0.0000, "page_f1": 0.0000
        },
        "long/real_sm0801_eco_full": {
            "word_f1": 0.9248, "word_precision": 0.9250, "word_recall": 0.9246, "page_f1": 0.9757
        },
        "medium/sec_13f_0031_loomis_sayles": {
            "word_f1": 0.8600, "word_precision": 0.8759, "word_recall": 0.8448, "page_f1": 0.9248
        },
    }

    test_docs = [
        "short/W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025)",
        "short/bianco-2024",
        "short/real_wyo_Goshen_2024",
        "medium/real_pueblo_oct_2025",
        "medium/veralto_earnings_deck_q4fy25",
        "long/real_sm0801_eco_full",
        "medium/sec_13f_0031_loomis_sayles",
    ]

    exp_dir = root_dir / "research" / "experiments" / "EXP-028C"
    smoke_preds_dir = exp_dir / "smoke_predictions" / "tonerhound"
    smoke_eval_dir = exp_dir / "smoke_eval_cache"

    print("=" * 105)
    print("EXP-028C PHASE 1: QUICK SANITY CHECK SMOKE BENCHMARK")
    print("=" * 105)

    results = []
    for tid in test_docs:
        pdf_path = root_dir / "research" / "data" / "full" / f"{tid}.pdf"
        if not pdf_path.exists():
            pdf_path = root_dir / "research" / "data" / "test" / f"{tid}.pdf"
        b1_pred_path = root_dir / "research" / "experiments" / "EXP-028B1" / "predictions" / "tonerhound" / f"{tid}.result.json"
        out_pred = smoke_preds_dir / f"{tid}.result.json"
        eval_cache = smoke_eval_dir / f"{tid}.eval.json"

        print(f"Running {tid}...")
        res = run_doc_phase1(
            doc_id=tid,
            pdf_path=pdf_path,
            base_pred_path=b1_pred_path,
            out_pred_path=out_pred,
            eval_cache_path=eval_cache,
            mode=ResolverMode.JOINT_RECORD,
        )
        results.append(res)

        b1_m = b1_smoke_metrics.get(tid, {})
        b1_f1 = b1_m.get("word_f1", 0.0) * 100
        delta = (res["word_f1"] * 100) - b1_f1
        sign = "+" if delta >= 0 else ""
        print(
            f"  -> Word F1: {res['word_f1']*100:6.2f}% (B1: {b1_f1:6.2f}%, {sign}{delta:5.2f}pp) | "
            f"Prec: {res['word_precision']*100:6.2f}% | Rec: {res['word_recall']*100:6.2f}% | "
            f"Page F1: {res['page_f1']*100:6.2f}% | False Grd: {res['false_grounding_rate']*100:5.2f}% | "
            f"Abst: {res['abstention_rate']*100:4.1f}% | Time: {res['runtime_sec']:5.2f}s"
        )

    # 6-Document Smoke Set (official ExtractBench smoke set)
    official_6_res = [r for r in results if r["doc_id"] != "medium/sec_13f_0031_loomis_sayles"]
    avg_f1_6 = sum(r["word_f1"] for r in official_6_res) / 6.0
    avg_p_6 = sum(r["word_precision"] for r in official_6_res) / 6.0
    avg_r_6 = sum(r["word_recall"] for r in official_6_res) / 6.0
    avg_page_6 = sum(r["page_f1"] for r in official_6_res) / 6.0
    avg_false_g_6 = sum(r["false_grounding_rate"] for r in official_6_res) / 6.0
    avg_abst_6 = sum(r["abstention_rate"] for r in official_6_res) / 6.0

    b1_6_avg = sum(b1_smoke_metrics[r["doc_id"]]["word_f1"] for r in official_6_res) / 6.0
    delta_6 = (avg_f1_6 - b1_6_avg) * 100

    print("\n" + "=" * 105)
    print("PHASE 1 SUMMARY (6-DOCUMENT OFFICIAL SMOKE BENCHMARK):")
    print(f"  EXP-028B1 Smoke Word F1:     {b1_6_avg*100:.2f}%")
    print(f"  EXP-028C Smoke Word F1:       {avg_f1_6*100:.2f}%")
    print(f"  Delta:                        {('+' if delta_6 >= 0 else '')}{delta_6:.2f}pp")
    print(f"  Word Precision:               {avg_p_6*100:.2f}%")
    print(f"  Word Recall:                  {avg_r_6*100:.2f}%")
    print(f"  Page Grounding F1:            {avg_page_6*100:.2f}%")
    print(f"  False Grounding Rate:         {avg_false_g_6*100:.2f}%")
    print(f"  Abstention Rate:              {avg_abst_6*100:.2f}%")
    print("=" * 105)


if __name__ == "__main__":
    main()
