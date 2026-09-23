"""EXP-024: Candidate Coverage & Perception Ablation Runner.

Evaluates 5 candidate sources on the frozen ExtractBench stratified benchmark:
1. EXP-024A: Baseline
2. EXP-024B: Multi-Token Spans
3. EXP-024C: Independent OCR (Tesseract)
4. EXP-024D: Table / Layout Cells
5. EXP-024E: Union Pool (Deduplicated Upper Bound)

Produces:
- research/experiments/EXP-024/results.csv
- research/experiments/EXP-024/failure_recovery.csv
- research/experiments/EXP-024/candidate_ablation_analysis.md
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure root directory and dependencies are in path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

# Load candidate generators dynamically
spec = importlib.util.spec_from_file_location("cand_gen", str(Path(__file__).resolve().parent / "candidate_generators.py"))
cand_gen = importlib.util.module_from_spec(spec)
sys.modules["cand_gen"] = cand_gen
spec.loader.exec_module(cand_gen)

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    compute_unified_evidence_metrics,
    iou_xywh,
    path_leaf,
)
from extract_bench.schemas.extract_output import FieldCitation
from extract_bench.test_cases.loader import load_test_case
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import ExtractionInput
from tonerhound.normalization.normalizers import (
    normalize_unicode_and_case,
    parse_numeric_value,
)
from benchmarks.run_final_370_benchmark import _ValidationAdapter, _ValidationResolver


def _flatten_expected_leaves(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            p = f"{prefix}.{k}" if prefix else k
            leaves.extend(_flatten_expected_leaves(v, p))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            p = f"{prefix}[{i}]"
            leaves.extend(_flatten_expected_leaves(item, p))
    else:
        leaves.append((prefix, data))
    return leaves


def evaluate_document_ablation(doc_info: dict[str, Any]) -> dict[str, Any]:
    """Run candidate ablation across baseline, spans, ocr, cells, and union on a single document."""
    test_id = doc_info["test_id"]
    pdf_path = root_dir / "research" / "data" / "full" / f"{test_id}.pdf"
    if not pdf_path.exists():
        return {"success": False, "test_id": test_id, "error": f"PDF not found: {pdf_path}"}

    t0 = time.perf_counter()
    tc = load_test_case(pdf_path)
    doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="hybrid")

    # Build GT rule indexes
    field_rules = tc.get_extract_field_rules()
    alt_values, ev_boxes, ev_pages, normalizers = build_rule_indexes(field_rules)

    # Instantiate baseline adapter
    adapter = _ValidationAdapter(doc_index, doc_id=test_id)
    payload = adapter.ground_extracted_data(tc.expected_output, example_id=test_id)
    raw_cits = payload.get("field_citations", [])
    pred_cits_by_path = {
        c["field_path"]: c
        for c in raw_cits
        if c.get("page") is not None
    }

    # Instantiate research-only candidate generators
    span_gen = cand_gen.MultiTokenSpanGenerator(doc_index)
    ocr_gen = cand_gen.TesseractOCRGenerator(str(pdf_path))
    cell_gen = cand_gen.TableLayoutCellGenerator(doc_index)

    expected_leaves = _flatten_expected_leaves(tc.expected_output)
    total_pages = len(doc_index.pages)

    # Per-source statistics
    sources = ["baseline", "spans", "ocr", "cells", "union"]
    doc_stats = {
        s: {
            "text_covered": 0,
            "geom_covered": 0,
            "hits_at_1": 0,
            "hits_at_5": 0,
            "hits_at_20": 0,
            "oracle_cits": {1: [], 5: [], 20: []},
        }
        for s in sources
    }

    doc_gradeable_count = 0
    field_recovery_rows = []

    for path, val in expected_leaves:
        if val is None:
            continue

        gold_boxes = ev_boxes.get(path, [])
        gold_pages = sorted(list(ev_pages.get(path, set())))
        is_gradeable = len(gold_boxes) > 0
        if not is_gradeable:
            continue

        doc_gradeable_count += 1
        val_str = str(val).strip()
        norm_val = normalize_unicode_and_case(val_str).text.strip()
        val_num = parse_numeric_value(val_str)
        p_hint = gold_pages[0] if gold_pages else None

        # 1. Baseline candidates
        cand_inp = ExtractionInput(field=path, value=val, field_context=path, page_hint=p_hint)
        base_cands_raw = adapter.resolver.collect_candidates(cand_inp)
        if len(base_cands_raw) > 1:
            scored = adapter.resolver._score_candidates_with_context(base_cands_raw, path)
            scored.sort(key=lambda it: it[1], reverse=True)
            base_cands = [
                cand_gen.ResearchCandidate(
                    page=c.page,
                    bbox=c.bbox,
                    matched_text=c.matched_text,
                    normalized_text=normalize_unicode_and_case(c.matched_text).text.strip(),
                    source="baseline",
                    confidence=s,
                    raw_similarity=c.raw_similarity,
                )
                for c, s in scored
            ]
        elif len(base_cands_raw) == 1:
            c = base_cands_raw[0]
            base_cands = [
                cand_gen.ResearchCandidate(
                    page=c.page,
                    bbox=c.bbox,
                    matched_text=c.matched_text,
                    normalized_text=normalize_unicode_and_case(c.matched_text).text.strip(),
                    source="baseline",
                    confidence=1.0,
                    raw_similarity=c.raw_similarity,
                )
            ]
        else:
            base_cands = []

        # 2. Spans candidates
        span_cands = span_gen.extract_candidates(val, page_hint=p_hint)

        # 3. OCR candidates
        ocr_cands = ocr_gen.extract_candidates(val, page_hint=p_hint, total_pages=total_pages)

        # 4. Table cell candidates
        cell_cands = cell_gen.extract_candidates(val, page_hint=p_hint)

        # 5. Union candidates
        all_combined = base_cands + span_cands + ocr_cands + cell_cands
        union_cands = cand_gen.deduplicate_candidate_pool(all_combined, iou_thresh=0.80)

        # Candidate pools dict
        pools = {
            "baseline": base_cands,
            "spans": span_cands,
            "ocr": ocr_cands,
            "cells": cell_cands,
            "union": union_cands,
        }

        # Track failure recovery status
        base_hit_at_5 = False
        base_best_iou = 0.0
        for rank_idx, c in enumerate(base_cands[:5], start=1):
            for gp, gb in gold_boxes:
                if gp == c.page:
                    iou = iou_xywh((c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height), gb)
                    if iou > base_best_iou:
                        base_best_iou = iou
                    if iou >= 0.50:
                        base_hit_at_5 = True

        rec_spans = any(
            any(iou_xywh((c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height), gb) >= 0.50 for gp, gb in gold_boxes if gp == c.page)
            for c in span_cands
        )
        rec_ocr = any(
            any(iou_xywh((c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height), gb) >= 0.50 for gp, gb in gold_boxes if gp == c.page)
            for c in ocr_cands
        )
        rec_cells = any(
            any(iou_xywh((c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height), gb) >= 0.50 for gp, gb in gold_boxes if gp == c.page)
            for c in cell_cands
        )
        rec_union = any(
            any(iou_xywh((c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height), gb) >= 0.50 for gp, gb in gold_boxes if gp == c.page)
            for c in union_cands
        )

        sel_pred = pred_cits_by_path.get(path)
        sel_iou = 0.0
        if sel_pred and sel_pred.get("bbox") and sel_pred.get("page") in gold_pages:
            sel_b = tuple(sel_pred["bbox"])
            sel_iou = max((iou_xywh(sel_b, gb) for gp, gb in gold_boxes if gp == sel_pred["page"]), default=0.0)

        # Baseline failure class
        if sel_iou >= 0.50:
            base_failure_class = "SUCCESS"
        elif not base_cands:
            base_failure_class = "RETRIEVAL_NO_CANDIDATE"
        elif base_hit_at_5:
            base_failure_class = "ASSOCIATION_RANK_MISS"
        elif base_best_iou >= 0.25:
            base_failure_class = "COORDINATE_DRIFT"
        else:
            base_failure_class = "INCOMPLETE_PERCEPTION"

        if base_failure_class != "SUCCESS":
            field_recovery_rows.append({
                "document_id": test_id,
                "field_path": path,
                "baseline_failure_class": base_failure_class,
                "recovered_by_spans": rec_spans,
                "recovered_by_ocr": rec_ocr,
                "recovered_by_cells": rec_cells,
                "recovered_by_union": rec_union,
            })

        # Evaluate each pool
        for s_name, pool in pools.items():
            # Text coverage
            has_text = any(
                c.normalized_text == norm_val or (val_num is not None and parse_numeric_value(c.matched_text) == val_num)
                for c in pool
            )
            if has_text:
                doc_stats[s_name]["text_covered"] += 1

            # Geometric coverage & hits
            hit_ranks = []
            best_c = None
            best_c_iou = -1.0

            for r_idx, c in enumerate(pool[:25], start=1):
                c_b = (c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height)
                c_best_iou = max(
                    (iou_xywh(c_b, gb) for gp, gb in gold_boxes if gp == c.page),
                    default=0.0,
                )
                if c_best_iou > best_c_iou:
                    best_c_iou = c_best_iou
                    best_c = c
                if c_best_iou >= 0.50:
                    hit_ranks.append(r_idx)

            if best_c_iou >= 0.50:
                doc_stats[s_name]["geom_covered"] += 1

            if any(r <= 1 for r in hit_ranks):
                doc_stats[s_name]["hits_at_1"] += 1
            if any(r <= 5 for r in hit_ranks):
                doc_stats[s_name]["hits_at_5"] += 1
            if any(r <= 20 for r in hit_ranks):
                doc_stats[s_name]["hits_at_20"] += 1

            # Build oracle citations for K=1, 5, 20
            for K in (1, 5, 20):
                sub_pool = pool[:K]
                if sub_pool:
                    best_in_k = max(
                        sub_pool,
                        key=lambda cand: max(
                            (
                                iou_xywh((cand.bbox.x, cand.bbox.y, cand.bbox.width, cand.bbox.height), gb)
                                for gp, gb in gold_boxes
                                if gp == cand.page
                            ),
                            default=-1.0,
                        ),
                    )
                    doc_stats[s_name]["oracle_cits"][K].append(
                        FieldCitation(
                            field_path=path,
                            page=best_in_k.page,
                            bbox=[best_in_k.bbox.x, best_in_k.bbox.y, best_in_k.bbox.width, best_in_k.bbox.height],
                            reference_text=best_in_k.matched_text,
                            confidence=1.0,
                            source=f"oracle_{s_name}_top_{K}",
                        )
                    )
                elif sel_pred and sel_pred.get("bbox"):
                    doc_stats[s_name]["oracle_cits"][K].append(
                        FieldCitation(
                            field_path=path,
                            page=sel_pred["page"],
                            bbox=sel_pred["bbox"],
                            reference_text=sel_pred.get("reference_text"),
                            confidence=0.5,
                            source="fallback",
                        )
                    )

    # Compute official oracle grounding metrics for each source
    source_metrics = {}
    for s_name in sources:
        oracle_wf1 = {}
        for K in (1, 5, 20):
            cits = doc_stats[s_name]["oracle_cits"][K]
            m = compute_unified_evidence_metrics(
                expected_output=tc.expected_output,
                extracted_data=tc.expected_output,
                field_rules=field_rules,
                field_citations=cits,
                data_schema=tc.data_schema,
            )
            m_map = {item.metric_name: item.value for item in m}
            oracle_wf1[K] = m_map.get("extract_unified_grounded_f1")

        # Official baseline metrics with actual prediction citations
        base_cits = [
            FieldCitation(
                field_path=c["field_path"],
                page=c["page"],
                bbox=c.get("bbox"),
                reference_text=c.get("reference_text"),
                confidence=c.get("confidence"),
                source="tonerhound",
            )
            for c in raw_cits
            if c.get("page") is not None
        ]
        b_metrics = compute_unified_evidence_metrics(
            expected_output=tc.expected_output,
            extracted_data=tc.expected_output,
            field_rules=field_rules,
            field_citations=base_cits,
            data_schema=tc.data_schema,
        )
        b_map = {item.metric_name: item.value for item in b_metrics}

        source_metrics[s_name] = {
            "text_covered": doc_stats[s_name]["text_covered"],
            "geom_covered": doc_stats[s_name]["geom_covered"],
            "hits_at_1": doc_stats[s_name]["hits_at_1"],
            "hits_at_5": doc_stats[s_name]["hits_at_5"],
            "hits_at_20": doc_stats[s_name]["hits_at_20"],
            "oracle_wf1_1": oracle_wf1[1],
            "oracle_wf1_5": oracle_wf1[5],
            "oracle_wf1_20": oracle_wf1[20],
            "word_f1": b_map.get("extract_unified_grounded_f1"),
            "page_f1": b_map.get("extract_unified_page_f1"),
        }

    elapsed = time.perf_counter() - t0
    return {
        "success": True,
        "test_id": test_id,
        "elapsed_sec": elapsed,
        "gradeable_count": doc_gradeable_count,
        "source_metrics": source_metrics,
        "recovery_rows": field_recovery_rows,
    }


def main():
    exp_dir = root_dir / "research" / "experiments" / "EXP-024"
    exp_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root_dir / "benchmarks" / "exp005_local_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    docs = manifest["documents"]
    print(f"=== EXP-024: Running Candidate Coverage & Perception Ablation across {len(docs)} Stratified Documents ===")

    all_results = []
    all_recovery_rows = []
    total_gradeable = 0

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_document_ablation, d): d["test_id"] for d in docs}
        for fut in as_completed(futures):
            res = fut.result()
            if res.get("success"):
                all_results.append(res)
                all_recovery_rows.extend(res["recovery_rows"])
                total_gradeable += res["gradeable_count"]
                print(f"  [OK] {res['test_id']} ({res['gradeable_count']} gradeable fields, {res['elapsed_sec']:.1f}s)")
            else:
                print(f"  [FAIL] {res.get('test_id')}: {res.get('error')}")

    total_time = time.perf_counter() - t_start
    mean_latency_ms = (total_time / len(all_results)) * 1000 if all_results else 0.0

    print(f"\nCompleted EXP-024 evaluation across {len(all_results)} documents ({total_gradeable} gradeable fields) in {total_time:.1f}s.")

    # Aggregate per candidate source
    sources = ["baseline", "spans", "ocr", "cells", "union"]
    source_names_map = {
        "baseline": "Baseline (EXP-024A)",
        "spans": "Multi-Token Spans (EXP-024B)",
        "ocr": "Tesseract OCR (EXP-024C)",
        "cells": "Table Layout Cells (EXP-024D)",
        "union": "Union Deduplicated Pool (EXP-024E)",
    }

    results_rows = []
    for s in sources:
        tot_text = sum(r["source_metrics"][s]["text_covered"] for r in all_results)
        tot_geom = sum(r["source_metrics"][s]["geom_covered"] for r in all_results)
        tot_h1 = sum(r["source_metrics"][s]["hits_at_1"] for r in all_results)
        tot_h5 = sum(r["source_metrics"][s]["hits_at_5"] for r in all_results)
        tot_h20 = sum(r["source_metrics"][s]["hits_at_20"] for r in all_results)

        # Valid oracle docs
        valid_o1 = [r["source_metrics"][s]["oracle_wf1_1"] for r in all_results if r["source_metrics"][s]["oracle_wf1_1"] is not None]
        valid_o5 = [r["source_metrics"][s]["oracle_wf1_5"] for r in all_results if r["source_metrics"][s]["oracle_wf1_5"] is not None]
        valid_o20 = [r["source_metrics"][s]["oracle_wf1_20"] for r in all_results if r["source_metrics"][s]["oracle_wf1_20"] is not None]
        valid_wf1 = [r["source_metrics"][s]["word_f1"] for r in all_results if r["source_metrics"][s]["word_f1"] is not None]
        valid_pf1 = [r["source_metrics"][s]["page_f1"] for r in all_results if r["source_metrics"][s]["page_f1"] is not None]

        denom = max(1, total_gradeable)
        results_rows.append({
            "candidate_source": source_names_map[s],
            "text_coverage": f"{tot_text / denom * 100:.2f}%",
            "geometric_coverage": f"{tot_geom / denom * 100:.2f}%",
            "recall_at_1": f"{tot_h1 / denom * 100:.2f}%",
            "recall_at_5": f"{tot_h5 / denom * 100:.2f}%",
            "recall_at_20": f"{tot_h20 / denom * 100:.2f}%",
            "oracle_grounding_at_1": f"{sum(valid_o1)/len(valid_o1)*100:.2f}%" if valid_o1 else "N/A",
            "oracle_grounding_at_5": f"{sum(valid_o5)/len(valid_o5)*100:.2f}%" if valid_o5 else "N/A",
            "oracle_grounding_at_20": f"{sum(valid_o20)/len(valid_o20)*100:.2f}%" if valid_o20 else "N/A",
            "word_grounding_f1": f"{sum(valid_wf1)/len(valid_wf1)*100:.2f}%" if valid_wf1 else "N/A",
            "page_grounding_f1": f"{sum(valid_pf1)/len(valid_pf1)*100:.2f}%" if valid_pf1 else "N/A",
            "mean_latency_ms": f"{mean_latency_ms:.1f}",
        })

    # Write results.csv
    csv_path = exp_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "candidate_source",
            "text_coverage",
            "geometric_coverage",
            "recall_at_1",
            "recall_at_5",
            "recall_at_20",
            "oracle_grounding_at_1",
            "oracle_grounding_at_5",
            "oracle_grounding_at_20",
            "word_grounding_f1",
            "page_grounding_f1",
            "mean_latency_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"Written results to {csv_path}")

    # Write failure_recovery.csv
    rec_path = exp_dir / "failure_recovery.csv"
    with open(rec_path, "w", newline="", encoding="utf-8") as f:
        rec_fields = [
            "document_id",
            "field_path",
            "baseline_failure_class",
            "recovered_by_spans",
            "recovered_by_ocr",
            "recovered_by_cells",
            "recovered_by_union",
        ]
        writer = csv.DictWriter(f, fieldnames=rec_fields)
        writer.writeheader()
        writer.writerows(all_recovery_rows)
    print(f"Written failure recovery data ({len(all_recovery_rows)} failing fields) to {rec_path}")

    # Print summary table
    print("\n=== EXP-024 Candidate Coverage & Perception Ablation Results ===")
    for row in results_rows:
        print(f"{row['candidate_source']:<35} | Text Cov: {row['text_coverage']} | Geom Cov: {row['geometric_coverage']} | Rec@5: {row['recall_at_5']} | Oracle@5: {row['oracle_grounding_at_5']}")


if __name__ == "__main__":
    main()
