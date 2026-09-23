"""Authoritative Benchmark Microscope V2 for TonerHound on ExtractBench.

Measures the exact candidate pool and official metrics on the frozen production stack
across all 370 ExtractBench documents without modifying production behavior.
Generates per-field authoritative records (field_records.parquet) and aggregates (aggregates.json).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Set up paths
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

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
from tonerhound.models.types import ExtractionInput, ResolutionResult
from tonerhound.normalization.normalizers import normalize_unicode_and_case
from benchmarks.run_final_370_benchmark import _ValidationAdapter, _ValidationResolver


def _flatten_expected_leaves(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dict/list into leaf paths and values."""
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


def _doc_type_from_schema_and_id(data_schema: Any, test_id: str) -> str:
    """Discover standardized document type name."""
    if isinstance(data_schema, dict) and data_schema.get("title"):
        title = data_schema["title"].strip()
        if title and title != "Unknown":
            return title
    tid = test_id.lower()
    if "00581" in tid or "07021" in tid or "k1" in tid or "k-1" in tid:
        return "Schedule K-1 (Form 1065 / 1120-S)"
    if "1040" in tid or "cabrera" in tid or "bar-lev" in tid:
        return "IRS Form 1040 Tax Package"
    if "w2" in tid or "w-2" in tid:
        return "Form W-2 Wage Statement"
    if "1099" in tid:
        return "Form 1099 Consolidated Statement"
    if "rrc" in tid or "texas" in tid or any(k in tid for k in ("h-12", "h12", "w-14", "w14", "p-4", "p4", "h-5", "h5")):
        return "Texas RRC Regulatory Form"
    if "sec" in tid or "13f" in tid or "10-k" in tid or "10-q" in tid:
        return "SEC Financial Filing"
    if "ccc" in tid or "mitchell" in tid or "valuation" in tid:
        return "Vehicle Market Valuation Report"
    if "gsa" in tid or "mas" in tid:
        return "GSA MAS IT Pricelist"
    if "closing" in tid or "trid" in tid:
        return "Closing Disclosure TRID"
    if "purchase_order" in tid or "po" in tid:
        return "Purchase Order"
    if "dd1155" in tid or "clin" in tid or "sf1449" in tid:
        return "DoD / Government Contract Schedule"
    if "medicaid" in tid or "remittance" in tid or "eob" in tid:
        return "Medical Remittance / EOB"
    if "investment" in tid or "vanguard" in tid or "ishares" in tid:
        return "Schedule of Investments"
    if "ofac" in tid or "unclaimed" in tid or "freer" in tid or "ftx" in tid or "imedia" in tid:
        return "Public Register / Schedule Table"
    return "Structured Form / Table Document"


class AuthoritativeInstrumentedResolver(_ValidationResolver):
    """Instruments TonerHound's production resolver to record exact candidate rankings."""

    def __init__(self, index: DocumentIndex, doc_id: str) -> None:
        super().__init__(index=index, doc_id=doc_id)
        self.field_observations: dict[str, dict[str, Any]] = {}

    def resolve(self, extraction: ExtractionInput) -> ResolutionResult:
        res = super().resolve(extraction)
        cands = self.last_field_candidates.get(extraction.field, [])

        if not cands:
            self.field_observations[extraction.field] = {
                "result": res,
                "ranked_candidates": [],
            }
            return res

        if len(cands) == 1:
            ranked = [cands[0]]
        else:
            scored = self._score_candidates_with_context(
                cands, extraction.field_context or extraction.field, y_hint=extraction.y_hint
            )
            scored.sort(key=lambda item: item[1], reverse=True)

            if self.reranker is not None and self.reranker.enabled:
                base_cands = [c for c, _ in scored]
                base_scores = [s for _, s in scored]
                is_num = isinstance(extraction.value, (int, float)) and not isinstance(extraction.value, bool)
                target_p = extraction.target_page or (
                    extraction.page_hint if (extraction.page_confidence and extraction.page_confidence > 0) else None
                )
                p_conf = (
                    extraction.page_confidence
                    if extraction.page_confidence > 0
                    else (1.0 if target_p is not None else 0.0)
                )
                reranked = self.reranker.rank_candidates(
                    candidates=base_cands,
                    base_scores=base_scores,
                    column_corridor=extraction.column_corridor,
                    column_peers=extraction.column_peers,
                    target_y=extraction.expected_row_y,
                    row_corridor=extraction.row_corridor,
                    sibling_boxes=extraction.sibling_boxes,
                    expected_row_y=extraction.expected_row_y,
                    prev_row_y=extraction.prev_row_y,
                    next_row_y=extraction.next_row_y,
                    target_page=target_p,
                    page_confidence=p_conf,
                    is_numeric=is_num,
                    is_header_field=extraction.is_header,
                    total_pages=len(self.index.pages),
                )
                ranked = [r.candidate for r in reranked]
            else:
                ranked = [c for c, _ in scored]

        # Flat form reranker promotion if applicable
        if res.is_grounded and res.page is not None and res.bbox is not None:
            sel_idx = next(
                (
                    i
                    for i, c in enumerate(ranked)
                    if c.page == res.page and c.bbox.iou(res.bbox) >= 0.85
                ),
                None,
            )
            if sel_idx is not None and sel_idx > 0:
                promoted = ranked.pop(sel_idx)
                ranked.insert(0, promoted)

        self.field_observations[extraction.field] = {
            "result": res,
            "ranked_candidates": ranked,
        }
        return res


def process_single_doc(task: dict[str, Any]) -> dict[str, Any]:
    """Execute authoritative microscope on a single document."""
    test_id = task["test_id"]
    pdf_path = Path(task["pdf_path"])
    cache_path = task.get("cache_path")
    force = task.get("force", False)

    if cache_path and Path(cache_path).exists() and not force:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("success") and "field_records" in cached:
                return cached
        except Exception:
            pass

    t0 = time.perf_counter()
    try:
        tc = load_test_case(pdf_path)
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=True, backend="hybrid")
        tags = tc.tags or []

        doc_type = _doc_type_from_schema_and_id(tc.data_schema, test_id)
        domain = next((t.split(":")[-1] for t in tags if t.startswith("domain:")), "Unknown")
        split = next((t.split(":")[-1] for t in tags if t.startswith("length:")), test_id.split("/")[0])

        # 1. Run authoritative instrumented adapter
        adapter = _ValidationAdapter(doc_index, doc_id=test_id)
        resolver = AuthoritativeInstrumentedResolver(doc_index, doc_id=test_id)
        adapter.resolver = resolver

        payload = adapter.ground_extracted_data(tc.expected_output, example_id=test_id)
        raw_cits = payload.get("field_citations", [])
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
        pred_cits_by_path = {c.field_path: c for c in base_cits}

        # 2. Build ground truth rule indexes
        field_rules = tc.get_extract_field_rules()
        alt_values, ev_boxes, ev_pages, normalizers = build_rule_indexes(field_rules)

        # 3. Official Baseline Metrics
        base_metrics = compute_unified_evidence_metrics(
            expected_output=tc.expected_output,
            extracted_data=tc.expected_output,
            field_rules=field_rules,
            field_citations=base_cits,
            data_schema=tc.data_schema,
        )
        base_m_map = {m.metric_name: m.value for m in base_metrics}

        # 4. Detailed Per-Field Microscope
        expected_leaves = _flatten_expected_leaves(tc.expected_output)
        field_records = []
        topk_cits = {1: [], 3: [], 5: [], 10: [], 20: []}
        r_hits = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
        doc_gradeable_count = 0

        for path, val in expected_leaves:
            if val is None:
                continue

            gold_boxes = ev_boxes.get(path, [])
            gold_pages = sorted(list(ev_pages.get(path, set())))
            gold_accepted_vals = alt_values.get(path, [val])
            is_gradeable = len(gold_boxes) > 0

            # Formatted gold evidence entries
            gold_evidence_entries = [
                {
                    "page": gp,
                    "bbox": list(gb),
                }
                for gp, gb in gold_boxes
            ]

            # Selected prediction citation
            sel_cit = pred_cits_by_path.get(path)
            sel_page = sel_cit.page if sel_cit else None
            sel_bbox = sel_cit.bbox if sel_cit else None

            # Get candidate pool from instrumented resolver
            if not is_gradeable:
                ranked_cands = []
            else:
                obs = resolver.field_observations.get(path)
                if obs is not None:
                    ranked_cands = obs["ranked_candidates"]
                else:
                    # Fallback query with page hint if not resolved in adapter passes
                    p_hint = gold_pages[0] if gold_pages else None
                    cand_inp = ExtractionInput(field=path, value=val, field_context=path, page_hint=p_hint)
                    cands = resolver.collect_candidates(cand_inp)
                    if len(cands) == 1:
                        ranked_cands = [cands[0]]
                    elif len(cands) > 1:
                        scored = resolver._score_candidates_with_context(cands, path)
                        scored.sort(key=lambda it: it[1], reverse=True)
                        ranked_cands = [c for c, _s in scored]
                    else:
                        ranked_cands = []

            # Evaluate each candidate in candidate pool
            candidate_pool = []
            hit_ranks = []
            cand_best_ious = []

            for rank_idx, cand in enumerate(ranked_cands[:25], start=1):
                c_p = cand.bbox.page if hasattr(cand.bbox, "page") else cand.page
                c_b = (cand.bbox.x, cand.bbox.y, cand.bbox.width, cand.bbox.height)

                # IoU against every gold evidence entry
                iou_by_gold = []
                for gp, gb in gold_boxes:
                    iou = iou_xywh(c_b, gb) if gp == c_p else 0.0
                    iou_by_gold.append(round(iou, 4))

                best_iou = max(iou_by_gold) if iou_by_gold else 0.0
                best_gold_idx = int(np.argmax(iou_by_gold)) if iou_by_gold else -1
                cand_best_ious.append(best_iou)

                # Consistent Candidate Hit Definition: max IoU >= 0.50
                is_hit = best_iou >= 0.50
                if is_hit:
                    hit_ranks.append(rank_idx)

                cand_norm = normalize_unicode_and_case(cand.matched_text).text.strip() if cand.matched_text else ""
                candidate_pool.append({
                    "rank": rank_idx,
                    "source": cand.match_type,
                    "text": cand.matched_text,
                    "normalized_text": cand_norm,
                    "page": c_p,
                    "bbox": list(c_b),
                    "iou_by_gold": iou_by_gold,
                    "best_iou": round(best_iou, 4),
                    "best_gold_index": best_gold_idx,
                })

            best_candidate_iou = max(cand_best_ious) if cand_best_ious else 0.0

            # Hit at K
            cand_hit_at_1 = any(r <= 1 for r in hit_ranks)
            cand_hit_at_3 = any(r <= 3 for r in hit_ranks)
            cand_hit_at_5 = any(r <= 5 for r in hit_ranks)
            cand_hit_at_10 = any(r <= 10 for r in hit_ranks)
            cand_hit_at_20 = any(r <= 20 for r in hit_ranks)

            # Selected candidate IoU against gold evidence
            selected_cand_iou = 0.0
            if sel_bbox and gold_boxes and sel_page is not None:
                sel_b_tuple = (sel_bbox[0], sel_bbox[1], sel_bbox[2], sel_bbox[3])
                selected_cand_iou = max(
                    (iou_xywh(sel_b_tuple, gb) for gp, gb in gold_boxes if gp == sel_page),
                    default=0.0,
                )

            page_correct = bool(sel_page is not None and sel_page in gold_pages)
            grounded_correct = is_gradeable and page_correct and (selected_cand_iou >= 0.50)
            value_correct = True

            # Track hit counts for gradeable fields
            if is_gradeable:
                doc_gradeable_count += 1
                if cand_hit_at_1:
                    r_hits[1] += 1
                if cand_hit_at_3:
                    r_hits[3] += 1
                if cand_hit_at_5:
                    r_hits[5] += 1
                if cand_hit_at_10:
                    r_hits[10] += 1
                if cand_hit_at_20:
                    r_hits[20] += 1

                # Build Top-K Oracle Citations
                for K in (1, 3, 5, 10, 20):
                    pool_k = ranked_cands[:K]
                    if pool_k:
                        best_cand = max(
                            pool_k,
                            key=lambda c: max(
                                (
                                    iou_xywh(
                                        (c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height),
                                        gb,
                                    )
                                    for gp, gb in gold_boxes
                                    if gp == (c.bbox.page if hasattr(c.bbox, "page") else c.page)
                                ),
                                default=-1.0,
                            ),
                        )
                        c_p = best_cand.bbox.page if hasattr(best_cand.bbox, "page") else best_cand.page
                        c_b = [best_cand.bbox.x, best_cand.bbox.y, best_cand.bbox.width, best_cand.bbox.height]
                        topk_cits[K].append(
                            FieldCitation(
                                field_path=path,
                                page=c_p,
                                bbox=c_b,
                                reference_text=getattr(best_cand, "matched_text", "oracle"),
                                confidence=1.0,
                                source=f"oracle_top_{K}",
                            )
                        )
                    elif sel_cit:
                        topk_cits[K].append(sel_cit)
            else:
                if sel_cit:
                    for K in (1, 3, 5, 10, 20):
                        topk_cits[K].append(sel_cit)

            # Failure Classification
            if not is_gradeable:
                failure_class = "UNGRADEABLE_PAGE_ONLY" if page_correct else "UNGRADEABLE_WRONG_PAGE"
            elif grounded_correct:
                failure_class = "SUCCESS"
            elif not candidate_pool:
                failure_class = "RETRIEVAL_NO_CANDIDATE"
            elif not hit_ranks:
                # No candidate has IoU >= 0.50
                cand_on_page = [c for c in candidate_pool if c["page"] in gold_pages]
                if cand_on_page:
                    c0 = cand_on_page[0]
                    c0_w = c0["bbox"][2]
                    g0_w = gold_boxes[0][1][2]
                    w_ratio = c0_w / g0_w if g0_w > 0 else 1.0
                    if w_ratio < 0.70:
                        failure_class = "BBOX_TOO_NARROW"
                    elif w_ratio > 1.35:
                        failure_class = "BBOX_TOO_WIDE"
                    elif best_candidate_iou >= 0.25:
                        failure_class = "COORDINATE_DRIFT"
                    else:
                        failure_class = "OCR_GEOMETRY"
                else:
                    failure_class = "RETRIEVAL_WRONG_PAGE"
            else:
                # Candidate hit exists in pool (IoU >= 0.50)
                first_hit_rank = hit_ranks[0]
                if first_hit_rank == 1:
                    failure_class = "SELECTED_CITATION_GEOMETRY_FAILURE"
                elif first_hit_rank <= 5:
                    if not page_correct:
                        failure_class = "ASSOCIATION_WRONG_PAGE"
                    elif "[" in path and "]" in path:
                        failure_class = "ASSOCIATION_WRONG_ROW"
                    else:
                        failure_class = "ASSOCIATION_RANK_MISS"
                elif first_hit_rank <= 20:
                    failure_class = "RETRIEVED_RANK_6_20"
                else:
                    failure_class = "RETRIEVED_RANK_GT_20"

            field_records.append({
                "document_id": test_id,
                "document_type": doc_type,
                "domain": domain,
                "split": split,
                "field_path": path,
                "gold_value": str(val)[:100],
                "gold_evidence_entries": json.dumps(gold_evidence_entries),
                "gold_bboxes_count": len(gold_boxes),
                "gold_pages": ";".join(str(p) for p in gold_pages),
                "candidate_count": len(candidate_pool),
                "candidate_pool": json.dumps(candidate_pool[:10]),
                "best_candidate_iou": round(best_candidate_iou, 4),
                "candidate_hit_at_1": cand_hit_at_1,
                "candidate_hit_at_3": cand_hit_at_3,
                "candidate_hit_at_5": cand_hit_at_5,
                "candidate_hit_at_10": cand_hit_at_10,
                "candidate_hit_at_20": cand_hit_at_20,
                "selected_candidate": json.dumps({
                    "page": sel_page,
                    "bbox": sel_bbox,
                    "reference_text": sel_cit.reference_text if sel_cit else None,
                }) if sel_cit else None,
                "selected_candidate_iou": round(selected_cand_iou, 4),
                "value_correct": value_correct,
                "page_correct": page_correct,
                "grounded_correct": grounded_correct,
                "failure_class": failure_class,
                "is_gradeable": is_gradeable,
            })

        # 5. Evaluate Top-K Oracle Citations
        topk_eval = {}
        for K in (1, 3, 5, 10, 20):
            c_metrics = compute_unified_evidence_metrics(
                expected_output=tc.expected_output,
                extracted_data=tc.expected_output,
                field_rules=field_rules,
                field_citations=topk_cits[K],
                data_schema=tc.data_schema,
            )
            c_map = {m.metric_name: m.value for m in c_metrics}
            topk_eval[str(K)] = {
                "grounded_f1": c_map.get("extract_unified_grounded_f1"),
                "grounded_precision": c_map.get("extract_unified_grounded_precision"),
                "grounded_recall": c_map.get("extract_unified_grounded_recall"),
                "hits_count": r_hits[K],
                "gradeable_count": doc_gradeable_count,
                # Pure doc-level recall: None if doc is ungradeable!
                "recall_at_k": (r_hits[K] / doc_gradeable_count) if doc_gradeable_count > 0 else None,
            }

        elapsed = time.perf_counter() - t0
        doc_result = {
            "success": True,
            "test_id": test_id,
            "document_type": doc_type,
            "domain": domain,
            "split": split,
            "pages": len(doc_index.pages),
            "elapsed": round(elapsed, 2),
            "is_grounded_doc": bool(base_m_map.get("extract_unified_grounded_f1") is not None),
            "gradeable_fields_count": doc_gradeable_count,
            "total_leaves_count": len(field_records),
            "base_metrics": {
                "word_grounding_f1": base_m_map.get("extract_unified_grounded_f1"),
                "word_grounding_precision": base_m_map.get("extract_unified_grounded_precision"),
                "word_grounding_recall": base_m_map.get("extract_unified_grounded_recall"),
                "page_grounding_f1": base_m_map.get("extract_unified_page_f1"),
                "page_grounding_precision": base_m_map.get("extract_unified_page_precision"),
                "page_grounding_recall": base_m_map.get("extract_unified_page_recall"),
                "value_f1": base_m_map.get("extract_unified_value_f1", 1.0),
            },
            "topk_eval": topk_eval,
            "field_records": field_records,
        }

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(doc_result, f)

        return doc_result

    except Exception as exc:
        traceback.print_exc()
        return {
            "success": False,
            "test_id": test_id,
            "error": f"{type(exc).__name__}: {str(exc)}",
        }


def main():
    obs_dir = root_dir / "research" / "observer"
    obs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = obs_dir / "cache_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root_dir / "research" / "data" / "full"

    # Schedule short documents first, then medium, then long to maximize pipeline throughput
    pdf_files = sorted(
        list(data_dir.glob("**/*.pdf")),
        key=lambda p: (0 if "short" in str(p) else (1 if "medium" in str(p) else 2), p.stat().st_size),
    )
    total_docs = len(pdf_files)
    print(f"=== Authoritative Microscope V2: Found {total_docs} Documents ===")

    tasks = [
        {
            "test_id": p.relative_to(data_dir).as_posix().removesuffix(".pdf"),
            "pdf_path": str(p),
            "cache_path": str(cache_dir / f"{p.relative_to(data_dir).as_posix().removesuffix('.pdf')}.json"),
            "force": False,
        }
        for p in pdf_files
    ]

    t_start = time.perf_counter()
    doc_results: list[dict[str, Any]] = []
    completed = 0
    failed = 0

    print("Executing microscope across worker pool (6 processes)...")
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_single_doc, t): t["test_id"] for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            if res.get("success"):
                doc_results.append(res)
            else:
                failed += 1
                print(f"FAILED on {res.get('test_id')}: {res.get('error')}")

            if completed % 25 == 0 or completed == total_docs:
                elapsed_cur = time.perf_counter() - t_start
                print(f"[{completed:3d}/{total_docs:3d}] Processed ({len(doc_results)} succeeded, {failed} failed) in {elapsed_cur:.1f}s")

    doc_results.sort(key=lambda r: r["test_id"])
    total_time = time.perf_counter() - t_start
    print(f"\nCompleted microscope processing in {total_time:.1f}s ({total_time/60:.2f}m).")

    # Aggregate metrics
    grounded_docs = [r for r in doc_results if r["is_grounded_doc"]]
    ungrounded_docs = [r for r in doc_results if not r["is_grounded_doc"]]
    print(f"Total Documents: {len(doc_results)}")
    print(f"Grounded Documents (carrying GT bboxes): {len(grounded_docs)}")
    print(f"Ungrounded Documents (value/page only): {len(ungrounded_docs)}")

    def mean_valid(vals: list[float | None]) -> float:
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else 0.0

    # 1. Official Baseline Metrics
    base_wf1 = mean_valid([r["base_metrics"]["word_grounding_f1"] for r in grounded_docs])
    base_wprec = mean_valid([r["base_metrics"]["word_grounding_precision"] for r in grounded_docs])
    base_wrec = mean_valid([r["base_metrics"]["word_grounding_recall"] for r in grounded_docs])

    base_pf1 = mean_valid([r["base_metrics"]["page_grounding_f1"] for r in grounded_docs])
    base_pprec = mean_valid([r["base_metrics"]["page_grounding_precision"] for r in grounded_docs])
    base_prec = mean_valid([r["base_metrics"]["page_grounding_recall"] for r in grounded_docs])

    base_vf1 = mean_valid([r["base_metrics"]["value_f1"] for r in doc_results])

    print("\n--- Official Baseline Metrics ---")
    print(f"Word Grounding F1:        {base_wf1*100:.2f}% (Precision: {base_wprec*100:.2f}%, Recall: {base_wrec*100:.2f}%) [N={len(grounded_docs)}]")
    print(f"Page Grounding F1:        {base_pf1*100:.2f}% (Precision: {base_pprec*100:.2f}%, Recall: {base_prec*100:.2f}%) [N={len(grounded_docs)}]")
    print(f"Value F1:                 {base_vf1*100:.2f}% [N={len(doc_results)}]")

    # 2. Extract all field records
    all_field_records = []
    gradeable_field_records = []
    for r in doc_results:
        for fld in r.get("field_records", []):
            all_field_records.append(fld)
            if fld.get("is_gradeable"):
                gradeable_field_records.append(fld)

    total_gradeable_fields = len(gradeable_field_records)
    print(f"\nTotal Field Records:      {len(all_field_records)}")
    print(f"Total Gradeable Fields:   {total_gradeable_fields}")

    # 3. Candidate Hit & Recall at K
    # A. Micro Recall@K (across all gradeable fields)
    micro_hits = {
        1: sum(1 for f in gradeable_field_records if f["candidate_hit_at_1"]),
        3: sum(1 for f in gradeable_field_records if f["candidate_hit_at_3"]),
        5: sum(1 for f in gradeable_field_records if f["candidate_hit_at_5"]),
        10: sum(1 for f in gradeable_field_records if f["candidate_hit_at_10"]),
        20: sum(1 for f in gradeable_field_records if f["candidate_hit_at_20"]),
    }

    # B. Macro Recall@K (across 236 grounded documents)
    macro_recall = {}
    for K in (1, 3, 5, 10, 20):
        doc_recalls = [r["topk_eval"][str(K)]["recall_at_k"] for r in grounded_docs if r["topk_eval"][str(K)]["recall_at_k"] is not None]
        macro_recall[K] = sum(doc_recalls) / len(doc_recalls) if doc_recalls else 0.0

    # C. Flawed Macro Recall@K (across all 370 documents with 134 ungrounded set to 1.0)
    flawed_recall = {}
    for K in (1, 3, 5, 10, 20):
        flawed_vals = []
        for r in doc_results:
            rec = r["topk_eval"][str(K)]["recall_at_k"]
            flawed_vals.append(rec if rec is not None else 1.0)
        flawed_recall[K] = sum(flawed_vals) / len(flawed_vals)

    print("\n--- Candidate Recall Matrix ---")
    for K in (1, 3, 5, 10, 20):
        print(f"K={K:2d}: Micro={micro_hits[K]/total_gradeable_fields*100:6.2f}% ({micro_hits[K]}/{total_gradeable_fields}) | True Macro (N=236)={macro_recall[K]*100:6.2f}% | Flawed Macro (N=370)={flawed_recall[K]*100:6.2f}%")

    # 4. Top-K Oracle Grounding Ceilings
    oracle_wf1 = {}
    for K in (1, 3, 5, 10, 20):
        oracle_wf1[K] = mean_valid([r["topk_eval"][str(K)]["grounded_f1"] for r in grounded_docs])
        print(f"Top-{K:2d} Oracle Word Grounding F1: {oracle_wf1[K]*100:.2f}% (Lift over baseline: +{(oracle_wf1[K] - base_wf1)*100:.2f} pp)")

    # 5. Failure Class Distribution
    failure_counts = Counter(f["failure_class"] for f in gradeable_field_records)
    print("\n--- Failure Class Distribution (Gradeable Fields) ---")
    for fc, cnt in failure_counts.most_common():
        pct = cnt / total_gradeable_fields * 100
        print(f"  {fc:<38}: {cnt:6d} ({pct:6.2f}%)")

    # 6. Save Parquet
    print(f"\nWriting field_records.parquet ({len(gradeable_field_records)} rows)...")
    arrow_table = pa.Table.from_pylist(gradeable_field_records)
    pq.write_table(arrow_table, obs_dir / "field_records.parquet", compression="snappy")
    print(f"Parquet file written to {obs_dir / 'field_records.parquet'}")

    # 7. Save Aggregates JSON
    aggregates = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_documents": total_docs,
        "grounded_documents": len(grounded_docs),
        "ungrounded_documents": len(ungrounded_docs),
        "total_gradeable_fields": total_gradeable_fields,
        "total_field_records": len(all_field_records),
        "baseline_metrics": {
            "word_grounding_f1": base_wf1,
            "word_grounding_precision": base_wprec,
            "word_grounding_recall": base_wrec,
            "page_grounding_f1": base_pf1,
            "page_grounding_precision": base_pprec,
            "page_grounding_recall": base_prec,
            "value_f1": base_vf1,
        },
        "candidate_recall": {
            "micro": {str(k): micro_hits[k] / total_gradeable_fields for k in (1, 3, 5, 10, 20)},
            "micro_numerators": {str(k): micro_hits[k] for k in (1, 3, 5, 10, 20)},
            "micro_denominator": total_gradeable_fields,
            "macro_true_grounded_docs": {str(k): macro_recall[k] for k in (1, 3, 5, 10, 20)},
            "macro_true_denominator_docs": len(grounded_docs),
            "macro_flawed_previous": {str(k): flawed_recall[k] for k in (1, 3, 5, 10, 20)},
            "macro_flawed_denominator_docs": total_docs,
        },
        "oracle_ceilings": {
            str(k): oracle_wf1[k] for k in (1, 3, 5, 10, 20)
        },
        "failure_classes": {
            fc: {"count": cnt, "percentage": cnt / total_gradeable_fields * 100}
            for fc, cnt in failure_counts.items()
        },
    }

    with open(obs_dir / "aggregates.json", "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2)
    print(f"Aggregates written to {obs_dir / 'aggregates.json'}")


if __name__ == "__main__":
    main()
