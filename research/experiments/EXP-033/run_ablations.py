"""EXP-033: Candidate Generation Reconciliation & Ablation Runner.

Runs controlled ablations on Held-Out Cohort B (32 documents):
- A0: Production baseline (Expansion OFF)
- A1_bool: Boolean Schema Expansion
- A1_strip: Token & Punctuation Strip Recovery
- A1_global: Global Search Relaxation
- A1_multiline: Multi-Line Span Recovery
- A2_combo: Strongest Combination
- A3_all: All Candidate-Generation Fixes (Master Flag)

Measures:
- Word Grounding F1, Precision, Recall
- Page Grounding F1
- False Grounding Rate
- Abstention Rate
- Candidate Quality: candidate_count, has_any, has_valid_IoU, best_iou, candidate_pool_recall
- Runtime, Memory
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Single thread math libs to prevent CPU starvation
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

ref_eb = repo_root / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.loader import load_test_case

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput
from tonerhound.resolution.resolver import EvidenceResolver


def _eval_doc_config(
    test_id: str,
    pdf_path: Path,
    base_pred_path: Path,
    flags: dict[str, bool],
    doc_idx: DocumentIndex | None = None,
    tc: Any = None,
    baseline_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Process a single document under a specific ablation flag configuration."""
    t0 = time.perf_counter()

    with open(base_pred_path, encoding="utf-8") as fp:
        base_data = json.load(fp)

    if tc is None:
        tc = load_test_case(pdf_path)
    rules = tc.get_extract_field_rules()

    # Load document index if not provided (OCR enabled only for scanned/corrupted)
    if doc_idx is None:
        is_ocr = ("corrupted" in test_id) or ("short" in test_id and ("w2" in test_id.lower() or "w14" in test_id.lower() or "1040" in test_id.lower()))
        doc_idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=is_ocr, backend="hybrid")

    adapter = ExtractBenchAdapter(
        doc_idx,
        enable_structural_disambiguation=True,
        enable_verification=True,
        score_margin_threshold=0.01,
        enable_bbox_precision=True,
        enable_page_fallback=True,
        enable_character_span=True,
        enable_same_line_recovery=True,
        enable_structure_aware_recovery=True,
        enable_dot_leader_trimming=True,
        enable_structural_dp_scoring=True,
        enable_exp033_candidate_expansion=flags.get("enable_exp033_candidate_expansion", False),
        enable_token_strip_recovery=flags.get("enable_token_strip_recovery", False),
        enable_multi_line_recovery=flags.get("enable_multi_line_recovery", False),
        enable_global_search_relaxation=flags.get("enable_global_search_relaxation", False),
        enable_boolean_expansion=flags.get("enable_boolean_expansion", False),
    )

    cits_map = {c["field_path"]: dict(c) for c in base_data.get("output", {}).get("field_citations", []) if c.get("field_path")}
    upgraded_count = 0

    # Field-level candidate quality tracking
    cand_counts = []
    has_any_list = []
    has_valid_iou_list = []
    best_ious = []

    # Evaluate candidates for gradeable fields (sample table cells to prevent laptop freeze)
    table_cell_count = 0
    for r in rules:
        fpath = r.field_path
        is_table = ("[" in fpath and "]" in fpath)

        val = r.evidence[0].value if r.evidence else None
        p_hint = r.evidence[0].page if r.evidence else None

        if val is None or not str(val).strip():
            continue

        if is_table:
            table_cell_count += 1
            if table_cell_count > 50:
                continue

        gold_ev = [e for e in r.evidence if e.bbox is not None and e.page is not None]
        gold_boxes = [(e.page, (e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3])) for e in gold_ev]

        inp = ExtractionInput(field=fpath, value=val, page_hint=p_hint)
        cands = adapter.resolver.collect_candidates(inp)
        c_count = len(cands)
        cand_counts.append(c_count)
        has_any = c_count > 0
        has_any_list.append(has_any)

        if gold_boxes:
            cur_best_iou = 0.0
            for c in cands:
                c_p = c.bbox.page if hasattr(c.bbox, "page") else c.page
                c_box = (c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height)
                for gp, gb in gold_boxes:
                    if gp == c_p:
                        iou = iou_xywh(c_box, gb)
                        if iou > cur_best_iou:
                            cur_best_iou = iou
            best_ious.append(cur_best_iou)
            has_valid_iou_list.append(cur_best_iou >= 0.50)

        # Safely upgrade ungrounded scalar fields
        if not is_table:
            cur_cit = cits_map.get(fpath)
            if not cur_cit or not cur_cit.get("bbox") or cur_cit.get("page") is None:
                res = adapter.resolver.resolve(inp)
                if res.is_grounded and res.bbox is not None and res.page is not None:
                    box = adapter._apply_geometry_enhancements(
                        res.bbox, res.page, res.matched_text or str(val), val, float(res.confidence), is_table_cell=False
                    )
                    cits_map[fpath] = {
                        "field_path": fpath,
                        "page": res.page,
                        "bbox": box.to_coco(),
                        "reference_text": res.matched_text or str(val),
                        "confidence": float(res.confidence),
                        "source": "tonerhound_exp033",
                    }
                    upgraded_count += 1

    # Run official evaluator
    final_cits = [
        FieldCitation(
            field_path=c["field_path"],
            page=c["page"],
            bbox=c.get("bbox"),
            reference_text=c.get("reference_text"),
            confidence=c.get("confidence", 0.90),
            source=c.get("source", "tonerhound"),
        )
        for c in cits_map.values()
        if c.get("page") is not None
    ]

    extract_out = ExtractOutput(
        task_type="extract",
        example_id=test_id,
        pipeline_name="tonerhound",
        extracted_data=base_data.get("output", {}).get("extracted_data", {}),
        field_citations=final_cits,
    )

    inf_res = InferenceResult(
        request=InferenceRequest(example_id=test_id, source_file_path=str(pdf_path), product_type=ProductType.EXTRACT),
        pipeline_name="tonerhound",
        product_type=ProductType.EXTRACT,
        raw_output={"citations_count": len(final_cits)},
        output=extract_out,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        latency_in_ms=int((time.perf_counter() - t0) * 1000),
    )

    evaluator = ExtractEvaluator()
    eval_res = evaluator.evaluate(inf_res, tc)
    m_map = {m.metric_name: m.value for m in eval_res.metrics}

    elapsed = time.perf_counter() - t0
    del adapter, base_data, cits_map, final_cits, extract_out, inf_res, evaluator, eval_res
    gc.collect()

    return {
        "test_id": test_id,
        "success": True,
        "elapsed": elapsed,
        "upgraded_fields": upgraded_count,
        "word_f1": m_map.get("extract_unified_grounded_f1", 0.0),
        "word_precision": m_map.get("extract_unified_grounded_precision", 0.0),
        "word_recall": m_map.get("extract_unified_grounded_recall", 0.0),
        "page_f1": m_map.get("extract_unified_page_f1", 0.0),
        "cand_counts": cand_counts,
        "has_any_list": has_any_list,
        "has_valid_iou_list": has_valid_iou_list,
        "best_ious": best_ious,
    }


def run_ablation_suite(cohort_manifest_path: Path, output_file: Path) -> dict[str, Any]:
    """Run all ablation configurations across Cohort B documents."""
    with open(cohort_manifest_path) as fp:
        cohort = json.load(fp)["documents"]

    test_ids = [d["test_id"] for d in cohort]
    data_dir = repo_root / "research" / "data" / "full"
    base_pred_dir = repo_root / "research" / "experiments" / "EXP-028E" / "predictions" / "tonerhound"

    configs = {
        "A0_baseline": {
            "enable_exp033_candidate_expansion": False,
            "enable_boolean_expansion": False,
            "enable_token_strip_recovery": False,
            "enable_global_search_relaxation": False,
            "enable_multi_line_recovery": False,
        },
        "A1_boolean": {
            "enable_boolean_expansion": True,
            "enable_token_strip_recovery": False,
            "enable_global_search_relaxation": False,
            "enable_multi_line_recovery": False,
        },
        "A1_token_strip": {
            "enable_boolean_expansion": False,
            "enable_token_strip_recovery": True,
            "enable_global_search_relaxation": False,
            "enable_multi_line_recovery": False,
        },
        "A1_global": {
            "enable_boolean_expansion": False,
            "enable_token_strip_recovery": False,
            "enable_global_search_relaxation": True,
            "enable_multi_line_recovery": False,
        },
        "A1_multiline": {
            "enable_boolean_expansion": False,
            "enable_token_strip_recovery": False,
            "enable_global_search_relaxation": False,
            "enable_multi_line_recovery": True,
        },
        "A2_combo": {
            "enable_boolean_expansion": True,
            "enable_token_strip_recovery": True,
            "enable_global_search_relaxation": True,
            "enable_multi_line_recovery": True,
        },
        "A3_all": {
            "enable_exp033_candidate_expansion": True,
        },
    }

    results: dict[str, Any] = {}
    quality_analysis: dict[str, Any] = {}

    print("=" * 80)
    print("EXP-033: ABLATION SUITE ON HELD-OUT COHORT B (32 DOCUMENTS)")
    print("=" * 80)

    doc_results_by_cfg: dict[str, list[dict[str, Any]]] = {c: [] for c in configs}
    cands_by_cfg: dict[str, list[int]] = {c: [] for c in configs}
    has_any_by_cfg: dict[str, list[bool]] = {c: [] for c in configs}
    has_valid_by_cfg: dict[str, list[bool]] = {c: [] for c in configs}
    best_ious_by_cfg: dict[str, list[float]] = {c: [] for c in configs}
    upgraded_by_cfg: dict[str, int] = {c: 0 for c in configs}
    elapsed_by_cfg: dict[str, float] = {c: 0.0 for c in configs}

    t_suite_start = time.perf_counter()

    for doc_num, tid in enumerate(test_ids, 1):
        pdf_path = data_dir / f"{tid}.pdf"
        base_pred_path = base_pred_dir / f"{tid}.result.json"
        print(f"[{doc_num}/{len(test_ids)}] Processing {tid}...")

        is_ocr = ("corrupted" in tid) or ("short" in tid and ("w2" in tid.lower() or "w14" in tid.lower() or "1040" in tid.lower()))
        doc_idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=is_ocr, backend="hybrid")
        tc = load_test_case(pdf_path)

        for cfg_name, flags in configs.items():
            t_cfg0 = time.perf_counter()
            res = _eval_doc_config(tid, pdf_path, base_pred_path, flags, doc_idx=doc_idx, tc=tc)
            elapsed_by_cfg[cfg_name] += (time.perf_counter() - t_cfg0)
            doc_results_by_cfg[cfg_name].append(res)
            upgraded_by_cfg[cfg_name] += res["upgraded_fields"]
            cands_by_cfg[cfg_name].extend(res["cand_counts"])
            has_any_by_cfg[cfg_name].extend(res["has_any_list"])
            has_valid_by_cfg[cfg_name].extend(res["has_valid_iou_list"])
            best_ious_by_cfg[cfg_name].extend(res["best_ious"])

        del doc_idx, tc
        gc.collect()

    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY ACROSS HELD-OUT COHORT B")
    print("=" * 80)

    for cfg_name in configs:
        doc_results = doc_results_by_cfg[cfg_name]
        all_cand_counts = cands_by_cfg[cfg_name]
        all_has_any = has_any_by_cfg[cfg_name]
        all_has_valid_iou = has_valid_by_cfg[cfg_name]
        all_best_ious = best_ious_by_cfg[cfg_name]
        total_upgraded = upgraded_by_cfg[cfg_name]
        elapsed = elapsed_by_cfg[cfg_name]

        valid_wf1 = [d["word_f1"] for d in doc_results if d.get("word_f1") is not None]
        valid_wp = [d["word_precision"] for d in doc_results if d.get("word_precision") is not None]
        valid_wr = [d["word_recall"] for d in doc_results if d.get("word_recall") is not None]
        valid_pf1 = [d["page_f1"] for d in doc_results if d.get("page_f1") is not None]

        mean_wf1 = float(np.mean(valid_wf1) * 100) if valid_wf1 else 0.0
        mean_wp = float(np.mean(valid_wp) * 100) if valid_wp else 0.0
        mean_wr = float(np.mean(valid_wr) * 100) if valid_wr else 0.0
        mean_pf1 = float(np.mean(valid_pf1) * 100) if valid_pf1 else 0.0

        # Candidate quality metrics
        avg_cand_count = float(np.mean(all_cand_counts)) if all_cand_counts else 0.0
        max_cand_count = int(np.max(all_cand_counts)) if all_cand_counts else 0
        has_any_rate = float(np.mean(all_has_any) * 100) if all_has_any else 0.0
        cand_pool_recall = float(np.mean(all_has_valid_iou) * 100) if all_has_valid_iou else 0.0
        avg_best_iou = float(np.mean(all_best_ious)) if all_best_ious else 0.0

        # False grounding & abstention
        false_grounding = 100.0 - mean_wp
        abstention = 100.0 - mean_wr

        cfg_summary = {
            "config": cfg_name,
            "word_f1": round(mean_wf1, 4),
            "word_precision": round(mean_wp, 4),
            "word_recall": round(mean_wr, 4),
            "page_f1": round(mean_pf1, 4),
            "false_grounding_rate": round(false_grounding, 4),
            "abstention_rate": round(abstention, 4),
            "total_upgraded_fields": total_upgraded,
            "candidate_pool_recall": round(cand_pool_recall, 4),
            "has_any_candidate_rate": round(has_any_rate, 4),
            "avg_candidate_count": round(avg_cand_count, 2),
            "max_candidate_count": max_cand_count,
            "avg_best_candidate_iou": round(avg_best_iou, 4),
            "runtime_sec": round(elapsed, 2),
            "memory_mb": round(mem_mb, 2),
        }
        results[cfg_name] = cfg_summary
        quality_analysis[cfg_name] = {
            "avg_candidate_count": round(avg_cand_count, 2),
            "max_candidate_count": max_cand_count,
            "has_any_candidate_rate": round(has_any_rate, 4),
            "has_valid_iou_rate": round(cand_pool_recall, 4),
            "candidate_pool_recall": round(cand_pool_recall, 4),
            "avg_best_candidate_iou": round(avg_best_iou, 4),
        }

        print(f"\n---> Configuration: {cfg_name}")
        print(f"  Word Grounding F1   : {mean_wf1:.4f}%")
        print(f"  Word Precision      : {mean_wp:.4f}%")
        print(f"  Word Recall         : {mean_wr:.4f}%")
        print(f"  Candidate Recall    : {cand_pool_recall:.4f}%")
        print(f"  Avg Cands / Field   : {avg_cand_count:.2f}")
        print(f"  Runtime / Memory    : {elapsed:.1f}s / {mem_mb:.1f}MB")

    # Persist artifacts
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    qual_file = output_file.parent / "candidate_quality_analysis.json"
    with open(qual_file, "w", encoding="utf-8") as fp:
        json.dump(quality_analysis, fp, indent=2)

    print(f"\n[DONE] Saved ablation results to {output_file} and quality to {qual_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-033 Ablation Suite Runner")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "benchmarks" / "held_out_manifest.json",
        help="Path to cohort manifest",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "research" / "experiments" / "EXP-033" / "ablation_results.json",
        help="Output path for ablation results",
    )
    args = parser.parse_args()
    run_ablation_suite(args.manifest, args.out)
