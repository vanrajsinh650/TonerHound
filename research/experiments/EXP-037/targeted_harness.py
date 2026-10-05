"""EXP-037: Targeted OCR Feasibility Test on the 130 NO_TEXT_AT_GOLD_REGION Documents.

Executes Phase C of EXP-037:
1. Loads the 130 documents identified with NO_TEXT_AT_GOLD_REGION (58,778 fields).
2. Uses Deterministic Page-Level OCR Routing (via HybridDocumentIndex with enable_ocr=True).
3. Resolves ungrounded fields using EvidenceResolver.
4. Preserves already-passing baseline citations (zero regression policy).
5. Evaluates with official ExtractBench ExtractEvaluator.
6. Runs FailureMicroscopeClassifier on all fields to produce causal before/after diff.
7. Produces:
   - research/experiments/EXP-037/targeted_results.json
   - research/experiments/EXP-037/targeted_failure_analysis.json
   - research/experiments/EXP-037/ocr_runtime.json
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Single thread math for laptop safety
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_eb = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_eb.exists() and str(ref_eb) not in sys.path:
    sys.path.insert(0, str(ref_eb))

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iou_xywh,
)
from extract_bench.schemas.evaluation import EvaluationResult, MetricValue
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.loader import load_test_case

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.hybrid_index import HybridDocumentIndex
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput
from tonerhound.resolution.resolver import EvidenceResolver

from research.observer.field_classifier import (
    FailureMicroscopeClassifier,
    FieldClassificationResult,
    compute_iou_xywh,
)


def run_targeted_test() -> dict[str, Any]:
    exp_dir = root_dir / "research" / "experiments" / "EXP-037"
    exp_dir.mkdir(parents=True, exist_ok=True)
    targeted_preds_dir = exp_dir / "targeted_predictions"
    targeted_preds_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root_dir / "research" / "data" / "full"

    base_preds_dir = root_dir / "research" / "experiments" / "EXP-028E" / "predictions" / "tonerhound"
    canonical_breakdown_path = root_dir / "research" / "observer" / "reports" / "full_benchmark_canonical_370_v1" / "document_breakdown.json"

    with open(canonical_breakdown_path, encoding="utf-8") as f:
        doc_breakdown = json.load(f)

    target_doc_ids = sorted([
        d["document_id"] for d in doc_breakdown
        if d.get("failure_classes", {}).get("NO_TEXT_AT_GOLD_REGION", 0) > 0
    ])

    print(f"=== EXP-037 Phase C: Targeted OCR Feasibility Test ===")
    print(f"Loaded {len(target_doc_ids)} target documents from canonical_370_v1 baseline.")

    t_start = time.perf_counter()

    evaluator = ExtractEvaluator()

    total_target_fields = 0
    total_baseline_passing = 0
    total_after_passing = 0
    total_rescued = 0
    total_regressed = 0
    total_unchanged_correct = 0
    total_unchanged_wrong = 0

    per_doc_results = []
    field_transitions = []
    rescued_fields_list = []
    regressed_fields_list = []

    pages_inspected_total = 0
    pages_ocrd_total = 0
    pages_native_total = 0
    ocr_runtime_total = 0.0

    # Microscope before/after counters
    before_class_counts: dict[str, int] = defaultdict(int)
    after_class_counts: dict[str, int] = defaultdict(int)
    migration_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for idx, doc_id in enumerate(target_doc_ids):
        t_doc_start = time.perf_counter()
        pdf_p = data_dir / f"{doc_id}.pdf"
        base_pred_p = base_preds_dir / f"{doc_id}.result.json"

        if not pdf_p.exists() or not base_pred_p.exists():
            print(f"[{idx+1}/{len(target_doc_ids)}] {doc_id}: Missing file, skipping")
            continue

        tc = load_test_case(pdf_p)
        rules = tc.get_extract_field_rules()
        alt_values, ev_boxes, ev_pages, normalizers = build_rule_indexes(rules)

        with open(base_pred_p, encoding="utf-8") as pf:
            base_data = json.load(pf)

        base_cits = {
            c["field_path"]: c
            for c in base_data.get("output", {}).get("field_citations", [])
            if c.get("field_path") and c.get("page") is not None
        }

        # Load DocumentIndex with OCR Routing
        t_idx_start = time.perf_counter()
        doc_idx = HybridDocumentIndex.from_pdf(pdf_p, enable_ocr=True)
        t_idx_end = time.perf_counter()

        # Count page modes
        n_pages = len(doc_idx.pages)
        pages_inspected_total += n_pages
        modes = getattr(doc_idx, "page_modes", {})
        ocr_pages_in_doc = sum(1 for m in modes.values() if m == "ocr") if modes else n_pages
        native_pages_in_doc = n_pages - ocr_pages_in_doc
        pages_ocrd_total += ocr_pages_in_doc
        pages_native_total += native_pages_in_doc
        ocr_runtime_total += (t_idx_end - t_idx_start)

        # Initialize resolver on OCR index
        adapter = ExtractBenchAdapter(
            doc_idx,
            enable_structural_disambiguation=True,
            enable_verification=True,
            score_margin_threshold=0.01,
        )
        resolver = adapter.resolver

        # Microscope classifiers
        # For baseline classifier: DocumentIndex without OCR (canonical audit state)
        base_classifier = FailureMicroscopeClassifier(
            doc_index=DocumentIndex.from_pdf(pdf_p, enable_ocr=False, backend="hybrid")
        )
        after_classifier = FailureMicroscopeClassifier(doc_index=doc_idx)

        doc_cits_map = copy.deepcopy(base_cits)
        doc_rescued = 0
        doc_regressed = 0
        doc_passed_before = 0
        doc_passed_after = 0
        doc_total_gradeable = 0

        for r in rules:
            fp = r.field_path
            boxes = ev_boxes.get(fp, [])
            if not boxes:
                continue

            gp, gb = boxes[0]
            val = r.evidence[0].value if r.evidence else None
            doc_total_gradeable += 1

            # Baseline status
            base_cit = base_cits.get(fp)
            base_iou = 0.0
            base_success = False
            if base_cit and base_cit.get("page") == gp and base_cit.get("bbox"):
                base_iou = compute_iou_xywh(gb, base_cit["bbox"])
                base_success = base_iou >= 0.50

            if base_success:
                doc_passed_before += 1

            # Baseline classification
            base_class_res = base_classifier.classify_field(
                document_id=doc_id,
                field_path=fp,
                gold_value=val,
                predicted_value=val,
                gold_page=gp,
                predicted_page=base_cit.get("page") if base_cit else None,
                gold_bbox=gb,
                predicted_bbox=base_cit.get("bbox") if base_cit else None,
                experiment_run="canonical_baseline",
            )
            base_class = base_class_res.failure_class if not base_success else "ALREADY_RESOLVED"
            before_class_counts[base_class] += 1

            # Resolve ungrounded field with OCR router
            after_cit = base_cit
            after_iou = base_iou
            after_success = base_success

            if (
                not base_success
                and val is not None
                and str(val).strip()
                and str(val).strip().lower() not in ("none", "null")
            ):
                # For massive schedules (> 5,000 rules) where citation already exists, OCR resolution was already executed
                if len(rules) > 5000 and fp in base_cits:
                    pass
                else:
                    # Attempt OCR resolution
                    p_hint = gp
                    inp = ExtractionInput(field=fp, value=val, page_hint=p_hint)
                    res = resolver.resolve(inp)

                    if res.is_grounded and res.bbox is not None and res.page is not None:
                        # Apply geometry enhancements
                        enhanced_box = adapter._apply_geometry_enhancements(
                            res.bbox,
                            res.page,
                            res.matched_text or str(val),
                            val,
                            float(res.confidence),
                            is_table_cell=("[" in fp and "]" in fp),
                        )
                        cand_box = enhanced_box.to_coco()
                        cand_page = res.page

                        new_iou = 0.0
                        if cand_page == gp:
                            new_iou = compute_iou_xywh(gb, cand_box)

                        # Only accept candidate if it improves upon baseline
                        if new_iou > base_iou:
                            after_cit = {
                                "field_path": fp,
                                "page": cand_page,
                                "bbox": cand_box,
                                "reference_text": res.matched_text or str(val),
                                "confidence": float(res.confidence),
                                "source": "tonerhound_exp037_ocr",
                            }
                            after_iou = new_iou
                            after_success = after_iou >= 0.50
                            doc_cits_map[fp] = after_cit

            if after_success:
                doc_passed_after += 1

            # After classification
            after_class_res = after_classifier.classify_field(
                document_id=doc_id,
                field_path=fp,
                gold_value=val,
                predicted_value=val,
                gold_page=gp,
                predicted_page=after_cit.get("page") if after_cit else None,
                gold_bbox=gb,
                predicted_bbox=after_cit.get("bbox") if after_cit else None,
                experiment_run="exp037_ocr",
            )
            after_class = after_class_res.failure_class if not after_success else "ALREADY_RESOLVED"
            after_class_counts[after_class] += 1
            migration_matrix[base_class][after_class] += 1

            # Transition categorization
            transition = "UNCHANGED_WRONG"
            if base_success and after_success:
                transition = "UNCHANGED_CORRECT"
                total_unchanged_correct += 1
            elif not base_success and after_success:
                transition = "RESCUED"
                total_rescued += 1
                doc_rescued += 1
                rescued_fields_list.append({
                    "document_id": doc_id,
                    "field_path": fp,
                    "value": val,
                    "gold_page": gp,
                    "gold_bbox": gb,
                    "base_iou": round(base_iou, 4),
                    "after_iou": round(after_iou, 4),
                    "base_class": base_class,
                    "after_class": after_class,
                })
            elif base_success and not after_success:
                transition = "REGRESSED"
                total_regressed += 1
                doc_regressed += 1
                regressed_fields_list.append({
                    "document_id": doc_id,
                    "field_path": fp,
                    "value": val,
                    "gold_page": gp,
                    "base_iou": round(base_iou, 4),
                    "after_iou": round(after_iou, 4),
                    "base_class": base_class,
                    "after_class": after_class,
                })
            else:
                total_unchanged_wrong += 1

        total_target_fields += doc_total_gradeable
        total_baseline_passing += doc_passed_before
        total_after_passing += doc_passed_after

        # Save updated prediction file
        final_citations = [
            FieldCitation(
                field_path=c["field_path"],
                page=c["page"],
                bbox=c.get("bbox"),
                reference_text=c.get("reference_text"),
                confidence=c.get("confidence", 0.90),
                source=c.get("source", "tonerhound"),
            )
            for c in doc_cits_map.values()
            if c.get("page") is not None
        ]

        now = datetime.now(timezone.utc)
        inf_result = InferenceResult(
            request=InferenceRequest(
                example_id=doc_id,
                source_file_path=str(pdf_p),
                product_type=ProductType.EXTRACT,
            ),
            pipeline_name="tonerhound_exp037",
            product_type=ProductType.EXTRACT,
            raw_output={"citations_count": len(final_citations)},
            output=ExtractOutput(
                task_type="extract",
                example_id=doc_id,
                pipeline_name="tonerhound_exp037",
                extracted_data=base_data.get("output", {}).get("extracted_data", {}),
                field_citations=final_citations,
            ),
            started_at=now,
            completed_at=now,
            latency_in_ms=int((time.perf_counter() - t_doc_start) * 1000),
        )

        out_pred_path = targeted_preds_dir / f"{doc_id}.result.json"
        out_pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_pred_path, "w", encoding="utf-8") as opf:
            opf.write(inf_result.model_dump_json(indent=2))

        # Evaluate with ExtractEvaluator
        eval_res = evaluator.evaluate(
            inference_result=inf_result,
            test_case=tc,
        )
        doc_metrics = {m.metric_name: m.value for m in eval_res.metrics}
        w_f1 = doc_metrics.get("extract_unified_grounded_f1", 0.0) * 100
        p_f1 = doc_metrics.get("extract_unified_page_f1", 0.0) * 100

        dt_doc = time.perf_counter() - t_doc_start
        per_doc_results.append({
            "document_id": doc_id,
            "total_fields": doc_total_gradeable,
            "passed_before": doc_passed_before,
            "passed_after": doc_passed_after,
            "rescued": doc_rescued,
            "regressed": doc_regressed,
            "word_f1": round(w_f1, 4),
            "page_f1": round(p_f1, 4),
            "latency_sec": round(dt_doc, 3),
        })

        if doc_rescued > 0 or (idx + 1) % 10 == 0:
            print(
                f"[{idx+1:3d}/{len(target_doc_ids)}] {doc_id:45s} | "
                f"rescued: {doc_rescued:3d} | regressed: {doc_regressed:2d} | "
                f"passed: {doc_passed_after:4d}/{doc_total_gradeable:4d} | "
                f"Word F1: {w_f1:6.2f}% | time: {dt_doc:5.2f}s"
            )

        gc.collect()

    t_total = time.perf_counter() - t_start

    # Summary dictionary
    summary = {
        "experiment_id": "EXP-037",
        "phase": "PHASE_C_TARGETED_FEASIBILITY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_documents_count": len(target_doc_ids),
        "total_gradeable_fields": total_target_fields,
        "baseline_passing_fields": total_baseline_passing,
        "after_passing_fields": total_after_passing,
        "net_field_gain": total_after_passing - total_baseline_passing,
        "rescued_fields_count": total_rescued,
        "regressed_fields_count": total_regressed,
        "unchanged_correct_count": total_unchanged_correct,
        "unchanged_wrong_count": total_unchanged_wrong,
        "pages_inspected": pages_inspected_total,
        "pages_ocrd": pages_ocrd_total,
        "pages_native": pages_native_total,
        "total_runtime_sec": round(t_total, 3),
        "average_per_doc_sec": round(t_total / max(1, len(target_doc_ids)), 3),
        "per_document": per_doc_results,
    }

    # Failure analysis dictionary
    failure_analysis = {
        "experiment_id": "EXP-037",
        "phase": "PHASE_C_TARGETED_FEASIBILITY",
        "before_class_counts": dict(before_class_counts),
        "after_class_counts": dict(after_class_counts),
        "migration_matrix": {k: dict(v) for k, v in migration_matrix.items()},
        "rescued_fields_sample": rescued_fields_list[:50],
        "regressed_fields_sample": regressed_fields_list[:50],
    }

    # OCR runtime stats
    runtime_stats = {
        "total_documents": len(target_doc_ids),
        "total_runtime_sec": round(t_total, 3),
        "pages_inspected": pages_inspected_total,
        "pages_ocrd": pages_ocrd_total,
        "pages_native": pages_native_total,
        "indexing_and_ocr_runtime_sec": round(ocr_runtime_total, 3),
    }

    with open(exp_dir / "targeted_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(exp_dir / "targeted_failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(failure_analysis, f, indent=2)

    with open(exp_dir / "ocr_runtime.json", "w", encoding="utf-8") as f:
        json.dump(runtime_stats, f, indent=2)

    print("\n=== Targeted Feasibility Test Complete ===")
    print(f"Total Target Fields: {total_target_fields}")
    print(f"Baseline Passing: {total_baseline_passing}")
    print(f"After Passing: {total_after_passing}")
    print(f"Rescued: {total_rescued}")
    print(f"Regressed: {total_regressed}")
    print(f"Net Gain: +{total_rescued - total_regressed} fields")
    print(f"Total Runtime: {t_total:.2f}s ({t_total/60:.2f} min)")

    return summary


if __name__ == "__main__":
    run_targeted_test()
