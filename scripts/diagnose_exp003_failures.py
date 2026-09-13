"""Diagnostic failure analysis and candidate recall measurement script for EXP-003.

Performs Step 2 (Failure-First Analysis across 17 categories)
and Step 3 (Recall@1, Recall@5, Recall@10, Recall@20 measurement).
"""

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir / "src") not in sys.path:
    sys.path.insert(0, str(root_dir / "src"))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

from extract_bench.test_cases.loader import load_test_cases
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    build_rule_indexes,
    iter_rule_evidence,
    _validated_bbox,
)
from tonerhound.benchmark.adapter import ExtractBenchAdapter, _flatten_leaves_with_context
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import ExtractionInput, ProvenanceStatus


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def run_diagnostics():
    data_dir = root_dir / "research" / "data" / "test"
    cases = load_test_cases(data_dir)

    all_failures = []
    category_counter = Counter()
    doc_recall_stats = {}

    print("=== Starting EXP-003 Diagnostic Failure & Recall Analysis ===")

    for case in cases:
        print(f"\nProcessing case: {case.test_id} ({case.file_path.name})...")
        pdf_path = Path(case.file_path)

        # EXP-002E settings: OCR fallback enabled, structural disambiguation enabled
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=True)
        adapter = ExtractBenchAdapter(doc_index, enable_structural_disambiguation=True)

        payload = adapter.ground_extracted_data(case.expected_output, example_id=case.test_id)
        citations = payload["field_citations"]
        cit_by_field = {c["field_path"]: c for c in citations}

        # Build GT rule index
        alt, ev_boxes, ev_pages, normalizers = build_rule_indexes(case.test_rules)

        leaves = _flatten_leaves_with_context(case.expected_output)
        leaf_by_path = {leaf[0]: leaf for leaf in leaves}

        # Measure Recall@K on all GT rules that have bbox
        total_bbox_rules = 0
        hit_at_1 = 0
        hit_at_5 = 0
        hit_at_10 = 0
        hit_at_20 = 0

        # Also inspect each field with GT bbox
        for rule in case.test_rules:
            path = rule.field_path
            gt_boxes_for_path = ev_boxes.get(path, [])
            if not gt_boxes_for_path:
                continue

            total_bbox_rules += 1
            leaf = leaf_by_path.get(path)
            val = leaf[1] if leaf else None
            ctx = leaf[3] if leaf else ""
            p_hint = leaf[2] if leaf else None

            # Generate all candidates using matcher
            is_num = isinstance(val, (int, float)) and not isinstance(val, bool)
            raw_cands = []
            if is_num:
                raw_cands.extend(adapter.resolver.matcher.find_normalized_numeric_candidates(val, page_hint=p_hint))
            if isinstance(val, str):
                raw_cands.extend(adapter.resolver.matcher.find_normalized_date_candidates(val, page_hint=p_hint))
                raw_cands.extend(adapter.resolver.matcher.find_exact_candidates(val, page_hint=p_hint))
                if not raw_cands:
                    raw_cands.extend(adapter.resolver.matcher.find_normalized_numeric_candidates(val, page_hint=p_hint))
            elif not raw_cands and isinstance(val, (int, float)):
                raw_cands.extend(adapter.resolver.matcher.find_exact_candidates(str(val), page_hint=p_hint))

            if not raw_cands and p_hint is not None:
                if is_num:
                    raw_cands.extend(adapter.resolver.matcher.find_normalized_numeric_candidates(val, page_hint=None))
                if isinstance(val, str):
                    raw_cands.extend(adapter.resolver.matcher.find_normalized_date_candidates(val, page_hint=None))
                    raw_cands.extend(adapter.resolver.matcher.find_exact_candidates(val, page_hint=None))
                    if not raw_cands:
                        raw_cands.extend(adapter.resolver.matcher.find_normalized_numeric_candidates(val, page_hint=None))
                elif not raw_cands and isinstance(val, (int, float)):
                    raw_cands.extend(adapter.resolver.matcher.find_exact_candidates(str(val), page_hint=None))

            if not raw_cands and isinstance(val, str) and len(val.strip()) >= 4:
                raw_cands.extend(adapter.resolver.matcher.find_fuzzy_candidates(val, threshold=0.80, page_hint=p_hint))

            # Deduplicate candidates
            cands = []
            for c in raw_cands:
                if not any(u.page == c.page and u.bbox.iou(c.bbox) >= 0.90 for u in cands):
                    cands.append(c)

            # Score candidates
            scored = adapter.resolver._score_candidates_with_context(cands, ctx or "")
            scored.sort(key=lambda item: item[1], reverse=True)
            ranked_cands = [item[0] for item in scored]
            cand_scores = [float(item[1]) for item in scored]

            # Check candidate recall against GT boxes
            cand_ious = []
            for c in ranked_cands:
                best_iou = 0.0
                for gt_p, gt_b in gt_boxes_for_path:
                    if c.page == gt_p:
                        best_iou = max(best_iou, compute_iou(c.bbox.to_coco(), gt_b))
                cand_ious.append(best_iou)

            has_match_at_1 = any(cand_ious[:1][i] >= 0.50 for i in range(len(cand_ious[:1])))
            has_match_at_5 = any(cand_ious[:5][i] >= 0.50 for i in range(len(cand_ious[:5])))
            has_match_at_10 = any(cand_ious[:10][i] >= 0.50 for i in range(len(cand_ious[:10])))
            has_match_at_20 = any(cand_ious[:20][i] >= 0.50 for i in range(len(cand_ious[:20])))

            if has_match_at_1:
                hit_at_1 += 1
            if has_match_at_5:
                hit_at_5 += 1
            if has_match_at_10:
                hit_at_10 += 1
            if has_match_at_20:
                hit_at_20 += 1

            # Check TonerHound's actual prediction
            pred_cit = cit_by_field.get(path)
            is_correct = False
            best_pred_iou = 0.0
            matched_gt_box = None
            matched_gt_page = None

            if pred_cit:
                p_page = pred_cit["page"]
                p_box = pred_cit["bbox"]
                for gt_p, gt_b in gt_boxes_for_path:
                    if p_page == gt_p:
                        iou = compute_iou(p_box, gt_b)
                        if iou > best_pred_iou:
                            best_pred_iou = iou
                            matched_gt_box = gt_b
                            matched_gt_page = gt_p
                if best_pred_iou >= 0.50:
                    is_correct = True
            else:
                p_page = None
                p_box = None

            if not matched_gt_box and gt_boxes_for_path:
                matched_gt_page, matched_gt_box = gt_boxes_for_path[0]

            # Classification
            classification = ""
            failure_reason = ""

            is_ocr_doc = ("bianco" in case.test_id or "W14" in case.test_id)

            if is_correct:
                if pred_cit and pred_cit.get("source") == "tonerhound":
                    # Check matching technique
                    matched_text = pred_cit.get("reference_text", "")
                    if str(val).strip() == matched_text.strip():
                        classification = "correct exact"
                    elif is_num or any(ch in str(val) for ch in "-/"):
                        classification = "correct normalized"
                    else:
                        classification = "correct fuzzy"
            else:
                # Classify error
                if not pred_cit:
                    if len(cands) == 0:
                        if is_ocr_doc:
                            classification = "OCR error"
                            failure_reason = "No candidates found on OCR processed document"
                        else:
                            classification = "not-found"
                            failure_reason = "Matcher found 0 candidates in document text index"
                    else:
                        classification = "not-found"
                        failure_reason = f"No citation emitted although {len(cands)} candidates generated"
                else:
                    # Citation was emitted, but IoU < 0.50 or wrong page
                    if p_page != matched_gt_page:
                        classification = "wrong page"
                        failure_reason = f"Predicted page {p_page} does not match GT page {matched_gt_page}"
                    elif best_pred_iou < 0.50:
                        # On correct page, but IoU < 0.50
                        # Check coordinate anomalies
                        if any(coord < 0.0 or coord > 1.05 for coord in p_box):
                            classification = "coordinate conversion error"
                            failure_reason = f"BBox coordinates {p_box} out of range [0, 1]"
                        elif matched_gt_box:
                            p_w, p_h = p_box[2], p_box[3]
                            gt_w, gt_h = matched_gt_box[2], matched_gt_box[3]
                            # Over-wide check
                            if p_w > 1.8 * gt_w and best_pred_iou < 0.50:
                                classification = "over-wide bbox"
                                failure_reason = f"Predicted width {p_w:.4f} is over-wide vs GT width {gt_w:.4f}"
                            elif p_w < 0.55 * gt_w and best_pred_iou < 0.50:
                                classification = "under-wide bbox"
                                failure_reason = f"Predicted width {p_w:.4f} is under-wide vs GT width {gt_w:.4f}"
                            elif is_ocr_doc:
                                classification = "OCR error"
                                failure_reason = f"OCR bounding box misalignment on {case.test_id} (IoU={best_pred_iou:.3f})"
                            elif len(cands) > 1:
                                # Check if multiple candidates have identical values
                                matching_val_cands = [c for c in cands if c.matched_text == pred_cit.get("reference_text")]
                                if "table" in path.lower() or "row" in path.lower() or "items" in path.lower() or "lines" in path.lower() or "[" in path:
                                    classification = "table ambiguity"
                                    failure_reason = f"Table cell disambiguation picked wrong row/occurrence (IoU={best_pred_iou:.3f})"
                                elif len(matching_val_cands) > 1:
                                    classification = "duplicate ambiguity"
                                    failure_reason = f"Multiple duplicate candidates ({len(matching_val_cands)}) with identical text"
                                else:
                                    classification = "wrong occurrence"
                                    failure_reason = f"Selected candidate at y={p_box[1]:.3f} instead of GT occurrence at y={matched_gt_box[1]:.3f}"
                            else:
                                classification = "wrong bbox"
                                failure_reason = f"Token bounding box does not match GT boundary (IoU={best_pred_iou:.3f})"

                all_failures.append({
                    "document": case.test_id,
                    "field": path,
                    "predicted_value": val,
                    "predicted_bbox": p_box,
                    "predicted_page": p_page,
                    "expected_bbox": matched_gt_box,
                    "expected_page": matched_gt_page,
                    "iou": round(best_pred_iou, 4),
                    "candidate_count": len(cands),
                    "candidate_scores": [round(s, 4) for s in cand_scores[:5]],
                    "selected_candidate": (
                        {"page": p_page, "bbox": p_box, "text": pred_cit.get("reference_text")}
                        if pred_cit else None
                    ),
                    "rejected_candidates_count": max(0, len(cands) - (1 if pred_cit else 0)),
                    "resolution_status": "GROUNDED" if pred_cit else "NOT_FOUND",
                    "classification": classification,
                    "why_selected_won": (
                        "Highest contextual similarity score / closest row anchor"
                        if pred_cit else "No candidates met acceptance criteria"
                    ),
                    "failure_reason": failure_reason,
                })

            category_counter[classification] += 1

        if total_bbox_rules > 0:
            r1 = hit_at_1 / total_bbox_rules
            r5 = hit_at_5 / total_bbox_rules
            r10 = hit_at_10 / total_bbox_rules
            r20 = hit_at_20 / total_bbox_rules
            doc_recall_stats[case.test_id] = {
                "total_bbox_rules": total_bbox_rules,
                "recall@1": r1,
                "recall@5": r5,
                "recall@10": r10,
                "recall@20": r20,
            }
            print(f"  Recall@1: {r1*100:.2f}% | Recall@5: {r5*100:.2f}% | Recall@10: {r10*100:.2f}% | Recall@20: {r20*100:.2f}%")
        else:
            doc_recall_stats[case.test_id] = {
                "total_bbox_rules": 0,
                "recall@1": None,
                "recall@5": None,
                "recall@10": None,
                "recall@20": None,
            }
            print("  No bbox rules in ground truth.")

    # Save JSON report
    failures_json_path = root_dir / "research" / "failures" / "EXP-003.json"
    failures_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "category_counts": dict(category_counter),
            "doc_recall_stats": doc_recall_stats,
            "total_failures": len(all_failures),
            "failures": all_failures,
        }, f, indent=2)

    # Save Markdown report
    failures_md_path = root_dir / "research" / "failures" / "EXP-003.md"
    with open(failures_md_path, "w", encoding="utf-8") as f:
        f.write("# EXP-003 Failure-First Analysis & Candidate Recall Report\n\n")
        f.write("## 1. Classification of All Grounding Predictions\n\n")
        f.write("| Category | Count | Percentage |\n| :--- | :---: | :---: |\n")
        total_eval = sum(category_counter.values())
        for cat, cnt in category_counter.most_common():
            pct = (cnt / max(1, total_eval)) * 100.0
            f.write(f"| `{cat}` | {cnt} | {pct:.2f}% |\n")

        f.write("\n\n## 2. Candidate Recall@K Measurement\n\n")
        f.write("| Document | GT BBoxes | Recall@1 | Recall@5 | Recall@10 | Recall@20 |\n| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for doc_id, stat in doc_recall_stats.items():
            if stat["total_bbox_rules"] > 0:
                f.write(f"| `{doc_id}` | {stat['total_bbox_rules']} | {stat['recall@1']*100:.2f}% | {stat['recall@5']*100:.2f}% | {stat['recall@10']*100:.2f}% | {stat['recall@20']*100:.2f}% |\n")
            else:
                f.write(f"| `{doc_id}` | 0 | N/A | N/A | N/A | N/A |\n")

        # Average recall
        valid_recalls = [s for s in doc_recall_stats.values() if s["total_bbox_rules"] > 0]
        avg_r1 = sum(s["recall@1"] for s in valid_recalls) / len(valid_recalls)
        avg_r5 = sum(s["recall@5"] for s in valid_recalls) / len(valid_recalls)
        avg_r10 = sum(s["recall@10"] for s in valid_recalls) / len(valid_recalls)
        avg_r20 = sum(s["recall@20"] for s in valid_recalls) / len(valid_recalls)
        f.write(f"| **AVERAGE** | **{sum(s['total_bbox_rules'] for s in valid_recalls)}** | **{avg_r1*100:.2f}%** | **{avg_r5*100:.2f}%** | **{avg_r10*100:.2f}%** | **{avg_r20*100:.2f}%** |\n")

        f.write("\n\n## 3. Detailed Failure Case Log (Sample of Representative Failures)\n\n")
        for i, fail in enumerate(all_failures[:30]):
            f.write(f"### Failure #{i+1}: `{fail['field']}` in `{fail['document']}`\n")
            f.write(f"- **Classification**: `{fail['classification']}`\n")
            f.write(f"- **Predicted Value**: `{fail['predicted_value']}`\n")
            f.write(f"- **Predicted BBox**: `{fail['predicted_bbox']}` (Page {fail['predicted_page']})\n")
            f.write(f"- **Expected BBox**: `{fail['expected_bbox']}` (Page {fail['expected_page']})\n")
            f.write(f"- **IoU**: {fail['iou']}\n")
            f.write(f"- **Candidates**: Count = {fail['candidate_count']}, Top Scores = {fail['candidate_scores']}\n")
            f.write(f"- **Resolution Status**: `{fail['resolution_status']}`\n")
            f.write(f"- **Why Selected Won**: {fail['why_selected_won']}\n")
            f.write(f"- **Failure Reason**: {fail['failure_reason']}\n\n")

    print(f"\n[Analysis Complete] Results saved to:\n  {failures_json_path}\n  {failures_md_path}")


if __name__ == "__main__":
    run_diagnostics()
