"""EXP-036C Phase 5: Frozen Held-Out Cohort B Evaluation.

Evaluates the deterministic visual detector on the 32 unseen documents from
benchmarks/held_out_manifest.json containing 25 boolean/checkbox failure targets
across 5 documents.

Measures all Section 15 requirements:
- Detection: target localization rate, candidate recall, false detections per page
- State: checked precision/recall, unchecked precision/recall, state accuracy
- Grounding: IoU >= 0.50 rate, mean IoU, median IoU
- End-to-end: exact field grounding recovery, abstention rate, false positive rate

Outputs:
- research/experiments/EXP-036/heldout_results.json
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz

from phase3_full_page_detector import compute_iou, detect_page_candidates

repo_root = Path(__file__).resolve().parent.parent.parent.parent


def run_phase5():
    manifest_path = repo_root / "benchmarks" / "held_out_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    held_out_doc_ids = [d["test_id"] for d in manifest["documents"]]

    audit_path = repo_root / "research" / "experiments" / "EXP-035" / "failure_audit.json"
    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)

    # 25 target cases in Cohort B
    held_out_targets = [
        c for c in audit["cases"]
        if c["document_id"] in held_out_doc_ids and c["failure_mechanism"] == "WRONG_CLASSIFICATION"
    ]

    print("=" * 80)
    print(f"EXP-036C PHASE 5: EVALUATING ON FROZEN HELD-OUT COHORT B ({len(held_out_doc_ids)} DOCS, {len(held_out_targets)} TARGETS)")
    print("=" * 80)

    # Group targets by (doc_id, gold_page)
    target_pages = defaultdict(list)
    for t in held_out_targets:
        target_pages[(t["document_id"], t["gold_page"])].append(t)

    # Run detection on all target pages plus representative pages from non-target held-out docs
    eval_pages = list(target_pages.keys())
    # Add page 1 for other held-out docs to test false detection rates on non-form pages
    other_docs = [d for d in held_out_doc_ids if not any(t["document_id"] == d for t in held_out_targets)]
    for od in other_docs[:10]:
        eval_pages.append((od, 1))

    page_results = []
    target_evaluations = []
    all_ious = []
    candidate_counts = []
    total_render_sec = 0.0
    total_detect_sec = 0.0

    t0_all = time.perf_counter()

    for doc_id, page_num in eval_pages:
        pdf_path = repo_root / "research" / "data" / "full" / f"{doc_id}.pdf"
        if not pdf_path.exists():
            continue

        doc = fitz.open(pdf_path)
        if page_num > len(doc):
            doc.close()
            continue

        t0_rend = time.perf_counter()
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=300)
        img_w, img_h = pix.width, pix.height
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((img_h, img_w, pix.n))
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if pix.n >= 3 else arr[:, :, 0]
        rend_time = time.perf_counter() - t0_rend
        total_render_sec += rend_time

        t0_det = time.perf_counter()
        candidates = detect_page_candidates(gray)
        det_time = time.perf_counter() - t0_det
        total_detect_sec += det_time

        candidate_counts.append(len(candidates))

        page_rec = {
            "document_id": doc_id,
            "page": page_num,
            "image_dimensions": [img_w, img_h],
            "render_ms": round(rend_time * 1000, 2),
            "detect_ms": round(det_time * 1000, 2),
            "candidates_count": len(candidates),
        }
        page_results.append(page_rec)

        # Targets on this page
        cur_targets = target_pages.get((doc_id, page_num), [])
        for t in cur_targets:
            gb = t["gold_bbox"]
            gb_px = [gb[0] * img_w, gb[1] * img_h, gb[2] * img_w, gb[3] * img_h]
            gt_val = t["value"]
            gt_state = "CHECKED" if gt_val is True else ("UNCHECKED" if gt_val is False else "OTHER")

            best_iou = 0.0
            best_c = None
            for c in candidates:
                iou = compute_iou(c["bbox_px"], gb_px)
                if iou > best_iou:
                    best_iou = iou
                    best_c = c

            all_ious.append(best_iou)
            is_loc_50 = best_iou >= 0.50
            is_loc_30 = best_iou >= 0.30
            pred_state = best_c["state"] if best_c else "NONE"
            state_match = (pred_state == gt_state)
            grounding_success = is_loc_50 and state_match

            t_eval = {
                "document_id": doc_id,
                "field_path": t["field_path"],
                "gold_page": page_num,
                "gold_bbox": gb,
                "ground_truth_state": gt_state,
                "best_iou": round(best_iou, 4),
                "is_localized_50": is_loc_50,
                "is_localized_30": is_loc_30,
                "predicted_state": pred_state,
                "state_match": state_match,
                "grounding_success": grounding_success,
                "matched_candidate_confidence": best_c["confidence"] if best_c else 0.0,
            }
            target_evaluations.append(t_eval)

        doc.close()

    total_time = time.perf_counter() - t0_all
    num_eval_pages = len(page_results)

    # 1. Detection Metrics
    n_targets = len(target_evaluations)
    loc50_count = sum(1 for t in target_evaluations if t["is_localized_50"])
    loc30_count = sum(1 for t in target_evaluations if t["is_localized_30"])
    loc50_rate = (loc50_count / n_targets) if n_targets > 0 else 0.0
    loc30_rate = (loc30_count / n_targets) if n_targets > 0 else 0.0

    mean_cands_per_page = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    # Spurious candidates: total candidates on target pages minus localized targets
    target_page_keys = set(target_pages.keys())
    cands_on_target_pages = sum(p["candidates_count"] for p in page_results if (p["document_id"], p["page"]) in target_page_keys)
    spurious_cands_per_target_page = (cands_on_target_pages - loc50_count) / max(1, len(target_page_keys))

    # 2. State Confusion Matrix
    tp_checked = sum(1 for t in target_evaluations if t["ground_truth_state"] == "CHECKED" and t["predicted_state"] == "CHECKED")
    fp_checked = sum(1 for t in target_evaluations if t["ground_truth_state"] != "CHECKED" and t["predicted_state"] == "CHECKED")
    fn_checked = sum(1 for t in target_evaluations if t["ground_truth_state"] == "CHECKED" and t["predicted_state"] != "CHECKED")

    tp_unchecked = sum(1 for t in target_evaluations if t["ground_truth_state"] == "UNCHECKED" and t["predicted_state"] == "UNCHECKED")
    fp_unchecked = sum(1 for t in target_evaluations if t["ground_truth_state"] != "UNCHECKED" and t["predicted_state"] == "UNCHECKED")
    fn_unchecked = sum(1 for t in target_evaluations if t["ground_truth_state"] == "UNCHECKED" and t["predicted_state"] != "UNCHECKED")

    prec_checked = tp_checked / max(1, tp_checked + fp_checked)
    rec_checked = tp_checked / max(1, tp_checked + fn_checked)
    prec_unchecked = tp_unchecked / max(1, tp_unchecked + fp_unchecked)
    rec_unchecked = tp_unchecked / max(1, tp_unchecked + fn_unchecked)
    state_acc = sum(1 for t in target_evaluations if t["state_match"]) / max(1, n_targets)

    # 3. Grounding Metrics
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    median_iou = float(np.median(all_ious)) if all_ious else 0.0
    grounding_success_count = sum(1 for t in target_evaluations if t["grounding_success"])
    grounding_success_rate = grounding_success_count / max(1, n_targets)

    # 4. End-to-end Recovery
    abstention_count = sum(1 for t in target_evaluations if t["predicted_state"] in ("AMBIGUOUS", "NONE"))
    abstention_rate = abstention_count / max(1, n_targets)
    false_positives = sum(1 for t in target_evaluations if not t["grounding_success"])
    fp_rate = false_positives / max(1, n_targets)

    # Breakdown by document in Held-out
    doc_breakdown = defaultdict(lambda: {"total": 0, "loc50": 0, "state_match": 0, "success": 0})
    for t in target_evaluations:
        d = t["document_id"]
        doc_breakdown[d]["total"] += 1
        if t["is_localized_50"]:
            doc_breakdown[d]["loc50"] += 1
        if t["state_match"]:
            doc_breakdown[d]["state_match"] += 1
        if t["grounding_success"]:
            doc_breakdown[d]["success"] += 1

    summary = {
        "experiment": "EXP-036C_PHASE_5_HELDOUT",
        "held_out_manifest_docs_count": len(held_out_doc_ids),
        "held_out_target_fields_count": n_targets,
        "held_out_target_docs_count": len(target_pages),
        "pages_evaluated_count": num_eval_pages,
        "total_runtime_seconds": round(total_time, 2),
        "mean_render_ms_per_page": round((total_render_sec / num_eval_pages) * 1000, 2),
        "mean_detect_ms_per_page": round((total_detect_sec / num_eval_pages) * 1000, 2),
        "mean_candidates_per_page": round(mean_cands_per_page, 2),
        "spurious_candidates_per_target_page": round(spurious_cands_per_target_page, 2),
        "detection_metrics": {
            "target_localization_rate_iou_50": round(loc50_rate, 4),
            "target_localization_rate_iou_30": round(loc30_rate, 4),
            "localized_iou_50_count": loc50_count,
            "localized_iou_30_count": loc30_count,
            "total_targets": n_targets,
        },
        "state_metrics": {
            "state_accuracy": round(state_acc, 4),
            "checked_precision": round(prec_checked, 4),
            "checked_recall": round(rec_checked, 4),
            "unchecked_precision": round(prec_unchecked, 4),
            "unchecked_recall": round(rec_unchecked, 4),
            "confusion_counts": {
                "tp_checked": tp_checked,
                "fp_checked": fp_checked,
                "fn_checked": fn_checked,
                "tp_unchecked": tp_unchecked,
                "fp_unchecked": fp_unchecked,
                "fn_unchecked": fn_unchecked,
            },
        },
        "grounding_metrics": {
            "iou_ge_50_rate": round(loc50_rate, 4),
            "mean_iou": round(mean_iou, 4),
            "median_iou": round(median_iou, 4),
            "grounding_success_count": grounding_success_count,
            "grounding_success_rate": round(grounding_success_rate, 4),
        },
        "end_to_end_metrics": {
            "exact_field_grounding_recovery_count": grounding_success_count,
            "exact_field_grounding_recovery_rate": round(grounding_success_rate, 4),
            "abstention_count": abstention_count,
            "abstention_rate": round(abstention_rate, 4),
            "false_positive_rate": round(fp_rate, 4),
        },
        "document_breakdown": dict(doc_breakdown),
    }

    out_file = repo_root / "research" / "experiments" / "EXP-036" / "heldout_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "targets": target_evaluations,
            "pages": page_results,
        }, f, indent=2)

    print("=" * 80)
    print("PHASE 5 HELD-OUT COHORT B RESULTS SUMMARY:")
    print("=" * 80)
    print(f"Target Fields: {n_targets} across {len(doc_breakdown)} documents ({num_eval_pages} pages evaluated)")
    print(f"Localization (IoU >= 0.50): {loc50_count} / {n_targets} ({loc50_rate * 100:.1f}%)")
    print(f"Localization (IoU >= 0.30): {loc30_count} / {n_targets} ({loc30_rate * 100:.1f}%)")
    print(f"Mean IoU: {mean_iou:.4f} | Median IoU: {median_iou:.4f}")
    print(f"State Accuracy: {state_acc * 100:.1f}%")
    print(f"  Checked   : Prec={prec_checked*100:.1f}%, Rec={rec_checked*100:.1f}%")
    print(f"  Unchecked : Prec={prec_unchecked*100:.1f}%, Rec={rec_unchecked*100:.1f}%")
    print(f"Exact Field Grounding Recovery: {grounding_success_count} / {n_targets} ({grounding_success_rate * 100:.1f}%)")
    print(f"Abstention Rate: {abstention_rate * 100:.1f}% | False Positive Rate: {fp_rate * 100:.1f}%")
    print("-" * 80)
    print("Document Breakdown:")
    for d, st in doc_breakdown.items():
        print(f"  {d:<36} | {st['success']}/{st['total']} success ({st['success']/st['total']*100:.1f}%) | {st['loc50']}/{st['total']} loc50")
    print("=" * 80)
    print(f"Saved held-out results to {out_file}")


if __name__ == "__main__":
    run_phase5()
