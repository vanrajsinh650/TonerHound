#!/usr/bin/env python3
"""EXP-005 Part 4 Pass 4: Forensic IoU Analysis of G2 Failures on real_ftx_full_corrupted.pdf.

Author: Agent A (IoU Forensics Lead)
Analyzes the 3,936 G2 failures on research/data/full/long/real_ftx_full_corrupted.pdf,
focusing on the 0.35–0.49 IoU near-miss band.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Ensure extract_bench and tonerhound are importable
repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
if _REF.exists():
    sys.path.insert(0, str(_REF))
sys.path.insert(0, str(repo_root / "src"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from extract_bench.test_cases.loader import load_test_case
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.benchmark.correlation import FailureRegistry
from tonerhound.document.index import DocumentIndex


def compute_iou_xywh(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    return float(iou_xywh(box_a, box_b))


def simulate_counterfactual_ious(gt_box: tuple[float, float, float, float], pred_box: tuple[float, float, float, float]) -> dict[str, float]:
    """Counterfactual experiment: what would IoU be if individual dimensions were set to ground truth?"""
    gx, gy, gw, gh = gt_box
    px, py, pw, ph = pred_box

    # Fix only width
    fix_w = (px, py, gw, ph)
    # Fix only x
    fix_x = (gx, py, pw, ph)
    # Fix horizontal (x and w)
    fix_xw = (gx, py, gw, ph)
    # Fix only height
    fix_h = (px, py, pw, gh)
    # Fix only y
    fix_y = (px, gy, pw, ph)
    # Fix vertical (y and h)
    fix_yh = (px, gy, pw, gh)

    return {
        "raw": compute_iou_xywh(gt_box, pred_box),
        "fix_w": compute_iou_xywh(gt_box, fix_w),
        "fix_x": compute_iou_xywh(gt_box, fix_x),
        "fix_xw": compute_iou_xywh(gt_box, fix_xw),
        "fix_h": compute_iou_xywh(gt_box, fix_h),
        "fix_y": compute_iou_xywh(gt_box, fix_y),
        "fix_yh": compute_iou_xywh(gt_box, fix_yh),
    }


def main():
    pdf_path = repo_root / "research/data/full/long/real_ftx_full_corrupted.pdf"
    cache_cits_path = repo_root / "research/cache/current_adapter_ftx_citations.json"
    cache_cits_path.parent.mkdir(parents=True, exist_ok=True)

    print("[Forensics] Loading failure registry and test case...")
    reg = FailureRegistry.load(repo_root / "experiments/EXP-005-ftx-failure-registry.json")
    tc = load_test_case(pdf_path)

    citations: list[dict[str, Any]]
    if cache_cits_path.exists():
        print(f"[Forensics] Loading cached predictions from {cache_cits_path}...")
        with open(cache_cits_path, "r", encoding="utf-8") as f:
            citations = json.load(f)
    else:
        print(f"[Forensics] Running ExtractBenchAdapter on {pdf_path}...")
        idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True)
        adapter = ExtractBenchAdapter(
            idx,
            enable_structural_disambiguation=True,
            enable_verification=True,
            score_margin_threshold=0.01,
            enable_bbox_precision=True,
        )
        payload = adapter.ground_extracted_data(tc.expected_output, example_id="long/real_ftx_full_corrupted")
        citations = payload["field_citations"]
        with open(cache_cits_path, "w", encoding="utf-8") as f:
            json.dump(citations, f)
        print(f"[Forensics] Saved cached predictions ({len(citations)} citations) to {cache_cits_path}")

    # Map current citations by field_path
    cits_map = {c["field_path"]: c for c in citations if c.get("field_path")}

    # Baseline G2 records
    baseline_g2_records = {p: r for p, r in reg.records.items() if r.category == "G2"}
    print(f"[Forensics] Baseline G2 records: {len(baseline_g2_records)}")

    # Classify each baseline G2 record under current predictions
    still_failing_baseline_g2 = {}
    newly_passing_baseline_g2 = {}

    for p, rec in baseline_g2_records.items():
        c = cits_map.get(p)
        if not c or c.get("bbox") is None or c.get("page") != rec.gt_page:
            still_failing_baseline_g2[p] = (rec, c, 0.0)
        else:
            iou = compute_iou_xywh(rec.gt_box, tuple(c["bbox"]))
            if iou >= 0.50:
                newly_passing_baseline_g2[p] = (rec, c, iou)
            else:
                still_failing_baseline_g2[p] = (rec, c, iou)

    print(f"[Forensics] Baseline G2 newly passing: {len(newly_passing_baseline_g2)}")
    print(f"[Forensics] Baseline G2 still failing: {len(still_failing_baseline_g2)}")
    assert len(still_failing_baseline_g2) == 3936, f"Expected 3936 remaining baseline G2 failures, got {len(still_failing_baseline_g2)}"

    # Fields of interest
    target_fields = [
        "name",
        "address_1",
        "address_2",
        "address_3",
        "address_4",
        "city",
        "state",
        "postal_code",
        "country",
    ]

    # Organize data for still-failing G2 records
    records_by_field = defaultdict(list)

    for p, (rec, c, curr_iou) in still_failing_baseline_g2.items():
        fld = rec.field
        if fld not in target_fields:
            continue

        gt_box = rec.gt_box
        pred_box = tuple(c["bbox"]) if (c and c.get("bbox")) else (rec.pred_box if rec.pred_box else None)
        if pred_box is None:
            continue

        gx, gy, gw, gh = gt_box
        px, py, pw, ph = pred_box

        dx = px - gx
        dw = pw - gw
        dy = py - gy
        dh = ph - gh

        cf = simulate_counterfactual_ious(gt_box, pred_box)

        records_by_field[fld].append({
            "path": p,
            "row_idx": rec.row_idx,
            "page": rec.gt_page,
            "is_multi_line_row": rec.is_multi_line_row,
            "gt_box": gt_box,
            "pred_box": pred_box,
            "iou": curr_iou,
            "baseline_iou": rec.iou,
            "dx": dx,
            "dw": dw,
            "dy": dy,
            "dh": dh,
            "cf": cf,
        })

    # Output detailed forensic results
    print("\n" + "=" * 110)
    print(f"{'EXP-005 PART 4 PASS 4: G2 NEAR-MISS FORENSIC REPORT (3,936 FAILURES)':^110}")
    print("=" * 110)

    summary_rows = []

    for fld in target_fields:
        items = records_by_field[fld]
        n_total = len(items)
        if n_total == 0:
            continue

        ious = np.array([it["iou"] for it in items])
        near_miss_mask = (ious >= 0.35) & (ious < 0.50)
        n_near_miss = int(np.sum(near_miss_mask))

        dxs = np.array([it["dx"] for it in items])
        dws = np.array([it["dw"] for it in items])
        dys = np.array([it["dy"] for it in items])
        dhs = np.array([it["dh"] for it in items])

        # Counterfactual recoveries (how many cross 0.50 IoU?)
        rec_w = sum(1 for it in items if it["cf"]["fix_w"] >= 0.50)
        rec_x = sum(1 for it in items if it["cf"]["fix_x"] >= 0.50)
        rec_xw = sum(1 for it in items if it["cf"]["fix_xw"] >= 0.50)
        rec_h = sum(1 for it in items if it["cf"]["fix_h"] >= 0.50)
        rec_y = sum(1 for it in items if it["cf"]["fix_y"] >= 0.50)
        rec_yh = sum(1 for it in items if it["cf"]["fix_yh"] >= 0.50)

        # Clipping vs overextending
        pct_clipping = np.mean(dws < -0.002) * 100
        pct_overext = np.mean(dws > 0.002) * 100

        # Mean and median deltas
        summary_rows.append({
            "field": fld,
            "count": n_total,
            "near_miss_count": n_near_miss,
            "near_miss_pct": n_near_miss / n_total * 100,
            "iou_mean": float(np.mean(ious)),
            "iou_median": float(np.median(ious)),
            "dx_mean": float(np.mean(dxs)),
            "dx_median": float(np.median(dxs)),
            "dw_mean": float(np.mean(dws)),
            "dw_median": float(np.median(dws)),
            "dy_mean": float(np.mean(dys)),
            "dy_median": float(np.median(dys)),
            "dh_mean": float(np.mean(dhs)),
            "dh_median": float(np.median(dhs)),
            "rec_w": rec_w,
            "rec_x": rec_x,
            "rec_xw": rec_xw,
            "rec_h": rec_h,
            "rec_y": rec_y,
            "rec_yh": rec_yh,
            "pct_clipping": pct_clipping,
            "pct_overext": pct_overext,
        })

    # Print Summary Table
    print(f"\n{'Field':<14} | {'G2 Count':<8} | {'0.35-0.49':<9} | {'Mean IoU':<8} | {'Med IoU':<8} | {'Med dx':<10} | {'Med dw':<10} | {'Med dy':<10} | {'Med dh':<10}")
    print("-" * 100)
    for r in summary_rows:
        print(f"{r['field']:<14} | {r['count']:<8} | {r['near_miss_count']:<9} | {r['iou_mean']:<8.4f} | {r['iou_median']:<8.4f} | {r['dx_median']:<+10.5f} | {r['dw_median']:<+10.5f} | {r['dy_median']:<+10.5f} | {r['dh_median']:<+10.5f}")

    print("\n" + "=" * 110)
    print(f"{'COUNTERFACTUAL RECOVERY MATRIX (Citations crossing >= 0.50 IoU if error axis eliminated)':^110}")
    print("=" * 110)
    print(f"{'Field':<14} | {'G2 Count':<8} | {'Fix W Only':<12} | {'Fix X Only':<12} | {'Fix (X+W)':<12} | {'Fix H Only':<12} | {'Fix Y Only':<12} | {'Fix (Y+H)':<12}")
    print("-" * 100)
    for r in summary_rows:
        print(f"{r['field']:<14} | {r['count']:<8} | {r['rec_w']:<5} ({r['rec_w']/r['count']*100:4.1f}%) | {r['rec_x']:<5} ({r['rec_x']/r['count']*100:4.1f}%) | {r['rec_xw']:<5} ({r['rec_xw']/r['count']*100:4.1f}%) | {r['rec_h']:<5} ({r['rec_h']/r['count']*100:4.1f}%) | {r['rec_y']:<5} ({r['rec_y']/r['count']*100:4.1f}%) | {r['rec_yh']:<5} ({r['rec_yh']/r['count']*100:4.1f}%)")

    # Save detailed JSON summary
    out_json_path = repo_root / "research/EXP-005-g2-forensics-summary.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n[Forensics] Detailed JSON saved to {out_json_path}")


if __name__ == "__main__":
    main()
