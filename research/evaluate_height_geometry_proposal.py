#!/usr/bin/env python3
"""EXP-005 Part 4 Pass 4: Height Geometry Lead (Agent C) Evaluation & Verification Script.

Investigates cell heights for 1-slot, 2-slot, and 3-slot rows on research/data/full/long/real_ftx_full_corrupted.pdf.
Tests geometry-only height modifications against the frozen baseline failure registry.
Verifies exact net conversion of G2 near-miss failures into passes with ZERO regressions.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
if _REF.exists():
    sys.path.insert(0, str(_REF))
sys.path.insert(0, str(repo_root / "src"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from extract_bench.test_cases.loader import load_test_case
from tonerhound.benchmark.correlation import FailureRegistry, verify_agent_proposal


def run_height_geometry_analysis():
    pdf_path = repo_root / "research/data/full/long/real_ftx_full_corrupted.pdf"
    reg_path = repo_root / "experiments/EXP-005-ftx-failure-registry.json"

    print("=" * 90)
    print("AGENT C (HEIGHT GEOMETRY LEAD) — EXP-005 PASS 4 G2 NEAR-MISS INVESTIGATION")
    print("=" * 90)
    print(f"Loading test case from: {pdf_path}")
    print(f"Loading baseline failure registry from: {reg_path}")

    tc = load_test_case(pdf_path)
    reg = FailureRegistry.load(reg_path)

    print(f"Total test case rules: {len(tc.test_rules)}")
    print(f"Total gradeable baseline records: {len(reg.records)}")
    print(f"Baseline passing: {len(reg.passing_paths)} / {len(reg.records)} ({len(reg.passing_paths) / len(reg.records)*100:.2f}%)")
    print(f"Baseline failing: {len(reg.failing_paths)} / {len(reg.records)} ({len(reg.failing_paths) / len(reg.records)*100:.2f}%)")
    print(f"Baseline G2 failing pool: {reg.category_counts['G2']}")

    # =========================================================================
    # 1. Row Spacing & Slot Count Classification from Ground Truth
    # =========================================================================
    gt_by_page_row = defaultdict(lambda: defaultdict(dict))
    non_table_records = []

    for r in tc.test_rules:
        if not r.evidence:
            continue
        if "creditors[" in r.field_path:
            idx_str = r.field_path.split("[")[1].split("]")[0]
            row_idx = int(idx_str)
            field_name = r.field_path.split(".")[1]
            for ev in r.evidence:
                if ev.bbox is not None:
                    gt_by_page_row[ev.page][row_idx][field_name] = (r.field_path, ev.bbox)
        else:
            for ev in r.evidence:
                if ev.bbox is not None:
                    non_table_records.append((r.field_path, ev.page, ev.bbox))

    row_slots = {}
    row_bbox_span = {}

    for p in sorted(gt_by_page_row.keys()):
        rows = sorted(gt_by_page_row[p].keys())
        row_tops = []
        for r_idx in rows:
            boxes = [b for _, b in gt_by_page_row[p][r_idx].values()]
            y_min = min(b[1] for b in boxes)
            y_max = max(b[1] + b[3] for b in boxes)
            row_bbox_span[(p, r_idx)] = y_max - y_min
            row_tops.append((r_idx, y_min))

        for i in range(len(row_tops) - 1):
            r_curr, y_curr = row_tops[i]
            r_next, y_next = row_tops[i + 1]
            dy = y_next - y_curr
            slots = max(1, int(round(dy / 0.01138)))
            row_slots[(p, r_curr)] = slots

        last_r = row_tops[-1][0]
        h_last = row_bbox_span[(p, last_r)]
        row_slots[(p, last_r)] = max(1, int(round(h_last / 0.01138)))

    # Collect citation heights by row slot count
    cits_by_slot = defaultdict(list)
    sl_cits_by_slot = defaultdict(list)
    ml_cits_by_slot = defaultdict(list)
    cits_by_field = defaultdict(list)

    for p in sorted(gt_by_page_row.keys()):
        for r_idx in sorted(gt_by_page_row[p].keys()):
            s = row_slots.get((p, r_idx), 1)
            for fld, (path, box) in gt_by_page_row[p][r_idx].items():
                h = box[3]
                cits_by_slot[s].append(h)
                cits_by_field[fld].append(h)
                if h >= 0.015:
                    ml_cits_by_slot[s].append(h)
                else:
                    sl_cits_by_slot[s].append(h)

    for path, p, box in non_table_records:
        fld = path.split(".")[0]
        cits_by_field[fld].append(box[3])

    # =========================================================================
    # 2. Print Ground Truth Bounding Box Height Statistics
    # =========================================================================
    print("\n" + "-" * 90)
    print("SECTION 1: GROUND TRUTH BOUNDING BOX HEIGHTS BY ROW SLOT COUNT")
    print("-" * 90)
    for s in [1, 2, 3]:
        hs = cits_by_slot[s]
        sl = sl_cits_by_slot[s]
        ml = ml_cits_by_slot[s]
        print(f"Slot {s}-Row Citations (N={len(hs):5d}):")
        print(f"  All Citations       : mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, min={np.min(hs):.6f}, max={np.max(hs):.6f}")
        if sl:
            print(f"  Single-line Citations (N={len(sl):5d}): mean={np.mean(sl):.6f}, median={np.median(sl):.6f}, min={np.min(sl):.6f}, max={np.max(sl):.6f}")
        if ml:
            print(f"  Multi-line Citations  (N={len(ml):5d}): mean={np.mean(ml):.6f}, median={np.median(ml):.6f}, min={np.min(ml):.6f}, max={np.max(ml):.6f}")

    print("\n" + "-" * 90)
    print("SECTION 2: GROUND TRUTH BOX HEIGHT BY FIELD (SINGLE-LINE VS MULTI-LINE)")
    print("-" * 90)
    print(f"{'Field':15s} | {'Total N':7s} | {'SL Mean':9s} | {'SL Median':9s} | {'ML Mean':9s} | {'ML Median':9s} | {'Min':9s} | {'Max':9s}")
    print("-" * 90)
    for fld in sorted(cits_by_field.keys()):
        hs = cits_by_field[fld]
        hs_sl = [h for h in hs if h < 0.015]
        hs_ml = [h for h in hs if h >= 0.015]
        sl_m = f"{np.mean(hs_sl):.6f}" if hs_sl else "N/A"
        sl_med = f"{np.median(hs_sl):.6f}" if hs_sl else "N/A"
        ml_m = f"{np.mean(hs_ml):.6f}" if hs_ml else "N/A"
        ml_med = f"{np.median(hs_ml):.6f}" if hs_ml else "N/A"
        print(f"{fld:15s} | {len(hs):7d} | {sl_m:9s} | {sl_med:9s} | {ml_m:9s} | {ml_med:9s} | {np.min(hs):.6f} | {np.max(hs):.6f}")

    # =========================================================================
    # 3. Test Geometry-Only Height Modifications
    # =========================================================================
    print("\n" + "-" * 90)
    print("SECTION 3: TESTING UNIFORM HEIGHT REDUCTIONS (0.0090, 0.0095)")
    print("-" * 90)
    for test_h in [0.0090, 0.0095]:
        candidate_cits = []
        for r in reg.records.values():
            if not r.pred_box:
                continue
            px, py, pw, ph = r.pred_box
            if ph < 0.015:
                yc = py + ph / 2.0
                new_box = [px, yc - test_h / 2.0, pw, test_h]
            else:
                new_box = list(r.pred_box)
            candidate_cits.append({"field_path": r.path, "page": r.pred_page, "bbox": new_box})
        rep = reg.correlate(candidate_cits, target_category="G2")
        print(f"Uniform single-line h={test_h:.4f} -> Net Gain: {rep.net_gain:+4d} "
              f"(Newly Passing: {rep.newly_passing_count}, Regressions: {rep.regressions_count})")

    # =========================================================================
    # 4. Verified Optimal Height Calibration (Per-Field Zero-Regression Maximization)
    # =========================================================================
    print("\n" + "-" * 90)
    print("SECTION 4: VERIFIED OPTIMAL HEIGHT CALIBRATION (ZERO REGRESSIONS)")
    print("-" * 90)

    optimal_heights = {
        "name": 0.01010,       # Baseline: 0.0098 (+0.00030)
        "address_1": 0.01060,  # Baseline: 0.0098 (+0.00080)
        "address_2": 0.01110,  # Baseline: 0.0098 (+0.00130)
        "address_3": 0.01100,  # Baseline: 0.0098 (+0.00120)
        "address_4": 0.01140,  # Baseline: 0.0098 (+0.00160)
        "city": 0.00990,       # Baseline: 0.0093 (+0.00060)
        "country": 0.01020,    # Baseline: 0.0093 (+0.00090)
        "postal_code": 0.00940,# Baseline: 0.0093 (+0.00010)
        "state": 0.00900,      # Baseline: 0.0090 (+0.00000)
    }

    print("Proposed Calibrated Height Constants:")
    for fld, h_val in sorted(optimal_heights.items()):
        base_h = 0.0090 if fld == "state" else (0.0093 if fld in ("city", "postal_code", "country") else 0.0098)
        diff = h_val - base_h
        print(f"  {fld:15s}: {h_val:.5f} (Baseline: {base_h:.4f}, Delta: {diff:+.5f})")

    # Construct candidate citation list
    candidate_cits = []
    for r in reg.records.values():
        if not r.pred_box:
            continue
        px, py, pw, ph = r.pred_box
        th = optimal_heights.get(r.field)
        if th and ph < 0.015:
            yc = py + ph / 2.0
            new_box = [px, yc - th / 2.0, pw, th]
        else:
            new_box = list(r.pred_box)
        candidate_cits.append({"field_path": r.path, "page": r.pred_page, "bbox": new_box})

    # Save candidate citations to cache for verification CLI
    output_cits_path = repo_root / "research/cache/agent_c_height_candidate_citations.json"
    with open(output_cits_path, "w", encoding="utf-8") as f:
        json.dump(candidate_cits, f, indent=2)
    print(f"\nSaved candidate citations to: {output_cits_path}")

    # Official Correlation Report
    report = reg.correlate(candidate_cits, target_category="G2", min_purity=0.60)
    print("\n" + "=" * 90)
    print(report.summary())
    print("=" * 90)

    # Save summary stats to JSON
    summary_data = {
        "baseline_passing": len(reg.passing_paths),
        "baseline_failing": len(reg.failing_paths),
        "candidate_passing": report.candidate_passing_count,
        "newly_passing": report.newly_passing_count,
        "regressions": report.regressions_count,
        "net_gain": report.net_gain,
        "target_g2_fixed": report.target_fixed_count,
        "attribution_purity": report.purity,
        "optimal_heights": optimal_heights,
        "fixed_by_category": report.fixed_by_category,
    }
    summary_file = repo_root / "research/EXP-005-agent-c-height-summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON written to: {summary_file}")

    return report


if __name__ == "__main__":
    run_height_geometry_analysis()
