#!/usr/bin/env python3
"""EXP-005 Part 4 Pass 4: Simulation of Recommended Empirical Parameter Adjustments for G2.

Evaluates how many of the 3,936 G2 failures are recovered by applying
the recommended column parameters (without touching src/tonerhound/benchmark/adapter.py).
"""

import json
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
import sys
if _REF.exists(): sys.path.insert(0, str(_REF))
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh

def main():
    reg_path = repo_root / "experiments/EXP-005-ftx-failure-registry.json"
    cits_path = repo_root / "research/cache/current_adapter_ftx_citations.json"

    with open(reg_path) as f:
        reg_data = json.load(f)
    with open(cits_path) as f:
        cits = json.load(f)

    cits_map = {c["field_path"]: c for c in cits if c.get("field_path")}

    # Recommended parameter adjustments:
    # 1. state: cell_x = 0.7337 (was 0.7325), cell_w = 0.0078 (was 0.0105), cell_h = 0.00895 (was 0.0090)
    # 2. city: char_w = 0.00365 (was 0.00325), min_w = 0.0180 (was 0.0150)
    # 3. postal_code: char_w = 0.00350 (was 0.00325), min_w = 0.0175
    # 4. country: char_w = 0.00360 (was 0.00325), min_w = 0.0160
    # 5. address_3: max_w = 0.0580 (was 0.0680)
    # 6. address_4: max_w = 0.0400 (was 0.0480)

    # Let's test single-parameter adjustment vs combined horizontal adjustment
    target_failing = {}
    for p, rec in reg_data.items():
        if rec["category"] == "G2":
            c = cits_map.get(p)
            if not c or not c.get("bbox"):
                continue
            iou = iou_xywh(rec["gt_box"], tuple(c["bbox"]))
            if iou < 0.50:
                target_failing[p] = (rec, c, iou)

    print(f"Total baseline G2 still failing: {len(target_failing)}")

    # Evaluate column adjustments
    recovered_by_col = {}
    recovered_overall = 0

    for fld in ["name", "address_1", "address_2", "address_3", "address_4", "city", "state", "postal_code", "country"]:
        col_recs = [v for v in target_failing.values() if v[0]["field"] == fld]
        if not col_recs:
            continue

        n_col = len(col_recs)
        raw_passes = sum(1 for rec, c, iou in col_recs if iou >= 0.50)
        
        # Test simulated horizontal parameter adjustments
        sim_passes = 0
        ious_before = []
        ious_after = []

        for rec, c, orig_iou in col_recs:
            gx, gy, gw, gh = rec["gt_box"]
            px, py, pw, ph = c["bbox"]
            ref_txt = str(c.get("reference_text", ""))
            L = len(ref_txt)

            new_px = px
            new_pw = pw
            new_py = py
            new_ph = ph

            if fld == "state":
                # Shift X right by +0.0012, narrow width to 0.0078
                new_px = px + 0.0012
                new_pw = 0.0078
                new_ph = 0.00895
            elif fld == "city":
                # char_w from 0.00325 to 0.00365
                text_w = max(0.0180, L * 0.00365)
                new_pw = min(0.0850, text_w)
                new_px = px - (new_pw - pw) / 2.0  # keep center or align left
                # actually city is left-aligned at col_x = 0.6427
                new_px = 0.6427
            elif fld == "postal_code":
                text_w = max(0.0175, L * 0.00350)
                new_pw = min(0.0350, text_w)
                new_px = 0.7970
            elif fld == "country":
                text_w = max(0.0160, L * 0.00360)
                new_pw = min(0.0500, text_w)
                new_px = 0.8513
            elif fld == "address_3":
                if pw > 0.0580:
                    new_pw = 0.0580
            elif fld == "address_4":
                if pw > 0.0400:
                    new_pw = 0.0400

            new_box = (new_px, new_py, new_pw, new_ph)
            new_iou = iou_xywh(rec["gt_box"], new_box)
            ious_before.append(orig_iou)
            ious_after.append(new_iou)
            if new_iou >= 0.50:
                sim_passes += 1

        recovered_by_col[fld] = {
            "total": n_col,
            "raw_passes": raw_passes,
            "sim_passes": sim_passes,
            "net_gain": sim_passes - raw_passes,
            "mean_iou_before": float(np.mean(ious_before)),
            "mean_iou_after": float(np.mean(ious_after)),
        }
        recovered_overall += (sim_passes - raw_passes)

    print("\n--- SIMULATED PARAMETER ADJUSTMENT IMPACT (HORIZONTAL ONLY) ---")
    print(f"{'Field':<14} | {'Failing':<8} | {'Recovered':<10} | {'Recov %':<8} | {'Mean IoU Pre':<12} | {'Mean IoU Post':<12}")
    print("-" * 80)
    for fld, res in recovered_by_col.items():
        pct = res['sim_passes'] / res['total'] * 100
        print(f"{fld:<14} | {res['total']:<8} | {res['sim_passes']:<10} | {pct:5.1f}%   | {res['mean_iou_before']:<12.4f} | {res['mean_iou_after']:<12.4f}")
    print(f"\nTotal G2 Citations Recovered by Pure Horizontal Calibration: +{recovered_overall} citations")

if __name__ == "__main__":
    main()
