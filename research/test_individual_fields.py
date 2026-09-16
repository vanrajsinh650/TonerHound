#!/usr/bin/env python3
"""EXP-005 Part 4 Pass 4: Per-Field Column Adjustment Exploration.

Tests data-derived width and horizontal padding adjustments for each column
individually on real_ftx_full_corrupted.pdf using the failure registry.
"""

import sys
import json
from pathlib import Path
from collections import Counter
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
if _REF.exists():
    sys.path.insert(0, str(_REF))
sys.path.insert(0, str(repo_root / "src"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from extract_bench.test_cases.loader import load_test_case

def load_data():
    pdf_path = repo_root / "research/data/full/long/real_ftx_full_corrupted.pdf"
    tc = load_test_case(pdf_path)

    reg_path = repo_root / "experiments/EXP-005-ftx-failure-registry.json"
    with open(reg_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    creditors = tc.expected_output.get("creditors", [])
    return tc, registry, creditors

def test_field_adjustments(target_fld, registry, creditors):
    items = [v for v in registry.values() if v.get("field") == target_fld and v.get("pred_box")]
    baseline_pass = sum(1 for v in items if v["status"] == "PASS")
    baseline_g2 = sum(1 for v in items if v["category"] == "G2")
    print(f"\n{'='*80}")
    print(f"FIELD: {target_fld} (Total: {len(items)}, Baseline Passing: {baseline_pass}, G2 Failures: {baseline_g2})")
    print(f"{'='*80}")

    # Extract text lengths and widths
    data = []
    for v in items:
        row_idx = v.get("row_idx")
        val = creditors[row_idx].get(target_fld) if (row_idx is not None and row_idx < len(creditors)) else None
        txt = str(val).strip() if val is not None else ""
        L = len(txt)
        data.append({
            "rec": v,
            "L": L,
            "gt_box": v["gt_box"],
            "pred_box": v["pred_box"],
            "status": v["status"],
            "category": v["category"],
            "iou": v["iou"],
        })

    # Let's test a parameter grid for this field:
    # Model:
    # w = min(max_w, max(min_w, L * char_w + pad_w))
    # x = px + dx (or left margin shift)
    # Also evaluate width-only scaling: new_w = pw * scale_w (or pw + delta_w)
    # Also evaluate char_w tuning

    # First: evaluate delta_w (padding added/subtracted to current predicted width)
    print("--- Option A: Additive Width Padding (delta_w) & Shift (delta_x) ---")
    best_opt_a = None
    best_net_a = -999

    for dx in np.linspace(-0.0020, 0.0020, 41):
        for dw in np.linspace(-0.0050, 0.0080, 53):
            fixed = 0
            g2_fixed = 0
            reg = 0
            for d in data:
                px, py, pw, ph = d["pred_box"]
                new_w = max(0.005, pw + dw)
                new_x = px + dx
                new_box = [new_x, py, new_w, ph]
                new_iou = iou_xywh(d["gt_box"], new_box)
                was_pass = (d["status"] == "PASS")
                is_pass = (new_iou >= 0.50)

                if is_pass and not was_pass:
                    fixed += 1
                    if d["category"] == "G2":
                        g2_fixed += 1
                elif was_pass and not is_pass:
                    reg += 1

            net = fixed - reg
            if reg == 0 and net > best_net_a:
                best_net_a = net
                best_opt_a = (dx, dw, fixed, reg, g2_fixed, net)

    if best_opt_a and best_opt_a[5] > 0:
        dx, dw, fixed, reg, g2_fixed, net = best_opt_a
        print(f"  Best Zero-Regression: dx={dx:+.5f}, dw={dw:+.5f} -> +{net} net gain ({g2_fixed} G2 fixed, {reg} regressed)")
    else:
        print("  No positive net gain with zero regressions found in delta_w / delta_x grid.")

    # Also search with small tolerance for regressions or check Pareto frontier
    print("  Pareto frontier (allowing up to 5 regressions):")
    best_p = []
    for max_reg in [0, 1, 2, 5, 10]:
        best_net = -999
        best_tuple = None
        for dx in np.linspace(-0.0020, 0.0020, 21):
            for dw in np.linspace(-0.0050, 0.0080, 27):
                fixed = 0
                g2_fixed = 0
                reg = 0
                for d in data:
                    px, py, pw, ph = d["pred_box"]
                    new_w = max(0.005, pw + dw)
                    new_x = px + dx
                    new_box = [new_x, py, new_w, ph]
                    new_iou = iou_xywh(d["gt_box"], new_box)
                    was_pass = (d["status"] == "PASS")
                    is_pass = (new_iou >= 0.50)
                    if is_pass and not was_pass:
                        fixed += 1
                        if d["category"] == "G2":
                            g2_fixed += 1
                    elif was_pass and not is_pass:
                        reg += 1
                net = fixed - reg
                if reg <= max_reg and net > best_net:
                    best_net = net
                    best_tuple = (dx, dw, fixed, reg, g2_fixed, net)
        if best_tuple:
            dx, dw, fixed, reg, g2_fixed, net = best_tuple
            print(f"    Max Reg <= {max_reg:2d}: dx={dx:+.5f}, dw={dw:+.5f} -> Fixed={fixed:3d} (G2={g2_fixed:3d}), Reg={reg:2d}, Net={net:+3d}")

    # Option B: Character width constant (char_w) and padding (pad_w)
    print("\n--- Option B: Character Width Model (char_w * L + pad_w) ---")
    best_opt_b = None
    best_net_b = -999

    for char_w in np.linspace(0.0024, 0.0040, 33):
        for pad_w in np.linspace(-0.003, 0.006, 19):
            for dx in [0.0, 0.0005, -0.0005]:
                fixed = 0
                g2_fixed = 0
                reg = 0
                for d in data:
                    px, py, pw, ph = d["pred_box"]
                    calc_w = max(0.006, d["L"] * char_w + pad_w)
                    new_x = px + dx
                    new_box = [new_x, py, calc_w, ph]
                    new_iou = iou_xywh(d["gt_box"], new_box)
                    was_pass = (d["status"] == "PASS")
                    is_pass = (new_iou >= 0.50)
                    if is_pass and not was_pass:
                        fixed += 1
                        if d["category"] == "G2":
                            g2_fixed += 1
                    elif was_pass and not is_pass:
                        reg += 1
                net = fixed - reg
                if reg == 0 and net > best_net_b:
                    best_net_b = net
                    best_opt_b = (char_w, pad_w, dx, fixed, reg, g2_fixed, net)

    if best_opt_b and best_opt_b[6] > 0:
        char_w, pad_w, dx, fixed, reg, g2_fixed, net = best_opt_b
        print(f"  Best Zero-Regression: char_w={char_w:.6f}, pad_w={pad_w:+.5f}, dx={dx:+.5f} -> +{net} net gain ({g2_fixed} G2 fixed, {reg} regressed)")
    else:
        print("  No positive net gain with zero regressions found in char_w grid.")

def main():
    tc, registry, creditors = load_data()
    fields = ["city", "postal_code", "country", "address_4", "address_3", "address_2", "address_1", "name", "state"]
    for fld in fields:
        test_field_adjustments(fld, registry, creditors)

if __name__ == "__main__":
    main()
