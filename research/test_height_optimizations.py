#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
if _REF.exists(): sys.path.insert(0, str(_REF))
sys.path.insert(0, str(repo_root / "src"))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from tonerhound.benchmark.correlation import FailureRegistry

def main():
    reg = FailureRegistry.load(repo_root / "experiments/EXP-005-ftx-failure-registry.json")
    print(f"Loaded registry: {len(reg.records)} records, {len(reg.passing_paths)} passing baseline, {len(reg.failing_paths)} failing baseline.")
    print(f"Baseline G2 count: {reg.category_counts['G2']}")

    # Baseline records dictionary
    # For each record: path, status, category, gt_page, gt_box, pred_page, pred_box, field, is_multi_line_row
    records = list(reg.records.values())

    def evaluate_height_transform(h_transform_fn):
        """
        h_transform_fn(field, pred_box, is_multi_line, gt_box) -> new_pred_box
        Returns:
            newly_passing: list of paths
            regressions: list of paths
            g2_fixed: int
            net_gain: int
        """
        newly_passing = []
        regressions = []
        fixed_by_cat = defaultdict(int)
        reg_by_cat = defaultdict(int)

        for r in records:
            if not r.pred_box:
                continue
            
            # Apply transform
            new_box = h_transform_fn(r.field, r.pred_box, r.is_multi_line_row)
            if new_box is None:
                new_box = r.pred_box

            new_iou = float(iou_xywh(r.gt_box, new_box))
            passes_now = (new_iou >= 0.50)

            if passes_now:
                if r.status == "FAIL":
                    newly_passing.append(r.path)
                    fixed_by_cat[r.category] += 1
            else:
                if r.status == "PASS":
                    regressions.append(r.path)
                    reg_by_cat[r.category] += 1

        net = len(newly_passing) - len(regressions)
        return {
            "newly_passing": len(newly_passing),
            "regressions": len(regressions),
            "net_gain": net,
            "g2_fixed": fixed_by_cat["G2"],
            "fixed_by_cat": dict(fixed_by_cat),
            "reg_by_cat": dict(reg_by_cat),
        }

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: UNIFORM SINGLE-LINE HEIGHT SWEEP (KEEPING CENTER FIXED)")
    print("=" * 80)
    # If cell is single line (ph < 0.015), change ph to target_h, preserving yc:
    # yc = py + ph / 2.0 -> py_new = yc - target_h / 2.0 = py + (ph - target_h) / 2.0
    for target_h in [0.0085, 0.0088, 0.0090, 0.0092, 0.0093, 0.0094, 0.0095, 0.0096, 0.0097, 0.0098, 0.0100, 0.0102]:
        def transform(field, pred_box, is_ml):
            px, py, pw, ph = pred_box
            if ph < 0.015:
                yc = py + ph / 2.0
                new_y = yc - target_h / 2.0
                return (px, new_y, pw, target_h)
            return pred_box

        res = evaluate_height_transform(transform)
        print(f"Single-line h={target_h:.4f}: Net={res['net_gain']:+4d} | "
              f"New={res['newly_passing']:4d} (G2={res['g2_fixed']:4d}) | Reg={res['regressions']:4d}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: PER-FIELD SINGLE-LINE HEIGHT SWEEP")
    print("=" * 80)
    # Test each field individually to find its optimal height
    fields_to_test = ["name", "address_1", "address_2", "address_3", "address_4", "city", "state", "postal_code", "country"]
    optimal_per_field_h = {}

    for fld in fields_to_test:
        print(f"\n--- Field: {fld} ---")
        best_net = -9999
        best_h = None
        best_reg = 9999
        for test_h in np.linspace(0.0085, 0.0105, 21):
            test_h = round(float(test_h), 5)
            def transform(field, pred_box, is_ml, target_fld=fld, target_h=test_h):
                if field == target_fld:
                    px, py, pw, ph = pred_box
                    if ph < 0.015:
                        yc = py + ph / 2.0
                        new_y = yc - target_h / 2.0
                        return (px, new_y, pw, target_h)
                return pred_box

            res = evaluate_height_transform(transform)
            if res['newly_passing'] > 0 or res['regressions'] > 0:
                print(f"  h={test_h:.5f}: Net={res['net_gain']:+3d} | New={res['newly_passing']:3d} (G2={res['g2_fixed']:3d}) | Reg={res['regressions']:3d}")
                if res['regressions'] == 0 and res['net_gain'] > best_net:
                    best_net = res['net_gain']
                    best_h = test_h
                    best_reg = 0
                elif best_h is None and res['net_gain'] > best_net:
                    best_net = res['net_gain']
                    best_h = test_h
                    best_reg = res['regressions']

        print(f"Optimal for {fld}: h={best_h} (Net={best_net:+d}, Reg={best_reg})")
        optimal_per_field_h[fld] = (best_h, best_net, best_reg)

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: MULTI-LINE HEIGHT SWEEP (ph >= 0.015)")
    print("=" * 80)
    # For multi-line cells, anchor is top (y stays fixed): new_h tested
    for ml_h in [0.0180, 0.0190, 0.0195, 0.0200, 0.0202, 0.0204, 0.0205, 0.0206, 0.0208, 0.0210, 0.0215, 0.0220]:
        def transform(field, pred_box, is_ml, target_h=ml_h):
            px, py, pw, ph = pred_box
            if ph >= 0.015:
                # anchor is top
                return (px, py, pw, target_h)
            return pred_box

        res = evaluate_height_transform(transform)
        print(f"Multi-line h={ml_h:.4f} (fixed top): Net={res['net_gain']:+3d} | "
              f"New={res['newly_passing']:3d} (G2={res['g2_fixed']:3d}) | Reg={res['regressions']:3d}")

    # Also test multi-line with fixed center:
    print("\nMulti-line with fixed center:")
    for ml_h in [0.0180, 0.0190, 0.0195, 0.0200, 0.0202, 0.0204, 0.0205, 0.0206, 0.0208, 0.0210]:
        def transform(field, pred_box, is_ml, target_h=ml_h):
            px, py, pw, ph = pred_box
            if ph >= 0.015:
                yc = py + ph / 2.0
                return (px, yc - target_h / 2.0, pw, target_h)
            return pred_box

        res = evaluate_height_transform(transform)
        print(f"Multi-line h={ml_h:.4f} (fixed center): Net={res['net_gain']:+3d} | "
              f"New={res['newly_passing']:3d} (G2={res['g2_fixed']:3d}) | Reg={res['regressions']:3d}")

if __name__ == "__main__":
    main()
