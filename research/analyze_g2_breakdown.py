#!/usr/bin/env python3
"""EXP-005 Part 4 Pass 4: In-depth G2 Failure Decomposition.

Analyzes the exact spatial discrepancies (dx, dw, dy, dh, IoU) for all 6,056 G2 citations.
"""

import json
from pathlib import Path
import numpy as np

def main():
    repo_root = Path(__file__).resolve().parent.parent
    registry_path = repo_root / "experiments/EXP-005-ftx-failure-registry.json"

    with open(registry_path) as f:
        registry = json.load(f)

    fields = ["name", "address_1", "address_2", "address_3", "address_4", "city", "state", "postal_code", "country"]

    print("=" * 115)
    print("G2 CITATION GEOMETRIC DISCREPANCY ANALYSIS (6,056 CITATIONS)")
    print("=" * 115)
    header = f"{'Field':12s} | {'Count':5s} | {'IoU Med':7s} | {'dx Med':8s} | {'dw Med':8s} | {'dy Med':8s} | {'dh Med':8s} | {'gt_w Med':8s} | {'pr_w Med':8s} | {'IoU>=0.45':9s}"
    print(header)
    print("-" * 115)

    for fld in fields:
        items = [v for v in registry.values() if v["category"] == "G2" and v["field"] == fld]
        if not items:
            continue
        ious = np.array([v["iou"] for v in items])
        dxs = np.array([v["pred_box"][0] - v["gt_box"][0] for v in items])
        dws = np.array([v["pred_box"][2] - v["gt_box"][2] for v in items])
        dys = np.array([v["pred_box"][1] - v["gt_box"][1] for v in items])
        dhs = np.array([v["pred_box"][3] - v["gt_box"][3] for v in items])
        gt_ws = np.array([v["gt_box"][2] for v in items])
        pr_ws = np.array([v["pred_box"][2] for v in items])

        frac_near_miss = np.mean(ious >= 0.45) * 100
        frac_40_45 = np.mean((ious >= 0.40) & (ious < 0.45)) * 100

        print(
            f"{fld:12s} | {len(items):5d} | {np.median(ious):.4f}  | {np.median(dxs):+.5f} | "
            f"{np.median(dws):+.5f} | {np.median(dys):+.5f} | {np.median(dhs):+.5f} | "
            f"{np.median(gt_ws):.5f}  | {np.median(pr_ws):.5f}  | {frac_near_miss:5.1f}%"
        )

    print("\nIoU Distribution of G2 citations:")
    all_g2_ious = np.array([v["iou"] for v in registry.values() if v["category"] == "G2"])
    print(f"  [0.45, 0.50): {np.sum((all_g2_ious >= 0.45) & (all_g2_ious < 0.50))} ({np.mean((all_g2_ious >= 0.45) & (all_g2_ious < 0.50))*100:.1f}%)")
    print(f"  [0.40, 0.45): {np.sum((all_g2_ious >= 0.40) & (all_g2_ious < 0.45))} ({np.mean((all_g2_ious >= 0.40) & (all_g2_ious < 0.45))*100:.1f}%)")
    print(f"  [0.35, 0.40): {np.sum((all_g2_ious >= 0.35) & (all_g2_ious < 0.40))} ({np.mean((all_g2_ious >= 0.35) & (all_g2_ious < 0.40))*100:.1f}%)")
    print(f"  [0.30, 0.35): {np.sum((all_g2_ious >= 0.30) & (all_g2_ious < 0.35))} ({np.mean((all_g2_ious >= 0.30) & (all_g2_ious < 0.35))*100:.1f}%)")

if __name__ == "__main__":
    main()
