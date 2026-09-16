#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

def main():
    reg_path = Path("experiments/EXP-005-ftx-failure-registry.json")
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    g2_records = [r for r in data.values() if r["category"] == "G2"]
    print(f"Total G2 records in baseline: {len(g2_records)}")

    # Breakdown of G2 by field
    fld_counts = defaultdict(int)
    for r in g2_records:
        fld_counts[r.get("field", "")] += 1
    print("\n--- G2 Failures by Field ---")
    for fld, cnt in sorted(fld_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fld:15s}: {cnt:5d} ({cnt / len(g2_records) * 100:.2f}%)")

    # IoU distribution of G2
    ious = [r["iou"] for r in g2_records]
    print("\n--- G2 IoU Distribution ---")
    print(f"Mean IoU   : {np.mean(ious):.4f}")
    print(f"Median IoU : {np.median(ious):.4f}")
    print(f"Min IoU    : {np.min(ious):.4f}")
    print(f"Max IoU    : {np.max(ious):.4f}")

    # Buckets of IoU
    buckets = [
        (0.48, 0.50),
        (0.45, 0.48),
        (0.40, 0.45),
        (0.35, 0.40),
        (0.30, 0.35),
    ]
    for low, high in buckets:
        cnt = sum(1 for i in ious if low <= i < high)
        print(f"  IoU [{low:.2f}, {high:.2f}): {cnt:5d} ({cnt / len(g2_records) * 100:.2f}%)")

    # Analyze dimension differences:
    # dx = pred_x - gt_x
    # dy = pred_y - gt_y
    # dw = pred_w - gt_w
    # dh = pred_h - gt_h
    # for each field in G2:
    print("\n--- Dimension Differences in G2 Failures (pred - gt) ---")
    fld_diffs = defaultdict(lambda: defaultdict(list))
    for r in g2_records:
        fld = r.get("field", "")
        pb = r.get("pred_box")
        gb = r.get("gt_box")
        if pb and gb:
            dx = pb[0] - gb[0]
            dy = pb[1] - gb[1]
            dw = pb[2] - gb[2]
            dh = pb[3] - gb[3]
            fld_diffs[fld]["dx"].append(dx)
            fld_diffs[fld]["dy"].append(dy)
            fld_diffs[fld]["dw"].append(dw)
            fld_diffs[fld]["dh"].append(dh)
            fld_diffs[fld]["gt_h"].append(gb[3])
            fld_diffs[fld]["pred_h"].append(pb[3])
            fld_diffs[fld]["iou"].append(r["iou"])

    for fld in sorted(fld_diffs.keys()):
        d = fld_diffs[fld]
        print(f"Field {fld:15s} (N={len(d['iou']):4d}, mean IoU={np.mean(d['iou']):.3f}):")
        print(f"  dh (pred_h - gt_h): mean={np.mean(d['dh']):+.6f}, median={np.median(d['dh']):+.6f} | gt_h mean={np.mean(d['gt_h']):.6f}, pred_h mean={np.mean(d['pred_h']):.6f}")
        print(f"  dy (pred_y - gt_y): mean={np.mean(d['dy']):+.6f}, median={np.median(d['dy']):+.6f}")
        print(f"  dw (pred_w - gt_w): mean={np.mean(d['dw']):+.6f}, median={np.median(d['dw']):+.6f}")
        print(f"  dx (pred_x - gt_x): mean={np.mean(d['dx']):+.6f}, median={np.median(d['dx']):+.6f}")

if __name__ == "__main__":
    main()
