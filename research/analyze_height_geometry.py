#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
    reg_path = Path("experiments/EXP-005-ftx-failure-registry.json")
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total records in registry: {len(data)}")
    
    # Inspect a few sample records
    sample_keys = list(data.keys())[:3]
    for k in sample_keys:
        print(f"Sample key {k}: {data[k]}")

    # Collect stats across all gradeable citations
    all_gt_h = []
    fld_gt_h = defaultdict(list)
    cat_gt_h = defaultdict(list)
    multi_line_gt_h = []
    single_line_gt_h = []

    for path, rec in data.items():
        gt_box = rec["gt_box"]
        gt_h = gt_box[3]
        all_gt_h.append(gt_h)
        fld = rec.get("field", "")
        fld_gt_h[fld].append(gt_h)
        cat_gt_h[rec["category"]].append(gt_h)
        
        is_ml = rec.get("is_multi_line_row", False)
        if is_ml:
            multi_line_gt_h.append(gt_h)
        else:
            single_line_gt_h.append(gt_h)

    print("\n--- Overall GT Box Height Distribution ---")
    print(f"All (N={len(all_gt_h)}): mean={np.mean(all_gt_h):.6f}, median={np.median(all_gt_h):.6f}, "
          f"min={np.min(all_gt_h):.6f}, max={np.max(all_gt_h):.6f}, std={np.std(all_gt_h):.6f}")
    
    print("\n--- By is_multi_line_row in registry ---")
    print(f"Single-line row (N={len(single_line_gt_h)}): mean={np.mean(single_line_gt_h):.6f}, median={np.median(single_line_gt_h):.6f}, "
          f"min={np.min(single_line_gt_h):.6f}, max={np.max(single_line_gt_h):.6f}")
    print(f"Multi-line row (N={len(multi_line_gt_h)}): mean={np.mean(multi_line_gt_h):.6f}, median={np.median(multi_line_gt_h):.6f}, "
          f"min={np.min(multi_line_gt_h):.6f}, max={np.max(multi_line_gt_h):.6f}")

    print("\n--- By Field ---")
    for fld in sorted(fld_gt_h.keys()):
        hs = fld_gt_h[fld]
        print(f"Field {fld:15s} (N={len(hs):5d}): mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, "
              f"min={np.min(hs):.6f}, max={np.max(hs):.6f}, p25={np.percentile(hs, 25):.6f}, p75={np.percentile(hs, 75):.6f}")

    print("\n--- By Baseline Category ---")
    for cat in sorted(cat_gt_h.keys()):
        hs = cat_gt_h[cat]
        print(f"Category {cat:10s} (N={len(hs):5d}): mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, "
              f"min={np.min(hs):.6f}, max={np.max(hs):.6f}")

if __name__ == "__main__":
    main()
