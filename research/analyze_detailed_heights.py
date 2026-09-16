#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import sys

_REF = Path("research/reference/ExtractBench/src")
if _REF.exists(): sys.path.insert(0, str(_REF))
from extract_bench.test_cases.loader import load_test_case

def main():
    pdf_path = Path("research/data/full/long/real_ftx_full_corrupted.pdf")
    tc = load_test_case(pdf_path)

    # Map row index to page and gt bounding boxes
    gt_by_page_row = defaultdict(lambda: defaultdict(dict))
    for r in tc.test_rules:
        if not r.evidence:
            continue
        if "creditors[" in r.field_path:
            idx_str = r.field_path.split("[")[1].split("]")[0]
            row_idx = int(idx_str)
            field_name = r.field_path.split(".")[1]
            for ev in r.evidence:
                if ev.bbox is not None:
                    gt_by_page_row[ev.page][row_idx][field_name] = ev.bbox

    # For each page, determine consecutive row y spacing (dy) to identify row slot size (1-slot, 2-slot, 3-slot)
    row_slot_counts = {}  # (page, row_idx) -> slot_count
    
    # We can also measure row bounding box height: max(y+h) - min(y) across all fields in the row!
    row_bbox_heights = {} # (page, row_idx) -> total_height
    
    for p in sorted(gt_by_page_row.keys()):
        rows = sorted(gt_by_page_row[p].keys())
        # row tops
        row_tops = []
        for r_idx in rows:
            boxes = list(gt_by_page_row[p][r_idx].values())
            y_min = min(b[1] for b in boxes)
            y_max = max(b[1] + b[3] for b in boxes)
            row_bbox_heights[(p, r_idx)] = y_max - y_min
            # Unrotated top approximation or simple y_min
            # Using name or first box top
            row_tops.append((r_idx, y_min))
            
        for i in range(len(row_tops) - 1):
            r_curr, y_curr = row_tops[i]
            r_next, y_next = row_tops[i+1]
            dy = y_next - y_curr
            slots = max(1, int(round(dy / 0.01138)))
            row_slot_counts[(p, r_curr)] = slots
        # For the last row on page, infer from row_bbox_height or default to 1
        last_r = row_tops[-1][0]
        h_last = row_bbox_heights[(p, last_r)]
        row_slot_counts[(p, last_r)] = max(1, int(round(h_last / 0.01138)))

    # Distribution of row slot counts
    slot_dist = defaultdict(int)
    for slots in row_slot_counts.values():
        slot_dist[slots] += 1
    print(f"Row slot counts distribution across table rows: {dict(slot_dist)}")

    # Distribution of row bbox heights (max y - min y) by slot count
    h_by_slot = defaultdict(list)
    for (p, r_idx), slots in row_slot_counts.items():
        h = row_bbox_heights[(p, r_idx)]
        h_by_slot[slots].append(h)

    print("\n--- Row Total BBox Height (max_y - min_y across row fields) by Row Slot Count ---")
    for s in sorted(h_by_slot.keys()):
        hs = h_by_slot[s]
        print(f"Slot {s} (N={len(hs)} rows): mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, min={np.min(hs):.6f}, max={np.max(hs):.6f}")

    # Now let's analyze individual field citation GT box heights:
    # 1. Field heights in 1-slot rows
    # 2. Field heights in 2-slot rows
    # 3. Field heights in 3-slot rows
    # Also separate by whether the field itself is single-line (h < 0.015) vs multi-line (h >= 0.015)
    fld_h_by_row_slot = defaultdict(lambda: defaultdict(list))
    all_h_by_row_slot = defaultdict(list)
    single_line_field_h = []
    two_line_field_h = []
    three_line_field_h = []

    for p in sorted(gt_by_page_row.keys()):
        for r_idx in sorted(gt_by_page_row[p].keys()):
            slots = row_slot_counts.get((p, r_idx), 1)
            for fld, box in gt_by_page_row[p][r_idx].items():
                h = box[3]
                all_h_by_row_slot[slots].append(h)
                fld_h_by_row_slot[slots][fld].append(h)
                if h < 0.015:
                    single_line_field_h.append((fld, h))
                elif h < 0.026:
                    two_line_field_h.append((fld, h))
                else:
                    three_line_field_h.append((fld, h))

    print("\n--- Citation GT Box Heights by Row Slot Count ---")
    for s in sorted(all_h_by_row_slot.keys()):
        hs = all_h_by_row_slot[s]
        print(f"Row Slot {s} (N={len(hs)} citations): mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, min={np.min(hs):.6f}, max={np.max(hs):.6f}")

    print("\n--- Field Citation Heights by Line Count of the Field BBox ---")
    sl_hs = [h for _, h in single_line_field_h]
    tl_hs = [h for _, h in two_line_field_h]
    thl_hs = [h for _, h in three_line_field_h]
    print(f"Single-line fields (h < 0.015, N={len(sl_hs)}): mean={np.mean(sl_hs):.6f}, median={np.median(sl_hs):.6f}, min={np.min(sl_hs):.6f}, max={np.max(sl_hs):.6f}")
    if tl_hs:
        print(f"Two-line fields (0.015 <= h < 0.026, N={len(tl_hs)}): mean={np.mean(tl_hs):.6f}, median={np.median(tl_hs):.6f}, min={np.min(tl_hs):.6f}, max={np.max(tl_hs):.6f}")
    if thl_hs:
        print(f"Three-line fields (h >= 0.026, N={len(thl_hs)}): mean={np.mean(thl_hs):.6f}, median={np.median(thl_hs):.6f}, min={np.min(thl_hs):.6f}, max={np.max(thl_hs):.6f}")

    print("\n--- Single-line Field Citation Heights by Field Name (h < 0.015) ---")
    fld_sl_hs = defaultdict(list)
    for fld, h in single_line_field_h:
        fld_sl_hs[fld].append(h)
    for fld in sorted(fld_sl_hs.keys()):
        hs = fld_sl_hs[fld]
        print(f"  Field {fld:15s} (N={len(hs):5d}): mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, min={np.min(hs):.6f}, max={np.max(hs):.6f}, p25={np.percentile(hs, 25):.6f}, p75={np.percentile(hs, 75):.6f}")

    print("\n--- Multi-line Field Citation Heights by Field Name (h >= 0.015) ---")
    fld_ml_hs = defaultdict(list)
    for fld, h in two_line_field_h + three_line_field_h:
        fld_ml_hs[fld].append(h)
    for fld in sorted(fld_ml_hs.keys()):
        hs = fld_ml_hs[fld]
        print(f"  Field {fld:15s} (N={len(hs):5d}): mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, min={np.min(hs):.6f}, max={np.max(hs):.6f}, p25={np.percentile(hs, 25):.6f}, p75={np.percentile(hs, 75):.6f}")

if __name__ == "__main__":
    main()
