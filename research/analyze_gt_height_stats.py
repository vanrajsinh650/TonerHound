#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
if _REF.exists(): sys.path.insert(0, str(_REF))
sys.path.insert(0, str(repo_root / "src"))

from extract_bench.test_cases.loader import load_test_case
from tonerhound.benchmark.correlation import FailureRegistry

def main():
    pdf_path = repo_root / "research/data/full/long/real_ftx_full_corrupted.pdf"
    tc = load_test_case(pdf_path)
    reg = FailureRegistry.load(repo_root / "experiments/EXP-005-ftx-failure-registry.json")

    # Map row index to page and gt bounding boxes
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

    # Determine row slot count from physical GT row spacing
    row_slots = {}  # (page, row_idx) -> slots
    row_bbox_span = {} # (page, row_idx) -> max_y - min_y

    for p in sorted(gt_by_page_row.keys()):
        rows = sorted(gt_by_page_row[p].keys())
        row_tops = []
        for r_idx in rows:
            boxes = [b for path, b in gt_by_page_row[p][r_idx].values()]
            y_min = min(b[1] for b in boxes)
            y_max = max(b[1] + b[3] for b in boxes)
            row_bbox_span[(p, r_idx)] = y_max - y_min
            row_tops.append((r_idx, y_min))
            
        for i in range(len(row_tops) - 1):
            r_curr, y_curr = row_tops[i]
            r_next, y_next = row_tops[i+1]
            dy = y_next - y_curr
            slots = max(1, int(round(dy / 0.01138)))
            row_slots[(p, r_curr)] = slots
            
        last_r = row_tops[-1][0]
        h_last = row_bbox_span[(p, last_r)]
        row_slots[(p, last_r)] = max(1, int(round(h_last / 0.01138)))

    # Collect citation heights by row slot count
    cits_by_slot = defaultdict(list)
    # Also collect by row slot AND field
    cits_by_slot_field = defaultdict(lambda: defaultdict(list))
    # Also collect by field across all citations
    cits_by_field = defaultdict(list)
    # Also collect multi-line citations in 2-slot and 3-slot rows
    ml_cits_by_slot = defaultdict(list)
    sl_cits_by_slot = defaultdict(list)

    total_citations_counted = 0
    for p in sorted(gt_by_page_row.keys()):
        for r_idx in sorted(gt_by_page_row[p].keys()):
            s = row_slots.get((p, r_idx), 1)
            for fld, (path, box) in gt_by_page_row[p][r_idx].items():
                h = box[3]
                total_citations_counted += 1
                cits_by_slot[s].append(h)
                cits_by_slot_field[s][fld].append(h)
                cits_by_field[fld].append(h)
                if h >= 0.015:
                    ml_cits_by_slot[s].append(h)
                else:
                    sl_cits_by_slot[s].append(h)

    for path, p, box in non_table_records:
        total_citations_counted += 1
        fld = path.split(".")[0]
        cits_by_field[fld].append(box[3])

    print(f"Total gradeable citations counted: {total_citations_counted} (Registry total: {len(reg.records)})")

    print("\n================================================================================")
    print("1. GROUND TRUTH BOUNDING BOX HEIGHT BY ROW SLOT COUNT")
    print("================================================================================")
    for s in [1, 2, 3]:
        hs = cits_by_slot[s]
        if not hs: continue
        print(f"\nRow Slot {s} (Total Citations N={len(hs)}):")
        print(f"  All Citations in Row  : mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, min={np.min(hs):.6f}, max={np.max(hs):.6f}, std={np.std(hs):.6f}")
        sl = sl_cits_by_slot[s]
        ml = ml_cits_by_slot[s]
        if sl:
            print(f"  Single-line Citations : N={len(sl):5d} | mean={np.mean(sl):.6f}, median={np.median(sl):.6f}, min={np.min(sl):.6f}, max={np.max(sl):.6f}")
        if ml:
            print(f"  Multi-line Citations  : N={len(ml):5d} | mean={np.mean(ml):.6f}, median={np.median(ml):.6f}, min={np.min(ml):.6f}, max={np.max(ml):.6f}")

    print("\n================================================================================")
    print("2. SINGLE-LINE BOXES VS GROUND TRUTH HEIGHT")
    print("================================================================================")
    all_sl = []
    for s in cits_by_slot:
        all_sl.extend(sl_cits_by_slot[s])
    print(f"All Single-Line Citations (h < 0.015, N={len(all_sl)}):")
    print(f"  Mean  : {np.mean(all_sl):.6f}")
    print(f"  Median: {np.median(all_sl):.6f}")
    print(f"  Min   : {np.min(all_sl):.6f}")
    print(f"  Max   : {np.max(all_sl):.6f}")
    print(f"  p10   : {np.percentile(all_sl, 10):.6f}")
    print(f"  p25   : {np.percentile(all_sl, 25):.6f}")
    print(f"  p75   : {np.percentile(all_sl, 75):.6f}")
    print(f"  p90   : {np.percentile(all_sl, 90):.6f}")

    print("\n================================================================================")
    print("3. GROUND TRUTH BOX HEIGHT VARIATION BY FIELD")
    print("================================================================================")
    for fld in sorted(cits_by_field.keys()):
        hs = cits_by_field[fld]
        hs_sl = [h for h in hs if h < 0.015]
        hs_ml = [h for h in hs if h >= 0.015]
        print(f"\nField: {fld} (Total N={len(hs)}):")
        print(f"  Overall           : mean={np.mean(hs):.6f}, median={np.median(hs):.6f}, min={np.min(hs):.6f}, max={np.max(hs):.6f}")
        if hs_sl:
            print(f"  Single-line (N={len(hs_sl):5d}): mean={np.mean(hs_sl):.6f}, median={np.median(hs_sl):.6f}, min={np.min(hs_sl):.6f}, max={np.max(hs_sl):.6f}")
        if hs_ml:
            print(f"  Multi-line  (N={len(hs_ml):5d}): mean={np.mean(hs_ml):.6f}, median={np.median(hs_ml):.6f}, min={np.min(hs_ml):.6f}, max={np.max(hs_ml):.6f}")

if __name__ == "__main__":
    main()
