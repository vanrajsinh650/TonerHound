import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

_REF = Path("research/reference/ExtractBench/src")
if _REF.exists(): sys.path.insert(0, str(_REF))
from extract_bench.test_cases.loader import load_test_case
from tonerhound.document.index import DocumentIndex

print("Loading test case and document index...")
pdf_path = Path("research/data/full/long/real_ftx_full_corrupted.pdf")
tc = load_test_case(pdf_path)
idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True)

# 1. Map row index to page and gt bounding boxes
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

pages = sorted(gt_by_page_row.keys())

# For each page, analyze row geometry at x = 0.50
page_stats = []

for p in pages:
    rows = sorted(gt_by_page_row[p].keys())
    
    # First: estimate page slope from all rows with >= 2 fields
    slope_samples = []
    for r_idx in rows:
        boxes = [(fld, b) for fld, b in gt_by_page_row[p][r_idx].items() if b is not None]
        if len(boxes) >= 2:
            # Fit line through center points of boxes: yc vs xc
            xcs = [b[0] + b[2]/2.0 for fld, b in boxes]
            ycs = [b[1] + b[3]/2.0 for fld, b in boxes]
            if max(xcs) - min(xcs) > 0.3:
                # Least squares slope
                slope = np.polyfit(xcs, ycs, 1)[0]
                slope_samples.append(slope)
                
    page_gt_slope = float(np.median(slope_samples)) if slope_samples else 0.0
    
    # Now for each row, compute unrotated y_center at x = 0.50 and top y at x = 0.50
    # For a box (x, y, w, h), its center is (xc, yc).
    # Unrotated yc at x=0.50: yc_050 = yc - page_gt_slope * (xc - 0.50)
    # top y at x=0.50: y_top_050 = (y + h/2.0) - page_gt_slope * (x + w/2.0 - 0.50) - h/2.0
    row_geoms = []
    for r_idx in rows:
        boxes = [(fld, b) for fld, b in gt_by_page_row[p][r_idx].items() if b is not None]
        # Calculate row y_top at x = 0.50
        y_tops = []
        for fld, b in boxes:
            xc = b[0] + b[2]/2.0
            yc = b[1] + b[3]/2.0
            # If multi-line address, only line 1 has top near true row top
            # Check h
            y_top_fld = (yc - b[3]/2.0) - page_gt_slope * (xc - 0.50)
            y_tops.append(y_top_fld)
        # Median y_top across fields gives the unrotated row top at x = 0.50
        row_y_top = float(np.min(y_tops))
        row_geoms.append({
            "row_idx": r_idx,
            "y_top": row_y_top,
        })
        
    # Analyze slot spacing between consecutive rows
    y_diffs = []
    for i in range(len(row_geoms) - 1):
        dy = row_geoms[i+1]["y_top"] - row_geoms[i]["y_top"]
        y_diffs.append(dy)
        
    # Slot counts
    slot_counts = [max(1, int(round(dy / 0.01138))) for dy in y_diffs]
    total_slots_span = sum(slot_counts)
    total_page_slots = total_slots_span + 1
    
    slot_indices = [0]
    for sc in slot_counts:
        slot_indices.append(slot_indices[-1] + sc)
        
    xs = np.array(slot_indices)
    ys = np.array([rg["y_top"] for rg in row_geoms])
    A = np.vstack([xs, np.ones(len(xs))]).T
    s_fit, y_start_fit = np.linalg.lstsq(A, ys, rcond=None)[0]
    residuals = ys - (y_start_fit + s_fit * xs)
    rmse = np.sqrt(np.mean(residuals**2))
    max_res = np.max(np.abs(residuals))
    
    # Non-linear drift down the page: residuals in top, mid, bot thirds
    n_pts = len(ys)
    t_res = float(np.mean(residuals[:n_pts//3])) if n_pts >= 3 else 0.0
    m_res = float(np.mean(residuals[n_pts//3: 2*n_pts//3])) if n_pts >= 3 else 0.0
    b_res = float(np.mean(residuals[2*n_pts//3:])) if n_pts >= 3 else 0.0
    
    end_y_fit = y_start_fit + s_fit * (total_page_slots - 1)
    
    page_stats.append({
        "page": p,
        "n_rows": len(rows),
        "total_slots": total_page_slots,
        "n_two_slot": sum(1 for sc in slot_counts if sc == 2),
        "n_three_slot": sum(1 for sc in slot_counts if sc >= 3),
        "page_gt_slope": page_gt_slope,
        "y_start_050": float(y_start_fit),
        "y_end_050": float(end_y_fit),
        "s_fit": float(s_fit),
        "rmse": float(rmse),
        "max_res": float(max_res),
        "top_res": t_res,
        "mid_res": m_res,
        "bot_res": b_res,
    })

full_pages = [ps for ps in page_stats if ps["page"] < 114]
p114 = [ps for ps in page_stats if ps["page"] == 114][0]

print("\n========================================================")
print("TRUE PHYSICAL GEOMETRY AT x = 0.50 (AFTER TILT DECOUPLING)")
print("========================================================")
print(f"Number of full pages: {len(full_pages)}")
y_starts = [ps["y_start_050"] for ps in full_pages]
y_ends = [ps["y_end_050"] for ps in full_pages]
s_fits = [ps["s_fit"] for ps in full_pages]
total_slots = [ps["total_slots"] for ps in full_pages]
slopes = [ps["page_gt_slope"] for ps in full_pages]
rmses = [ps["rmse"] for ps in full_pages]

print(f"\ny_start (at x=0.50):")
print(f"  Mean: {np.mean(y_starts):.5f}")
print(f"  Std : {np.std(y_starts):.5f}")
print(f"  Min : {np.min(y_starts):.5f}")
print(f"  Max : {np.max(y_starts):.5f}")
print(f"  Range: {np.max(y_starts) - np.min(y_starts):.5f}")

print(f"\nend_y (at x=0.50):")
print(f"  Mean: {np.mean(y_ends):.5f}")
print(f"  Std : {np.std(y_ends):.5f}")
print(f"  Min : {np.min(y_ends):.5f}")
print(f"  Max : {np.max(y_ends):.5f}")
print(f"  Range: {np.max(y_ends) - np.min(y_ends):.5f}")

print(f"\nSlot Pitch s:")
print(f"  Mean: {np.mean(s_fits):.6f}")
print(f"  Std : {np.std(s_fits):.6f}")
print(f"  Min : {np.min(s_fits):.6f}")
print(f"  Max : {np.max(s_fits):.6f}")
print(f"  Range: {np.max(s_fits) - np.min(s_fits):.6f}")

print(f"\nTotal Slots per Page:")
slot_dist = defaultdict(int)
for ts in total_slots:
    slot_dist[ts] += 1
print(f"  Distribution: {dict(slot_dist)}")

print(f"\nPage Slope (tilt):")
print(f"  Mean: {np.mean(slopes):.5f}")
print(f"  Std : {np.std(slopes):.5f}")
print(f"  Min : {np.min(slopes):.5f}")
print(f"  Max : {np.max(slopes):.5f}")

print(f"\nLinear Fit Quality (Residuals):")
print(f"  Mean RMSE: {np.mean(rmses):.6f}")
print(f"  Max RMSE : {np.max(rmses):.6f}")
top_res_all = [ps["top_res"] for ps in full_pages]
mid_res_all = [ps["mid_res"] for ps in full_pages]
bot_res_all = [ps["bot_res"] for ps in full_pages]
print(f"  Top 1/3 Residual Mean   : {np.mean(top_res_all):.6f}")
print(f"  Middle 1/3 Residual Mean: {np.mean(mid_res_all):.6f}")
print(f"  Bottom 1/3 Residual Mean: {np.mean(bot_res_all):.6f}")

print(f"\nPage 114:")
print(f"  Rows: {p114['n_rows']}, Slots: {p114['total_slots']}, y_start: {p114['y_start_050']:.5f}, s: {p114['s_fit']:.6f}, slope: {p114['page_gt_slope']:.5f}")
