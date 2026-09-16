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
print(f"Loaded! tc rules: {len(tc.test_rules)}, index pages: {len(idx.pages)}")

# 1. Ground truth mapping
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

# Analyze each page
page_results = []
for p in pages:
    rows = sorted(gt_by_page_row[p].keys())
    row_info = []
    for r_idx in rows:
        boxes = list(gt_by_page_row[p][r_idx].values())
        y_min = min(b[1] for b in boxes)
        y_max = max(b[1] + b[3] for b in boxes)
        h = y_max - y_min
        # name box or first box
        name_box = gt_by_page_row[p][r_idx].get("name")
        addr_box = gt_by_page_row[p][r_idx].get("address_1")
        ref_y = (name_box or addr_box or boxes[0])[1]
        ref_h = (name_box or addr_box or boxes[0])[3]
        row_info.append({
            "row_idx": r_idx,
            "y": ref_y,
            "h": ref_h,
            "y_min": y_min,
            "y_max": y_max,
        })
    
    # Measure diffs between consecutive rows
    y_diffs = []
    for i in range(len(row_info) - 1):
        dy = row_info[i+1]["y"] - row_info[i]["y"]
        y_diffs.append(dy)
    
    # A single slot step is approx 0.0112 - 0.0115
    # Two slots step is approx 0.0224 - 0.0230
    slot_counts = []
    for dy in y_diffs:
        # Determine if 1 slot or 2 slots or more
        s_est = round(dy / 0.0114)
        slot_counts.append(max(1, s_est))
    
    # Total slots between first row and last row
    total_slots_span = sum(slot_counts)
    # Total slots on page if full table
    total_page_slots = total_slots_span + 1
    
    # Assign slot index to each row
    slot_indices = [0]
    for sc in slot_counts:
        slot_indices.append(slot_indices[-1] + sc)
        
    # Fit linear regression: y = y_start + s * slot_index
    xs = np.array(slot_indices)
    ys = np.array([ri["y"] for ri in row_info])
    A = np.vstack([xs, np.ones(len(xs))]).T
    s_fit, y_start_fit = np.linalg.lstsq(A, ys, rcond=None)[0]
    residuals = ys - (y_start_fit + s_fit * xs)
    max_res = np.max(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Linear drift across top, mid, bot
    n_pts = len(ys)
    top_res = residuals[:n_pts//3] if n_pts >= 3 else residuals
    mid_res = residuals[n_pts//3: 2*n_pts//3] if n_pts >= 3 else residuals
    bot_res = residuals[2*n_pts//3:] if n_pts >= 3 else residuals
    
    # OCR features on page
    page_obj = idx.get_page(p)
    header_y = None
    subhead_y = None
    if page_obj and page_obj.tokens:
        h_toks = [t.bbox.y for t in page_obj.tokens if t.bbox.y < 0.035 and any(k in t.text.lower() for k in ("case", "22-11068", "doc", "page", "114"))]
        if h_toks:
            header_y = float(np.median(h_toks))
        sh_toks = [t.bbox.y for t in page_obj.tokens if 0.035 <= t.bbox.y <= 0.060 and any(k in t.text.lower() for k in ("consolidated", "comsolidated", "list", "creditors"))]
        if sh_toks:
            subhead_y = float(np.median(sh_toks))
            
    page_results.append({
        "page": p,
        "n_rows": len(rows),
        "total_slots": total_page_slots,
        "n_two_slot": sum(1 for sc in slot_counts if sc == 2),
        "y_start_raw": row_info[0]["y"],
        "y_end_raw": row_info[-1]["y"],
        "span_raw": row_info[-1]["y"] - row_info[0]["y"],
        "y_start_fit": float(y_start_fit),
        "s_fit": float(s_fit),
        "rmse": float(rmse),
        "max_res": float(max_res),
        "top_res_mean": float(np.mean(top_res)),
        "mid_res_mean": float(np.mean(mid_res)),
        "bot_res_mean": float(np.mean(bot_res)),
        "header_y": header_y,
        "subhead_y": subhead_y,
    })

print("\n--- Physical Pitch and Geometry Summary across all 113 pages ---")
# Exclude page 114 (which is partial, only 19 rows) for full page slot summary
full_pages = [pr for pr in page_results if pr["page"] < 114]
total_slots_list = [pr["total_slots"] for pr in full_pages]
s_fit_list = [pr["s_fit"] for pr in full_pages]
y_start_fit_list = [pr["y_start_fit"] for pr in full_pages]
rmse_list = [pr["rmse"] for pr in full_pages]
max_res_list = [pr["max_res"] for pr in full_pages]

print(f"Total full pages: {len(full_pages)}")
print(f"Total slots per page distribution: {dict(defaultdict(int, {x: total_slots_list.count(x) for x in set(total_slots_list)}))}")
print(f"Fitted slot pitch s: mean={np.mean(s_fit_list):.6f}, std={np.std(s_fit_list):.6f}, min={np.min(s_fit_list):.6f}, max={np.max(s_fit_list):.6f}")
print(f"Fitted y_start     : mean={np.mean(y_start_fit_list):.6f}, std={np.std(y_start_fit_list):.6f}, min={np.min(y_start_fit_list):.6f}, max={np.max(y_start_fit_list):.6f}")
print(f"Fit RMSE           : mean={np.mean(rmse_list):.6f}, max={np.max(rmse_list):.6f}")
print(f"Fit Max Residual   : mean={np.mean(max_res_list):.6f}, max={np.max(max_res_list):.6f}")

print("\nPage 114 (partial page):")
p114 = [pr for pr in page_results if pr["page"] == 114][0]
print(f"Page 114: rows={p114['n_rows']}, slots_used={p114['total_slots']}, y_start_fit={p114['y_start_fit']:.5f}, s_fit={p114['s_fit']:.6f}, rmse={p114['rmse']:.6f}")
