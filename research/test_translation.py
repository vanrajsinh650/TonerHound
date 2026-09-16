import sys
sys.path.insert(0, ".")
from tonerhound.document.index import DocumentIndex
from research.analyze_true_geometry import page_stats
import numpy as np

print("Loading index...")
idx = DocumentIndex.from_pdf("research/data/full/long/real_ftx_full_corrupted.pdf", enable_ocr=True)

offsets = []
for ps in page_stats:
    p = ps["page"]
    if p >= 114: continue
    page_obj = idx.get_page(p)
    slope = ps["page_gt_slope"]
    true_ys = ps["y_start_050"]
    true_ye = ps["y_end_050"]
    
    # 1. Header (Case 22-11068-JTD)
    case_toks = [t for t in page_obj.tokens if t.bbox.y < 0.035 and any(k in t.text.lower() for k in ("case", "22-11068", "doc", "page"))]
    case_y050 = [t.bbox.y + t.bbox.height/2.0 - slope * (t.bbox.x + t.bbox.width/2.0 - 0.50) for t in case_toks]
    med_case_y = np.median(case_y050) if case_y050 else None
    
    # 2. Subhead (Consolidated List...)
    subhead_toks = [t for t in page_obj.tokens if 0.035 <= t.bbox.y <= 0.065 and any(k in t.text.lower() for k in ("consolidated", "comsolidated", "list", "creditors"))]
    subhead_y050 = [t.bbox.y + t.bbox.height/2.0 - slope * (t.bbox.x + t.bbox.width/2.0 - 0.50) for t in subhead_toks]
    med_subhead_y = np.median(subhead_y050) if subhead_y050 else None
    
    # 3. Content lines between 0.065 and 0.90
    lines = [l for l in page_obj.lines if 0.060 <= l.bbox.y <= 0.92]
    line_y050 = [l.bbox.y + l.bbox.height/2.0 - slope * (l.bbox.x + l.bbox.width/2.0 - 0.50) for l in lines]
    line_y050.sort()
    
    # Filter out header/subhead
    table_lines = [ly for ly in line_y050 if (med_subhead_y is None or ly > med_subhead_y + 0.015)]
    first_tbl_line = table_lines[0] if table_lines else None
    last_tbl_line = table_lines[-1] if table_lines else None
    
    offsets.append({
        "page": p,
        "true_ys": true_ys,
        "true_ye": true_ye,
        "case_y": med_case_y,
        "subhead_y": med_subhead_y,
        "first_tbl_line": first_tbl_line,
        "last_tbl_line": last_tbl_line,
    })

print(f"Total analyzed pages: {len(offsets)}")
# Compare true_ys with subhead_y
valid_sh = [o for o in offsets if o["subhead_y"] is not None]
sh_diffs = [o["true_ys"] - o["subhead_y"] for o in valid_sh]
print(f"true_ys - subhead_y: mean={np.mean(sh_diffs):.5f}, std={np.std(sh_diffs):.5f}, min={np.min(sh_diffs):.5f}, max={np.max(sh_diffs):.5f}")

# Compare true_ys with case_y
valid_case = [o for o in offsets if o["case_y"] is not None]
case_diffs = [o["true_ys"] - o["case_y"] for o in valid_case]
print(f"true_ys - case_y   : mean={np.mean(case_diffs):.5f}, std={np.std(case_diffs):.5f}, min={np.min(case_diffs):.5f}, max={np.max(case_diffs):.5f}")

# Compare true_ys with first_tbl_line
valid_ftl = [o for o in offsets if o["first_tbl_line"] is not None]
ftl_diffs = [o["true_ys"] - o["first_tbl_line"] for o in valid_ftl]
print(f"true_ys - first_tbl_line: mean={np.mean(ftl_diffs):.5f}, std={np.std(ftl_diffs):.5f}")

# Compare true_ye with last_tbl_line
valid_ltl = [o for o in offsets if o["last_tbl_line"] is not None]
ltl_diffs = [o["true_ye"] - o["last_tbl_line"] for o in valid_ltl]
print(f"true_ye - last_tbl_line : mean={np.mean(ltl_diffs):.5f}, std={np.std(ltl_diffs):.5f}")
