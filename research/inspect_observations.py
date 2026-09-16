import sys
sys.path.insert(0, ".")
from pathlib import Path
from tonerhound.document.index import DocumentIndex
from research.analyze_true_geometry import page_stats
import numpy as np

print("Loading index...")
idx = DocumentIndex.from_pdf("research/data/full/long/real_ftx_full_corrupted.pdf", enable_ocr=True)

# For each page, let us inspect what lines exist and their y positions
test_pages = [2, 4, 10, 27, 46, 50, 66, 67, 75, 85, 94, 100, 105, 113, 114]

for p in test_pages:
    page_obj = idx.get_page(p)
    ps = next(x for x in page_stats if x["page"] == p)
    true_ys = ps["y_start_050"]
    true_ye = ps["y_end_050"]
    true_s = ps["s_fit"]
    true_ts = ps["total_slots"]
    slope = ps["page_gt_slope"]
    
    # 1. Candidate table lines (0.05 <= y <= 0.92)
    lines = [l for l in page_obj.lines if 0.05 <= l.bbox.y <= 0.92]
    # Unrotate line y_center to x=0.50
    line_y050 = []
    for l in lines:
        yc = l.bbox.y + l.bbox.height / 2.0
        xc = l.bbox.x + l.bbox.width / 2.0
        y050 = yc - slope * (xc - 0.50)
        line_y050.append((y050, l))
    line_y050.sort(key=lambda x: x[0])
    
    # Header tokens (Case...)
    case_toks = [t for t in page_obj.tokens if t.bbox.y < 0.035 and any(k in t.text.lower() for k in ("case", "22-11068", "doc", "page"))]
    case_y050 = []
    for t in case_toks:
        yc = t.bbox.y + t.bbox.height / 2.0
        xc = t.bbox.x + t.bbox.width / 2.0
        case_y050.append(yc - slope * (xc - 0.50))
    med_case_y = np.median(case_y050) if case_y050 else 0.02
    
    # Subhead (Consolidated List...)
    subhead_toks = [t for t in page_obj.tokens if 0.035 <= t.bbox.y <= 0.065 and any(k in t.text.lower() for k in ("consolidated", "comsolidated", "list", "creditors"))]
    subhead_y050 = []
    for t in subhead_toks:
        yc = t.bbox.y + t.bbox.height / 2.0
        xc = t.bbox.x + t.bbox.width / 2.0
        subhead_y050.append(yc - slope * (xc - 0.50))
    med_subhead_y = np.median(subhead_y050) if subhead_y050 else 0.048
    
    # Table column header line (Name, Address...) between 0.06 and 0.08
    col_hdr_toks = [t for t in page_obj.tokens if 0.060 <= t.bbox.y <= 0.082 and any(k in t.text.lower() for k in ("name", "address", "city", "state", "postal", "zip", "country"))]
    col_hdr_y050 = []
    for t in col_hdr_toks:
        yc = t.bbox.y + t.bbox.height / 2.0
        xc = t.bbox.x + t.bbox.width / 2.0
        col_hdr_y050.append(yc - slope * (xc - 0.50))
    med_col_hdr_y = np.median(col_hdr_y050) if col_hdr_y050 else None
    
    # First content line after subhead/headers (y050 > 0.070)
    content_lines = [ly for ly, l in line_y050 if ly > med_subhead_y + 0.02]
    first_line_y = content_lines[0] if content_lines else None
    last_line_y = content_lines[-1] if content_lines else None
    
    print(f"Page {p:3d} (TrueSlots={true_ts:2d}, M={ps['n_rows']:2d}):")
    print(f"  True  : y_start={true_ys:.5f}, end_y={true_ye:.5f}, span={true_ye-true_ys:.5f}, s={true_s:.6f}")
    print(f"  Obsrv : case_y={med_case_y:.5f}, subhead_y={med_subhead_y:.5f}, col_hdr_y={med_col_hdr_y if med_col_hdr_y else 0:.5f}")
    print(f"          first_line_y={first_line_y if first_line_y else 0:.5f}, last_line_y={last_line_y if last_line_y else 0:.5f}")
    if first_line_y:
        print(f"  Diff  : first_line_y - true_y_start = {first_line_y - true_ys:+.5f}")
    if last_line_y:
        print(f"  Diff  : last_line_y - true_end_y = {last_line_y - true_ye:+.5f}")
