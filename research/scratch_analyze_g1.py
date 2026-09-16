import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

_REF = Path("research/reference/ExtractBench/src")
if _REF.exists(): sys.path.insert(0, str(_REF))
from extract_bench.test_cases.loader import load_test_case
from tonerhound.document.index import DocumentIndex

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
            gt_by_page_row[ev.page][row_idx][field_name] = ev.bbox

pages = sorted(gt_by_page_row.keys())
print(f"Total pages with creditor GT: {len(pages)} (min={min(pages)}, max={max(pages)})")

page_stats = []
all_y_starts = []
all_y_ends = []
all_spans = []

for p in pages:
    rows = sorted(gt_by_page_row[p].keys())
    row_y = []
    for r_idx in rows:
        boxes = [b for b in gt_by_page_row[p][r_idx].values() if b is not None]
        if not boxes:
            continue
        y_mins = [b[1] for b in boxes]
        y_maxs = [b[1] + b[3] for b in boxes]
        row_y.append((r_idx, min(y_mins), max(y_maxs)))
    if not row_y:
        continue
    
    first_y = row_y[0][1]
    last_y = row_y[-1][1]
    last_y_bottom = row_y[-1][2]
    span = last_y - first_y
    all_y_starts.append(first_y)
    all_y_ends.append(last_y)
    all_spans.append(span)
    
    page_stats.append({
        "page": p,
        "n_rows": len(rows),
        "first_row_idx": rows[0],
        "last_row_idx": rows[-1],
        "y_start": first_y,
        "y_end": last_y,
        "y_end_bottom": last_y_bottom,
        "span": span,
    })

print("\n--- Summary across all 113 creditor pages (2 to 114) ---")
print(f"y_start: min={min(all_y_starts):.5f}, max={max(all_y_starts):.5f}, mean={np.mean(all_y_starts):.5f}, std={np.std(all_y_starts):.5f}")
print(f"y_end  : min={min(all_y_ends):.5f}, max={max(all_y_ends):.5f}, mean={np.mean(all_y_ends):.5f}, std={np.std(all_y_ends):.5f}")
print(f"span   : min={min(all_spans):.5f}, max={max(all_spans):.5f}, mean={np.mean(all_spans):.5f}, std={np.std(all_spans):.5f}")

print("\nSample page stats (first 10 pages):")
for ps in page_stats[:10]:
    p = ps["page"]
    nr = ps["n_rows"]
    r0, r1 = ps["first_row_idx"], ps["last_row_idx"]
    ys, ye, sp = ps["y_start"], ps["y_end"], ps["span"]
    print(f"Page {p:3d}: rows={nr:2d}, range=[{r0:4d}:{r1:4d}], y_start={ys:.5f}, y_end={ye:.5f}, span={sp:.5f}")

print("\nSample page stats (pages 50-55):")
for ps in page_stats[48:54]:
    p = ps["page"]
    nr = ps["n_rows"]
    r0, r1 = ps["first_row_idx"], ps["last_row_idx"]
    ys, ye, sp = ps["y_start"], ps["y_end"], ps["span"]
    print(f"Page {p:3d}: rows={nr:2d}, range=[{r0:4d}:{r1:4d}], y_start={ys:.5f}, y_end={ye:.5f}, span={sp:.5f}")

print("\nSample page stats (last 5 pages):")
for ps in page_stats[-5:]:
    p = ps["page"]
    nr = ps["n_rows"]
    r0, r1 = ps["first_row_idx"], ps["last_row_idx"]
    ys, ye, sp = ps["y_start"], ps["y_end"], ps["span"]
    print(f"Page {p:3d}: rows={nr:2d}, range=[{r0:4d}:{r1:4d}], y_start={ys:.5f}, y_end={ye:.5f}, span={sp:.5f}")
