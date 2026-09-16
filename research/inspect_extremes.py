import sys
sys.path.insert(0, ".")
from research.scratch_analyze_full import page_results, gt_by_page_row

sorted_by_ystart = sorted(page_results, key=lambda x: x["y_start_raw"])
print("5 lowest y_start_raw pages:")
for pr in sorted_by_ystart[:5]:
    p = pr["page"]
    ys = pr["y_start_raw"]
    ye = pr["y_end_raw"]
    nr = pr["n_rows"]
    ts = pr["total_slots"]
    print(f"Page {p:3d}: y_start_raw={ys:.5f}, y_end_raw={ye:.5f}, n_rows={nr}, total_slots={ts}")

print("\n5 highest y_start_raw pages:")
for pr in sorted_by_ystart[-5:]:
    p = pr["page"]
    ys = pr["y_start_raw"]
    ye = pr["y_end_raw"]
    nr = pr["n_rows"]
    ts = pr["total_slots"]
    print(f"Page {p:3d}: y_start_raw={ys:.5f}, y_end_raw={ye:.5f}, n_rows={nr}, total_slots={ts}")

# Let's inspect the first 3 rows of the lowest page and highest page
p_low = sorted_by_ystart[0]["page"]
p_high = sorted_by_ystart[-1]["page"]

print(f"\nDetails for lowest page {p_low}:")
for r_idx in sorted(gt_by_page_row[p_low].keys())[:5]:
    print(f"  Row {r_idx}:")
    for fld, box in gt_by_page_row[p_low][r_idx].items():
        if box:
            print(f"    {fld:12s}: y={box[1]:.5f}, h={box[3]:.5f}, x={box[0]:.4f}")

print(f"\nDetails for highest page {p_high}:")
for r_idx in sorted(gt_by_page_row[p_high].keys())[:5]:
    print(f"  Row {r_idx}:")
    for fld, box in gt_by_page_row[p_high][r_idx].items():
        if box:
            print(f"    {fld:12s}: y={box[1]:.5f}, h={box[3]:.5f}, x={box[0]:.4f}")
