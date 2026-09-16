import sys
sys.path.insert(0, ".")
from research.analyze_true_geometry import page_stats

worst_pages = [105, 114, 100, 101, 27, 46, 84, 75, 66, 32, 48, 50, 24, 94, 67]

print("True Geometry for Worst Performing Pages:")
print("Page | TrueSlots | N_Rows | N_TwoSlot | True y_start | True end_y | True s    | Page Slope")
print("---------------------------------------------------------------------------------------")
for p in worst_pages:
    ps = next(x for x in page_stats if x["page"] == p)
    ts = ps["total_slots"]
    nr = ps["n_rows"]
    n2 = ps["n_two_slot"]
    ys = ps["y_start_050"]
    ye = ps["y_end_050"]
    s = ps["s_fit"]
    sl = ps["page_gt_slope"]
    print(f"{p:4d} | {ts:9d} | {nr:6d} | {n2:9d} | {ys:12.5f} | {ye:10.5f} | {s:9.6f} | {sl:+.5f}")

best_pages = [58, 87, 57, 4, 59, 65, 77, 26, 49, 10, 90, 7, 85, 113]
print("\nTrue Geometry for Best Performing Pages:")
print("Page | TrueSlots | N_Rows | N_TwoSlot | True y_start | True end_y | True s    | Page Slope")
print("---------------------------------------------------------------------------------------")
for p in best_pages:
    ps = next(x for x in page_stats if x["page"] == p)
    ts = ps["total_slots"]
    nr = ps["n_rows"]
    n2 = ps["n_two_slot"]
    ys = ps["y_start_050"]
    ye = ps["y_end_050"]
    s = ps["s_fit"]
    sl = ps["page_gt_slope"]
    print(f"{p:4d} | {ts:9d} | {nr:6d} | {n2:9d} | {ys:12.5f} | {ye:10.5f} | {s:9.6f} | {sl:+.5f}")
