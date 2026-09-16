import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

_REF = Path("research/reference/ExtractBench/src")
if _REF.exists(): sys.path.insert(0, str(_REF))
from extract_bench.test_cases.loader import load_test_case
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex

print("Loading test case and document index...")
pdf_path = Path("research/data/full/long/real_ftx_full_corrupted.pdf")
tc = load_test_case(pdf_path)
idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True)

print("Running adapter.ground_extracted_data...")
t0 = time.time()
adapter = ExtractBenchAdapter(
    idx,
    enable_structural_disambiguation=True,
    enable_verification=True,
    score_margin_threshold=0.01,
    enable_bbox_precision=True,
)
payload = adapter.ground_extracted_data(
    tc.expected_output,
    example_id=tc.test_id,
    pipeline_name="tonerhound",
)
t1 = time.time()
print(f"Grounding completed in {t1 - t0:.2f}s! Citations count: {len(payload['field_citations'])}")

# Build map of predictions: field_path -> citation
pred_map = {c["field_path"]: c for c in payload["field_citations"]}

# Evaluate against tc.test_rules
# Group by page
page_eval = defaultdict(lambda: {
    "total_gradeable": 0,
    "passing": 0,
    "iou_list": [],
    "dy_list": [],
    "dx_list": [],
    "slot_drift_count": 0, # |dy| >= 0.006 (approx >= 0.5 slot)
    "severe_drift_count": 0, # |dy| >= 0.011 (>= 1 slot)
    "wrong_page_count": 0,
    "none_count": 0,
    "top_third_dy": [],
    "mid_third_dy": [],
    "bot_third_dy": [],
})

def compute_iou(b1, b2):
    # b1, b2: [x, y, w, h]
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi0 = max(x1, x2)
    yi0 = max(y1, y2)
    xi1 = min(x1 + w1, x2 + w2)
    yi1 = min(y1 + h1, y2 + h2)
    if xi1 <= xi0 or yi1 <= yi0:
        return 0.0
    inter = (xi1 - xi0) * (yi1 - yi0)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

for r in tc.test_rules:
    if not r.evidence:
        continue
    ev = r.evidence[0]
    if ev.bbox is None:
        continue
    gt_page = ev.page
    gt_box = ev.bbox
    
    stats = page_eval[gt_page]
    stats["total_gradeable"] += 1
    
    pred = pred_map.get(r.field_path)
    if not pred or pred.get("bbox") is None:
        stats["none_count"] += 1
        stats["iou_list"].append(0.0)
        continue
    
    pred_page = pred.get("page")
    if pred_page != gt_page:
        stats["wrong_page_count"] += 1
        stats["iou_list"].append(0.0)
        continue
        
    p_box = pred["bbox"]
    iou = compute_iou(p_box, gt_box)
    stats["iou_list"].append(iou)
    if iou >= 0.50:
        stats["passing"] += 1
        
    # Vertical and horizontal drift
    # Compare centers
    pred_yc = p_box[1] + p_box[3] / 2.0
    gt_yc = gt_box[1] + gt_box[3] / 2.0
    dy = pred_yc - gt_yc
    
    pred_xc = p_box[0] + p_box[2] / 2.0
    gt_xc = gt_box[0] + gt_box[2] / 2.0
    dx = pred_xc - gt_xc
    
    stats["dy_list"].append(dy)
    stats["dx_list"].append(dx)
    
    if abs(dy) >= 0.006:
        stats["slot_drift_count"] += 1
    if abs(dy) >= 0.011:
        stats["severe_drift_count"] += 1
        
    # Categorize by vertical position on page
    if gt_yc < 0.35:
        stats["top_third_dy"].append(dy)
    elif gt_yc < 0.65:
        stats["mid_third_dy"].append(dy)
    else:
        stats["bot_third_dy"].append(dy)

# Global stats
all_gradeable = sum(s["total_gradeable"] for s in page_eval.values())
all_passing = sum(s["passing"] for s in page_eval.values())
all_slot_drift = sum(s["slot_drift_count"] for s in page_eval.values())
all_severe_drift = sum(s["severe_drift_count"] for s in page_eval.values())

print(f"\n========================================================")
print(f"CURRENT ADAPTER ACCURACY & DRIFT ON ALL 114 PAGES")
print(f"========================================================")
print(f"Total Gradeable Rules : {all_gradeable}")
print(f"Passing Citations (IoU >= 0.50): {all_passing} ({all_passing / all_gradeable * 100:.2f}%)")
print(f"Failing Citations     : {all_gradeable - all_passing} ({(all_gradeable - all_passing) / all_gradeable * 100:.2f}%)")
print(f"Rules with Slot Drift (|dy| >= 0.006): {all_slot_drift} ({all_slot_drift / all_gradeable * 100:.2f}%)")
print(f"Rules with Severe Drift (|dy| >= 0.011): {all_severe_drift} ({all_severe_drift / all_gradeable * 100:.2f}%)")

# Page level statistics
f1_per_page = []
for p in sorted(page_eval.keys()):
    st = page_eval[p]
    p_f1 = st["passing"] / st["total_gradeable"] if st["total_gradeable"] > 0 else 0
    f1_per_page.append((p, p_f1, st["total_gradeable"], st["passing"], st["slot_drift_count"]))

f1_vals = [x[1] for x in f1_per_page]
print(f"\nPage-level Word Grounding F1 summary:")
print(f"  Mean Page F1  : {np.mean(f1_vals) * 100:.2f}%")
print(f"  Median Page F1: {np.median(f1_vals) * 100:.2f}%")
print(f"  Min Page F1   : {np.min(f1_vals) * 100:.2f}% (Page {min(f1_per_page, key=lambda x: x[1])[0]})")
print(f"  Max Page F1   : {np.max(f1_vals) * 100:.2f}% (Page {max(f1_per_page, key=lambda x: x[1])[0]})")

# Distribution of page F1
b_0_20 = sum(1 for v in f1_vals if v < 0.20)
b_20_50 = sum(1 for v in f1_vals if 0.20 <= v < 0.50)
b_50_80 = sum(1 for v in f1_vals if 0.50 <= v < 0.80)
b_80_100 = sum(1 for v in f1_vals if v >= 0.80)
print(f"  Pages < 20% F1 : {b_0_20}")
print(f"  Pages 20-50% F1: {b_20_50}")
print(f"  Pages 50-80% F1: {b_50_80}")
print(f"  Pages >= 80% F1: {b_80_100}")

# Cumulative drift down the page across all pages
top_dys = [dy for s in page_eval.values() for dy in s["top_third_dy"]]
mid_dys = [dy for s in page_eval.values() for dy in s["mid_third_dy"]]
bot_dys = [dy for s in page_eval.values() for dy in s["bot_third_dy"]]

print(f"\nCumulative Drift Down the Page (Predicted Y - GT Y):")
print(f"  Top 1/3 (y < 0.35)    : mean dy = {np.mean(top_dys):+.5f}, std = {np.std(top_dys):.5f}, mean |dy| = {np.mean(np.abs(top_dys)):.5f}")
print(f"  Middle 1/3 (0.35-0.65): mean dy = {np.mean(mid_dys):+.5f}, std = {np.std(mid_dys):.5f}, mean |dy| = {np.mean(np.abs(mid_dys)):.5f}")
print(f"  Bottom 1/3 (y > 0.65) : mean dy = {np.mean(bot_dys):+.5f}, std = {np.std(bot_dys):.5f}, mean |dy| = {np.mean(np.abs(bot_dys)):.5f}")

# Save detailed stats for further inspection
with open("research/page_eval_stats.json", "w") as f:
    json_ready = {
        p: {
            "total_gradeable": s["total_gradeable"],
            "passing": s["passing"],
            "f1": s["passing"] / s["total_gradeable"] if s["total_gradeable"] > 0 else 0,
            "slot_drift_count": s["slot_drift_count"],
            "severe_drift_count": s["severe_drift_count"],
            "mean_dy": float(np.mean(s["dy_list"])) if s["dy_list"] else 0.0,
            "mean_abs_dy": float(np.mean(np.abs(s["dy_list"]))) if s["dy_list"] else 0.0,
            "top_dy": float(np.mean(s["top_third_dy"])) if s["top_third_dy"] else 0.0,
            "mid_dy": float(np.mean(s["mid_third_dy"])) if s["mid_third_dy"] else 0.0,
            "bot_dy": float(np.mean(s["bot_third_dy"])) if s["bot_third_dy"] else 0.0,
        }
        for p, s in page_eval.items()
    }
    json.dump(json_ready, f, indent=2)
