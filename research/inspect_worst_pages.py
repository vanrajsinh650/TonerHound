import json
from pathlib import Path

with open("research/page_eval_stats.json") as f:
    stats = json.load(f)

# Sort by F1 ascending
sorted_pages = sorted(stats.items(), key=lambda x: x[1]["f1"])
print("15 Worst Performing Pages:")
for p, s in sorted_pages[:15]:
    f1 = s["f1"] * 100
    pass_cnt = s["passing"]
    tot = s["total_gradeable"]
    drift_cnt = s["slot_drift_count"]
    mean_dy = s["mean_dy"]
    top_dy = s["top_dy"]
    mid_dy = s["mid_dy"]
    bot_dy = s["bot_dy"]
    print(f"Page {p:3s}: F1={f1:5.2f}%, Passing={pass_cnt:3d}/{tot:3d}, DriftCount={drift_cnt:3d}, MeanDy={mean_dy:+.5f}, TopDy={top_dy:+.5f}, MidDy={mid_dy:+.5f}, BotDy={bot_dy:+.5f}")

print("\n15 Best Performing Pages:")
for p, s in sorted_pages[-15:]:
    f1 = s["f1"] * 100
    pass_cnt = s["passing"]
    tot = s["total_gradeable"]
    drift_cnt = s["slot_drift_count"]
    mean_dy = s["mean_dy"]
    top_dy = s["top_dy"]
    mid_dy = s["mid_dy"]
    bot_dy = s["bot_dy"]
    print(f"Page {p:3s}: F1={f1:5.2f}%, Passing={pass_cnt:3d}/{tot:3d}, DriftCount={drift_cnt:3d}, MeanDy={mean_dy:+.5f}, TopDy={top_dy:+.5f}, MidDy={mid_dy:+.5f}, BotDy={bot_dy:+.5f}")
