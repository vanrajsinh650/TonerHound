#!/usr/bin/env python3
"""EXP-005 Part 4 Pass 4: Targeted State Column Tilt & Width Sweep.

Tests data-derived left margin alignment, tilt compensation factor (alpha),
and width for the state column across all 2,576 state citations.
"""

import json
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import iou_xywh
from tonerhound.document.index import DocumentIndex

def main():
    reg_path = repo_root / "experiments/EXP-005-ftx-failure-registry.json"
    with open(reg_path) as f:
        registry = json.load(f)

    pdf_path = repo_root / "research/data/full/long/real_ftx_full_corrupted.pdf"
    idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True)

    page_skews = {}
    for p_num in range(1, len(idx.pages) + 1):
        page_obj = idx.get_page(p_num)
        page_slope = 0.0
        if page_obj and page_obj.tokens:
            left_toks = [t for t in page_obj.tokens if t.bbox.y < 0.04 and t.bbox.x < 0.40 and any(k in t.text.lower() for k in ("case", "22-11068", "doc", "form", "page"))]
            right_toks = [t for t in page_obj.tokens if t.bbox.y < 0.04 and t.bbox.x > 0.60 and any(k in t.text.lower() for k in ("page", "114", "of", "filed"))]
            if left_toks and right_toks:
                lt = left_toks[0]
                rt = right_toks[-1]
                dx = (rt.bbox.x + rt.bbox.width / 2.0) - (lt.bbox.x + lt.bbox.width / 2.0)
                dy = (rt.bbox.y + rt.bbox.height / 2.0) - (lt.bbox.y + lt.bbox.height / 2.0)
                if abs(dx) > 0.1:
                    page_slope = dy / dx
        page_skews[p_num] = page_slope

    state_items = [v for v in registry.values() if v.get("field") == "state" and v.get("pred_box")]
    baseline_pass = sum(1 for v in state_items if v["status"] == "PASS")
    baseline_g2 = sum(1 for v in state_items if v["category"] == "G2")
    print(f"State: {len(state_items)} citations. Baseline Passing: {baseline_pass}, G2 Failures: {baseline_g2}")

    # Sweep base_x, base_w, alpha
    # baseline was base_x = 0.7325, base_w = 0.0105, alpha = 0.50
    best_zero_reg = None
    best_gain = -999

    pareto = []

    for base_x in np.linspace(0.7320, 0.7345, 26):
        for base_w in np.linspace(0.0075, 0.0105, 31):
            for alpha in [0.0, 0.3, 0.5, 0.7, 0.85, 1.0, 1.15]:
                fixed = 0
                g2_fixed = 0
                reg = 0
                for v in state_items:
                    px, py, pw, ph = v["pred_box"]
                    p_page = v["pred_page"]
                    slope = page_skews.get(p_page, 0.0)
                    cy = py + ph / 2.0

                    new_x = base_x - alpha * slope * (cy - 0.50)
                    new_w = base_w
                    new_box = [new_x, py, new_w, ph]
                    new_iou = iou_xywh(v["gt_box"], new_box)

                    was_pass = (v["status"] == "PASS")
                    is_pass = (new_iou >= 0.50)

                    if is_pass and not was_pass:
                        fixed += 1
                        if v["category"] == "G2":
                            g2_fixed += 1
                    elif was_pass and not is_pass:
                        reg += 1

                net = fixed - reg
                if reg == 0 and net > best_gain:
                    best_gain = net
                    best_zero_reg = (base_x, base_w, alpha, fixed, reg, g2_fixed, net)

                if net > 0 and reg <= 10:
                    pareto.append((reg, net, fixed, g2_fixed, base_x, base_w, alpha))

    print("\n--- Best Zero-Regression Configuration for State ---")
    if best_zero_reg:
        bx, bw, a, f, r, g2, n = best_zero_reg
        print(f"  base_x={bx:.5f}, base_w={bw:.5f}, alpha={a:.2f}")
        print(f"  Fixed: {f} (G2: {g2}), Regressions: {r}, Net Gain: +{n}")
    else:
        print("  None found with exactly zero regressions.")

    print("\n--- Pareto Frontier for State (Regressions vs Net Gain) ---")
    for max_r in [0, 1, 2, 5, 10]:
        cands = [p for p in pareto if p[0] <= max_r]
        if cands:
            best_c = max(cands, key=lambda x: x[1])
            print(f"  Max Reg <= {max_r:2d}: Reg={best_c[0]:2d}, Net={best_c[1]:+3d}, Fixed={best_c[2]:3d} (G2={best_c[3]:3d}) | base_x={best_c[4]:.5f}, base_w={best_c[5]:.5f}, alpha={best_c[6]:.2f}")

if __name__ == "__main__":
    main()
