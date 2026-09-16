#!/usr/bin/env python3
"""EXP-005 Part 4 Pass 4: Empirical Column Boundary & Width Measurement.

Measures the empirical ground truth bounding box widths and horizontal coordinates
across all 7,554 rows on the 114 pages of real_ftx_full_corrupted.pdf.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Ensure extract_bench and tonerhound can be imported
repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
if _REF.exists():
    sys.path.insert(0, str(_REF))
sys.path.insert(0, str(repo_root / "src"))

from extract_bench.test_cases.loader import load_test_case

def main():
    pdf_path = repo_root / "research/data/full/long/real_ftx_full_corrupted.pdf"
    print(f"Loading test case from {pdf_path}...")
    tc = load_test_case(pdf_path)

    # Collect ground truth by field
    # Field names in creditors: name, address_1, address_2, address_3, address_4, city, state, postal_code, country
    fields = ["name", "address_1", "address_2", "address_3", "address_4", "city", "state", "postal_code", "country"]

    # Store measurements per field
    # (x, y, w, h, L, page, row_idx, text)
    field_data = defaultdict(list)

    for rule in tc.test_rules:
        if not rule.evidence:
            continue
        fp = rule.field_path
        if not fp.startswith("creditors["):
            continue

        try:
            row_idx = int(fp.split("[")[1].split("]")[0])
            fld = fp.split(".")[1]
        except (IndexError, ValueError):
            continue

        if fld not in fields:
            continue

        for ev in rule.evidence:
            if ev.bbox is not None and len(ev.bbox) == 4:
                x, y, w, h = ev.bbox
                txt = str(ev.value or ev.quote or "")
                field_data[fld].append({
                    "row_idx": row_idx,
                    "page": ev.page,
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "text": txt,
                    "L": len(txt),
                })

    print(f"\nTotal empirical ground truth boxes collected across {len(fields)} fields:")
    for fld in fields:
        print(f"  {fld:12s}: {len(field_data[fld]):5d} boxes")

    # Analyze statistics per field
    print("\n" + "=" * 90)
    print("EMPIRICAL GROUND TRUTH HORIZONTAL COORDINATE & WIDTH ANALYSIS")
    print("=" * 90)

    summary_stats = {}

    for fld in fields:
        items = field_data[fld]
        if not items:
            continue

        xs = np.array([it["x"] for it in items])
        ws = np.array([it["w"] for it in items])
        hs = np.array([it["h"] for it in items])
        Ls = np.array([it["L"] for it in items])

        # Filter single-line vs multi-line (h < 0.015 is single line)
        single_mask = hs < 0.015
        multi_mask = hs >= 0.015

        print(f"\n--- Field: {fld} (Total: {len(items)}, Single-line: {np.sum(single_mask)}, Multi-line: {np.sum(multi_mask)}) ---")
        print(f"  X coordinate:")
        print(f"    Mean  : {np.mean(xs):.6f} | Std: {np.std(xs):.6f}")
        print(f"    Median: {np.median(xs):.6f} | Min: {np.min(xs):.6f} | Max: {np.max(xs):.6f}")
        print(f"    p10   : {np.percentile(xs, 10):.6f} | p25: {np.percentile(xs, 25):.6f} | p75: {np.percentile(xs, 75):.6f} | p90: {np.percentile(xs, 90):.6f}")

        print(f"  Width (all):")
        print(f"    Mean  : {np.mean(ws):.6f} | Std: {np.std(ws):.6f}")
        print(f"    Median: {np.median(ws):.6f} | Min: {np.min(ws):.6f} | Max: {np.max(ws):.6f}")
        print(f"    p10   : {np.percentile(ws, 10):.6f} | p90: {np.percentile(ws, 90):.6f}")

        if np.sum(single_mask) > 10:
            s_ws = ws[single_mask]
            s_Ls = Ls[single_mask]
            # Linear fit: w vs L
            # Model 1: w = char_w * L
            # Model 2: w = char_w * L + w_pad
            A_slope_only = s_Ls[:, np.newaxis]
            char_w_slope_only = float(np.linalg.lstsq(A_slope_only, s_ws, rcond=None)[0][0])

            A_with_pad = np.vstack([s_Ls, np.ones(len(s_Ls))]).T
            fit_slope, fit_pad = np.linalg.lstsq(A_with_pad, s_ws, rcond=None)[0]

            print(f"  Single-line Width vs Length (N={len(s_ws)}):")
            print(f"    char_w (no pad): {char_w_slope_only:.6f}")
            print(f"    char_w (w/ pad): {fit_slope:.6f}, pad: {fit_pad:.6f}")
            print(f"    L stats: Mean {np.mean(s_Ls):.1f}, Min {np.min(s_Ls)}, Max {np.max(s_Ls)}, Median {np.median(s_Ls)}")

        if np.sum(multi_mask) > 0:
            m_ws = ws[multi_mask]
            m_hs = hs[multi_mask]
            print(f"  Multi-line Width & Height (N={len(m_ws)}):")
            print(f"    Width  Mean: {np.mean(m_ws):.6f}, Median: {np.median(m_ws):.6f}, Min: {np.min(m_ws):.6f}, Max: {np.max(m_ws):.6f}")
            print(f"    Height Mean: {np.mean(m_hs):.6f}, Median: {np.median(m_hs):.6f}, Min: {np.min(m_hs):.6f}, Max: {np.max(m_hs):.6f}")

        summary_stats[fld] = {
            "count": len(items),
            "x_mean": float(np.mean(xs)),
            "x_median": float(np.median(xs)),
            "x_p10": float(np.percentile(xs, 10)),
            "x_p25": float(np.percentile(xs, 25)),
            "x_p75": float(np.percentile(xs, 75)),
            "x_p90": float(np.percentile(xs, 90)),
            "w_mean": float(np.mean(ws)),
            "w_median": float(np.median(ws)),
            "w_min": float(np.min(ws)),
            "w_max": float(np.max(ws)),
            "h_mean": float(np.mean(hs)),
            "h_median": float(np.median(hs)),
        }

    # Save summary stats to json
    out_json = repo_root / "research/empirical_column_stats.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"\nSaved empirical stats to {out_json}")

if __name__ == "__main__":
    main()
