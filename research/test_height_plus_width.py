#!/usr/bin/env python3
import sys
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
if _REF.exists():
    sys.path.insert(0, str(_REF))
sys.path.insert(0, str(repo_root / "src"))

from tonerhound.benchmark.correlation import FailureRegistry
from extract_bench.test_cases.loader import load_test_case

def main():
    pdf_path = repo_root / "research/data/full/long/real_ftx_full_corrupted.pdf"
    tc = load_test_case(pdf_path)
    creditors = tc.expected_output.get("creditors", [])

    registry_path = repo_root / "experiments/EXP-005-ftx-failure-registry.json"
    registry = FailureRegistry.load(registry_path)

    optimal_heights = {
        "name": 0.01010,
        "address_1": 0.01060,
        "address_2": 0.01110,
        "address_3": 0.01100,
        "address_4": 0.01140,
        "city": 0.00990,
        "country": 0.01020,
        "postal_code": 0.00940,
        "state": 0.00900,
    }

    candidate_citations = []

    for path, rec in registry.records.items():
        if not rec.pred_box:
            continue

        px, py, pw, ph = rec.pred_box
        p_page = rec.pred_page
        fld = rec.field
        row_idx = rec.row_idx

        new_x = px
        new_w = pw
        new_h = ph
        new_y = py

        val = None
        if row_idx is not None and 0 <= row_idx < len(creditors):
            val = creditors[row_idx].get(fld)
        txt = str(val).strip() if val is not None else ""
        L = len(txt)

        # Height adjustments (only for single-line boxes)
        th = optimal_heights.get(fld)
        if th and ph < 0.015:
            yc = py + ph / 2.0
            new_h = th
            new_y = yc - th / 2.0

        # Width and X adjustments
        if fld == "postal_code":
            new_x = px - 0.00110
            new_w = pw + 0.00200
        elif fld == "country":
            new_x = px - 0.00040
            new_w = max(0.010, L * 0.003350 + 0.00050)
        elif fld == "city":
            new_x = px - 0.00010
            new_w = pw + 0.00030
        elif fld == "address_1":
            new_x = px - 0.00045
            new_w = pw + 0.00090
        elif fld == "address_3":
            new_x = px + 0.00010
            new_w = pw + 0.00040
        elif fld == "name":
            new_x = px - 0.00005
            new_w = pw + 0.00010
        elif fld == "address_2":
            new_x = px - 0.00005
            new_w = pw
        elif fld == "address_4":
            new_x = px - 0.00135
            new_w = pw + 0.00110

        candidate_citations.append({
            "field_path": path,
            "page": p_page,
            "bbox": [new_x, new_y, new_w, new_h],
            "confidence": 0.90,
            "source": "tonerhound",
        })

    report = registry.correlate(candidate_citations, target_category="G2", min_purity=0.60)
    print("=" * 80)
    print("COMBINED HEIGHT + WIDTH CORRELATION REPORT")
    print("=" * 80)
    print(report.summary())

if __name__ == "__main__":
    main()
