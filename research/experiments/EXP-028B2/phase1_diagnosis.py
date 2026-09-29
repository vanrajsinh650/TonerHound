"""Phase 1: Verification of Diagnosis for EXP-028B2.

Analyzes EXP-028B1 field_failure_analysis.parquet to:
1. Determine genuine Oracle-to-Production gap fields (where Oracle achieved IoU >= 0.50, but Production failed).
2. Distinguish real failures from false classifications in the parquet.
3. Sample 100-200 fields classified as Category D and verify true underlying failure modes:
   - Repeated values across rows (same column)
   - Repeated values across columns (same row)
   - Identical codes/numbers across the table
   - Wrong row assignment (same X column, different Y)
   - Wrong column assignment (same Y row, different X)
   - Missing row/header context (unanchored)
   - Omitted / ungrounded
4. Sample control categories A, C, G, and F.
5. Provide a rigorous empirical breakdown for the Phase 1 Decision Gate.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
ref_extractbench = root_dir / "research" / "reference" / "ExtractBench" / "src"
if ref_extractbench.exists() and str(ref_extractbench) not in sys.path:
    sys.path.insert(0, str(ref_extractbench))

from extract_bench.test_cases.loader import load_test_case


def iou_xywh(b1: list[float] | tuple[float, float, float, float] | None,
             b2: list[float] | tuple[float, float, float, float] | None) -> float:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def run_phase1_analysis():
    print("=" * 80)
    print("EXP-028B2 PHASE 1: VERIFY THE DIAGNOSIS")
    print("=" * 80)

    parquet_path = root_dir / "research" / "experiments" / "EXP-028B1" / "field_failure_analysis.parquet"
    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist!")
        return

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Total rows in parquet: {len(df)}")
    print("Reported category distribution in parquet:")
    for cat, count in df["failure_category"].value_counts().items():
        print(f"  {cat:<35}: {count:7d} ({count / len(df) * 100:5.2f}%)")

    # Set up prediction paths
    b0_dir = root_dir / "research" / "experiments" / "EXP-028B0" / "predictions" / "tonerhound"
    b1_dir = root_dir / "research" / "experiments" / "EXP-028B1" / "predictions" / "tonerhound"
    data_dir = root_dir / "research" / "data" / "full"

    # Group by document to load predictions efficiently
    docs = sorted(df["document_id"].unique())
    print(f"\nAnalyzing documents across {len(docs)} files...")

    # We will sample 200 documents or test cases across diverse splits
    random.seed(42)
    sample_size = 200

    # Let us sample across categories from df
    cat_d_df = df[df["failure_category"] == "D. table row/column assignment"]
    cat_a_df = df[df["failure_category"] == "A. wrong candidate selected"]
    cat_c_df = df[df["failure_category"] == "C. repeated occurrence"]
    cat_g_df = df[df["failure_category"] == "G. normalization"]
    cat_f_df = df[df["failure_category"] == "F. multiline"]

    sampled_d = cat_d_df.sample(n=min(sample_size, len(cat_d_df)), random_state=42)
    sampled_a = cat_a_df.sample(n=min(50, len(cat_a_df)), random_state=42)
    sampled_c = cat_c_df.sample(n=min(50, len(cat_c_df)), random_state=42)
    sampled_g = cat_g_df.sample(n=min(50, len(cat_g_df)), random_state=42)
    sampled_f = cat_f_df.sample(n=min(50, len(cat_f_df)), random_state=42)

    all_sampled = pd.concat([sampled_d, sampled_a, sampled_c, sampled_g, sampled_f])
    needed_docs = set(all_sampled["document_id"].unique())

    print(f"Loaded {len(all_sampled)} sample fields across {len(needed_docs)} documents.")

    # Cache predictions and test cases for needed docs
    b0_cache: dict[str, dict[str, Any]] = {}
    b1_cache: dict[str, dict[str, Any]] = {}
    tc_rules_cache: dict[str, dict[str, Any]] = {}

    for doc_id in needed_docs:
        p0_path = b0_dir / f"{doc_id}.result.json"
        p1_path = b1_dir / f"{doc_id}.result.json"
        pdf_path = data_dir / f"{doc_id}.pdf"

        if p0_path.exists():
            with open(p0_path) as fp:
                d = json.load(fp)
                b0_cache[doc_id] = {
                    c["field_path"]: c for c in d.get("output", {}).get("field_citations", [])
                }
        if p1_path.exists():
            with open(p1_path) as fp:
                d = json.load(fp)
                b1_cache[doc_id] = {
                    c["field_path"]: c for c in d.get("output", {}).get("field_citations", [])
                }
        if pdf_path.exists():
            try:
                tc = load_test_case(pdf_path)
                tc_rules_cache[doc_id] = {
                    r.field_path: r for r in tc.get_extract_field_rules()
                }
            except Exception as e:
                pass

    # Now verify Category D samples
    print("\n" + "=" * 80)
    print("DETAILED VERIFICATION OF CATEGORY D (TABLE ROW/COLUMN ASSIGNMENT)")
    print("=" * 80)

    d_stats = {
        "total_sampled": len(sampled_d),
        "false_alarm_passed_in_b1": 0,  # Actually passed in B1 (not a failure!)
        "oracle_also_failed": 0,        # Oracle failed too (not in oracle-prod gap)
        "genuine_gap_failure": 0,       # Oracle succeeded (IoU >= 0.50), B1 failed (IoU < 0.50 or missing)
    }

    genuine_d_failures = []

    for _, row in sampled_d.iterrows():
        doc_id = row["document_id"]
        fpath = row["field_path"]
        gold_bbox = json.loads(row["gold_bbox"]) if isinstance(row["gold_bbox"], str) else row["gold_bbox"]
        gold_page = row["gold_page"]

        rule = tc_rules_cache.get(doc_id, {}).get(fpath)
        gold_val = rule.evidence[0].value if rule and rule.evidence else None

        c0 = b0_cache.get(doc_id, {}).get(fpath)
        c1 = b1_cache.get(doc_id, {}).get(fpath)

        b0_iou = iou_xywh(c0.get("bbox") if c0 else None, gold_bbox) if (c0 and c0.get("page") == gold_page) else 0.0
        b1_iou = iou_xywh(c1.get("bbox") if c1 else None, gold_bbox) if (c1 and c1.get("page") == gold_page) else 0.0

        if b1_iou >= 0.50:
            d_stats["false_alarm_passed_in_b1"] += 1
            continue

        if b0_iou < 0.50:
            d_stats["oracle_also_failed"] += 1
            continue

        # This is a genuine oracle-to-production gap failure!
        d_stats["genuine_gap_failure"] += 1

        # Classify the specific mechanism
        b1_page = c1.get("page") if c1 else None
        b1_box = c1.get("bbox") if c1 else None

        fail_mode = "unknown"
        if c1 is None or b1_box is None:
            fail_mode = "omitted_unresolved"
        elif b1_page != gold_page:
            fail_mode = "wrong_page_assignment"
        else:
            gx, gy, gw, gh = gold_bbox
            bx, by, bw, bh = b1_box
            # Check if same column (similar X), different row (different Y)
            x_overlap = max(0.0, min(gx + gw, bx + bw) - max(gx, bx))
            y_overlap = max(0.0, min(gy + gh, by + bh) - max(gy, by))
            
            if x_overlap / max(gw, bw) >= 0.50 and y_overlap / max(gh, bh) < 0.30:
                fail_mode = "wrong_row_same_column"
            elif y_overlap / max(gh, bh) >= 0.50 and x_overlap / max(gw, bw) < 0.30:
                fail_mode = "wrong_column_same_row"
            else:
                fail_mode = "different_cell_or_table"

        genuine_d_failures.append({
            "doc_id": doc_id,
            "field_path": fpath,
            "value": gold_val,
            "gold_page": gold_page,
            "gold_bbox": gold_bbox,
            "b1_page": b1_page,
            "b1_bbox": b1_box,
            "b0_iou": b0_iou,
            "b1_iou": b1_iou,
            "fail_mode": fail_mode,
        })

    print(f"Sampled Category D fields: {d_stats['total_sampled']}")
    print(f"  1. False Alarms (Actually PASSED in B1 with IoU >= 0.50): {d_stats['false_alarm_passed_in_b1']} ({d_stats['false_alarm_passed_in_b1']/d_stats['total_sampled']*100:.1f}%)")
    print(f"  2. Oracle Also Failed (IoU < 0.50 in Oracle too):         {d_stats['oracle_also_failed']} ({d_stats['oracle_also_failed']/d_stats['total_sampled']*100:.1f}%)")
    print(f"  3. Genuine Oracle-to-Prod Gap Failures (Oracle>=0.50, B1<0.50): {d_stats['genuine_gap_failure']} ({d_stats['genuine_gap_failure']/d_stats['total_sampled']*100:.1f}%)")

    # Now break down the genuine Category D failures
    print("\nBreakdown of Genuine Category D Failures:")
    d_mode_counts = defaultdict(int)
    for f in genuine_d_failures:
        d_mode_counts[f["fail_mode"]] += 1
    for mode, count in sorted(d_mode_counts.items(), key=lambda x: -x[1]):
        pct = count / len(genuine_d_failures) * 100 if genuine_d_failures else 0.0
        print(f"  {mode:<30}: {count:4d} ({pct:5.1f}%)")

    print("\nRepresentative Examples of Genuine Category D Failures:")
    for i, ex in enumerate(genuine_d_failures[:8], 1):
        print(f"  [{i}] Doc: {ex['doc_id']}")
        print(f"      Field: {ex['field_path']}")
        print(f"      Value: {ex['value']!r}")
        print(f"      Mode:  {ex['fail_mode']}")
        print(f"      Gold:  page={ex['gold_page']}, bbox={[round(x, 4) for x in ex['gold_bbox']]}")
        print(f"      B1:    page={ex['b1_page']}, bbox={[round(x, 4) for x in ex['b1_bbox']] if ex['b1_bbox'] else None} (IoU={ex['b1_iou']:.3f})")
        print()

    # Now verify Control Categories: A, C, G, F
    print("=" * 80)
    print("VERIFICATION OF CONTROL CATEGORIES (A, C, G, F)")
    print("=" * 80)

    for cat_name, cat_sample in [
        ("A. wrong candidate selected", sampled_a),
        ("C. repeated occurrence", sampled_c),
        ("G. normalization", sampled_g),
        ("F. multiline", sampled_f),
    ]:
        passed = 0
        ora_fail = 0
        gap_fail = 0
        for _, row in cat_sample.iterrows():
            doc_id = row["document_id"]
            fpath = row["field_path"]
            gold_bbox = json.loads(row["gold_bbox"]) if isinstance(row["gold_bbox"], str) else row["gold_bbox"]
            gold_page = row["gold_page"]
            c0 = b0_cache.get(doc_id, {}).get(fpath)
            c1 = b1_cache.get(doc_id, {}).get(fpath)

            b0_iou = iou_xywh(c0.get("bbox") if c0 else None, gold_bbox) if (c0 and c0.get("page") == gold_page) else 0.0
            b1_iou = iou_xywh(c1.get("bbox") if c1 else None, gold_bbox) if (c1 and c1.get("page") == gold_page) else 0.0

            if b1_iou >= 0.50:
                passed += 1
            elif b0_iou < 0.50:
                ora_fail += 1
            else:
                gap_fail += 1

        total = len(cat_sample)
        print(f"Category {cat_name} (N={total}):")
        print(f"  - Passed in B1 (False Alarm):    {passed:3d} ({passed/total*100:5.1f}%)")
        print(f"  - Oracle Also Failed:            {ora_fail:3d} ({ora_fail/total*100:5.1f}%)")
        print(f"  - Genuine Oracle-Prod Gap:       {gap_fail:3d} ({gap_fail/total*100:5.1f}%)")

    # Now compute true population estimates across the full benchmark
    print("\n" + "=" * 80)
    print("POPULATION-LEVEL AUDIT: TRUE EXTENT OF TABLE FAILURES IN BENCHMARK")
    print("=" * 80)

    # Let's count how many fields in the entire 370-document benchmark
    # have Oracle IoU >= 0.50 and Production IoU < 0.50
    # To do this safely and quickly, let us inspect a representative cross-section of 50 documents
    cross_docs = sorted(list(docs))[:50]
    total_gradeable = 0
    total_oracle_ok = 0
    total_prod_ok = 0
    total_gap = 0
    table_gap = 0
    scalar_gap = 0

    for doc_id in cross_docs:
        p0_path = b0_dir / f"{doc_id}.result.json"
        p1_path = b1_dir / f"{doc_id}.result.json"
        pdf_path = data_dir / f"{doc_id}.pdf"
        if not p0_path.exists() or not p1_path.exists() or not pdf_path.exists():
            continue

        with open(p0_path) as fp0, open(p1_path) as fp1:
            d0 = json.load(fp0)
            d1 = json.load(fp1)

        c0_map = {c["field_path"]: c for c in d0.get("output", {}).get("field_citations", [])}
        c1_map = {c["field_path"]: c for c in d1.get("output", {}).get("field_citations", [])}

        tc = load_test_case(pdf_path)
        for r in tc.get_extract_field_rules():
            if not r.evidence or r.evidence[0].page is None or r.evidence[0].bbox is None:
                continue
            total_gradeable += 1
            gp = r.evidence[0].page
            gb = r.evidence[0].bbox
            fp_path = r.field_path

            c0 = c0_map.get(fp_path)
            c1 = c1_map.get(fp_path)

            iou0 = iou_xywh(c0.get("bbox") if c0 else None, gb) if (c0 and c0.get("page") == gp) else 0.0
            iou1 = iou_xywh(c1.get("bbox") if c1 else None, gb) if (c1 and c1.get("page") == gp) else 0.0

            if iou0 >= 0.50:
                total_oracle_ok += 1
            if iou1 >= 0.50:
                total_prod_ok += 1
            if iou0 >= 0.50 and iou1 < 0.50:
                total_gap += 1
                if "[" in fp_path or "table" in fp_path.lower():
                    table_gap += 1
                else:
                    scalar_gap += 1

    print(f"In 50-document cross section:")
    print(f"  Total gradeable fields: {total_gradeable}")
    print(f"  Oracle OK (IoU>=0.50):  {total_oracle_ok} ({total_oracle_ok / total_gradeable * 100:.2f}%)")
    print(f"  Prod OK (IoU>=0.50):    {total_prod_ok} ({total_prod_ok / total_gradeable * 100:.2f}%)")
    print(f"  Genuine Gap Fields:     {total_gap}")
    if total_gap > 0:
        print(f"    - Table fields (`[` or `table`): {table_gap} ({table_gap / total_gap * 100:.2f}%)")
        print(f"    - Scalar fields:                {scalar_gap} ({scalar_gap / total_gap * 100:.2f}%)")

    # Decision Gate Verdict
    print("\n" + "=" * 80)
    print("PHASE 1 DECISION GATE SUMMARY")
    print("=" * 80)
    table_share = (table_gap / total_gap * 100) if total_gap > 0 else 0.0
    print(f"Table Field Share of Real Oracle-to-Production Gap: {table_share:.2f}%")
    if table_share >= 80.0:
        print("DECISION GATE VERDICT: CONFIRMED.")
        print("Table row/column assignment is indeed the dominant real failure mode.")
        print("Proceed to Phase 2: Build Structural Context Signals.")
    else:
        print("DECISION GATE VERDICT: REJECTED.")
        print("Table failures are not the dominant mode; re-evaluate failure taxonomy.")
    print("=" * 80)


if __name__ == "__main__":
    run_phase1_analysis()
