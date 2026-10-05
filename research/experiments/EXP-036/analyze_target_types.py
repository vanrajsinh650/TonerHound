"""EXP-036C: Initial target geometry and document family distribution analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import pymupdf as fitz

repo_root = Path(__file__).resolve().parent.parent.parent.parent

with open(repo_root / "research" / "experiments" / "EXP-036" / "checkbox_inventory.json") as f:
    records = json.load(f)["inventory"]

print(f"Loaded {len(records)} records from inventory.")

# Categorize targets into:
# 1. SQUARE_CHECKBOX (w/h in [0.6, 1.6], w <= 0.04, h <= 0.035, area <= 0.001)
# 2. RADIO_OR_SMALL_FORM_BOX (w/h in [0.4, 2.5], area <= 0.002, not signature)
# 3. SIGNATURE_OR_HANDWRITTEN_MARK (signature in field_path or area >= 0.005 or w/h >= 3.0)
# 4. OTHER_FORM_MARK (intermediate boxes, rectangular checkboxes)
# 5. AMBIGUOUS (non-standard geometries or heavily cropped regions)

categories = defaultdict(list)
doc_families = defaultdict(lambda: defaultdict(int))
page_checkbox_counts = defaultdict(int)
dimensions_px = defaultdict(list)

for r in records:
    doc_id = r["document_id"]
    split, doc_name = doc_id.split("/", 1)
    
    # Identify document family
    if doc_name.startswith("W14") or "W-14" in doc_name or "W14" in doc_name:
        fam = "TEXAS_RRC_W14"
    elif doc_name.startswith("H-12") or "H-12" in doc_name or "H12" in doc_name:
        fam = "TEXAS_RRC_H12"
    elif any(k in doc_name for k in ("W-1", "W-2", "P4", "P-4", "2A")):
        fam = "TEXAS_RRC_OTHER"
    elif any(k in doc_name for k in ("arif", "bar-lev", "becerra", "1040", "8879", "8949", "8960", "8812")):
        fam = "IRS_TAX_RETURNS"
    elif any(k in doc_name for k in ("credit", "ishares", "ofac", "oklahoma", "sm0801", "sec_")):
        fam = "SEC_FINANCIAL_REGULATORY"
    else:
        fam = "OTHER_DOCUMENTS"

    gb = r["gold_bbox"]
    w, h = gb[2], gb[3]
    ar = w / max(1e-5, h)
    area = w * h
    fp = r["field_path"].lower()

    # Typical pixel dimensions at 300 DPI (approx 8.5 x 11 inch = 2550 x 3300 px)
    w_px = w * 2550
    h_px = h * 3300

    # Categorization rule
    if "signature" in fp or "signed" in fp or (area >= 0.004 and ar >= 2.5):
        obj_type = "SIGNATURE_OR_HANDWRITTEN_MARK"
    elif 0.65 <= ar <= 1.55 and w <= 0.035 and h <= 0.030 and area <= 0.0008:
        obj_type = "SQUARE_CHECKBOX"
    elif 0.40 <= ar <= 2.50 and w <= 0.045 and h <= 0.035 and area <= 0.0015:
        obj_type = "RADIO_OR_SMALL_FORM_BOX"
    elif area > 0.003:
        obj_type = "AMBIGUOUS"
    else:
        obj_type = "OTHER_FORM_MARK"

    categories[obj_type].append(r)
    doc_families[fam][obj_type] += 1
    page_key = f"{doc_id}:p{r['gold_page']}"
    if obj_type in ("SQUARE_CHECKBOX", "RADIO_OR_SMALL_FORM_BOX", "OTHER_FORM_MARK"):
        page_checkbox_counts[page_key] += 1
    dimensions_px[obj_type].append((w_px, h_px, ar))

print("\n" + "=" * 75)
print(f"{'Visual Target Category':<35} | {'Count':<8} | {'% of Total':<10}")
print("-" * 75)
for cat, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
    pct = len(items) / len(records) * 100
    print(f"{cat:<35} | {len(items):>8} | {pct:>8.2f}%")
print("=" * 75)

print("\nDocument Family Distribution:")
for fam, cat_map in sorted(doc_families.items(), key=lambda x: sum(x[1].values()), reverse=True):
    tot = sum(cat_map.values())
    print(f"  {fam:<28} (Total: {tot:>3}) -> {dict(cat_map)}")

print("\nPages with Repeated Checkbox Structure (>= 3 boxes on page):")
repeated_pages = {k: v for k, v in page_checkbox_counts.items() if v >= 3}
print(f"  Total pages with >= 3 checkboxes: {len(repeated_pages)} pages (covering {sum(repeated_pages.values())} fields)")
for pk, cnt in sorted(repeated_pages.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    {pk}: {cnt} checkboxes")

print("\nTypical Pixel Dimensions at 300 DPI (median width x median height, aspect ratio):")
import numpy as np
for cat, dims in dimensions_px.items():
    ws = [d[0] for d in dims]
    hs = [d[1] for d in dims]
    ars = [d[2] for d in dims]
    print(f"  {cat:<35}: {np.median(ws):.1f} x {np.median(hs):.1f} px (aspect ratio {np.median(ars):.2f})")
