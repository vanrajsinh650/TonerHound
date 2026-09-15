# EXP-005 Tax Form Error Analysis Report

**Focus Cluster**: IRS Form 1040 Tax Returns (`cabrera-2023`, `cabrera-2022`, `becerra-2024`, `bianco-2022`, `bar-lev-2022`)  
**Baseline Metric**: 12.3% – 17.2% Word Grounding F1 (Mean: 14.1%)  
**Evaluator**: Official ExtractBench Unified Evidence Metric  

---

## 1. Executive Summary

Across the 5 representative tax return filings in the frozen local benchmark, there are 259 non-None extracted fields with ground-truth evidence. The current system grounds only 93 correctly, resulting in **202 critical field failures**.

Empirical classification of all 202 failed fields according to categories A through L reveals that three failure modes dominate over 75% of all errors:

1. **Category D: Checkbox Confusion (45.5% of failures, 92 fields)**: The resolver matches boolean `False` / `True` against arbitrary empty box glyphs on the page without tying the checkbox to its specific question/line label. All 19 checkboxes on Form 1040 collapse onto a single coordinate.
2. **Category C: Duplicated Numbers on Later Schedules (22.3% of failures, 45 fields)**: Form 1040 summary lines (e.g., Line 9 Total Income, Line 10 Adjustments, Line 11 AGI) repeat across later schedules and California 540 state returns. Without Form 1040 page constraints, the resolver matches numbers on Pages 4, 8, 17, and 24.
3. **Category A: Correct Number, Wrong Line on Same Page (7.4% of failures, 15 fields)**: When identical dollar amounts occur on multiple lines (e.g., Line 1a and Line 1z both $125,500; Lines 25a, 25d, and 33 all $15,292; Lines 16, 18, 22, 24 all $29,881), the resolver arbitrarily assigns all fields to the first or last occurrence.

---

## 2. Failure Category Distribution

| Category | Description | Count | Percentage | Primary Impact |
| :--- | :--- | :---: | :---: | :--- |
| **D** | Checkbox confusion (unanchored box tokens) | 92 | **45.5%** | 19 checkboxes per Form 1040 |
| **C** | Duplicated number elsewhere in document | 45 | **22.3%** | Line amounts resolving to later schedules/state returns |
| **E** | Label / field anchor not detected by resolver | 18 | **8.9%** | Multi-token names (`CABRERA`, `BAR-LEV`) marked ungrounded |
| **A** | Correct number, wrong line on same page | 15 | **7.4%** | Duplicate lines on Form 1040 (e.g. Lines 1a vs 1z, 25a vs 33) |
| **K** | OCR / text extraction missing token | 14 | **6.9%** | Right column monetary boxes omitted by default PSM 3 |
| **I** | Page confusion for names/entities | 10 | **5.0%** | Taxpayer/spouse name resolving to later schedules |
| **F** | Label/value spatial distance & IoU mismatch | 6 | **3.0%** | Multi-token bounding box alignment |
| **H** | Section confusion (Designee vs Preparer) | 2 | **1.0%** | AYOUB CPA mapped to preparer instead of designee |
| **Total** | | **202** | **100.0%** | |

---

## 3. Concrete Benchmark Failure Case Studies

### Case 1: The Checkbox Collapse (Category D — 45.5%)
- **Document**: `medium/cabrera-2023` (Form 1040 Page 1)
- **Extracted Fields**:
  - `presidential_campaign_you_box`: Expected `False` next to Presidential Campaign Fund label at `y=0.206, x=0.812`.
  - `someone_can_claim_you_box`: Expected `False` next to Standard Deduction Age/Blindness label at `y=0.339, x=0.264`.
  - `you_born_before_jan2_1959_box`: Expected `False` next to Age label at `y=0.385, x=0.182`.
  - `you_blind_box`: Expected `False` next to Blindness label at `y=0.386, x=0.392`.
- **Observed Behavior**: The resolver predicted `[0.8098, 0.2058, 0.022, 0.018]` for **all four distinct fields**!
- **Root Cause**: `find_boolean_candidates` searched the page for glyphs representing `False` (`"[ ]"` / Dingbat / OCR artifact), found the first available empty box token at `y=0.206`, and assigned it to every boolean field on the page without inspecting the surrounding horizontal line text!

### Case 2: Duplicated Summary Number Straying to Later Pages (Category C — 22.3%)
- **Document**: `medium/cabrera-2023` (28 pages total)
- **Extracted Field**: `line_9_total_income` (`val = 250658`)
- **Ground Truth**: Form 1040 Page 1, Line 9, bounding box `[0.8322, 0.7733, 0.0902, 0.0153]`.
- **Observed Behavior**: Grounded on **Page 24**, bounding box `[0.8633, 0.6760, 0.0850, 0.0180]`.
- **Root Cause**: In multi-page tax filings, Total Income is carried forward to California state returns and subsidiary statements. Because `matcher.py` searches all pages globally when `page_hint is None`, the match on page 24 tied or outscored the Form 1040 line. Form 1040 fields (`line_1a` through `line_37`) must be structurally scoped to the Form 1040 page region.

### Case 3: Intra-Page Duplicate Line Collapse (Category A — 7.4%)
- **Document**: `medium/cabrera-2023` (Form 1040 Page 1 & 2)
- **Extracted Fields**:
  - `line_1a_total_w2_wages`: `val = 125500` (Line 1a at `y=0.498`)
  - `line_1z_total_wages`: `val = 125500` (Line 1z at `y=0.633`)
  - `line_25a_withholding_w2`: `val = 15292` (Line 25a at `y=0.201`)
  - `line_25d_total_withholding`: `val = 15292` (Line 25d at `y=0.254`)
  - `line_33_total_payments`: `val = 15292` (Line 33 at `y=0.374`)
- **Observed Behavior**:
  - Both Line 1a and Line 1z were mapped to Line 1z at `y=0.635`.
  - Lines 25a, 25d, and 33 were all mapped to Line 33 at `y=0.375`.
- **Root Cause**: The resolver treats numeric matching as a purely scalar value lookup. It ignores the explicit line prefix (`"1a"`, `"1z"`, `"25a"`, `"25d"`, `"33"`) present in the line's visual row.

---

## 4. Architectural Prescription: Form-Aware Structural Grounding

To systematically eliminate these failure modes, we must implement:

1. **Form 1040 Page Detection & Scoping**:
   - Detect the exact pages containing IRS Form 1040 (typically Page 1 and Page 2 in standard tax return filings).
   - Form 1040 schema fields (`taxpayer_*`, `spouse_*`, `line_1a` through `line_15` on Page 1; `line_16` through `line_37`, `third_party_designee_*`, `preparer_*` on Page 2) must be anchored strictly to their corresponding Form 1040 page!
2. **Line-Number Anchor Extraction**:
   - Parse the line identifier directly from the field path (e.g. `line_1a_...` &rarr; `"1a"`; `line_25a_...` &rarr; `"25a"`; `line_9_...` &rarr; `"9"`).
   - Locate the visual line on the form that starts with or contains that line label at $x \in [0.05, 0.25]$.
   - Restrict the candidate value search to the horizontal stripe $[y - 0.015, y + 0.015]$ and right column $x \in [0.70, 0.98]$.
3. **Spatially Anchored Checkbox Resolution**:
   - For every boolean / checkbox field, identify its anchor text (e.g., `you_blind_box` &rarr; `"blind"`; `presidential_campaign_you_box` &rarr; `"presidential election campaign"`, `"you"`; `digital_assets_yes` &rarr; `"digital assets"`, `"yes"`).
   - Locate the label line and select the checkbox glyph immediately adjacent to that label ($|y_{\text{box}} - y_{\text{label}}| \le 0.015$ and adjacent horizontally).
4. **Taxpayer vs. Spouse Entity Disambiguation**:
   - Taxpayer name & SSN are on row 1 ($y \approx 0.09\text{--}0.11$).
   - Spouse name & SSN are on row 2 ($y \approx 0.12\text{--}0.14$).
   - Distinct y-coordinate bands prevent name collisions.
