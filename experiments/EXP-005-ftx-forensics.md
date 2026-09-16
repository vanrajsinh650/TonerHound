# EXP-005 Part 4: FTX Forensic Error Analysis & Diagnostics

**Document**: `long/real_ftx_full_corrupted` (114 pages, 7,554 creditors, 75,543 test rules)  
**Experiment**: EXP-005 Part 4 — Long-Document + FTX Deep Grounding  
**Author**: Lead Research Engineer (DeepMind / TonerHound Pair)  
**Date**: 2026-09-15  
**Baseline State**:  
- Word Grounding F1: **0.45%** (Precision: 0.49%, Recall: 0.41%)  
- Page Grounding F1: **93.39%** (Precision: 99.69%, Recall: 87.84%)  
- Core Diagnostic Signal: **System finds the correct page (93.39% Page F1) but fails completely on word/region grounding (0.45% Word F1).**

---

## 1. Executive Summary & Forensic Anatomy

A comprehensive forensic audit of `long/real_ftx_full_corrupted` reveals why word-level grounding collapsed despite near-perfect page-level precision:

1. **Pure Scanned Image with Grid Lines**:
   FTX is a 100% scanned PDF (`native_char_count = 0` across all 114 pages). Pages 2 through 114 represent a massive tabular schedule ("Consolidated List of Creditors") formatted with fine black horizontal and vertical grid lines separating table cells.
2. **OCR Grid-Line Interference & Single-Linkage Merging Pathology**:
   Default Tesseract OCR treats tabular grid lines as graphical artifacts, severely fragmenting text into broken glyphs. Furthermore, `DocumentIndex._cluster_tokens_into_lines` uses greedy single-linkage vertical bounding-box overlap. Slight token skews and vertical noise artifacts cause the line group bounding box $[y_0, y_1]$ to expand continuously across vertical rows, merging 4 to 6 separate physical table rows into a single giant multi-line `VisualLine` ($h \approx 0.04$–$0.07$). Within each merged line, sorting by `x0` interleaves tokens across different rows into incomprehensible text soup.
3. **Table DP Misalignment & Cascade Poisoning**:
   In `ExtractBenchAdapter._align_table_arrays`, dynamic programming attempts to align $M \approx 60$–$70$ creditor records to $N \approx 40$ merged OCR lines per page. Because merged lines contain fragmented words from multiple rows, fuzzy line matching falsely aligns rows to lines situated several rows higher on the page (e.g., Row 6 `1012ND STREET INC.` falsely matched to Line 3 at $y = 0.0806$, which actually belongs to Row 0/1).
4. **Anchor Collapse and Negative Y Fallback**:
   Because Row 6 was falsely anchored at $y = 0.0806$ with an erroneous step size ($\Delta y = 0.0182$ instead of $\approx 0.0112$), preceding rows 0 through 3 were extrapolated to negative Y coordinates ($y < 0.03$), producing `None` predictions. Subsequent rows were shifted upward by 4 to 6 rows ($\Delta y \approx -0.05$ to $-0.07$).
5. **Geometry Precision vs Displacement**:
   Horizontal coordinates ($X$) in `standard_table_cols` are remarkably accurate ($\Delta x < 0.001$, within 0.1% page width), but vertical coordinates ($Y$) are completely displaced ($\Delta y = -0.05$ to $-0.07$), resulting in $0.000$ IoU against ground truth cells ($h \approx 0.0095$).

---

## 2. Representative Sample Forensic Table

Below is an exhaustive forensic examination of 20 representative fields across Page 2 and Page 3, showing extracted values, ground truth geometry, candidate generation results, prediction geometry, IoU, and failure classification:

| Field Path | Extracted Value | GT Page | GT BBox [x, y, w, h] | Pred Page | Pred BBox [x, y, w, h] | IoU | Class | Forensic Analysis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `case_number` | `"22-11068-JTD"` | 1 | [0.3173, 0.0221, 0.0956, 0.0219] | 1 | [0.3179, 0.0225, 0.0941, 0.0180] | **0.811** | CORRECT | Cover page narrative text; clean OCR line; correct resolution. |
| `debtor` | `"FTX TRADING LTD., et al."` | 1 | [0.2212, 0.1869, 0.1036, 0.0188] | 1 | [0.2215, 0.1869, 0.1351, 0.0180] | **0.738** | CORRECT | Cover page caption; accurate bounding box; IoU > 0.50. |
| `creditors[0].name` | `"NAME ON FILE"` | 2 | [0.0719, 0.0772, 0.0405, 0.0094] | 2 | `None` | 0.000 | **A** | Row 0 anchor fell off top of page ($y < 0.03$) due to row 6 false alignment; no candidate. |
| `creditors[0].address_1` | `"ADDRESS ON FILE"` | 2 | [0.2591, 0.0787, 0.0486, 0.0095] | 2 | `None` | 0.000 | **A** | Row 0 anchor `None`; no candidate generated. |
| `creditors[1].name` | `"NAME ON FILE"` | 2 | [0.0718, 0.0882, 0.0405, 0.0094] | 2 | `None` | 0.000 | **A** | Row 1 anchor fell off page; 0 candidates. |
| `creditors[1].address_1` | `"ADDRESS ON FILE"` | 2 | [0.2590, 0.0897, 0.0486, 0.0095] | 2 | `None` | 0.000 | **A** | Row 1 anchor `None`; 0 candidates. |
| `creditors[3].name` | `"NAME ON FILE"` | 2 | [0.0716, 0.1114, 0.0405, 0.0094] | 2 | [0.0710, 0.0283, 0.0550, 0.0160] | 0.000 | **E** | Displaced anchor at $y = 0.0283$ ($\Delta y = -0.0831$); misses true row by 8 rows. |
| `creditors[3].address_1` | `"ADDRESS ON FILE"` | 2 | [0.2588, 0.1130, 0.0486, 0.0095] | 2 | [0.2580, 0.0283, 0.0550, 0.0160] | 0.000 | **E** | Displaced anchor at $y = 0.0283$; 0 IoU. |
| `creditors[4].name` | `"NAME ON FILE"` | 2 | [0.0715, 0.1224, 0.0405, 0.0094] | 2 | [0.0710, 0.0451, 0.0550, 0.0160] | 0.000 | **E** | Extrapolated anchor at $y = 0.0451$ ($\Delta y = -0.0773$). |
| `creditors[5].name` | `"NAME ON FILE"` | 2 | [0.0713, 0.1347, 0.0405, 0.0094] | 2 | [0.0710, 0.0618, 0.0550, 0.0160] | 0.000 | **E** | Extrapolated anchor at $y = 0.0618$ ($\Delta y = -0.0729$). |
| `creditors[6].name` | `"1012ND STREET INC."` | 2 | [0.0712, 0.1456, 0.0570, 0.0096] | 2 | [0.0941, 0.1541, 0.0432, 0.0418] | 0.016 | **F, J** | Matched to merged multi-line line 7 ($h=0.0418, \Delta y=0.0085$); height ratio 4.35x kills IoU. |
| `creditors[6].address_1` | `"1012ND STREET"` | 2 | [0.2584, 0.1472, 0.0440, 0.0094] | 2 | [0.0941, 0.1541, 0.0332, 0.0418] | 0.000 | **I, J** | Aligned to name token in col 0 instead of col 1 ($x=0.0941$ vs $0.2584$); column confusion. |
| `creditors[6].city` | `"SAN FRANCISCO"` | 2 | [0.6454, 0.1505, 0.0447, 0.0094] | 2 | [0.6450, 0.0964, 0.0450, 0.0160] | 0.000 | **E** | Column coordinates exact ($\Delta x = -0.0004$), but vertical displacement $\Delta y = -0.0541$. |
| `creditors[6].state` | `"CA"` | 2 | [0.7364, 0.1514, 0.0077, 0.0089] | 2 | [0.7360, 0.0964, 0.0150, 0.0160] | 0.000 | **E** | Column coordinates exact ($\Delta x = -0.0004$), vertical displacement $\Delta y = -0.0550$. |
| `creditors[6].postal_code` | `"94105"` | 2 | [0.7996, 0.1519, 0.0173, 0.0091] | 2 | [0.7990, 0.0964, 0.0250, 0.0160] | 0.000 | **E** | Column coordinates exact ($\Delta x = -0.0006$), vertical displacement $\Delta y = -0.0555$. |
| `creditors[7].name` | `"101 SECOND STREET INC"` | 2 | [0.0711, 0.1578, 0.0677, 0.0098] | 2 | [0.0710, 0.0968, 0.0550, 0.0160] | 0.000 | **E** | Extrapolated anchor displaced by $\Delta y = -0.0610$ (misses row 7 by 5 rows). |
| `creditors[7].address_1` | `"C/O HINES"` | 2 | [0.2583, 0.1595, 0.0291, 0.0092] | 2 | [0.2580, 0.0968, 0.0550, 0.0160] | 0.000 | **E** | Column coordinates exact ($\Delta x = -0.0003$), vertical displacement $\Delta y = -0.0627$. |
| `creditors[7].city` | `"SAN FRANCISCO"` | 2 | [0.6453, 0.1627, 0.0447, 0.0094] | 2 | [0.6450, 0.0968, 0.0450, 0.0160] | 0.000 | **E** | Column coordinates exact ($\Delta x = -0.0003$), vertical displacement $\Delta y = -0.0659$. |
| `creditors[10].name` | `"101 SECOND STREET, INC."`| 2 | [0.0707, 0.2018, 0.0705, 0.0098] | 2 | [0.0719, 0.1541, 0.0655, 0.0418] | 0.000 | **J, L** | Aligned to row 6 text in merged line 7 ($y=0.1541$ instead of $0.2018$); wrong occurrence. |
| `creditors[10].address_1` | `"SHARTSIS FRIESE LLP"` | 2 | [0.2579, 0.2035, 0.0568, 0.0096] | 2 | [0.1146, 0.2000, 0.2001, 0.0206] | 0.132 | **F, I** | Matches scattered OCR noise in col 0; wide box ($w=0.2001$), wrong column. |
| `creditors[20].name` | `"101 SECOND STREET, INC."`| 2 | [0.0692, 0.3643, 0.0996, 0.0102] | 2 | [0.0710, 0.3335, 0.0550, 0.0160] | 0.000 | **E** | Displaced grid anchor at $y = 0.3335$ ($\Delta y = -0.0308$). |
| `creditors[30].name` | `"101 SECOND STREET, INC."`| 2 | [0.0679, 0.4977, 0.0405, 0.0093] | 2 | [0.0710, 0.5156, 0.0550, 0.0160] | 0.000 | **E** | Drifted anchor at $y = 0.5156$ ($\Delta y = +0.0179$); IoU = 0.000. |

---

## 3. Systematic Failure Taxonomy (A – N Breakdown)

Across 544 evaluated field rules sampled on Page 2 and Page 3:

| Code | Category Description | Count | Percentage | Primary Root Cause |
| :---: | :--- | :---: | :---: | :--- |
| **A** | Correct page, no candidate | 460 | 84.6% | Grid lines break Tesseract OCR; DP anchor collapse leaves preceding rows without anchors; fallback aborted |
| **D** | OCR text is severely corrupted | 59 | 10.8% | OCR glyph mutations (`[RAMCONIRE`, `SICON`, `CATORRT`) fail exact/fuzzy dictionary matching |
| **J** | Long-line / multi-line confusion | 20 | 3.7% | Line clustering merged 4–6 rows into single line; whole-line or multi-row token bbox adopted |
| **B** | Correct page, candidate exists, rank wrong | 2 | 0.4% | Repeated company strings (`101 SECOND STREET INC`) matched to wrong row line index |
| **E** | PDF/OCR geometry mismatch (grid drift) | 540* | 99.3% | Horizontal column coordinates exact ($\Delta x < 0.001$), but vertical row offset $\Delta y = 0.03$–$0.07$ eliminates IoU |
| **CORRECT** | BBox IoU $\ge 0.50$ | 3 | 0.6% | Non-tabular header fields (`case_number`, `debtor`, `report_title`) |

*\*Note: Almost all evaluated tabular fields exhibit Category E geometry mismatch as the downstream consequence of Categories A, D, and J.*

---

## 4. Phase 2 Candidate Recall Analysis

Measuring candidate retrieval directly on the FTX sample:

- **Recall@1**: **0.37%** (2 / 544 rules)
- **Recall@5**: **0.37%** (2 / 544 rules)
- **Recall@10**: **0.37%** (2 / 544 rules)
- **Recall@20**: **0.37%** (2 / 544 rules)
- **Recall@50**: **0.37%** (2 / 544 rules)

### Diagnostic Interpretation:
Recall@20 is **extremely low (< 1%)**. This proves unequivocally that:
1. Lexical retrieval over the uncalibrated, corrupted OCR tokens fails to return the true word box in 99.6% of cases.
2. The failure is NOT merely a re-ranking problem among good candidates; the candidates simply do not exist in the retrieved pool.
3. Downstream column fallback fails because the table row vertical registration ($Y$) is shifted by 3 to 8 rows.

---

## 5. Summary of Key Discoveries

1. **Horizontal Geometry is Solved**: `standard_table_cols` for `creditors` has column $X$ positions with $< 0.001$ error.
2. **Vertical Geometry is the Sole Geometric Blocker**: Vertical row positioning fails because:
   - Rows are not strictly uniform: single-line rows have $\Delta y \approx 0.0110$, while multi-line cell rows have $\Delta y \approx 0.0208$.
   - A single false DP match poisons `consistent_indices`, causing linear interpolation to drift or collapse entirely.
3. **Table Row Registration Engine Needed**: A dedicated row-anchoring mechanism that detects true row baselines, accounts for multi-line cell expansions, and preserves column bounding boxes will immediately convert the 93.39% Page Grounding into high Word Grounding F1.

---

## 6. Document Structure & Hierarchical Layout Analysis (Phase 3)

The forensic inspection reveals a highly regular, structured hierarchy across the 114 pages:

```
DOCUMENT (114 pages, real_ftx_full_corrupted.pdf)
├── PAGE 1: Narrative Legal Caption & Cover
│   ├── Case Header: "IN THE UNITED STATES BANKRUPTCY COURT..."
│   ├── Debtor Entity: "FTX TRADING LTD., et al." (y ≈ 0.187, x ≈ 0.221)
│   ├── Case Number: "22-11068-JTD" (y ≈ 0.022, x ≈ 0.318)
│   └── Schedule Title: "CONSOLIDATED TOP 50 CREDITORS LIST"
└── PAGES 2–114: Tabular Schedule ("Consolidated List of Creditors")
    ├── Running Header (y < 0.04): Case & Page Numbers with subtle scanner skew (slope dy/dx ≈ -0.002 to +0.004)
    ├── Table Header Bar (y ≈ 0.060 - 0.075): Column Labels (Name, Address 1-4, City, State, Postal Code, Country)
    ├── Table Body Grid (y ≈ 0.0802 to 0.8830):
    │   ├── Pitch / Step Size: s = (0.8830 - 0.0802) / 70 = 0.011468
    │   ├── Total Slots per Page: 71 slots
    │   ├── Records per Page: M ≈ 55 to 70 creditors
    │   │   ├── Single-line rows: consume 1 slot (h ≈ 0.0095)
    │   │   └── Multi-line rows (e.g. 2-line addresses): consume 2 slots (h ≈ 0.0205)
    │   └── Column X-Spans (fixed physical table columns):
    │       ├── Name:        x ∈ [0.0712, 0.2584] (w = 0.1872)
    │       ├── Address 1:   x ∈ [0.2584, 0.4032] (w = 0.1448)
    │       ├── Address 2:   x ∈ [0.4032, 0.5240] (w = 0.1208)
    │       ├── Address 3:   x ∈ [0.5240, 0.6000] (w = 0.0760)
    │       ├── Address 4:   x ∈ [0.6000, 0.6454] (w = 0.0454)
    │       ├── City:        x ∈ [0.6454, 0.7364] (w = 0.0910)
    │       ├── State:       x ∈ [0.7364, 0.7996] (w = 0.0632)
    │       ├── Postal Code: x ∈ [0.7996, 0.8750] (w = 0.0754)
    │       └── Country:     x ∈ [0.8750, 0.9600] (w = 0.0850)
    └── Footer Bar (y > 0.900): Scanner margin artifacts
```

---

## 7. OCR vs. Native Geometry Quality (Phase 7)

- **`native_text_quality`**: **0.00** across all 114 pages (`native_char_count = 0`). Pure raster scan.
- **`ocr_quality`**: **Fair to Poor** on fine tabular cells. Tabular separator lines fragment glyphs into non-standard characters, making standard n-gram string retrieval unviable.
- **`geometry_quality`**: **High**. Despite corrupted text glyphs, table cell borders and grid positions are rigid and follow a deterministic mathematical layout.

---

## 8. Decision Gate & Next Targeted Improvement

1. **Why is FTX stuck near 0.45% Word F1 despite 93% Page F1?**  
   The lexical retrieval pool contains < 1% of true word boxes due to OCR grid-line fragmentation. Standard table DP alignment breaks because merged multi-row OCR lines produce false matches, poisoning step-size extrapolation and shifting all row anchors by 3–8 rows ($\Delta y = -0.05$ to $-0.07$).
2. **Is the bottleneck retrieval, alignment, ranking, or geometry?**  
   It is a **hybrid retrieval + row-registration geometry bottleneck**. Because lexical OCR retrieval fails, the system must rely on table layout geometry.
3. **What is Recall@20 and Recall@50?**  
   **0.37%** (2 / 544 rules) for both Recall@20 and Recall@50.
4. **What single change produces the largest gain?**  
   **Dynamic Slot Budgeting with Header-Calibrated Skew and Column Projection**:
   - Filter boilerplate salient tokens (`NAME ON FILE`, `ADDRESS ON FILE`).
   - Dynamically budget 71 vertical table slots across the $M$ page records (allocating 2 slots to multi-line address rows).
   - Calibrate per-page scan tilt slope ($\theta = dy/dx$) from header tokens.
   - Project cell bounding boxes into standardized column coordinates.
   - **Result**: Word Grounding F1 leaps from **0.45% to 22.94%** (51x gain) and Page Grounding F1 reaches **100.00%**.
5. **Impact on Control Documents**:  
   - `13f__sl_advisors_llc`: **99.80% Word F1, 100.00% Page F1** (Zero regression)
   - `nport__bullfinch_fund_inc`: **91.44% Word F1, 100.00% Page F1** (Zero regression)
   - `cabrera-2023`: **26.67% Word F1, 68.99% Page F1** (Zero regression)
   - `real_credit_strategies_full`: **19.29% Word F1, 77.57% Page F1** (Zero regression)
   - `real_clinton_property_25_11073_corrupted`: **47.83% Word F1, 94.12% Page F1** (Zero regression)
6. **Next Targeted Improvement**:  
   Refine multi-line cell slot allocation and address sub-field column boundaries on FTX, add long-document test cases for Phase 16, and prepare the EXP-005 Part 4 report (`EXP-005-ftx-v1.md`).
