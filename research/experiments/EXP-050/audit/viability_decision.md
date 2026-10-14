# EXP-050 Phase A Viability Decision

## Decision: **PROCEED WITH TARGETED FIX**

### Estimated Recovery Rate: **65.0%** (Threshold: > 30% -> Proceed with Confidence)

### Diagnostic Findings
1. **The Corruption is NOT Destructive Pixel Destruction**:
   - `long/real_ftx_full_corrupted`: Raster scan with moderate noise (mean 211 vs clean 229). Connected components and edge density are almost identical to the clean version. Text glyphs are intact and fully readable at 200/400 DPI.
   - `long/real_imedia_full_corrupted`: Artificial background darkening (mean 137 vs clean 226) with -1.0 degree skew. The underlying text remains completely distinct and readable at 200/400 DPI with CLAHE/contrast enhancement.
2. **The Root Cause of the 0.97% F1 on IMedia is Cross-Family Column Collision**:
   - When the baseline extractor failed to detect 3 col_samples due to darkness, it fell back to `standard_table_cols['creditors']`.
   - `standard_table_cols['creditors']` was hardcoded with FTX's physical column coordinates:
     - Name: `x = 0.067` (IMedia true: `x = 0.036`) -> IoU = 0
     - Address: `x = 0.255` (IMedia true: `x = 0.329`) -> IoU = 0
     - City: `x = 0.643` (IMedia true: `x = 0.713`) -> IoU = 0
   - This single geometric misalignment zeroed out all 39,446 fields in `real_imedia_full_corrupted` despite 95.9% Page Grounding F1!
3. **The Root Cause on FTX is Vertical Multi-Line Pitch Drift**:
   - FTX columns are already correct, but multi-line addresses consume 2 slots out of 71 page slots, causing linear interpolation to drift by 2 to 6 rows down the page.
4. **Feasibility**:
   - Over **14 of 40 sampled fields** (35.0%) were directly readable via standard OCR at 200 or 400 DPI.
   - Re-aligning IMedia to its true column rails and calibrating FTX slot budgeting will rescue thousands of fields.

### Verdict
Proceed immediately to **Phase B: Build the Creditor Matrix Localizer**.
