# Microscope Self-Audit — canonical_370_v1

## 1. What the Microscope Proves with High Confidence
1. **Zero Text Layer Absence**: Proves deterministically whether MuPDF extracts any character tokens within the gold bounding box (`NO_TEXT_AT_GOLD_REGION`).
2. **Boolean / Checkbox Non-Text Evidence**: Evaluates whether boolean schema targets contain text vs raster glyphs, adhering to Section 9 `bool` type checks.
3. **Exact vs Offset Geometry**: Distinguishes between wrong page (`PAGE_ROUTING`), wrong table row (`WRONG_ROW`), and narrow/wide bounding box truncations.

## 2. What the Microscope CANNOT Prove from Available Metadata
1. **OCR Recoverability**: A finding of `NO_TEXT_AT_GOLD_REGION` proves the text layer is empty. It does NOT prove that an OCR engine would recognize the text with sufficient quality to pass IoU >= 0.50.
2. **Semantic Disambiguation vs Ambiguous Layout**: In dense tables with repeated numeric values (e.g. `0.00` across 20 columns), the microscope classifies a mismatch as `WRONG_ROW` or `WRONG_COLUMN` based on coordinate offsets, but cannot prove whether the pipeline's failure was semantic label confusion or spatial coordinate drifting.
3. **Multi-Cause Precedence**: Where a page has no text layer AND the field is a checkbox, the classifier prioritizes `NON_TEXT_BOOLEAN_GROUNDING` over `NO_TEXT_AT_GOLD_REGION`. While causally justified, both factors contribute.

## 3. Denominator Clarification
- **Field-Weighted vs Macro-Document-Weighted**:
  - `long/real_imedia_full_corrupted` contains **39,064** failures in a single document (28.7% of all failure fields).
  - However, in the official macro-averaged benchmark, this single document accounts for only **0.42 pp** of Word Grounding F1.
  - The microscope explicitly reports `macro_weighted_opportunity_pp` to prevent single massive documents from distorting research priorities.
