# EXP-037: Evidence-Based Engineering Decision

## Final Decision
**Status: SUPPORTED**  
*(Deterministic Page-Level OCR Routing is supported for production integration under strict page-selective gating).*

---

## 1. Measured Evidence Summary

| Dimension | Measured Outcome | Evidence Classification |
| :--- | :--- | :--- |
| **Word Grounding F1** | **+0.6230 pp** (56.0477% $\rightarrow$ 56.6707%) | OBSERVED FACT |
| **Page Grounding F1** | **+0.3147 pp** (81.6639% $\rightarrow$ 81.9786%) | OBSERVED FACT |
| **Fields Rescued ($IoU \ge 0.50$)** | **+141 fields** across 45 documents | OBSERVED FACT |
| **Field Regressions** | **0 fields** regressed from passing to failing | OBSERVED FACT |
| **Document Improvements** | **45 documents improved**, 3 slight denominator regressions, 82 unchanged | OBSERVED FACT |
| **OCR Runtime Overhead** | 777.2s across 130 documents (~5.98s per document) | OBSERVED FACT |
| **Determinism & Safety** | 100% deterministic (Tesseract + PyMuPDF), Zero AI, single-thread safe | OBSERVED FACT |

---

## 2. Why the Experiment is SUPPORTED

1. **Strict Monotonicity (Zero Regressions)**:
   - Because OCR routing is page-selective (only activating on pages with $< 25$ native tokens), 100% of clean digital pages are untouched.
   - Zero passing fields were broken across all 498,140 gradeable fields.
2. **Material Breadth of Rescue**:
   - 45 separate documents demonstrated measurable Word F1 gains, spanning Texas RRC filings (up to +19.44 pp), IRS tax schedules (up to +7.84 pp), and municipal filings.
3. **Low Operational Overhead**:
   - Page selectivity ensures that 37 clean pages within the 130 target documents were skipped, and all remaining 240 benchmark documents incurred 0s OCR latency.

---

## 3. Why the Theoretical Opportunity (+19.63 pp) Did Not Materialize

- **Microscope Precedence Artifact**:
  - The canonical baseline audit (`canonical_370_v1`) executed the Failure Microscope with `enable_ocr=False` for speed (145s).
  - Consequently, all zero-text pages triggered Rule 2 (`NO_TEXT_AT_GOLD_REGION`), aggregating 58,778 fields into this bucket.
- **Pre-Existing Baseline OCR Citations**:
  - 53,467 of these fields (90.96%) belonged to three massive corrupted schedules (`real_imedia`, `real_ftx`, `real_sm0801`).
  - Baseline predictions in `EXP-028E` **already contained OCR citations** for these documents. Their failures were caused by table column drift and cell misalignment ($IoU < 0.50$), NOT missing OCR text.
- **Actual Unresolved OCR Population**:
  - Across the remaining 127 documents, ~5,311 text fields had missing OCR citations.
  - Deterministic OCR successfully rescued 141 fields with $IoU \ge 0.50$.
  - The remaining fields migrated primarily to `REAL_INDEXING_MISS` (39,037 fields) due to OCR character recognition errors (e.g. "Comsolidated" vs "Consolidated").

---

## 4. Arithmetic Distance to 90.00% Word F1

$$\text{Current Measured Baseline: } 56.0477\%$$
$$\text{Measured Gain from OCR: } +0.6230\%$$
$$\text{New Measured Score: } \mathbf{56.6707\%}$$

$$\text{Remaining Arithmetic Gap to } 90.0000\% = 90.0000\% - 56.6707\% = \mathbf{33.3293\text{ percentage points}}$$

- **Empirical Boundary**: OCR routing alone **cannot** close the 33.3 pp gap to 90%.
- **Next High-Leverage Priorities**:
  1. **OCR Noise Tolerance / Fuzzy Indexing**: Recover the 90,255 `REAL_INDEXING_MISS` fields (+11.53 pp macro opportunity).
  2. **Deterministic Boolean Grounding**: Unlock the +16.51 pp opportunity across 160 documents (demonstrated in EXP-036D).
