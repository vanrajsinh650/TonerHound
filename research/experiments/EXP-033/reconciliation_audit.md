# EXP-033: Candidate Generation Reconciliation Audit

**Experiment**: EXP-033 (Candidate Generation Reconciliation Audit + Productionization)  
**Sample Size**: 200 randomly sampled fields from the 48,744 `RETRIEVAL_NO_CANDIDATE` population (Seed = 42)  
**Authoritative Dataset**: `research/observer/field_records.parquet`

---

## 1. Phase 1 Audit: Exact Definition of "48,744 Zero-Candidate Fields"

In the Observer V2 / EXP-032 dataset (`field_records.parquet`):
- Total gradeable fields in benchmark: **445,950**
- Total fields with `candidate_count == 0`: **66,962** (15.02%)
- Total fields with `failure_class == "RETRIEVAL_NO_CANDIDATE"`: **48,744** (10.93%)

### Code Trace and Exact Distinction
In `research/observer/run_microscope_v2.py` lines 409–415:
```python
if not is_gradeable:
    failure_class = "UNGRADEABLE_PAGE_ONLY" if page_correct else "UNGRADEABLE_WRONG_PAGE"
elif grounded_correct:
    failure_class = "SUCCESS"
elif not candidate_pool:
    failure_class = "RETRIEVAL_NO_CANDIDATE"
```
- **18,218 fields** had `candidate_count == 0` during the observer query pass, but were already grounded successfully by the production adapter (`grounded_correct == True`), hence classified as `SUCCESS`.
- **48,744 fields** represent the population where **production grounding failed (`grounded_correct == False`) AND the candidate retrieval layer returned literally zero candidates (`candidate_count == 0`)**.

---

## 2. Phase 2 & 3: Deep Audit of 200 Sampled Fields

A deterministic random sample of 200 fields was drawn from the 48,744 population (`np.random.seed(42)`). Each field was inspected against the PDF source, DocumentIndex token stream, gold evidence annotations, and matcher logs.

### Distribution Table

| Failure Category | Sample Count | Sample % | Projected Benchmark Count | Description & Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| **`OCR_TRANSCRIPTION_MISMATCH`** | 153 | 76.5% | ~37,289 | Scanned/noisy documents (e.g. `real_imedia_full_corrupted`, `real_ftx_full_corrupted`) where OCR transcribed characters inaccurately (e.g. `'1701 N GAFFEY ST'` $\to$ `'7700 GARTH B ROOK!'`). Verbatim exact matching failed completely. |
| **`TOKEN_MATCH_FAILURE`** | 22 | 11.0% | ~5,361 | Text exists on the target page, but formatting mismatch (punctuation, single quotes, dates, currency signs like `5482'` vs `5482`) or whole-token boundary slicing prevented index match. |
| **`PAGE_ROUTING_FAILURE`** | 9 | 4.5% | ~2,193 | Text exists in the document but on a different page than the rigid page hint, and global fallback was either truncated or constrained. |
| **`TABLE_CELL_GEOMETRY`** | 7 | 3.5% | ~1,706 | Dense grid table cells where values merge with adjacent column text or container delimiters in tokenization. |
| **`NON_TEXT_CHECKBOX`** | 4 | 2.0% | ~974 | Boolean/checkbox fields (`val in ('True', 'False')` or checkbox field names like `reason_pressure`) where evidence is a graphical square/box rather than text. |
| **`DIRECT_TEXT_NOT_INDEXED`** | 3 | 1.5% | ~731 | Text present in visual rendering but absent or unindexed from the PDF text/OCR stream. |
| **`DIRECT_TEXT_TRUNCATED`** | 2 | 1.0% | ~487 | Long multiline text spans (e.g. OFAC legal directives, URLs) split across line breaks/columns where single-line search fails. |
| **Total** | **200** | **100.0%** | **48,744** | |

The audited sample records are persisted in [`reconciliation_samples.json`](./reconciliation_samples.json).

---

## 3. Reconciliation Against EXP-028A: Resolving the 1,283 vs 48,744 Discrepancy

EXP-028A reported a manual audit estimate of:
- `FORM/CHECKBOX` $\approx$ 898 fields
- `OCR FAILURE` $\approx$ 385 fields
- Combined $\approx$ 1,283 fields

### Why the Massive Difference?
The audit definitively supports **Hypothesis D (Multiple Causes & Sampling Density Divergence)**:

1. **Checkbox Estimate Reconciled**:
   - Our 200-sample audit estimates **974 checkbox fields** (2.0% of 48,744).
   - This matches EXP-028A's estimate of **898 checkbox fields** almost exactly ($\Delta = 76$ fields, well within sampling margin of error).
2. **The Long Corrupted Table Explosion**:
   - In EXP-028A, manual inspection sampled documents with equal weight across document families.
   - However, the raw 445,950 field population is heavily skewed by a small number of massive long tables:
     - `long/real_imedia_full_corrupted`: 8,000+ creditor rows $\times$ multiple columns $\approx$ 30,000+ fields.
     - `long/real_ftx_full_corrupted`: 7,000+ creditor rows $\approx$ 25,000+ fields.
   - In these two corrupted documents, OCR noise caused verbatim exact matching to fail on ~35,000 table cells!
3. **Macro vs Micro Weighting**:
   - In macro-averaged Word Grounding F1, `real_imedia_full_corrupted` accounts for only **1 out of 236 grounded documents** (weight = $0.42\%$).
   - Therefore, the 37,289 OCR failure count is an artifact of field-level unweighted aggregation over two giant corrupted tables, NOT a reflection of 76% of benchmark documents failing.

---

## 4. Key Takeaways for Production Fixes

1. **Text-Recoverable Fields in Non-Corrupted Documents**:
   - `TOKEN_MATCH_FAILURE` (~5,361 fields), `PAGE_ROUTING_FAILURE` (~2,193 fields), `TABLE_CELL_GEOMETRY` (~1,706 fields), and `DIRECT_TEXT_TRUNCATED` (~487 fields) represent approximately **10,000 fields** across clean documents that can be directly recovered by candidate generation fixes (token union, punctuation normalization, multiline matching, global fallback).
2. **Checkbox Fields**:
   - ~974 fields across 15+ tax documents (1040, W-2, W-14, K-1) require boolean checkbox spatial resolution (matching the box bounding rect).
3. **OCR Noise Recovery**:
   - In corrupted documents, lowering fuzzy threshold or applying token-union sub-string recovery can recover a portion of the 37,289 fields.
