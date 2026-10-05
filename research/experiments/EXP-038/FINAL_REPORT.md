# EXP-038 FINAL REPORT

## 1. Experiment Identity

- **Run ID**: `exp038_full_run_v1` (Targeted Held-Out Phase: `exp038_heldout_v1`)
- **Timestamp**: `2026-10-04T04:21:40Z`
- **Git Commit**: `178f81c`
- **Benchmark Version**: Official ExtractBench 370-Document Full Benchmark Corpus (`research/data/full`)
- **Evaluator**: Official `extract_bench.evaluation.evaluators.extract.ExtractEvaluator` (`ExtractAssociationF1Metric` / Unified Evidence Metric)
- **Environment**: Linux x86_64, Python 3.12.14, single-threaded math (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`)
- **Production Changes**: Zero permanent modifications to `src/tonerhound/` prior to integration gate verification. Multi-problem deterministic resolvers implemented in `research/experiments/EXP-038/` consuming existing `tonerhound.document.index` and `HybridDocumentIndex` primitives.
- **Exact Command**:
  ```bash
  ./.venv/bin/python research/experiments/EXP-038/run_targeted.py
  ./.venv/bin/python research/experiments/EXP-038/run_full_benchmark.py
  ./.venv/bin/python research/observer/run_all_validation_gates.py
  ```

---

## 2. Baseline

*(Reference Run: `canonical_baseline_v2`, frozen post-EXP-037 at commit `178f81c`)*

- **Word Grounding F1**: 56.6707%
- **Page Grounding F1**: 81.9786%
- **Word Grounding Precision**: 62.4511%
- **Word Grounding Recall**: 53.1233%
- **Total Gradeable Fields**: 498,140
- **Passing Fields ($IoU \ge 0.50$)**: 307,514
- **Failing Fields**: 190,626
- **Evaluated Documents (Word Grounding)**: 236
- **Evaluated Documents (Page Grounding)**: 293

*(Historical Reference Run: `canonical_370_v1`, commit `178f81c`, Word F1 = 56.0477%, Page F1 = 81.6639%)*

---

## 3. Targeted Population & Multi-Problem Resolution Scope

EXP-038 attacked all remaining major failure classes identified by Failure Microscope V4:

| Fix ID | Failure Class Targeted | Baseline Failing Fields | Baseline Doc Breadth | Fix Mechanism Implemented |
| :--- | :--- | :---: | :---: | :--- |
| **Fix 1** | `REAL_INDEXING_MISS` (OCR Pages) | 39,037 | ~120 | Bounded character 3-gram index with edit distance $\le 1$ |
| **Fix 2** | `NON_TEXT_BOOLEAN_GROUNDING` | 2,845 | 160 | Morphological wireframe + Hough diagonal line checkbox provider |
| **Fix 3** | `NO_TEXT_AT_GOLD_REGION` | 16,886 | 130 | 300 DPI high-resolution render + adaptive CLAHE/Otsu retry |
| **Fix 4** | `DATE_INDEX_MISS` | 2,650 | 79 | Deterministic date normalizer regex for non-standard formats |
| **Fix 5** | `MULTI_LINE_SPLIT` | 563 | 42 | Bounded multi-line bounding box union assembler |
| **Fix 6** | `NORMALIZATION_MISMATCH` / `REAL_INDEXING_MISS` | 46,158 / 90,255 | 229 | Accounting negative parser (`(1,234.56)` $\rightarrow$ `-1234.56`) & currency stripper |
| **Fix 7** | `TOKEN_SLICING` | 23,646 | 114 | Proportional character-span sub-box slicer for column bleed |
| **Fix 8** | `HYPHENATION` | 7,623 | 68 | Terminal line hyphen joiner |

---

## 4. Phase E — Targeted Held-Out Evaluation

Evaluated across the frozen 32-document held-out cohort (`benchmarks/held_out_manifest.json`):

- **Documents Evaluated**: 32 documents (short: 14, medium: 12, long: 6)
- **Total Fields Evaluated**: 178,934
- **Baseline Passing Fields**: 88,802
- **After Passing Fields**: 89,017
- **Net Fields Rescued**: **+215 fields** (Passes $\ge 200$ target gate)
- **Fields Regressed**: **0 fields** (Passes zero-regression gate)
- **Execution Runtime**: 161.29s (2.69 min, average 5.04s per doc)
- **Gate Verdict**: **PASS** on all four criteria.

---

## 5. Full 370-Document Benchmark Results

Evaluated across the entire official ExtractBench 370-document corpus:

- **Word Grounding F1**: **58.1118%**
  - **Delta vs Canonical Baseline V2**: **+1.4411 pp**
  - **Delta vs Canonical Baseline V1**: **+2.0641 pp**
- **Page Grounding F1**: **82.2750%**
  - **Delta vs Canonical Baseline V2**: **+0.2964 pp**
  - **Delta vs Canonical Baseline V1**: **+0.6111 pp**
- **Word Grounding Precision**: **63.9733%** (+1.5222 pp vs baseline V2)
- **Word Grounding Recall**: **54.5086%** (+1.3853 pp vs baseline V2)
- **Total Benchmark Documents**: 370
- **Evaluated Documents (Word Grounding)**: 236
- **Evaluated Documents (Page Grounding)**: 293
- **Total Gradeable Fields**: 498,140
- **Passing Fields ($IoU \ge 0.50$)**: **308,756** (+1,242 fields rescued)
- **Failing Fields**: **189,384** (-1,242 fields)
- **Full Benchmark Evaluation Runtime**: 186.2s (3.10 min)

---

## 6. Before vs After Comparison

| Metric | Canonical Baseline V2 (`canonical_baseline_v2`) | EXP-038 After (`exp038_full_run_v1`) | Measured Delta |
| :--- | :---: | :---: | :---: |
| **Word Grounding F1** | **56.6707%** | **58.1118%** | **+1.4411 pp** |
| **Page Grounding F1** | **81.9786%** | **82.2750%** | **+0.2964 pp** |
| **Word Grounding Precision** | **62.4511%** | **63.9733%** | **+1.5222 pp** |
| **Word Grounding Recall** | **53.1233%** | **54.5086%** | **+1.3853 pp** |
| **Passing Fields ($IoU \ge 0.50$)** | 307,514 | 308,756 | **+1,242 fields** |
| **Failing Fields** | 190,626 | 189,384 | **-1,242 fields** |
| **Documents Modified** | — | 108 | **108 docs** |
| **Documents Rescued** | — | 108 | **108 docs** |
| **Documents Regressed** | — | 0 | **0 docs** |
| **Documents Unchanged** | — | 262 | **262 docs** |

---

## 7. Field-Level Rescue Analysis & Fix Attribution

Across the 1,242 rescued fields ($IoU < 0.50 \rightarrow IoU \ge 0.50$):

| Fix Component | Module | Fields Rescued | Share of Rescues | Target Failure Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **standard_resolver** | `ExtractBenchAdapter` fallback | 723 | 58.21% | Grounded unattempted / ungrounded valid candidates |
| **fix6_normalization** | `normalization_v2.py` | 494 | 39.77% | Parenthesized negatives `(123.45)` & currency prefixes |
| **fix1_ocr_noise** | `ocr_noise_index.py` | 16 | 1.29% | OCR substitution noise (Levenshtein $\le 1$) on scanned pages |
| **fix2_checkbox** | `visual_provider_v2.py` | 5 | 0.40% | Visual checkbox wireframe + state detection |
| **fix4_dates** | `date_normalizer.py` | 4 | 0.32% | Non-standard date format normalizations |
| **Total** | — | **1,242** | **100.0%** | **Zero Regressions ($IoU \ge 0.50 \rightarrow IoU < 0.50$)** |

---

## 8. Failure Migration

| Failure Class | Baseline V2 Field Count | EXP-038 Post-Run Count | Net Field Change | Migration to RECOVERED | Remaining Realistic Expected Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **REAL_INDEXING_MISS** | 90,255 | 89,022 | **-1,233** | 1,233 fields | +0.7394 pp |
| **NORMALIZATION_MISMATCH** | 46,158 | 46,158 | 0 | 0 fields | +0.5901 pp |
| **TOKEN_SLICING** | 23,646 | 23,646 | 0 | 0 fields | +0.3023 pp |
| **NO_TEXT_AT_GOLD_REGION** | 16,886 | 16,886 | 0 | 0 fields | +0.2070 pp |
| **HYPHENATION** | 7,623 | 7,623 | 0 | 0 fields | +0.0683 pp |
| **NON_TEXT_BOOLEAN_GROUNDING** | 2,845 | 2,840 | **-5** | 5 fields | +0.7746 pp |
| **DATE_INDEX_MISS** | 2,650 | 2,646 | **-4** | 4 fields | +0.7180 pp |
| **MULTI_LINE_SPLIT** | 563 | 563 | 0 | 0 fields | +0.0252 pp |
| **Total** | **190,626** | **189,384** | **-1,242** | **1,242 fields** | **+3.4249 pp** |

---

## 9. What Worked

1. **Accounting Negative Normalization (Fix 6)**: [OBSERVED FACT]  
   Rescued **494 fields** across financial statements and tax forms (e.g., `real_vg_divappr_full`: +291 fields, `sched_i__akron_community_foundation`: +150 fields, `medium/bar-lev-2021`: +15 fields). Inverting parenthesized accounting strings `(1,234.56)` into numeric query `-1234.56` resolved massive tabular losses.
2. **Standard Resolver Grounding Fallback**: [OBSERVED FACT]  
   Rescued **723 fields** by running the existing structural disambiguation resolver with geometry enhancement on ungrounded target rules that had received no citation in baseline.
3. **OCR Noise Tolerance on Degraded Scans (Fix 1)**: [OBSERVED FACT]  
   Rescued **16 fields** across scanned tax forms (`passcoag-2020-w2`, `07021-2016`) by resolving 1-character OCR substitutions (e.g. `0` $\leftrightarrow$ `O`, `1` $\leftrightarrow$ `l`, `S` $\leftrightarrow$ `5`) using bounded character 3-gram indexing.
4. **Zero Regressions Across All 498,140 Fields**: [OBSERVED FACT]  
   Unconditional preservation of passing baseline citations guaranteed exactly 0 field regressions across the corpus.

---

## 10. What Did Not Work

1. **Checkbox Rescues on Dense Complex Schedules**: [OBSERVED FACT]  
   Only 5 checkbox fields were rescued out of 2,845 failing booleans. While visual wireframe contour extraction works well on clean isolated forms, dense schedule tables (e.g. Schedule K-1, W-2 matrices) have tiny 6x6 pixel boxes embedded within grid lines that merged into table cell borders, evading square contour detection.
2. **Date Normalization Volume**: [OBSERVED FACT]  
   Only 4 date fields were rescued out of 2,650 `DATE_INDEX_MISS` failures. The majority of remaining date failures involve dates formatted identically to native digital text, but failing due to month name abbreviations or OCR noise in the year digits.
3. **Token Slicing / Multi-line Visual Wraps**: [OBSERVED FACT]  
   0 fields rescued by standalone multi-line or token slicing heuristics in this cycle. Table column bleed requires cell-aware coordinate bounding rather than pure proportional string length slicing.

---

## 11. What the Microscope Could Not Prove

1. **Recoverability of Degraded Visual Checkboxes**: [OBSERVED FACT]  
   The microscope confirms that 2,840 boolean fields fail $IoU \ge 0.50$. It cannot prove whether line-removal preprocessing (horizontal/vertical morphological structuring elements) would disentangle the boxes from grid lines without degrading box borders.
2. **Ambiguity Resolution in Identical Tabular Digits**: [OBSERVED FACT]  
   On dense IRS Schedules where `$0.00` appears dozens of times on the same page, the microscope identifies a spatial miss, but cannot prove whether row-header graph walking or column-coordinate anchoring is causally responsible.

---

## 12. What We Were Wrong About / What We Were Missing

1. **The Magnitude of Unattempted Standard Groundings**: [DIRECTLY SUPPORTED EXPLANATION]  
   We previously assumed that all failures in `REAL_INDEXING_MISS` required novel fuzzy indexing or normalization algorithms. In reality, **723 fields (58.2% of all rescues)** were recovered simply by running TonerHound's standard `ExtractBenchAdapter` with geometry enhancements on ungrounded targets that baseline had skipped or dropped.
2. **Dominance of Accounting Formats in Real Financial Benchmark**: [DIRECTLY SUPPORTED EXPLANATION]  
   Negative numbers represented with parentheses `(xxx)` were the single largest algorithmic blocker, accounting for 494 rescued fields in one deterministic fix.

---

## 13. New Failure Modes / Regressions

- **Zero Field Regressions**: [OBSERVED FACT]  
  Exactly **0 fields regressed** from passing to failing ($IoU \ge 0.50 \rightarrow IoU < 0.50$).
- **Zero Document Regressions**: [OBSERVED FACT]  
  Across all 370 benchmark documents, 108 improved, 0 regressed, and 262 remained unchanged.

---

## 14. Runtime / Operational Latency Cost

| Component | Measured Latency | Operational Notes |
| :--- | :---: | :--- |
| **Phase E Held-Out Resolution (32 docs)** | 161.3s (2.69 min) | Average 5.04s per document |
| **Full Corpus Resolution (108 modified docs)** | ~450s (7.5 min) | Average 4.17s per modified document |
| **Official ExtractBench Evaluator (108 docs)** | 179.2s (2.98 min) | Official Hungarian matching execution |
| **Failure Microscope V4 Audit (370 docs)** | 7.0s | Post-run causal re-classification & migration |
| **Overhead on Unmodified Digital Documents** | **0.00s** | Completely skipped |

---

## 15. Current Score vs 90% Target

$$\text{Canonical Baseline V1: } 56.0477\%$$
$$\text{EXP-037 OCR Routing Gain: } +0.6230\text{ pp}$$
$$\text{Canonical Baseline V2: } 56.6707\%$$
$$\text{EXP-038 Multi-Fix Gain: } \mathbf{+1.4411\text{ pp}}$$
$$\text{Current Measured Score: } \mathbf{58.1118\%}$$

$$\text{Target Score: } 90.0000\%$$
$$\text{Remaining Arithmetic Gap: } 90.0000\% - 58.1118\% = \mathbf{31.8882\text{ percentage points}}$$

- **Empirical Boundary**: [OBSERVED FACT]  
  The multi-problem resolution suite in EXP-038 lifted TonerHound past 58% Word F1 (+1.4411 pp net gain). The remaining gap to 90% is **31.8882 pp**. Failure Microscope V4 reports a remaining realistic expected opportunity of **+3.4249 pp** across all remaining classes under deterministic methods.

---

## 16. Decision

**SUPPORTED — VALIDATED FOR PRODUCTION INTEGRATION**

- **Criteria Evaluation**:
  - Word Grounding F1 Delta: **+1.4411 pp** (Strictly positive)
  - Page Grounding F1 Delta: **+0.2964 pp** (Strictly positive)
  - Fields Rescued: **+1,242 fields** across 108 documents
  - Fields Regressed: **0 fields**
  - Performance: Minimal overhead, digital pages untouched.
  - Microscope Calibration: Gate V8 Backtesting passed (1.92x actual/expected ratio within 2x requirement).

---

## 17. Next Investigation

1. **Table Grid Line Removal for Checkbox Extraction**: [DIRECTLY SUPPORTED EXPLANATION]  
   - *Target*: 2,840 remaining failing boolean fields embedded in dense schedule tables.
   - *Approach*: Apply morphological line subtraction (horizontal/vertical kernel filters) before wireframe contour detection to isolate 6x6 pixel boxes from cell borders.
2. **Cell-Aware Coordinate Bounding for Token Slicing**: [DIRECTLY SUPPORTED EXPLANATION]  
   - *Target*: 23,646 failing fields in `TOKEN_SLICING`.
   - *Approach*: Use table column grid bounds from PyMuPDF vector drawing paths rather than string-length proportional heuristics to eliminate column bleed.
