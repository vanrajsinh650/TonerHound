# TONERHOUND — FORENSIC AUDIT: EXP-037 FULL BENCHMARK RUN
**Run ID:** `exp037_full_run_v1`  
**Git Commit:** `178f81c`  
**Execution Timestamp:** 2026-10-03T10:33:57Z  
**Benchmark Version:** Official ExtractBench 370-Document Benchmark Corpus  
**Evaluator:** Official `ExtractEvaluator` (`ExtractAssociationF1Metric` / Unified Grounded Metric)  

---

## 1. Executive Summary

This forensic audit evaluates the results of **EXP-037: Deterministic Page-Level OCR Routing** across the full official 370-document ExtractBench benchmark.

### Headline Benchmark Metrics

| Metric | Canonical Baseline (`canonical_370_v1`) | EXP-037 After (`exp037_full_run_v1`) | Measured Delta |
| :--- | :---: | :---: | :---: |
| **Word Grounding F1** | **56.0477%** | **56.6707%** | **+0.6230 pp** |
| **Page Grounding F1** | **81.6639%** | **81.9786%** | **+0.3147 pp** |
| **Word Grounding Precision** | **61.7275%** | **62.4511%** | **+0.7236 pp** |
| **Word Grounding Recall** | **52.5686%** | **53.1233%** | **+0.5547 pp** |
| **Total Evaluated Documents** | 236 | 236 | 0 |
| **Total Gradeable Fields** | 498,140 | 498,140 | 0 |
| **Passing Fields ($IoU \ge 0.50$)** | 307,373 | 307,514 | **+141 fields** |
| **Failing Fields** | 190,767 | 190,626 | **-141 fields** |
| **Field Regressions** | 0 | 0 | **0** |

---

## 2. Forensic Investigation of the "58,778" Target Population

In the canonical baseline audit (`canonical_370_v1`), the Failure Microscope reported **58,778 fields across 130 documents** classified as `NO_TEXT_AT_GOLD_REGION`, carrying a theoretical macro-weighted opportunity of **+19.6271 pp**.

### Forensic Findings:
1. **The Origin of the 58,778 Count**:
   - In `canonical_370_v1`, the audit classifier constructed `DocumentIndex` using `enable_ocr=False` to execute the audit rapidly in 145 seconds without running OCR.
   - For all scanned/image pages without a digital vector text layer, `doc_index.get_page(p).tokens` was empty.
   - Consequently, Rule 2 of the classifier (`if not tokens_in_gold: return NO_TEXT_AT_GOLD_REGION`) caught all failing fields on those pages before deeper causal analysis could be performed.
2. **Pre-Existing Baseline OCR Citations**:
   - Out of the 58,778 fields, **53,467 fields (90.96%)** belonged to just three massive corrupted schedules:
     - `long/real_imedia_full_corrupted`: 39,071 failing fields
     - `long/real_ftx_full_corrupted`: 12,661 failing fields
     - `long/real_sm0801_eco_full_corrupted`: 1,735 failing fields
   - Inspection revealed that the baseline predictions in `EXP-028E` **ALREADY HAD OCR CITATIONS** generated for all 34,137 non-null fields in `real_ftx` and 48,069 non-null fields in `real_imedia`.
   - The failure mode for these fields was **NOT missing OCR text**; it was **OCR bounding box drift and table cell alignment offsets** (e.g. `creditors[6].postal_code` had IoU = 0.4992, falling just 0.0008 below the 0.50 cutoff).
3. **Genuine Unresolved OCR Population**:
   - Across the remaining 127 documents (primarily IRS tax forms, Texas RRC oil & gas filings, and court schedules), ~5,311 text fields had missing or incomplete OCR citations.
   - Enabling deterministic page-level OCR routing successfully extracted tokens on these zero-text pages, allowing `EvidenceResolver` to rescue **141 previously failed fields** with $IoU \ge 0.50$ across 45 documents.

---

## 3. Failure Migration Analysis

When the Failure Microscope evaluated the post-OCR state across all 498,140 fields, `NO_TEXT_AT_GOLD_REGION` collapsed dramatically, revealing the true underlying causal mechanisms:

| Failure Class | Canonical Baseline (`canonical_370_v1`) | EXP-037 After (`exp037_full_run_v1`) | Field Delta | Macro Opportunity Before | Macro Opportunity After |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NO_TEXT_AT_GOLD_REGION** | 58,778 | 16,886 | **-41,892 (-71.27%)** | +19.6271 pp | +7.8108 pp |
| **REAL_INDEXING_MISS** | 51,218 | 90,255 | **+39,037 (+76.22%)** | +5.5731 pp | +11.5335 pp |
| **NORMALIZATION_MISMATCH** | 45,015 | 46,158 | **+1,143 (+2.54%)** | +2.8554 pp | +5.4780 pp |
| **TOKEN_SLICING** | 22,857 | 23,646 | **+789 (+3.45%)** | +2.9992 pp | +4.5411 pp |
| **HYPHENATION** | 7,311 | 7,623 | **+312 (+4.27%)** | +0.5229 pp | +1.2252 pp |
| **NON_TEXT_BOOLEAN_GROUNDING** | 2,847 | 2,845 | **-2 (-0.07%)** | +16.4542 pp | +16.5056 pp |
| **DATE_INDEX_MISS** | 2,436 | 2,650 | **+214 (+8.78%)** | +0.7105 pp | +1.1990 pp |
| **MULTI_LINE_SPLIT** | 305 | 563 | **+258 (+84.59%)** | +0.1044 pp | +0.1964 pp |

### Key Insight:
41,892 fields migrated out of `NO_TEXT_AT_GOLD_REGION`. Once OCR tokens were present at the gold coordinates, the microscope was able to observe the true bottleneck: **exact-string indexing failure caused by OCR character noise** (migrated to `REAL_INDEXING_MISS`).

---

## 4. Per-Document Improvements and Breadth

- **Documents with Word F1 Improved**: **45 documents**
- **Documents with Word F1 Regressed**: **3 documents** (slight precision denominator effect from generating candidates with $IoU < 0.50$)
- **Documents Unchanged**: **82 documents**

### Top Improved Documents:
1. `short/P4-83-631_246`: 8.33% $\rightarrow$ 27.78% (**+19.44 pp**)
2. `short/2A-9-160258_109909`: 31.19% $\rightarrow$ 42.20% (**+11.01 pp**)
3. `short/P4-27-51022_74198`: 16.84% $\rightarrow$ 27.37% (**+10.53 pp**)
4. `short/2A-9-160282_109921`: 32.73% $\rightarrow$ 41.82% (**+9.09 pp**)
5. `short/passcoag-2020-w2-p0003-r4`: 54.90% $\rightarrow$ 62.75% (**+7.84 pp**)
6. `short/2A-9-160266_109913`: 29.41% $\rightarrow$ 35.29% (**+5.88 pp**)
7. `short/2A-9-160318_109939`: 31.07% $\rightarrow$ 36.89% (**+5.83 pp**)
8. `short/2A-9-160298_109929`: 46.15% $\rightarrow$ 51.92% (**+5.77 pp**)
9. `medium/becerra-2022`: 35.76% $\rightarrow$ 37.81% (**+2.05 pp**, 10 fields rescued)
10. `medium/becerra-2021`: 28.51% $\rightarrow$ 30.48% (**+1.97 pp**, 8 fields rescued)

---

## 5. Arithmetic Distance to 90.00% Word F1

- **Baseline Word F1 ($A$)**: 56.0477%
- **EXP-037 Measured Gain ($B$)**: +0.6230 pp
- **New Measured Score ($A + B$)**: **56.6707%**
- **Target Score**: 90.0000%
- **Remaining Arithmetic Gap ($90.0000 - 56.6707$)**: **33.3293 percentage points**
