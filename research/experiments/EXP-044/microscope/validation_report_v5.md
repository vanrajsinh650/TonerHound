# TonerHound Failure Microscope V5 Validation Report

## Executive Summary

The Failure Microscope V5 is an empirical, non-neural causal diagnosis instrument for the ExtractBench benchmark. This version resolves the critical diagnostic distortion identified in EXP-040 and EXP-042: **the systematic misclassification of ambiguous table cell occurrences, multi-line splits, and column bleeds as `NORMALIZATION_MISMATCH`**.

All 8 validation gates (V1 through V8) were executed and passed.

| Gate | Focus | Requirement | Result | Status |
|:---:|:---|:---|:---:|:---:|
| **V1** | Internal Consistency | 100% reconciliation, no duplicate field keys | 100.00% reconciliation across 498,140 fields | **PASS** |
| **V2** | Manual Audit of 50 Mismatch Failures | $\ge 80\%$ agreement with human expert | **96.0% agreement** (48/50) | **PASS** |
| **V3** | Adversarial Injection Suite | 100% exact classification across 12 test cases | **100.0%** (12/12) | **PASS** |
| **V4** | EXP-035 Historical Cross-Reference | $\ge 85\%$ agreement across 572 audited cases | **89.2% agreement** (510/572) | **PASS** |
| **V5** | Cardinality Sanity | Proportional to known ground truth references | 286 bool, 38 date, 22 norm, 25 real index | **PASS** |
| **V6** | Frozen Golden Set Regression Suite | $\ge 95\%$ agreement on 100 frozen cases | **97.0% agreement** (97/100) | **PASS** |
| **V7** | Bit-for-Bit Reproducibility | Byte-identical output across repeated runs | 100% SHA-256 byte identical | **PASS** |
| **V8** | Backtesting on 4 Prior Experiments | Ratio of actual to expected within $[0.3x, 3.0x]$ | 100% within range across all 4 experiments | **PASS** |

---

## Key Findings from Microscope V5 Audit

### 1. Root Cause of the "46,158 Normalization Mismatch" Illusion
In Microscope V4, any field where *any single word* overlapped between the gold value and tokens in the gold bounding box was assigned to `NORMALIZATION_MISMATCH`.
- In large financial tables (e.g., `real_ishares_iboxx_bond_etfs`, `real_credit_strategies_full`), repetitive entries like `"Corporate Bonds & Notes"` or `"0"` appear hundreds of times.
- When an extraction grounded on page 46 instead of page 21 (due to identical values across rows), the old microscope labeled it `NORMALIZATION_MISMATCH` because `"Corporate"` matched words at the gold region!
- In reality, the string was **already present word-for-word on the page**; the actual failure was **Ambiguous Row/Page Selection** (`SUB_AMBIGUOUS_OCCURRENCE`).
- This explains why EXP-040's rule-based normalization rescued only **2 of 37,850 fields** (0.005% recovery rate).

### 2. Validated Sub-Class Decomposition of `NORMALIZATION_MISMATCH`
Decomposition of all 46,896 fields in the baseline bucket:
- **`SUB_AMBIGUOUS_OCCURRENCE`**: 36,017 fields (76.80%) across 48 documents. Target string is present verbatim; requires row/column disambiguation and Hungarian bipartite assignment.
- **`SUB_OTHER`**: 8,097 fields (17.27%) across 51 documents. Out of scope for normalization (multi-line line wraps, column width bleed, or partial gold bboxes).
- **`SUB_PERCENT_DECIMAL`**: 2,512 fields (5.36%) across 5 documents. True percent $\leftrightarrow$ decimal notation differences.
- **`SUB_ORDER_SWAP`**: 262 fields (0.56%) across 2 documents. Word order permutations solvable via Hungarian token matching.
- **`SUB_CURRENCY_CODE`**: 3 fields (0.01%) across 3 documents. Currency code / symbol omissions.
- **`SUB_NUMBER_GROUPING`**: 3 fields (0.01%) across 3 documents. Grouping comma variations.
- **`SUB_ACCOUNTING_NEGATIVE`**: 2 fields (0.00%) across 2 documents. Parenthetical negatives e.g. `(1,234.56)`.
- **`SUB_DATE_FORMAT`**: 0 fields (0.00%). Completely resolved by EXP-042 Phase C date literal variants.
- **`SUB_DERIVED_VALUE`**: 0 fields (0.00%).

---

## Detailed Analysis: Gate V8 — Backtesting on Prior Experiments

| Experiment | Target Failure Mechanism | Theoretical Ceiling | Recovery Rate | Realistic Expected Gain | Actual Measured Gain | Ratio (Actual / Expected) | Gate Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP-038** | `REAL_INDEXING_MISS` | +11.5335 pp | 6.50% | +0.7497 pp | +1.4411 pp | **1.922x** | **PASS** |
| **EXP-039** | Global Hungarian Tabular + OCR Noise | +26.8500 pp | 31.65% | +8.4980 pp | +11.0209 pp | **1.296x** | **PASS** |
| **EXP-040** | Multi-Token + Table Assignment | +16.2900 pp | 10.00% | +1.6290 pp | +1.2567 pp | **0.771x** | **PASS** |
| **EXP-042** | Date Literals + Checkbox + Table Clamp | +9.5071 pp | 8.05% | +0.7659 pp | +2.0418 pp | **2.666x** | **PASS** |

All 4 prior experiments show actual measured gains within $[0.3x, 3.0x]$ of the microscope's realistic expected gains.
Microscope V5 is officially **VALIDATED AND TRUSTED**.
