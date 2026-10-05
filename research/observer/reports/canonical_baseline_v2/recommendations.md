# Failure Microscope V4 Calibrated Audit — canonical_baseline_v2

## 1. Executive Baseline Metrics
- **Word Grounding F1**: **56.6707%**
- **Page Grounding F1**: **81.9786%**
- **Word Grounding Precision**: 62.4511%
- **Word Grounding Recall**: 53.1233%
- **Total Evaluated Documents**: 236 documents
- **Total Gradeable Fields**: 498,140 fields
- **Passing Fields ($IoU \ge 0.50$)**: 307,514 fields (61.73%)
- **Failing Fields**: 190,626 fields (38.27%)
- **Total Theoretical Opportunity**: +48.4896 pp (upper bound ceiling)
- **Total Realistic Expected Opportunity**: **+4.3801 pp** (empirically calibrated)

## 2. Calibrated Failure Class Rankings (Microscope V4 Format)

| Rank | Failure Class | Fields | Breadth | Theoretical Ceiling | Recovery Rate | Realistic Expected Gain | Fix Type | Confidence |
|:---:|:---|---:|---:|---:|---:|---:|:---|:---:|
| 1 | `REAL_INDEXING_MISS` | 90,255 | 97.0% | +11.5335 pp | 6.50% | **+0.7497 pp** | Indexing improvement | HIGH |
| 2 | `NORMALIZATION_MISMATCH` | 46,158 | 86.4% | +5.4780 pp | 10.00% | **+0.5478 pp** | Normalization expansion | LOW |
| 3 | `TOKEN_SLICING` | 23,646 | 66.1% | +4.5411 pp | 10.00% | **+0.4541 pp** | Token slicing refinement | LOW |
| 4 | `NO_TEXT_AT_GOLD_REGION` | 16,886 | 51.7% | +7.8108 pp | 2.65% | **+0.2070 pp** | OCR recovery (scanned pages) | HIGH |
| 5 | `HYPHENATION` | 7,623 | 42.4% | +1.2252 pp | 70.00% | **+0.8576 pp** | Hyphenation join | MEDIUM |
| 6 | `NON_TEXT_BOOLEAN_GROUNDING` | 2,845 | 67.8% | +16.5056 pp | 4.70% | **+0.7758 pp** | Checkbox visual detection | HIGH |
| 7 | `DATE_INDEX_MISS` | 2,650 | 56.8% | +1.1990 pp | 60.00% | **+0.7194 pp** | Date normalization | MEDIUM |
| 8 | `MULTI_LINE_SPLIT` | 563 | 2.5% | +0.1964 pp | 35.00% | **+0.0687 pp** | Multi-line assembly | LOW |

## 3. Class-by-Class Calibrated Profiles

```
FAILURE CLASS: REAL_INDEXING_MISS
  Field count:              90255
  Document breadth:         229 (97.03% of grounded docs)
  THEORETICAL_CEILING:      +11.5335 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  6.50%      (empirical, source: EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%))
  REALISTIC_EXPECTED_GAIN:  +0.7497 pp  (theoretical × recovery)
  Fix type:                 Indexing improvement
  Historical confidence:    HIGH
  Source evidence:          EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%)
  Recommended next step:    OCR noise-tolerant bounded n-gram/fuzzy index on OCR pages
```

```
FAILURE CLASS: NORMALIZATION_MISMATCH
  Field count:              46158
  Document breadth:         204 (86.44% of grounded docs)
  THEORETICAL_CEILING:      +5.4780 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.5478 pp  (theoretical × recovery)
  Fix type:                 Normalization expansion
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Parenthesized negative numbers + currency code normalization
```

```
FAILURE CLASS: TOKEN_SLICING
  Field count:              23646
  Document breadth:         156 (66.10% of grounded docs)
  THEORETICAL_CEILING:      +4.5411 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.4541 pp  (theoretical × recovery)
  Fix type:                 Token slicing refinement
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Narrow character-span reconstruction on wide table cells
```

```
FAILURE CLASS: NO_TEXT_AT_GOLD_REGION
  Field count:              16886
  Document breadth:         122 (51.69% of grounded docs)
  THEORETICAL_CEILING:      +7.8108 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  2.65%      (empirical, source: EXP-037 (141 rescued / 5,311 true target fields = 2.65%))
  REALISTIC_EXPECTED_GAIN:  +0.2070 pp  (theoretical × recovery)
  Fix type:                 OCR recovery (scanned pages)
  Historical confidence:    HIGH
  Source evidence:          EXP-037 (141 rescued / 5,311 true target fields = 2.65%)
  Recommended next step:    Higher-DPI OCR retry (300 DPI) + adaptive binarization
```

```
FAILURE CLASS: HYPHENATION
  Field count:              7623
  Document breadth:         100 (42.37% of grounded docs)
  THEORETICAL_CEILING:      +1.2252 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  70.00%      (empirical, source: EXP-033 hyphenation analysis)
  REALISTIC_EXPECTED_GAIN:  +0.8576 pp  (theoretical × recovery)
  Fix type:                 Hyphenation join
  Historical confidence:    MEDIUM
  Source evidence:          EXP-033 hyphenation analysis
  Recommended next step:    Trailing hyphen join with next-line token
```

```
FAILURE CLASS: NON_TEXT_BOOLEAN_GROUNDING
  Field count:              2845
  Document breadth:         160 (67.80% of grounded docs)
  THEORETICAL_CEILING:      +16.5056 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  4.70%      (empirical, source: EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%))
  REALISTIC_EXPECTED_GAIN:  +0.7758 pp  (theoretical × recovery)
  Fix type:                 Checkbox visual detection
  Historical confidence:    HIGH
  Source evidence:          EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%)
  Recommended next step:    Extended deterministic checkbox CV provider across all 160 docs
```

```
FAILURE CLASS: DATE_INDEX_MISS
  Field count:              2650
  Document breadth:         134 (56.78% of grounded docs)
  THEORETICAL_CEILING:      +1.1990 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  60.00%      (empirical, source: EXP-032 / EXP-035 deterministic pattern evaluation)
  REALISTIC_EXPECTED_GAIN:  +0.7194 pp  (theoretical × recovery)
  Fix type:                 Date normalization
  Historical confidence:    MEDIUM
  Source evidence:          EXP-032 / EXP-035 deterministic pattern evaluation
  Recommended next step:    Deterministic multi-format regex patterns (YYYY-MMM-DD, DD-MM-YYYY)
```

```
FAILURE CLASS: MULTI_LINE_SPLIT
  Field count:              563
  Document breadth:         6 (2.54% of grounded docs)
  THEORETICAL_CEILING:      +0.1964 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  35.00%      (empirical, source: EXP-033 multi-line recovery estimate)
  REALISTIC_EXPECTED_GAIN:  +0.0687 pp  (theoretical × recovery)
  Fix type:                 Multi-line assembly
  Historical confidence:    LOW
  Source evidence:          EXP-033 multi-line recovery estimate
  Recommended next step:    VisualLine clustering union bbox for vertically adjacent lines
```

