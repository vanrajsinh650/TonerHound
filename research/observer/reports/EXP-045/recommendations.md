# TonerHound Failure Microscope Diagnostic Report — EXP-045

## 1. Executive Summary
- **Evaluated Fields**: 498140
- **Grounding Successes**: 338077
- **Grounding Failures**: 160063
- **Total Macro-Weighted Opportunity**: +31.4767 pp

## 2. Macro-Weighted Failure Ranking

| Rank | Failure Class | Fields | Documents | % of Failures | Macro Opp (pp) | % of Macro Opp |
|:----:|:--------------|-------:|----------:|--------------:|---------------:|---------------:|
| 1 | `NO_TEXT_AT_GOLD_REGION` | 58488 | 132 | 36.54% | +18.6884 pp | 59.37% |
| 2 | `NON_TEXT_BOOLEAN_GROUNDING` | 962 | 134 | 0.60% | +5.5411 pp | 17.60% |
| 3 | `REAL_INDEXING_MISS` | 42564 | 102 | 26.59% | +3.5085 pp | 11.15% |
| 4 | `NORMALIZATION_MISMATCH` | 39629 | 78 | 24.76% | +1.4574 pp | 4.63% |
| 5 | `TOKEN_SLICING` | 11697 | 61 | 7.31% | +1.4542 pp | 4.62% |
| 6 | `DATE_INDEX_MISS` | 715 | 47 | 0.45% | +0.3913 pp | 1.24% |
| 7 | `HYPHENATION` | 5722 | 32 | 3.57% | +0.3376 pp | 1.07% |
| 8 | `MULTI_LINE_SPLIT` | 286 | 4 | 0.18% | +0.0982 pp | 0.31% |

## 3. Calibrated Failure Class Evaluations (Microscope V4)

```
FAILURE CLASS: NO_TEXT_AT_GOLD_REGION
  Field count:              58488
  Document breadth:         132 (55.93% of grounded docs)
  THEORETICAL_CEILING:      +18.6884 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  2.65%      (empirical, source: EXP-037 (141 rescued / 5,311 true target fields = 2.65%))
  REALISTIC_EXPECTED_GAIN:  +0.4952 pp  (theoretical × recovery)
  Fix type:                 OCR recovery (scanned pages)
  Historical confidence:    HIGH
  Source evidence:          EXP-037 (141 rescued / 5,311 true target fields = 2.65%)
  Recommended next step:    Higher-DPI OCR retry (300 DPI) + adaptive binarization
```

```
FAILURE CLASS: NON_TEXT_BOOLEAN_GROUNDING
  Field count:              962
  Document breadth:         134 (56.78% of grounded docs)
  THEORETICAL_CEILING:      +5.5411 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  4.70%      (empirical, source: EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%))
  REALISTIC_EXPECTED_GAIN:  +0.2604 pp  (theoretical × recovery)
  Fix type:                 Checkbox visual detection
  Historical confidence:    HIGH
  Source evidence:          EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%)
  Recommended next step:    Extended deterministic checkbox CV provider across all 160 docs
```

```
FAILURE CLASS: REAL_INDEXING_MISS
  Field count:              42564
  Document breadth:         102 (43.22% of grounded docs)
  THEORETICAL_CEILING:      +3.5085 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  6.50%      (empirical, source: EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%))
  REALISTIC_EXPECTED_GAIN:  +0.2281 pp  (theoretical × recovery)
  Fix type:                 Indexing improvement
  Historical confidence:    HIGH
  Source evidence:          EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%)
  Recommended next step:    OCR noise-tolerant bounded n-gram/fuzzy index on OCR pages
```

```
FAILURE CLASS: NORMALIZATION_MISMATCH
  Field count:              39629
  Document breadth:         78 (33.05% of grounded docs)
  THEORETICAL_CEILING:      +1.4574 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.1457 pp  (theoretical × recovery)
  Fix type:                 Normalization expansion
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Parenthesized negative numbers + currency code normalization
```

```
FAILURE CLASS: TOKEN_SLICING
  Field count:              11697
  Document breadth:         61 (25.85% of grounded docs)
  THEORETICAL_CEILING:      +1.4542 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.1454 pp  (theoretical × recovery)
  Fix type:                 Token slicing refinement
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Narrow character-span reconstruction on wide table cells
```

```
FAILURE CLASS: DATE_INDEX_MISS
  Field count:              715
  Document breadth:         47 (19.92% of grounded docs)
  THEORETICAL_CEILING:      +0.3913 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  60.00%      (empirical, source: EXP-032 / EXP-035 deterministic pattern evaluation)
  REALISTIC_EXPECTED_GAIN:  +0.2348 pp  (theoretical × recovery)
  Fix type:                 Date normalization
  Historical confidence:    MEDIUM
  Source evidence:          EXP-032 / EXP-035 deterministic pattern evaluation
  Recommended next step:    Deterministic multi-format regex patterns (YYYY-MMM-DD, DD-MM-YYYY)
```

```
FAILURE CLASS: HYPHENATION
  Field count:              5722
  Document breadth:         32 (13.56% of grounded docs)
  THEORETICAL_CEILING:      +0.3376 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  70.00%      (empirical, source: EXP-033 hyphenation analysis)
  REALISTIC_EXPECTED_GAIN:  +0.2363 pp  (theoretical × recovery)
  Fix type:                 Hyphenation join
  Historical confidence:    MEDIUM
  Source evidence:          EXP-033 hyphenation analysis
  Recommended next step:    Trailing hyphen join with next-line token
```

```
FAILURE CLASS: MULTI_LINE_SPLIT
  Field count:              286
  Document breadth:         4 (1.69% of grounded docs)
  THEORETICAL_CEILING:      +0.0982 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  35.00%      (empirical, source: EXP-033 multi-line recovery estimate)
  REALISTIC_EXPECTED_GAIN:  +0.0344 pp  (theoretical × recovery)
  Fix type:                 Multi-line assembly
  Historical confidence:    LOW
  Source evidence:          EXP-033 multi-line recovery estimate
  Recommended next step:    VisualLine clustering union bbox for vertically adjacent lines
```

## 4. Evidence-Based Research Recommendations

### Primary Realistic Research Target: `NO_TEXT_AT_GOLD_REGION`
- **Realistic Expected Gain**: +0.4952 pp Word F1 (Theoretical Ceiling: +18.6884 pp).
- **Empirical Recovery Rate**: 2.65% (Confidence: HIGH, Source: EXP-037 (141 rescued / 5,311 true target fields = 2.65%)).
- **Action**: Higher-DPI OCR retry (300 DPI) + adaptive binarization.

### Secondary Realistic Research Target: `NON_TEXT_BOOLEAN_GROUNDING`
- **Realistic Expected Gain**: +0.2604 pp Word F1 (Theoretical Ceiling: +5.5411 pp).
- **Empirical Recovery Rate**: 4.70% (Confidence: HIGH, Source: EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%)).
- **Action**: Extended deterministic checkbox CV provider across all 160 docs.

