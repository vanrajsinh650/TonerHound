# TonerHound Failure Microscope Diagnostic Report — EXP-044

## 1. Executive Summary
- **Evaluated Fields**: 498140
- **Grounding Successes**: 326723
- **Grounding Failures**: 171417
- **Total Macro-Weighted Opportunity**: +32.0637 pp

## 2. Macro-Weighted Failure Ranking

| Rank | Failure Class | Fields | Documents | % of Failures | Macro Opp (pp) | % of Macro Opp |
|:----:|:--------------|-------:|----------:|--------------:|---------------:|---------------:|
| 1 | `NO_TEXT_AT_GOLD_REGION` | 58488 | 132 | 34.12% | +18.6884 pp | 58.29% |
| 2 | `NON_TEXT_BOOLEAN_GROUNDING` | 962 | 134 | 0.56% | +5.5411 pp | 17.28% |
| 3 | `REAL_INDEXING_MISS` | 49256 | 102 | 28.73% | +3.7836 pp | 11.80% |
| 4 | `TOKEN_SLICING` | 13941 | 62 | 8.13% | +1.5979 pp | 4.98% |
| 5 | `NORMALIZATION_MISMATCH` | 40620 | 81 | 23.70% | +1.5334 pp | 4.78% |
| 6 | `DATE_INDEX_MISS` | 1891 | 49 | 1.10% | +0.4639 pp | 1.45% |
| 7 | `HYPHENATION` | 5973 | 36 | 3.48% | +0.3573 pp | 1.11% |
| 8 | `MULTI_LINE_SPLIT` | 286 | 4 | 0.17% | +0.0982 pp | 0.31% |

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
  Field count:              49256
  Document breadth:         102 (43.22% of grounded docs)
  THEORETICAL_CEILING:      +3.7836 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  6.50%      (empirical, source: EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%))
  REALISTIC_EXPECTED_GAIN:  +0.2459 pp  (theoretical × recovery)
  Fix type:                 Indexing improvement
  Historical confidence:    HIGH
  Source evidence:          EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%)
  Recommended next step:    OCR noise-tolerant bounded n-gram/fuzzy index on OCR pages
```

```
FAILURE CLASS: TOKEN_SLICING
  Field count:              13941
  Document breadth:         62 (26.27% of grounded docs)
  THEORETICAL_CEILING:      +1.5979 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.1598 pp  (theoretical × recovery)
  Fix type:                 Token slicing refinement
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Narrow character-span reconstruction on wide table cells
```

```
FAILURE CLASS: NORMALIZATION_MISMATCH
  Field count:              40620
  Document breadth:         81 (34.32% of grounded docs)
  THEORETICAL_CEILING:      +1.5334 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.1533 pp  (theoretical × recovery)
  Fix type:                 Normalization expansion
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Parenthesized negative numbers + currency code normalization
```

```
FAILURE CLASS: DATE_INDEX_MISS
  Field count:              1891
  Document breadth:         49 (20.76% of grounded docs)
  THEORETICAL_CEILING:      +0.4639 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  60.00%      (empirical, source: EXP-032 / EXP-035 deterministic pattern evaluation)
  REALISTIC_EXPECTED_GAIN:  +0.2783 pp  (theoretical × recovery)
  Fix type:                 Date normalization
  Historical confidence:    MEDIUM
  Source evidence:          EXP-032 / EXP-035 deterministic pattern evaluation
  Recommended next step:    Deterministic multi-format regex patterns (YYYY-MMM-DD, DD-MM-YYYY)
```

```
FAILURE CLASS: HYPHENATION
  Field count:              5973
  Document breadth:         36 (15.25% of grounded docs)
  THEORETICAL_CEILING:      +0.3573 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  70.00%      (empirical, source: EXP-033 hyphenation analysis)
  REALISTIC_EXPECTED_GAIN:  +0.2501 pp  (theoretical × recovery)
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

### Secondary Realistic Research Target: `DATE_INDEX_MISS`
- **Realistic Expected Gain**: +0.2783 pp Word F1 (Theoretical Ceiling: +0.4639 pp).
- **Empirical Recovery Rate**: 60.00% (Confidence: MEDIUM, Source: EXP-032 / EXP-035 deterministic pattern evaluation).
- **Action**: Deterministic multi-format regex patterns (YYYY-MMM-DD, DD-MM-YYYY).

