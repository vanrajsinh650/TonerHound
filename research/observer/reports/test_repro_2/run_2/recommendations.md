# TonerHound Failure Microscope Diagnostic Report — run_2

## 1. Executive Summary
- **Evaluated Fields**: 470
- **Grounding Successes**: 161
- **Grounding Failures**: 309
- **Total Macro-Weighted Opportunity**: +63.8676 pp

## 2. Macro-Weighted Failure Ranking

| Rank | Failure Class | Fields | Documents | % of Failures | Macro Opp (pp) | % of Macro Opp |
|:----:|:--------------|-------:|----------:|--------------:|---------------:|---------------:|
| 1 | `NON_TEXT_BOOLEAN_GROUNDING` | 141 | 5 | 45.63% | +27.4690 pp | 43.01% |
| 2 | `NO_TEXT_AT_GOLD_REGION` | 102 | 4 | 33.01% | +21.7882 pp | 34.11% |
| 3 | `REAL_INDEXING_MISS` | 41 | 2 | 13.27% | +9.2695 pp | 14.51% |
| 4 | `BBOX_RECONSTRUCTION` | 8 | 2 | 2.59% | +1.5747 pp | 2.47% |
| 5 | `PAGE_ROUTING` | 5 | 1 | 1.62% | +1.1364 pp | 1.78% |
| 6 | `BBOX_TOO_NARROW` | 5 | 2 | 1.62% | +1.0877 pp | 1.70% |
| 7 | `NORMALIZATION_MISMATCH` | 3 | 2 | 0.97% | +0.6331 pp | 0.99% |
| 8 | `OTHER` | 1 | 1 | 0.32% | +0.2273 pp | 0.36% |
| 9 | `HYPHENATION` | 1 | 1 | 0.32% | +0.2273 pp | 0.36% |
| 10 | `DATE_INDEX_MISS` | 1 | 1 | 0.32% | +0.2273 pp | 0.36% |
| 11 | `BBOX_TOO_WIDE` | 1 | 1 | 0.32% | +0.2273 pp | 0.36% |

## 3. Calibrated Failure Class Evaluations (Microscope V4)

```
FAILURE CLASS: NON_TEXT_BOOLEAN_GROUNDING
  Field count:              141
  Document breadth:         5 (100.00% of grounded docs)
  THEORETICAL_CEILING:      +27.4690 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  4.70%      (empirical, source: EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%))
  REALISTIC_EXPECTED_GAIN:  +1.2910 pp  (theoretical × recovery)
  Fix type:                 Checkbox visual detection
  Historical confidence:    HIGH
  Source evidence:          EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%)
  Recommended next step:    Extended deterministic checkbox CV provider across all 160 docs
```

```
FAILURE CLASS: NO_TEXT_AT_GOLD_REGION
  Field count:              102
  Document breadth:         4 (80.00% of grounded docs)
  THEORETICAL_CEILING:      +21.7882 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  2.65%      (empirical, source: EXP-037 (141 rescued / 5,311 true target fields = 2.65%))
  REALISTIC_EXPECTED_GAIN:  +0.5774 pp  (theoretical × recovery)
  Fix type:                 OCR recovery (scanned pages)
  Historical confidence:    HIGH
  Source evidence:          EXP-037 (141 rescued / 5,311 true target fields = 2.65%)
  Recommended next step:    Higher-DPI OCR retry (300 DPI) + adaptive binarization
```

```
FAILURE CLASS: REAL_INDEXING_MISS
  Field count:              41
  Document breadth:         2 (40.00% of grounded docs)
  THEORETICAL_CEILING:      +9.2695 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  6.50%      (empirical, source: EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%))
  REALISTIC_EXPECTED_GAIN:  +0.6025 pp  (theoretical × recovery)
  Fix type:                 Indexing improvement
  Historical confidence:    HIGH
  Source evidence:          EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%)
  Recommended next step:    OCR noise-tolerant bounded n-gram/fuzzy index on OCR pages
```

```
FAILURE CLASS: BBOX_RECONSTRUCTION
  Field count:              8
  Document breadth:         2 (40.00% of grounded docs)
  THEORETICAL_CEILING:      +1.5747 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.1575 pp  (theoretical × recovery)
  Fix type:                 General heuristic
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Targeted forensic audit
```

```
FAILURE CLASS: PAGE_ROUTING
  Field count:              5
  Document breadth:         1 (20.00% of grounded docs)
  THEORETICAL_CEILING:      +1.1364 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.1136 pp  (theoretical × recovery)
  Fix type:                 Page selection reranking
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Document structure global page priors
```

```
FAILURE CLASS: BBOX_TOO_NARROW
  Field count:              5
  Document breadth:         2 (40.00% of grounded docs)
  THEORETICAL_CEILING:      +1.0877 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  25.00%      (empirical, source: EXP-026 estimate (20-30%))
  REALISTIC_EXPECTED_GAIN:  +0.2719 pp  (theoretical × recovery)
  Fix type:                 Bbox geometry refinement
  Historical confidence:    LOW
  Source evidence:          EXP-026 estimate (20-30%)
  Recommended next step:    Multi-token union expansion
```

```
FAILURE CLASS: NORMALIZATION_MISMATCH
  Field count:              3
  Document breadth:         2 (40.00% of grounded docs)
  THEORETICAL_CEILING:      +0.6331 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.0633 pp  (theoretical × recovery)
  Fix type:                 Normalization expansion
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Parenthesized negative numbers + currency code normalization
```

```
FAILURE CLASS: OTHER
  Field count:              1
  Document breadth:         1 (20.00% of grounded docs)
  THEORETICAL_CEILING:      +0.2273 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.0227 pp  (theoretical × recovery)
  Fix type:                 General heuristic
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Targeted forensic audit
```

```
FAILURE CLASS: HYPHENATION
  Field count:              1
  Document breadth:         1 (20.00% of grounded docs)
  THEORETICAL_CEILING:      +0.2273 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  70.00%      (empirical, source: EXP-033 hyphenation analysis)
  REALISTIC_EXPECTED_GAIN:  +0.1591 pp  (theoretical × recovery)
  Fix type:                 Hyphenation join
  Historical confidence:    MEDIUM
  Source evidence:          EXP-033 hyphenation analysis
  Recommended next step:    Trailing hyphen join with next-line token
```

```
FAILURE CLASS: DATE_INDEX_MISS
  Field count:              1
  Document breadth:         1 (20.00% of grounded docs)
  THEORETICAL_CEILING:      +0.2273 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  60.00%      (empirical, source: EXP-032 / EXP-035 deterministic pattern evaluation)
  REALISTIC_EXPECTED_GAIN:  +0.1364 pp  (theoretical × recovery)
  Fix type:                 Date normalization
  Historical confidence:    MEDIUM
  Source evidence:          EXP-032 / EXP-035 deterministic pattern evaluation
  Recommended next step:    Deterministic multi-format regex patterns (YYYY-MMM-DD, DD-MM-YYYY)
```

```
FAILURE CLASS: BBOX_TOO_WIDE
  Field count:              1
  Document breadth:         1 (20.00% of grounded docs)
  THEORETICAL_CEILING:      +0.2273 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  25.00%      (empirical, source: EXP-026 estimate (20-30%))
  REALISTIC_EXPECTED_GAIN:  +0.0568 pp  (theoretical × recovery)
  Fix type:                 Bbox geometry refinement
  Historical confidence:    LOW
  Source evidence:          EXP-026 estimate (20-30%)
  Recommended next step:    Boundary snapping to word edge
```

## 4. Evidence-Based Research Recommendations

### Primary Realistic Research Target: `NON_TEXT_BOOLEAN_GROUNDING`
- **Realistic Expected Gain**: +1.2910 pp Word F1 (Theoretical Ceiling: +27.4690 pp).
- **Empirical Recovery Rate**: 4.70% (Confidence: HIGH, Source: EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%)).
- **Action**: Extended deterministic checkbox CV provider across all 160 docs.

### Secondary Realistic Research Target: `REAL_INDEXING_MISS`
- **Realistic Expected Gain**: +0.6025 pp Word F1 (Theoretical Ceiling: +9.2695 pp).
- **Empirical Recovery Rate**: 6.50% (Confidence: HIGH, Source: EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%)).
- **Action**: OCR noise-tolerant bounded n-gram/fuzzy index on OCR pages.

