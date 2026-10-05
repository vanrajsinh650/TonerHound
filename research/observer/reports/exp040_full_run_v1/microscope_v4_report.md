# Failure Microscope V4 Causal Audit — exp040_full_run_v1

**Date:** 2026-10-04T08:25:07.255639+00:00  
**Word Grounding F1:** 70.3894% (+1.2567 pp vs EXP-039 baseline)  
**Passing Fields:** 320,159 (+4817 rescued)  
**Failing Fields:** 177,981  
**Total Realistic Remaining Opportunity:** +2.5468 pp (Theoretical Ceiling: +22.9739 pp)

## Calibrated Failure Class Breakdown

```
FAILURE CLASS: REAL_INDEXING_MISS
  Field count:              81069
  Document breadth:         0 (0.00% of grounded docs)
  THEORETICAL_CEILING:      +10.3804 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  6.50%      (empirical, source: EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%))
  REALISTIC_EXPECTED_GAIN:  +0.6747 pp  (theoretical × recovery)
  Fix type:                 Indexing improvement
  Historical confidence:    HIGH
  Source evidence:          EXP-035 (+0.2293 pp / +3.4991 pp = 6.55%)
  Recommended next step:    OCR noise-tolerant bounded n-gram/fuzzy index on OCR pages

FAILURE CLASS: NORMALIZATION_MISMATCH
  Field count:              46156
  Document breadth:         0 (0.00% of grounded docs)
  THEORETICAL_CEILING:      +5.9100 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.5910 pp  (theoretical × recovery)
  Fix type:                 Normalization expansion
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Parenthesized negative numbers + currency code normalization

FAILURE CLASS: TOKEN_SLICING
  Field count:              23542
  Document breadth:         0 (0.00% of grounded docs)
  THEORETICAL_CEILING:      +3.0144 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  10.00%      (empirical, source: Conservative prior (10%))
  REALISTIC_EXPECTED_GAIN:  +0.3014 pp  (theoretical × recovery)
  Fix type:                 Token slicing refinement
  Historical confidence:    LOW
  Source evidence:          Conservative prior (10%)
  Recommended next step:    Narrow character-span reconstruction on wide table cells

FAILURE CLASS: NO_TEXT_AT_GOLD_REGION
  Field count:              14983
  Document breadth:         0 (0.00% of grounded docs)
  THEORETICAL_CEILING:      +1.9185 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  2.65%      (empirical, source: EXP-037 (141 rescued / 5,311 true target fields = 2.65%))
  REALISTIC_EXPECTED_GAIN:  +0.0508 pp  (theoretical × recovery)
  Fix type:                 OCR recovery (scanned pages)
  Historical confidence:    HIGH
  Source evidence:          EXP-037 (141 rescued / 5,311 true target fields = 2.65%)
  Recommended next step:    Higher-DPI OCR retry (300 DPI) + adaptive binarization

FAILURE CLASS: HYPHENATION
  Field count:              7623
  Document breadth:         0 (0.00% of grounded docs)
  THEORETICAL_CEILING:      +0.9761 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  70.00%      (empirical, source: EXP-033 hyphenation analysis)
  REALISTIC_EXPECTED_GAIN:  +0.6833 pp  (theoretical × recovery)
  Fix type:                 Hyphenation join
  Historical confidence:    MEDIUM
  Source evidence:          EXP-033 hyphenation analysis
  Recommended next step:    Trailing hyphen join with next-line token

FAILURE CLASS: NON_TEXT_BOOLEAN_GROUNDING
  Field count:              2840
  Document breadth:         0 (0.00% of grounded docs)
  THEORETICAL_CEILING:      +0.3636 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  4.70%      (empirical, source: EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%))
  REALISTIC_EXPECTED_GAIN:  +0.0171 pp  (theoretical × recovery)
  Fix type:                 Checkbox visual detection
  Historical confidence:    HIGH
  Source evidence:          EXP-036D (185 rescued / 2,847 fields, +0.776 pp / +16.45 pp = 4.7%)
  Recommended next step:    Extended deterministic checkbox CV provider across all 160 docs

FAILURE CLASS: DATE_INDEX_MISS
  Field count:              2646
  Document breadth:         0 (0.00% of grounded docs)
  THEORETICAL_CEILING:      +0.3388 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  60.00%      (empirical, source: EXP-032 / EXP-035 deterministic pattern evaluation)
  REALISTIC_EXPECTED_GAIN:  +0.2033 pp  (theoretical × recovery)
  Fix type:                 Date normalization
  Historical confidence:    MEDIUM
  Source evidence:          EXP-032 / EXP-035 deterministic pattern evaluation
  Recommended next step:    Deterministic multi-format regex patterns (YYYY-MMM-DD, DD-MM-YYYY)

FAILURE CLASS: MULTI_LINE_SPLIT
  Field count:              563
  Document breadth:         0 (0.00% of grounded docs)
  THEORETICAL_CEILING:      +0.0721 pp  (upper bound, not achievable)
  REALISTIC_RECOVERY_RATE:  35.00%      (empirical, source: EXP-033 multi-line recovery estimate)
  REALISTIC_EXPECTED_GAIN:  +0.0252 pp  (theoretical × recovery)
  Fix type:                 Multi-line assembly
  Historical confidence:    LOW
  Source evidence:          EXP-033 multi-line recovery estimate
  Recommended next step:    VisualLine clustering union bbox for vertically adjacent lines

```
