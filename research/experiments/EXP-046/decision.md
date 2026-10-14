# EXP-046: Architectural Decision & Integration Recommendation

## Final Decision: REFINE (Do NOT integrate all 34 stages directly into production)

### Justification & Analysis

1. **Held-Out Evaluation Gate**:
   - Rescues: +14 fields on held-out cohort (against EXP-045 baseline).
   - Regressions: 0 (Strict zero regressions verified across all 32 documents).
   - Latency: 1.25x increase factor (within <= 5x threshold).
   - The gate threshold of >= 3,000 marginal rescues over EXP-045 was **not met** on the held-out set alone because EXP-045 had already integrated the primary table mapping gains (+8,303 fields), leaving a long-tail distribution where general fallbacks have diminishing returns.

2. **Benchmark Results (Full 370 Documents)**:
   - Word Grounding F1: 72.8448% (identical to EXP-045 baseline, delta +0.0000 pp).
   - Page Grounding F1: **90.0061%** (+6.1035 pp gain over 83.9026%).
   - Total Net Rescued Fields: **+32** fields with **0 regressions**.

3. **Stage Marginal Contribution**:
   - Out of 34 parallel stages, only **5 stages** produced non-zero rescues in the benchmark:
     - `stage_07_row_index_mapper`: 24 rescues (75.0% of total)
     - `stage_20_char_span_slicer`: 4 rescues (12.5%)
     - `stage_16_numeric_canonical_v2`: 2 rescues (6.25%)
     - `stage_02_fuzzy_matcher`: 1 rescue (3.125%)
     - `stage_15_consonant_skeleton`: 1 rescue (3.125%)
   - 29 stages produced 0 marginal rescues across the entire benchmark.

### Stages to Promote into Production
Promote only the 5 empirically verified, zero-regression stages:
1. `stage_07_row_index_mapper` (with line-wrap merging and array-bounds early exit)
2. `stage_20_char_span_slicer` (sub-token character-span slicing)
3. `stage_16_numeric_canonical_v2` (numeric float tolerance canonicalization)
4. `stage_02_fuzzy_matcher` (spatial-constrained Levenshtein fuzzy matching)
5. `stage_15_consonant_skeleton` (consonant skeleton matching)

### Stages to Reject / Prune
Reject all 29 stages that yielded 0 marginal rescues to eliminate dead weight and latency overhead:
- Stages 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34.

### Next Priority Recommendation
Focus on the dominant remaining failure class: **REAL_INDEXING_MISS** (48,120 fields, 31.3% of all remaining failures). The system currently lacks an index-level semantic anchor when target strings are omitted or expressed under different header hierarchies.
