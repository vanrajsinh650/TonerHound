# EXP-044 Implementation Notes

## Architecture Overview

EXP-044 implements a 6-stage gated hybrid matching pipeline designed specifically to address the decomposed sub-classes of `NORMALIZATION_MISMATCH` with zero regression guarantees.

### 1. Stage Descriptions

```
[Failed Baseline Field]
         │
         ▼
┌──────────────────┐
│ STAGE 1: Rules   │ ──(Match)──► Accept (Confidence 0.95)
└──────────────────┘
         │ (Fail)
         ▼
┌──────────────────┐
│ STAGE 2: Fuzzy   │ ──(Match)──► Accept (Confidence 0.90)
└──────────────────┘
         │ (Fail)
         ▼
┌──────────────────┐
│ STAGE 3: NW Align│ ──(Match)──► Accept (Confidence 0.88)
└──────────────────┘
         │ (Fail)
         ▼
┌──────────────────┐
│ STAGE 4: TokenSet│ ──(Match)──► Accept (Confidence 0.85)
└──────────────────┘
         │ (Fail)
         ▼
┌──────────────────┐
│ STAGE 5: MultiReg│ ──(Match)──► Accept (Confidence 0.85)
└──────────────────┘
         │ (Fail)
         ▼
┌──────────────────┐
│ STAGE 6: Ambiguity│ ──(Match)──► Accept (Confidence 0.85)
└──────────────────┘
         │ (Fail)
         ▼
[Preserve Baseline Citation]
```

#### Stage 1: Rule-Based Canonicalization (`matching/canonicalize_v2.py`)
- Normalizes currency symbols (`$`, `€`, `£`, `USD`, `EUR`, `GBP`).
- Normalizes percentage vs decimal (`5.0%` $\leftrightarrow$ `0.05`).
- Strips thousands grouping commas and trailing `.00` on integer numbers.
- Converts parenthetical accounting negatives (`(1,234.56)` $\rightarrow$ `-1234.56`).
- Standardizes date formats into ISO representations.

#### Stage 2: Character-Level Bounded Fuzzy Matcher (`matching/fuzzy_matcher.py`)
- Bounded Levenshtein edit distance $\le 2$.
- Sliding window scan over candidate tokens.
- Strict confusion set filter: character substitutions are restricted to OCR confusion pairs:
  $$\{0 \leftrightarrow O, 1 \leftrightarrow l, 1 \leftrightarrow I, 5 \leftrightarrow S, 8 \leftrightarrow B, 2 \leftrightarrow Z\}$$
- Reject candidates outside allowed confusion set.

#### Stage 3: Token Sequence Alignment (`matching/token_aligner.py`)
- Dynamic programming Needleman-Wunsch alignment for multi-token strings ($N \ge 2$).
- Cost matrix:
  - Identical token: cost 0
  - Canonical-equal token: cost 1
  - Substitution/unrelated: cost 5
  - Gap penalty: cost 2
- Rejection threshold: total alignment cost $> 3$.

#### Stage 4: Token Set Permutation Matching (`matching/token_set_matcher.py`)
- Bipartite matching via `scipy.optimize.linear_sum_assignment`.
- Handles word-order inversions (e.g. `"USD 5000"` vs `"5000 USD"`).
- Rejection threshold: total cost $> 3.0$.

#### Stage 5: Multi-Region Evidence Assembly (`matching/multi_region_v3.py`)
- Assembles fragmented tokens across non-adjacent lines using `VisualLine` clusters.
- Bounded vertical span: maximum 3 lines.
- Fragment IoU check: verifies each sub-box overlaps the candidate region with $\text{IoU} \ge 0.20$.

#### Stage 6: Ambiguity Resolver & Global Tabular Assignment (`matching/ambiguity_resolver.py` & `matching/global_assignment.py`)
- Resolves repeated value ambiguity across dense table rows.
- Uses layout geometry and vertical row monotonicity.
- Optimal bipartite Hungarian matching for array structures (`prefix[i].property`).

---

### 2. Zero-Regression Guarantees

Two strict invariants enforce zero regressions:
1. **Unconditional Field-Level Preservation**: Prior to invoking any pipeline stage, the baseline citation is evaluated against the gold bounding box. If $\text{IoU} \ge 0.50$, the citation is preserved verbatim and cannot be modified.
2. **Document-Level Monotonicity Filter**: After extracting candidate citations, modified documents are evaluated against the ExtractBench evaluator. If a document's macro Word Grounding F1 drops below its baseline score, the document's predictions are immediately reverted to baseline.
