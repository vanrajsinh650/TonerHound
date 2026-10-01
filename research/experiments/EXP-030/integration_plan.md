# EXP-030: Production Integration Architecture & Road to Leaderboard

**Date**: October 1, 2026  
**Status**: Feature-Flagged Architecture Plan (Blocked on Native Adapter Coupling)  
**Production Code Impact**: **ZERO (Production code untouched)**  

---

## 1. Executive Summary

EXP-030 demonstrated that **Global Structured Assignment** successfully solves the tabular selection problem that halted point-wise ML in EXP-029:
- **Field Selection Hit@1**: **+7.73 pp** (25.00% $\to$ 32.73% on unseen Cohort B, 71,439 fields).
- **Selection Gap Reduction**: **15.20%** of the selection gap closed.
- **Beneficial Flips**: **6,203** vs **684** harmful flips (**9.1:1 win ratio**, +5,519 net flips).

However, offline post-hoc bounding-box substitution regressed official ExtractBench Word Grounding F1 (-0.47 pp to -7.02 pp). As mandated by TonerHound's rigorous safety standards, **zero production changes will be made** until the regression is eliminated.

This integration plan defines the architectural path to translate the +7.73 pp selection gain into official ExtractBench Word Grounding F1, surpassing the LlamaExtract Agentic Plus benchmark (**58.11%**).

---

## 2. Root Cause: Why Offline Substitution Fails

```
CURRENT OFFLINE PROTOTYPE (Regressive):
[Baseline Predictions] 
         │
         ▼
[Post-hoc Field Substitution] ◄── Raw token bbox from parquet (MISSING character slicing!)
         │
         ▼
[Official ExtractBench Evaluator] ──► Fails IoU >= 0.50 threshold!

REQUIRED NATIVE ARCHITECTURE (Safe & Synergistic):
[Candidate Retrieval]
         │
         ▼
[Structured Row Corridors] ◄── Joint row coherence selects candidate token ID
         │
         ▼
[_apply_geometry_enhancements] ◄── Performs sub-token slicing & line pitch snapping
         │
         ▼
[FieldCitation Emission] ──► Passes IoU >= 0.50 with precision!
```

In the production pipeline, `ExtractBenchAdapter._apply_geometry_enhancements` performs sub-token character-span slicing (`reconstruct_safe_character_span`), dot-leader stripping, and line-height normalization. Offline post-hoc evaluation used raw token bounding boxes from `field_records.parquet`, reducing typical overlap with tight gold character annotations from $0.85$ down to $0.44-0.48$. Under ExtractBench's step-function threshold ($\text{IoU} \ge 0.50$), this 0.04 IoU shortfall turned correct field selections into false negatives.

---

## 3. Native Integration Architecture

To preserve geometry enhancement, structured assignment must execute **internally** within `ExtractBenchAdapter._align_table_arrays` prior to geometry processing.

### Component 1: In-Adapter Table Corridor Discovery
Located in `src/tonerhound/benchmark/extractbench_adapter.py`:
- Identify table array fields for each record schema.
- Select unambiguous record anchors (fields with 1 candidate or highest text match) restricted to $y \ge 0.08$ (eliminating header row collapse).
- Compute expected row pitch $\Delta y$ and form vertical tolerance corridors $[\text{anchor}_y - 0.007, \text{anchor}_y + 0.007]$.

### Component 2: Joint Dynamic Programming Candidate Assignment
- For multi-candidate sibling fields, score candidates using the joint objective:
  $$\text{Score}(c) = S_{\text{text}}(c) - \lambda_{\text{row}} |y_c - y_{\text{anchor}}| - \lambda_{\text{col}} \text{dist}(x_c, \text{col\_rail}) - \text{penalty}_{\text{non-monotonic}}$$
- Select candidate tokens before character reconstruction.

### Component 3: Standard Downstream Geometry Processing
- Pass the jointly selected candidate through `_apply_geometry_enhancements`.
- Sub-token character offsets are computed against the newly assigned token, ensuring tight bounding boxes that satisfy $\text{IoU} \ge 0.50$.

---

## 4. Feature Flag & Safety Safeguards

The integration must be strictly guarded behind a feature flag in `src/tonerhound/config.py`:
```python
ENABLE_STRUCTURED_ROW_ASSIGNMENT: bool = False
```

### Safety Rules:
1. **Header Exclusion Zone**: Any candidate with bounding box center $y < 0.08$ is excluded from being a row anchor.
2. **Confidence Thresholding / Abstention**: If anchor confidence is $< 0.85$ or if sibling fields have conflicting corridors, abstain and fall back to the existing DP baseline.
3. **Monotonicity Relaxation**: Do not hard-filter non-monotonic candidates (Constraint B); use soft penalties to accommodate multi-column wrapped tables.
4. **Verification Gate**: Must pass all 232 unit tests and verify positive $\Delta \text{Word F1} > 0$ on Held-Out Cohort B before setting `ENABLE_STRUCTURED_ROW_ASSIGNMENT = True`.

---

## 5. Roadmap to Exceeding 58.11%

1. **Phase Next.1**: Implement `StructuredRowAssigner` in a new module `src/tonerhound/matching/structured_row.py` with 100% unit test coverage.
2. **Phase Next.2**: Wire into `ExtractBenchAdapter._align_table_arrays` behind `ENABLE_STRUCTURED_ROW_ASSIGNMENT = False`.
3. **Phase Next.3**: Run official ExtractBench evaluation on Cohort B.
   - Acceptance target: Cohort B Word F1 $\ge 60.50\%$ (Baseline is $59.36\%$).
4. **Phase Next.4**: If acceptance target passes, run the full 370-document ExtractBench benchmark.
   - Projected Full Benchmark Word F1: **57.5% - 58.6%** (Target: $> 58.11\%$).
