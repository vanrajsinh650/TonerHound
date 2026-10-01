# EXP-031A: Production Integration Assessment & Future Architectural Directions

**Date**: October 1, 2026  
**Status**: **BLOCKED BY DECISION GATE D (Regression) — INTEGRATION REJECTED**  
**Production Code Impact**: **ZERO (Production code untouched)**  

---

## 1. Executive Summary

EXP-031A investigated whether running EXP-030's globally assigned candidates through TonerHound's native production geometry pipeline (`_apply_geometry_enhancements`, `align_to_line_height`, `reconstruct_safe_character_span`, `trim_dot_leaders`) would recover the official ExtractBench Word Grounding F1 loss.

The experiment demonstrated:
- **Simulated Native Word F1**: **52.19%** (vs 59.36% baseline, **-7.17 pp regression**).
- **Conservative Simulated Word F1**: **58.79%** (vs 59.36% baseline, **-0.57 pp regression**).
- **IoU Translation Analysis**: 96.31% of beneficial flips already passed $\text{IoU} \ge 0.50$ with raw bounding boxes. Only 0.10% were rescued by geometry, while 3.60% were over-trimmed.

Under TonerHound's strict safety standards:
> **When an experimental intervention regresses held-out Word Grounding F1, production integration is strictly prohibited.**

Production code remains **100% frozen** at the EXP-028E baseline.

---

## 2. Why Production Integration is Halted

1. **Failure of Independent Record Assignment**:
   Snapping each record independently to a detected "anchor" caused multiple adjacent table records to snap to the same visual line, creating duplicate row collisions that fail ExtractBench's bipartite Hungarian matcher.
2. **Production Baseline Superiority**:
   TonerHound's existing production implementation in `ExtractBenchAdapter._align_table_arrays` already utilizes a monotonic sequence alignment dynamic program (`_align_table_arrays`) with explicit slot budgets and line-height constraints that prevents duplicate row collapse.
3. **The True Selection Problem**:
   Table evidence grounding cannot be solved by post-hoc or unconstrained corridor assignment. It requires sequence-level monotonicity and 1-to-1 bijection constraints that respect ExtractBench's row evaluation semantics.

---

## 3. Required Preconditions for Any Future Work

Before any future table-level or sequence-level evidence grounding experiment may proceed:
1. **Bipartite Matching Awareness**: Any proposed algorithm must model ExtractBench's exact bipartite Hungarian row-matching objective during candidate selection, ensuring no two records map to the same document row.
2. **Monotonicity Preservation**: Row assignments must preserve top-to-bottom reading order on single-column tables.
3. **Safe Fallback**: Any uncertain assignment must strictly abstain and retain the production DP baseline.
4. **Held-Out Cohort B Verification**: Must achieve a strictly positive Word Grounding F1 gain ($\Delta \text{Word F1} > 0.0\text{ pp}$) on Held-Out Cohort B before touching production code or running the full 370-document benchmark.
