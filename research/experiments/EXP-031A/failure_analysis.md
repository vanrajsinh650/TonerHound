# EXP-031A: Failure Analysis & Root Cause Investigation

**Date**: October 1, 2026  
**Status**: Completed  
**Focus**: Forensic analysis of why EXP-030's +7.73 pp candidate selection gain failed to translate into official ExtractBench Word Grounding F1 under simulated native geometry  
**Production Code Impact**: **ZERO (Production code untouched)**  

---

## 1. Executive Summary

EXP-031A tested the core hypothesis of EXP-030:
> *Did EXP-030 fail to translate its +7.73 pp selection gain into Word Grounding F1 because offline post-hoc evaluation bypassed the production geometry refinement pipeline (`align_to_line_height`, `reconstruct_safe_character_span`, `trim_dot_leaders`)?*

The empirical answer from EXP-031A is an unequivocal **NO**:
1. **IoU Translation Reality**:
   - Out of 6,203 beneficial flips, **5,974 fields (96.31%) ALREADY PASSED $\text{IoU} \ge 0.50$** with raw candidate bounding boxes.
   - Production geometry rescued only **6 fields (0.10%)** from fail ($\text{IoU} < 0.50$) to pass ($\text{IoU} \ge 0.50$).
   - In contrast, geometry refinement actually dropped **223 fields (3.60%)** from passing to failing due to over-trimming.
2. **Official ExtractBench Evaluator Results on Held-Out Cohort B**:
   - Baseline Word Grounding F1: **59.36%**
   - EXP-030 Raw Post-Hoc Word F1: **52.34%** (-7.02 pp)
   - EXP-031A Simulated Native Word F1: **52.19%** (-7.17 pp)
   - EXP-031A Conservative Simulated Word F1: **58.79%** (-0.57 pp)

Geometry pipeline decoupling was **not** the bottleneck. The true failure mechanisms lie in table array topology and ExtractBench's bipartite Hungarian row-matching dynamics.

---

## 2. Root Cause Analysis: The Three True Failure Mechanisms

### Mechanism 1: Hungarian Row Collision & Duplicate Collapse
ExtractBench does not evaluate individual fields in isolation. For tabular arrays (e.g. `transactions[i]`, `checks[i]`, `grants[i]`), ExtractBench runs **maximum-weight bipartite Hungarian matching** between predicted rows and ground-truth rows:
- In `short/593338187_200912_990PF-p0033` (Form 1099-B), transactions 1, 4, and 5 had an ambiguous CUSIP field (`166764100`).
- Because all three records shared a non-unique anchor, the structured assigner pulled all three records to the identical visual line corridor at $y = 0.4995$.
- When Hungarian matching aligned the predicted rows with ground truth, **only one record could pair with the single row at $y=0.4995$**; the other two records collided as duplicate rows and were rejected.
- Rejecting a row invalidates **all** sibling fields in that record (`federal_income_tax_withheld`, `gross_proceeds`, `date_of_sale`, `description`), penalizing precision and recall simultaneously.
- Consequently, `593338187_200912_990PF-p0033` collapsed from **78.53% down to 59.55%** in unconstrained mode.

### Mechanism 2: Header Anchor Collapse
On documents with column headers (`medium/real_pueblo_oct_2025`):
- Column headers like `TYPE`, `DATE`, `CHECK NUMBER` appeared once on the page at $y \approx 0.049$.
- A record whose `type` field had only 1 candidate on the page selected the header token as its anchor.
- All sibling fields in that record were dragged up into the page header margin, causing `real_pueblo_oct_2025` to plunge from **99.60% to 54.71%**.
- The `y >= 0.08` header exclusion safeguard prevented this collapse (lifting Pueblo back to **98.48%**), but residual anchor drift persisted across other tabular documents.

### Mechanism 3: Disparity Between Field-Level Hit@1 and Table Sequence Alignment
- When candidate selection is evaluated field-by-field, picking any token that matches a gold evidence annotation on that page with $\text{IoU} \ge 0.50$ is scored as a "Hit@1".
- In a document with 10 identical values (e.g. `0.00` or `CA` or common dates) across 10 table rows:
  - If field 3 picks candidate 7's box, candidate-level evaluation scores it as a **hit** because it overlaps *some* gold evidence on the page.
  - However, in table-level Hungarian evaluation, field 3 must belong to row 3! If row 3's fields are scattered across different rows, the entire row fails bipartite alignment.
- Thus, field-level Hit@1 (+7.73 pp) was inflated by cross-row token matches that are invalid under strict row-index alignment.

---

## 3. Failure Classification Taxonomy

| Failure Class | Frequency | Mechanism Description | Primary Document Examples |
| :--- | :---: | :--- | :--- |
| `HUNGARIAN_COLLISION` | 44.8% | Multiple records collapsed into the same row corridor due to shared non-unique anchors, causing bipartite rejection. | `short/593338187_200912_990PF-p0033`, `medium/sched_i__united_way_worldwide_ty2024` |
| `CROSS_ROW_MAPPING` | 28.5% | Candidate overlaps a gold box for a *different* row index; counted as Hit@1 locally but rejected by table evaluator. | `medium/real_vg_equity_income_full`, `long/real_ishares_iboxx_bond_etfs` |
| `HEADER_ANCHOR_DRIFT` | 14.2% | Low-y header tokens chosen as anchors, corrupting vertical row pitch. | `medium/real_pueblo_oct_2025`, `short/h5Filing-46302-1` |
| `GEOMETRY_OVERTRIM` | 3.6% | Sub-token character span or dot-leader trimmer reduced an already passing box below $\text{IoU} = 0.50$. | `long/real_freer_register_full`, `medium/real_sm0801_dispositions_full` |
| `ANCHOR_ABSENCE` | 8.9% | Records lacking any distinctive anchor fell back to baseline or drifted arbitrarily. | `long/real_imedia_full_corrupted`, `long/real_ofac_ssi_full` |

---

## 4. Conclusion & Decision Gate Verdict

1. **Geometry Enhancement Decoupling is Disproven**:
   Simulated native geometry refinement (`_apply_geometry_enhancements`) produced **52.19% Word F1** (vs **52.34%** raw), confirming that bounding box character slicing was not the cause of the gap.
2. **Decision Gate D Triggered**:
   EXP-031A Word F1 (**58.79%** conservative / **52.19%** unconstrained) regressed below the **59.36%** baseline.
3. **Mandatory Action**:
   In strict adherence to project safety rules, **STOP STRUCTURED ASSIGNMENT**. Production code remains 100% untouched.
