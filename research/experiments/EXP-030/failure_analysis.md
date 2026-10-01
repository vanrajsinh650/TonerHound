# EXP-030: Failure Analysis & Metric Divergence Diagnostics

**Date**: October 1, 2026  
**Status**: Completed  
**Focus**: Root-cause analysis of the divergence between Field-Level Selection Hit@1 (+7.73 pp) and Official ExtractBench Word Grounding F1 (-0.47 pp to -7.02 pp)  
**Production Code Impact**: **ZERO (Production code untouched)**  

---

## 1. The Core Empirical Paradox of EXP-030

EXP-030 yielded two seemingly contradictory results on Held-Out Cohort B (32 documents, 71,439 multi-candidate fields):

| Evaluation Lens | Metric | Baseline | EXP-030 | Delta | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Field-Level Candidate Selection** | Multi-Candidate Hit@1 | **25.00%** | **32.73%** | **+7.73 pp** | ✅ Decisive Gain (+5,519 Net Flips) |
| **Field-Level Selection Gap** | Gap Closed | — | — | **15.20%** | ✅ Substantial Closure |
| **Official ExtractBench Evaluator (Raw)** | Word Grounding F1 | **59.36%** | **52.34%** | **-7.02 pp** | ❌ Severe Regression |
| **Official ExtractBench Evaluator (Conservative)** | Word Grounding F1 | **59.36%** | **58.89%** | **-0.47 pp** | ❌ Slight Regression |

In candidate selection, EXP-030 successfully recovered **6,203 previously wrong fields** while regressing only 684 fields (a **9.1:1 beneficial-to-harmful ratio**).  
Yet when evaluated by the official ExtractBench evaluator, Word Grounding F1 declined.

This forensic analysis isolates the three structural mechanisms responsible for this divergence.

---

## 2. Failure Mechanism 1: The Geometry Pipeline Decoupling (Sub-Token IoU Mismatch)

### The Production Pipeline
In TonerHound's production architecture, candidate selection and citation emission are deeply coupled:
```
Candidate Retrieval -> Row Alignment -> _apply_geometry_enhancements -> FieldCitation
```
`_apply_geometry_enhancements` performs four vital transformations on the selected bounding box:
1. `reconstruct_safe_character_span`: Slices multi-character tokens down to the exact sub-token characters of the extracted value (e.g. trimming leading `$`, commas, or enclosing parentheses).
2. `trim_dot_leaders`: Strips dot leaders (`...`) commonly found in accounting schedules and investment tables.
3. `extend_same_line_tokens`: Expands multi-word spans across contiguous token gaps.
4. `align_to_line_height`: Snaps the vertical height to the table's canonical row pitch.

### The Offline Post-Hoc Vulnerability
In an offline candidate reranking simulation, the assigned candidate's `bbox` in `field_records.parquet` is the **raw token bounding box** from candidate generation.
- For a value like `"1250"`, the underlying PDF token may be `"$1,250.00"` with bbox width $0.065$.
- Ground truth evidence covers ONLY the characters `"1250"` with bbox width $0.038$.
- Without sub-token slicing, the IoU between the raw candidate bbox and the gold evidence is $\sim 0.44 - 0.48$.
- Because ExtractBench enforces a strict step-function threshold ($\text{IoU} \ge 0.50$), this 0.04 IoU discrepancy converts a topologically correct field selection into an official evaluation failure!

In the production baseline, `_apply_geometry_enhancements` trims the token to $0.038$, achieving $\text{IoU} = 0.88 \ge 0.50$. When offline assignment blindly substituted raw candidate boxes, hundreds of valid fields in documents like `medium/nport__sei_etf_2025q4` and `medium/sched_i__united_way_worldwide_ty2024` failed solely due to missing character slicing.

---

## 3. Failure Mechanism 2: Header Anchor Collapse (The "Zero/State/Bank" Trap)

In unconstrained table assignment, anchor discovery selects the field in a record with the fewest candidate occurrences.
However, in financial check registers (e.g. `medium/real_pueblo_oct_2025`) and regulatory tables:
- A column header like `BANK` or `TYPE` or `DATE` may appear once on the page at $y \approx 0.049$.
- If `checks[0].bank` or `checks[0].type` has only 1 candidate on the page, the naive assigner selects the header token as the record anchor.
- Once the record anchor is set to $y = 0.049$, all sibling fields (`checks[0].date`, `checks[0].amount`) are dragged up to the header line.
- When `checks[1]` also has an ambiguous bank field, it too collapses onto $y = 0.049$.

This caused `real_pueblo_oct_2025` to collapse from **99.60%** down to **54.77%**.
Restricting anchor candidates to $y \ge 0.08$ and enforcing corridor conservatism eliminated 95% of this defect (restoring Pueblo to 99.60% and lifting Cohort B F1 from 52.34% to 58.89%), but residual anchor drift still caused a -0.47pp regression across edge-case documents.

---

## 4. Failure Mechanism 3: Disparity Between Per-Field IoU and Hungarian Matrix Alignment

ExtractBench does not evaluate fields independently.
In table arrays, ExtractBench runs **Hungarian maximum-weight bipartite matching** across table rows to align predicted rows with gold rows:
1. If structured assignment improves field selection on rows that are already matched, Word F1 improves.
2. But if a single noisy field assignment pulls an entire row into an incorrect Hungarian pairing, **all sibling fields in that row are simultaneously invalidated**, penalizing precision and recall twice.

A single misassigned anchor causes correlated multi-field failure across the entire record.

---

## 5. Summary of Constraint Efficacy

| Constraint | Name | Isolated Effect | Verdict |
| :---: | :--- | :---: | :--- |
| **A** | **Row Coherence** | **+11.47 pp** Hit@1 | **CRITICAL SUCCESS**: The primary driver of closing the selection gap. |
| **B** | **Row Monotonicity** | **-1.14 pp** Hit@1 | **MIXED / HURT**: Naive non-decreasing filtering prunes legitimate multi-column or wrapped table rows. |
| **C** | **Column Rail Consistency** | **+1.31 pp** Hit@1 | **HELPED**: Provides horizontal anchoring that resolves identical numeric values across columns. |
| **D** | **Sibling Proximity** | **+0.85 pp** Hit@1 | **HELPED**: Reinforces row coherence within the same physical visual line. |
| **E** | **Table Membership** | **+0.12 pp** Hit@1 | **NEUTRAL / SAFE**: Prevents cross-table page pollution. |
| **F** | **Conservatism / Abstention** | **+6.55 pp** Word F1 | **VITAL SAFEGUARD**: Prevents catastrophic overwriting of already-grounded high-precision baseline rows. |

---

## 6. Conclusion

The selection gap on tables is **empirically real and solvable** through joint row coherence (+7.73pp Hit@1 gain, +5,519 net flips).
However, post-hoc substitution of raw candidate bounding boxes into prediction files is inherently flawed due to geometry enhancement decoupling.
To realize the +7.73pp selection gain as an official Word Grounding F1 gain, structured assignment must be integrated **inside the adapter's DP alignment engine prior to geometry enhancement**, rather than applied as an external post-hoc replacement.
