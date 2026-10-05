# Official Metric Contract & Evaluator Independence Verification

**Experiment**: EXP-034R (True Text Oracle Re-Measurement + Macro-Weighted Gap Characterization)  
**Evaluator Implementation File**: `research/reference/ExtractBench/src/extract_bench/evaluation/metrics/extract/unified_evidence_metric.py`  
**Evaluator**: `ExtractEvaluator` / `compute_unified_evidence_metrics`

---

## 1. Verified Evaluator Properties

Inspection of `unified_evidence_metric.py` confirms the following mathematical properties:

1. **Row Alignment is Exclusively Value-Driven**:
   - The Hungarian cost matrix between predicted rows `act_rows[j]` and ground-truth rows `exp_rows[i]` is built strictly from value mismatch indicators (`mismatch_cost_matrix`, lines 602–615).
   - Citation bounding boxes, pages, and confidence values do **NOT** enter the cost matrix.
2. **Fixed Identity Row Alignment**:
   - Because TonerHound evaluated predictions operate on extraction payloads where `extracted_data == expected_output`, the diagonal value cost is identically zero ($C_{ii} = 0$).
   - Hungarian matching via `scipy.optimize.linear_sum_assignment` assigns row $i$ to gold row $i$ for all tables ($j = i$).
3. **Strict Per-Cell Grounding Credit**:
   - For each aligned cell path $p$ (e.g. `line_items[3].unit_price` or scalar `invoice_number`), grounding credit is awarded if and only if:
     $$\exists (gp, gb) \in \text{ev\_boxes}[p], (pp, pb) \in \text{pred\_boxes}[p] : gp == pp \land \text{IoU}(gb, pb) \ge 0.50$$
4. **Independent Cell Decoupling**:
   - The credit for cell $p$ depends *only* on whether the citation for path $p$ matches any gold box in $\text{ev\_boxes}[p]$.
   - Field $p$'s citation has zero cross-coupling with field $q$'s citation.
   - Field $p$'s citation has zero effect on the Hungarian alignment of any other rows or fields.
5. **Precision Penalty Mechanics on Ungroundable Cells**:
   - Under `_note_grounded_claim` (lines 379–383):
     $$\text{If } \text{ev\_boxes}[p] \neq \emptyset \land \text{pred\_boxes}[p] \neq \emptyset \implies g\_claims \leftarrow g\_claims + 1$$
   - Emitting an invalid candidate ($\text{IoU} < 0.50$) increments the precision denominator $g\_claims$ without incrementing $g\_correct$, reducing Precision.
   - Emitting a valid candidate ($\text{IoU} \ge 0.50$) increments both $g\_correct$ and $g\_claims$, maximizing both Recall and Precision.

---

## 2. Independence Theorem for Optimal Evidence Assignment

> **Theorem (Field-Level Oracle Independence)**:
> Given fixed correct predicted values and identity Hungarian row alignment, the global official Word Grounding F1 is strictly maximized by solving each field independently:
> For each gradeable field path $p$:
> 1. If $\exists c \in \mathcal{C}_p$ such that $c.\text{page} == gp$ and $\text{IoU}(c.\text{bbox}, gb) \ge 0.50$:
>    Assign $c^* = \arg\max_{c \in \mathcal{C}_p} \text{IoU}(c.\text{bbox}, gb)$.
> 2. If no qualifying candidate exists in $\mathcal{C}_p$:
>    Retain baseline emission (Mode A) or abstain/omit citation (Mode B).
>
> **Proof**:
> Let $F1 = 2 \frac{P \cdot R}{P + R} = \frac{2 g\_correct}{g\_claims + g\_expected}$.
> Since $g\_expected$ is constant for the document, $F1$ is strictly monotonically increasing with respect to $g\_correct$.
> Furthermore, $g\_correct = \sum_{p \in \text{fields}} \mathbb{I}(\text{hit}(p))$.
> Because the hit indicator $\mathbb{I}(\text{hit}(p))$ depends solely on $c_p$ and is independent of $c_q$ for all $q \neq p$, maximizing each $\mathbb{I}(\text{hit}(p))$ independently maximizes $\sum_p \mathbb{I}(\text{hit}(p)) = g\_correct$.
> Under Mode A (baseline emission), $g\_claims$ is constant, so maximizing $g\_correct$ unconditionally maximizes $F1$.
> Under Mode B (optimal omission), omitting ungroundable claims strictly minimizes $g\_claims$ while preserving maximum $g\_correct$, yielding the absolute global maximum of $F1$. $\blacksquare$

---

## 3. Practical Implications for EXP-034R

1. **No complex joint table combinatorial solver is required for the oracle**: Because row alignment is fixed to identity, optimal candidate selection decomposes into independent per-field optimization across all 445,950 fields.
2. **Oracle Construction is Exact and Exhaustive**: For every field, we evaluate all available text/OCR candidates against the gold evidence and select the maximal-credit candidate.
