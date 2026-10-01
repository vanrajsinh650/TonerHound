# EXP-032: True Record-Level Oracle under the Official ExtractBench Evaluator

## 1. Executive Summary

EXP-031A falsified the hypothesis that geometry integration caused the disconnect between selection gains and Word Grounding F1. Furthermore, field-level Hit@1 was proven to be misleading for repeated table values (e.g. repeated states or numbers like `CA`, `0.00` across table rows), where matching *any* gold box on the page produced false-positive selection credit that the official evaluator penalized.

**EXP-032 is a diagnostic ceiling experiment.**
The core question answered by EXP-032 is:
> **How much Word Grounding F1 is actually achievable from the current candidate pool when candidate selection is optimized against the unmodified official ExtractBench evaluator?**

---

## 2. Research Discipline & Absolute Constraints

1. **Production Code Untouched**: No modifications to `src/tonerhound/`.
2. **Evaluator Untouched**: The official ExtractBench evaluator (`extract_bench.evaluation.evaluators.extract.ExtractEvaluator` and `unified_evidence_metric.py`) must remain completely unmodified.
3. **No Proxy Metrics as Ceilings**: Hit@1, candidate recall, and independent field IoU are NOT ceilings. The ceiling is defined solely by the official evaluator's Word Grounding F1.
4. **No Synthesized / Invented Evidence**: Every evidence region selected by the oracle MUST originate from the existing candidate pool in `research/observer/field_records.parquet` or baseline citations.
5. **No Extrapolations**: Evaluated on Held-Out Cohort B first, followed by an exhaustive evaluation across all 370 documents of the official benchmark.

---

## 3. Official Evaluator Mechanics Summary

As documented in [`evaluator_mechanics.md`](./evaluator_mechanics.md):
1. **Value-Only Hungarian Assignment**: Hungarian table row alignment operates exclusively on text/numeric values. Bounding boxes and pages do not enter the cost matrix.
2. **Identity Alignment**: Because extracted values match expected values along the diagonal, Hungarian row assignment between predicted and gold rows is the exact identity mapping ($i \leftrightarrow i$).
3. **Per-Cell Grounding**: Once row $i$ is mapped to gold row $i$, each subfield path `table[i].col` is evaluated strictly against the gold evidence for `table[i].col`.
4. **Credit Criterion**: Grounded credit is awarded if and only if:
   $$\exists (gp, gb) \in \text{ev\_boxes}[table[i].col], (pp, pb) \in \text{pred\_boxes}[table[i].col] : gp == pp \land \text{IoU}(gb, pb) \ge 0.50$$
5. **Precision Penalty on Ungroundable Cells**: If a predicted cell emits a citation for a cell that has gold bboxes, but fails $\text{IoU} \ge 0.50$, it increments $g\_claims$ without incrementing $g\_correct$, reducing Precision.

---

## 4. Mathematical Oracle Formulation

For every field path $p$ (both scalar fields and table cell paths `table[i].col`):
1. Gold evidence for $p$ is retrieved: $G_p = \{(gp, gb) \in \text{evidence}(p) \mid gb \text{ is not None}\}$.
2. Candidate pool for $p$ is retrieved from `field_records.parquet`: $C_p = \{c_1, c_2, \dots, c_K\}$.
3. If $G_p \neq \emptyset$:
   - We check if $\exists c \in C_p$ such that $\exists (gp, gb) \in G_p$ with $c.\text{page} == gp$ and $\text{IoU}(c.\text{bbox}, gb) \ge 0.50$.
   - If one or more such candidates exist, the oracle selects the candidate $c^*$ that maximizes $\text{IoU}(c.\text{bbox}, gb)$. This guarantees $g\_correct += 1$ and $g\_claims += 1$.
   - If NO candidate in $C_p$ achieves $\text{IoU} \ge 0.50$ for $G_p$:
     - *Mode A (Baseline Emission)*: Retain baseline candidate selection to measure ceiling under current emission policy.
     - *Mode B (Optimal Abstention / Omission)*: Withhold citation for $p$ so $g\_claims$ is not incremented, maximizing Precision without loss of Recall.
4. If $G_p = \emptyset$ (cell has no gold bbox):
   - Withhold citation or omit bbox (does not affect $g\_claims$ or $g\_correct$ since ungradeable).

---

## 5. Decision Gates

- **Oracle F1 $\ge 75\%$**: Large selection headroom exists. Candidate pool is healthy. Proceed to record-level selector design.
- **Oracle F1 $65\% - 75\%$**: Selection can contribute materially, but selection alone cannot reach high 80s/90s.
- **Oracle F1 $60\% - 65\%$**: Selection headroom is limited. Prioritize candidate generation redesign.
- **Oracle F1 $< 60\%$**: Current candidate pool provides almost no recoverable headroom over baseline (56.05% full / 59.36% Cohort B). STOP selection research.

---

## 6. Official Verified Results

### A. Held-Out Cohort B (32 Unseen Documents)

| Metric | Production Baseline | Mode A (Selection Oracle) | Mode B (Optimal Abstention) | Headroom (Mode A) |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | **59.3588%** | **66.2218%** | **67.7861%** | **+6.86 pp** |
| Word Precision | 63.7774% | 71.0432% | 99.7997% | +7.27 pp |
| Word Recall | 56.5164% | 63.0824% | 56.8802% | +6.57 pp |
| Page Grounding F1 | 85.4408% | 87.7680% | 61.3592% | +2.33 pp |

### B. Full Official 370-Document Benchmark

| Metric | Production Baseline | Mode A (Selection Oracle) | Mode B (Optimal Abstention) | Headroom (Mode A) |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | **56.0477%** | **60.9591%** | **62.5639%** | **+4.91 pp** |
| Word Precision | 61.7275% | 67.0437% | 99.0757% | +5.32 pp |
| Word Recall | 52.5686% | 57.1762% | 50.3276% | +4.61 pp |
| Page Grounding F1 | 81.6639% | 82.4392% | 46.3507% | +0.78 pp |

---

## 7. Decision Gate & Final Verdict

**TRIGGERED GATE**: **GATE 3 — ORACLE 60% – 65% (LIMITED SELECTION HEADROOM)**  
$$\text{Official Oracle F1 (Mode A)} = 60.9591\% \in [60.00\%, 65.00\%]$$

**MANDATED DECISION**: **RESEARCH CANDIDATE GENERATION**

Selection research (point-wise reranking, joint Hungarian table assignment, post-hoc substitution) is formally **STOPPED**. Research shifts to **Candidate Generation and Visual Perception** (addressing the 141,238 zero-candidate fields).

