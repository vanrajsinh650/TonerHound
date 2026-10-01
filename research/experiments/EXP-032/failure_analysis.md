# EXP-032: Evaluator-Grounded Failure Analysis

**Experiment**: EXP-032 (True Record-Level Oracle under the Official ExtractBench Evaluator)  
**Dataset**: Full 370-Document ExtractBench Benchmark (445,950 gradeable fields across 236 grounded documents)

---

## 1. Executive Summary

EXP-032 evaluates the maximum Word Grounding F1 achievable through candidate selection alone, under the exact, unmodified ExtractBench evaluator.

The empirical findings establish that:
1. **Selection Headroom is Narrow (+4.91 pp)**:
   - Current Production Baseline: **56.05%** Word Grounding F1.
   - True Record-Level Selection Oracle (Mode A): **60.96%** Word Grounding F1 (Cohort B: **66.26%**).
   - Selection Gap: **+4.91 pp** on the full benchmark.
2. **The 39.04 pp Remaining Gap is Not a Selection Problem**:
   - Out of 445,950 gradeable fields in the official benchmark, only **304,712 fields (68.33%)** have ANY candidate in the pool with $\text{IoU} \ge 0.50$.
   - Exactly **141,238 gradeable fields (31.67%)** have **ZERO valid candidates** in the candidate pool.
   - No selection algorithm, Hungarian assigner, or reranker can ever ground these fields, because no candidate exists to be selected.

---

## 2. Root Cause Breakdown of Unreachable Grounding

From the full-population inventory analysis of all 445,950 gradeable fields:

| Failure Mechanism | Field Count | % of Benchmark | Root Cause Description |
| :--- | :--- | :--- | :--- |
| **Candidate Missing from Pool** | 48,744 | 10.93% | Resolver produced zero candidates for the field value (e.g. OCR transcription differences, non-text checkboxes, dates formatted differently). |
| **BBox Geometry Narrow** | 5,809 | 1.30% | Candidate bbox spans only a portion of the true multi-token or wrapped span ($\text{IoU} < 0.50$). |
| **BBox Geometry Wide** | 20,872 | 4.68% | Candidate bbox captures the entire table row, header banner, or container rather than the isolated cell value. |
| **OCR Geometry & Coordinate Drift** | 15,352 | 3.44% | Token bounding box drift due to skewed scans, low DPI, or OCR word boundary misalignment ($\text{IoU} \in [0.20, 0.49]$). |
| **Retrieval Wrong Page** | 7,921 | 1.78% | Candidate was only retrieved on a non-target page (e.g. repeated cover page text, recurring headers/footers). |
| **Repeated Value Ambiguity** | 16,257 | 3.65% | Multiple table rows share identical text (e.g., `0.00`, `None`, `Active`, `N/A`, `CA`). Local candidates are indistinguishable. |
| **Selected Citation Geometry Failure** | 57,856 | 12.97% | Candidate rank 1 had correct text match but failed $\text{IoU} \ge 0.50$ due to dot-leaders, line snapping, or character span truncation. |
| **Low-Rank Hits (Rank 6–20+)** | 13,873 | 3.11% | Valid candidates exist but are buried deep in candidate pools ($> 5$ candidates), beyond the reach of greedy search. |

---

## 3. Table Value Collisions and Evaluator Alignment

In table extraction, fields often repeat across rows:
- **Total table fields**: 342,109
- **Repeated table value fields**: 184,520 (53.94% of all table fields)

### The Collision Trap
1. In ExtractBench, table row alignment (`_score_array`) uses Hungarian assignment strictly on **value matching**.
2. Because TonerHound passes extracted data matching expected output, Hungarian row assignment is fixed to the **identity mapping** ($i \leftrightarrow i$).
3. When multiple rows share identical text, naive or point-wise rerankers select the single candidate that scores highest locally (e.g., the top-most `CA` or the bold `0.00`).
4. As a result, multiple predicted rows emit citations pointing to the exact same bounding box on the page.
5. In official evaluation:
   - Row 1 matches gold row 1 ($\text{IoU} \ge 0.50 \to \text{TP}$).
   - Row 2 cites row 1's bbox instead of row 2's bbox ($\text{IoU} = 0.00 \to \text{FP}$ and $\text{FN}$).
   - Row 3 cites row 1's bbox ($\text{IoU} = 0.00 \to \text{FP}$ and $\text{FN}$).
6. This collision dynamic explains why EXP-029 point-wise reranker degraded performance (harmful flips outnumbered beneficial flips) and why EXP-030 field-level Hit@1 (+7.73 pp) was illusory.

---

## 4. Why Selection Cannot Beat LlamaExtract Agentic Plus Alone

- **Official ExtractBench Leaderboard**:
  - **LlamaExtract Agentic Plus**: **58.11%** Word Grounding F1.
  - **TonerHound Frozen Production**: **56.05%** Word Grounding F1.
  - **TonerHound True Candidate Oracle (Mode A)**: **60.96%** Word Grounding F1.
- While the theoretical ceiling (60.96%) is above 58.11%, the total headroom available to candidate selection is only **4.91 pp**.
- A production selector would need to achieve **> 42% of all theoretically possible oracle selection gains** without suffering even minor cross-row regressions to surpass 58.11%.
- In contrast, expanding candidate generation (as demonstrated by EXP-028B0 True Text Oracle = 75.12%) unlocks **19.07 pp of headroom**, providing a much wider and safer path to surpassing 58.11%.
