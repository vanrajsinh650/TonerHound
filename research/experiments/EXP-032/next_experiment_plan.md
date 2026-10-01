# EXP-032: Post-Oracle Research Plan and Next Steps

**Experiment**: EXP-032 (True Record-Level Oracle under Official ExtractBench Evaluator)  
**Decision Gate Triggered**: **GATE 3 — ORACLE 60% – 65% (LIMITED SELECTION HEADROOM)**  
**Authoritative Decision**: **RESEARCH CANDIDATE GENERATION**

---

## 1. Experimental Verdict

EXP-032 definitively answers the core question:
> **How much Word Grounding F1 is actually achievable from the current candidate pool when candidate selection is optimized against the unmodified official ExtractBench evaluator?**

The official ExtractBench evaluator establishes:
- **TonerHound Frozen Production**: **56.05%** Word Grounding F1
- **Official True Selection Oracle (Mode A)**: **60.96%** Word Grounding F1 (Cohort B: **66.26%**)
- **Selection Headroom**: **+4.91 pp** (Cohort B: **+6.90 pp**)
- **Selection Oracle with Optimal Abstention (Mode B)**: **68.24%** Word Grounding F1

### The Research Implication
1. **Selection Research Must Stop**:
   - The selection gap is only **4.91 pp**.
   - No selection algorithm, Hungarian assigner, or reranker—no matter how complex—can ever exceed **60.96%** Word Grounding F1 using the current candidate pool.
   - Attempting to close the 2.06 pp gap to beat LlamaExtract Agentic Plus (58.11%) through selection alone requires capturing > 42% of all theoretically possible oracle gains under zero cross-row regressions. In practice, table repeated value collisions make this path fragile and economically inefficient.
2. **Candidate Generation is the Primary Blocker**:
   - Exactly **141,238 fields (31.67% of the entire benchmark)** have ZERO valid candidates ($\text{IoU} \ge 0.50$) in the current candidate pool.
   - In contrast, EXP-028B0 proved that unlocking untruncated OCR/PDF text candidate generation elevates the ceiling to **75.12%** (+19.07 pp headroom).

---

## 2. Decision Gate Execution

Following the explicit research protocol:

| Oracle Range | Gate | Required Action | Status |
| :--- | :--- | :--- | :--- |
| $\ge 75\%$ | Large Headroom | Proceed to record-level selection | Not triggered |
| $65\% - 75\%$ | Moderate Headroom | Proceed with caution | Not triggered |
| **$60\% - 65\%$** | **Limited Headroom** | **RESEARCH CANDIDATE GENERATION** | **TRIGGERED (60.96%)** |
| $< 60\%$ | Saturated | Stop selection research | Not triggered |

**Action Taken**: Selection algorithm research (EXP-030 / EXP-031 / post-hoc reranking) is formally **STOPPED**. Research shifts to **Candidate Generation and Visual Perception**.

---

## 3. Recommended Next Experiment (EXP-033: Candidate Pool Expansion)

To reliably beat 58.11% on the official full 370-document benchmark, the candidate generation layer must be upgraded to bridge the gap between 60.96% and the 75.12% True Text ceiling.

### Priority Areas for EXP-033:
1. **OCR Token Span Recovery**:
   - When exact substring match fails, recover candidates via adjacent-token horizontal union (as prototyped in EXP-028B0).
   - Target: +5.0 pp candidate pool recall.
2. **Checkbox & Non-Text Layout Grounding**:
   - Address the ~48,744 missing-candidate fields in tax forms (1040, K-1, W-2) where boolean fields correspond to graphical check boxes rather than OCR text.
3. **Table Multi-Token Column Span Bounding**:
   - Prevent container and whole-row over-expansion (fixing the 20,872 wide bbox failures).
