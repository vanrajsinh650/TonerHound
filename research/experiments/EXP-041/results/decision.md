# EXP-041 DECISION GATE

## 1. Quantitative Verification
- **Baseline (EXP-040):** 70.3894% Word Grounding F1
- **Result (EXP-041):** 70.5761% Word Grounding F1
- **Net Delta:** +0.1867 pp
- **Regressions:** 0 (Zero Regressions Verified)
- **Net Rescued Fields:** +2953

## 2. Gate Criteria Evaluation
- Score ≥ 74.00%, 0 regressions: **INTEGRATE**
- Score 71.50–74.00%, 0 regressions: **REFINE**
- Score < 71.50% OR any regressions: **REJECT**

## 3. Official Verdict: **REJECT**

### Operational Recommendation:
While EXP-041 achieved a positive net gain (+0.1867 pp, +2,953 fields rescued) with 0 regressions, the achieved Word Grounding F1 of 70.5761% falls below the 71.50% threshold specified in the Decision Gate (Score < 71.50% -> REJECT). In accordance with the project constraints, production code `src/tonerhound/` remains unmodified, and the techniques must be reassessed and fundamentally redesigned before integration.
