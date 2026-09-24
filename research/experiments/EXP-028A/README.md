# EXP-028A: 90% Reachability Audit

## Objective
Classify the 64,130 unclassified zero-candidate fields to determine whether 90% Word Grounding F1 is still plausibly reachable.

## Freeze Baseline
- Commit: `c5fe68a` (EXP-027 Frozen Stack)
- Benchmark Evaluator: Completely Unchanged Official ExtractBench Evaluator

## Key Artifacts
- `sample_manifest.json`: Proportional stratified sampling manifest of N=1,000 fields across 6 dimensions (Seed: 20260924).
- `manual_classification.parquet`: Item-level diagnostic classification of all 1,000 sampled fields.
- `class_distribution.json`: Class distributions, population counts, 95% Wilson confidence intervals, and recoverability categories.
- `reachability_analysis.md`: Detailed reachability analysis, mathematical feasibility test, and Kill/Continue gate decision.

## Verdict
- Visually Recoverable: **100.00%** (95% CI: [99.62%, 100.00%])
- Fundamentally Unrecoverable: **0.00%** (95% CI: [0.00%, 0.38%])
- Gate Decision: **CONTINUE → EXP-028B VISUAL REACHABILITY TEST**
