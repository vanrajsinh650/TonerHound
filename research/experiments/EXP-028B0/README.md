# EXP-028B0: True Text Ceiling

## Objective
Determine how high TonerHound can get using only the document's existing text + geometry
if there is absolutely no candidate truncation.

## Results
- Production Baseline: **55.98%**
- Old Ceiling B: **64.58%**
- **TRUE TEXT ORACLE: 75.12%** (+10.55pp vs Old Ceiling B, +19.15pp vs Production)

## Artifacts
- `ceiling_b_audit.md`: Analysis of all 10 information-loss points in Old Ceiling B.
- `results.json`: Complete official metrics and cohort breakdowns.
- `true_text_ceiling_report.md`: Full technical evaluation report.
