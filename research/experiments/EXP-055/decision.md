# EXP-055: Decision Record

## Evaluation Summary
- **Baseline (EXP-054):** 75.2099% Word F1 (237 docs), 411,098 passing fields.
- **Post Mega-Attack (EXP-055):** **80.5120% Word F1** (237 docs), **466,099 passing fields**.
- **Net Fields Rescued:** **+55,001 fields** across 10 sequential phases.
- **Regressions:** Strict **0** across all 370 documents (498,140 fields).
- **Unit Test Suite:** 276 / 276 passed (100%).

---

## Production Release Gate Rules
- If Word F1 >= 90%: Ship v1.0.0
- If Word F1 >= 85%: Ship v0.6.0
- If Word F1 < 85%: Ship **v0.5.1** with honest report

**Score Achieved:** **80.5120%**  
**Gate Result:** < 85.0% threshold for v0.6.0.  
**Version Shipped:** **v0.5.1**

---

## Decision
**DECISION: RELEASE v0.5.1 AND CONTINUE**

1. **Release v0.5.1**: Ship the accepted fixes from Phases 1 through 10 as release `v0.5.1`, solidifying TonerHound crossing the **80% Word Grounding F1 milestone (80.5120%)**.
2. **Deterministic Ceiling Confirmed Reachable**: The myth of an 80% ceiling is broken. With 466,099 passing fields (93.57% passing volume), the remaining 32,041 failures are strictly bounded.
3. **Next Priority (EXP-056)**: Target the remaining 32,041 fields with special focus on corrupted bankruptcy OCR repair (`real_imedia_full_corrupted` remaining ~31,871 fields) to push directly through 85% to 90%.
