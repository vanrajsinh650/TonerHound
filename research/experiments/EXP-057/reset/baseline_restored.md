# EXP-057 Part A: Baseline Restored & Verified

**Status:** VERIFIED AND RESTORED  
**Timestamp:** 2026-10-14  
**Verified Macro Word Grounding F1 (237 docs):** **75.7252%**  
**Verified Macro Word Grounding F1 (236 docs):** **75.4585%**  
**Verified Passing Fields:** **418,290 / 498,140**  
**Verified Failing Fields:** **79,850**  
**Regressions:** **Strict 0** across all 370 documents

---

## Restorations Executed
1. **Reverted Fake v0.5.1 Tag**: Deleted remote tag `v0.5.1` from GitHub; reset local `pyproject.toml` and `src/tonerhound/__init__.py` to version `0.5.0`.
2. **Fixed Phase 1 Regression**:
   - Confirmed `long/real_ftx_full` restored to **96.7197% Word F1** (33,017 passing / 34,137 rules).
   - Confirmed `long/real_imedia_full` restored to **94.0298% Word F1** (45,199 passing / 48,070 rules).
3. **Verified Exact Match with EXP-056 Audit**:
   - True Word F1: **75.7252%** ($\Delta = 0.00\text{ pp}$ against audit).
   - Zero regressions verified across all documents.
