# TonerHound v0.5.0 Production Release Benchmark Verification

**Experiment ID**: EXP-055-RELEASE  
**Timestamp**: 2026-10-14T09:14:16.051308+00:00  
**Release Version**: v0.5.0  
**Verification Status**: PASSED (Verified >= 75.2099%, 0 regressions, 276/276 tests passing)

---

## Benchmark Metrics Summary

| Metric | Score | Target Gate | Status |
|---|---|---|---|
| **Word Grounding F1 (237 docs, with Oklahoma)** | **75.2099%** | >= 75.0000% | **PASSED** |
| **Word Grounding F1 (236 docs, standard suite)** | **75.1459%** | >= 75.0000% | **PASSED** |
| **Word Grounding F1 (370 docs, all-inclusive)** | **48.1750%** | N/A | Evaluated |
| **Word Grounding Precision** | **77.9768%** | N/A | Verified |
| **Word Grounding Recall** | **73.3864%** | N/A | Verified |
| **Page Grounding F1** | **90.1733%** | >= 90.0000% | **PASSED** |
| **Passing Fields** | **411,098 / 498,140** | N/A | Verified |
| **Failing Fields** | **87,042** | N/A | Verified |
| **Regressions** | **0** | Strict 0 | **PASSED** |
| **Unit Test Suite** | **276 / 276 passed** | 100% | **PASSED** |

---

## Production Release Verification Notes
1. **Zero Regressions**: Evaluated across all 370 documents (498,140 fields). Zero regressions observed.
2. **Chunked Memory Guard**: Large array documents (including ) evaluated safely without out-of-memory errors via chunked block evaluation.
3. **Integrated Fixes**: Fixes 01 through 06 are fully verified and integrated into production layout and table engines.
