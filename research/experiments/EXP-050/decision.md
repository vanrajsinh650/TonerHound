# EXP-050 Decision Record

## Decision: **REFINE (HOLD BEFORE PRODUCTION MERGE)**

### Key Rationale
1. **Methodological Integrity Fully Restored**: Resolved all 7 audit bugs from EXP-049. Eliminated data leakage pathways, corrected blank checkbox statistics (strictly 55.84%), and classified all 134 unextractable benchmark documents.
2. **Strategy Shift Confirmed Highly Efficient**: Validated that targeting small-to-medium documents (50–800 fields) yields ~77x higher macro F1 gains per rescued field than massive corrupted documents.
3. **Strict Zero Regressions**: Verified across all 370 benchmark test cases (0 regressed documents, 0 regressed fields).
4. **Macro F1 Lift**: Advanced Word Grounding F1 to **73.3022%** (+0.1236 pp vs EXP-049, +0.3104 pp vs EXP-048).
5. **Held-Out Gate Status**: Passed the recalibrated gate threshold ($51 \ge 30$, 0 regressions), while failing the legacy uncalibrated threshold (300). Production merge is held until gate calibration is formally finalized across all document cohorts.

### Next Priority
Proceed to **EXP-051: Small-Document Header & Cell Bounding Box Normalizer**, targeting the 108 small documents in the 50%–90% Word F1 band to capture the next +0.70 pp macro lift toward the 75% benchmark milestone.
