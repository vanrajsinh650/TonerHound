# EXP-049 Decision Record

## Decision: **HOLD BEFORE PRODUCTION MERGE (REFINE RESEARCH MODULE)**

### Executive Justification
1. **Positive Macro F1 Impact**: EXP-049 moved macro-averaged Word Grounding F1 by **+0.1868 pp** (from 72.9918% to 73.1786%) and Page Grounding F1 by **+0.0704 pp** (from 90.0613% to 90.1317%).
2. **Substantial Field Recovery**: Achieved **+117 net field rescues** across 13 tax documents, with top documents experiencing dramatic double-digit jumps (`medium/bar-lev-2024`: +13.75 pp; `medium/bar-lev-2023`: +9.16 pp).
3. **Strict Zero Regressions**: Verified across all 370 benchmark documents (0 regressed documents, 0 regressed fields).
4. **Negligible Latency**: Overhead is only 2.95s across all target documents and 16.43s across the held-out cohort.
5. **Held-Out Gate Analysis**:
   - The frozen held-out gate threshold required **≥ 300 net rescues**.
   - EXP-049 delivered **+51 net rescues** on the held-out split, resulting in a formal **FAIL** verdict.
   - **Root Cause of Shortfall**: The 32-document frozen held-out cohort contains only 4 tax documents (`bar-lev-2024`, `becerra-2021`, `becerra-2022`, `arif-2021`). Across those 4 documents, there are only ~120 failing fields with physical OCR text tokens; over 70% of the failures are empty/blank boxes (`[ ]`) or redacted cells with zero OCR tokens in the gold region. Rescuing 51 fields represents a strong 42.5% recovery rate of reachable fields, but mathematically could not reach 300.

### Integration Strategy
- **Hold Production Integration in `src/tonerhound/`**: In accordance with the hard constraint that fixes failing the held-out gate must not be merged into production code prior to gate satisfaction.
- **Preserve Research Fix**: Retain `research/experiments/EXP-049/fixes/irs_line_grid.py` as an approved, verified research module.
- **Future Pathway**: To clear the gate for production inclusion, develop a synthetic geometric box generator for blank/empty checkboxes in IRS schedules, OR adjust the held-out gate threshold to be proportional to target family representation within the split.

### Next Priority
Proceed directly to **EXP-050: Bankruptcy Creditor Schedule Multi-Column Block Localizer**, targeting the massive ungrounded block of 34,575 fields in `real_ftx_full_corrupted` and `real_imedia_full_corrupted` (4.32 pp macro lift potential).
