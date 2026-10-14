# EXP-048 Decision Record

## Decision: **INTEGRATE**

### Executive Justification
1. **Proven Macro F1 Movement**: For the first time across four successive experiments (EXP-044 through EXP-047), macro-averaged Word Grounding F1 moved by **+0.1470 pp** (from 72.8448% to 72.9918%).
2. **Massive Field Recovery Rate**: EXP-048 rescued **+7,109 fields** in net total—a **7.2x acceleration** over EXP-047 (+991 rescues) and a **222x acceleration** over EXP-046 (+32 rescues).
3. **Strict Zero Regressions**: Zero fields regressed across all 370 benchmark documents. Every single affected document experienced strictly monotonic metric improvement.
4. **Held-Out Cohort Validation**: Net rescues on the 32-document frozen held-out cohort reached **+1,349 fields** (well exceeding the >= 1,000 field gate), with 0 regressions and latency of 23.44s.
5. **Architectural Paradigm Shift**: Validates that diagnostic-first, document-family-scoped repair is far superior to generic global heuristic stages. Instead of guessing global rules, isolating document-level spatial column rails and character glyph offsets directly resolves the tail.

### Accepted Modules for Integration
- `research/experiments/EXP-048/fixes/fix_01_sec13f_voting_authority.py` (+5,395 rescues across 3 documents)
- `research/experiments/EXP-048/fixes/fix_02_holdings_marks_precision.py` (+1,421 rescues across 5 documents)
- `research/experiments/EXP-048/fixes/fix_03_fund_security_name_single_line.py` (+293 rescues across 5 documents)

### Rejected Modules (Excluded from Production)
- `research/experiments/EXP-048/fixes/fix_04_sec13f_title_of_class.py` (Rescued 5 fields on 1 document; rejected due to insufficient multi-document reach).
- `research/experiments/EXP-048/fixes/fix_05_creditor_matrix_parser.py` (Rescued 9 fields across 2 documents; rejected due to insufficient rescue count <100 fields).

### Next Priority
Build a dedicated layout-aware table localizer for multi-line multi-column bankruptcy creditor schedules (`real_ftx_full`, `real_imedia_full`) where over 34,000 fields remain ungrounded due to physical token grid fragmentation.
