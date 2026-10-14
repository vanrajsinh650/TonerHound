# EXP-051: Engineering Decision Record

## Status: ACCEPTED (EXPERIMENT FIX ACCEPTED; INTEGRATION PENDING 75% GATE)

### Evaluation Against Acceptance Gates

1. **Isolation Acceptance Criteria (fix_01_cell_padding_normalizer)**:
   - Rescued Fields $\ge 500$: **PASS (663 fields rescued)**
   - Regressed Fields $== 0$: **PASS (0 regressions)**
   - Documents Affected $\ge 10$: **PASS (46 documents affected)**
   - Runtime $\le 15.0\text{s}$: **PASS (0.06 seconds runtime)**
   - **Fix Status**: `ACCEPTED`

2. **Benchmark Acceptance Criteria (Part D)**:
   - Macro Word Grounding F1 Lift: **+0.4376 pp (73.3022% -> 73.7398%)**
   - Zero Regressions across all 370 benchmark documents: **VERIFIED**
   - Remaining gap to 75.0%: **1.2602 pp**

3. **Part B Free Extractor Diagnosis**:
   - LiteParse extracted 192 pages in 3.4 seconds, achieving 24,365 passes (67.12%) vs TonerHound's 8,886 (24.48%).
   - Diagnosis: Spatial extractors produce tight token envelopes. Discrepancies in OFAC table structures are caused by cell-boundary width framing rather than missing characters.
   - Reference fix `fix_02_paddle_derived_cell_detection.py` documented.

4. **Part C Microscope V6 Upgrade**:
   - Code written: [`research/observer/microscope_v6.py`](file:///home/vanrajsinh/Projects/TonerHound/research/observer/microscope_v6.py).
   - Execution status: **NOT RUN** (Preserved strictly for EXP-052).
   - Unit tests: **100% PASS** ([`tests/test_microscope_v6.py`](file:///home/vanrajsinh/Projects/TonerHound/tests/test_microscope_v6.py)).

---

## Production Integration Decision
- **Decision**: **REFINE / HOLD PRODUCTION INTEGRATION**.
- **Rationale**: Strict project hard constraints mandate that production code in `src/tonerhound/` remains untouched until reaching the 75.0% threshold. At 73.74% Word Grounding F1, the project is within **1.26 pp** of production release. Fix 01 is staged in `research/experiments/EXP-051/fixes/` ready for final integration once the 75% gate is crossed.

---

## Recommended Next Experiment: EXP-052
- **Objective**: Execute Microscope V6 on the complete 370-document corpus to perform the project's first exhaustive field accounting across all 498,140 fields.
- **Target**: Resolve the top remaining failure cluster (`WRONG_ROW` in SEC 13F holdings tables, 48,535 fields) to close the remaining 1.26 pp gap to 75%.
