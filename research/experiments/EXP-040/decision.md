# EXP-040 DECISION MEMORANDUM

## 1. Recommendation

**VERDICT: REFINE AND RETRY (GATE VERDICT: MIXED)**

- **Baseline Word Grounding F1 (EXP-039)**: 69.1327%
- **Measured Word Grounding F1 (EXP-040)**: **70.3894%**
- **Net F1 Gain**: **+1.2567 pp**
- **Field Regressions**: **0** (Zero regressions across 498,140 fields)
- **Net Fields Rescued**: **+4,817 fields**
- **Decision Gate Criteria**:
  - Score $\ge 74.00\%$ with zero regressions $\rightarrow$ INTEGRATE
  - Score $70.00\% - 74.00\%$ with zero regressions $\rightarrow$ **REFINE AND RETRY**
  - Score $< 70.00\%$ OR any regressions $\rightarrow$ REJECT

---

## 2. Rationale

1. **Historic 70% Barrier Broken**:
   TonerHound has achieved **70.3894% Word Grounding F1** on the official ExtractBench benchmark harness, rescuing 4,817 net fields without a single regression across the entire 498,140-field corpus.
2. **Solid Progress but Below Integration Gate**:
   While the score of 70.3894% represents a verified +1.2567 pp gain over EXP-039 (and +14.34 pp over Canonical Baseline V1), it falls short of the ambitious 74.00% production integration threshold.
3. **Identified Bottlenecks**:
   The forensic audit revealed that although `NORMALIZATION_MISMATCH` is the second-largest failure mode (46,156 fields, +5.91 pp theoretical ceiling), token-level variants rescued only 2 fields. This is because ExtractBench's official evaluator requires exact coordinate alignment matching the raw document evidence, which in dense tables requires a column rail coordinate constraint (`SUB_COLUMN_DRIFT`) and tabular layout structure.

---

## 3. Production Readiness Assessment

| Component | Module | Status | Latency Impact | Recommendation |
| :--- | :--- | :---: | :---: | :--- |
| **Multi-Token Sequence Matcher** | `multi_token_matcher.py` | **READY** | +10 ms / page | Promote to production candidate library |
| **Multi-Region Assembler** | `multi_region_assembler.py` | **READY** | +5 ms / page | Promote to production candidate library |
| **Hyphenation Joiner** | `hyphen_joiner.py` | **READY** | +2 ms / page | Promote to production candidate library |
| **Normalization Variants** | `normalization_variants.py`| **NEEDS REVISION** | Negligible | Refactor into column-level schema mapper |
| **Global Table Assigner** | `global_assignment.py` | **READY** | +12 ms / table | Promote to production candidate library |

---

## 4. Next Priority & Action Plan (EXP-041)

To bridge the 3.61 pp gap to 74.00%, the immediate highest-leverage engineering targets are:

1. **Column Rail Constraint Grounder (`SUB_COLUMN_DRIFT`: 7,804 fields, +0.99 pp realistic gain)**:
   In dense financial statements (e.g. Schedule of Investments), repeated values (such as `USD`, `0.00`, `1,000`) drift horizontally into adjacent column cells. By detecting column rails via PyMuPDF table grid vertical lines, candidates can be strictly bounded to the target column rail.
2. **Trailed Line Hyphen Joining at Scale (`HYPHENATION`: 7,623 fields, +0.68 pp realistic gain)**:
   Integrate a dictionary-backed prefix-suffix hyphen joiner across OCR text lines to recover broken tokens.
3. **Targeted Tabular Schema Normalizer (`NORMALIZATION_MISMATCH`: 46,156 fields, +0.59 pp realistic gain)**:
   Parse cell formatting directly from table headers rather than ad-hoc string variations.
