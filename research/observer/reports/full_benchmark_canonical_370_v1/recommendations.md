# Evidence-Backed Research Recommendations

Based strictly on observed macro-weighted opportunity and document breadth from run `canonical_370_v1`:

## 1. Priority 1: Page-Level OCR Routing for Zero-Text Documents
- **Evidence**: `NO_TEXT_AT_GOLD_REGION` affects 130 documents and 58778 fields, representing a macro-weighted opportunity of 19.63 pp.
- **Action**: Implement deterministic page-level OCR detection only when `len(page.tokens) == 0`.

## 2. Priority 2: Deterministic Visual Provider Integration (EXP-036D)
- **Evidence**: `NON_TEXT_BOOLEAN_GROUNDING` affects 160 documents and 2847 fields, representing a macro-weighted opportunity of 16.45 pp.
- **Action**: Integrate EXP-036D Policy A/D to rescue 185 fields (+0.7760 pp Word F1) with zero regressions.

## 3. Priority 3: Multi-Format Date Parsing
- **Evidence**: `DATE_INDEX_MISS` affects 79 documents and 2436 fields.
- **Action**: Expand deterministic date normalizer to handle non-standard delimiters (`YYYY-MMM-DD`, `DD-MM-YYYY`).
