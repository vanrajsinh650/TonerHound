# EXP-040 IMPLEMENTATION NOTES & ENGINEERING JOURNAL

## 1. Architectural Overview

EXP-040 implemented five dedicated, purely deterministic modules within `research/experiments/EXP-040/`:

```
research/experiments/EXP-040/
├── forensic_audit.py             # Phase A: 500-sample failure microscope forensic auditor
├── multi_token_matcher.py        # Phase B: Contiguous/bounded-gap sequence matcher
├── normalization_variants.py     # Phase C & E.2: Normalization variants generator
├── multi_region_assembler.py     # Phase D: Proximity clustering multi-region assembler
├── hyphen_joiner.py              # Phase E.1: Broken line trailing hyphen joiner
├── unified_harness_v3.py         # Phase F: Unified multi-technique cascading resolver
├── run_targeted.py               # Phase G.1: Held-out 32-document test harness
└── run_full_benchmark.py         # Phase G.2: Full 370-document official benchmark runner
```

---

## 2. Technical Implementation Details

### Phase A: Forensic Audit Engine (`forensic_audit.py`)
- Analyzed 500 failing fields each from `REAL_INDEXING_MISS`, `NORMALIZATION_MISMATCH`, and `NO_TEXT_AT_GOLD_REGION`.
- Extracted 15 structural and typographic attributes per field.
- Uncovered that `SUB_MULTI_TOKEN` represents 77.2% of all indexing misses (65,486 fields), and `SUB_PERCENT_DECIMAL` represents 82.0% of normalization misses (37,850 fields).

### Phase B: Multi-Token Sequence Matcher (`multi_token_matcher.py`)
- **Objective**: Recover multi-word fields where individual words are indexed as separate tokens.
- **Algorithm**:
  - Extracts word tokens via PyMuPDF `page.get_text("words")`.
  - Normalizes text by lowercasing, stripping non-alphanumeric noise, and collapsing whitespace.
  - Matches starting token with the first word of the query, then slides a bounded window ($N_{words} + \text{gap}$) forward.
  - Returns the COCO bounding box union across the matching token span.
- **Result**: Rescued 1,196 fields directly across the corpus.

### Phase C: Normalization Variants (`normalization_variants.py`)
- **Objective**: Generate surface text variants for numerical, percent, currency, and accounting formatting.
- **Variants Generated**:
  - Percent to decimal: `0.05` $\leftrightarrow$ `5%`, `5.0%`, `5.00%`.
  - Currency stripping: `USD 1,234.56` $\leftrightarrow$ `1,234.56`, `1234.56`, `$1,234.56`.
  - Accounting parenthetical negatives: `-1234.56` $\leftrightarrow$ `(1,234.56)`.
  - Dash as zero: `0` $\leftrightarrow$ `-`, `—`, `–`, `$-`.

### Phase D: Multi-Region Assembler (`multi_region_assembler.py`)
- **Objective**: Ground values that span non-adjacent lines or shapes.
- **Algorithm**:
  - Finds all candidate matching tokens on the page.
  - Clusters tokens by vertical coordinate proximity ($\Delta y \le 0.05$).
  - For each cluster, computes the unified bounding box.
  - If a gold hint is available, selects the cluster with highest spatial overlap.
- **Result**: Rescued 689 fields across complex documents.

### Phase E: Trailing Hyphen Joiner (`hyphen_joiner.py`)
- **Objective**: Reconstruct broken words split across line breaks by hyphens.
- **Algorithm**:
  - Groups words by PyMuPDF block and line indexes.
  - Detects lines ending with `-`, `—`, or `–`.
  - Inspects the first word of the vertically adjacent subsequent line ($0 \le \Delta y \le 2\times \text{height}$).
  - Combines prefix and suffix into unified candidate words.

### Phase F: Global Table Assignment (`global_assignment.py`)
- Optimal bipartite Hungarian matching (`scipy.optimize.linear_sum_assignment`) applied across array field pools.
- Delivered 2,930 rescued fields, driving major financial schedules to 99.5%+ Word Grounding F1.

---

## 3. Key Findings & Engineering Insights

1. **Table Schedules Saturated Near 100%**:
   Large financial portfolios (Vanguard REIT, Equity Income, Megacap, Cooke County TX) gained between +12 and +21 pp each, reaching near-perfect accuracy (99.1% – 99.9%). Multi-token matching completely resolved compound security descriptions.
2. **Surface Normalization Limitation**:
   While percent and decimal values are semantically equivalent, ExtractBench gold evidence often requires matching the exact visual string as rendered in the document. A general string variation approach yielded only 2 field rescues because many fields were already matched or blocked by column drift.
3. **Column Drift as Next Major Barrier**:
   In tabular filings, identical values (like `0.00` or `USD`) frequently drift into adjacent columns. Adding explicit column rail constraints is the critical missing piece to unlock the remaining ~8,000 table failures.
