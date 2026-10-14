# EXP-045 Decision Document

## Status: ACCEPTED / READY FOR INTEGRATION GATE (FEATURE-FLAGGED)

### 1. Decision Summary
Following the implementation and validation of **Visual Line-Wrap Merging**, **Intermediate Sub-Header Filtering**, and **Multi-Column Reading Order Flow**, we formally **ACCEPT** promoting Stage 7 (`RowIndexMapper`) into the TonerHound architecture behind a feature flag gate.

The empirical results across the held-out cohort and full benchmark confirm:
- **8,303 net fields rescued** by Stage 7 alone.
- **Strictly zero regressions (0 fields)** across all evaluated documents.
- **+0.0367 pp macro Word Grounding F1 gain** across the benchmark (Word F1: 72.8081% → 72.8448%).
- **Total passing fields increased**: 336,135 → 344,438.
- **Latency overhead bounded**: Average pipeline latency increased by 1.18x, comfortably within the $\le 3\times$ latency gate.

The elimination of row accumulation drift successfully resolved the 1:1 bijection between schema array index $N$ and physical table row $R$ across dense multi-page tables.

---

### 2. Validation Gate Checklist

| Gate | Requirement | Actual Result | Status |
| :--- | :--- | :---: | :---: |
| **Phase A Viability Trace** | Sample trace viability $\ge 30\%$ | 20 / 20 (100.0%) | **PASS** |
| **Zero Regressions** | Strictly 0 regressed fields | 0 regressed fields | **PASS** |
| **Held-Out Evaluation Gate** | Rescued $\ge 1,500$ net, 0 regressed | +8,303 Stage 7, 0 regressed | **PASS** |
| **Full Benchmark Marginal Rescue** | Measurable net positive recovery | +8,303 marginal fields | **PASS** |
| **Production Code Isolation** | `src/tonerhound/` completely untouched | Verified untouched | **PASS** |
| **Latency Gate** | Latency factor $\le 3.0\times$ | 1.18x factor | **PASS** |

---

### 3. Key Technical Advances in Stage 7
1. **Visual Line-Wrap Merging**: Seamlessly stitches multi-line text descriptions and trailing footnote/series tokens into a single logical `TableRow` anchored by the quantity/value numeric cells.
2. **Intermediate Sub-Header Filtering**: Disregards category sub-headers (percentages, asset class banners, and unannotated issuer titles) so they do not increment the data row counter.
3. **Multi-Column Reading Order Flow**: Detects 2-column table layouts and reads column 1 top-to-bottom, followed by column 2 top-to-bottom.
4. **Strict Spatial Row Windowing**: Constrains token candidate searches strictly within the mapped `TableRow`'s normalized $[x, y, w, h]$ window, completely eliminating ambiguous cross-row false positives.
