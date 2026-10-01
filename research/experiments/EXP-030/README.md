# TONERHOUND — EXP-030: Global Structured Assignment for Table-Level Evidence Grounding

## 1. Verified Starting State

### A. Performance Benchmarks
- **Production Baseline (EXP-028E full 370-document benchmark)**:
  - **Overall Word Grounding F1**: **56.05%** (56.0477%)
  - **Overall Word Precision**: 61.73%
  - **Overall Word Recall**: 52.57%
  - **Overall Page Grounding F1**: 81.66%
  - **False Grounding Rate**: 23.87%
  - **Abstention Rate**: 28.19%
- **Benchmark Splits**:
  - **Short (175 docs)**: 52.00% Word F1 | 84.47% Page F1
  - **Medium (48 docs)**: 67.35% Word F1 | 75.26% Page F1
  - **Long (13 docs)**: 68.81% Word F1 | 77.29% Page F1
  - **Table-Heavy (13 docs)**: 87.57% Word F1 | 84.91% Page F1
  - **Non-Table (223 docs)**: 54.21% Word F1 | 81.46% Page F1
- **Cohorts**:
  - **Cohort A (Development, 32 docs)**: 65.33% Word F1 | 92.50% Page F1
  - **Cohort B (Held-Out, 32 docs)**: 59.36% Word F1 | 85.44% Page F1
- **Smoke Benchmark (7 docs, EXP-028F Phase 1.2)**: 62.90% Word F1
- **Competitive Target**:
  - **LlamaExtract Agentic Plus Leaderboard**: **58.11% Overall Word Grounding F1**
  - **EXP-030 Success Target**: **$\ge$ 58.12% Overall Word Grounding F1** on the official full 370-document benchmark.
  - **Distance to Target**: **+2.06 pp**

---

### B. Ceiling Analysis & The Selection Gap
- **EXP-028B0 True Text Oracle**: **75.12% Word F1** (theoretical text reachability upper bound).
- **Ceiling A (Perfect Selection from existing candidate inventory)**: **60.96% Word F1**.
- **Global Selection Gap**: **4.91 pp** (60.96% vs 56.05%).
- **Held-Out Cohort B Multi-Candidate Fields (71,439 fields)**:
  - **Baseline Multi-Candidate Hit@1**: **25.00%** (17,861 fields)
  - **Candidate Pool Recall ($\text{IoU} \ge 0.50$)**: **75.83%** (54,171 fields)
  - **Selection Gap on Multi-Candidate Fields**: **50.83 pp**

---

### C. Pipeline Implementation Status
1. **Active in Production Stack**:
   - `EvidenceMatcher` multi-tier matching (exact, numeric, date, boolean, fuzzy).
   - `CandidateRecoveryEngine` modular recovery (fragmented, interleaved, spaced numeric, split symbol).
   - `ENABLE_GLOBAL_FALLBACK = True` (EXP-028F Phase 1.2 token-gated global fallback for long documents).
   - `ExtractBenchAdapter` monotonic dynamic programming row alignment with embedded structural DP scoring (`enable_structural_dp_scoring = True`, EXP-028D).
   - Adaptive character-span sub-token slicing and dot-leader trimming.
2. **Halted / NOT Implemented**:
   - EXP-028F Phase 1.3 (Per-Page Cap relaxation): STOPPED.
   - EXP-028F Phase 1.4 (Fuzzy recovery expansion): STOPPED.
   - Candidate generation expansion is STOPPED.
3. **Definitively Ruled Out**:
   - EXP-029 point-wise machine learning reranking: Regressed held-out Cohort B Word F1 from 59.36% to 57.94% (-1.42 pp) because isolated candidate scoring cannot resolve repeated scalar entries across table rows.

---

## 2. EXP-030 Core Hypothesis

> Table disambiguation is not a point-wise classification problem; it is a **global structured assignment problem**. When fields belonging to the same record (e.g. `issuer`, `class`, `cusip`, `value`, `shares`) are evaluated jointly against candidate table rows, the unique high-entropy fields (CUSIP, issuer) anchor the row, enabling deterministic and coherent assignment for all low-entropy repeated values (`0.00`, `SOLE`, `CA`, dates).

---

## 3. Experimental Protocol

1. **Phase 1**: Audit candidate generation layer to verify that candidate inventory has not suffered truncation or deduplication collapse.
2. **Phase 2 & 3**: Construct Oracle Table Structure to measure the maximum theoretical gain achievable by joint row-aware assignment in isolation.
3. **Phase 4 & 5**: Formulate and test Global Assignment with independent constraint ablations:
   - Constraint A: Row coherence (same record $\to$ same row)
   - Constraint B: Row monotonicity (record order follows document order)
   - Constraint C: Column consistency (field $\to$ consistent column corridor)
   - Constraint D: Sibling proximity (vertical baseline compatibility)
   - Constraint E: Table membership (isolate distinct tables)
   - Constraint F: Ambiguity / Abstention gating
4. **Phase 7 & 8**: Evaluate on Held-Out Cohort B (32 documents). Measure Hit@1, selection gap reduction, beneficial vs harmful flips, and official ExtractBench Word Grounding F1.
5. **Phase 9**: Full 370-document benchmark (strictly conditional on passing Decision Gate A/B).
