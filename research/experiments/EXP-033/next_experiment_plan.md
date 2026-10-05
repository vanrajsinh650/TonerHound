# EXP-034: Next Experiment Plan — Joint Hungarian Table Row Alignment & Production Selection

**Preceding Experiment**: EXP-033 (Candidate Generation Reconciliation Audit + Productionization)  
**Status**: Mandated by EXP-033 Decision Gate D (Reassess Bottleneck)  
**Primary Target**: $\ge 58.11\%$ Word Grounding F1 on the full 370-document ExtractBench benchmark

---

## 1. Context & Motivation

EXP-032 and EXP-033 established two foundational empirical facts:
1. **Selection Headroom Exists**: The current candidate pool has an oracle ceiling of **60.9591%** (+4.91 pp above the 56.0477% baseline).
2. **Scalar Candidate Expansion Alone Cannot Close the Gap**: EXP-033 proved that candidate expansion on scalar fields yields only +0.0071 pp because >90% of benchmark field mass consists of table cells evaluated under Hungarian row assignment.

To beat the **58.11%** LlamaExtract target, TonerHound needs to recover **+2.06 pp** of the available **+4.91 pp** selection headroom. This requires resolving table cells in their joint row context rather than as independent scalar queries.

---

## 2. Core Hypothesis for EXP-034

> **Hypothesis**: By implementing joint Hungarian table row alignment inside `ExtractBenchAdapter._ground_table()`, table cells with identical values (e.g. repeated numbers, zero balances, two-letter state abbreviations) will map to their exact spatial row coordinates without cross-row collision, capturing $\ge 2.50$ pp of selection headroom and moving production F1 above $58.11\%$.

---

## 3. Planned Architecture for EXP-034

1. **Joint Row Assignment Matrix**:
   - For each table, compute a cost matrix between predicted row extractions and visual line candidates using token overlap and vertical order.
   - Run Hungarian linear sum assignment (`scipy.optimize.linear_sum_assignment`) to match predicted table records to physical document rows.
2. **Anchor-Constrained Cell Bounding**:
   - Once a row is mapped to visual line $L_i$, constrain all column cell lookups strictly to the horizontal band $[L_i.y0 - \epsilon, L_i.y1 + \epsilon]$.
   - This eliminates the dominant cause of table cell precision loss in the official evaluator.
3. **Safety & Fallback Policy**:
   - If bipartite matching confidence is low or table structure is unstructured/grid-corrupted, fallback to the existing safe baseline DP alignment.
4. **Controlled Evaluation**:
   - Evaluate on Held-Out Cohort B (32 documents) first.
   - Run the optimized full 370-document production runner (`run_production_370.py`).
   - Validate against the 58.11% LlamaExtract target.

---

## 4. Execution Sequence

```text
1. Implement Joint Table Bipartite Matching in src/tonerhound/benchmark/adapter.py
        ↓
2. Gate behind ENABLE_JOINT_TABLE_ASSIGNMENT feature flag
        ↓
3. Run full unit test suite (ensure 236/236 pass)
        ↓
4. Evaluate on Held-Out Cohort B (verify delta >= +2.0 pp)
        ↓
5. Run Full 370-Document Benchmark
        ↓
6. Compare directly against 58.11% LlamaExtract target
```
