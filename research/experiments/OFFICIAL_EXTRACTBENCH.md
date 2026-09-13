# Official ExtractBench Benchmark Evaluation: TonerHound EXP-003

**Evaluator**: ExtractBench Official Evaluation Runner (`ExtractEvaluator`)  
**ExtractBench Commit**: `94ceac15d457881b3d6f1c0f35c15bdea6af4b95`  
**Timestamp**: `2026-09-13T13:46:05.356423+00:00`  
**Total Examples**: 6 (Successful: 6, Failed: 0)  

## 1. Headline Aggregate Metrics

| Metric | Score | Note |
| :--- | :--- | :--- |
| `avg_extract_unified_grounded_f1` | **69.33%** | Official Unified Grounded F1 |
| `avg_extract_unified_grounded_precision` | **78.94%** | Official Unified Grounded Precision |
| `avg_extract_unified_grounded_recall` | **64.69%** | Official Unified Grounded Recall |
| `avg_extract_unified_page_f1` | **59.71%** | Official Unified Page F1 |
| `avg_extract_unified_page_precision` | **73.92%** | Official Unified Page Precision |
| `avg_extract_unified_page_recall` | **55.91%** | Official Unified Page Recall |
| `avg_extract_unified_value_f1` | **100.00%** | Official Unified Value F1 |
| `avg_extract_evidence_value_pass_rate` | **99.84%** | Evidence Value Pass Rate |
| `avg_extract_evidence_page_pass_rate` | **67.46%** | Evidence Page Pass Rate |
| `avg_extract_evidence_bbox_coverage` | **74.77%** | Evidence Bbox Coverage |

## 2. Per-Document Detailed Results

| Test ID | Unified Grounded F1 | Unified Page F1 | Unified Value F1 | Citations | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `medium/veralto_earnings_deck_q4fy25` | N/A | 0.00% | 100.00% | 49 metrics | ✅ Passed |
| `short/W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025)` | 43.86% | 27.93% | 100.00% | 56 metrics | ✅ Passed |
| `short/bianco-2024` | 20.71% | 40.00% | 100.00% | 56 metrics | ✅ Passed |
| `short/real_wyo_Goshen_2024` | 99.49% | 99.65% | 100.00% | 56 metrics | ✅ Passed |
| `long/real_sm0801_eco_full` | 83.20% | 91.04% | 100.00% | 56 metrics | ✅ Passed |
| `medium/real_pueblo_oct_2025` | 99.37% | 99.64% | 100.00% | 56 metrics | ✅ Passed |

## 3. Comparison with Official ExtractBench Leaderboard

| System | Grounded F1 (Word Grounding) | Page F1 | Status / Scope |
| :--- | :--- | :--- | :--- |
| **LlamaExtract Agentic Plus** | **46.43%** | 80.31% | **Official Leaderboard Leader** (Full Benchmark) |
| LlamaExtract Agentic | 44.91% | 79.15% | Official Leaderboard Baseline |
| **TonerHound EXP-003 (Official Harness on Test Set)** | **69.33%** | 59.71% | Evaluated via official `EvaluationRunner` (6-case test set) |

> [!IMPORTANT]
> TonerHound EXP-003 is evaluated locally on the 6 representative ExtractBench test cases using the official `EvaluationRunner` from ExtractBench commit `94ceac15d457881b3d6f1c0f35c15bdea6af4b95`.
> The official leaderboard leader remains **LlamaExtract Agentic Plus at 46.43% overall Word Grounding F1** across the entire leaderboard suite.
> TonerHound's results represent performance on the 6 test cases, strictly separated from official leaderboard claims.
