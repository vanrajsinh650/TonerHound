# EXP-007B Full Benchmark Report: TonerHound vs ExtractBench (370 Documents)

**Experiment ID**: `EXP-007B`  
**Benchmark**: ExtractBench Official Benchmark  
**Benchmark Git Commit**: `94ceac15d457881b3d6f1c0f35c15bdea6af4b95`  
**Dataset Revision**: `f6180e917a050a84582e6366cff85b7dc1e84e58`  
**Timestamp**: `2026-09-19T05:38:16.866723+00:00`  
**Dataset Scope**: **370 documents (4869 pages)** — 252 Short, 98 Medium, 20 Long  
**Evaluator**: Official ExtractBench `EvaluationRunner` (`ExtractEvaluator`)  
**Total Runtime**: 11185.4s (186.42 minutes)  
- Prediction Time: 10305.2s  
- Evaluation Time: 880.2s  

---

## 1. Executive Summary & Complete Leaderboard Comparison

| Rank | System / Model | Overall Value F1 | Word Grounding F1 | Page Grounding F1 | Short Word F1 | Medium Word F1 | Long Word F1 | Delta vs Leader |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **TonerHound EXP-007B (Ours)** | **100.00%** | **50.40%** | **70.37%** | **48.84%** | **54.53%** | **56.18%** | **+3.97 pp** |
| 2 | **LlamaExtract Agentic Plus** (Previous #1) | 95.59% | 46.43% | 84.92% | 43.74% | 54.01% | 54.67% | Baseline (0.00 pp) |
| 3 | **TonerHound EXP-004** (Previous Baseline) | 100.00% | 45.49% | 59.11% | 44.03% | 53.68% | 34.82% | -0.94 pp |
| 4 | **LlamaExtract Agentic** | 89.55% | 44.14% | 66.12% | 42.30% | 50.47% | 45.68% | -2.29 pp |
| 5 | **Reducto Deep Extract** | 90.44% | 43.30% | 71.71% | 42.84% | 45.57% | 41.13% | -3.13 pp |
| 6 | **LlamaExtract Cost-Effective** | 86.78% | 40.43% | 64.15% | 40.20% | 42.30% | 36.67% | -6.00 pp |
| 7 | **Extend (Max Context)** | 88.62% | 25.20% | 49.04% | 33.93% | 0.21% | 0.02% | -21.23 pp |
| 8 | **Extend Extract** | 85.72% | 15.96% | 53.58% | 21.13% | 1.03% | 0.01% | -30.47 pp |
| 9 | **Datalab (Accurate + Balanced)** | 85.70% | 2.02% | 48.50% | 2.67% | 0.24% | 0.00% | -44.41 pp |
| — | **Codex (GPT-5.5)** | 93.57% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | -46.43 pp |
| — | **OpenAI GPT-6 Astra** | 91.91% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | -46.43 pp |
| — | **Google Gemini 3.8 Flash** | 80.71% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | -46.43 pp |
| — | **Qwen3.6 35B** | 88.11% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | -46.43 pp |
| — | **Claude Code (Opus 4.8)** | 87.09% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | -46.43 pp |

---

## 2. Detailed Official Metrics

| Metric | Overall | Short (≤10 pgs) | Medium (11–50 pgs) | Long (>50 pgs) |
| :--- | :---: | :---: | :---: | :---: |
| **Word Grounding F1** | **50.40%** | 48.84% | 54.53% | 56.18% |
| **Word Grounding Precision** | **56.72%** | 56.40% | 57.79% | 56.99% |
| **Word Grounding Recall** | **46.57%** | 44.40% | 52.05% | 55.44% |
| **Page Grounding F1** | **70.37%** | 69.75% | 70.54% | 76.95% |
| **Page Grounding Precision** | **86.57%** | — | — | — |
| **Page Grounding Recall** | **62.71%** | — | — | — |
| **Value F1** | **100.00%** | 100.00% | 100.00% | 100.00% |
| **False-Grounding Rate** | **43.28%** | — | — | — |

---

## 3. Domain Performance Breakdown (D1–D8)

| Domain Code | Description | Documents | Word Grounding F1 | Page Grounding F1 | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **domain:D1** | Financial / SEC / 13F / N-PORT | 145 | 59.88% | 77.95% | 66.19% | 56.00% |
| **domain:D2** | Legal / Court / Bankruptcy / Claims | 98 | 36.83% | 58.21% | 44.47% | 32.29% |
| **domain:D3** | Tax / Government / IRS Forms (990, W-2, 1040) | 49 | 43.68% | 67.93% | 44.01% | 43.35% |
| **domain:D4** | Invoices / Receipts / Billing | 27 | 0.00% | 57.08% | 0.00% | 0.00% |
| **domain:D5** | Healthcare / Medical / Clinical | 20 | 0.00% | 83.33% | 0.00% | 0.00% |
| **domain:D6** | Real Estate / Deeds / Titles / Mortgages | 15 | 67.92% | 94.98% | 68.61% | 67.27% |
| **domain:D7** | Corporate / Contracts / Commercial Agreements | 10 | 57.64% | 98.71% | 58.35% | 56.98% |
| **domain:D8** | Academic / Scientific / Technical Reports | 6 | 0.00% | 0.00% | 0.00% | 0.00% |

---

## 4. Diagnostics & System Capabilities

| Measurement | Result | Description |
| :--- | :--- | :--- |
| **Candidate Recall@1** | **59.43%** | Ground truth box matches top-1 candidate |
| **Candidate Recall@5** | **95.73%** | Ground truth box in top-5 candidates |
| **Candidate Recall@10** | **95.73%** | Ground truth box in top-10 candidates |
| **Candidate Recall@20** | **95.73%** | Ground truth box in top-20 candidates |
| **False-Grounding Rate** | **43.28%** | Fraction of emitted citations with wrong IoU/page |
| **Ambiguity Rate** | **93.75%** | Fields with near-identical competing candidates |
| **Not-Found Rate** | **0.01%** | Expected values with 0 textual candidate matches |
| **OCR Pages Invoked** | **0 / 4869** | Scanned / bitmap pages processed with Tesseract OCR |
| **Throughput / Latency** | **0.5 pages/sec** | Mean per-document latency: 27.85s |

---

## 5. Hardware and Environment Information
- **OS**: Linux
- **Architecture**: x86_64
- **Workers**: 6 concurrent processes
- **Engine**: TonerHound + Hybrid Backend (LiteParse + PDFium/Tesseract OCR fallback)
- **Adapter**: Gated Sibling-Co-Occurrence with Median Delta Y and IQR ratio filtering
- **Benchmark Harness**: ExtractBench commit `94ceac15d457881b3d6f1c0f35c15bdea6af4b95`
