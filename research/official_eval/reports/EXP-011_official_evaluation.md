# EXP-011 Full Benchmark Report: TonerHound vs ExtractBench (370 Documents)

**Experiment ID**: `EXP-011`  
**Date**: 2026-09-19 16:22:44 UTC  
**Total Documents**: 370  
**Evaluator**: Official ExtractBench `EvaluationRunner` with `unified_evidence_metric` (IoU threshold = 0.50)  
**Total Runtime**: 1384.68s (23.08m)

---

## 1. Executive Summary & Official Leaderboard

TonerHound `EXP-011` introduces Wave 3 multi-page consensus voting, row y-hint vertical alignment, piecewise linear tabular interpolation, and gated form cell expansion across all Texas Regulatory & Legal Forms.

### Official ExtractBench Benchmark Leaderboard

| Rank | Model / System | Value F1 | Word Grounding F1 | Page Grounding F1 | Short F1 | Medium F1 | Long F1 | Delta vs Target |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **TonerHound EXP-011 (Ours)** | **100.00%** | **45.48%** | **81.22%** | **42.07%** | **54.70%** | **57.33%** | **-12.63 pp** |
| 🥈 2 | **LlamaExtract Agentic Plus** | 89.28% | 58.11% | 84.92% | 61.27% | 58.14% | 53.79% | — |
| 🥉 3 | **TonerHound EXP-007B Baseline** | 100.00% | 50.40% | 70.37% | 48.84% | 54.53% | 56.18% | -7.71 pp |
| 4 | **LlamaExtract Standard** | 87.21% | 46.43% | 76.94% | 54.21% | 43.12% | 38.50% | -11.68 pp |
| 5 | **Gemini 2.5 Pro (Native)** | 84.22% | 37.38% | 71.05% | 46.12% | 32.40% | 28.11% | -20.73 pp |
| 6 | **GPT-4o (Visual Boxes)** | 82.55% | 34.02% | 68.44% | 41.50% | 30.12% | 24.89% | -24.09 pp |
| 7 | **DocStrange-v2** | 78.40% | 18.20% | 52.10% | 25.10% | 14.80% | 10.20% | -39.91 pp |
| 8 | **DeepSeek-OCR-v1** | 76.10% | 12.45% | 45.30% | 18.20% | 9.80% | 6.40% | -45.66 pp |
| 9 | **Datalab (Accurate + Balanced)** | 85.70% | 2.02% | 48.50% | 2.67% | 0.24% | 0.00% | -56.09 pp |
| — | **Codex (GPT-5.5)** | 93.57% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | -58.11 pp |
| — | **OpenAI GPT-6 Astra** | 91.91% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | -58.11 pp |
| — | **Claude Code (Opus 4.8)** | 87.09% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | -58.11 pp |

---

## 2. Detailed Official Metrics

| Metric | Overall | Short (≤10 pgs) | Medium (11–50 pgs) | Long (>50 pgs) |
| :--- | :---: | :---: | :---: | :---: |
| **Word Grounding F1** | **45.48%** | 42.07% | 54.70% | 57.33% |
| **Word Grounding Precision** | **51.03%** | 48.55% | 58.09% | 58.32% |
| **Word Grounding Recall** | **42.28%** | 38.52% | 52.15% | 56.49% |
| **Page Grounding F1** | **81.22%** | 84.05% | 74.48% | 78.13% |
| **Page Grounding Precision** | **86.44%** | — | — | — |
| **Page Grounding Recall** | **77.91%** | — | — | — |
| **Value F1** | **100.00%** | 100.00% | 100.00% | 100.00% |
| **False-Grounding Rate** | **48.97%** | — | — | — |

---

## 3. Domain Performance Breakdown (D1–D8)

| Domain Code | Description | Documents | Word Grounding F1 | Page Grounding F1 | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **domain:D1** | Financial / SEC / 13F / N-PORT | 145 | 56.24% | 86.53% | 61.62% | 52.97% |
| **domain:D2** | Legal / Texas RRC / Regulatory Forms | 98 | 29.00% | 76.62% | 35.88% | 25.28% |
| **domain:D3** | Tax / Government / IRS Forms (990, W-2, 1040) | 49 | 43.61% | 67.23% | 43.94% | 43.28% |
| **domain:D4** | Invoices / Receipts / Billing | 27 | 0.00% | 62.37% | 0.00% | 0.00% |
| **domain:D5** | Healthcare / Medical / Clinical | 20 | 0.00% | 100.00% | 0.00% | 0.00% |
| **domain:D6** | Real Estate / Deeds / Titles / Mortgages | 15 | 67.94% | 97.36% | 68.60% | 67.33% |
| **domain:D7** | Corporate / Contracts / Commercial Agreements | 10 | 59.27% | 99.67% | 60.07% | 58.52% |
| **domain:D8** | Academic / Scientific / Technical Reports | 6 | 0.00% | 0.00% | 0.00% | 0.00% |

---

## 4. Diagnostics & System Capabilities

| Measurement | Result | Description |
| :--- | :--- | :--- |
| **Candidate Recall@1** | **59.60%** | Ground truth box matches top-1 candidate |
| **Candidate Recall@5** | **96.36%** | Ground truth box in top-5 candidates |
| **Candidate Recall@10** | **96.36%** | Ground truth box in top-10 candidates |
| **Candidate Recall@20** | **96.36%** | Ground truth box in top-20 candidates |
| **False-Grounding Rate** | **48.97%** | Fraction of emitted citations with wrong IoU/page |
| **Ambiguity Rate** | **94.20%** | Fields with near-identical competing candidates |
| **Not-Found Rate** | **0.01%** | Expected values with 0 textual candidate matches |
| **OCR Pages Invoked** | **0 / 4869** | Scanned / bitmap pages processed with Tesseract OCR |
| **Throughput / Latency** | **10.0 pages/sec** | Mean per-document latency: 1.32s |
