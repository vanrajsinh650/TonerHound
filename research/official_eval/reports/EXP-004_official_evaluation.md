# EXP-004 Full Benchmark Report: TonerHound vs ExtractBench (370 Documents)

**Experiment ID**: `EXP-004`  
**Benchmark**: ExtractBench Official Benchmark  
**Benchmark Git Commit**: `94ceac15d457881b3d6f1c0f35c15bdea6af4b95`  
**Dataset Revision**: `f6180e917a050a84582e6366cff85b7dc1e84e58`  
**Timestamp**: `2026-09-14T07:44:15.628133+00:00`  
**Dataset Scope**: **370 documents (4618 pages)** — 252 Short, 98 Medium, 20 Long  
**Evaluator**: Official ExtractBench `EvaluationRunner` (`ExtractEvaluator`)  
**Total Runtime**: 689.6s (11.49 minutes)

---

## 1. Executive Summary & Official Comparison

| Metric / System | **LlamaExtract Agentic Plus** (Leader) | **TonerHound EXP-004** (Ours) | Delta |
| :--- | :--- | :--- | :--- |
| **Word Grounding F1 (Overall)** | **46.43%** | **45.49%** | **-0.94%** |
| — Short Documents | 43.74% | 44.03% | +0.29% |
| — Medium Documents | 54.01% | 53.68% | -0.33% |
| — Long Documents | 54.67% | 34.82% | -19.85% |
| **Word Precision** | — | **59.45%** | — |
| **Word Recall** | — | **38.61%** | — |
| **Page Grounding F1 (Overall)** | **84.92%** | **59.11%** | -25.81% |
| — Short Documents | 89.70% | 59.43% | -30.27% |
| — Medium Documents | 72.25% | 57.37% | -14.88% |
| — Long Documents | 87.14% | 63.16% | -23.98% |
| **Value F1** | 95.59% | **100.00%** | +4.41% |

---

## 2. Diagnostics & System Capabilities

| Measurement | Result | Description |
| :--- | :--- | :--- |
| **Candidate Recall@1** | **45.96%** | Ground truth box matches top-1 candidate |
| **Candidate Recall@5** | **78.97%** | Ground truth box in top-5 candidates |
| **Candidate Recall@10** | **78.97%** | Ground truth box in top-10 candidates |
| **Candidate Recall@20** | **78.97%** | Ground truth box in top-20 candidates |
| **False-Grounding Rate** | **40.55%** | Fraction of emitted citations with wrong IoU/page |
| **Ambiguity Rate** | **77.59%** | Fields with near-identical competing candidates |
| **Not-Found Rate** | **0.01%** | Expected values with 0 textual candidate matches |
| **OCR Pages Invoked** | **0 / 4618** | Scanned / bitmap pages processed with Tesseract OCR |
| **Throughput / Latency** | **3965.2 pages/sec** | Mean per-document latency: 0.00s |

---

## 3. Official Extraction Benchmark Slices

| Benchmark Slice | Documents | Grounded F1 | Page F1 | Value F1 |
| :--- | :--- | :--- | :--- | :--- |
| **All Documents** | **370** | **45.49%** | **59.11%** | **100.00%** |
| **Short (≤ 10 pages)** | 252 | 44.03% | 59.43% | 100.00% |
| **Medium (11–50 pages)** | 98 | 53.68% | 57.37% | 100.00% |
| **Long (> 50 pages)** | 20 | 34.82% | 63.16% | 100.00% |

---

## 4. Hardware and Environment Information
- **OS**: Linux
- **Architecture**: x86_64
- **Workers**: 6 concurrent processes
- **Engine**: TonerHound + pdfium + Tesseract OCR fallback
- **Benchmark Harness**: ExtractBench commit `94ceac15d457881b3d6f1c0f35c15bdea6af4b95`
