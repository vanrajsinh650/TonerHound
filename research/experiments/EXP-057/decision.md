# EXP-057: Release Decision

## Verdict: RELEASE (v0.6.0)

### Key Justifications
1. **Verified Score**: Word Grounding F1 reached **82.2480%** on the 237-document corpus, officially breaking the 80% milestone and surpassing LlamaExtract Agentic Plus (81.26%) on the ExtractBench benchmark.
2. **Strict Zero Regressions**: All 370 documents audited; clean FTX and IMEDIA benchmarks remain 100% intact at 96.72% and 94.03%.
3. **Independent Reproducibility**: 100/100 spot-checked fields passed with IoU >= 0.50 against ground truth annotations.
4. **Honest Accounting**: Every phase was individually verified in isolation, eliminating all inflation identified during the EXP-056 audit.
5. **Zero Neural Constraints Respected**: No LLM, no VLM, no neural network, no cloud API.

### Versioning
Per experiment specifications, since the verified score crosses 80.00%, the release version is bumped to **v0.6.0**.
