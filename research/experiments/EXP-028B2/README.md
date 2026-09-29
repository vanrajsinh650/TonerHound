# EXP-028B2: Table Row/Column Disambiguation

## Overview
This experiment addresses the dominant failure mode identified in EXP-028B1 (99.43% of real Oracle-to-Production gap fields) where repeated tokens across rows and columns in structured tables resulted in incorrect spatial assignments.

## Directory Structure
- `production_report.md`: Detailed engineering report and empirical comparison.
- `controlled_experiments_metrics.json`: Full metrics payload across Modes A through F.
- `experiment_config.json`: Experiment configuration and ablation parameters.
- `representative_examples.json`: Detailed case studies showing before/after IoU and coordinate improvements.
- `failure_samples.json`: Residual failure analysis and classification.
- `phase1_diagnosis.py`: Phase 1 verification script auditing EXP-028B1 failure classifications.
- `run_controlled_experiments.py`: Controlled experiment suite executing Modes A through F.

## How to Run
```bash
# Run unit tests
pytest tests/test_joint_record_resolver.py

# Run Phase 1 verification
python research/experiments/EXP-028B2/phase1_diagnosis.py

# Run Phase 4 controlled experiments
python research/experiments/EXP-028B2/run_controlled_experiments.py
```
