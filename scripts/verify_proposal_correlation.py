#!/usr/bin/env python3
"""CLI utility for EXP-005 Pass 4: Correlate Agent Proposals with Failure Taxonomy.

Usage:
  # Verify current working tree / adapter output against G1:
  python scripts/verify_proposal_correlation.py --target G1

  # Verify current working tree / adapter output against G3:
  python scripts/verify_proposal_correlation.py --target G3

  # Verify a saved citation payload against G1:
  python scripts/verify_proposal_correlation.py --target G1 --citations path/to/payload.json
"""

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
_REF = repo_root / "research/reference/ExtractBench/src"
if _REF.exists(): sys.path.insert(0, str(_REF))
sys.path.insert(0, str(repo_root / "src"))

from tonerhound.benchmark.correlation import FailureRegistry, verify_agent_proposal


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate Agent Proposals with Failure Taxonomy")
    parser.add_argument("--target", required=True, choices=["G1", "G2", "G3", "G4", "T1"], help="Target failure taxonomy category")
    parser.add_argument("--citations", help="Path to JSON file containing candidate field_citations list")
    parser.add_argument("--min-purity", type=float, default=0.60, help="Minimum acceptable attribution purity (default: 0.60)")
    args = parser.parse_args()

    if args.citations:
        cit_path = Path(args.citations)
        with open(cit_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            candidate_citations = data.get("field_citations", data) if isinstance(data, dict) else data
    else:
        print("[Agent C Correlation Harness] Grounding candidate output with current adapter...")
        from extract_bench.test_cases.loader import load_test_case
        from tonerhound.benchmark.adapter import ExtractBenchAdapter
        from tonerhound.document.index import DocumentIndex

        pdf_path = repo_root / "research/data/full/long/real_ftx_full_corrupted.pdf"
        tc = load_test_case(pdf_path)
        idx = DocumentIndex.from_pdf(pdf_path, enable_ocr=True)
        adapter = ExtractBenchAdapter(idx)
        payload = adapter.ground_extracted_data(tc.expected_output)
        candidate_citations = payload.get("field_citations", [])

    registry_path = repo_root / "experiments" / "EXP-005-ftx-failure-registry.json"
    report = verify_agent_proposal(
        candidate_citations,
        args.target,
        registry_path=registry_path,
        min_purity=args.min_purity,
    )

    print("\n" + "=" * 80)
    print(report.summary())
    print("=" * 80)

    if not report.is_verified:
        print(f"\n[CORRELATION FAILED] Fix for {args.target} could not be verified.")
        sys.exit(1)
    else:
        print(f"\n[CORRELATION VERIFIED] Fix for {args.target} verified on-target with {report.purity*100:.2f}% purity.")
        sys.exit(0)


if __name__ == "__main__":
    main()
