"""CLI entrypoint for running TonerHound benchmark evaluations."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="TonerHound Benchmark Runner")
    parser.add_argument("--test-case", type=str, help="Path to test case JSON or directory")
    args = parser.parse_args()

    print("=== TonerHound Benchmark Harness ===")
    print("ExtractBench Official Evaluation Adapter Active.")
    print("Baseline Leader: LlamaExtract Agentic Plus (46.43% Word Grounding F1)")
    if args.test_case:
        print(f"Target test case: {args.test_case}")


if __name__ == "__main__":
    main()
