"""CLI entrypoint for running TonerHound benchmark evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from tonerhound.benchmark.runner import run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="TonerHound Benchmark Runner")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="research/data/test",
        help="Path to ExtractBench dataset directory (default: research/data/test)",
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default="EXP-001",
        help="Experiment identifier (default: EXP-001)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="research/experiments/EXP-001.json",
        help="Output path for structured JSON results",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="research/experiments/EXP-001.md",
        help="Output path for Markdown report",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_json = Path(args.output_json) if args.output_json else None
    out_md = Path(args.output_md) if args.output_md else None

    run_benchmark_suite(
        data_dir=data_dir,
        experiment_id=args.exp_id,
        output_json=out_json,
        output_md=out_md,
    )


if __name__ == "__main__":
    main()
