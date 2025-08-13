"""Command-line interface for leakcheck.

Examples:
    leakcheck data.csv --target y
    leakcheck train.csv --target y --test test.csv --json
    leakcheck --demo
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .core import check
from .data import make_leaky_dataset
from .detectors import LeakageThresholds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leakcheck",
        description="Detect data leakage and contamination in tabular datasets.",
    )
    parser.add_argument("data", nargs="?", help="Path to a training/full CSV file.")
    parser.add_argument("--target", "-t", help="Name of the target column.")
    parser.add_argument("--test", help="Optional held-out test CSV for contamination checks.")
    parser.add_argument(
        "--task",
        choices=["auto", "classification", "regression"],
        default="auto",
        help="Task type (default: auto-detect).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run against the built-in synthetic leaky dataset.",
    )
    parser.add_argument(
        "--fail-on-blocking",
        action="store_true",
        help="Exit with status 2 if any CRITICAL/HIGH finding is present.",
    )
    parser.add_argument(
        "--critical", type=float, default=0.98, help="Leakage score for CRITICAL (default 0.98)."
    )
    parser.add_argument(
        "--high", type=float, default=0.85, help="Leakage score for HIGH (default 0.85)."
    )
    parser.add_argument(
        "--medium", type=float, default=0.70, help="Leakage score for MEDIUM (default 0.70)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    thresholds = LeakageThresholds(critical=args.critical, high=args.high, medium=args.medium)

    if args.demo:
        _full, train, test = make_leaky_dataset()
        report = check(train, target="target", test=test, thresholds=thresholds)
    else:
        if not args.data or not args.target:
            parser.error("provide DATA and --target, or use --demo")
        df = pd.read_csv(args.data)
        test_df = pd.read_csv(args.test) if args.test else None
        report = check(
            df,
            target=args.target,
            test=test_df,
            task=args.task,
            thresholds=thresholds,
        )

    output = report.to_json() if args.json else report.to_text()
    print(output)

    if args.fail_on_blocking and report.has_blocking():
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
