"""leakcheck -- a tabular data-leakage detector.

Public API:
    check                 -- run all detectors, return a Report.
    make_leaky_dataset    -- deterministic synthetic data with planted leaks.
    Report, Finding       -- structured result types.
    Severity, Category    -- enums used throughout.
"""

from __future__ import annotations

from .core import check
from .data import make_leaky_dataset
from .detectors import (
    LeakageThresholds,
    detect_contamination,
    detect_duplicate_rows,
    detect_target_leakage,
    detect_trivial_features,
)
from .report import Category, Finding, Report, Severity

__all__ = [
    "Category",
    "Finding",
    "LeakageThresholds",
    "Report",
    "Severity",
    "check",
    "detect_contamination",
    "detect_duplicate_rows",
    "detect_target_leakage",
    "detect_trivial_features",
    "make_leaky_dataset",
]


def main() -> int:
    """Console-script entry point (delegates to the CLI)."""
    from .cli import main as _main

    return _main()
