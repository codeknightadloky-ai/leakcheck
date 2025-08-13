"""High-level entry point tying the detectors into a single :class:`Report`."""

from __future__ import annotations

import pandas as pd

from .detectors import (
    LeakageThresholds,
    detect_contamination,
    detect_duplicate_rows,
    detect_target_leakage,
    detect_trivial_features,
)
from .report import Report


def check(
    df: pd.DataFrame,
    target: str,
    test: pd.DataFrame | None = None,
    task: str = "auto",
    thresholds: LeakageThresholds | None = None,
    random_state: int = 0,
) -> Report:
    """Run every detector over ``df`` and return a combined :class:`Report`.

    Args:
        df: Training frame (or the full dataset if ``test`` is omitted).
        target: Name of the label column.
        test: Optional held-out frame; if given, train/test contamination is
            checked in addition to the single-frame checks.
        task: ``"classification"``, ``"regression"`` or ``"auto"``.
        thresholds: Custom leakage-score cut-offs.
        random_state: Seed for mutual-information estimation.
    """
    report = Report()
    report.extend(
        detect_target_leakage(
            df,
            target,
            task=task,
            thresholds=thresholds,
            random_state=random_state,
        )
    )
    report.extend(detect_trivial_features(df, target=target))
    report.extend(detect_duplicate_rows(df))
    if test is not None:
        report.extend(detect_contamination(df, test))
    return report
