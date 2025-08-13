"""Unit tests for the individual detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leakcheck.detectors import (
    detect_contamination,
    detect_duplicate_rows,
    detect_target_leakage,
    detect_trivial_features,
)
from leakcheck.report import Category, Severity


def _classification_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "leak": y.astype(float),  # perfect copy of target
            "weak": y + rng.normal(scale=2.0, size=n),  # noisy, low signal
            "noise": rng.normal(size=n),
            "target": y,
        }
    )


def test_perfect_copy_is_critical_leakage() -> None:
    df = _classification_frame()
    findings = detect_target_leakage(df, target="target")
    leaks = [f for f in findings if f.feature == "leak"]
    assert leaks, "expected the perfect-copy feature to be flagged"
    assert leaks[0].severity == Severity.CRITICAL
    assert leaks[0].metrics["leakage_score"] > 0.98
    assert all(f.category is Category.TARGET_LEAKAGE for f in findings)


def test_pure_noise_is_not_flagged() -> None:
    df = _classification_frame()
    findings = detect_target_leakage(df, target="target")
    flagged = {f.feature for f in findings}
    assert "noise" not in flagged
    assert "weak" not in flagged


def test_target_leakage_is_deterministic() -> None:
    df = _classification_frame()
    a = detect_target_leakage(df, target="target", random_state=0)
    b = detect_target_leakage(df, target="target", random_state=0)
    assert [f.to_dict() for f in a] == [f.to_dict() for f in b]


def test_regression_correlated_feature_flagged() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=500).cumsum()  # many unique values -> regression
    df = pd.DataFrame(
        {
            "almost_target": y + rng.normal(scale=1e-3, size=500),
            "noise": rng.normal(size=500),
            "target": y,
        }
    )
    findings = detect_target_leakage(df, target="target", task="auto")
    features = {f.feature for f in findings}
    assert "almost_target" in features
    assert "noise" not in features


def test_contamination_detects_shared_rows() -> None:
    base = pd.DataFrame({"a": range(100), "b": range(100, 200), "target": [0, 1] * 50})
    train = base.iloc[:80].reset_index(drop=True)
    test = base.iloc[80:].reset_index(drop=True)
    # Copy 10 train rows into test -> 10 / (20 + 10) overlap.
    test = pd.concat([test, train.iloc[:10]], ignore_index=True)
    findings = detect_contamination(train, test)
    assert len(findings) == 1
    f = findings[0]
    assert f.category is Category.CONTAMINATION
    assert f.metrics["overlapping_rows"] == 10.0
    assert f.severity == Severity.CRITICAL  # > 10% overlap


def test_no_contamination_when_disjoint() -> None:
    train = pd.DataFrame({"a": range(0, 50), "target": [0, 1] * 25})
    test = pd.DataFrame({"a": range(50, 100), "target": [0, 1] * 25})
    assert detect_contamination(train, test) == []


def test_duplicate_rows_detected() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [1, 1, 2, 3]})
    findings = detect_duplicate_rows(df)
    assert len(findings) == 1
    assert findings[0].metrics["duplicate_rows"] == 1.0


def test_constant_and_id_like_features() -> None:
    n = 200
    df = pd.DataFrame(
        {
            "const": [7] * n,
            "row_id": [f"ID{i}" for i in range(n)],
            "value": np.random.default_rng(0).normal(size=n),
            "target": np.random.default_rng(0).integers(0, 2, size=n),
        }
    )
    findings = detect_trivial_features(df, target="target")
    cats = {(f.feature, f.category) for f in findings}
    assert ("const", Category.CONSTANT_FEATURE) in cats
    assert ("row_id", Category.ID_LIKE_FEATURE) in cats
    # A normal numeric column must not be flagged.
    assert all(f.feature != "value" for f in findings)


def test_trivial_features_skips_target() -> None:
    df = pd.DataFrame({"const": [1, 1, 1, 1], "target": [1, 1, 1, 1]})
    findings = detect_trivial_features(df, target="target")
    assert all(f.feature != "target" for f in findings)
