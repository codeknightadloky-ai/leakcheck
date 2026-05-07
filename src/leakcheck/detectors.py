"""Leakage / data-quality detectors operating on pandas DataFrames.

Three families of check:

1. Target leakage    -- features that predict the target almost perfectly.
2. Contamination     -- rows shared between a train and a test split, plus
                        duplicate rows inside a single frame.
3. Trivial features  -- constant columns and ID-like (near-unique) columns.

The detectors are deterministic: mutual-information estimation is seeded, so
the same DataFrame always yields the same report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .report import Category, Finding, Severity


@dataclass(frozen=True)
class LeakageThresholds:
    """Score cut-offs for target-leakage severity (score in [0, 1])."""

    critical: float = 0.98
    high: float = 0.85
    medium: float = 0.70


def _infer_task(y: pd.Series, max_classes: int = 20) -> str:
    """Return ``"classification"`` or ``"regression"`` for a target column."""
    if y.dtype == bool or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nunique = y.nunique(dropna=True)
    if nunique <= max_classes:
        return "classification"
    return "regression"


def _encode(series: pd.Series) -> tuple[np.ndarray, bool]:
    """Encode a column to a numeric array.

    Returns the array and a flag indicating whether it should be treated as a
    discrete (categorical) feature for mutual-information estimation.
    """
    if series.dtype == bool:
        return series.astype(float).to_numpy(), True
    if not pd.api.types.is_numeric_dtype(series):
        # object / string / category -> integer codes, treated as discrete.
        codes, _ = pd.factorize(series, use_na_sentinel=False)
        return codes.astype(float), True
    numeric = series.astype(float)
    filled = numeric.fillna(numeric.median())
    discrete = bool(series.nunique(dropna=True) <= 20)
    return filled.to_numpy(), discrete


def _target_entropy(codes: np.ndarray) -> float:
    """Shannon entropy (nats) of a discrete target encoded as integer codes."""
    _, counts = np.unique(codes, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def _severity_for(score: float, thresholds: LeakageThresholds) -> Severity | None:
    if score >= thresholds.critical:
        return Severity.CRITICAL
    if score >= thresholds.high:
        return Severity.HIGH
    if score >= thresholds.medium:
        return Severity.MEDIUM
    return None


def detect_target_leakage(
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    thresholds: LeakageThresholds | None = None,
    random_state: int = 0,
) -> list[Finding]:
    """Flag features that predict ``target`` suspiciously well.

    For each feature two signals are computed and the larger is used as the
    leakage score:

    * absolute Pearson correlation with the (numeric-encoded) target, and
    * normalized mutual information -- MI divided by target entropy for
      classification, or absolute correlation for regression.

    A score near ``1.0`` means the feature essentially *is* the target, which in
    real pipelines almost always signals leakage.
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in DataFrame")
    thresholds = thresholds or LeakageThresholds()
    resolved_task = _infer_task(df[target]) if task == "auto" else task
    if resolved_task not in ("classification", "regression"):
        raise ValueError(f"unknown task {resolved_task!r}")

    y_codes, _ = _encode(df[target])
    features = [c for c in df.columns if c != target]
    if not features:
        return []

    n_rows = len(df)
    encoded: dict[str, np.ndarray] = {}
    discrete_mask: list[bool] = []
    # A discrete feature whose values are nearly all distinct makes mutual
    # information degenerate: every value maps to a single row, so MI trivially
    # equals the target entropy (norm_mi -> 1.0). That is an artifact of an
    # identifier, not real predictive leakage, so we distrust MI for such
    # columns and let the ID-like detector flag them instead.
    mi_unreliable: list[bool] = []
    for col in features:
        arr, discrete = _encode(df[col])
        encoded[col] = arr
        discrete_mask.append(discrete)
        unique_ratio = df[col].nunique(dropna=False) / n_rows if n_rows else 0.0
        mi_unreliable.append(discrete and unique_ratio > 0.5)

    x_matrix = np.column_stack([encoded[c] for c in features])

    if resolved_task == "classification":
        mi = mutual_info_classif(
            x_matrix,
            y_codes.astype(int),
            discrete_features=np.array(discrete_mask),
            random_state=random_state,
        )
        entropy = _target_entropy(y_codes.astype(int))
        norm_mi = mi / entropy if entropy > 0 else np.zeros_like(mi)
        norm_mi = np.where(mi_unreliable, 0.0, norm_mi)
    else:
        mi = mutual_info_regression(
            x_matrix,
            y_codes,
            discrete_features=np.array(discrete_mask),
            random_state=random_state,
        )
        # MI is unbounded for regression; fall back to correlation for the
        # normalized signal and keep raw MI as evidence only.
        norm_mi = np.zeros_like(mi)

    findings: list[Finding] = []
    for i, col in enumerate(features):
        feat = encoded[col]
        if np.std(feat) == 0 or np.std(y_codes) == 0:
            corr = 0.0
        else:
            corr = float(abs(np.corrcoef(feat, y_codes)[0, 1]))
        score = max(corr, float(np.clip(norm_mi[i], 0.0, 1.0)))
        severity = _severity_for(score, thresholds)
        if severity is None:
            continue
        findings.append(
            Finding(
                category=Category.TARGET_LEAKAGE,
                severity=severity,
                message=(
                    f"feature {col!r} predicts target {target!r} with "
                    f"score={score:.3f} (likely leakage)"
                ),
                feature=col,
                metrics={
                    "leakage_score": score,
                    "abs_corr": corr,
                    "norm_mi": float(np.clip(norm_mi[i], 0.0, 1.0)),
                    "raw_mi": float(mi[i]),
                },
                recommendation=(
                    "Confirm this column is available at prediction time and is "
                    "not derived from the target; drop it if it is."
                ),
            )
        )
    # Most suspicious feature first.
    findings.sort(key=lambda f: f.metrics["leakage_score"], reverse=True)
    return findings


def detect_duplicate_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> list[Finding]:
    """Flag exact duplicate rows within a single frame."""
    dup_mask = df.duplicated(subset=subset, keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup == 0:
        return []
    frac = n_dup / len(df)
    severity = Severity.MEDIUM if frac >= 0.05 else Severity.LOW
    return [
        Finding(
            category=Category.DUPLICATE_ROWS,
            severity=severity,
            message=f"{n_dup} duplicate row(s) ({frac:.1%} of the frame)",
            metrics={"duplicate_rows": float(n_dup), "duplicate_fraction": frac},
            recommendation="Deduplicate before splitting to avoid inflated metrics.",
        )
    ]


def detect_contamination(
    train: pd.DataFrame,
    test: pd.DataFrame,
    subset: list[str] | None = None,
) -> list[Finding]:
    """Flag rows in ``test`` that also appear in ``train`` (split contamination).

    Comparison is by row identity over ``subset`` columns (default: the columns
    common to both frames). Any overlap means the model is evaluated on rows it
    trained on, so the test score is optimistic.
    """
    if subset is None:
        subset = [c for c in train.columns if c in test.columns]
    if not subset:
        raise ValueError("no shared columns to compare train and test on")

    train_keys = train[subset].apply(lambda r: hash(tuple(r)), axis=1)
    test_keys = test[subset].apply(lambda r: hash(tuple(r)), axis=1)
    overlap_mask = test_keys.isin(set(train_keys))
    n_overlap = int(overlap_mask.sum())
    if n_overlap == 0:
        return []

    frac = n_overlap / len(test)
    if frac >= 0.10:
        severity = Severity.CRITICAL
    elif frac >= 0.01:
        severity = Severity.HIGH
    else:
        severity = Severity.MEDIUM
    return [
        Finding(
            category=Category.CONTAMINATION,
            severity=severity,
            message=(
                f"{n_overlap} test row(s) ({frac:.1%}) also occur in train "
                f"-- train/test contamination"
            ),
            metrics={
                "overlapping_rows": float(n_overlap),
                "overlap_fraction": frac,
                "test_rows": float(len(test)),
            },
            recommendation=(
                "Re-split with grouping/deduplication so no record appears in both train and test."
            ),
        )
    ]


def detect_trivial_features(
    df: pd.DataFrame,
    target: str | None = None,
    id_ratio: float = 0.95,
) -> list[Finding]:
    """Flag constant columns and ID-like (near-unique) columns."""
    findings: list[Finding] = []
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        nunique = int(series.nunique(dropna=False))
        if nunique <= 1:
            findings.append(
                Finding(
                    category=Category.CONSTANT_FEATURE,
                    severity=Severity.LOW,
                    message=f"feature {col!r} is constant (single value)",
                    feature=col,
                    metrics={"n_unique": float(nunique)},
                    recommendation="Drop constant columns; they carry no signal.",
                )
            )
            continue
        ratio = nunique / n if n else 0.0
        is_idish = pd.api.types.is_string_dtype(series) or pd.api.types.is_integer_dtype(series)
        if is_idish and ratio >= id_ratio and n >= 20:
            findings.append(
                Finding(
                    category=Category.ID_LIKE_FEATURE,
                    severity=Severity.MEDIUM,
                    message=(f"feature {col!r} is ID-like ({ratio:.0%} of rows are unique)"),
                    feature=col,
                    metrics={"unique_ratio": ratio, "n_unique": float(nunique)},
                    recommendation=(
                        "Identifiers leak row identity and don't generalize; drop or hash them out."
                    ),
                )
            )
    return findings
