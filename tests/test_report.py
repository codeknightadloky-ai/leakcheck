"""Tests for the Report / Finding data types."""

from __future__ import annotations

import json

from leakcheck.report import Category, Finding, Report, Severity


def _finding(sev: Severity, feature: str = "f") -> Finding:
    return Finding(
        category=Category.TARGET_LEAKAGE,
        severity=sev,
        message="msg",
        feature=feature,
        metrics={"leakage_score": 0.9},
    )


def test_severity_ordering() -> None:
    assert Severity.CRITICAL.rank < Severity.HIGH.rank < Severity.LOW.rank


def test_report_sorted_by_severity() -> None:
    report = Report()
    report.add(_finding(Severity.LOW))
    report.add(_finding(Severity.CRITICAL))
    report.add(_finding(Severity.MEDIUM))
    order = [f.severity for f in report.sorted()]
    assert order == [Severity.CRITICAL, Severity.MEDIUM, Severity.LOW]


def test_worst_and_blocking() -> None:
    report = Report()
    assert report.worst is None
    assert not report.has_blocking()
    report.add(_finding(Severity.MEDIUM))
    assert report.worst == Severity.MEDIUM
    assert not report.has_blocking()
    report.add(_finding(Severity.CRITICAL))
    assert report.worst == Severity.CRITICAL
    assert report.has_blocking()


def test_counts() -> None:
    report = Report()
    report.add(_finding(Severity.CRITICAL))
    report.add(_finding(Severity.CRITICAL))
    report.add(_finding(Severity.LOW))
    counts = report.counts()
    assert counts["critical"] == 2
    assert counts["low"] == 1
    assert counts["high"] == 0


def test_json_roundtrip_shape() -> None:
    report = Report()
    report.add(_finding(Severity.HIGH))
    payload = json.loads(report.to_json())
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["worst"] == "high"
    assert payload["findings"][0]["category"] == "target_leakage"
    assert payload["findings"][0]["metrics"]["leakage_score"] == 0.9


def test_to_text_empty_and_nonempty() -> None:
    assert "no issues" in Report().to_text()
    report = Report()
    report.add(_finding(Severity.CRITICAL))
    text = report.to_text()
    assert "CRITICAL" in text
    assert "leakage_score" in text
