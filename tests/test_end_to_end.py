"""End-to-end tests over the synthetic dataset and the CLI."""

from __future__ import annotations

import json

from leakcheck import check, make_leaky_dataset
from leakcheck.cli import main
from leakcheck.report import Category, Finding, Severity


def test_synthetic_dataset_is_deterministic() -> None:
    full_a, train_a, test_a = make_leaky_dataset(seed=0)
    full_b, train_b, test_b = make_leaky_dataset(seed=0)
    assert full_a.equals(full_b)
    assert train_a.equals(train_b)
    assert test_a.equals(test_b)


def test_check_flags_all_planted_problems() -> None:
    _full, train, test = make_leaky_dataset(seed=0, overlap=30)
    report = check(train, target="target", test=test)

    by_cat: dict[Category, list[Finding]] = {}
    for f in report.findings:
        by_cat.setdefault(f.category, []).append(f)

    # 1. planted target leak
    leaks = by_cat.get(Category.TARGET_LEAKAGE, [])
    assert any(f.feature == "leaky_probe" for f in leaks)
    probe = next(f for f in leaks if f.feature == "leaky_probe")
    assert probe.severity in (Severity.CRITICAL, Severity.HIGH)

    # 2. planted contamination
    contam = by_cat.get(Category.CONTAMINATION, [])
    assert contam and contam[0].metrics["overlapping_rows"] == 30.0

    # 3. constant column
    assert any(f.feature == "batch_flag" for f in by_cat.get(Category.CONSTANT_FEATURE, []))

    # 4. ID-like column
    assert any(f.feature == "customer_id" for f in by_cat.get(Category.ID_LIKE_FEATURE, []))

    # Overall report should block a pipeline.
    assert report.has_blocking()


def test_clean_column_not_flagged_as_leak() -> None:
    _full, train, _test = make_leaky_dataset(seed=0)
    report = check(train, target="target")
    leak_features = {f.feature for f in report.findings if f.category is Category.TARGET_LEAKAGE}
    assert "clean_signal" not in leak_features
    assert "noise_a" not in leak_features


def test_cli_demo_text(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["--demo"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "leaky_probe" in out
    assert "contamination" in out.lower()


def test_cli_demo_json_and_fail_flag(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = main(["--demo", "--json", "--fail-on-blocking"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["summary"]["has_blocking"] is True
    assert rc == 2  # blocking findings -> non-zero exit


def test_cli_reads_csv(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    _full, train, test = make_leaky_dataset(seed=1)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    rc = main([str(train_path), "--target", "target", "--test", str(test_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["summary"]["total"] > 0
    features = {f.get("feature") for f in payload["findings"]}
    assert "leaky_probe" in features
