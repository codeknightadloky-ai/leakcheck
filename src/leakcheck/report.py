"""Structured report types for leakcheck.

A :class:`Report` is an ordered collection of :class:`Finding` objects, each of
which describes one suspected data-leakage / data-quality problem together with
a :class:`Severity`. Everything here is plain data so a report can be turned
into JSON, printed, or asserted on in tests without touching pandas.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    """Ordered severity levels (CRITICAL is worst)."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        """Lower rank == more severe (CRITICAL == 0)."""
        return _SEVERITY_ORDER.index(self)


_SEVERITY_ORDER: list[Severity] = [
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
]


class Category(StrEnum):
    """The kind of problem a finding describes."""

    TARGET_LEAKAGE = "target_leakage"
    CONTAMINATION = "contamination"
    DUPLICATE_ROWS = "duplicate_rows"
    CONSTANT_FEATURE = "constant_feature"
    ID_LIKE_FEATURE = "id_like_feature"


@dataclass(frozen=True)
class Finding:
    """A single detected issue.

    Attributes:
        category: What class of problem this is.
        severity: How bad it is.
        message: Human-readable one-line explanation.
        feature: Column the finding relates to, if any.
        metrics: Numeric evidence backing the finding (e.g. correlation).
        recommendation: Suggested remediation.
    """

    category: Category
    severity: Severity
    message: str
    feature: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        d["metrics"] = {k: round(float(v), 6) for k, v in self.metrics.items()}
        return d


@dataclass
class Report:
    """An ordered set of findings plus convenience accessors."""

    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def extend(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    def sorted(self) -> list[Finding]:
        """Findings ordered by severity (worst first), stable within a level."""
        return sorted(self.findings, key=lambda f: f.severity.rank)

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def counts(self) -> dict[str, int]:
        """Number of findings per severity, in severity order."""
        out: dict[str, int] = {s.value: 0 for s in _SEVERITY_ORDER}
        for f in self.findings:
            out[f.severity.value] += 1
        return out

    @property
    def worst(self) -> Severity | None:
        """The most severe level present, or None for an empty report."""
        if not self.findings:
            return None
        return min((f.severity for f in self.findings), key=lambda s: s.rank)

    def has_blocking(self) -> bool:
        """True if any CRITICAL or HIGH finding exists (fails CI / gates)."""
        return any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total": len(self.findings),
                "worst": self.worst.value if self.worst else None,
                "counts": self.counts(),
                "has_blocking": self.has_blocking(),
            },
            "findings": [f.to_dict() for f in self.sorted()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Compact human-readable rendering."""
        if not self.findings:
            return "leakcheck: no issues detected."
        lines: list[str] = []
        counts = self.counts()
        summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        lines.append(f"leakcheck: {len(self.findings)} finding(s) [{summary}]")
        for f in self.sorted():
            tag = f.severity.value.upper()
            loc = f" [{f.feature}]" if f.feature else ""
            lines.append(f"  {tag:8}{loc} {f.message}")
            if f.metrics:
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in f.metrics.items())
                lines.append(f"           metrics: {metric_str}")
            if f.recommendation:
                lines.append(f"           fix: {f.recommendation}")
        return "\n".join(lines)
