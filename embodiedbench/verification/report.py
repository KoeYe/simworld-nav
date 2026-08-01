"""The milestone report structure PLAN.md's verification standard requires."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from embodiedbench.artifacts.hashing import sha256_file, sha256_json


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"


@dataclass
class Assertion:
    """One machine-readable claim with its expected and observed values.

    ``blocked`` is distinct from ``fail``. PLAN.md's verification standard says a
    conditional skip is not a pass; it is equally important that an assertion we
    could not evaluate is not recorded as one we evaluated and failed.
    """

    name: str
    expected: Any
    observed: Any
    status: Status
    evidence: str | None = None
    note: str | None = None

    @classmethod
    def check(cls, name: str, expected: Any, observed: Any, **kwargs: Any) -> "Assertion":
        return cls(
            name=name,
            expected=expected,
            observed=observed,
            status=Status.PASS if expected == observed else Status.FAIL,
            **kwargs,
        )

    @classmethod
    def blocked(cls, name: str, expected: Any, reason: str, **kwargs: Any) -> "Assertion":
        return cls(
            name=name, expected=expected, observed=None, status=Status.BLOCKED, note=reason, **kwargs
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "expected": self.expected,
            "observed": self.observed,
            "status": self.status.value,
        }
        if self.evidence:
            out["evidence"] = self.evidence
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class ReportCommand:
    """A non-interactive command that produced evidence for this milestone."""

    description: str
    command: str
    exit_code: int | None = None
    log: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"description": self.description, "command": self.command}
        if self.exit_code is not None:
            out["exit_code"] = self.exit_code
        if self.log:
            out["log"] = self.log
        return out


def host_manifest() -> dict[str, Any]:
    """Hardware and OS identity of the machine that produced the evidence."""
    gpus: list[str] = []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if proc.returncode == 0:
            gpus = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        gpus = []
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": __import__("os").cpu_count(),
        "gpus": gpus,
    }


@dataclass
class MilestoneReport:
    """A milestone's verification evidence."""

    milestone: str
    title: str
    owner: str
    verifier: str
    started_at: str
    inputs: dict[str, Any] = field(default_factory=dict)
    commands: list[ReportCommand] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    deviations: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    human_review: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def start(cls, milestone: str, title: str, owner: str, verifier: str) -> "MilestoneReport":
        return cls(
            milestone=milestone,
            title=title,
            owner=owner,
            verifier=verifier,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def add_artifact(self, path: Path | str, role: str) -> None:
        """Record a produced artifact by content hash, so it cannot be swapped."""
        path = Path(path)
        entry: dict[str, Any] = {"path": str(path), "role": role, "exists": path.exists()}
        if path.exists() and path.is_file():
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = path.stat().st_size
        self.artifacts.append(entry)

    def absorb(self, prefix: str, result: dict[str, Any]) -> None:
        """Fold a sub-report's assertions in under a namespace prefix."""
        for raw in result.get("assertions", []):
            status = raw.get("status", "fail")
            self.assertions.append(
                Assertion(
                    name=f"{prefix}.{raw['name']}",
                    expected=raw.get("expected"),
                    observed=raw.get("observed"),
                    status=Status(status),
                )
            )

    @property
    def status(self) -> Status:
        """``fail`` if anything failed, else ``blocked`` if anything is blocked."""
        if any(a.status is Status.FAIL for a in self.assertions):
            return Status.FAIL
        if any(a.status is Status.BLOCKED for a in self.assertions):
            return Status.BLOCKED
        return Status.PASS

    def summary_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Status}
        for assertion in self.assertions:
            counts[assertion.status.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": "embodiedbench/milestone_report/v0.1",
            "milestone": self.milestone,
            "title": self.title,
            "status": self.status.value,
            "owner": self.owner,
            "verifier": self.verifier,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": host_manifest(),
            "inputs": self.inputs,
            "summary": self.summary_counts(),
            "commands": [c.to_dict() for c in self.commands],
            "assertions": [a.to_dict() for a in self.assertions],
            "artifacts": self.artifacts,
            "findings": self.findings,
            "deviations": self.deviations,
            "decisions": self.decisions,
            "human_review": self.human_review,
            "notes": self.notes,
        }
        # Self-hash last, over everything above, so the report cannot be edited
        # after the fact without detection.
        body["report_sha256"] = sha256_json(body)
        return body

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path
