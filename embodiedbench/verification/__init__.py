"""Milestone verification reports.

PLAN.md's verification standard requires every milestone to emit
``artifacts/verification/M<N>/report.json`` with input hashes, the
non-interactive commands used, machine-readable assertions carrying expected and
observed values, raw log and result hashes, timing, owner and verifier, and a
status of ``pass``, ``fail``, or ``blocked``.

This package implements that shape once so M0 through M12 report identically and
a reader can compare them.
"""

from embodiedbench.verification.report import (
    Assertion,
    MilestoneReport,
    ReportCommand,
    Status,
)

__all__ = ["Assertion", "MilestoneReport", "ReportCommand", "Status"]
