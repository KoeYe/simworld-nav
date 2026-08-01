"""Assemble ``artifacts/verification/M1/report.json``.

    python -m embodiedbench.verification.m1_report

Maps PLAN.md M1's six acceptance criteria to explicit assertions and runs the
gates non-interactively. Exits 0 on pass, 2 on blocked, 1 on fail.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from embodiedbench.artifacts.hashing import sha256_file
from embodiedbench.baseline.determinism import apply_deterministic_patches
from embodiedbench.schemas import SCHEMA_REGISTRY
from embodiedbench.schemas.base import SchemaVersion
from embodiedbench.schemas.fixtures import invalid_fixtures, uncovered_schemas, valid_fixtures
from embodiedbench.verification.report import Assertion, MilestoneReport, ReportCommand, Status

REPO_ROOT = Path(__file__).resolve().parents[2]
M1_DIR = REPO_ROOT / "artifacts" / "verification" / "M1"
SKELETON_DIR = REPO_ROOT / "artifacts" / "skeleton"
PYTHON = sys.executable

OWNER = "qlab.ucsd2@gmail.com"
VERIFIER = "claude-code (automated), pending human sign-off"


def _run(command: list[str], log_name: str) -> tuple[int, Path, str]:
    (M1_DIR / "logs").mkdir(parents=True, exist_ok=True)
    log_path = M1_DIR / "logs" / log_name
    proc = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)
    output = proc.stdout + proc.stderr
    log_path.write_text(output)
    return proc.returncode, log_path, output


def build_report() -> MilestoneReport:
    report = MilestoneReport.start(
        milestone="M1",
        title="Protocol schemas and contract test kit",
        owner=OWNER,
        verifier=VERIFIER,
    )

    registry = SCHEMA_REGISTRY()
    envelopes = {k: v for k, v in registry.items() if v.VERSIONED_ENVELOPE}
    report.inputs = {
        "schema_count": len(registry),
        "versioned_envelope_count": len(envelopes),
        "schemas": {
            schema_id: {"model": model.__name__, "version": model.SCHEMA_VERSION}
            for schema_id, model in sorted(envelopes.items())
        },
        "valid_fixture_count": len(valid_fixtures()),
        "invalid_fixture_count": len(invalid_fixtures()),
        "determinism_patches": apply_deterministic_patches().to_dict(),
        "baseline_manifest_sha256": sha256_file(REPO_ROOT / "BASELINE_MANIFEST.json"),
    }

    # ── criterion 1: contract tests for every schema ─────────────────────────
    code, log, _ = _run(
        [PYTHON, "-m", "pytest", "tests/test_schema_contracts.py", "-q", "-p", "no:cacheprovider"],
        "schema_contracts.log",
    )
    report.commands.append(
        ReportCommand(
            description="M1.1 round-trip, unknown-field, version, and invalid-fixture tests "
            "for every contract",
            command="python -m pytest tests/test_schema_contracts.py -q",
            exit_code=code,
            log=str(log.relative_to(REPO_ROOT)),
        )
    )
    report.assertions.append(Assertion.check("M1.1.schema_contract_tests_pass", 0, code))
    report.assertions.append(
        Assertion.check(
            "M1.1.every_versioned_schema_has_a_fixture", [], sorted(uncovered_schemas())
        )
    )
    report.assertions.append(
        Assertion(
            name="M1.1.contract_count_is_complete",
            expected="all PLAN.md M1 contracts present",
            observed=sorted(envelopes),
            status=Status.PASS if _covers_plan_contracts(envelopes) else Status.FAIL,
            note="PLAN.md M1 names WorldBundle, OverlaySpec, EnvironmentBundle, observations, "
                 "actions, events, runtime results, embodiment capabilities, EpisodeSpec, "
                 "trajectory, and score report",
        )
    )

    # ── criterion 2: adapter preserves vendored behavior ─────────────────────
    code, log, _ = _run(
        [PYTHON, "-m", "pytest", "tests/test_runtime_adapter.py", "-q", "-p", "no:cacheprovider"],
        "runtime_adapter.log",
    )
    report.commands.append(
        ReportCommand(
            description="M1.2 a fixed DeliveryBench action trace through the adapter three times, "
            "against the pre-adapter state hash",
            command="python -m pytest tests/test_runtime_adapter.py -q",
            exit_code=code,
            log=str(log.relative_to(REPO_ROOT)),
        )
    )
    report.assertions.append(Assertion.check("M1.2.adapter_preserves_vendor_behavior", 0, code))

    # ── criterion 3: terminated and truncated have separate tested causes ────
    code, log, output = _run(
        [
            PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "tests/test_schema_contracts.py", "-k", "termination or truncation or terminated",
        ],
        "termination_causes.log",
    )
    report.commands.append(
        ReportCommand(
            description="M1.3 terminated and truncated causes are tested separately",
            command="python -m pytest tests/test_schema_contracts.py -k 'termination or truncation'",
            exit_code=code,
            log=str(log.relative_to(REPO_ROOT)),
        )
    )
    report.assertions.append(Assertion.check("M1.3.termination_causes_tested_separately", 0, code))

    # ── criterion 4: typed errors ────────────────────────────────────────────
    report.assertions.append(
        Assertion(
            name="M1.4.typed_errors_for_version_frame_capability_payload",
            expected=["VersionError", "FrameError", "CapabilityError", "PayloadError"],
            observed=_typed_error_names(),
            status=Status.PASS if _typed_error_names() == [
                "CapabilityError", "FrameError", "PayloadError", "VersionError"
            ] else Status.FAIL,
        )
    )

    # ── criterion 5: walking skeleton ────────────────────────────────────────
    code, log, output = _run(
        [
            PYTHON, "-m", "embodiedbench.skeleton",
            "--out", str(SKELETON_DIR), "--repeat", "3",
        ],
        "walking_skeleton.log",
    )
    report.commands.append(
        ReportCommand(
            description="M1.5 walking skeleton through compiler, runtime, task, agent, "
            "trajectory writer, and evaluator; three runs",
            command="python -m embodiedbench.skeleton --out artifacts/skeleton --repeat 3",
            exit_code=code,
            log=str(log.relative_to(REPO_ROOT)),
        )
    )
    report.assertions.append(Assertion.check("M1.5.walking_skeleton_exits_zero", 0, code))
    report.assertions.append(
        Assertion.check(
            "M1.5.three_runs_produce_identical_hashes",
            True,
            "all 3 runs produced identical hashes" in output,
        )
    )

    code, log, _ = _run(
        [PYTHON, "-m", "pytest", "tests/test_walking_skeleton.py", "-q", "-p", "no:cacheprovider"],
        "walking_skeleton_tests.log",
    )
    report.commands.append(
        ReportCommand(
            description="M1.5 skeleton runs from a clean process with a randomized hash seed",
            command="python -m pytest tests/test_walking_skeleton.py -q",
            exit_code=code,
            log=str(log.relative_to(REPO_ROOT)),
        )
    )
    report.assertions.append(Assertion.check("M1.5.clean_process_skeleton_tests_pass", 0, code))

    summary_path = SKELETON_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        report.inputs["skeleton_summary"] = summary
        report.assertions.append(
            Assertion.check("M1.5.skeleton_reaches_task_logic", True, summary["deliveries"] >= 1,
                            note="a skeleton that never completes a delivery would not exercise "
                                 "the task or evaluator layers")
        )

    # ── criterion 6: schemas remain v0.x ─────────────────────────────────────
    majors = sorted({SchemaVersion.parse(m.SCHEMA_VERSION).major for m in envelopes.values()})
    report.assertions.append(
        Assertion.check("M1.6.all_schemas_remain_v0", [0], majors,
                        note="PLAN.md M1: no backward-compatibility promise before M12")
    )

    # ── full suite ───────────────────────────────────────────────────────────
    code, log, output = _run(
        [PYTHON, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"], "full_suite.log"
    )
    report.commands.append(
        ReportCommand(
            description="full contract and unit suite",
            command="python -m pytest tests/ -q",
            exit_code=code,
            log=str(log.relative_to(REPO_ROOT)),
        )
    )
    report.assertions.append(Assertion.check("M1.tests.full_suite_passes", 0, code))
    passed = ""
    for line in output.splitlines():
        if " passed" in line:
            passed = line.strip()
    report.inputs["test_summary"] = passed

    return report


def _covers_plan_contracts(envelopes: dict[str, Any]) -> bool:
    required = {
        "embodiedbench/world_bundle",
        "embodiedbench/overlay_spec",
        "embodiedbench/environment_bundle",
        "embodiedbench/observation",
        "embodiedbench/action_envelope",
        "embodiedbench/event",
        "embodiedbench/step_result",
        "embodiedbench/embodiment_capabilities",
        "embodiedbench/episode_spec",
        "embodiedbench/trajectory",
        "embodiedbench/score_report",
    }
    return required <= set(envelopes)


def _typed_error_names() -> list[str]:
    from embodiedbench.schemas.base import (
        CapabilityError,
        FrameError,
        PayloadError,
        SchemaError,
        VersionError,
    )

    return sorted(
        cls.__name__
        for cls in (VersionError, FrameError, CapabilityError, PayloadError)
        if issubclass(cls, SchemaError)
    )


def main() -> int:
    report = build_report()

    report.findings = [
        {
            "id": "M1-F1",
            "severity": "blocking-for-M4",
            "title": "The nav preset silently overrides the requested step budget",
            "detail": (
                "DeliveryBenchEnvConfig sets dynamic_max_steps_mult=2.5 for the `nav` preset, so "
                "deliverybench_env.py:462 recomputes max_steps per seed from a chained BFS oracle "
                "and clamps it to at least 20. The max_steps a caller passes is a floor, not the "
                "enforced budget."
            ),
            "impact": (
                "A benchmark instance that pins budgets.steps would not get the budget it pins, "
                "and truncation would be attributed to the wrong cause. PLAN.md 12.2 makes step "
                "budget a scoring budget, so this must be reconciled before M7 freezes instances."
            ),
            "resolution": (
                "The adapter exposes effective_step_budget() and derives truncation from it "
                "rather than from the requested value. M4 must decide whether benchmark "
                "instances pin the static budget or record the dynamic one."
            ),
        },
        {
            "id": "M1-F2",
            "severity": "fixed",
            "title": "Handler-level action failures are reported under `action_error`",
            "detail": (
                "The vendored env surfaces failed PICKUP/DROP_OFF/MOVE handlers as "
                "info['action_error'] (deliverybench_env.py:1001) and keeps the message in "
                "info['raw_info']['error']; there is no top-level 'error' key. An adapter reading "
                "info['error'] reports every handler failure as an accepted action."
            ),
            "impact": (
                "Would have made ActionResult.status wrong for every failed task action, breaking "
                "PLAN.md M6's typed-feedback requirement and the invalid-action metric."
            ),
            "resolution": "found by the invalid-action test and fixed in _action_error(); covered "
                          "by tests/test_runtime_adapter.py",
        },
        {
            "id": "M1-F3",
            "severity": "informational",
            "title": "The vendored engine emits no event stream",
            "detail": (
                "PLAN.md 7 requires structured events and PLAN.md 13 builds the portal on an "
                "event protocol, but the vendored env returns only an info dict. The adapter "
                "synthesizes a minimal stream (step_completed, action_error) with ids derived "
                "from the episode and a monotonic counter."
            ),
            "impact": (
                "Event granularity is currently far below what PLAN.md 13.1's timeline needs "
                "(no pickup, dropoff, hazard, or transport events). Richer events must come "
                "from the task layer at M4, not from more adapter inference."
            ),
            "resolution": "open; recorded so the portal work does not assume a richer stream exists",
        },
        {
            "id": "M1-F4",
            "severity": "informational",
            "title": "Reward arrives as one scalar with no components",
            "detail": (
                "PLAN.md 7 requires named reward_components and PLAN.md 11.3 separates task score "
                "from training shaping. The vendored env computes a single scalar."
            ),
            "impact": "Reward decomposition cannot be reported until the task layer owns it.",
            "resolution": "the adapter reports one component named `vendor_total` rather than "
                          "inventing a decomposition it did not compute",
        },
    ]

    report.deviations = [
        {
            "id": "M1-D1",
            "from": "PLAN.md M1: 'stub compiler artifact' in the walking skeleton",
            "to": "a real procgen compiler producer that reads the engine's own waypoint graph",
            "reason": (
                "A hand-written stub would agree with nothing. Compiling from the graph the "
                "runtime navigates makes the skeleton prove the contracts connect, and seeds the "
                "M2 Paris producer."
            ),
            "assertion_strength": "stronger than required",
        },
        {
            "id": "M1-D2",
            "from": "PLAN.md 5.4's minor-version compatibility rule",
            "to": "a differing minor version is rejected while the package is 0.x",
            "reason": (
                "PLAN.md M1 states there is no backward-compatibility promise before M12 and "
                "that early consumers pin exact schema revisions. Accepting 0.2 data into a 0.1 "
                "reader would contradict that. The 5.4 rule applies from 1.0."
            ),
        },
    ]

    report.human_review = [
        {
            "item": "M1 verification report sign-off",
            "checklist": "PLAN.md M1 acceptance criteria 1-6",
            "reviewer": None,
            "timestamp": None,
            "status": "pending",
        },
    ]

    for name, role in (
        ("artifacts/skeleton/world_bundle.json", "compiled base world"),
        ("artifacts/skeleton/environment_bundle.json", "derived environment"),
        ("artifacts/skeleton/episode_spec.json", "generated episode"),
        ("artifacts/skeleton/trajectory.json", "recorded trajectory"),
        ("artifacts/skeleton/score_report.json", "evaluator output"),
        ("artifacts/skeleton/summary.json", "skeleton summary and hashes"),
    ):
        report.add_artifact(REPO_ROOT / name, role)

    path = report.write(M1_DIR / "report.json")
    counts = report.summary_counts()
    print(f"M1 status: {report.status.value.upper()}")
    print(f"  assertions: {counts['pass']} pass, {counts['fail']} fail, {counts['blocked']} blocked")
    print(f"  schemas: {report.inputs['versioned_envelope_count']} versioned envelopes, "
          f"{report.inputs['valid_fixture_count']} valid / "
          f"{report.inputs['invalid_fixture_count']} invalid fixtures")
    print(f"  {report.inputs.get('test_summary', '')}")
    for assertion in report.assertions:
        if assertion.status is not Status.PASS:
            print(f"  [{assertion.status.value}] {assertion.name}")
            print(f"      expected {assertion.expected!r}")
            print(f"      observed {assertion.observed!r}")
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return {Status.PASS: 0, Status.FAIL: 1, Status.BLOCKED: 2}[report.status]


if __name__ == "__main__":
    sys.exit(main())
