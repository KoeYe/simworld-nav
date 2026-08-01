"""Run the pipeline against every adversarial map and check its verdicts.

    python -m embodiedbench.compiler.stress.run

A stress case passes when the pipeline reaches a *verdict* and that verdict
matches what the case expects. Crashing is always a failure; concluding "this
map is unusable, here is why" is a success, because that is the pipeline doing
its job.

Expectations are deliberately partial. Most cases only assert "do not crash",
because over-specifying turns a stress suite into a change-detector. The cases
that do assert a verdict assert the one thing they were built to probe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from embodiedbench.compiler.pipeline import compile_map
from embodiedbench.compiler.stress.generator import STRESS_CASES, StressCase, build_stress_workspace

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = Path("/data/murray/stress_maps")
DEFAULT_OUT = REPO_ROOT / "artifacts" / "verification" / "STRESS"


def evaluate(case: StressCase, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare a pipeline verdict against the case's expectations."""
    checks: list[dict[str, Any]] = []

    def add(name: str, expected: Any, observed: Any, passed: bool) -> None:
        checks.append(
            {
                "name": name,
                "expected": expected,
                "observed": observed,
                "status": "pass" if passed else "fail",
            }
        )

    crashed = "error" in result
    add("reaches_a_verdict_without_crashing", True, not crashed, not crashed)
    if crashed:
        return checks

    if case.expect_nav_mode is not None:
        observed = result["navigation"]["mode"]
        add("navigation_mode", case.expect_nav_mode, observed, observed == case.expect_nav_mode)

    if case.expect_grade is not None:
        add("grade", case.expect_grade, result["grade"], result["grade"] == case.expect_grade)

    for flag in case.expect_flags:
        observed = [f["code"] for f in result["quality_findings"]]
        add(f"flag[{flag}]", flag, observed, flag in observed)

    if case.expect_solvable is not None:
        observed = bool(result.get("validation", {}).get("passes"))
        add("solvable", case.expect_solvable, observed, observed == case.expect_solvable)

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedbench.compiler.stress.run", description=__doc__)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--keep", action="store_true", help="keep the generated maps")
    args = parser.parse_args(argv)

    cases = STRESS_CASES
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in STRESS_CASES if c.name in wanted]

    workspace = Path(args.workspace).resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    build_stress_workspace(workspace, cases)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    failures = 0
    print(f"stress workspace: {workspace}")
    for case in cases:
        started = time.time()
        try:
            result = compile_map(case.name, base_dir=str(workspace))
        except Exception as exc:  # noqa: BLE001 - a crash is the finding
            result = {"map": case.name, "error": f"{type(exc).__name__}: {exc}"[:400]}
        elapsed = round(time.time() - started, 1)

        checks = evaluate(case, result)
        failed = [c for c in checks if c["status"] == "fail"]
        failures += bool(failed)

        record = {
            "case": case.name,
            "attacks": case.attacks,
            "note": case.note,
            "seconds": elapsed,
            "status": "fail" if failed else "pass",
            "checks": checks,
            "result": result,
        }
        records.append(record)

        if "error" in result:
            summary = f"CRASH {result['error'][:70]}"
        elif result.get("unusable"):
            summary = f"unusable  code={result['failure']['code']:22s} stage={result['failure']['stage']}"
        else:
            analysis = result["analysis"]
            summary = (
                f"grade={result['grade']:4s} nav={result['navigation']['mode']:14s} "
                f"card={analysis['cardinal_fraction']:5.1%} n={analysis['node_count']:5d} "
                f"solv={result['validation'].get('solvability_rate', 0):4.0%} "
                f"flags={[f['code'] for f in result['quality_findings']]}"
            )
        marker = "FAIL" if failed else "ok  "
        print(f"  [{marker}] {case.name:24s} {elapsed:5.1f}s  {summary}")
        for check in failed:
            print(f"           expected {check['name']}={check['expected']!r} "
                  f"got {check['observed']!r}")

    summary = {
        "schema": "embodiedbench/stress_suite/v0.1",
        "cases": len(records),
        "failed": failures,
        "workspace": str(workspace),
        "records": records,
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"{len(records) - failures}/{len(records)} stress cases behaved as expected")
    print(f"wrote {out_dir / 'report.json'}")

    if not args.keep:
        shutil.rmtree(workspace, ignore_errors=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
