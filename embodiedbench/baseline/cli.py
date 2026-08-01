"""Non-interactive CLI for the M0 baseline freeze.

    python -m embodiedbench.baseline.cli freeze
    python -m embodiedbench.baseline.cli verify        # the M0 gate command

``verify`` exits 0 only when every recomputed digest matches the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from embodiedbench import __version__
from embodiedbench.artifacts.hashing import sha256_file
from embodiedbench.baseline.baseline_spec import BASELINE_ENTRIES, MISSING_BASELINES
from embodiedbench.baseline.manifest import (
    build_manifest,
    load_manifest,
    load_roots,
    verify_manifest,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "BASELINE_MANIFEST.json"
DEFAULT_ROOTS = REPO_ROOT / "baseline_roots.local.json"


def _generator_record() -> dict[str, str]:
    spec_file = Path(__file__).parent / "baseline_spec.py"
    return {
        "tool": "embodiedbench.baseline.cli",
        "package_version": __version__,
        "spec_sha256": sha256_file(spec_file),
        "manifest_module_sha256": sha256_file(Path(__file__).parent / "manifest.py"),
    }


def cmd_freeze(args: argparse.Namespace) -> int:
    roots = load_roots(Path(args.roots))
    manifest = build_manifest(
        BASELINE_ENTRIES,
        roots,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generator=_generator_record(),
    )
    out = manifest.to_dict()
    out["missing_baselines"] = MISSING_BASELINES
    Path(args.manifest).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.manifest} with {len(manifest.entries)} entries")
    for entry in manifest.entries:
        detail = entry.commit[:12] if entry.commit else ""
        if entry.digests:
            detail = f"{entry.digests[0]['level']}={entry.digests[0]['digest'][:12]}"
        print(f"  {entry.id:28s} {entry.kind:14s} {detail}")
    if MISSING_BASELINES:
        print(f"  ({len(MISSING_BASELINES)} PLAN.md baselines recorded as absent/substituted)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    roots = load_roots(Path(args.roots))
    result = verify_manifest(manifest, roots, skip_content=args.skip_content)

    for assertion in result.assertions:
        if not assertion.passed:
            print(f"FAIL {assertion.name}")
            print(f"     expected: {assertion.expected}")
            print(f"     observed: {assertion.observed}")
    for error in result.errors:
        print(f"ERROR {error}")

    summary = result.to_dict()
    print(
        f"{summary['status'].upper()}: {summary['assertion_count'] - summary['failed_count']}"
        f"/{summary['assertion_count']} assertions passed"
        + (" (content digests skipped)" if args.skip_content else "")
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.report}")
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodiedbench.baseline.cli", description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--roots", default=str(DEFAULT_ROOTS))
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="compute digests and write BASELINE_MANIFEST.json")
    freeze.set_defaults(func=cmd_freeze)

    verify = sub.add_parser("verify", help="recompute digests and compare (M0 gate)")
    verify.add_argument(
        "--skip-content",
        action="store_true",
        help="omit byte-level tree digests (fast smoke check; not valid for the gate)",
    )
    verify.add_argument("--report", help="write a machine-readable result JSON here")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
