"""Replay one episode in a clean process and print its hashes as JSON.

Used by the M0 gate to show determinism across process boundaries, not just
within one interpreter. Same-process replay cannot detect a fix that depends on
warm module state, and PLAN.md M4 requires byte-identical results from two clean
processes.

    python -m embodiedbench.baseline.replay_worker <spec.json>

``spec.json`` holds the ``run_episode`` keyword arguments including ``actions``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

from embodiedbench.baseline.replay import run_episode_sync


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m embodiedbench.baseline.replay_worker <spec.json>", file=sys.stderr)
        return 2

    spec = json.loads(Path(argv[0]).read_text())
    # The vendored engine prints per-move narration to stdout. Capture it so the
    # only thing on stdout is the JSON result the parent parses.
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise):
        record = run_episode_sync(**spec)

    json.dump(
        {
            "label": record.label,
            "step_count": len(record.steps),
            "hashes": record.hashes(),
            "score": record.score,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
