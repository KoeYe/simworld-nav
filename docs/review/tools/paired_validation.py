"""Did the policy improve? A paired comparison of two validations.

Two validations are the same 64 seeds, run greedily, at two different steps.
Comparing their means compares two proportions, and with 64 binary episodes
that carries a standard error of about five points -- so a real five-point
gain and pure noise produce the same number. This project has already read
noise as signal twice that way: a baseline of 23.4% and one of 14.1% were
called a 16% improvement and then a 43% regression, and the only change
between them was ``Job 0`` becoming ``Job 1`` in one line of a tool's reply.
Greedy decoding is deterministic, but it is not stable: one token earlier in
the prompt sends a forty-turn trajectory somewhere else entirely.

The episodes are paired by construction -- same seeds, same order, one
trajectory each -- and pairing is what makes the comparison sensitive. Most of
the variance between two validations is *which seeds are hard*, and that
cancels when each seed is compared with itself. What is left is the seeds
whose outcome actually changed, which is what McNemar's test counts and what
a paired difference in earnings measures.

Reports, for a pair of steps:

* how many seeds went 0 -> paid and how many went paid -> 0, with McNemar's
  exact p, which uses only those two counts because the seeds that did not
  change carry no information about a change;
* the mean paired difference in earnings with a bootstrap interval, because
  earnings is the objective and a seed can improve without crossing zero;
* the seeds themselves, so a claimed improvement can be read.

Usage:
    python paired_validation.py <validation_data_dir> [--from 0] [--to 10]
    python paired_validation.py <dir> --all      # every consecutive pair
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load(path: Path) -> list[dict]:
    """One validation: its episodes in the order the dataset ran them."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact p for b vs c under H0: a change is equally likely either way.

    The seeds that did not change are deliberately absent from the arithmetic.
    Under the null the discordant pairs split like a fair coin, so this is a
    binomial sign test on those pairs alone -- which is the whole point: 60 seeds
    that failed both times tell you nothing about whether anything moved.
    """
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(diffs: list[float], draws: int = 20000, seed: int = 0) -> tuple[float, float]:
    """A 95% interval for the mean paired difference, resampling the pairs."""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(draws):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * draws)], means[int(0.975 * draws)])


def compare(before: list[dict], after: list[dict], label: str) -> dict:
    if len(before) != len(after):
        raise SystemExit(
            f"{label}: {len(before)} episodes vs {len(after)} -- the two "
            "validations must be the same seeds in the same order, or they are "
            "not paired and nothing below is valid.")

    diffs, gained, lost, moved = [], [], [], []
    for i, (b, a) in enumerate(zip(before, after)):
        eb, ea = float(b.get("score") or 0.0), float(a.get("score") or 0.0)
        diffs.append(ea - eb)
        if eb <= 0 < ea:
            gained.append(i)
        elif ea <= 0 < eb:
            lost.append(i)
        if abs(ea - eb) > 1e-9:
            moved.append((i, eb, ea))

    mean_before = sum(float(b.get("score") or 0.0) for b in before) / len(before)
    mean_after = sum(float(a.get("score") or 0.0) for a in after) / len(after)
    mean_diff = sum(diffs) / len(diffs)
    low, high = bootstrap_ci(diffs)
    p = mcnemar_exact(len(gained), len(lost))

    print(f"\n=== {label} ===")
    print(f"  earnings  {mean_before:.4f} -> {mean_after:.4f}   "
          f"paired mean difference {mean_diff:+.4f}")
    print(f"  95% bootstrap interval on that difference: "
          f"[{low:+.4f}, {high:+.4f}]"
          f"{'   (excludes zero)' if low > 0 or high < 0 else '   (includes zero)'}")
    print(f"  seeds that started earning : {len(gained)}  {gained if gained else ''}")
    print(f"  seeds that stopped earning : {len(lost)}  {lost if lost else ''}")
    print(f"  unchanged                  : {len(before) - len(gained) - len(lost)}")
    print(f"  McNemar exact p            : {p:.4f}"
          f"{'  -- a real change' if p < 0.05 else '  -- not distinguishable from noise'}")
    if moved and len(moved) <= 12:
        print("  every seed whose earnings moved:")
        for i, eb, ea in moved:
            print(f"    #{i:<3} {eb:6.2f} -> {ea:6.2f}")
    return {"label": label, "mean_before": mean_before, "mean_after": mean_after,
            "mean_diff": mean_diff, "ci": [low, high], "gained": len(gained),
            "lost": len(lost), "p": p}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path,
                        help="trainer.validation_data_dir")
    parser.add_argument("--from", dest="start", type=int, default=None)
    parser.add_argument("--to", dest="end", type=int, default=None)
    parser.add_argument("--all", action="store_true",
                        help="every consecutive pair, and first vs last")
    args = parser.parse_args()

    files = sorted(args.directory.glob("*.jsonl"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if len(files) < 2:
        print(f"{len(files)} validation dump(s) in {args.directory} -- "
              "need two before anything can be compared.")
        return 1
    steps = [int(f.stem) for f in files]
    print(f"validations found at steps: {steps}")

    if args.all:
        for a, b in zip(files, files[1:]):
            compare(load(a), load(b), f"step {a.stem} -> {b.stem}")
        if len(files) > 2:
            compare(load(files[0]), load(files[-1]),
                    f"step {files[0].stem} -> {files[-1].stem} (whole run)")
        return 0

    first = args.start if args.start is not None else steps[0]
    last = args.end if args.end is not None else steps[-1]
    a = args.directory / f"{first}.jsonl"
    b = args.directory / f"{last}.jsonl"
    for path in (a, b):
        if not path.exists():
            raise SystemExit(f"no validation dump at {path}")
    compare(load(a), load(b), f"step {first} -> {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
