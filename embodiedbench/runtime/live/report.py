"""Turn episode telemetry into the two numbers a live run is judged by.

    python -m embodiedbench.runtime.live.report /data/koe/nav_telemetry

**How fast is it really.** Not the engine's tick rate in a microbenchmark --
the acceleration a *training run* actually received, which is the sim seconds
the couriers walked divided by the wall seconds the fleet was alive. Every gap
between the two is somewhere the fleet was not walking: queueing behind another
episode, waiting on the policy, booting.

**How much of the failure is the simulator's.** ``stuck`` and ``walk_timeout``
arrive at the policy wearing the same clothes as a bad action -- they are
refusals, they charge time, and the return goes down. That is defensible only
while the rate is low and *known*. Unknown, it is a systematic bias with the
policy's name on it: the gradient learns to avoid edges the navmesh cannot
walk, which is a fact about the map, not about being a good courier.

Everything here is read-only over the JSONL the envs append to. It answers on
partial runs, because the interesting moment to ask is usually mid-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def load(directory: Path) -> Iterator[dict[str, Any]]:
    """Every record under a telemetry directory, tolerating a live writer.

    The last line of a file being appended to right now can be short; that is
    normal, not corruption, and it costs one episode out of the report rather
    than the report itself.
    """
    for path in sorted(directory.glob("episodes-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"episodes": 0}

    outcomes: Counter[str] = Counter()
    walk_sim_seconds = 0.0
    hops = recoveries = degraded = busy_waits = 0
    busy_seconds = wall_seconds = 0.0
    pose_errors: list[float] = []
    engine_seconds = graph_seconds = 0.0
    workers: Counter[int] = Counter()
    delivered = 0

    for record in records:
        counters = record.get("counters", {})
        outcomes.update(counters.get("outcomes", {}))
        hops += int(counters.get("hops", 0))
        recoveries += int(counters.get("recoveries", 0))
        degraded += 1 if counters.get("degraded") else 0
        busy_waits += int(counters.get("busy_waits", 0))
        busy_seconds += float(counters.get("busy_wait_seconds", 0.0))
        wall_seconds += float(record.get("wall_seconds") or 0.0)
        workers[record.get("pid", -1)] += 1
        for hop in record.get("hops", []):
            if "sim_seconds" in hop:
                walk_sim_seconds += float(hop["sim_seconds"])
            if "pose_error_cm" in hop:
                pose_errors.append(float(hop["pose_error_cm"]))
            # Only arrived hops: a refused hop is charged a floor, not a
            # travel price, so including them would compare two different
            # things and flatter the ratio.
            if hop.get("outcome") == "arrived" and "graph_seconds" in hop:
                engine_seconds += float(hop.get("sim_seconds", 0.0))
                graph_seconds += float(hop["graph_seconds"])
        summary = record.get("summary") or {}
        if summary.get("delivered"):
            delivered += int(summary["delivered"])

    # The wall clock the fleet was actually alive, not the sum of episode
    # durations: episodes on different instances overlap, and summing them
    # would report an acceleration the run never had.
    stamps = [r["closed_at"] for r in records if r.get("closed_at")]
    span = (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0

    total_walks = sum(outcomes.values())
    sim_failures = outcomes.get("stuck", 0) + outcomes.get("timeout", 0) \
        + outcomes.get("walk_timeout", 0)

    return {
        "episodes": len(records),
        "worker_processes": len(workers),
        "episodes_per_worker": dict(sorted(workers.items())),
        "hops": hops,
        "outcomes": dict(outcomes),
        # THE bias number: what fraction of walks failed for reasons the
        # policy could not have avoided, and which the policy is charged for.
        "sim_failure_rate": (round(sim_failures / total_walks, 4)
                             if total_walks else None),
        "recoveries": recoveries,
        "episodes_degraded_to_album": degraded,
        "delivered": delivered,
        "walk_sim_seconds": round(walk_sim_seconds, 1),
        "wall_span_seconds": round(span, 1),
        # THE throughput number: sim seconds walked per wall second, across
        # the whole fleet. Above 1.0 means the world moved faster than the
        # clock on the wall; it is bounded by how much of the wall clock goes
        # to the policy rather than the engine.
        "sim_seconds_per_wall_second": (round(walk_sim_seconds / span, 3)
                                        if span > 0 else None),
        "busy_waits": busy_waits,
        "busy_wait_seconds": round(busy_seconds, 1),
        # Queueing as a share of the run: the honest measure of "is the fleet
        # too small for this rollout width".
        "busy_share_of_wall": (round(busy_seconds / span, 4)
                               if span > 0 else None),
        # How much more the engine charges for the same hop than the graph
        # price every budget in the task is still computed from. 1.0 means the
        # two agree; above 1.0, the live env is quietly running a harder
        # version of the benchmark than the offline one.
        "clock_inflation": (round(engine_seconds / graph_seconds, 4)
                            if graph_seconds > 0 else None),
        "engine_seconds_arrived": round(engine_seconds, 1),
        "graph_seconds_arrived": round(graph_seconds, 1),
        "pose_error_cm": {
            "n": len(pose_errors),
            "max": round(max(pose_errors), 1) if pose_errors else None,
            "p50": round(sorted(pose_errors)[len(pose_errors) // 2], 1)
            if pose_errors else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", action="store_true",
                        help="machine-readable, for a dashboard")
    args = parser.parse_args(argv)

    records = list(load(args.directory))
    report = summarize(records)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if not report["episodes"]:
        print(f"no episode records under {args.directory}")
        return 1
    print(f"episodes            {report['episodes']} "
          f"across {report['worker_processes']} worker processes")
    print(f"hops                {report['hops']}  {report['outcomes']}")
    print(f"sim failure rate    {report['sim_failure_rate']}   "
          f"(stuck/timeout charged to the policy)")
    print(f"recoveries          {report['recoveries']}   "
          f"degraded episodes {report['episodes_degraded_to_album']}")
    print(f"throughput          {report['sim_seconds_per_wall_second']} "
          f"sim-sec per wall-sec  "
          f"({report['walk_sim_seconds']}s walked / "
          f"{report['wall_span_seconds']}s wall)")
    print(f"queueing            {report['busy_waits']} waits, "
          f"{report['busy_wait_seconds']}s "
          f"({report['busy_share_of_wall']} of wall)")
    print(f"clock inflation     {report['clock_inflation']}x  "
          f"(engine {report['engine_seconds_arrived']}s vs graph "
          f"{report['graph_seconds_arrived']}s on arrived hops)")
    print(f"pose error cm       {report['pose_error_cm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
