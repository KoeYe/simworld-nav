"""Many UE instances behind one ``render``: least-loaded dispatch, quarantine.

The multiplexing decision comes from the spec (section 3): renders are
self-contained and the service restores the level between requests, so there
is nothing to lease per episode. A batch goes to whichever healthy instance
has the least in flight, ties broken round-robin, and more envs than
instances is the normal case, not a degraded one.

Failure handling is the spec's two-strikes rule (section 5): an instance
failing health twice in a row is quarantined and readmitted on a successful
probe. A transport failure during a render counts as a strike too -- it is
the same evidence a probe would have gathered, arriving earlier -- and the
batch fails over to another instance within the same call, so a dying
instance costs latency, not an episode. The pool never kills a process; the
fleet owns lifecycle, and the nav side's whole authority over an instance is
declining to send it work.

Quarantine is persisted to a small statefile beside the endpoints file, so a
restarted trainer does not spend its first batches rediscovering a dead
instance. **No file lock, and that is a documented decision, not an
oversight**: today one trainer process runs per host (verl's env workers
share this pool in-process), so the statefile has a single writer. The write
is atomic (temp file + rename) so a reader never sees a torn file; if two
trainer processes ever share a host, the worst case is one relearning a
quarantine the other already knew, which costs two probes. A lock buys
nothing until then, and file locks held across NFS are their own incident.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .client import RenderServiceError, ServiceUnreachable, UERenderClient
from .protocol import RenderBatch, RenderResult

# The env var the endpoints file is found under when no explicit path is given.
ENDPOINTS_ENV = "EB_UE_ENDPOINTS"
# Strikes in a row before an instance stops being offered work.
QUARANTINE_STRIKES = 2


class NoHealthyInstance(RenderServiceError):
    """Every instance is quarantined or failed for this batch.

    Still a ``RenderServiceError``, so a caller that degrades to album mode on
    "the render backend is unavailable" needs exactly one except clause.
    """

    code = "no_healthy_instance"


class EndpointsError(ValueError):
    """The endpoints file is missing or malformed. Refused loudly at
    construction: a pool that silently starts empty renders nothing and looks
    like a network problem."""


@dataclass
class _Member:
    """One instance and everything the pool believes about it."""

    id: str
    base_url: str
    map_name: str
    gpu_uuid: str
    client: UERenderClient
    strikes: int = 0
    quarantined: bool = False
    in_flight: int = 0
    dispatched: int = 0     # lifetime batches, the round-robin tiebreak
    quarantined_at: float = 0.0

    def state(self) -> dict[str, Any]:
        return {"id": self.id, "base_url": self.base_url,
                "quarantined": self.quarantined, "strikes": self.strikes,
                "in_flight": self.in_flight, "dispatched": self.dispatched}


class RenderPool:
    """Loads endpoints.json, dispatches batches, quarantines the dying."""

    def __init__(
        self,
        endpoints_path: str | Path | None = None,
        *,
        client_factory: Callable[..., UERenderClient] = UERenderClient,
        render_timeout_s: float | None = None,
    ):
        raw = endpoints_path or os.environ.get(ENDPOINTS_ENV)
        if not raw:
            raise EndpointsError(
                f"no endpoints file: pass a path or set {ENDPOINTS_ENV}. A pool "
                "with no instances is not a fallback, it is a typo.")
        self.endpoints_path = Path(raw)
        if not self.endpoints_path.exists():
            raise EndpointsError(f"endpoints file does not exist: {self.endpoints_path}")
        try:
            data = json.loads(self.endpoints_path.read_text())
        except ValueError as error:
            raise EndpointsError(
                f"endpoints file is not JSON: {self.endpoints_path} ({error})") from None
        instances = data.get("instances") or []
        if not instances:
            raise EndpointsError(f"endpoints file lists no instances: {self.endpoints_path}")

        kwargs: dict[str, Any] = {}
        if render_timeout_s is not None:
            kwargs["render_timeout_s"] = render_timeout_s
        self.members: list[_Member] = []
        for row in instances:
            base_url = str(row["base_url"])
            member_id = str(row.get("id") or base_url)
            self.members.append(_Member(
                id=member_id, base_url=base_url,
                map_name=str(row.get("map_name") or ""),
                gpu_uuid=str(row.get("gpu_uuid") or ""),
                client=client_factory(base_url, instance_id=member_id, **kwargs),
            ))

        # Beside the endpoints file, as the spec says, so whoever looks at the
        # fleet's config sees the nav side's opinion of it in the same place.
        self.statefile = self.endpoints_path.with_name(
            self.endpoints_path.stem + ".quarantine.json")
        self._load_quarantine()

    # ── dispatch ─────────────────────────────────────────────────────────────

    def render(self, batch: RenderBatch) -> tuple[RenderResult, ...]:
        """Send one batch to the best instance, failing over on transport death.

        Per-*item* failures come back in the results untouched -- they are the
        caller's information, not evidence against the instance. Only "no HTTP
        conversation happened" and "the engine is down" are strikes.
        """
        tried: set[str] = set()
        while True:
            member = self._pick(tried)
            if member is None:
                if not self._readmit_one(tried):
                    raise NoHealthyInstance(
                        f"no healthy instance for batch of {len(batch.requests)} "
                        f"({len(self.members)} configured, "
                        f"{sum(m.quarantined for m in self.members)} quarantined)")
                continue
            member.in_flight += 1
            member.dispatched += 1
            try:
                results = member.client.render(batch)
            except ServiceUnreachable:
                self._strike(member)
                tried.add(member.id)
                continue
            except RenderServiceError as error:
                if error.code == "engine_down":
                    self._strike(member)
                    tried.add(member.id)
                    continue
                raise
            finally:
                member.in_flight -= 1
            member.strikes = 0
            return results

    def _pick(self, exclude: set[str]) -> _Member | None:
        candidates = [m for m in self.members
                      if not m.quarantined and m.id not in exclude]
        if not candidates:
            return None
        return min(candidates, key=lambda m: (m.in_flight, m.dispatched, m.id))

    # ── health ───────────────────────────────────────────────────────────────

    def check_health(self) -> dict[str, str]:
        """Probe every instance; apply strikes, quarantine, and readmission.

        Returns ``{instance_id: "ok" | "degraded" | "quarantined"}`` so a
        caller can log the fleet's shape in one line.
        """
        out: dict[str, str] = {}
        for member in self.members:
            if self._probe(member):
                out[member.id] = "ok"
            else:
                out[member.id] = "quarantined" if member.quarantined else "degraded"
        self._save_quarantine()
        return out

    def _probe(self, member: _Member) -> bool:
        try:
            member.client.healthz()
        except RenderServiceError:
            self._strike(member)
            return False
        member.strikes = 0
        if member.quarantined:
            member.quarantined = False
            member.quarantined_at = 0.0
        return True

    def _readmit_one(self, exclude: set[str]) -> bool:
        """When nothing is admitted, probe the quarantined before giving up.

        Readmit-on-probe is what makes quarantine a pause rather than a
        verdict: an instance the fleet restarted comes back the moment it
        answers, without anyone editing a file.
        """
        changed = False
        for member in self.members:
            if member.quarantined and member.id not in exclude:
                if self._probe(member):
                    changed = True
        if changed:
            self._save_quarantine()
        return changed

    def _strike(self, member: _Member) -> None:
        member.strikes += 1
        if member.strikes >= QUARANTINE_STRIKES and not member.quarantined:
            member.quarantined = True
            member.quarantined_at = time.time()
        self._save_quarantine()

    # ── the statefile ────────────────────────────────────────────────────────

    def _load_quarantine(self) -> None:
        if not self.statefile.exists():
            return
        try:
            data = json.loads(self.statefile.read_text())
        except ValueError:
            # A torn or hand-mangled statefile must not take the pool down;
            # the worst it recorded was pessimism, and probes rebuild that.
            return
        quarantined = set(data.get("quarantined") or [])
        strikes = data.get("strikes") or {}
        for member in self.members:
            if member.id in quarantined:
                member.quarantined = True
            member.strikes = int(strikes.get(member.id, 0))

    def _save_quarantine(self) -> None:
        payload = {
            "version": 0,
            "quarantined": sorted(m.id for m in self.members if m.quarantined),
            "strikes": {m.id: m.strikes for m in self.members if m.strikes},
        }
        # Atomic on POSIX: a reader sees the old file or the new one, never a
        # prefix. This is the whole of the "no lock needed for v0" story --
        # single writer per host, torn reads impossible by construction.
        temporary = self.statefile.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.statefile)

    # ── reporting ────────────────────────────────────────────────────────────

    def state(self) -> list[dict[str, Any]]:
        return [m.state() for m in self.members]
