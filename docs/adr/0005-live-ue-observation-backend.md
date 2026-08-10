# ADR-0005: UE as a stateless renderer behind the album contract

- **Status:** accepted (experimental branch `live-ue`)
- **Date:** 2026-08-09
- **Milestone:** live-UE track A (affects M10)
- **Deciders:** repository owner

## Context

The goal is online GRPO training for the courier environment with UE as the
observation backend: many instances in parallel, sim time faster than wall
clock. The audit that preceded this branch established the decisive fact:
**`CourierEnv.sim_seconds` is event-driven bookkeeping advanced by declared
action costs. There is no wall-clock coupling anywhere in the transition
system.** The Python environment already steps in near-zero wall time, and
every pose it could ever want photographed is fully determined by state it
already owns -- node position, bearing to the neighbour, signal phase from
`sim_seconds` parity, obstacle kind from the (map, seed) hash. "Faster than
wall clock" therefore reduces to render throughput per pose, and UE is needed
as a camera, not as a transition engine.

The observation side of `CourierEnv` is pure filesystem convention: up to five
album roots serving `images/<node>/toward_<neighbour>[suffix].png`, gated by
two visibility sidecars whose absence silently disables the corresponding
charging mechanic. All album paths are constructed in exactly three methods --
`_plain_frame`, `signal_frame_for`, `obstacle_frame_for` -- plus the
`album_coverage` report.

The wire protocol, cache contract and pool discipline are specified in
`docs/LIVE_UE_SPEC.md` (a verbatim copy shared with the SimWorld2 branch that
carries the service half; the golden fixtures under
`tests/golden/nav_render_v0/` pin both sides to one set of bytes).

## Decision

**UE is a deterministic on-demand frame renderer standing behind the album
contract, and the integration is a `CourierEnv` subclass, not a refactor.**

`LiveCourierEnv` (`embodiedbench/runtime/live/env.py`) materialises a
per-episode cache directory shaped exactly like an album, points every album
root of the stock constructor at it, and overrides only the three frame
lookups (plus `album_coverage`) to render-on-miss through a `RenderPool` and
then defer to the stock path lookup. Everything that decides *what* may be
shown -- the visibility gates, the one-lamp grouping, the served-size gate,
the signal phase, the obstacle field, the viewpoint entitlement -- is the
stock code running unchanged over a directory that fills itself in.
Downstream consumers (CourierSession, FrameAliases, the training adapter's
`PIL.Image.open`) cannot tell a live album from a baked one; a test makes the
claim literal by pointing a stock `CourierEnv` at the cache and getting the
same files.

The visibility sidecars are **copied from a baked album, never invented**:
visibility is a property of scene and camera geometry, so identical poses give
identical visibility, and the measured claims stay valid for live renders of
the same poses. *(Superseded in part -- see the addendum: the poses are
identical for street/obstacle frames only, so the copy is split by validity
and the signal sidecar is a warned opt-in.)* Absent a sidecar source, the
mechanics are silently off -- the same "silence is not consent" default as a
bare album.

`LiveCourierGymEnv` (`embodiedbench/runtime/live/gym_adapter.py`) is the
training adapter: a `CourierGymEnv` subclass that overrides only `reset` (the
stock one constructs `CourierEnv` inline, entangled with `/data/murray` album
defaulting, and offers no construction seam -- copying its eight lines of
bookkeeping was cheaper than editing a file this branch avoids touching). It
registers the same CLI way the stock adapter does, so nothing in the
gitignored vendor/ checkout changes:

    +env_registry.Courier=embodiedbench.runtime.live.gym_adapter.LiveCourierGymEnv

## Alternatives considered

**A provider-interface refactor of `CourierEnv`** -- extract a `FrameProvider`
protocol, inject album/live implementations. Rejected for this branch: it
edits the most load-bearing file in the repository to serve an experiment, and
the three lookup methods *are* the provider interface in all but name. If a
third backend ever appears, the refactor can be done then, with this subclass
as one of its two worked examples.

**An `EmbodiedRuntime` LIVE mode** -- implement the protocol stack's contract
(`runtime/core.py`), for which `RuntimeMode.LIVE` (`schemas/runtime.py`) is
reserved. Rejected because the courier agent does not live on the protocol
stack: `CourierEnv` is driven by `CourierSession` and `CourierGymEnv`, is not
an `EmbodiedRuntime`, and has no state-digest machinery. A LIVE runtime over
the vendored text engine would render frames for an environment the training
loop does not use. **`RuntimeMode.LIVE` stays reserved for what it names**: a
future runtime where UE owns transitions (track B in the spec -- the embodied
pawn under `global_sync` stepping). That is a different contract with a
different determinism story, and giving its name to a frame cache would spend
the enum value on the wrong thing.

## The design tension with PLAN.md 7.1

PLAN.md 7.1 says of the live mode: "Never required for every RL worker." An
online-rollout trainer wants every worker rendering through UE, which reads as
a contradiction. Track A respects the constraint's *substance*:

- **Transitions never depend on UE.** The renderer cannot touch the clock,
  the charges or the trajectory by construction -- it only adds files to a
  directory. The conformance test drives the same seed and action script over
  a live backend, a dead one and no album at all, and requires identical
  nodes, `sim_seconds` and rewards.
- **A dead instance degrades to album mode.** A render failure marks the
  episode degraded and every lookup falls through to whatever the cache
  already holds; the episode continues, text-navigable, exactly as over a
  bare album. No RL worker *requires* a live instance to finish its episode
  -- live UE is how the frames get good, not how the world keeps turning.

What 7.1 was guarding against -- a training loop that stalls when UE does --
cannot happen here. What it did not anticipate -- wanting fresh frames on
every worker -- is throughput, not coupling, and the pool's least-loaded
dispatch over N stateless instances is the whole answer to it.

## Consequences

- The live package is additive: `embodiedbench/runtime/live/` plus tests and
  fixtures. No existing file changes on this branch.
- Byte-identity of frames holds only within one service process (the cache
  guarantees per-(episode, key) idempotency; nothing guarantees two service
  processes rasterise identically). Replay conformance across processes must
  keep media digests out of state comparison, which PLAN.md 7.2 already
  prescribes: pixels need not match; state must.
- The lamp close-up camera matches the bake (1280 px long edge, FOV 40, eye
  165 cm) but v0 aims along the crossing from the node rather than at the
  lens, because the lamp-pose sidecar the bake's aiming used has no live
  export yet. `LiveCourierEnv._lamp_item` is the one place to refine when it
  does.
- The pool's quarantine statefile is written without a file lock, documented
  in `pool.py`: one trainer process per host today, atomic replace, and the
  worst cross-process cost is one relearned quarantine. Revisit if two
  trainers ever share a host.
- The SimWorld2 branch owns the service, the fleet and the track-B substrate
  (spec sections 6). Changes to the wire protocol are made in the spec copy
  and golden fixtures of **both** repos or not at all.

## Addendum (2026-08-09): sidecar transfer is split by validity

The decision text above argued the sidecar copy from "identical poses give
identical visibility" and, three bullets later, admitted the v0 lamp-aiming
simplification -- without ever connecting the two. The connection is the
whole point: the premise holds only where the live camera stands where the
bake's camera stood.

- **Street and obstacle frames: the premise holds.** The live env renders
  them from the same street camera pose the obstacle bake photographed, so
  `obstacle_visibility.json` transfers and is copied from
  `obstacle_sidecar_root`.
- **Lamp close-ups: it does not.** `bake_real_lamps.py` stands the camera on
  the lamp-junction line, yawed at the *lens* and pitched at the head -- its
  own analysis notes that the 24 cm bracket offset alone is five degrees of
  aim at 3 m, and that aiming at the pole base photographs the pavement. The
  v0 live renderer stands at the node with pitch 0 (until this branch the
  wire protocol could not even express pitch). The bake's legible set and
  `lamp_px` measurements therefore certify frames the live env never
  produces, and copying them can attach red-light charges to frames that do
  not show the lamp -- the exact defect the visibility gate exists to
  prevent.

Consequently the single `sidecar_source_root` is replaced by
`obstacle_sidecar_root` (valid transfer; obstacles chargeable whenever it is
given) and `signal_sidecar_root` (an explicit opt-in that logs a WARNING and
is documented as invalid for scored runs until a lamp_pose export lands and
`_lamp_item` sends the bake's aimed pose with the protocol's new
`pitch_deg`). The default live env therefore runs with obstacles chargeable
and signals off. The split also matches the stock album layout -- the two
sidecars live in *different* albums (real-lamp vs viewpoint-matched
obstacle), which one source directory could never express. Finally, LiveAlbum
now re-syncs the sidecars to its constructor's parameters whenever an episode
directory is reused, so a stale file left by a differently-configured run
cannot switch on a mechanic the current config did not ask for.
