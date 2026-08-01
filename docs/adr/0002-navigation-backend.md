# ADR-0002: The UE navigation backend the M0 gate asserts against

- **Status:** accepted
- **Date:** 2026-07-31
- **Milestone:** M0 (affects M2, M10)
- **Deciders:** repository owner

## Context

PLAN.md M0 accepts when "an archived command/log proves UE 5.8 opens
`ParisCity_FinalBlueprints`, builds NavMesh, and `vget /nav/random_points 100`
returns exactly 100 unique valid in-bounds Paris points", and PLAN.md 14.1 lists
"UE 5.8 plugin build containing SimWorld/UnrealCV and `/nav/*`" as an input
needed by M0/M2.

Two facts on this host contradict that wording.

1. `$SPEAR_ROOT/utils/simworld_task/navmesh.py` opens with: "The supported
   public backend is SPEAR's `navigation_service`. The legacy `vget /nav/...`
   command builders remain for migration notes and old scripts, but new release
   evidence and docs should use the SPEAR request helpers below." The `/nav/*`
   API PLAN.md names is deprecated in the very module PLAN.md points at.
2. `unrealcv` is present in `$SPEAR_ROOT/Plugins/` as source, but is **not** in
   `SimWorld.uproject`'s enabled-plugin list, and has no built Linux binaries.
   Asserting `vget /nav/*` would first require enabling and compiling a plugin
   the project has moved off.

## Decision

Keep the assertion, change the backend it is evaluated against.

The gate still requires, unchanged and falsifiable: the map opens in UE 5.8, a
NavMesh build completes, and exactly 100 **unique**, **finite**, **in-bounds**
navigable Paris points are returned. Only the API producing those points
changes, from UnrealCV `vget /nav/random_points` to the engine navigation system
that SPEAR's `navigation_service` itself queries.

The M0 probe (`tools/ue/m0_paris_nav_probe.py`) runs inside the editor's Python
interpreter and uses `unreal.NavigationSystemV1`. This is the same NavMesh data
`navigation_service` exposes; it avoids requiring a live PIE session and a
rendering context for a gate whose purpose is to prove the map and its
navigation data are usable at all.

"In-bounds" is checked against the Paris export bounds PLAN.md 3.2 records
(x ∈ [-40521.9, 20267.3] cm, y ∈ [-34177.5, 37414.4] cm) with a 200 m margin,
because NavMesh legitimately extends past the authored road graph. A point
outside that is not a Paris point and fails the gate.

## Consequences

- M0's UE gate no longer depends on building a plugin the project has retired.
- M10's live runtime should target `navigation_service` (and SPEAR's session
  state machine), not UnrealCV. `navmesh.py`'s `SpearNavigationRequest`
  helpers — `find_paths`, `get_random_reachable_points_in_radius`,
  `get_random_points` — are the interface to build against.
- M2's NavMesh validation of certified edges and both long synthetic connectors
  uses `navigation_service.find_paths`, not `vget /nav/path`.
- PLAN.md's text should be corrected; until then this ADR is the deviation
  record. The deviation is in the *mechanism*, not in the *strength* of the
  assertion.
