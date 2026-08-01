# ADR-0001: Repository ownership, authoritative locations, and branch policy

- **Status:** accepted
- **Date:** 2026-07-31
- **Milestone:** M0
- **Deciders:** repository owner (qlab.ucsd2@gmail.com)

PLAN.md M0 requires "an approved ownership ADR [that] names the repository,
owners, branch policy, and authoritative locations for schemas, Delivery task
logic, compiler, and benchmark instances."

## Context

PLAN.md 14.1 lists the integration repository as "not named", with the default
"create `/home/murray/embodiedbench` only after approval". None of the five
local baselines PLAN.md 2.2 builds on exist on this host: `deliverybench`,
`nav_task`, `simworld_arena`, `embodied_data/molmo-motion`, and a VAGEN
checkout are all absent, as are `IMPLEMENTATION_PLAN.md` and
`PLAN_REVIEW_FEEDBACK.md`. What is present is CityCore Paris content, an
Unreal Engine 5.8.0-preview-1 install, a read-only SimWorld_SPEAR checkout with
built Linux editor binaries, and eight L40S GPUs.

## Decision

**Repository.** `/home/murray/simworld_nav`, a fresh git repository holding
PLAN.md alongside the implementation. Not `/home/murray/embodiedbench`: PLAN.md
offered that name only as a default in the absence of a decision, and PLAN.md
itself already lives here. The Python package inside it is `embodiedbench`, so
the module namespace PLAN.md 14 describes is unchanged.

**Baselines.** `vendor/vagen` holds a read-only clone of
`ymzhang0303/VAGEN@dev/env_v2`, pinned in `BASELINE_MANIFEST.json`. It is
excluded from git — it is 3.6 GB and is reference material, not our source.
`/data/koe/SimWorld_SPEAR/utils/simworld_task` substitutes for the absent
`nav_task`. `simworld_arena` and `molmo-motion` have no substitute and are
recorded as gaps; neither is on the M0-M7 path.

**Authoritative locations.**

| Concern | Authoritative location |
|---|---|
| Protocol schemas | `embodiedbench/schemas/` — this repository only |
| Delivery task logic | `vendor/vagen/.../vlm_delivery` **until** the M1 adapter lands; thereafter `embodiedbench/tasks/delivery/` |
| Map compiler | `embodiedbench/compiler/`, seeded from the VAGEN Paris exporter |
| Benchmark instances | `embodiedbench/benchmark/instances/`, published as hashed JSONL + manifest |
| Baseline pins | `BASELINE_MANIFEST.json` (portable) + `baseline_roots.local.json` (host, untracked) |
| Milestone evidence | `artifacts/verification/M<N>/report.json`, tracked; bulky logs untracked |

**Authoritative Delivery transition implementation.** The VAGEN `dev/env_v2`
pure-Python engine, adapted behind our runtime contract. PLAN.md 2.2 already
judged it the stronger of the two candidates, and the alternative (the released
DeliveryBench wrapper with a placeholder reward) does not exist on this host, so
the decision is forced rather than merely preferred. One consequence: the
de-vendoring milestone in PLAN.md 14 has one input, not two.

**Branch policy.** `main` is the integration branch. Milestone work happens on
`m<N>/<topic>` branches and merges only when that milestone's
`artifacts/verification/M<N>/report.json` has `status: pass` or a recorded,
justified `blocked`. Released artifacts are immutable; corrections create a new
version plus migration notes (PLAN.md 5.4).

**Vendor policy.** `vendor/` is never edited. Defects found in it are fixed in
our layer as recorded runtime patches that carry the original source's sha256
(see `embodiedbench/baseline/determinism.py` and ADR-0003), so a vendor bump
that changes the patched code fails loudly instead of silently reverting a fix.

## Consequences

- One authoritative Delivery implementation from the start; no fifth copy.
- Reference material is reproducible from a commit pin without bloating the repo,
  at the cost of requiring network access on a fresh checkout.
- `nav_task`'s navigation measures are unavailable. SPL and SoftSPL are not
  present in the SPEAR substitute and must be implemented for M7.
- No portal code to reuse; M12 starts from scratch or from
  `$SPEAR_ROOT/utils/city_livestream`.
