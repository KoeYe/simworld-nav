# ADR-0003: How defects in vendored code are fixed

- **Status:** accepted
- **Date:** 2026-07-31
- **Milestone:** M0
- **Deciders:** repository owner

## Context

M0's replay gate uncovered a defect in the vendored DeliveryBench engine that
blocks not only M0 but M4's "byte-identical `EpisodeSpec` in two clean
processes" and every cross-runtime conformance claim in PLAN.md 7.2:

`vendor/vagen/.../vlm_delivery/base/graph.py` lines 339, 354, 399, 417 use
`id(node)` — a CPython memory address — as the tie-breaker in Dijkstra's
priority queue. Equal-cost routes are routine on a street grid, so which of
several equally short routes is returned depends on where the allocator placed
the node objects. Measured on `small-city-11` seed 42: **9 of 12** clean-process
trials produced a different route, and therefore different distance, deadline,
energy, and arrival ordering, for identical inputs.

PLAN.md M0 also requires that "dirty user changes in existing repositories
remain untouched", and PLAN.md 14 says to adapt existing implementations behind
stable interfaces rather than fork them.

## Decision

Defects in `vendor/` are fixed by **recorded runtime patches in our layer**.
`vendor/` files are never edited.

Each patch is a `PatchRecord` carrying:

- the fully-qualified target it replaces;
- a written reason stating the defect and its consequence;
- the **sha256 of the vendored function source it replaced**;
- whether it was applied, and if not, why it refused.

A patch refuses to apply when the vendored source no longer matches its
premise — for the tie-break patch, when `id(` has disappeared from the function.
Applying anyway would turn a fix into an unreviewed behavior change. Refusing
loudly means a vendor bump surfaces as a decision rather than as a silent
revert or a silent double-fix.

Every milestone report embeds the active patch set, so no result is ever
reported without the modifications that produced it.

The same discipline applies to state-digest exclusions. When a field must be
left out of an authoritative-state hash, it goes in `documented_exclusions` with
the evidence, and the exclusion list is printed in the report — never into a
general denylist where an inert field and a real nondeterminism would be hidden
by the same mechanism. Two exclusions exist today (M0-F2 `Order.start_time`,
M0-F3 `dm.run_dir`), both wall-clock values that no code reads.

## Alternatives rejected

- **Edit `vendor/` directly.** Violates M0's untouched-repositories requirement
  and makes the diff invisible to anyone reading our source.
- **Fork the engine now.** PLAN.md 14 explicitly warns against a fifth copy, and
  a 31k-line fork on day one buys nothing that a two-function patch does not.
- **Fix upstream first.** Correct eventually, but it makes our milestone chain
  depend on an external review cycle. Upstreaming the tie-break fix remains
  worth doing; it is not a prerequisite.
- **Accept nondeterminism and loosen the gate.** This would make every
  reproducibility claim in the benchmark false, which is the one thing PLAN.md
  section 1 says the project is for.

## Consequences

- Determinism holds only in processes that call `apply_deterministic_patches()`.
  The runtime layer must apply it centrally at M1 so no consumer can forget.
- Routes chosen by the patched code may differ from any route the unpatched
  engine happened to return. No golden artifact predates this change, so nothing
  needs migrating; after M1, changing a patch is a breaking change under
  PLAN.md 5.4.
- The patch is deliberately minimal: same algorithm, same weights, same path
  reconstruction, different tie-breaker. Path *costs* are unaffected — only
  which of several equal-cost routes is returned becomes stable.
