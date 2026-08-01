# ADR-0004: M0 frozen decisions

- **Status:** accepted
- **Date:** 2026-07-31
- **Milestone:** M0
- **Deciders:** repository owner

PLAN.md M0 accepts when "the v1 courier profile set, reference
correctness/performance machines, target R1 model candidates, and minimum Paris
certified-road-coverage threshold are frozen in the M0 report."

Where PLAN.md 14.1 supplies a default for an unresolved input, that default is
taken and recorded here as the frozen value. Where it says a claim is blocked
until an owner decides, that stays **unresolved** and the dependent claim stays
blocked. Freezing a value we cannot justify would make a later gate pass on
fiction.

## 1. v1 courier profile set — FROZEN

`walker_novice`, `scooter_standard`, `multi_modal_courier`.

PLAN.md 8.2's fourth example, `scooter_veteran`, is **excluded from v1**. PLAN.md
conditions it on being "scientifically justified and disclosed", and no such
justification exists. PLAN.md 14.1's default for this input is exactly the three
above, and PLAN.md 17 lists hidden courier attributes as a risk.

Every outcome-relevant field of each profile must be visible in the reset
observation and recorded in `EpisodeSpec`, trajectory metadata, and evaluator
output (PLAN.md 8.2). Exact field values are defined at M3, when the overlay
compiler determines which affordances Paris can actually support.

## 2. Minimum Paris certified-road-coverage threshold — FROZEN at 90%

PLAN.md M2 requires "the certified episode region covers at least the M0-pinned
fraction of non-degenerate authored road length, preventing a vacuous
certificate over a tiny subgraph", with a default of 90% unless an approved
bounded-district ADR changes the evaluation bounds before M2.

Frozen at **90% of non-degenerate authored road length**. Non-degenerate excludes
the zero-length segments PLAN.md 3.3 names (current indices 186 and 187, both
under 5 mm).

If M2 measurement shows Paris cannot reach 90%, PLAN.md 17's go/no-go applies:
repair the source or select a bounded district by a new ADR. Do not fabricate
long links, and do not lower this number to fit the result.

## 3. Reference machines — PARTIALLY FROZEN

**Correctness reference machine: frozen.** This host: Linux 7.0.0-28-generic,
8× NVIDIA L40S (46 GB each), `/data` on a 140 T array, Python 3.11.15 in the
`simnav` conda environment. Its exact package set is pinned by the
`verification_tool_env` entry of `BASELINE_MANIFEST.json`.

**Performance reference machine: frozen as the same host, provisionally.**
Latency and throughput numbers measured here are valid only against this
manifest and must be reported with it.

**Second independent machine: UNRESOLVED.** PLAN.md M7 requires "two pinned
machines reproduce exact discrete metrics and per-instance float metrics within
1e-6", and M12 requires two clean reference machines. Only one host is available.
Consequence, per PLAN.md 14.1: correctness work proceeds; the cross-machine
reproducibility assertions of M7 and M12 cannot pass until a second machine is
named.

## 4. Target R1 model candidates — UNRESOLVED

PLAN.md 14.1 lists "target VLM checkpoint, tokenizer/processor revision, trainer
containers, and GPU allocation" as not frozen, with the default: "no trainer
winner or throughput claim until pinned."

No VLM checkpoint, processor revision, or trainer container is present on this
host, and no GPU allocation has been assigned. This stays unresolved.

Consequence: **R1/M8 cannot start.** Its acceptance criteria are stated in terms
of "model, processor, trainer-container, and hardware hashes frozen in the R1
report", which cannot be produced. PLAN.md 16 schedules R1 from week 1 in
parallel with M0; that parallelism is unavailable, and the week-3 VLM-RL go/no-go
gate has no input. M9 depends on a passing R1 gate and is blocked behind it.

This is the single largest schedule consequence recorded at M0, and it is an
input decision, not an engineering blocker.

## 5. First benchmark claim — FROZEN

The Paris + DeliveryBench vertical slice. **Not** AnyMap generalization.

PLAN.md 12.3 requires at least one held-out map before any map-generalization
claim and at least one held-out content pack before any visual-domain claim.
PLAN.md 17's final go/no-go: if held-out maps are unavailable before M12, release
a Paris development environment, not a general benchmark.

## 6. Multi-agent — FROZEN out of scope

Per PLAN.md 1. DeliveryBench collaboration mechanics may remain in the vendored
implementation but are not exposed as a v1 feature, and no v1 protocol, track,
budget, or score accounts for them.
