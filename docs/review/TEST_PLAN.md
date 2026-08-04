# Test plan — courier environment, adversarial review

Written before any episode was played. Reviewer: an agent acting first as the
courier policy, then as a debugger of the environment it just played.

Goal stated by the requester: this benchmark should be good enough that an
embodied agent trained on it could plausibly deliver parcels in real Paris.
That sets the bar for every check below — not "does the code run" but "is what
the agent is shown sufficient, honest, and non-wasteful".

## Rules I hold myself to

From `docs/EVAL_BRIEF.md`, and they constrain the design of every experiment:

1. **While playing, I read only `session.observe()`** — its text, its
   photographs, its SVG map. No `courier_env.py`, no `env.light_here()`, no
   `route_nodes`. Play that consults the source is not evidence.
2. **One turn per process / full transcript.** I drive episodes through a
   harness script that persists session state to disk between turns, so I
   cannot smuggle Python state a real policy would not carry.
3. **I look at the pictures.** Photographs get read with the image tool, SVG
   maps get rasterised to PNG and read. A finding of the form "the text did not
   say X" is only valid once I have checked the picture did not say it either.
4. **Findings need the observation attached** — paste of the turn, the action,
   the result, and what I expected.

Order of work: play first (uncontaminated), then open the source and run the
config sweep, then write reports.

## Phase 0 — provenance and preconditions

Before trusting any number the environment reports.

| check | how | pass condition |
|---|---|---|
| test suite | `pytest tests/ -q` | 950 passed, 4 skipped (the brief's claim) |
| migration | `python -m embodiedbench.tools.migration_check` | exit 0 |
| albums present | count images, cross-check against `manifest.jsonl` | every manifest row resolves to a file on disk |
| graph attach | do album keys match compiled graph nodes | no orphan albums (the failure mode the migration doc describes) |
| interpreter | `/data/murray/miniconda3/bin/python` | imports `embodiedbench`, `cairosvg` |

Any mismatch here invalidates results downstream, so it is done first and
reported as a number, not a yes/no.

## Phase 1 — play as the courier (the part that earns the review)

Two episodes minimum required by the brief; I plan four, so that stride and
tier are each varied with the other held fixed.

| episode | tier | stride | why this cell |
|---|---|---|---|
| A | `solo` | `block` | the floor. If a competent reader cannot do this, nothing above matters |
| B | `pair` or `triple` | `block` | sequencing appears; tests whether the observation supports scheduling |
| C | same seed as A | `waypoint` | isolates stride: same city, same job, 5× the decisions. Tests the "same world, more turns" claim and hunts for waypoint-only redundancy |
| D | `solo`/`pair` | `block`, hazards on | deliberately walk toward a barrier and a red light to see what the frames show and what the world charges |

Per turn I record, in a live log: the observation text, which photographs I
actually opened, what I concluded from each, the action, the result, and one of
the five defect labels from the brief when applicable —
`missing` / `contradictory` / `misleading` / `inexpressible` / `unclear`.

Specific things I am hunting while playing:

- **Can I win without looking?** (the brief's #1 finding). I will run a
  deliberate arm where I choose actions from text alone and never open an
  image. If that delivers, perception is not being measured.
- **Is anything in the text giving away what only a picture should?** — barrier
  or light state leaking into candidate rows, feedback messages, or refusals.
- **Do refusals leak?** e.g. does `collect()` too far from the door state the
  distance (the brief says this was fixed — verify it stays fixed), does
  `hand_over()`, does `walk_to` on a blocked edge.
- **Are the nine verbs sufficient?** Every turn where I wanted something I could
  not express gets logged.
- **Is the numbered candidate list stable?** The brief admits it is re-derived;
  I test whether that produces an actual trap (same index, different street,
  no warning).
- **Does the frozen map mislead?** After `navigate()`, the route is frozen and
  position live. I will deviate deliberately and see whether the stale route
  reads as current advice.

## Phase 2 — configuration sweep (every setting, not a sample)

The requester asked for *all* settings/configs. The declared axes:

- `difficulty` ∈ {solo, pair, triple, shift, endless} — 5
- `stride` ∈ {waypoint, block} — 2
- `condition` ∈ {full, no_phone, visual} — 3
- album triple ∈ {street only, +signals, +obstacles, +both} — 4 mechanic states
- seeds — ≥3 per cell for anything I quote

Sweep design:

1. **Validity grid** — every `(difficulty × stride × condition)` = 30 cells
   constructed and reset, 3 seeds each, checked for: constructs without error,
   `allowed_tool_names()` matches the prompt, order count matches
   `Difficulty.SPEC`, clock is a fixed multiple of optimal (the docstring's
   central claim), episode terminates.
2. **Album-gate matrix** — for each of the 4 album states, confirm the claimed
   gating actually holds: no signal album ⇒ zero `red_crossings` charged and no
   light frames; no obstacle album ⇒ zero `blocked_attempts`. This is the
   "never charge for what the photographs cannot show" invariant, and it is the
   environment's main honesty claim.
3. **Reference policies** — floor (blind), floor (`sighted=True`), ceiling
   (router) on every validated cell, 6 seeds, to check the published table in
   `docs/RUNNING.md` reproduces. A published number that does not reproduce is
   a finding in itself.
4. **Invalid input** — unknown difficulty / stride / condition raise, not
   silently default; malformed replies, wrong arity, out-of-range indices,
   unknown tool names, three-strikes termination.
5. **Determinism** — same seed twice ⇒ identical trajectory digest; different
   seeds ⇒ different. Also across the two strides: same seed should give the
   same city and the same job.

## Phase 3 — input economy (the "no redundant, wasteful things" ask)

Measured, not asserted:

- **Token budget of the observation** — how many tokens per turn, what fraction
  is repeated verbatim from the previous turn, what fraction is the system
  prompt re-sent. Anything constant across every turn belongs in the system
  prompt, not the observation.
- **Image economy** — how many photographs per turn, how many are duplicates of
  each other (near-identical bearings, the known defect), how many are of
  streets the courier will never take, and the byte cost.
- **Field-level usefulness** — for each field in the observation, did any of my
  decisions across four episodes depend on it? A field no decision ever used is
  either dead weight or a symptom of a missing task.
- **Sufficiency, the other direction** — the count of turns where I needed
  something no field carried.

## Phase 4 — harness and tool contract

- prompt/dispatch symmetry under every condition (the constructor claims to
  enforce it — try to break it)
- cost accounting: does each verb charge what the table says, is time charged
  on rejected actions, can a rejected action be a free probe
- `check_map` / `navigate` / `look` — do they return more than they should
- budget/termination: three malformed replies, step ceiling, shift clock,
  endless hour
- `summary()` fields — every one cross-checked against what actually happened
  in a recorded episode (walked_m vs sum of legs, sim_seconds vs sum of
  charges, on_time vs deadlines)

## Phase 5 — deliverables

- `docs/review/REPORT.md` — detailed: every finding with observation, action,
  result, expectation, severity, and the sweep tables in full.
- `docs/review/report.html` — visual: the trajectories turn by turn with the
  actual photographs inline, the map, the action taken, and the defect markers,
  plus the sweep results as charts/tables.

Severity scale used in both, adapted from the brief's own ordering:

- **S1** win-without-looking / harness exploit — invalidates the measurement
- **S2** unwinnable state, livelock, or a number in the observation that is false
- **S3** missing or contradictory information that a policy must work around
- **S4** waste: redundant text, duplicate images, unused fields
- **S5** cosmetic / wording

## What would make me say this is not yet NeurIPS-oral

Stated in advance so the conclusion is not fitted after the fact:

1. a blind text-only policy matching a sighted one on the validated tier
2. any published table in the repo that does not reproduce
3. more than one condition rung declared but unvalidated (currently 2 of 3 —
   already true, and the honesty of admitting it is not the same as having a
   perception rung)
4. the observation containing information only a photograph should carry
5. an unwinnable episode arising from the environment rather than the policy
