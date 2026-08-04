# Paris Delivery Benchmark — Evaluation Report

**Audit date:** 2026-08-02 UTC  
**Revision:** `5130047c0851fbd4c5b4837cf34ba9e4917e6e70`  
**Evaluator:** embodied courier first, debugger second  
**Plan:** [`BENCHMARK_TEST_PLAN.md`](BENCHMARK_TEST_PLAN.md)

## Executive verdict

The current version is a strong research prototype with unusually thoughtful
contracts, deterministic replay, extensive tests, full cached-view coverage,
and genuinely useful visual hazards. It is **not yet a NeurIPS-oral-quality
benchmark release and is not evidence for deployment in real Paris**.

The principal reason is construct validity. The only validated condition,
`full`, is explicitly a planning floor rather than a perception benchmark. The
two perception-oriented conditions are declared but unvalidated. The “Paris”
road network and imagery are a synthetic CityCore environment, with implausible
topology/name adjacencies and no demonstrated correspondence to current Paris
streets, pedestrian rules, traffic, accessibility, weather, construction, or
delivery operations. No physical embodiment, live-UE run, or real-world trial
was available in this audit.

Release recommendation: **research preview / internal benchmark, major revision
required before a public benchmark claim; prohibit real-world deployment use.**

## What I actually tested

I wrote the plan before executing tests, preserved the existing architecture
plan, and followed the repository's blind-play rule. During blind play I used
only `CourierSession.observe()`, ordered frames, and `CourierSession.step()`.
Only after the embodied episodes did I inspect implementation source.

### Audit environment

- The documented `python` executable does not exist. System `python3` is 3.12.3.
- The system interpreter had none of the principal project dependencies.
- `python3 -m venv .venv` failed because `ensurepip`/`python3-venv` is absent.
- Dependencies declared in `.[dev]` were installed into an isolated
  `.audit_deps` target and invoked with `PYTHONPATH=.audit_deps:.`.
- The available Conda Python is 3.14.6 but also lacks `gymnasium`.
- Cached Paris albums were present under `/data/murray`: 856 street frames,
  694 signal frames, and 240 obstacle frames.
- The vendored CityCore-Paris map source used by the runner was available.

These setup facts are part of the result: the documented copy/paste commands do
not run in this supplied environment.

## Embodied experience

### Blind solo, seed 0, `full`, block stride, all cached albums

Outcome: **1/1 delivered on time** in **34 turns** and **10.59 simulated
minutes**. I walked 537.9 m, hit 0 barriers, crossed 0 red lights, waited at 2
red lights, incurred 2 rejected interactions, and encountered 1 slowing hazard.
The delivery earned 5.43.

The images were causally useful. I avoided a clearly visible road closure and
read separate red/green pedestrian-light frames correctly. The direct phone
route was blocked, however, and the obvious adjacent quay was a dead end. I had
to discover a much wider loop through Rue de Sévigné, Rue de la Roquette, and
Avenue de Crimée before reaching Rue Oberkampf.

The key trajectory was:

```text
Quai Beaubourg
  -> Rue Saint-Antoine pickup (5)
  -> Boulevard du Temple
  X  visible road closure
  -> Quai Ménilmontant dead end
  -> Rue Saint-Antoine loop
  -> Rue de Sévigné
  -> red light / wait / green
  -> Rue de la Roquette
  -> Avenue de Crimée
  -> Rue Oberkampf (18 -> 15 -> 17)
  -> hand-over
```

Policy-visible reproducibility excerpts:

1. At the start, candidates both said “next junction 18 m”; `navigate()` then
   said “Quai Beaubourg — 1 junction, 29 m.” The accepted block action reported
   “29 m ... through 2 junctions.” These units are not coherent to an agent.
2. At Rue Saint-Antoine, `look(3)` returned doors `1, 3`; one block action moved
   54 m through 3 junctions, making door-level localization difficult.
3. A rejected `collect()` returned: “It is **36 m away; you need to be within
   8 m**.” This is exactly the free rangefinder leak the evaluation brief says
   was removed.
4. Photograph 1 on Boulevard du Temple visibly showed a road closure. Avoiding
   it prevented a charged collision, demonstrating that visual input can matter
   under hazards even though `full` itself is text-solvable in clean settings.
5. All three signal frames at Rue de Sévigné were red. `wait()` changed the
   corresponding frame paths to green; walking then produced no violation.
6. At Rue Oberkampf, the environment displayed number 18 but rejected delivery
   to 17 as 36 m away. `look(1)` said `13, 16 falling`; `look(2)` said `15
   falling`. A fresh phone query was required to disambiguate direction.

### Documented observation-only pair runner, seed 1

Calling the public example literally, without `env.reset()`, produced a zero-step
`OracleResult` with `trace=['no order was generated']`. The example in
`docs/RUNNING.md` therefore omits a required lifecycle call.

After an explicit reset:

| Reference variant | Delivered | Policy steps | Env turns | Simulated time | Walked | Blocked attempts |
|---|---:|---:|---:|---:|---:|---:|
| observation-only, blind | 0/2 | 119 | 249 | 146.8 min | 5,556.5 m | 53 |
| observation-only, sighted | 0/2 | 318 | 643 | 146.8 min | 9,240.6 m | 0 |

Sight correctly removed barrier collisions but did not improve delivery and
more than doubled wandering. This single seed is not a benchmark estimate, but
it contradicts the impression created by the nearby six-seed table strongly
enough that the shipped claims must be regenerated from this exact revision and
configuration.

## Configuration coverage

| Axis | Shipped values | Audit status |
|---|---|---|
| Difficulty | solo, pair, triple, shift, endless | Constructor/tests cover all; manual solo and pair references executed |
| Stride | waypoint, block | Both covered by tests; manual block executed |
| Condition | full, no_phone, visual | Full is the only validated rung; other two must not be scored |
| Hazard albums | none, signal, obstacle, both | Unit matrix plus all-albums play; migration reports complete cached coverage |
| Runtime | text, cached, live | Text/cached conformance tested; live UE not available/certified here |
| Navigation actions | waypoint, point-3D, point-2D+depth, task | Schema/tests cover contracts; courier uses indexed graph actions |
| Courier profiles | walker_novice, scooter_standard, multi_modal_courier | Profile/schema tests; current courier play is walking only |
| Maps | 11 shipped compiled/env-spec artifacts including CityCore Paris | Compiler `--all` executed; long run status reported below |
| Policies | manual, observation-only, sighted, privileged router | Manual and references executed; privileged router kept out of score table |
| Training | logprobs, masks, policy update, verl adapter, R1 gate | Unit coverage; R1 CLI blocked by missing undeclared `torch` |

This is not a claim that every Cartesian product is valid. For example, live
visual execution requires external licensed/engine assets, `visual` is known
unsolvable, and transport profiles require affordances not exercised by the
walking courier. A publishable benchmark needs a generated compatibility matrix
with an explicit denominator instead of distributing these facts across code,
ADRs, and prose.

## Automated verification

| Check | Result | Evidence |
|---|---|---|
| Full pytest suite | PASS | 950 passed, 4 skipped in 379.81 s |
| Migration check | PASS | 366 nodes, 428 edges, 86 streets, 477 addresses, 105 signalised; 856/856 directions photographed; scripted 2/2 delivered |
| Stress compiler | PASS | 15/15 cases behaved as expected, including unusable malformed cases |
| Environment readiness | **FAIL** | 6 turns had images, but `expected_images_per_turn` failed; FPV path fallback was skipped first |
| Baseline manifest | **FAIL, 43/49** | Plugin tree digest/file count drift, CityCore content path absent, Python 3.11.15 expected vs 3.12.3 observed |
| Deterministic replay | PASS | 28/28 assertions in 78.993 s; text and visual recordings, two replays each, and clean-process checks matched |
| Training R1 gate | **BLOCKED** | `ModuleNotFoundError: torch`; torch is absent from project dependencies |
| Qwen baseline CLI | **BLOCKED** | same undeclared `torch` dependency before model execution |
| Compile every shipped map | PASS | 10/10 usable environments; CityCore Paris grade B, graph navigation, 23.1% cardinal, 1,162 nodes, four flags |

## Findings

### Critical — no basis for real-Paris deployment

The benchmark uses a synthetic map and cached visual world. It has no GPS/map
alignment certificate, current street or crossing data, sidewalk accessibility,
dynamic road users, weather/night diversity, localization uncertainty, legal
compliance, fail-safe behavior, human intervention protocol, physical dynamics,
or field validation. Street adjacencies observed during play (for example Rue de
la Roquette ending into Avenue de Crimée/Rue de Flandre) should not be treated
as geographic Paris. Training on this benchmark may study agent loops; it does
not qualify an agent to move through Paris.

Acceptance gate: a separate sim-to-real protocol, georeferenced dataset,
licensing/privacy review, safety case, shadow-mode trials, supervised pedestrian
tests, incident taxonomy, and explicit no-go/hand-off behavior.

### High — validated benchmark does not measure general visual navigation

`Condition.VALIDATED == ('full',)`. Source documentation itself says `full` is
a text planning floor, `no_phone` is unvalidated, and `visual` is impossible
because door numbers are not legible. Hazard images add a real perceptual
subtask, but a benchmark whose only valid rung is designed to be text-solvable
cannot support broad embodied-perception claims.

Acceptance gate: at least one perception-essential, observation-complete rung;
human and multiple VLM solvability studies; ablations showing a significant
drop when images are removed or shuffled; adversarial leakage tests.

### High — action refusals leak privileged metric distance

Both rejected `collect()` and `hand_over()` disclosed exact 36 m distances. The
evaluation brief explicitly claims the former exact-distance rangefinder was
removed. This provides a cheap, ground-truth localization oracle unavailable in
the stated task and makes exploit policies possible.

Acceptance gate: refusal says only that the courier is not at the address (or a
coarse observationally justified category); add a regression test forbidding
numbers/metric distance in refusal text.

### High — shipped documentation cannot reproduce its reference runner

The documented `ObservationOnlyCourier(env).run(seed)` call produces no order
unless `env.reset()` was called first. The earlier environment example does call
reset, but the reference-policy snippet appears standalone and its contract is
not self-initializing. In addition, the provided commands use nonexistent
`python`, and declared dev dependencies do not install `torch` required by
training/model verification CLIs.

Acceptance gate: clean-container smoke test running every documentation block
verbatim; lock file or supported environment; optional dependency groups such
as `train` and `vlm`; actionable preflight errors.

### High — release gates disagree with current environment

Environment readiness fails, and baseline integrity is 43/49. A benchmark
release must not require readers to infer which failed gates are harmless local
drift. The missing CityCore content assertion is particularly important because
the release claim is tied to that source.

Acceptance gate: one top-level release command whose required gates all pass;
conditional checks reported as skip, never pass; pinned provenance regenerated
or restored and reviewed.

### Medium — route and movement units are confusing

Candidate distance, route metres, route “junction” counts, actual block calls,
and reported waypoints are different abstractions. Source comments acknowledge
this class of bug, but blind play still found 18 m vs 29 m and “1 junction” vs
“through 2 junctions.” The agent cannot reliably tell whether it followed the
route correctly.

Acceptance gate: define `next waypoint`, `decision stop`, `graph edge`, and
`street junction`; expose only the quantities relevant to the configured stride;
property-test route leg counts against actual calls for all origin/goal pairs.

### Medium — observations are repetitive and token-expensive

The system prompt was measured by readiness at 5,917 characters (~1,479 rough
tokens). Each turn repeats notes, seen streets, candidates, photograph captions,
map caveats, and previous outcome. The persistent map is useful, but identical
photographs and long trail text are repeatedly resent. Training cost and context
position, not only simulated time, are material benchmark resources.

Acceptance gate: publish exact tokenizer counts; evaluate delta observations or
cache semantics; cap memory by relevance; report input and output tokens and
image pixels per delivered order.

### Medium — reference claims are stale or configuration-sensitive

The docs report much stronger six-seed floor results than seed 1 produced here,
and the source docstring contains still different 25/30-seed tables. These may
refer to clean rather than hazard runs, another stride, or another revision, but
the table does not bind every number to artifact hash, seed list, policy commit,
condition, albums, and reset procedure.

Acceptance gate: machine-generated result cards with hashes and confidence
intervals; prohibit manually copied score tables.

### Medium — metric anomalies undermine interpretability

The successful solo summary reported walked distance 537.9 m versus “optimal”
562.7 m, a ratio of 0.96. An agent can beat a shortest-path label if the two
metrics use different endpoints/stride semantics, but then `optimal_walk_m` is
misnamed or incomparable. Earlier in the episode it was 0 until completion.

Acceptance gate: recompute both from the same transition distances and endpoints;
assert optimal is a lower bound or rename/document the comparator.

### Low/medium — visual interface is usable but not yet publication-grade

Street renders are consistent and hazards are visually recognizable. Some
candidate views point substantially toward facades rather than down streets,
signal lamps occupy a very small fraction of a 1200×843 frame, map-edge grey/sky
artifacts are known, and near-identical bearings depend on captions. Frame
filenames also encode `red`, `green`, `road_block`, and `slow_pedestrian`; a
harness must ensure models receive pixels, not paths or alt text containing
labels. The sighted oracle intentionally reads filenames and must remain clearly
privileged.

Acceptance gate: perception study across resolutions/models; path sanitization;
randomized neutral media IDs; camera-pose and label correspondence audit.

## Harness and security review

Positive properties:

- One typed tool registry drives both prompt and dispatch.
- Exactly one fenced action is required; three consecutive format errors stop an
  episode.
- Photographs and SVG phone maps are typed separately and ordered explicitly.
- Agent-visible and privileged state are conceptually separated.
- Every tool consumes simulated time, and trajectories retain the exact prompt,
  image ordering, reply, outcome, spend, and memory.
- Cached/text conformance and deterministic replay receive serious test coverage.

Remaining risks:

- Exact-distance refusal leakage is an active privileged-state channel.
- File paths expose semantic hazard labels to any adapter that serializes paths.
- `follow_street` appears in generic reply-format examples even when block stride
  removes it; the tool menu is correct, but examples should also be generated
  from allowed tools.
- Default budgets in `CourierSession` and task profile budgets are separate
  systems, raising the possibility of incomparable truncation.
- Live/cached parity is transition-based by design, but no live execution was
  available in this audit.

## Research quality gaps for a NeurIPS oral bar

1. State a narrow hypothesis: planning with costly multimodal information,
   perception-grounded hazard avoidance, or real-world courier autonomy are
   different papers.
2. Freeze an evaluation protocol with train/dev/test geographic separation,
   hidden seeds, policy-visible budgets, and confidence intervals.
3. Provide strong baselines: text-only, image-shuffled, caption-only, VLM,
   memory ablations, oracle perception, oracle route, and human performance.
4. Demonstrate causal necessity of pixels and resistance to filename/text leaks.
5. Quantify dataset diversity, visual duplication, order overlap, shortest-path
   distributions, hazard density, and episode feasibility.
6. Publish failures and conditional coverage with denominators.
7. Separate simulator benchmark validity from real-world deployment readiness.
8. Add ethics, labor, pedestrian safety, accessibility, privacy, licensing, and
   environmental-impact analysis.

## Prioritized next work

Before another benchmark run:

1. Remove exact-distance refusal leakage and add regression tests.
2. Fix every public run block and provide a reproducible environment command.
3. Make readiness, provenance, full tests, replay, and dependency preflight one
   mandatory release gate.
4. Generate result tables from frozen trajectory artifacts, not prose.
5. Design and validate a perception-essential condition with ablations.
6. Publish a compatibility matrix for every map/profile/runtime/condition pair.
7. Treat real-Paris deployment as a separate, safety-governed program.

## Bottom line

This benchmark has a better engineering skeleton than many embodied-agent
prototypes: typed artifacts, explicit unvalidated conditions, image/state
correspondence checks, stress tests, and replay discipline are excellent
foundations. The blind trajectory was genuinely engaging and forced visual
hazard decisions. But today it principally measures planning and recovery in a
synthetic cached city. Claims should stay at that altitude until the perception
rung, reproducibility gates, statistical evaluation, and real-world evidence
exist.
