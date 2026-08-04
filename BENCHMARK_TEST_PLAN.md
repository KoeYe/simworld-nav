# Paris Delivery Benchmark — Debug and Evaluation Plan

**Prepared before test execution:** 2026-08-02 (UTC)  
**Repository:** `/home/murray/simworld_nav`  
**Evaluator role:** embodied courier policy first; benchmark debugger second

## 1. Goals and quality bar

Evaluate the current benchmark as a complete research instrument: task inputs,
multimodal observations, action tools, agent harness, environment transitions,
scoring, configuration coverage, reproducibility, artifacts, training interfaces,
and the claims needed for eventual deployment in Paris. Passing unit tests is
necessary but not sufficient. The final assessment will distinguish:

- implemented and verified behavior;
- declared but unvalidated behavior;
- simulated evidence versus real-Paris evidence;
- benchmark defects, policy failures, and expected difficulty;
- release blockers versus research limitations.

## 2. Evidence-preserving order of work

1. **Freeze the audit context.** Record revision, working-tree state, Python and
   dependency versions, available datasets/albums, and machine-specific paths.
2. **Blind embodied play.** Read only the public runner/evaluation instructions
   and interact through `CourierSession.observe()` and `CourierSession.step()`.
   Do not inspect environment source or privileged route/state fields until the
   required gameplay is complete. Play at least one `solo` and one `pair` or
   `triple` episode with `stride="block"`; preserve every prompt, observation,
   ordered frame reference, action, response, and outcome.
3. **Input/output/tool audit.** Check whether each decision has sufficient text
   and/or image evidence; identify redundancy, token/turn waste, contradictions,
   ambiguity, hidden privileged leakage, malformed-call handling, and actions
   that cannot express a reasonable courier choice.
4. **Configuration inventory and matrix.** Enumerate all supported values from
   public APIs, schemas, CLIs, fixtures, manifests, and tests. Cover runtime
   modes, difficulty tiers, stride modes, conditions, hazard albums, seeds,
   policies, maps, task/embodiment profiles, observation modes, and applicable
   training adapters. Mark impossible or unavailable combinations explicitly
   rather than silently skipping them.
5. **Automated verification.** Run the full test suite, migration check, baseline
   verification/replay, compiler/map checks, environment readiness, stress tests,
   and determinism/conformance checks exposed by the repository. Capture exact
   commands, durations, exit codes, warnings, failures, and produced artifacts.
6. **Adversarial harness tests.** Probe parser boundaries, repeated malformed
   replies, tool argument validation, refusal information leakage, stale action
   indices, missing/duplicate frames, path handling, episode termination,
   budget accounting, replay integrity, and separation of policy-visible versus
   evaluator-only state.
7. **Source and contract review.** Only after blind play, inspect implementation
   paths for observation construction, action dispatch, transition semantics,
   reward/scoring, hashes, media, runtime adapters, VLM/model adapters, and
   training masks/log-probability plumbing. Map tests to claims and find untested
   branches or contracts.
8. **Visual validation.** Open representative photographs, maps, overlays, and
   trajectory media. Check legibility, correspondence with text/state, ordering,
   camera/view defects, obstacle/signal visibility, and whether visual information
   is actually causally required. Do not infer image quality from filenames.
9. **Research and deployment assessment.** Evaluate construct validity,
   contamination/exploit resistance, statistical coverage, oracle/floor usage,
   reproducibility, generalization, realism, safety, accessibility, privacy,
   licensing, sim-to-real gaps, and whether current evidence supports any
   real-Paris deployment claim.
10. **Reports and verification.** Produce a detailed Markdown report and a
    self-contained HTML report with visual trajectory timelines, configuration
    coverage, metrics, findings, and reproducible evidence links. Validate HTML
    structure and links, and ensure both reports agree on all headline numbers.

## 3. Minimum test matrix

The inventory may expand this matrix after source inspection.

| Axis | Planned coverage |
|---|---|
| Difficulty | `solo`, `pair`, `triple`, `shift`, `endless` |
| Stride | `block`, `waypoint` |
| Condition | `full`, `no_phone`, `visual` (with declared validation status) |
| Perception/hazards | no albums; signal only; obstacle only; both; available street FPV |
| Policy | manual/blind evaluator play; observation-only; sighted reference; privileged oracle kept separate |
| Runtime/observation | text, cached visual, and live capability/readiness where available |
| Robustness | valid, invalid, malformed, stale, repeated, and boundary tool calls |
| Reproducibility | repeated identical seeds; replay/digest/hash comparison |
| Maps/profiles | every shipped map/environment spec and compatible task/embodiment profile |
| Training | trajectory schema, masks, log-probs, policy update/R1 gates, adapter import/contract tests |

Exhaustive Cartesian execution is not assumed when combinations are semantically
invalid or require unavailable licensed/live assets. The report will give a
denominator, status, and reason for every enumerated cell.

## 4. Finding record

Each finding will contain: severity, component/configuration, reproducible setup,
exact policy-visible observation or artifact reference, action, actual result,
expected result, frequency, research impact, deployment impact, and proposed
acceptance test. Policy mistakes will be recorded separately from benchmark bugs.

## 5. Deliverables

- `BENCHMARK_EVALUATION_REPORT.md`: detailed methods, matrix, raw command results,
  trajectory evidence, findings, limitations, and prioritized recommendations.
- `BENCHMARK_EVALUATION_REPORT.html`: self-contained visual report emphasizing
  trajectory timelines, observation/action flow, coverage, and severity.
- Supporting machine-readable logs/artifacts under a dedicated audit directory,
  without modifying canonical benchmark fixtures.

## 6. Stop/go rules

- Do not claim a configuration passed unless it was executed or its static-only
  status is explicit.
- Do not use privileged environment state during blind play.
- Do not rerun away an unfavorable policy outcome.
- Do not treat missing live UE, licensed assets, or real-world trials as success;
  report them as evidence gaps.
- Preserve unrelated user changes and avoid altering canonical benchmark outputs.

