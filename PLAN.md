# Autonomous Embodied Agent Environment and Benchmark

## Detailed implementation plan, Paris and DeliveryBench first

**Status:** proposed architecture and build plan  
**Date:** 2026-07-31  
**Revision:** updated after review in `PLAN_REVIEW_FEEDBACK.md`; accepted items are integrated, and point-mode cache sizes remain pilot estimates rather than certified measurements  
**Scope:** high-level embodied agents for long-horizon navigation and task execution; manipulation and low-level motor control are future integrations  
**First vertical slice:** CityCore Paris + DeliveryBench  

---

## 1. Objective

Build a platform that plays the role for embodied agents that SWE-bench plays for coding agents:

1. A versioned, reproducible environment artifact derived from a real UE map.
2. Frozen task instances with deterministic reset state, goals, constraints, budgets, and evaluators.
3. A common agent/runtime API so one harness can run text-only, cached-image, and live-UE episodes.
4. Execution-based scoring rather than subjective judging.
5. A training path for agentic RL and an evaluation path that uses the same environment semantics.
6. Reproducible trajectories, environment hashes, evaluator versions, costs, and failure evidence.

The initial task family is DeliveryBench. The architecture must also support PointNav and later task plugins without copying the simulator or coupling the agent to DeliveryBench internals. **Multi-agent execution is out of scope for v1**: the first protocol, benchmark tracks, budgets, and scores are single-agent. DeliveryBench collaboration mechanics may remain in the source implementation, but they are not exposed as a partially supported v1 feature.

The high-level agent is deliberately embodiment-independent. It chooses task actions, goals, and navigation targets. An `EmbodimentAdapter` or expert controller translates those requests into locomotion. A future robot, avatar, vehicle, or controller model should be attachable without retraining or rewriting the high-level harness solely because its low-level control API differs.

---

## 2. Review of the existing plans

The current [IMPLEMENTATION_PLAN.md](/home/murray/IMPLEMENTATION_PLAN.md) is a useful Paris/DeliveryBench execution plan. Its strongest parts are:

- It audits the actual Paris export rather than assuming the conversion is unfinished.
- It identifies non-cardinal Paris roads, degenerate segments, false intersections, synthetic connectors, missing DeliveryBench affordances, and the absent FPV album.
- It makes `MOVE_TO(k)` the primary graph action on an irregular city.
- It treats live UE as an optional runtime and keeps it out of the high-throughput RL loop.
- It calls out observation-token loss masking and fan-out reward accounting as correctness risks.
- Its milestones have falsifiable acceptance tests.

However, it is too narrow to serve as the platform implementation plan. It goes directly from Paris fixes to a slime harness and benchmark release without first defining stable contracts between the map compiler, runtime, task, embodiment, agent, trainer, evaluator, and portal.

### 2.1 Changes required to the current plan

1. **WorldSpec cannot remain deferred.** A versioned environment contract is needed before three runtimes and task-specific map modifications are implemented. It can start small and evolve, but the boundary must exist in the first vertical slice.
2. **There are three runtime modes, not two:** `text`, `cached`, and `live`. They share transition semantics, but observations differ by declared sensor capability.
3. **Task modifications need an overlay model.** DeliveryBench must not destructively edit the base Paris map. Scooters, bus stops, chargers, stores, and other task assets should be declared in a task overlay and materialized into derived artifacts.
4. **Task, courier, and embodiment configuration are distinct.** A delivery worker's shift, equipment, battery, energy, vehicle access, and carrying capacity belong to a courier profile consumed by the Delivery task. Robot dimensions, control modes, and sensors belong to the embodiment profile.
5. **The agent boundary needs a typed protocol.** `BaseModel` is only model inference. A general agent also needs context policy, action parsing, memory, tool policy, budgets, and trajectory emission.
6. **Trainer choice should be gated, not assumed.** verl currently has an official multimodal agent-loop test with image tool responses. Public slime documentation clearly supports custom multi-turn generation, but current public evidence for end-to-end VLM training is weaker. Run a bake-off before selecting the primary adapter.
7. **Live/cached conformance must compare transitions, not pixels.** Cached and live images cannot be byte-identical. State, events, reward components, termination, and controller outcomes can be compared.
8. **A benchmark needs instance and evaluator governance.** Frozen episodes, hidden evaluator inputs, artifact hashes, versioning rules, gold/oracle checks, cost accounting, and result bundles are first-class deliverables.
9. **The portal is absent from the milestone chain.** It needs an event-stream contract and read-only replay before live control is exposed.
10. **Metric navigation needs a research track.** It should not be mixed into the critical path for DeliveryBench v1.

### 2.2 Audited starting point

The practical baseline is the remote `VAGEN` branch `dev/env_v2`, not only `/home/murray/deliverybench`:

- The local released DeliveryBench checkout has a live-UE-oriented gym wrapper whose reward is documented as a placeholder.
- `VAGEN@dev/env_v2` already contains a pure-Python DeliveryBench environment, text and vision modes, cached FPV loading, waypoint marks, `MOVE_TO`, hazards, scripted rollouts, benchmark helpers, and AgentGym HTTP/gateway servers.
- The branch already contains `maps/citycore-paris` and the Paris export.
- `/home/murray/nav_task` contains reusable navigation measures and a navmesh interface.
- `/home/murray/simworld_arena` contains a useful React/Node streaming portal and UE viewport pattern, but it is a scene-building arena rather than this benchmark.

Do not fork all four systems into a fifth copy. Extract stable interfaces and initially adapt existing implementations behind them.

---

## 3. Paris facts and immediate blockers

### 3.1 Available source material

- UE content is available through the symlink:
  `/data/koe/simworld-content-store/releases/ue58-baidu-20260705/CityCore_Paris`
- Its resolved location is:
  `/data/koe/simworld-content-store/allow-ai/environment/CityCore_Paris`
- Relevant maps include `ParisCity_FinalBlueprints.umap`, `ParisCity_FinalBlueprints_Night.umap`, and `ParisCity_Editable.umap`.
- `/data/shared/CityCore_Paris` is not mounted on this machine and must not appear in build scripts.
- The public [CityCore_Paris repository](https://github.com/XTRose29/CityCore_Paris) contains extraction, conversion, render code, reports, and render outputs, but not the licensed source asset pack.
- Our current codebase is https://github.com/ymzhang0303/VAGEN.git on the non-default `dev/env_v2` branch.

### 3.2 Existing export facts

The current report records:

- UE 5.8 source map `/Game/CityCore_Paris/Scenes/ParisCity_FinalBlueprints`.
- 3,290 actors, 59 road actors, 59 road centerlines, and 190 DeliveryBench road segments.
- 573 authored buildings; 443 in the navigable DeliveryBench subset.
- 414 generic buildings, 18 restaurants, 11 stores, and 125 traffic lights.
- A 144-node road graph reported as one connected component.
- Four component connectors. Two are approximately 36.3 m and 45.3 m and require validation because they may cross non-road space.
- `progen_world_enriched.json` currently has 568 nodes and an empty `bus_routes` list.
- No Paris cached FPV album is present in the VAGEN branch.
- Only 33 of 190 segments (17.4%) are within 6 degrees of a cardinal axis, so cardinal `MOVE` is not a valid primary action abstraction for Paris.
- The export has 145 unique endpoints: 79 true junctions, 46 degree-2 pass-through vertices, and 20 dead ends. The median turn at the pass-through vertices is 50 degrees; 35 of 46 exceed the current 6-degree `_edge_group` tolerance.
- Degenerate road segments 186 and 187 are both shorter than 5 mm.

### 3.3 Required Paris corrections

1. Filter zero/near-zero road segments, including current indices 186 and 187, before normalization.
2. Preserve authored centerline identity and merge pass-through spline vertices correctly.
3. Remove cardinal-grid assumptions from street grouping and naming.
4. Make graph-neighbor navigation the default for Paris.
5. Mark every synthetic connector with provenance and exclude unvalidated connectors from episode generation.
6. Validate topology against UE NavMesh, especially both long connectors.
7. Add or assign required Delivery task affordances: scooter spawn/parking, charger, rest area, hospital, car rental, bus stops, and bus routes as enabled by a task profile.
8. Materialize visible task assets in a derived UE layer so text state and pixels agree.
9. Bake portable RGB/depth observations with relative paths and content hashes.
10. Produce a certificate that says exactly which task and embodiment profiles Paris supports.
11. Decide how `pedestrian_light` is represented. All 125 current records are silently dropped by `import_pois` because the type is in neither its building-like nor point-like set. If signals remain render-only, state that contract explicitly and certify the separate hazard/FPV sidecar.

---

## 4. Architecture

```text
Source UE map + content pack
        |
        v
Map compiler: inspect -> extract -> normalize -> annotate -> validate
        |
        +---- immutable Base WorldBundle
        |
Task overlay compiler: requirements -> placement -> UE data layer -> validate -> bake
        |
        +---- derived EnvironmentBundle (world + overlay + sensors + certificate)
        |
        v
Runtime API -------------------------------------------------------+
  text runtime       cached RGB/RGB-D runtime       live UE runtime|
        |                         |                         |        |
        +-------------------------+-------------------------+        |
                                  v                                  |
Task plugin <-> EmbodimentAdapter <-> high-level AgentHarness       |
                                  |                                  |
                    trajectory/event protocol                       |
                         /                 \                          |
               trainer adapter          evaluator                    |
                verl or slime       benchmark instances              |
                         \                 /                          |
                          artifact/result store -> portal ------------+
```

### 4.1 Dependency rule

Dependencies point downward:

- Tasks may read `WorldBundle` and call `Runtime`, but never import UE or cached-runtime implementation code.
- The agent sees only observations, action schemas, controller capabilities, and budgets.
- The trainer sees trajectory records, tokens, masks, rewards, and metadata, not simulator objects.
- The portal consumes event and artifact APIs, not internal Python objects.
- The evaluator owns privileged state. Agent observations never expose it.

---

## 5. Contract A: World and environment artifacts

### 5.1 Base `WorldBundle`

The base artifact represents the reusable map without a particular task's spawned assets:

```text
worlds/citycore-paris/1.0.0/
  world.json
  coordinate_frames.json
  nav/
    graph.json
    navmesh_metadata.json
    components.json
  semantics/
    entities.jsonl
    ontology.json
  affordances/
    interaction_sites.jsonl
  source/
    actor_inventory.json
    extraction_report.json
    provenance.json
  previews/
    topdown.webp
  checksums.sha256
```

Required properties:

- Stable IDs independent of array ordering.
- Explicit units, axes, handedness, map origin, and transforms between UE world, bundle world, camera, and embodiment frames.
- Semantic labels with provenance: authored metadata, rule, geometric heuristic, VLM suggestion, or human override.
- Navigation edges with polylines, length, slope, width/clearance, allowed locomotion modes, surface, and synthetic provenance.
- Content hash over the relevant source map, compiler version, rules, and manual overrides.
- No absolute machine-specific paths in portable artifacts.

### 5.2 Task `OverlaySpec`

An overlay is a declarative patch, not an edited copy of the source map:

```yaml
id: deliverybench-paris
version: 1.0.0
task_plugin: delivery@1
requires:
  affordances:
    restaurant: {min: 10}
    customer_dock: {min: 100}
    scooter_spawn: {min: 4}
    scooter_charger: {min: 2}
    bus_stop: {min: 6}
  routes:
    bus_route: {min: 1}
placements:
  policy: reuse_then_spawn
  seed: 0
assets:
  scooter_spawn: /Game/TaskAssets/Delivery/BP_ScooterDock
  bus_stop: /Game/TaskAssets/Delivery/BP_BusStop
```

Overlay compilation must:

1. Reuse valid authored assets where possible.
2. Assign roles to suitable existing entities when a visible asset is not necessary.
3. Spawn or modify only through a named UE Data Layer or generated sublevel.
4. Record every placement, source asset, transform, rule, and seed.
5. Check NavMesh reachability, clearance, visibility, collision, and minimum spacing.
6. Emit an inverse/removal manifest so the source project remains unchanged.

### 5.3 Derived `EnvironmentBundle`

The derived artifact binds a base world, overlay, supported embodiment profiles, and sensor cache:

```text
environments/citycore-paris-delivery/1.0.0/
  environment.json
  base_world.ref
  overlay/
    overlay.yaml
    placements.jsonl
    ue_data_layer_manifest.json
  navigation/
    graph.json
    controller_constraints.json
  observations/
    manifest.parquet
    rgb/
    depth/
    semantic/                 # optional, privileged unless task exposes it
    intrinsics.json
  certificates/
    map.json
    delivery.json
    cached_live_conformance.json
  checksums.sha256
```

`environment.json` must declare compatibility, not imply it. Example: graph navigation may be certified for `abstract_courier` and `scooter_courier`, while metric navigation remains experimental.

### 5.4 Schema and version policy

- Semantic versions for schemas and artifacts.
- Patch: metadata or evaluator fix that preserves instance behavior.
- Minor: backward-compatible fields or new optional capabilities.
- Major: changed transitions, action meanings, coordinate frames, task semantics, or scores.
- A benchmark release pins exact environment, task, evaluator, and episode hashes.
- Existing releases are immutable. Corrections create a new release plus migration notes.

---

## 6. Map compiler and task-aware preprocessing

Use the already working Paris exporter as the first structure-aware producer. Add a generic NavMesh producer later; do not discard authored spline data.

### 6.1 Compiler passes

#### P0: source inspection

- Resolve content-store links and verify allow-AI/license metadata.
- Record UE project, engine version, plugin versions, level package, asset registry hash, and source sizes.
- Open the map in the required UE editor version.
- Dump actor/class/folder/tag/component/asset histograms.
- Detect existing NavMesh, splines, roads, sidewalks, intersections, doors, traffic controls, and task-relevant assets.

#### P1: coordinate normalization

- Define UE, bundle-world, map-render, camera, and embodiment frames.
- Unit-test round trips with known landmarks.
- Reject implicit meters/centimeters conversion.
- Record camera intrinsics and depth convention.

#### P2: navigation extraction

- Paris primary path: `BP_SplineRoad`/authored centerlines from the existing exporter.
- Universal validation/fallback: UE Recast NavMesh.
- Generate a polyline graph with true junctions, pass-through nodes, docks, free-space samples, and edge clearance.
- Preserve disconnected components in the artifact; episode generation selects certified components rather than silently deleting data.
- Treat synthetic graph links as unsafe until a NavMesh path confirms them.

#### P3: semantic and affordance extraction

- Prefer actor class, tags, folders, asset paths, and authored metadata.
- Apply pack-specific rules from versioned YAML.
- Use geometry heuristics only after engine metadata.
- Use a VLM only to propose unresolved labels. Persist proposals with confidence for review; never call that VLM during runtime.
- Generate interaction sites separately from semantic entities: door/dock, pickup counter, charger, stop, rest point, parking site.

#### P4: task overlay synthesis

- Read `TaskRequirements` from the plugin.
- Produce a deficit report before modifying anything.
- Select valid placements deterministically.
- Materialize task assets in a generated Data Layer/sublevel.
- Rebuild navigation if collision-affecting assets were added.
- Export both privileged task state and observable visual state from the same placements.

#### P5: sensor bake

- Graph mode: bake at every certified node for each outgoing-edge bearing, plus optional scan headings.
- Metric mode research data: sample poses and executable short trajectories over free space, not only graph nodes.
- Bake RGB and metric depth first; segmentation is optional privileged/debug data.
- Store camera pose, intrinsics, near/far planes, condition, asset/overlay hashes, and relative paths per row.
- Make the bake resumable and idempotent.

#### P6: validation and certification

- Schema, checksums, ID uniqueness, and relative paths.
- Graph topology and NavMesh path agreement.
- Affordance count, reachability, clearance, and visible-asset agreement.
- Random start/goal solvability and task-specific oracle completion.
- Contact sheets and targeted screenshots for every spawned asset category.
- Text/cached transition conformance and, once available, cached/live conformance.

### 6.2 Certification grades

- **A:** eligible for public benchmark evaluation; all required checks pass, no unreviewed synthetic links or placements.
- **B:** eligible for training and development; known limitations are declared.
- **C:** visualization/research only; incomplete task or runtime support.
- **Fail:** artifact cannot load or contains invalid hashes/topology.

Certificates are per `(environment version, task plugin version, embodiment profile, navigation mode, runtime)`, not a single vague map grade.

---

## 7. Contract B: common runtime API

Use Gymnasium's reset/step semantics, including the distinction between task termination and budget truncation.

```python
class EmbodiedRuntime(Protocol):
    capabilities: RuntimeCapabilities

    def reset(self, instance: EpisodeSpec) -> tuple[Observation, ResetInfo]: ...
    def step(self, action: ActionEnvelope) -> StepResult: ...
    def close(self) -> None: ...
```

Snapshot/restore is an optional extension advertised by `capabilities.supports_snapshot`, not a mandatory runtime method. In all v1 benchmark tracks it is evaluator/harness-only and never agent-visible. A future search-enabled track would require a separate protocol and leaderboard.

`StepResult` contains:

- `observation`
- `reward` and named `reward_components`
- `terminated`
- `truncated`
- structured `events`
- `action_result` with accepted/rejected/failed/controller status
- `metrics_delta`
- `privileged_state_ref` for evaluator/debug use only

### 7.1 Runtime modes

#### `text`

- Pure Python state transition engine.
- Text and structured observations only.
- Default for simulator unit tests, oracle development, task logic, and high-throughput text RL.
- Target p95 environment step latency: at most 2 ms excluding model inference.

#### `cached`

- Same transition engine as `text`.
- Resolves RGB/depth observations from the environment manifest.
- Default for VLM RL at scale when the selected navigation mode declares a finite observation pose set.
- Target p95 step latency: at most 15 ms with warm local storage; cache hit rate at least 99% in a standard rollout batch.
- The existing waypoint album supports only `nav_waypoint`. Point-based modes require a separately baked pose-lattice album; they do not become cached-compatible merely because the runtime can load images.

#### `live`

- Packaged UE or controlled PIE instance behind a service.
- Same action and event schema.
- Used for final validation, unseen continuous poses, dynamics, videos, demos, and deployment checks.
- Never required for every RL worker.

Latency acceptance must also report effective environment throughput. For a reference 400-step episode, report measured episodes/hour at 32 and 128 concurrent environments, including image decode and cache misses. The per-step p95 target alone is not an RL capacity result.

### 7.2 Runtime conformance

The following must match for a seeded action sequence:

- action acceptance and error code
- authoritative pose or graph node within declared tolerance
- simulation clock
- task state and inventory
- economy and constraint state
- event sequence
- reward components
- terminated/truncated reason

Pixels do not need to match. Instead, a render conformance suite checks camera pose, visible task assets, depth scale, basic perceptual similarity, and absence of blank frames.

### 7.3 Service transport

Keep the Python protocol authoritative. Provide in-process and gRPC/HTTP adapters with the same serialized messages. Include protocol version negotiation, health, reset timeout, step idempotency key, episode ID, monotonic step index, and structured error codes.

---

## 8. Contract C: task plugins and courier profiles

### 8.1 Task plugin

```python
class TaskPlugin(Protocol):
    id: str
    version: str

    def requirements(self, config: TaskConfig) -> TaskRequirements: ...
    def generate(self, world: WorldView, seed: int, config: TaskConfig) -> EpisodeSpec: ...
    def reset(self, episode: EpisodeSpec, runtime: RuntimeView) -> TaskState: ...
    def action_schema(self, state: TaskState) -> list[ActionSpec]: ...
    def transition(self, state: TaskState, events: list[Event]) -> TaskTransition: ...
    def evaluate(self, trajectory: Trajectory, privileged: PrivilegedState) -> ScoreReport: ...
```

Initial plugins:

- `pointnav@1`: reach a goal under path/time/collision budgets.
- `delivery@1`: orders, pickup/drop-off, deadlines, earnings, food constraints, transport, and energy. Multi-agent collaboration is deferred beyond v1.

ObjectNav should follow after the semantics pipeline is certified on at least two maps.

### 8.2 Delivery configuration composition

Replace a single large mutable config with validated composition:

```text
DeliveryTaskConfig
  order_profile
  constraint_profile
  transport_profile
  courier_profile
  reward_profile
  observation_profile
  budget_profile
```

Example courier profiles:

- `walker_novice`: walking, low carrying capacity, no scooter ownership.
- `scooter_standard`: assigned scooter, battery/charger mechanics, standard bag.
- `scooter_veteran`: higher speed tolerance or efficiency only if scientifically justified and disclosed.
- `multi_modal_courier`: scooter, bus, and rental-car access.

Every profile field that changes outcomes must be visible to the agent at reset and recorded in `EpisodeSpec`, trajectory metadata, and evaluator output. Avoid hidden character bonuses.

### 8.3 Solvability

Episode generation is followed by an oracle feasibility check. Reject instances where:

- required sites are unreachable under the selected embodiment/navigation mode;
- deadlines are impossible under the declared controller model;
- required energy/charging or transport resources do not exist;
- a bus route is internally inconsistent;
- task success depends on an unvalidated synthetic edge;
- the documented evaluation upper bound is zero, undefined, or fails its tightness checks.

---

## 9. Contract D: three high-level navigation modes

All three modes produce a `NavigationRequest`, use the same coordinate-frame library, and terminate at poses that the selected embodiment controller can execute in UE. They differ in how the high-level target is chosen and how travel distance is obtained.

### 9.1 Mode 1: `nav_waypoint`

Action:

```json
{"type":"nav_waypoint","target_node":"wp_0123"}
```

For visual policy use, expose reachable candidates through numbered Set-of-Marks and accept the mark ID. The runtime resolves it to a stable node ID before recording the action.

This is the production path for DeliveryBench v1 because it is deterministic, compatible with the existing VAGEN implementation, and uses the small per-waypoint cached album. It primarily tests long-horizon task planning and route choice over enumerated candidates.

### 9.2 Mode 2: `nav_point_3d` with model-predicted distance

The VLM chooses a normalized image point and explicitly predicts metric travel distance:

```json
{
  "type": "nav_point_3d",
  "frame": "camera",
  "target": {"u_norm": 0.61, "v_norm": 0.73, "distance_m": 6.4}
}
```

This mode tests visual grounding plus metric distance estimation. It has two declared execution variants:

- `3d_snap`: unproject using the model distance, project to traversable space, then quantize to the nearest valid pose-lattice cell. Use this forgiving variant for supervised learning and RL.
- `3d_strict`: distance is load-bearing. Reject or stop at the model-specified distance rather than repairing it along the ray. Use this variant for the metric-distance benchmark claim.

Always log `abs(distance_model_m - distance_external_m)` using the external depth/raycast as diagnostic ground truth, even when execution uses the model distance. Otherwise a model that emits a constant distance can appear competent after snapping.

### 9.3 Mode 3: `nav_point_2d_depth` with external depth

The VLM selects only where to go in the current image:

```json
{
  "type": "nav_point_2d_depth",
  "frame": "camera",
  "target": {"u_norm": 0.61, "v_norm": 0.73}
}
```

An external metric-depth provider decides how far to travel. In cached UE simulation this may be the baked UE depth channel. In live UE or a real embodiment it may be a depth camera, stereo system, LiDAR projection, or a calibrated metric-depth model. A monocular relative-depth model is insufficient unless a separately validated scale-calibration stage turns it into metric depth.

Do not read a single depth pixel. Use a robust local estimator with invalid-depth, discontinuity, sky, transparent-surface, and dynamic-object handling. On flat ground with known camera pose, a ground-plane or NavMesh raycast can provide the distance, so on Paris this mode may be close to point selection plus environment geometry rather than a distinct metric-reasoning problem. Report it accordingly.

### 9.4 Shared geometry and UE execution

For both point modes:

```text
image point + model/external distance
        -> unproject with camera intrinsics
        -> camera-to-agent-to-UE transform
        -> range and ground-hit validation
        -> project to embodiment-traversable NavMesh
        -> quantize to declared pose lattice when that track uses cached observations
        -> request executable controller trajectory
```

Point modes must declare `max_range_m`. Start the Paris pilot near 18 m, approximately its median graph-edge length, so one point action and one waypoint action have comparable displacement and step budgets. Tune only from controller and benchmark evidence.

Typed outcomes include `accepted`, `clipped`, `no_ground_hit`, `not_traversable`, `out_of_range`, `no_controller_path`, `low_depth_confidence`, and `pose_lattice_unavailable`. `no_ground_hit` is distinct from hitting a wall or other non-traversable surface.

Every accepted request stores the original image coordinates, source of distance, continuous camera/agent/UE targets, projected target, quantized target when applicable, and controller trajectory. This makes all three modes convertible to and auditable in UE.

### 9.5 Cached-album consequence: finite pose lattice

The existing waypoint album cannot support `nav_point_3d` or `nav_point_2d_depth`, because those actions produce continuous end poses. There are two valid choices:

1. Run point modes only in live UE; or
2. Make pose quantization part of the runtime-independent action semantics and build a separate pose-lattice album.

For option 2, all runtimes execute the same rule:

```text
continuous target -> traversable projection -> nearest valid pose in lattice L
```

The live runtime must also end at the quantized pose. Snapping only in cached mode is forbidden because it would create different transition systems. Snapping to the existing road-graph nodes is also too coarse; the current Paris median edge is about 18.5 m.

The initial design point to pilot is a 2.0 m spatial lattice with 8 quantized headings. The review estimates roughly 12k poses, 97k images, 4.4 GB at 640x480 JPEG, and at most about 1.4 m planar snap error from Paris road-surface area. Treat these as planning estimates, not certified facts: the real count must be generated from the certified embodiment-traversable surface, and a 1% pilot bake must measure bytes/image, render seconds/image, depth storage, and coverage before approving the full bake. The lattice resolution and headings are versioned fields in `environment.json`.

Keep two separate observation artifacts:

- waypoint album for `nav_waypoint`, baked per node/outgoing bearing;
- pose-lattice RGB-D album for both point modes.

### 9.6 Embodiment controller boundary

```python
class EmbodimentAdapter(Protocol):
    capabilities: EmbodimentCapabilities

    def reset(self, spawn: Pose, profile: EmbodimentProfile) -> ControllerState: ...
    def execute(self, request: NavigationRequest) -> ControllerResult: ...
    def observe_pose(self) -> PoseEstimate: ...
    def emergency_stop(self) -> None: ...
```

`ControllerResult` includes requested target, accepted/projected target, trajectory reference, final pose, elapsed simulation time, distance, energy, collision/violation events, and failure reason.

Future manipulation is reserved through a separate `SkillRequest` capability. Do not add fake pick/place primitives to the high-level navigation interface.

### 9.7 Research basis and recommendation

- [Waypoint Models for Instruction-guided Navigation in Continuous Environments](https://openaccess.thecvf.com/content/ICCV2021/papers/Krantz_Waypoint_Models_for_Instruction-Guided_Navigation_in_Continuous_Environments_ICCV_2021_paper.pdf) predicts relative waypoints from panoramic RGB-D and language.
- [DREAMWALKER](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DREAMWALKER_Mental_Planning_for_Continuous_Vision-Language_Navigation_ICCV_2023_paper.pdf) represents nearby candidates over angle-distance bins.
- [RoboPoint](https://arxiv.org/abs/2406.10721) shows language-conditioned image keypoint prediction and downstream navigation use.
- [NaVid](https://arxiv.org/abs/2402.15852) demonstrates direct next-step VLM navigation from video, although it intentionally omits depth.
- [SmartWay](https://arxiv.org/abs/2503.10069) combines waypoint prediction with an MLLM navigator and backtracking.
- [Beyond Waypoints](https://arxiv.org/abs/2606.07244) highlights a key failure mode: a predicted point may be unreachable, and proposes coupling the waypoint to an executable trajectory.
- [PIVOT](https://arxiv.org/abs/2402.07872) iteratively annotates spatial/action proposals for VLM selection and is the required zero-shot bridge baseline between Set-of-Marks and free point prediction.

Therefore the recommended research sequence is:

1. Ship `nav_waypoint` as the production DeliveryBench v1 path.
2. Validate point unprojection, external metric depth, UE transforms, NavMesh projection, and pose-lattice quantization with an oracle.
3. Implement `nav_point_2d_depth`; the model learns visual target selection while external depth determines range.
4. Implement `nav_point_3d/3d_snap`; train distance-aware target selection with recoverable errors.
5. Evaluate `nav_point_3d/3d_strict` and report metric-distance error.
6. Compare all modes on identical frozen episodes, controller, embodiment, maximum displacement, and budgets.

Use `/home/murray/embodied_data/molmo-motion`, including its `pointmotionbench` subtree, as an existing source to audit for pointing supervision before generating a new dataset from scratch.

---

## 10. Contract E: general embodied agent and harness

Separate model access from agent behavior:

```text
BaseModelAdapter
  generate(messages, media, sampling) -> ModelTurn

AgentHarness
  prompt/context policy
  memory and compaction policy
  tool/action schema exposure
  action parser and repair policy
  budget accounting
  episode loop
  trajectory writer
```

### 10.1 Agent inputs

- Declared observation channels and coordinate conventions.
- Task goal and rules.
- Courier profile and embodiment/controller capabilities.
- Valid action schemas, not privileged valid targets unless the benchmark track allows them.
- Previous action result and structured failure.
- Remaining step, tool, token, simulation-time, and wall-clock budgets.

### 10.2 Agent outputs

Use schema-validated JSON actions. Preserve raw model output separately. One action envelope may include an optional high-level subgoal for interpretability, but task execution must depend only on the validated action field.

### 10.3 Memory and long horizons

- Short rolling interaction window.
- Agent-owned notes or episodic memory.
- Deterministic context compaction with before/after trajectory boundaries.
- Environment state is never summarized away; each new observation is authoritative.
- Checkpoint environment snapshot, agent memory, budgets, RNG states, and conversation reference together.

Long-horizon validation must include a model, not only a scripted replay: a zero-shot VLM must sustain at least 300 steps with at least two context-compaction cycles. The per-turn context must remain below a pinned limit and cumulative processed input tokens must grow linearly rather than quadratically with steps after compaction. Cumulative output tokens are expected to grow at least linearly because every step requires an action. Verify that authoritative task/environment state immediately after each compaction is identical to the uncompacted reference.

### 10.4 Trajectory schema

Every turn records:

- benchmark/environment/task/episode/model/harness versions;
- raw observation refs and exact observation text;
- model input messages/media refs;
- prompt and sampled token IDs when available;
- raw output, parsed action, validation result;
- environment events, reward components, termination state;
- token, latency, GPU/service, and environment cost;
- loss mask with environment/tool observations excluded;
- hashes linking large images/video/snapshots in artifact storage.

---

## 11. Contract F: training adapters

The simulator emits a trainer-neutral `Trajectory`. Framework adapters convert it to framework-specific samples. This prevents the environment API from becoming a slime or verl extension point.

### 11.1 Selection gate

Run the same toy RGB-D PointNav episode and small Delivery episode through both candidates.

Minimum proof:

- supported target VLM loads for training and rollout;
- images can appear on multiple environment turns;
- sampled tokens and log probabilities are preserved without lossy text re-tokenization;
- observation/image/tool tokens are excluded from policy loss;
- one policy update completes and weight sync changes rollout behavior;
- context compaction/fan-out preserves total reward exactly once;
- at least 32 concurrent waiting environments do not cause a global rollout stall, and rollout GPU utilization remains at least 70% of the same model/server's non-blocked baseline. Report the absolute utilization as well as the relative value.

### 11.2 Current recommendation

Use **verl as the default VLM integration candidate** because its official repository includes a multimodal agent-loop test with Qwen-VL and verifies that tool observations are masked from the response training mask. Keep a **slime adapter spike** because slime's current customization API is well suited to long-running agent workflows and asynchronous rollout scaling.

The final choice is made by the spike, not by this document. Text-only training may use either adapter earlier, but it must not be presented as proof that multimodal training works.

Relevant primary documentation:

- [verl agent loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- [verl multimodal agent-loop test](https://github.com/volcengine/verl/blob/main/tests/experimental/agent_loop/test_multi_modal.py)
- [slime customization guide](https://thudm.github.io/slime/get_started/customization.html)

### 11.3 Reward design

Use two layers:

- Task score: the benchmark outcome, kept stable and sparse enough to remain meaningful.
- Training shaping: versioned and reported separately, never used for public benchmark ranking.

Delivery training reward may include potential-based geodesic progress, pickup/drop-off milestones, net earnings, time cost, expired-order cost, and safety violations. Public evaluation reports the stable score components without training-only shaping.

All reward functions require adversarial tests for loops, tool spam, accept-and-abandon, idling, item churn, invalid-action farming, and snapshot/retry duplication.

---

## 12. Benchmark design

### 12.1 SWE-bench analogy

SWE-bench pins a repository state, issue, test patch/evaluator, and reproducible execution environment. This benchmark should pin:

| SWE-bench concept | Embodied benchmark concept |
|---|---|
| repository/base commit | `EnvironmentBundle` hash |
| issue/problem statement | task instruction and observable initial state |
| test/evaluator | task evaluator plus privileged terminal state |
| Docker image | runtime image + UE build/content reference where applicable |
| patch | agent trajectory/actions |
| resolved/unresolved | success plus continuous score report |

The [official SWE-bench harness](https://github.com/SWE-bench/SWE-bench) is a useful precedent for containerized, execution-based evaluation and per-instance logs. Embodied evaluation additionally needs sensor, controller, simulation-time, and stochasticity contracts.

### 12.2 Frozen `BenchmarkInstance`

```json
{
  "instance_id": "dbam_v1_paris_000123",
  "environment": {"id":"citycore-paris-delivery","version":"1.0.0","sha256":"..."},
  "task": {"plugin":"delivery","version":"1.0.0","config_sha256":"..."},
  "embodiment": {"profile":"abstract_courier_v1","navigation":"nav_waypoint"},
  "runtime_track": "cached",
  "seed": 4711,
  "spawn": {"node":"wp_0041","facing_deg":90},
  "order_schedule_ref": "private://dbam_v1/order_schedules/paris_000123.json",
  "observable_instruction": "...",
  "private_state_ref": "private://dbam_v1/...",
  "budgets": {"steps":400,"tool_calls":80,"sim_s":7200,"output_tokens":120000},
  "safety_timeout_s": 3600,
  "evaluator": {"id":"delivery_score","version":"1.0.0","sha256":"..."}
}
```

Release actual JSONL/Parquet files and a manifest with hashes. A seed alone is not a benchmark instance. Order content is already deterministic by `(environment seed, order id)` in the current DeliveryBench implementation; benchmark generation additionally pins a finite order-id stream and `available_from_sim_time` for every order. Pool capacity limits display/acceptance but agent actions do not decide when new order IDs become available.

Wall-clock time is a reported cost plus a generous non-scoring safety timeout. A timeout caused by evaluator infrastructure is recorded separately from task failure. Scoring budgets use hardware-independent steps, tool calls, simulation time, and output tokens.

### 12.3 Splits

- `dev`: public instances and evaluator details.
- `test-episode`: unseen episodes on seen maps.
- `test-map`: unseen layouts from seen content/generator domains.
- `test-pack`: unseen content packs and visual domains.
- `live-challenge`: periodically refreshed private maps/overlays if operationally feasible.

At least one held-out map, not only Paris, is required before claiming map generalization. At least one held-out content pack is required before claiming visual-domain generalization.

### 12.4 Tracks

- Text, cached, and live `nav_waypoint`, with assisted-routing and unassisted Set-of-Marks variants.
- Cached pose-lattice and live `nav_point_2d_depth`, where external metric depth supplies range.
- Cached pose-lattice and live `nav_point_3d/3d_snap` for training and recovery evaluation.
- Cached pose-lattice and live `nav_point_3d/3d_strict` for the metric-distance claim.

Do not combine track scores into one leaderboard number. Controller, embodiment, maximum action range, pose-lattice specification, observation channels, and depth provider must be fixed within a track.

### 12.5 Metrics

Primary Delivery metric:

- **`normalized_utility_vs_upper_bound`:** achieved net task utility divided by a documented upper-bound policy's utility for that exact frozen order schedule, with defined behavior when the bound is non-positive. Do not call this an exact oracle optimum unless optimality is actually proven. The upper-bound policy may see the frozen finite order stream `0..N`; its information advantage and construction must be disclosed.

Required secondary metrics:

- success and deliveries completed;
- on-time and constraint-satisfaction rates;
- net earnings and costs;
- SPL/SoftSPL or route efficiency;
- collisions, traffic violations, rescues, invalid actions, and controller rejections;
- environment steps, tool calls, output tokens, wall time, and estimated inference cost;
- mean, median, confidence interval, worst decile, and per-map result.

PointNav uses success, SPL, SoftSPL, final geodesic distance, collisions, and cost.

### 12.6 Baselines and release checks

Required baselines:

- random valid action;
- shortest-path PointNav oracle;
- full-information Delivery oracle or documented upper-bound heuristic;
- greedy nearest feasible order;
- zero-shot text model;
- zero-shot VLM;
- supervised and RL-trained agent when available.

Before release:

- gold/oracle succeeds on every valid instance;
- do-nothing and deliberately invalid agents fail as expected;
- evaluator is idempotent;
- retrying a step cannot duplicate reward/events;
- private fields and optional snapshot/restore capabilities never enter agent observations or portal streams;
- two deliberately divergent policies observe the same `available_from_sim_time` order schedule;
- the best non-oracle baseline reaches at least 0.25 `normalized_utility_vs_upper_bound` on dev, otherwise the bound is too loose to use as the primary ranking metric;
- at least two independent machines reproduce aggregate scores within tolerance;
- complete result bundles include configs, trajectories, evaluator logs, and artifact hashes.

---

## 13. Inference, live-stream, and replay portal

Build the portal on the event protocol, not directly on UE or trainer logs.

### 13.1 Portal v1: read-only replay

- Run list with environment/task/model/harness versions and status.
- Episode timeline with observation images, text, parsed actions, errors, reward components, and task events.
- Map view with agent path, goals, POIs, bus routes, and violations.
- Side-by-side cached versus live conformance replay.
- Downloadable trajectory and score report.
- Filters for map, task config, courier, model, failure reason, and benchmark split.

### 13.2 Portal v2: live monitoring

- Server-sent events or WebSocket stream using the same `EpisodeEvent` schema.
- Current RGB/depth preview, task state, budgets, controller state, and model-generation status.
- Pause/stop controls only through authenticated orchestration APIs.
- Backpressure and sampling for image frames; never block the environment step on a browser client.

### 13.3 Portal v3: evaluation submission

- Submit a model/harness manifest, not arbitrary server code in the first release.
- Queue status and resource limits.
- Per-instance logs with private evaluator details redacted.
- Aggregate reports and reproducibility bundle.

Reuse UI and streaming patterns from `/home/murray/simworld_arena`, but create a separate domain model. Arena battle/voting abstractions do not belong in the benchmark core.

---

## 14. Proposed repository structure

Create one new integration repository after M0, with adapters to existing code:

```text
embodiedbench/
  schemas/                    # JSON Schema / Pydantic protocol models
  compiler/
    core/
    ue/
    rules/
    overlays/
    validators/
  artifacts/
    worldbundle/
    environmentbundle/
    store/
  runtime/
    core/
    text/
    cached/
    live/
    conformance/
  embodiment/
    core.py
    abstract.py
    expert_controller.py
    profiles/
  tasks/
    core.py
    pointnav/
    delivery/
  agent/
    model_adapters/
    harness/
    memory/
    trajectory/
  training/
    core.py
    verl_adapter/
    slime_adapter/
  benchmark/
    instances/
    evaluator/
    baselines/
    reports/
  services/
    runtime_api/
    orchestrator/
    artifact_api/
  portal/
  tests/
```

Initial reuse:

- Adapt VAGEN `dev/env_v2` DeliveryBench transition logic; do not immediately rewrite it.
- Adapt its cached FPV loader and rendering tools behind the cached runtime.
- Reuse `/home/murray/nav_task` measures and NavMesh client.
- Reuse the current Paris extraction/conversion scripts as a compiler producer.
- Reuse applicable streaming/viewport components from SimWorld Arena.

Add a de-vendoring milestone after the vertical slice so there is one authoritative Delivery task implementation rather than released DeliveryBench plus multiple vendored copies.

### 14.1 Inputs still needed to execute the plan

Work can start now on M0 inventory, M1 schemas/walking skeleton, R1 adapter scaffolding, and R2 synthetic geometry fixtures. The following inputs must be resolved to complete later gates:

| Input | Needed by | Current state | Default if the owner does not choose |
|---|---|---|---|
| Authoritative integration repository, owner, and branch policy | M0/M1 | not named | create `/home/murray/embodiedbench` only after approval; do not place new core code into a vendored DeliveryBench copy |
| Named UE 5.8 host, SSH/access procedure, Paris project path, and launch command | M0/M2 | no accessible local UE 5.8 editor confirmed | none; M2-M5 remain blocked |
| UE 5.8 plugin build containing SimWorld/UnrealCV and `/nav/*` | M0/M2 | unverified | none; graph certification cannot be inferred from UE 5.3 |
| UE assets/classes for scooter docks, chargers, bus stops, and other Delivery overlay objects | M3 | exact asset paths not frozen | use placeholder development assets only for grade-C artifacts, never benchmark release |
| v1 Delivery task/courier profile set | M0/M3 | examples exist, release set not frozen | minimum: `walker_novice`, `scooter_standard`, `multi_modal_courier` |
| Target VLM checkpoint, tokenizer/processor revision, trainer containers, and GPU allocation | R1 | not frozen | no trainer winner or throughput claim until pinned |
| External metric-depth provider for cached and live tracks | R2/M11 | interface decided, implementation not selected | cached: baked UE metric depth; live UE: UE metric depth; real embodiment remains a separate profile |
| Artifact store path, quota, retention, and checksum/index mechanism | M5 | not assigned | a local development directory is allowed; no full lattice bake without a quota decision |
| Content licensing and `test-pack` delivery model | M5 | open | no public `test-pack` claim |
| Reference CI/performance machines | M0/M5 | not pinned | correctness can proceed; latency/throughput milestones cannot pass |
| Additional train/test maps and access rights | M12 | not selected | release only a Paris development set, not AnyMap benchmark v1 |

Secrets, hostnames, credentials, and licensed asset paths belong in deployment configuration or a private operations document, not in portable benchmark manifests.

---

## 15. Milestones and verifiable checkpoints

Milestones are ordered by dependency and risk. Time estimates assume two engineers with reliable access to a UE 5.8 machine and training GPUs; revise after M0.

### Verification standard

Every milestone must emit `artifacts/verification/M<N>/report.json` with:

- exact input artifact, source commit, schema, model, container, plugin, and hardware hashes;
- the non-interactive command(s) used to run verification;
- machine-readable assertions with expected and observed values;
- raw log and result-artifact hashes;
- wall-clock start/end, responsible owner, and verifier;
- status `pass`, `fail`, or `blocked`; a conditional skip is not `pass` unless the milestone explicitly defines a valid scope decision;
- links to human-review checklists where visual or licensing approval is required.

Performance checks use three runs after one warm-up, report median and p95, and pin the reference-machine manifest. Human visual checks require a versioned checklist, reviewer identity, timestamp, and hashes of the reviewed contact sheets. Research engineering completion and scientific capability gates are separate: a reproducible negative result can complete the engineering milestone while failing the capability go/no-go.

### M0: baseline freeze and ownership decision (1 week)

Work:

- Pin commits for VAGEN `dev/env_v2`, DeliveryBench, `nav_task`, SimWorld/UE plugins, and Paris scripts.
- Record licenses/access requirements and which artifacts can be published.
- Run existing unit/scripted tests and archive results.
- Decide the authoritative Delivery transition implementation and repository ownership.
- Define the first benchmark claim: Paris vertical slice, not yet AnyMap generalization.
- Identify a named, accessible UE 5.8 machine and verify that its SimWorld/UnrealCV build includes the `/nav/*` `NavigationHandler` used by `/home/murray/nav_task`.

Accept when:

- `BASELINE_MANIFEST.json` contains every commit/content/plugin hash and validates in a clean checkout with one non-interactive command.
- One existing procgen text episode and one visual episode replay twice from recorded actions with identical transition, terminal-state, and score hashes.
- An approved ownership ADR names the repository, owners, branch policy, and authoritative locations for schemas, Delivery task logic, compiler, and benchmark instances.
- On the named machine, an archived command/log proves UE 5.8 opens `ParisCity_FinalBlueprints`, builds NavMesh, and `vget /nav/random_points 100` returns exactly 100 unique valid in-bounds Paris points.
- The v1 courier profile set, reference correctness/performance machines, target R1 model candidates, and minimum Paris certified-road-coverage threshold are frozen in the M0 report. Default road-coverage threshold is 90% unless an approved bounded-district ADR changes the evaluation bounds before M2.
- Dirty user changes in existing repositories remain untouched.

### M1: protocol schemas and contract test kit (1-2 weeks)

Work:

- Implement versioned schemas for WorldBundle, OverlaySpec, EnvironmentBundle, observations, actions, events, runtime results, embodiment capabilities, EpisodeSpec, trajectory, and score report.
- Add schema fixtures and compatibility tests.
- Wrap the existing VAGEN text env without changing its behavior.
- Build a walking skeleton through every layer on an existing procgen map: stub compiler artifact, text runtime, Delivery task, trivial scripted agent, trajectory writer, and evaluator.

Accept when:

- JSON Schema/Pydantic round-trip, unknown-field/version, and invalid-fixture tests pass for every contract.
- A fixed existing DeliveryBench action trace runs through the adapter three times and matches the pinned pre-adapter state/event hash.
- `terminated` and `truncated` have separate tested causes.
- Invalid versions, coordinate frames, capabilities, and action payloads fail with typed errors.
- The walking-skeleton command exits zero from a clean environment and three runs produce identical trajectory and score hashes.
- All schemas remain `v0.x` with no backward-compatibility promise before M12; early consumers pin exact schema revisions.

### M2: Paris base WorldBundle (1-2 weeks)

Work:

- Turn the existing Paris export into WorldBundle v0.1.
- Fix degenerate segments and spline pass-through grouping.
- Validate graph edges and both long synthetic connectors against UE NavMesh.
- Produce actor inventory, coordinate tests, topology report, and map certificate.

Accept when:

- No zero-length edge, non-finite coordinate, duplicate stable ID, or schema error exists.
- At least 20 landmarks distributed across all Paris-bounds quadrants have UE->world->UE round-trip error at most 1 cm.
- Every graph edge is classified `certified` or `excluded`; every certified edge has an archived NavMesh path and every excluded edge is unavailable to episode generation.
- The certified episode region covers at least the M0-pinned fraction of non-degenerate authored road length, preventing a vacuous certificate over a tiny subgraph.
- Graph components and true junction/pass-through statistics are reported.
- Long synthetic connectors are either validated with evidence or excluded from episode generation.
- WorldBundle checksums validate after copying to a different directory.

### M3: task overlay compiler for Delivery Paris (2 weeks)

Work:

- Define Delivery `TaskRequirements` and courier profiles.
- Generate a Paris deficit report.
- Enrich roles/routes and materialize required scooter, charger, bus-stop, and other enabled assets in a generated UE Data Layer.
- Add placement validation and removal manifest.

Accept when:

- The M0-frozen v1 profile list is non-empty and includes at least `walker_novice`, `scooter_standard`, and `multi_modal_courier`; every profile's exact versioned affordance requirements are satisfied.
- 100% of interaction sites are reachable for their profile and pass versioned collision/clearance assertions.
- At least one valid bus route has ordered reachable stops and consistent travel times.
- A signed checklist reviews every spawned asset category and every placement at least once through generated contact sheets; zero unresolved visibility, floating, collision, or orientation defects remain.
- The source `.umap` and actor manifest hashes are unchanged; activating then deactivating the generated Data Layer returns the active actor manifest to its original hash.
- The Delivery task certificate is at least grade B.

### M4: Paris text runtime vertical slice (1-2 weeks)

Work:

- Load EnvironmentBundle through the common runtime API.
- Make graph/Set-of-Marks movement primary.
- Implement PointNav and Delivery plugin adapters.
- Make courier profiles observable and outcome-affecting.
- Implement deterministic EpisodeSpec generation and oracle feasibility filtering.

Accept when:

- A PointNav oracle succeeds on 200/200 certified Paris instances.
- A scripted Delivery oracle completes at least three deliveries in each M0-frozen courier profile on a pinned set of seeds.
- Same `(environment, task config, seed)` produces byte-identical EpisodeSpec in two clean processes.
- Two deliberately divergent scripted agents observe the same pre-materialized order `available_from_sim_time` schedule.
- 1,000 seeded resets and 100,000 scripted steps complete with zero crash, non-finite value, schema failure, impossible inventory transition, negative elapsed time, or duplicate reward/event ID.
- Existing `medium-city-22` golden trajectory remains unchanged or has an explicitly reviewed version migration.

### M5: cached RGB-D runtime and Paris bakes (2-3 weeks plus render time)

Work:

- Bake the `nav_waypoint` RGB-D album per certified node/outgoing bearing.
- Run a 1% pose-lattice pilot, then conditionally bake the point-mode RGB-D album using the approved lattice resolution and headings.
- Use relative paths, content hashes, resumable jobs, and portable compression.
- Add Set-of-Marks overlays from graph candidates.
- Implement cache indexing, preload/LRU, and observation manifests.
- Decide by this milestone whether `test-pack` uses redistributable assets, hosted private evaluation, or metadata/hash plus an external mount procedure.

Accept when:

- The waypoint manifest row count exactly equals the expected key set derived from the certified graph and requested outgoing bearings, and 100% have `status=ok`.
- The required 1% pose-lattice pilot covers every certified surface stratum. It produces a go/live-only decision before a full bake; cached point mode passes the gate only if projected full storage fits the approved quota, certified-surface coverage is at least 95%, and maximum planar quantization error is at most 1.5 m.
- If cached point mode passes that gate, its manifest row count exactly equals the certified pose-lattice key set times declared headings and 100% have `status=ok`.
- A deterministic sample of `max(100, 1% of rows)` re-renders with camera-position error at most 5 cm, yaw error at most 0.5 degrees, and valid-pixel median metric-depth relative error at most 1%.
- No manifest path is absolute.
- Contact sheets covering every manifest row pass the signed review checklist with zero unresolved blank-frame, clipping, marker mismatch, missing-task-asset, or orientation defect.
- Cached runtime p95 step latency is at most 15 ms on the reference machine with warm storage.
- Measured 400-step episodes/hour is reported at 1, 32, and 128 concurrent environments; aggregate environment throughput must be at least 2x the rollout demand measured by R1 on the pinned training configuration.
- Text and cached runtimes produce identical action-result, transition, event, reward, and terminal-state hashes for 100 seeded `nav_waypoint` traces and, if enabled, 100 point-mode pose-lattice traces.

### M6: general AgentHarness and replay artifacts (2 weeks)

Work:

- Implement model adapter, context policy, schema action parser, repair limits, budgets, memory, checkpoint/resume, and trajectory writer.
- Add a simple OpenAI-compatible inference adapter for baseline evaluation.
- Implement result bundle and deterministic replay CLI.

Accept when:

- The same harness runs text and cached Paris without backend-specific branches in the episode loop.
- Every invalid action receives typed feedback on the next turn.
- Observation/tool/image tokens are absent from the policy response mask for every turn in the fixture trajectory; all sampled response tokens are included exactly once.
- Text and cached runtimes advertise `supports_snapshot`; interruptions at three pinned steps in a 500-step scripted episode restore to the same subsequent transition sequence, terminal state, and score hash as uninterrupted execution. Live UE may explicitly report unsupported.
- A pinned zero-shot VLM sustains at least 300 steps and two context-compaction cycles without context overflow or harness failure; per-turn context stays below `max_context_tokens`, cumulative processed input-token growth is O(steps), and authoritative state hashes match the uncompacted reference at each boundary.
- A result bundle replays without access to the original run directory.

### M7: benchmark dev set and evaluator v1 (2 weeks)

Work:

- Freeze public Paris PointNav and Delivery dev instances.
- Implement score reports, oracle/greedy/random/do-nothing baselines, and evaluator isolation.
- Add assisted and unassisted graph tracks.

Accept when:

- Instance manifest pins all environment/task/evaluator hashes.
- The solvability oracle passes every instance; invalid and do-nothing controls have zero task success and no positive net task utility.
- Evaluating the same trajectory twice returns identical reports.
- Private state leak tests pass.
- Per-instance logs and 10,000-resample, fixed-seed bootstrap confidence intervals are generated.
- Two pinned machines reproduce exact discrete metrics and per-instance float metrics within `1e-6` absolute error and aggregate metrics within `1e-4`.

### M8/R1: trainer bake-off and primary adapter (starts week 1; 2-3 weeks)

Work:

- Implement minimal verl and slime trajectory adapters against the existing VAGEN environment and an existing procgen FPV album; do not wait for Paris or the new compiler.
- Run the selection-gate experiments in Section 11.
- Select and document the primary VLM trainer; keep the other adapter experimental if useful.

Accept when:

- One text and one RGB-D policy update complete end-to-end on the same small environment using model, processor, trainer-container, and hardware hashes frozen in the R1 report.
- Loss-mask, log-prob, tokenization, reward-once, and weight-sync tests pass.
- Throughput is measured at 1 and 32 concurrent environments; 128 is required only when the frozen allocation can host it and otherwise has an explicit `not_capacity_eligible` result rather than a silent skip.
- With 32 environments, there is no global rollout stall and rollout GPU utilization is at least 70% of the non-blocked baseline. Weight synchronization produces a changed checkpoint hash, nonzero parameter delta, and changed pinned-prompt logits.
- An approved architecture decision record selects the primary adapter from a fixed correctness, throughput, operational-complexity, and supported-model scorecard.

### M9: Delivery RL v1 (3-5 weeks of engineering and experiments)

Work:

- Add curriculum sampling over task complexity, courier profiles, seeds, and maps.
- Implement versioned training shaping and reward-hacking tests.
- Run SFT warm start if required, then RL on text and cached tracks.
- Freeze `M9_RUN_PROTOCOL.json` before training with instance hashes, seeds, model/checkpoint, algorithm thresholds, restart tolerance, baseline tests, and statistical procedure.

Accept when:

- A minimum 100-update smoke run completes with zero NaN/Inf, unrecovered OOM, corrupt checkpoint, lost reward, or environment deadlock; restart from the midpoint checkpoint reproduces the next pinned evaluation within the tolerance frozen in `M9_RUN_PROTOCOL.json`.
- On a held-out instance set frozen before training, the trained policy beats random and do-nothing baselines with a paired 95% confidence interval for the utility difference strictly above zero.
- Scientific success requires the paired 95% confidence interval against greedy feasible-order to be strictly above zero. Otherwise M9 engineering is complete with a negative result and the capability gate fails.
- Train/eval environment transition code path is identical; only instance source and shaping differ.
- Ablations report text vs cached, assisted vs unassisted, and shaping components.

### M10: live UE service and conformance (2-4 weeks)

Work:

- Package or reliably orchestrate UE 5.8 plus required plugins/content.
- Implement runtime service, health, reset, optional evaluator-only snapshot where feasible, watchdog, and artifact capture.
- Execute all declared navigation actions through the expert controller and the shared coordinate/quantization semantics.

Accept when:

- 100 consecutive reset/episode-close cycles have no leaked process or port.
- Cached/live transition conformance passes on Paris and one procgen map for 100 seeded waypoint traces: task/event/discrete state is exact, authoritative pose differs by at most 5 cm and 0.5 degrees, simulation time by at most one runtime tick, and configured floats by at most `1e-4`.
- Camera/depth/task-asset render conformance passes the M5 camera/depth tolerances and signed visual checklist.
- Three forced UE crashes each produce one structured infrastructure-failure episode, preserve other workers, and allow a healthy reset within the pinned watchdog timeout.
- One full Delivery instance completes through the public Runtime API.
- `nav_waypoint` passes live conformance before initial M10 completion. A point mode cannot be declared live-supported or included in M12 until 100 traces for that mode reach the same quantized poses in text, cached pose-lattice, and live runtimes under the numeric M10 tolerances above.

### M11/R2: point navigation research prototype (geometry starts week 1; model work after M5)

Work:

- Build oracle RGB-D unprojection, external-depth lookup, coordinate transforms, NavMesh/trajectory projection, range cap, and pose-lattice quantization.
- Audit `embodied_data/molmo-motion/pointmotionbench`, then generate only the additional supervised short-horizon point/trajectory data that is missing.
- Train/evaluate a VLM waypoint predictor and recovery behavior.
- Add `nav_point_2d_depth`, `nav_point_3d/3d_snap`, and `nav_point_3d/3d_strict` experimental tracks plus a PIVOT baseline.

Accept when:

- At least 1,000 calibrated fixtures covering image quadrants, valid depth range, slopes, no-ground hits, and frame transforms have pixel/depth unprojection error at most 5 cm.
- Every accepted point request records and round-trips its camera, agent, and UE coordinates; `nav_point_3d` reports absolute model-versus-external distance error.
- At least 99% of 1,000 valid oracle proposals return an executable trajectory; invalid fixtures return their expected typed rejection with 100% accuracy.
- A held-out-area set and `M11_EVAL_PROTOCOL.json` are hashed before model training. The model report includes proposal validity, navigation success, SPL, collision rate, rejection recovery, and distance MAE with fixed confidence intervals.
- Comparison includes graph Set-of-Marks, PIVOT proposals, external-depth 2D points, `3d_snap`, `3d_strict`, and oracle targets.
- Point modes do not alter `nav_waypoint` results or the shared API schemas.

### M12: portal and multi-map benchmark release (3-6 weeks)

Work:

- Implement replay then live monitoring.
- Compile at least two additional training maps and two held-out evaluation maps, including one unseen content pack.
- Freeze benchmark v1 splits and publish permitted artifacts, harness images, baselines, and result schema.

Accept when:

- Portal replays 100% of the frozen v1 result bundles from event artifacts with matching terminal score hashes. With zero and five connected viewers, live episode step p95 differs by less than 5% and no event blocks the runtime loop.
- The release contains at least two additional certified training maps and two held-out certified evaluation maps, including one unseen content pack; every released instance is oracle-solvable and pins a grade-A certificate.
- All released artifacts pass checksum validation and the documented smoke evaluation on two clean reference machines.
- Baselines include per-map results, confidence intervals, worst decile, token/wall-time cost, and failure taxonomy.
- Benchmark versioning, submission, disclosure, and correction policies are documented.

### Verification audit summary

| Milestone | Decisive evidence | Verifiability status |
|---|---|---|
| M0 | clean manifest validation, deterministic replays, archived UE 5.8 `/nav/*` probe, approved ADR | verifiable; currently externally blocked on named UE 5.8 access and ownership inputs |
| M1 | clean one-command walking skeleton repeated three times with identical hashes | verifiable with current workspace once repository ownership is chosen |
| M2 | exhaustive edge classification, NavMesh evidence, non-vacuous road-coverage threshold | verifiable; blocked by M0 UE gate |
| M3 | frozen non-empty profile set, 100% affordance/reachability checks, source hashes, signed visual review | verifiable; needs final UE asset paths |
| M4 | fixed oracle seeds, deterministic EpisodeSpec/order schedule, 100k-step invariant suite | verifiable |
| M5 | exact expected cache keys, quantitative rerender checks, required lattice pilot, capacity gate | verifiable; full point bake may validly resolve to `live-only` if its predeclared gate fails |
| M6 | exhaustive loss masks, deterministic snapshot replay, bounded-context 300-step model run | verifiable; needs pinned baseline VLM/API access |
| M7 | frozen manifests, control-agent outcomes, fixed bootstrap, numeric cross-machine tolerance | verifiable; needs second reference machine |
| R1/M8 | pinned model/container/hardware, policy-update correctness, utilization threshold, approved scorecard | verifiable; needs model and GPU allocation |
| M9 | pre-frozen run protocol, 100-update health assertions, paired confidence intervals | verifiable; engineering completion is distinct from scientific-success gate |
| M10 | lifecycle/crash tests and numeric state/pose/render conformance | verifiable; blocked by packaged/live UE service |
| R2/M11 | at least 1,000 geometry/oracle fixtures, pre-frozen held-out protocol, full baseline report | verifiable; model quality may be a negative result without hiding it |
| M12 | exact map counts, grade-A instance certificates, 100% replay hashes, two clean machines, portal overhead threshold | verifiable; blocked on licenses and additional maps |

---

## 16. Infra and research tracks

The project-killing research questions do not depend on the Paris compiler and start immediately:

```text
Infra:     M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 -> M9
             \                         \-> M10

Research:  R1 trainer bake-off on existing VAGEN env -> week-3 VLM-RL gate
           R2 point geometry/UE-coordinate fixtures -> pose-lattice pilot -> M11 model work

Merge:     M9 requires both M7 and a passing R1 trainer gate
Release:   M7 + M10 + M11 evidence + additional certified maps -> M12
```

- Portal replay can begin from M1 walking-skeleton event fixtures and mature during M6.
- R1 uses an existing procgen map and FPV album, so Paris access cannot block trainer correctness.
- R2 begins with synthetic calibrated geometry fixtures; full cached point-mode work waits for the M5 pose-lattice pilot.
- Additional map compiler rules can begin after Paris produces the first grade-B certificate.

Do not parallelize by duplicating task transition logic or schema definitions in separate repositories.

---

## 17. Risks and explicit go/no-go gates

| Risk | Consequence | Gate or mitigation |
|---|---|---|
| Paris export topology is visually plausible but not traversable | invalid episodes and misleading cached observations | M2 NavMesh evidence for every certified edge |
| Task assets exist in metadata but not pixels | VLM cannot perceive constraints | M3 placement screenshots and M5 bake review |
| Cached album causes map memorization | high score without general navigation | multi-map/test-pack splits, condition/camera variation, live validation |
| VLM trainer cannot preserve multimodal multi-turn masks/logprobs | invalid RL gradients | R1 bake-off and go/no-go by end of week 3 |
| Runtime implementations drift | benchmark depends on backend | transition/event/reward conformance suite |
| Config explosion creates unverified combinations | impossible or incomparable tasks | typed profile composition plus oracle certification per combination |
| Hidden courier attributes affect outcomes | unfair and uninterpretable instances | surface all outcome-relevant fields in reset observation and score metadata |
| Free-form coordinates are unreachable | undefined control and reward | short horizon, projection, executable trajectory, typed rejection |
| Point navigation has no finite cached observation set | cached/live semantic drift or uncontrolled storage growth | runtime-independent pose lattice, 1% pilot, separate point-mode album |
| Synthetic connectors leak into eval | agent traverses nonexistent roads | exclude unless NavMesh and visual checks pass |
| Benchmark overfits to Paris | no evidence of generality | restrict early claims; require M12 test-map/test-pack for v1 claim |
| Licensed UE content cannot be redistributed | unreproducible public benchmark | publish hashes/compiler/metadata where allowed; provide controlled runtime images or access procedure |
| Portal leaks privileged evaluator state | invalid public results | separate agent, evaluator, and public event projections with leak tests |

Go/no-go decisions:

- After M0: if no named UE 5.8 machine can open Paris and expose the required `/nav/*` API, mark Paris compiler M2-M5 blocked and continue only the procgen research/walking-skeleton tracks.
- After M2: if Paris cannot produce a certified connected task region, repair the source/overlay or select a bounded district; do not fabricate long links.
- After week 3: if neither trainer passes multimodal correctness, continue text research only and treat VLM RL as blocked.
- After M5: if cached observations cannot visually ground task assets, fix placement/bake before VLM training.
- After M9: if RL does not beat greedy, improve task signal/harness or report the negative result; do not tune the benchmark evaluator around the trained policy.
- Before M12: if held-out maps are unavailable, release a Paris development environment, not a general benchmark.

---

## 18. First four weeks

### Week 1

- Complete M0 manifests and ownership ADR.
- Pin the VAGEN branch and Paris exporter commits.
- Run existing DeliveryBench/VAGEN tests and two replay fixtures.
- Start R1 verl/slime multimodal bake-off on an existing procgen map and FPV album.
- Start R2 synthetic camera-unprojection and camera-to-agent-to-UE coordinate fixtures.
- Identify and exercise the named UE 5.8 + Paris + `/nav/*` machine.
- Agree on names and serialized shapes for the six core contracts.

### Week 2

- Implement M1 schemas and VAGEN text adapter.
- Complete the end-to-end walking skeleton.
- Add golden transition fixture and version negotiation.
- Start Paris graph diagnostic tool for degenerate edges, centerline grouping, and connector validation.
- Continue R1 through one multimodal policy update and weight-sync check.

### Week 3

- Produce Paris WorldBundle v0.1 and coordinate/topology reports.
- Run NavMesh validation in UE 5.8.
- Draft Delivery requirements and courier profiles from existing mechanics.
- Make the R1 primary-trainer go/no-go decision and publish its correctness/throughput report.
- Validate R2 external-depth lookup, range cap, rejection codes, and pose quantization on fixtures.

### Week 4

- Produce overlay deficit and placement plan.
- Materialize the smallest Stage-1 Delivery overlay.
- Run PointNav oracle and one end-to-end text Delivery oracle on Paris.
- Audit Molmo Motion/PointMotionBench supervision and define only the missing point-navigation dataset.
- Review M2/M3 evidence before committing render and training capacity.

The first month should end with a reproducible text-only Paris environment artifact and an evidence-backed decision about what must be added to the map. It should not end with a large RL run.

---

## 19. Definition of project success

The project is successful when an external team can:

1. Obtain or mount a permitted UE map and run the compiler to create a hashed, certified environment bundle.
2. Add a task through requirements, overlays, transitions, and an evaluator without modifying runtime internals.
3. Attach a high-level VLM agent to waypoint, external-depth 2D-point, or model-distance 3D-point navigation through the same harness.
4. Run the same frozen instance in text, cached, or live mode with declared capabilities and conformant task outcomes.
5. Train through a supported RL adapter without contaminating the policy loss with environment observations.
6. Evaluate in a pinned harness and obtain a reproducible result bundle.
7. Inspect the full run in the replay/live portal.
8. Compare agents on held-out maps and content packs with cost, robustness, and failure evidence, not only a single average score.

Paris plus DeliveryBench is the first proof of this system. It is not, by itself, proof of an AnyMap benchmark.

1. /data/koe/UnrealEngine-5.8.0-preview-1
2. /data/koe/SimWorld_SPEAR
3. /data/koe/simworld-content-store/releases/ue58-citynav-20260610 

分别是UE源码，SimWorld代码，和Assets的位置