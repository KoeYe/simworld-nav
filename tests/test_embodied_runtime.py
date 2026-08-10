"""Track B: the courier whose legs are a UE pawn.

The claim under test is UE ownership of locomotion and locomotion time
(spec 3b), asserted against a kinematic reference service whose tick counts
are exact by construction:

1. **the bytes** -- the Track B messages reproduce the ``track_b_*`` golden
   fixtures byte for byte, and ``RenderResult`` gained its ``pose`` without
   moving a single Track A byte;
2. **the client and the lease** -- the four stateful endpoints over real
   HTTP, and the pool lease that takes an instance out of /render dispatch
   for exactly the episode's duration;
3. **the env** -- on the real Paris graph: the node trajectory is the stock
   GRAPH's, the clock is the integrator's ``ticks * fixed_dt`` plus the
   declared non-movement costs and nothing else, a wall maps to the stock
   refusal semantics with an engine-measured charge, and the ``embodied``
   summary block carries the I/O evidence the spec demands;
4. **the adapter** -- the training contract (obs shape, info keys, success)
   unchanged over the embodied backend, hazards refused, the lease returned
   on close.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import build_road_network
from embodiedbench.runtime.city.courier_env import (
    REJECTED_ACTION_SECONDS,
    CourierEnv,
)
from embodiedbench.runtime.live.client import (
    BadRequestError,
    ServiceBusy,
    UERenderClient,
)
from embodiedbench.runtime.live.embodied_env import EmbodiedCourierEnv
from embodiedbench.runtime.live.gym_adapter import EmbodiedCourierGymEnv
from embodiedbench.runtime.live.pool import NoHealthyInstance, RenderPool
from embodiedbench.runtime.live.protocol import (
    CameraSpec,
    EpisodeEndRequest,
    EpisodeRequest,
    EpisodeResponse,
    ObserveRequest,
    Pose,
    ProtocolViolation,
    RenderResult,
    AgentSpec,
    WalkRequest,
    WalkResponse,
    dumps,
)

from live_stub import FakeTrackBService, write_endpoints

GOLDEN = Path(__file__).resolve().parent / "golden" / "nav_render_v0"
MAPS = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
        / "envs" / "deliverybench" / "maps")
PARIS = MAPS / "citycore-paris"
needs_maps = pytest.mark.skipif(not PARIS.exists(), reason="vendored maps not present")

EPISODE = "courier-citycore-paris-s5-embodied"
CAMERA = CameraSpec(640, 480, 90.0)
FIXED_DT = FakeTrackBService.FIXED_DT


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS, map_name="citycore-paris")


@pytest.fixture()
def service(tmp_path):
    stub = FakeTrackBService(tmp_path / "svc").start()
    yield stub
    stub.stop()


def embodied_env(paris, client, tmp_path, *, seed=5, **kwargs):
    env = EmbodiedCourierEnv(paris, client,
                             episode_id=EPISODE,
                             cache_root=tmp_path / "cache",
                             seed=seed, **kwargs)
    env.reset()
    return env


def stand_at(env, node_id):
    """Move the graph position directly, the way the reference tests do. The
    pawn deliberately stays where it is: the divergence is the scenario."""
    env.node_id = node_id
    env.arrived_from = None


def expected_ticks(start_xy, target_xy, arrive_cm, speed_cm_s=140.0):
    """The integrator's own arithmetic: ticks to bring the remaining distance
    within ``arrive_cm`` at ``speed * dt`` per tick, never overshooting."""
    distance = math.dist(start_xy, target_xy)
    if distance <= arrive_cm:
        return 0
    return math.ceil((distance - arrive_cm) / (speed_cm_s * FIXED_DT))


# ─────────────────────────────────────────────────────────────────────────────


class TestTheTrackBGoldenBytes:
    """The same compatibility contract as Track A: the fixture files are the
    protocol, shared byte for byte with the SimWorld2 branch."""

    @pytest.mark.parametrize("name, cls", [
        ("track_b_episode_request.json", EpisodeRequest),
        ("track_b_episode_response.json", EpisodeResponse),
        ("track_b_walk_request.json", WalkRequest),
        ("track_b_walk_response.json", WalkResponse),
        ("track_b_observe_request.json", ObserveRequest),
        ("track_b_observe_result.json", RenderResult),
        ("track_b_episode_end_request.json", EpisodeEndRequest),
    ])
    def test_each_track_b_message_round_trips_byte_exactly(self, name, cls):
        golden = (GOLDEN / name).read_text()
        message = cls.from_dict(json.loads(golden))
        assert dumps(message.to_dict()) == golden

    def test_the_walk_response_fixture_is_kinematically_coherent(self):
        """The exemplar's numbers must be the reference integrator's own, or
        the fixture teaches the wrong arithmetic to whoever implements the
        service side from it."""
        request = WalkRequest.from_dict(
            json.loads((GOLDEN / "track_b_walk_request.json").read_text()))
        response = WalkResponse.from_dict(
            json.loads((GOLDEN / "track_b_walk_response.json").read_text()))
        spawn = EpisodeRequest.from_dict(
            json.loads((GOLDEN / "track_b_episode_request.json").read_text())).spawn
        ticks = expected_ticks((spawn.x_cm, spawn.y_cm),
                               (request.target_x_cm, request.target_y_cm),
                               request.arrive_cm)
        assert response.ticks == ticks
        assert response.sim_seconds == pytest.approx(ticks * FIXED_DT, abs=1e-4)

    def test_a_result_without_a_pose_serialises_without_the_key(self):
        """``pose`` postdates the Track A fixtures; omitted-when-absent is
        what lets both repos gain it with zero golden bytes moving."""
        result = RenderResult(key="n0/toward_n1", status="ok",
                              path="/tmp/frame.png", sha256="00", width=64,
                              height=48)
        assert "pose" not in result.to_dict()

    def test_the_track_a_response_fixture_still_round_trips_unchanged(self):
        """The direct proof that the pose field cannot move the old bytes:
        the Track A response fixture parses to pose-less results and
        re-serialises to its own bytes."""
        golden = (GOLDEN / "render_response.json").read_text()
        parsed = json.loads(golden)
        results = [RenderResult.from_dict(r) for r in parsed["results"]]
        assert all(r.pose is None for r in results)
        assert dumps({"results": [r.to_dict() for r in results]}) == golden

    def test_a_pose_round_trips_through_a_result(self):
        result = RenderResult(
            key="observe/000001", status="ok", path="/tmp/f.png", sha256="00",
            width=640, height=480,
            pose=Pose(x_cm=1.5, y_cm=-2.5, z_cm=100.0, yaw_deg=90.0))
        assert RenderResult.from_dict(result.to_dict()) == result

    def test_a_wrong_protocol_string_is_refused(self):
        data = json.loads((GOLDEN / "track_b_episode_request.json").read_text())
        data["protocol"] = "nav-render/v1"
        with pytest.raises(ProtocolViolation):
            EpisodeRequest.from_dict(data)


def episode_request(episode_id=EPISODE, spawn=Pose(0.0, 0.0, 100.0, 0.0)):
    return EpisodeRequest(
        episode_id=episode_id, map_name="citycore-paris",
        agent=AgentSpec(speed_cm_s=140.0, eye_z_cm=160.0, camera=CAMERA),
        spawn=spawn)


class TestTheClientSpeaksTrackB:
    def test_episode_spawns_the_agent_and_reports_the_fixed_dt(self, service):
        client = UERenderClient(service.base_url)
        response = client.episode(episode_request(
            spawn=Pose(10.0, 20.0, 100.0, 45.0)))
        assert response.episode_id == EPISODE
        assert response.fixed_dt == pytest.approx(FIXED_DT)
        assert (response.pose.x_cm, response.pose.y_cm) == (10.0, 20.0)
        assert service.episodes[0]["agent"]["speed_cm_s"] == 140.0

    def test_a_walk_arrives_with_exact_tick_counts(self, service):
        """The integrator is deterministic on purpose: 950 cm of travel at
        4.662 cm per tick is 204 ticks, and ``sim_seconds`` is that times the
        fixed dt -- the arithmetic every clock assertion downstream uses."""
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        walk = client.walk(WalkRequest(
            episode_id=EPISODE, target_x_cm=1000.0, target_y_cm=0.0,
            arrive_cm=50.0))
        ticks = expected_ticks((0.0, 0.0), (1000.0, 0.0), 50.0)
        assert ticks == 204  # the worked example, pinned
        assert walk.arrived and not walk.stuck and not walk.timeout
        assert walk.ticks == ticks
        assert walk.sim_seconds == pytest.approx(ticks * FIXED_DT)
        assert walk.walked_cm == pytest.approx(ticks * 140.0 * FIXED_DT)
        assert math.dist((walk.pose.x_cm, walk.pose.y_cm),
                         (1000.0, 0.0)) <= 50.0

    def test_observe_returns_a_frame_with_the_pose_echo(self, service):
        from PIL import Image

        client = UERenderClient(service.base_url)
        client.episode(episode_request(spawn=Pose(5.0, 6.0, 100.0, 0.0)))
        result = client.observe(ObserveRequest(
            episode_id=EPISODE, camera=CAMERA, yaw_deg=90.0))
        assert result.ok and result.key.startswith("observe/")
        assert result.pose is not None
        assert (result.pose.x_cm, result.pose.y_cm) == (5.0, 6.0)
        assert result.pose.yaw_deg == 90.0
        with Image.open(result.path) as image:
            assert image.size == (640, 480)

    def test_episode_end_despawns_and_answers_ok(self, service):
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        assert client.episode_end(EpisodeEndRequest(episode_id=EPISODE))
        assert service.agent is None
        assert service.episode_ends == [EPISODE]

    def test_a_busy_instance_raises_service_busy_not_a_strike_signal(self, service):
        """A stateful endpoint's busy means "another episode holds this
        instance" -- the same transient taxonomy as a saturated render, and
        just as much not a health verdict."""
        client = UERenderClient(service.base_url)
        client.episode(episode_request())
        service.busy_track_b = True
        with pytest.raises(ServiceBusy):
            client.walk(WalkRequest(episode_id=EPISODE,
                                    target_x_cm=100.0, target_y_cm=0.0))

    def test_a_walk_for_an_unknown_episode_is_the_callers_bug(self, service):
        client = UERenderClient(service.base_url)
        with pytest.raises(BadRequestError):
            client.walk(WalkRequest(episode_id="never-spawned",
                                    target_x_cm=0.0, target_y_cm=0.0))


class TestTheEmbodiedLease:
    def batch(self, key="n0/toward_n1"):
        from embodiedbench.runtime.live.protocol import RenderBatch, RenderItem

        return RenderBatch(
            episode_id="ep-render", return_mode="path",
            camera=CameraSpec(64, 48, 90.0),
            requests=(RenderItem(key=key, x_cm=0.0, y_cm=0.0, z_cm=160.0,
                                 yaw_deg=0.0, render_kind="street_view"),))

    def test_a_leased_instance_is_skipped_by_render_dispatch(self, tmp_path):
        first = FakeTrackBService(tmp_path / "a", instance_id="ue-a").start()
        second = FakeTrackBService(tmp_path / "b", instance_id="ue-b").start()
        try:
            pool = RenderPool(write_endpoints(tmp_path / "endpoints.json",
                                              [first, second]))
            with pool.lease_embodied("ep-emb") as client:
                leased = next(m for m in pool.members
                              if m.client is client)
                assert leased.leased_to == "ep-emb"
                for index in range(4):
                    pool.render(self.batch(f"n0/toward_{index}"))
                spare = first if leased.id == "ue-b" else second
                busy = first if spare is second else second
                assert len(spare.batches) == 4, (
                    "every batch must land on the unleased instance")
                assert len(busy.batches) == 0
            # Released: dispatch spreads over both again.
            assert all(m.leased_to is None for m in pool.members)
            for index in range(2):
                pool.render(self.batch(f"n1/toward_{index}"))
            assert len(first.batches) + len(second.batches) == 6
            assert min(len(first.batches), len(second.batches)) >= 1
        finally:
            first.stop()
            second.stop()

    def test_a_fully_leased_fleet_refuses_renders_rather_than_sharing(
            self, tmp_path, service):
        """Exclusivity is the point of the lease: an embodied episode's
        instance must not also photograph other episodes' streets, because
        the stateful scene (the pawn) would be in them."""
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        with pool.lease_embodied("ep-emb"):
            with pytest.raises(NoHealthyInstance):
                pool.render(self.batch())
        assert pool.render(self.batch())[0].ok

    def test_the_lease_is_released_on_error_too(self, tmp_path, service):
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        with pytest.raises(RuntimeError, match="episode died"):
            with pool.lease_embodied("ep-emb"):
                raise RuntimeError("episode died mid-walk")
        assert all(m.leased_to is None for m in pool.members)

    def test_leasing_probes_the_quarantined_before_giving_up(
            self, tmp_path, service):
        pool = RenderPool(write_endpoints(tmp_path / "endpoints.json", [service]))
        service.healthz_ok = False
        pool.check_health()
        pool.check_health()
        assert pool.members[0].quarantined
        service.healthz_ok = True
        with pool.lease_embodied("ep-emb") as client:
            assert client is pool.members[0].client
        assert not pool.members[0].quarantined


# ─────────────────────────────────────────────────────────────────────────────


@needs_maps
class TestAnEmbodiedEpisodeWalksTheStockGraph:
    def scripted_nodes(self, env, turns=8):
        """Drive by geometry alone (street names and headings, never images)
        and record the node after each move -- the same decision rule for the
        embodied env and the stock one, so the sequences diverge only if the
        *transitions* diverge."""
        nodes = []
        for turn in range(turns):
            rows = env.candidates()
            row = rows[turn % len(rows)]
            env.walk_to(row["street"], row["heading"])
            nodes.append(env.node_id)
        return nodes

    def test_the_node_trajectory_is_the_stock_graphs(
            self, paris, service, tmp_path):
        """UE owns the seconds; the GRAPH still owns the topology. Same seed,
        same action script, same node sequence as a stock env -- the hop
        lands on the node ``_step_to`` names, however the pawn got there."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        bare = CourierEnv(paris, seed=5, difficulty="solo")
        bare.reset()
        assert env.node_id == bare.node_id, "same seed, same spawn"
        assert self.scripted_nodes(env) == self.scripted_nodes(bare)

    def test_the_clock_is_engine_ticks_plus_declared_costs_and_nothing_else(
            self, paris, service, tmp_path):
        """The Track B clock contract, end to end: after a scripted walk the
        env clock equals the integrator's ticks * fixed_dt summed over hops;
        a declared non-movement cost (wait, a refused action's floor) adds
        exactly its declared seconds on top."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        assert env.fixed_dt == pytest.approx(FIXED_DT)
        self.scripted_nodes(env, turns=6)

        total_ticks = sum(hop["ticks"] for hop in env.embodied_log)
        walk_seconds = sum(hop["sim_seconds"] for hop in env.embodied_log)
        assert walk_seconds == pytest.approx(total_ticks * FIXED_DT)
        assert env.sim_seconds == pytest.approx(walk_seconds), (
            "movement seconds must be the engine's, with no arithmetic beside")

        waited = env.wait()
        refused = env.walk_to("No Such Street Anywhere")
        assert not refused.ok
        assert refused.sim_seconds == REJECTED_ACTION_SECONDS
        assert env.sim_seconds == pytest.approx(
            walk_seconds + waited.sim_seconds + refused.sim_seconds)

    def test_every_hops_ticks_match_the_integrators_own_arithmetic(
            self, paris, service, tmp_path):
        """The tick math, hop by hop: each walk starts where the last one
        landed (within arrive_cm of the previous node, not on it), and its
        tick count is exactly ceil((distance - arrive) / (speed * dt))."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        self.scripted_nodes(env, turns=6)
        assert service.walks, "the script must have walked"
        for row in service.walks:
            ticks = expected_ticks(row["start_xy"], row["target_xy"],
                                   row["arrive_cm"])
            assert row["ticks"] == ticks
            assert row["sim_seconds"] == pytest.approx(ticks * FIXED_DT)

    def test_the_pawn_lands_within_arrive_cm_of_every_node(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        self.scripted_nodes(env, turns=6)
        errors = [hop["pose_error_cm"] for hop in env.embodied_log]
        assert errors and max(errors) <= env.arrive_cm
        assert env.summary()["embodied"]["max_pose_error_cm"] <= env.arrive_cm

    def test_the_episode_spawns_the_pawn_at_the_reset_node(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                           spawn_z_cm=120.0)
        node = paris.nodes[env.node_id]
        spawn = service.episodes[0]["spawn"]
        assert (spawn["x_cm"], spawn["y_cm"]) == (node.x_cm, node.y_cm)
        assert spawn["z_cm"] == 120.0
        assert service.episodes[0]["agent"]["speed_cm_s"] == pytest.approx(
            env.embodiment.speed_cm_s)

    def test_frames_land_in_album_shape_from_the_pawns_camera(
            self, paris, service, tmp_path):
        from PIL import Image

        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        rows = env.candidates()
        assert rows
        for row in rows:
            assert row["image"], "every neighbour view must be observed"
            path = Path(row["image"])
            assert path.parent.parent == env.live_album.images
            with Image.open(path) as image:
                image.load()
            # v1 hazards off: nothing may show a lamp or an obstacle.
            assert row["signal_image"] is None
        before = len(service.observes)
        env.candidates()
        assert len(service.observes) == before, (
            "a second look is a cache hit, not a second observe")

    def test_the_summary_carries_the_embodied_evidence_block(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        self.scripted_nodes(env, turns=4)
        block = env.summary()["embodied"]
        assert set(block) == {"hops", "recoveries", "total_ticks", "total_walk_seconds",
                              "max_pose_error_cm", "stuck_count"}
        assert block["hops"] == len(env.embodied_log) > 0
        assert block["total_ticks"] == sum(h["ticks"] for h in env.embodied_log)
        assert block["total_walk_seconds"] == pytest.approx(
            block["total_ticks"] * FIXED_DT)
        assert block["stuck_count"] == 0
        log_keys = {"target_node", "target_xy", "ticks", "sim_seconds",
                    "walked_cm", "end_pose", "node_xy", "pose_error_cm",
                    "outcome"}
        assert all(set(hop) == log_keys for hop in env.embodied_log)

    def test_close_ends_the_episode_on_the_service(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        env.close()
        assert service.episode_ends == [EPISODE]
        env.close()  # idempotent: no second despawn, no exception
        assert service.episode_ends == [EPISODE]


@needs_maps
class TestConstructionRefusals:
    @pytest.mark.parametrize("key", [
        "album_root", "signal_album_root", "obstacle_album_root",
        "pavement_album_root", "pavement_obstacle_album_root",
        "obstacle_sidecar_root", "signal_sidecar_root",
    ])
    def test_hazard_albums_and_sidecars_are_refused_at_the_door(
            self, paris, tmp_path, key):
        """v1 embodied runs hazards OFF: locomotion realism is the thing
        under test, and a hazard root smuggled in would attach charges to a
        renderer that cannot show them."""
        with pytest.raises(ValueError, match="hazards OFF"):
            EmbodiedCourierEnv(paris, object(), episode_id=EPISODE,
                               cache_root=tmp_path,
                               **{key: str(tmp_path / "somewhere")})

    def test_difficulty_defaults_solo_but_other_tiers_are_not_refused(
            self, paris, service, tmp_path):
        solo = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        assert solo.difficulty == "solo"
        pair = EmbodiedCourierEnv(paris, UERenderClient(service.base_url),
                                  episode_id=EPISODE,
                                  cache_root=tmp_path / "pair",
                                  seed=5, difficulty="pair")
        assert pair.difficulty == "pair"


@needs_maps
class TestStuckAndTimeoutMapToTheStockRefusal:
    def hop_beyond(self, env, min_cm=2000.0):
        """A hop whose /walk distance from the pawn's actual pose clears
        ``min_cm``, found deterministically: the graph position is stood at a
        junction whose neighbour is far from the spawn pose (the pawn stays
        put -- the /walk starts from where the pawn really is, which is the
        distance that decides the tick count). Every map has one, so the
        engine-measured-charge scenarios never skip."""
        origin = (env.ue_pose.x_cm, env.ue_pose.y_cm)
        for node_id in sorted(env.network.nodes):
            for neighbour in sorted(env.network.nodes[node_id].neighbours):
                there = env.network.nodes[neighbour]
                if math.dist(origin, (there.x_cm, there.y_cm)) >= min_cm:
                    stand_at(env, node_id)
                    rows = {row["node"]: row for row in env.candidates()}
                    return rows[neighbour]
        pytest.fail("no edge far enough from spawn on this map")

    def test_a_wall_maps_to_the_stock_refusal_with_the_burned_seconds(
            self, paris, service, tmp_path):
        """The way_blocked semantics with an engine-measured price: code
        ``stuck``, not ok, the turn and the rejected action counted, the
        courier still at the junction -- and the charge is exactly the sim
        seconds the engine burned walking into the wall."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        row = self.hop_beyond(env)
        service.wall_after_cm = 1000.0  # 10 m of progress, then nothing
        before = env.sim_seconds
        node_before = env.node_id
        outcome = env._step_to(row["k"])
        assert not outcome.ok and outcome.code == "stuck"
        assert env.node_id == node_before, "a stuck walk does not hop"
        assert env.rejected_actions == 1
        assert env.blocked_attempts == 1
        assert (node_before, row["node"]) in env.witnessed_blocks

        walk = service.walks[-1]
        assert walk["stuck"] and not walk["arrived"]
        burned = walk["ticks"] * FIXED_DT
        assert burned > REJECTED_ACTION_SECONDS, (
            "this scenario must clear the floor or it proves nothing")
        assert outcome.sim_seconds == pytest.approx(burned)
        assert env.sim_seconds - before == pytest.approx(burned)
        assert env.summary()["embodied"]["stuck_count"] == 1
        # The failed hop is followed by its recovery re-spawn entry — the
        # hop is one before the end now, and the recovery names its cause.
        assert env.embodied_log[-2]["outcome"] == "stuck"
        assert env.embodied_log[-1]["recovery"] == "respawn"
        assert env.embodied_log[-1]["after"] == "stuck"

    def test_a_cheap_wall_still_pays_the_stock_refusal_floor(
            self, paris, service, tmp_path):
        """A pawn that stalls after 10 cm burned 0.07 s of engine time; the
        charge is the stock REJECTED_ACTION_SECONDS floor, because a refusal
        cheaper than any other refusal would be a probe."""
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path)
        row = env.candidates()[0]
        service.wall_after_cm = 10.0
        before = env.sim_seconds
        outcome = env._step_to(row["k"])
        assert not outcome.ok and outcome.code == "stuck"
        assert env.sim_seconds - before == REJECTED_ACTION_SECONDS
        assert outcome.sim_seconds == REJECTED_ACTION_SECONDS

    def test_a_timeout_is_the_same_refusal_under_its_own_code(
            self, paris, service, tmp_path):
        env = embodied_env(paris, UERenderClient(service.base_url), tmp_path,
                           max_walk_seconds=6.0)
        row = self.hop_beyond(env)
        before = env.sim_seconds
        outcome = env._step_to(row["k"])
        assert not outcome.ok and outcome.code == "walk_timeout"
        walk = service.walks[-1]
        assert walk["timeout"] and not walk["arrived"]
        burned = walk["ticks"] * FIXED_DT
        assert burned == pytest.approx(int(6.0 / FIXED_DT) * FIXED_DT)
        assert env.sim_seconds - before == pytest.approx(
            max(burned, REJECTED_ACTION_SECONDS))
        # A timeout is not a witnessed barrier: nothing was walked into.
        assert env.blocked_attempts == 0
        assert env.summary()["embodied"]["stuck_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────


class _EmbodiedAdapterUnderTest(EmbodiedCourierGymEnv):
    """The embodied adapter with the phone map rasterised by PIL instead of
    cairosvg, exactly as the live adapter's tests do; the rasteriser is
    inherited stock code and not what these tests defend."""

    def _rasterise(self, svg, index):
        from PIL import Image

        return self._fit(Image.new("RGB", (720, 540), (240, 240, 240)))


def walk_reply(env) -> str:
    street, heading = env._env.street_at(1)
    return f'THOUGHT: go\n```\nwalk_to("{street}", "{heading}")\n```'


@needs_maps
class TestTheEmbodiedTrainingAdapter:
    @pytest.fixture()
    def config(self, service, tmp_path):
        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        return {"backend": "embodied",
                "ue_endpoints": str(endpoints),
                "live_cache_root": str(tmp_path / "cache"),
                "spawn_z_cm": 120.0,
                "difficulty": "solo", "stride": "block",
                "max_turns": 3, "max_images": 2}

    def test_the_observation_contract_holds_over_the_embodied_backend(
            self, config, service):
        """The training contract, unchanged: placeholder count matches the
        image list, images are PIL objects, the info dict keeps its keys --
        success and env_return included -- and max_turns ends the episode."""
        from PIL import Image

        env = _EmbodiedAdapterUnderTest(config)
        obs, info = run(env.reset(0))
        assert info["backend"] == "embodied"
        assert info["episode_id"].endswith(env._cfg8)
        images = obs["multi_modal_input"]["<image>"]
        assert obs["obs_str"].count("<image>") == len(images)
        assert images and all(isinstance(image, Image.Image) for image in images)

        done, steps = False, 0
        while not done and steps < 5:
            obs, reward, done, info = run(env.step(walk_reply(env)))
            assert isinstance(reward, float)
            for key in ("status", "sim_seconds", "success", "env_return",
                        "earnings", "turns"):
                assert key in info
            steps += 1
        assert done, "max_turns did not end the episode"
        assert service.walks, "the steps must have walked the pawn"
        run(env.close())
        assert service.episode_ends, "close must end the embodied episode"

    def test_close_returns_the_lease_to_the_pool(self, config, service):
        env = _EmbodiedAdapterUnderTest(config)
        run(env.reset(0))
        pool = env._render_pool
        assert any(m.leased_to for m in pool.members), (
            "reset must have taken the exclusive lease")
        run(env.close())
        assert all(m.leased_to is None for m in pool.members)

    def test_a_second_reset_releases_the_first_episodes_lease(
            self, config, service):
        env = _EmbodiedAdapterUnderTest(config)
        run(env.reset(0))
        observed = len(service.observes)
        assert observed
        run(env.reset(0))
        pool = env._render_pool
        assert sum(m.leased_to is not None for m in pool.members) == 1, (
            "a reset storm must not hold one lease per reset")
        assert len(service.episode_ends) == 1, (
            "the first episode must have been ended, once")
        assert len(service.observes) == observed, (
            "same seed, same album: the second reset re-serves cached frames")
        run(env.close())

    def test_hazards_true_is_refused_not_stripped(self):
        with pytest.raises(ValueError, match="hazards"):
            EmbodiedCourierGymEnv({"backend": "embodied", "hazards": True})

    def test_hazards_defaults_false_without_boilerplate(self, config):
        env = _EmbodiedAdapterUnderTest(config)
        assert env.hazards is False

    @pytest.mark.parametrize("key", ["album_root", "obstacle_sidecar_root",
                                     "signal_sidecar_root",
                                     "sidecar_source_root"])
    def test_album_and_sidecar_keys_are_refused(self, key):
        with pytest.raises(ValueError, match=key):
            EmbodiedCourierGymEnv({"backend": "embodied",
                                   key: "/data/somewhere"})

    def test_a_config_meant_for_another_adapter_is_refused(self):
        with pytest.raises(ValueError, match="backend"):
            EmbodiedCourierGymEnv({"backend": "live"})

    def test_spawn_z_reaches_the_wire(self, config, service):
        env = _EmbodiedAdapterUnderTest(config)
        run(env.reset(0))
        assert service.episodes[0]["spawn"]["z_cm"] == 120.0
        run(env.close())

    def test_the_registry_launch_line_names_a_real_class(self):
        import importlib

        module = importlib.import_module(
            "embodiedbench.runtime.live.gym_adapter")
        assert getattr(module, "EmbodiedCourierGymEnv") is EmbodiedCourierGymEnv


class TestAnOversubscribedFleetWaitsItsTurn:
    """A trainer drives more concurrent episodes than the fleet has
    instances; the second episode parks until the first ends, rather than
    failing the reset (NoHealthyInstance used to be immediate)."""

    def test_a_lease_waits_for_the_previous_episode_to_end(self, service, tmp_path):
        import threading
        import time as _time
        from embodiedbench.runtime.live.pool import RenderPool

        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        pool = RenderPool(endpoints, lease_timeout_s=10.0, lease_poll_s=0.05)
        release = threading.Event()
        held = threading.Event()

        def first():
            with pool.lease_embodied("ep-one"):
                held.set()
                release.wait(timeout=5.0)

        thread = threading.Thread(target=first)
        thread.start()
        assert held.wait(timeout=5.0)
        t0 = _time.monotonic()
        threading.Timer(0.3, release.set).start()
        with pool.lease_embodied("ep-two") as client:
            waited = _time.monotonic() - t0
            assert client is not None
        thread.join(timeout=5.0)
        assert waited >= 0.25, "second lease should have parked until release"

    def test_a_dead_fleet_still_fails_fast(self, tmp_path):
        import json as _json
        from embodiedbench.runtime.live.pool import NoHealthyInstance, RenderPool

        endpoints = tmp_path / "endpoints.json"
        endpoints.write_text(_json.dumps({"version": 0, "instances": [
            {"id": "gone", "base_url": "http://127.0.0.1:9", "map_name": "x"}]}))
        pool = RenderPool(endpoints, lease_timeout_s=30.0, lease_poll_s=0.05)
        for member in pool.members:
            member.quarantined = True
        import pytest as _pytest
        import time as _time
        t0 = _time.monotonic()
        with _pytest.raises(NoHealthyInstance):
            with pool.lease_embodied("ep-dead"):
                pass
        assert _time.monotonic() - t0 < 5.0, "all-dead must not wait out the lease timeout"
