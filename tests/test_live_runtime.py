"""The live backend: a UE renderer wearing a photo album's contract.

The claim under test is indistinguishability. ``LiveCourierEnv`` must look,
to every downstream consumer -- CourierSession's frames, FrameAliases'
laundering, the training adapter's ``PIL.Image.open`` -- exactly like a
CourierEnv over a baked album that happens to fill in lazily. Four properties
carry that claim, and each is a test class here:

1. the frames land in album shape and open as real PNGs;
2. **transitions never depend on the renderer** -- the same seed and action
   script walk the same trajectory over a live backend, a dead one, and no
   album at all, because UE is a camera, not a physics engine;
3. the same (episode, key) renders exactly once per process, which is what
   keeps FrameAliases and media digests stable within a run;
4. the visibility sidecars -- copied from a bake, never invented -- are the
   only thing that switches the perceptual charges on, exactly as for a
   baked album ("silence is not consent", third mechanic, same rule).

Plus the request-correctness claims (a lamp request names the phase the
charge will use; an obstacle request carries the caller's own world state)
and the training adapter over a real HTTP service.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from embodiedbench.compiler.road_network import bearing_deg, build_road_network
from embodiedbench.runtime.city.courier_env import (
    BLOCKED_SECONDS,
    RED_CROSSING_PENALTY,
    RED_CROSSING_PENALTY_S,
    CourierEnv,
    signal_state,
)
from embodiedbench.runtime.city.obstacles import (
    ROAD_BLOCK,
    approach_key,
    obstacle_sites,
)
from embodiedbench.runtime.live.client import UERenderClient
from embodiedbench.runtime.live.env import (
    KERB_MARGIN_CM,
    LAMP_EYE_CM,
    STREET_EYE_CM,
    LiveCourierEnv,
)
from embodiedbench.runtime.live.gym_adapter import LiveCourierGymEnv

from live_stub import FakeRenderService, write_endpoints

MAPS = (Path(__file__).resolve().parents[1] / "vendor" / "vagen" / "vagen"
        / "envs" / "deliverybench" / "maps")
PARIS = MAPS / "citycore-paris"
needs_maps = pytest.mark.skipif(not PARIS.exists(), reason="vendored maps not present")

EPISODE = "courier-citycore-paris-s3"


def run(coro):
    """asyncio.run per call, for the same reason test_vagen_courier does."""
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def paris():
    return build_road_network(PARIS, map_name="citycore-paris")


@pytest.fixture()
def service(tmp_path):
    stub = FakeRenderService(tmp_path / "svc").start()
    yield stub
    stub.stop()


@pytest.fixture()
def sidecars(tmp_path, paris):
    """A sidecar source claiming everything is visible.

    The live album copies visibility claims from a baked album; the tests need
    the mechanics exercisable without a bake on this machine, so the claims
    are generated -- every signalised approach legible, every obstacle site
    visible both ways. ``lamps_are_per_approach`` skips the one-lamp grouping,
    which needs lamp boxes only a real bake can measure.
    """
    root = tmp_path / "sidecars"
    root.mkdir()
    signalised = paris.signalised_nodes()
    legible = [f"{node}|{toward}" for node in sorted(signalised)
               for toward in sorted(paris.nodes[node].neighbours)]
    (root / "signal_visibility.json").write_text(json.dumps(
        {"map": paris.map_name, "legible": legible,
         "lamps_are_per_approach": True}))
    neighbours = {n: sorted(node.neighbours) for n, node in paris.nodes.items()}
    visible = [approach_key(x, y)
               for a, b in obstacle_sites(neighbours, paris.map_name)
               for x, y in ((a, b), (b, a))]
    (root / "obstacle_visibility.json").write_text(json.dumps({"visible": visible}))
    return root


def live_env(paris, renderer, tmp_path, *, sidecars=None, obstacle_sidecars=None,
             signal_sidecars=None, seed=3, **kwargs):
    """``sidecars`` points BOTH roots at one directory (the tests' generated
    claims live together); the split params exercise the split itself."""
    env = LiveCourierEnv(
        paris, renderer, episode_id=EPISODE, cache_root=tmp_path / "cache",
        obstacle_sidecar_root=obstacle_sidecars or sidecars,
        signal_sidecar_root=signal_sidecars or sidecars,
        seed=seed, **kwargs)
    env.reset()
    return env


def scripted_trace(env, turns=10):
    """Drive an env by geometry alone and record what the world did.

    The decision rule reads only fields both a live and an album-less env
    populate identically (street names, headings) -- never an image -- so two
    envs given the same script diverge only if their *transitions* diverge,
    which is the thing being tested.
    """
    trace = []
    for turn in range(turns):
        rows = env.candidates()
        row = rows[turn % len(rows)]
        outcome = env.walk_to(row["street"], row["heading"])
        if turn % 4 == 3:
            env.wait()
        trace.append((env.node_id, round(env.sim_seconds, 6),
                      round(outcome.reward, 6), outcome.ok, outcome.code))
    return trace


def stand_at(env, node_id):
    """Put the courier somewhere specific, the way the reference tests do:
    directly, because the claim under test is about the mechanics at that
    spot, not about how the courier got there."""
    env.node_id = node_id
    env.arrived_from = None


def road_block_edge(env):
    """An edge this episode's obstacle field actually blocks."""
    for key, kind in sorted(env.obstacles.by_site.items()):
        if kind != ROAD_BLOCK:
            continue
        a, b = key.split("|")
        if env.obstacles.in_effect(a, b) == ROAD_BLOCK:
            return a, b
    pytest.skip("this seed placed no visible road_block on this map")


# ─────────────────────────────────────────────────────────────────────────────


@needs_maps
class TestALiveEpisodeIsAnAlbumBeingWritten:
    def test_frames_land_in_album_shape_and_open_with_pil(
            self, paris, service, tmp_path, sidecars):
        """Every picture a turn offers must be a real PNG at the album path
        for its edge -- the exact contract a baked album keeps, because the
        training adapter's ``Image.open(frame.path)`` is the consumer and it
        was not told anything changed."""
        from PIL import Image

        env = live_env(paris, UERenderClient(service.base_url), tmp_path,
                       sidecars=sidecars)
        seen = []
        for turn in range(6):
            rows = env.candidates()
            for row in rows:
                assert row["image"], f"no frame for {env.node_id}->{row['node']}"
                path = Path(row["image"])
                seen.append((path, row))
                if row["signal_image"]:
                    seen.append((Path(row["signal_image"]), row))
            row = rows[turn % len(rows)]
            env.walk_to(row["street"], row["heading"])
        assert seen
        for path, row in seen:
            assert path.exists()
            assert path.parent.parent == env.live_album.images, (
                f"{path} is outside the episode album")
            assert path.name.startswith("toward_")
            with Image.open(path) as image:
                image.load()

    def test_the_cache_is_a_real_album_to_a_stock_env(
            self, paris, service, tmp_path, sidecars):
        """Indistinguishability made literal: a *stock* CourierEnv pointed at
        the cache directory serves the very same files. Nothing downstream of
        the album contract can tell who wrote the album."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path,
                       sidecars=sidecars)
        rows = env.candidates()
        replay = CourierEnv(paris, seed=3,
                            album_root=env.live_album.root,
                            signal_album_root=env.live_album.root,
                            obstacle_album_root=env.live_album.root)
        replay.reset()
        assert replay.node_id == env.node_id  # same seed, same spawn
        for row in rows:
            assert replay.frame_for(env.node_id, row["node"]) == row["image"]

    def test_frame_aliases_launder_live_frames_like_any_others(
            self, paris, service, tmp_path, monkeypatch):
        """The cache keeps the album's leaky names inside; the existing alias
        layer is what the policy sees, and it must neither fail on a live
        frame nor leak its name."""
        from embodiedbench.agent.courier.frame_alias import FrameAliases

        monkeypatch.setenv("EMBODIEDBENCH_FRAME_CACHE", str(tmp_path / "aliases"))
        env = live_env(paris, UERenderClient(service.base_url), tmp_path)
        row = env.candidates()[0]
        alias = FrameAliases().alias(row["image"])
        assert alias
        assert "toward" not in Path(alias).name


@needs_maps
class TestTransitionsNeverDependOnTheRenderer:
    @pytest.mark.parametrize("stride", ["waypoint", "block"])
    def test_live_and_albumless_walk_the_same_trajectory(
            self, paris, service, tmp_path, stride):
        """The decisive architectural fact, asserted: same network, seed and
        action script give identical nodes, clock and rewards whether frames
        are rendered or absent. UE is a camera; the world is Python."""
        live = live_env(paris, UERenderClient(service.base_url), tmp_path,
                        seed=5, stride=stride)
        bare = CourierEnv(paris, seed=5, stride=stride)
        bare.reset()
        assert scripted_trace(live, turns=10) == scripted_trace(bare, turns=10)

    def test_a_busy_backend_skips_the_batch_and_the_next_look_retries(
            self, paris, service, tmp_path):
        """Busy is transient by spec: the episode is NOT degraded, the miss
        stays a miss, and the very next lookup at the same key renders it.
        Treating one 503 as fleet death was how a reset storm silently turned
        whole episodes text-only."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path)
        service.busy_batches = 1
        env.candidates()
        assert env.live_busy_skips == 1
        assert not env.live_degraded, "busy must never degrade the episode"
        # The skipped keys are still cache misses; the next look renders them.
        rows = env.candidates()
        assert all(row["image"] for row in rows)
        assert not env.live_degraded

    def test_a_dead_fleet_changes_pictures_not_physics(self, paris, tmp_path):
        """PLAN.md 7.1's "never required for every RL worker", as a test: with
        the whole fleet unreachable the env degrades to album mode -- no
        frames, one warning, zero exceptions -- and the trajectory is the
        album-less one to the last decimal."""
        dead = FakeRenderService(tmp_path / "dead").start()
        url = dead.base_url
        dead.stop()
        live = live_env(
            paris, UERenderClient(url, render_timeout_s=2.0), tmp_path, seed=5)
        bare = CourierEnv(paris, seed=5)
        bare.reset()
        assert scripted_trace(live, turns=8) == scripted_trace(bare, turns=8)
        assert live.live_degraded
        assert all(row["image"] is None for row in live.candidates())
        coverage = live.album_coverage()
        assert coverage["degraded"] and coverage["with_frame"] == 0


@needs_maps
class TestSameKeyRendersExactlyOnce:
    def test_repeated_looks_hit_the_cache_not_the_gpu(
            self, paris, service, tmp_path, sidecars):
        """Within-runtime replay determinism rests on this: FrameAliases and
        ``observation_media_hash`` assume the same picture keeps the same
        bytes, and a GPU only guarantees that if it is asked once."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path,
                       sidecars=sidecars)
        env.candidates()
        env.candidates()
        row = env.candidates()[0]
        env.walk_to(row["street"], row["heading"])
        env.candidates()
        assert service.render_counts, "the episode rendered nothing at all"
        assert max(service.render_counts.values()) == 1

    def test_a_second_runtime_over_the_same_cache_renders_nothing(
            self, paris, service, tmp_path, sidecars):
        """Same (episode, key), new env object: the album answers and the
        service is never asked again -- which is what lets a replayed episode
        reproduce its observations without a GPU agreeing with itself."""
        first = live_env(paris, UERenderClient(service.base_url), tmp_path,
                         sidecars=sidecars)
        first.candidates()
        before = dict(service.render_counts)
        second = live_env(paris, UERenderClient(service.base_url), tmp_path,
                          sidecars=sidecars)
        second.candidates()
        assert dict(service.render_counts) == before


class TestConcurrentStoresOfOneKey:
    def test_two_albums_over_one_directory_race_without_corruption(self, tmp_path):
        """Two LiveAlbum objects (two env instances) sharing one episode dir
        must not be able to hurt each other: unique temp names mean nobody
        truncates a peer's half-written bytes, and losing the publish race is
        a success because the winner's file is the same frame. The old fixed
        '.part' name could raise FileNotFoundError from os.replace or publish
        torn bytes."""
        import base64
        import threading

        from PIL import Image

        from embodiedbench.runtime.live.cache import LiveAlbum
        from embodiedbench.runtime.live.protocol import RenderResult
        from live_stub import draw_frame

        keys = [f"race/toward_{index}" for index in range(60)]
        results = {
            key: RenderResult(
                key=key, status="ok",
                png_base64=base64.b64encode(draw_frame(key, 32, 24)).decode())
            for key in keys
        }
        albums = [LiveAlbum(tmp_path, "ep"), LiveAlbum(tmp_path, "ep")]
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def hammer(album: LiveAlbum) -> None:
            try:
                barrier.wait()
                for key in keys:
                    album.store(key, results[key])
            except Exception as error:  # noqa: BLE001 - the assert below reports it
                errors.append(error)

        threads = [threading.Thread(target=hammer, args=(album,))
                   for album in albums]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"a store raced into an exception: {errors}"
        for key in keys:
            path = albums[0].path_for(key)
            assert path.exists()
            with Image.open(path) as image:
                image.load()  # a torn publish would fail to decode
        assert not list(albums[0].images.rglob(".part-*")), (
            "a loser's temp file was left behind")


@needs_maps
class TestSidecarsDecideTheCharges:
    def test_with_sidecars_the_mechanics_bite_per_stock_rules(
            self, paris, service, tmp_path, sidecars):
        """Copied claims switch the charges on, and the numbers are the stock
        constants: a witnessed barrier costs 45 s and a refusal, a red
        crossing costs the -1.0 reward and 75 s at the kerb."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path,
                       sidecars=sidecars)
        assert env.enforce_signals and env.enforce_obstacles
        assert env.visible_signals
        assert env.obstacles.visible

        a, b = road_block_edge(env)
        stand_at(env, a)
        rows = {row["node"]: row for row in env.candidates()}
        before = env.sim_seconds
        outcome = env._step_to(rows[b]["k"])
        assert outcome.code == "way_blocked" and not outcome.ok
        assert env.blocked_attempts == 1
        assert env.sim_seconds - before == BLOCKED_SECONDS

        node = sorted(env.signalised)[0]
        stand_at(env, node)
        rows = env.candidates()
        row = rows[0]
        if signal_state(node, row["bearing"], env.sim_seconds) == "green":
            env.sim_seconds += 60.0  # the other phase; the whole city switches
        before = env.sim_seconds
        outcome = env._step_to(row["k"])
        assert outcome.ok and outcome.reward == -RED_CROSSING_PENALTY
        assert env.red_crossings == 1
        assert env.sim_seconds - before >= RED_CROSSING_PENALTY_S

    def test_signals_stay_off_without_the_explicit_opt_in(
            self, paris, service, tmp_path, sidecars):
        """The default live config: obstacle claims transfer (same street
        camera pose as the bake), signal claims do not (the bake aimed at the
        lens, the live camera stands at the node). With only
        obstacle_sidecar_root set, obstacles charge and signals are OFF -- no
        lamp frames, no red-light charge -- because a charge whose picture
        may not show the lamp is the defect the gate exists to prevent."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path,
                       obstacle_sidecars=sidecars)
        assert env.obstacles.visible, "obstacle claims should have transferred"
        assert env.visible_signals is None, "signal claims must not transfer"
        assert not (env.live_album.root / "signal_visibility.json").exists()

        node = sorted(env.signalised)[0]
        stand_at(env, node)
        rows = env.candidates()
        for row in rows:
            assert row["signal_image"] is None
        row = rows[0]
        if signal_state(node, row["bearing"], env.sim_seconds) == "green":
            env.sim_seconds += 60.0
        outcome = env._step_to(row["k"])
        assert outcome.ok and env.red_crossings == 0

    def test_the_signal_opt_in_charges_and_warns(
            self, paris, service, tmp_path, sidecars, caplog):
        """signal_sidecar_root is usable -- but only past a WARNING that says
        plainly the certification does not transfer in v0. With it set, the
        stock red-crossing charge fires (that is what opting in means)."""
        import logging

        with caplog.at_level(logging.WARNING,
                             logger="embodiedbench.runtime.live.cache"):
            env = live_env(paris, UERenderClient(service.base_url), tmp_path,
                           obstacle_sidecars=sidecars, signal_sidecars=sidecars)
        assert any("does not transfer" in record.message
                   for record in caplog.records), (
            "opting in to the untransferable signal claims must warn")
        assert env.visible_signals

        node = sorted(env.signalised)[0]
        stand_at(env, node)
        rows = env.candidates()
        row = rows[0]
        if signal_state(node, row["bearing"], env.sim_seconds) == "green":
            env.sim_seconds += 60.0
        outcome = env._step_to(row["k"])
        assert outcome.ok and outcome.reward == -RED_CROSSING_PENALTY
        assert env.red_crossings == 1

    def test_a_reused_dir_drops_sidecars_the_config_did_not_ask_for(
            self, paris, service, tmp_path, sidecars):
        """The cache dir is reusable across same-seed resets, so the sidecars
        must match the *current* constructor, not history: an episode dir
        populated by a both-sidecars run, reopened by an obstacle-only config,
        loses the signal file and the signal mechanics with it."""
        first = live_env(paris, UERenderClient(service.base_url), tmp_path,
                         obstacle_sidecars=sidecars, signal_sidecars=sidecars)
        assert (first.live_album.root / "signal_visibility.json").exists()

        second = live_env(paris, UERenderClient(service.base_url), tmp_path,
                          obstacle_sidecars=sidecars)
        assert not (second.live_album.root / "signal_visibility.json").exists()
        assert second.visible_signals is None
        assert second.obstacles.visible

        third = live_env(paris, UERenderClient(service.base_url), tmp_path)
        assert not (third.live_album.root / "obstacle_visibility.json").exists()
        assert third.obstacles.visible is None

    def test_without_sidecars_the_mechanics_are_silently_off(
            self, paris, service, tmp_path):
        """No claim, no charge -- byte for byte the bare-album default. The
        same blocked edge is walkable, nothing shows a lamp, and the summary
        says zero obstacles are in effect, so the silence is measurable."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path)
        assert env.visible_signals is None
        assert env.obstacles.visible is None
        assert env.summary()["obstacles"]["in_effect"] == 0

        sited = sorted(k for k, kind in env.obstacles.by_site.items()
                       if kind == ROAD_BLOCK)
        if not sited:
            pytest.skip("this seed placed no road_block site")
        a, b = sited[0].split("|")
        stand_at(env, a)
        rows = {row["node"]: row for row in env.candidates()}
        outcome = env._step_to(rows[b]["k"])
        assert outcome.ok and outcome.moved
        assert env.blocked_attempts == 0

        node = sorted(env.signalised)[0]
        stand_at(env, node)
        for row in env.candidates():
            assert row["signal_image"] is None
        assert env.red_crossings == 0


@needs_maps
class TestRenderRequestsCarryTheWorldState:
    def test_a_lamp_request_names_the_approach_and_the_phase(
            self, paris, service, tmp_path, sidecars):
        """The service never consults a clock -- the caller decides the phase
        from ``sim_seconds``, and it must be the same ``signal_state`` the
        charge uses, or picture and penalty disagree by one minute, which is
        the exact bake defect the docstrings pin."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path,
                       sidecars=sidecars)
        node = sorted(env.signalised)[0]
        stand_at(env, node)
        toward = sorted(paris.nodes[node].neighbours)[0]
        bearing = bearing_deg(env.position(node), env.position(toward))
        expected = signal_state(node, bearing, env.sim_seconds)

        frame = env.signal_frame_for(node, toward)
        assert frame and frame.endswith(f"toward_{toward}_{expected}.png")
        item = next(item for batch in service.batches
                    for item in batch["requests"]
                    if item["render_kind"] == "lamp")
        assert item["signal"] == {"approach": f"{node}|{toward}", "state": expected}
        assert item["camera"] == {"width": 1280, "height": 960, "fov_deg": 40.0}
        assert item["z_cm"] == LAMP_EYE_CM

        # A minute later the whole city has switched, and the *other* phase is
        # requested under its own key -- two frames, two cache entries, the
        # pair a baked signal album carries.
        env.sim_seconds += 60.0
        other = env.signal_frame_for(node, toward)
        assert other and other != frame
        states = {item["signal"]["state"] for batch in service.batches
                  for item in batch["requests"] if item.get("signal")}
        assert states == {"red", "green"}

    def test_an_obstacle_request_carries_kind_and_edge_geometry(
            self, paris, service, tmp_path, sidecars):
        """The caller owns the obstacle set (deterministic in map and seed);
        the service owns prop geometry. So the request must carry kind, both
        edge endpoints in world cm and the street's width -- everything the
        layout table needs and nothing the service could invent."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path,
                       sidecars=sidecars, embodiment="human_on_scooter")
        a, b = road_block_edge(env)
        stand_at(env, a)
        frame = env.obstacle_frame_for(a, b)
        assert frame and frame.endswith(f"toward_{b}_{ROAD_BLOCK}.png")

        item = next(item for batch in service.batches
                    for item in batch["requests"]
                    if item["render_kind"] == "obstacle")
        obstacle = item["obstacle"]
        assert obstacle["kind"] == ROAD_BLOCK
        assert tuple(obstacle["a_cm"]) == env.position(a)
        assert tuple(obstacle["b_cm"]) == env.position(b)
        assert obstacle["street_width_cm"] == env._edge_width_cm(a, b)
        # A scooter is in the carriageway: camera on the node, no kerb offset,
        # and the request says which viewpoint it is rendering.
        assert obstacle["viewpoint"] == "carriageway"
        assert (item["x_cm"], item["y_cm"]) == env.position(a)
        assert item["z_cm"] == STREET_EYE_CM

    def test_the_pavement_viewpoint_stands_the_camera_on_the_kerb(
            self, paris, service, tmp_path):
        """The default courier is on foot, and on foot means the pavement:
        the street camera moves off the centreline by the pavement bake's own
        rule -- half the street's width plus the kerb margin, perpendicular to
        the right of travel -- so live frames show the world the body actually
        stands in."""
        env = live_env(paris, UERenderClient(service.base_url), tmp_path)
        assert env.viewpoint_served == "pavement"
        node_id = env.node_id
        env.candidates()

        node = paris.nodes[node_id]
        toward = sorted(node.neighbours)[0]
        key = f"{node_id}/toward_{toward}"
        item = next(item for batch in service.batches
                    for item in batch["requests"] if item["key"] == key)
        yaw = bearing_deg((node.x_cm, node.y_cm), env.position(toward))
        offset = env.streets[node.street_index].width_cm / 2.0 + KERB_MARGIN_CM
        right = math.radians(yaw + 90.0)
        assert item["x_cm"] == pytest.approx(node.x_cm + offset * math.cos(right))
        assert item["y_cm"] == pytest.approx(node.y_cm + offset * math.sin(right))
        assert item["yaw_deg"] == pytest.approx(yaw)


class _AdapterUnderTest(LiveCourierGymEnv):
    """The live adapter with the phone map rasterised by PIL instead of
    cairosvg, which needs a native cairo this machine does not have. The
    rasteriser is inherited stock code and is not what these tests defend;
    everything else -- the env construction, the frames, the budget, the
    contract -- runs unmodified."""

    def _rasterise(self, svg, index):
        from PIL import Image

        return self._fit(Image.new("RGB", (720, 540), (240, 240, 240)))


def walk_reply(env) -> str:
    """A legal walk at whatever junction the courier stands, read off the
    environment the way test_vagen_courier's helper does."""
    street, heading = env._env.street_at(1)
    return f'THOUGHT: go\n```\nwalk_to("{street}", "{heading}")\n```'


@needs_maps
class TestTheLiveTrainingAdapter:
    @pytest.fixture()
    def config(self, service, tmp_path, sidecars):
        endpoints = write_endpoints(tmp_path / "endpoints.json", [service])
        return {"backend": "live",
                "ue_endpoints": str(endpoints),
                "live_cache_root": str(tmp_path / "cache"),
                "obstacle_sidecar_root": str(sidecars),
                "difficulty": "solo", "stride": "block",
                "max_turns": 3, "max_images": 2}

    def test_the_observation_contract_holds_over_live_frames(self, config):
        """The four async methods and the obs shape, end to end over real
        HTTP: placeholders match images, the images are PIL objects the
        rollout engine can hand to a processor, done ends the episode."""
        from PIL import Image

        env = _AdapterUnderTest(config)
        obs, info = run(env.reset(0))
        assert info["backend"] == "live"
        images = obs["multi_modal_input"]["<image>"]
        assert obs["obs_str"].count("<image>") == len(images)
        assert images and all(isinstance(image, Image.Image) for image in images)

        done, steps = False, 0
        while not done and steps < 5:
            obs, reward, done, info = run(env.step(walk_reply(env)))
            assert isinstance(reward, float)
            steps += 1
        assert done, "max_turns did not end the episode"
        run(env.close())

    def test_resetting_the_same_seed_reuses_the_episode_album(
            self, config, service):
        """The episode id carries the seed, so reset(0) twice is one album:
        the second reset opens the same directory and renders nothing new --
        idempotency across resets, not merely within one."""
        env = _AdapterUnderTest(config)
        run(env.reset(0))
        before = dict(service.render_counts)
        assert before
        run(env.reset(0))
        assert dict(service.render_counts) == before
        assert max(service.render_counts.values()) == 1
        run(env.close())

    def test_an_album_root_config_is_refused(self):
        """Two sources of truth about one directory is how training and
        evaluation drift; the config that asks for both dies at construction."""
        with pytest.raises(ValueError, match="album_root"):
            LiveCourierGymEnv({"backend": "live", "album_root": "/data/somewhere"})

    def test_a_config_meant_for_the_album_adapter_is_refused(self):
        with pytest.raises(ValueError, match="backend"):
            LiveCourierGymEnv({"backend": "album"})

    @pytest.mark.parametrize("key", ["obstacle_sidecar_root",
                                     "signal_sidecar_root"])
    def test_a_missing_sidecar_source_is_refused_not_skipped(self, tmp_path, key):
        """The same refusal the stock adapter makes for a missing album, for
        the same reason: a path that silently degrades to mechanics-off is a
        drift nobody decided on."""
        with pytest.raises(FileNotFoundError, match=key):
            LiveCourierGymEnv({"backend": "live",
                               key: str(tmp_path / "absent")})

    def test_the_retired_sidecar_source_root_key_is_refused(self):
        """The old single-root key cannot express the split-by-validity rule;
        a config still carrying it must fail loudly, not silently drop a
        sidecar."""
        with pytest.raises(ValueError, match="sidecar_source_root"):
            LiveCourierGymEnv({"backend": "live",
                               "sidecar_source_root": "/data/somewhere"})

    def test_the_registry_launch_line_names_a_real_class(self):
        """The launch line registers the adapter by dotted path
        (+env_registry.Courier=embodiedbench.runtime.live.gym_adapter
        .LiveCourierGymEnv); if the path stops resolving, every training run
        dies at hydra parse time, so it is pinned here."""
        import importlib

        module = importlib.import_module("embodiedbench.runtime.live.gym_adapter")
        assert getattr(module, "LiveCourierGymEnv") is LiveCourierGymEnv
