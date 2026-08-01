"""``delivery@1`` task plugin (PLAN.md 8.1).

M1 needs the plugin to do three things honestly: state its requirements,
generate a deterministic ``EpisodeSpec`` from ``(world, seed, config)``, and
score a trajectory. Delivery mechanics themselves stay in the vendored engine
behind the runtime — this is the contract layer, not a second implementation.

Determinism is the point. PLAN.md M4 requires the same ``(environment, task
config, seed)`` to produce a byte-identical ``EpisodeSpec`` in two clean
processes, so generation uses an explicitly seeded ``random.Random`` and sorted
inputs. No ``set`` iteration, no dict ordering assumptions, no wall-clock.
"""

from __future__ import annotations

import random
from typing import Any

from embodiedbench.schemas.environment import NavigationMode
from embodiedbench.schemas.episode import (
    Budgets,
    CourierProfile,
    EpisodeSpec,
    ScheduledOrder,
    TaskConfigRef,
)
from embodiedbench.schemas.geometry import FrameName, Pose, Vec3
from embodiedbench.schemas.runtime import RuntimeMode
from embodiedbench.schemas.trajectory import MetricValue, ScoreReport, Trajectory
from embodiedbench.schemas.world import WorldBundle
from embodiedbench.tasks.core import SolvabilityVerdict, TaskRequirements

# The M0-frozen v1 courier profile set (ADR-0004). scooter_veteran is
# deliberately absent: PLAN.md 8.2 conditions it on a scientific justification
# that does not exist.
COURIER_PROFILES: dict[str, CourierProfile] = {
    "walker_novice": CourierProfile(
        profile_id="walker_novice",
        transport_modes=["walk"],
        carrying_capacity=1,
        owns_scooter=False,
        battery_enabled=False,
        outcome_relevant_fields=["transport_modes", "carrying_capacity", "owns_scooter"],
    ),
    "scooter_standard": CourierProfile(
        profile_id="scooter_standard",
        transport_modes=["walk", "scooter"],
        carrying_capacity=3,
        owns_scooter=True,
        battery_enabled=True,
        outcome_relevant_fields=[
            "transport_modes", "carrying_capacity", "owns_scooter", "battery_enabled",
        ],
    ),
    "multi_modal_courier": CourierProfile(
        profile_id="multi_modal_courier",
        transport_modes=["walk", "scooter", "bus", "car"],
        carrying_capacity=4,
        owns_scooter=True,
        battery_enabled=True,
        bus_access=True,
        car_rental_access=True,
        outcome_relevant_fields=[
            "transport_modes", "carrying_capacity", "owns_scooter", "battery_enabled",
            "bus_access", "car_rental_access",
        ],
    ),
}


class DeliveryTask:
    """The Delivery task plugin."""

    id = "delivery"
    version = "0.1.0"

    def requirements(self, config: dict[str, Any]) -> TaskRequirements:
        orders = int(config.get("order_count", 3))
        return TaskRequirements(
            affordances={
                "restaurant_dock": max(1, orders),
                "building_dock": max(1, orders),
            }
        )

    # ── generation ───────────────────────────────────────────────────────────

    def generate(self, world: WorldBundle, seed: int, config: dict[str, Any]) -> EpisodeSpec:
        rng = random.Random(seed)
        profile_name = config.get("courier_profile", "scooter_standard")
        if profile_name not in COURIER_PROFILES:
            raise ValueError(
                f"unknown courier profile {profile_name!r}; the M0-frozen v1 set is "
                f"{sorted(COURIER_PROFILES)}"
            )

        # Sorted so generation never depends on bundle ordering.
        certified_nodes = sorted(
            {e.from_node for e in world.nav_graph.certified_edges()}
            | {e.to_node for e in world.nav_graph.certified_edges()}
        )
        if not certified_nodes:
            raise ValueError(f"world {world.world_id} has no certified nodes to spawn on")
        node_by_id = {n.node_id: n for n in world.nav_graph.nodes}

        spawn_node = node_by_id[rng.choice(certified_nodes)]
        sites = sorted(world.interaction_sites, key=lambda s: s.site_id)
        order_count = int(config.get("order_count", 3))
        orders: list[ScheduledOrder] = []
        if len(sites) >= 2:
            interval = float(config.get("order_interval_s", 300.0))
            for index in range(order_count):
                pickup = sites[rng.randrange(len(sites))]
                dropoff = sites[rng.randrange(len(sites))]
                orders.append(
                    ScheduledOrder(
                        order_id=f"o_{index}",
                        available_from_sim_time_s=index * interval,
                        pickup_site=pickup.site_id,
                        dropoff_site=dropoff.site_id,
                    )
                )

        return EpisodeSpec(
            instance_id=f"{world.world_id}_{self.id}_{seed:06d}",
            environment_id=config.get("environment_id", f"{world.world_id}-delivery"),
            environment_version=world.version,
            environment_sha256=world.content_hash(),
            task=TaskConfigRef(plugin=self.id, version=self.version),
            embodiment_profile=config.get("embodiment_profile", "abstract_courier_v1"),
            courier_profile=COURIER_PROFILES[profile_name],
            navigation_mode=NavigationMode.NAV_WAYPOINT,
            runtime_track=RuntimeMode.TEXT,
            seed=seed,
            spawn=Pose(
                frame=FrameName.BUNDLE_WORLD,
                position=Vec3(
                    x_cm=spawn_node.position.x_cm,
                    y_cm=spawn_node.position.y_cm,
                    z_cm=spawn_node.position.z_cm,
                ),
                yaw_deg=float(config.get("spawn_yaw_deg", 0.0)),
            ),
            order_schedule=orders,
            observable_instruction=(
                "Complete as many deliveries as you can before your budget runs out."
            ),
            budgets=Budgets(
                steps=int(config.get("max_steps", 400)),
                tool_calls=config.get("tool_calls"),
                sim_s=config.get("sim_s"),
                output_tokens=config.get("output_tokens"),
            ),
            evaluator_id="delivery_score",
            evaluator_version="0.1.0",
            private_state_ref=f"private://{world.world_id}/{seed}",
        )

    def check_solvable(self, spec: EpisodeSpec, world: WorldBundle) -> SolvabilityVerdict:
        """PLAN.md 8.3's rejection rules, as far as M1 can evaluate them."""
        reasons: list[str] = []
        known_sites = {s.site_id for s in world.interaction_sites}
        for order in spec.order_schedule:
            if order.pickup_site not in known_sites:
                reasons.append(f"order {order.order_id} pickup site is not in the world")
            if order.dropoff_site not in known_sites:
                reasons.append(f"order {order.order_id} dropoff site is not in the world")
        if spec.budgets.steps <= 0:
            reasons.append("step budget is not positive")
        # PLAN.md 8.3: task success must not depend on an unvalidated synthetic edge.
        certified = {e.edge_id for e in world.nav_graph.certified_edges()}
        if not certified:
            reasons.append("no certified edges: every route would use excluded geometry")
        return SolvabilityVerdict(feasible=not reasons, reasons=reasons)

    # ── action surface ───────────────────────────────────────────────────────

    def action_schema(self, state: Any = None) -> list[str]:
        return [
            "VIEW_ORDERS", "ACCEPT_ORDER", "MOVE", "MOVE_TO", "PICKUP", "DROP_OFF", "WAIT",
        ]

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, trajectory: Trajectory, privileged: dict[str, Any]) -> ScoreReport:
        """Score a finished episode (PLAN.md 8.1, 12.5).

        The primary metric is left ``None`` with a stated reason rather than
        fabricated: PLAN.md 12.5 defines ``normalized_utility_vs_upper_bound``
        against a documented upper-bound policy for that exact frozen order
        schedule, and no such policy exists before M7. Reporting a number here
        would be reporting a ratio to nothing.
        """
        deliveries = float(privileged.get("delivered_count") or 0)
        earnings = float(privileged.get("earnings_total") or 0.0)
        success = bool(privileged.get("success", False))
        invalid_actions = sum(
            1 for turn in trajectory.turns if turn.action_result.status.value != "accepted"
        )

        metrics = {
            "deliveries": MetricValue(value=deliveries),
            "net_earnings": MetricValue(value=earnings),
            "steps_taken": MetricValue(value=float(len(trajectory.turns))),
            "invalid_actions": MetricValue(value=float(invalid_actions)),
            "total_reward": MetricValue(value=float(trajectory.total_reward)),
        }
        return ScoreReport(
            instance_id=trajectory.instance_id,
            episode_id=trajectory.episode_id,
            evaluator_id="delivery_score",
            evaluator_version="0.1.0",
            success=success,
            normalized_utility_vs_upper_bound=None,
            upper_bound_undefined_reason=(
                "no documented upper-bound policy exists before M7; PLAN.md 12.5 defines the "
                "primary metric relative to one, so it is reported as undefined rather than "
                "estimated"
            ),
            metrics=metrics,
            costs={
                "environment_steps": float(len(trajectory.turns)),
                "output_tokens": float(
                    sum(t.tokens.response_tokens for t in trajectory.turns)
                ),
            },
            trajectory_sha256=trajectory.content_hash(),
        )
