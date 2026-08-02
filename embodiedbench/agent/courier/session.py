"""One shift, assembled: the environment on one side, a model reply on the other.

Everything needed to run a courier episode existed before this module and none
of it was joined up. ``prompts.py`` could render an observation from strings,
``loop.py`` could parse a reply into a call, ``memory.py`` could remember a
trail, and ``CourierEnv`` could execute a tool -- but nothing turned the runtime
into the strings, dispatched the parsed call, or charged the budget, so the
input/output contract the benchmark is built on had no implementation and no
test. Every number reported about the harness was therefore about a harness
nobody had run end to end.

This is that implementation, and it is deliberately thin. It decides nothing:

  observe   read the runtime, render the same fields ``prompts.py`` declares,
            attach the photographs the runtime says exist
  step      parse the reply against the tools *this* condition enables,
            dispatch by name, charge, record what happened

The two rules it enforces are the ones the rest of the design rests on. First,
a tool that is not in the prompt cannot be dispatched, and a tool that is in the
prompt must be dispatchable -- the dispatch table is built from the same list
the prompt is built from, and a mismatch is a startup error rather than a turn
the agent loses. Second, the observation states what the world is, never what to
do about it: the only routing advice in the whole loop comes from ``navigate()``,
which the courier has to ask for and pay for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from embodiedbench.agent.courier.loop import (
    Budgets,
    CourierRun,
    FormatError,
    ParsedAction,
    Spend,
    TurnLog,
    budget_exceeded,
    parse_reply,
)
from embodiedbench.agent.courier.memory import CourierMemory
from embodiedbench.agent.courier.prompts import (
    FORMAT_ERROR_TEMPLATE,
    build_observation,
    build_system_prompt,
    render_candidates,
    render_photographs,
)
from embodiedbench.agent.courier.tools import TOOLS_BY_NAME, ToolKind


@dataclass(frozen=True)
class Frame:
    """One picture put in front of the courier, and what it is a picture of.

    ``kind`` separates the two sources, and the separation is load-bearing
    rather than tidy. A ``photograph`` comes out of the world through the
    courier's eyes and is the only place a light, a barrier or a shopfront ever
    appears. A ``map`` comes off the phone's screen and is drawn from the survey
    -- it knows geometry and names and can see nothing. Putting them in one
    undifferentiated list would let a courier conclude the phone can see the
    street -- and nothing can correct that belief, because there is no way to
    tell the phone anything. The route will name a shut street every time it is
    asked; only the photograph says it is shut.
    """

    label: str
    path: str = ""
    kind: str = "photograph"
    svg: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "path": self.path,
                "kind": self.kind, "svg": self.svg}


@dataclass
class Observation:
    """What the courier is shown this turn: the text, and the images beside it."""

    text: str
    frames: list[Frame] = field(default_factory=list)

    @property
    def image_paths(self) -> list[str]:
        return [f.path for f in self.frames if f.path]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "frames": [f.to_dict() for f in self.frames]}


class CourierSession:
    """Drives one episode of ``CourierEnv`` through the courier prompt contract."""

    def __init__(
        self,
        env: Any,
        *,
        city: str = "Paris",
        budgets: Budgets | None = None,
        with_images: bool = True,
    ):
        self.env = env
        self.city = city
        self.budgets = budgets or Budgets()
        self.with_images = with_images
        self.memory = CourierMemory()
        self.spend = Spend()
        self.run = CourierRun(spend=self.spend, budgets=self.budgets, memory=self.memory)
        self.feedback = ""
        self.allowed = list(env.allowed_tool_names())
        self.tools = [TOOLS_BY_NAME[name] for name in self.allowed]
        self.dispatch: dict[str, Callable[..., Any]] = {}
        for name in self.allowed:
            call = getattr(env, name, None)
            if call is None:
                # The prompt is generated from this same list. A name in it that
                # the runtime cannot execute is a turn the agent is guaranteed to
                # lose, and the whole tool/prompt split exists to make that
                # impossible -- so it fails here, at construction, not there.
                raise ValueError(
                    f"the prompt would advertise {name!r} but {type(env).__name__} "
                    "cannot execute it"
                )
            self.dispatch[name] = call
        self._record_arrival()
        # The slip is in the courier's pocket before the shift starts. Setting
        # the goal only after the first action meant turn 1 was always spent on
        # check_order() to learn what the job was -- a turn, 2 s, and a wholly
        # avoidable one, since a rider is handed the job with the bag.
        self._set_goal()

    # ── what the courier is shown ────────────────────────────────────────────

    def system_prompt(self) -> str:
        return build_system_prompt(city=self.city, tools=self.tools)

    def observe(self) -> Observation:
        rows = self.env.candidates()
        text = build_observation(
            memory=self.memory.render(),
            location=self.env.location_text(),
            clock=self.env.clock_text(),
            candidates=render_candidates(rows),
            photographs=(render_photographs(rows, phone_map=self.phone_map_on())
                         if self.with_images else ""),
            extra=(f"\n### what just happened\n{self.feedback}" if self.feedback else ""),
        )
        return Observation(text=text, frames=self._frames(rows))

    def _frames(self, rows: list[dict[str, Any]]) -> list[Frame]:
        """The images, in the order their captions are listed.

        Street views first, then any pedestrian lights, because that is the order
        ``render_photographs`` writes the captions in and a model matching
        captions to images by position must not be misled by the harness.
        """
        if not self.with_images:
            return []
        frames = [
            Frame(f"[{row['k']}] {row['street']}, {row.get('relative') or row['heading']}",
                  row["image"])
            for row in rows if row.get("image")
        ]
        frames += [
            Frame(f"[light {row['k']}] pedestrian light for street {row['k']}",
                  row["signal_image"])
            for row in rows if row.get("signal_image")
        ]
        if self.phone_map_on():
            # Last, and named for what it is. The captions are read in order and
            # the map is the one picture that did not come from the courier's
            # eyes, so it goes after everything that did.
            #
            # Re-rendered every turn, which is the point: a map app does not
            # switch off when the phone goes in a pocket. The route stays where
            # it was drawn and the dot showing where you are keeps moving. It
            # used to be shown for one turn only, and that made asking for it a
            # tax -- an episode played by hand spent 45% of its turns on
            # ``navigate()``, 37 lookups at 15 s each, nine minutes of an hour
            # standing still reading a phone.
            frames.append(Frame("[map] the map on your phone",
                                kind="map", svg=self.env.map_drawing().svg))
        return frames

    def phone_map_on(self) -> bool:
        """Has the courier asked for a route this shift, and does it still have one?"""
        return bool(getattr(self.env, "screen_target", None))

    # ── one turn ─────────────────────────────────────────────────────────────

    def step(self, reply: str) -> TurnLog:
        """Parse one model reply, execute it, charge for it, and record it."""
        observation = self.observe()
        turn = TurnLog(
            step=len(self.run.turns) + 1,
            prompt=observation.text,
            image_paths=observation.image_paths,
            reply=reply,
        )
        stopped = budget_exceeded(self.spend, self.budgets)
        if stopped:
            turn.status, turn.error = "truncated", stopped
            self._finish(stopped)
            self.run.turns.append(turn)
            return turn

        try:
            action = parse_reply(reply, set(self.allowed))
        except FormatError as error:
            self.spend.format_errors += 1
            self.spend.consecutive_format_errors += 1
            turn.status, turn.error = "format_error", str(error)
            self.feedback = FORMAT_ERROR_TEMPLATE.format(error=error)
            self.run.turns.append(turn)
            if budget_exceeded(self.spend, self.budgets) == "repeated_format_errors":
                self._finish("repeated_format_errors")
            return turn

        self.spend.consecutive_format_errors = 0
        turn.thought, turn.action = action.thought, action.render()
        turn.tool_kind = TOOLS_BY_NAME[action.tool].kind.value

        before = self.env.sim_seconds
        outcome = self._execute(action)
        turn.sim_seconds = self.env.sim_seconds - before
        turn.reward = outcome.reward
        turn.status = "accepted" if outcome.ok else "rejected"
        if not outcome.ok:
            turn.error = outcome.code
        self.feedback = outcome.message

        self.spend.steps += 1
        self.spend.tool_calls += 1
        self.spend.sim_seconds += turn.sim_seconds
        self.run.total_reward += outcome.reward

        if outcome.moved:
            self._record_arrival()
        self._remember(action, outcome)
        turn.memory = self.memory.to_dict()
        self.run.turns.append(turn)

        if outcome.finished or self.env.shift_over:
            self._finish("delivered" if outcome.finished else "shift_over")
        else:
            stopped = budget_exceeded(self.spend, self.budgets)
            if stopped:
                self._finish(stopped)
        return turn

    def _execute(self, action: ParsedAction):
        call = self.dispatch[action.tool]
        try:
            return call(*action.args, **action.kwargs)
        except TypeError as error:
            # A call with the wrong arity is a format problem, not a crash. The
            # world did not change, so it is reported the way a refusal is.
            from embodiedbench.runtime.city.courier_env import StepOutcome

            return StepOutcome(
                ok=False, code="bad_arguments",
                message=(
                    f"{TOOLS_BY_NAME[action.tool].signature()} does not take those "
                    f"arguments ({error})."
                ),
            )

    # ── bookkeeping ──────────────────────────────────────────────────────────

    def _record_arrival(self) -> None:
        self.memory.arrive(
            step=len(self.run.turns) + 1,
            node_id=self.env.node_id,
            street=self.env.street_of(self.env.node_id),
            address_hint=self.env.house_numbers_near(self.env.node_id),
        )

    def _set_goal(self) -> None:
        """Name the job in the notes -- but only while there is just one.

        With several live orders the clock block already lists every one of them
        with its stage and its deadline, and the notes' single ``Job:`` line then
        names one of the several as though it were the job, which contradicts the
        block directly above it. Sequencing is the agent's decision at those
        tiers, so the harness must not appear to have made it.
        """
        live = self.env.live_orders() if hasattr(self.env, "live_orders") else []
        order = self.env.active_order() if len(live) <= 1 else None
        if order is None:
            self.memory.set_goal("", "")
        else:
            self.memory.set_goal(
                "deliver to" if order.picked_up else "collect from", order.target.text
            )

    def _remember(self, action: ParsedAction, outcome: Any) -> None:
        self._set_goal()
        if outcome.ok and action.tool in ("walk_to", "follow_street"):
            rows = self.env.candidates()
            if len(rows) <= 1:
                self.memory.mark_dead_end(self.env.node_id)
            for row in rows:
                self.memory.saw_street(row["street"], f"seen from {self.memory.place_label(self.env.node_id)}")

    def _finish(self, reason: str) -> None:
        self.run.finished = True
        self.run.termination_reason = reason

    @property
    def finished(self) -> bool:
        return self.run.finished

    def report(self) -> dict[str, Any]:
        out = self.run.to_dict()
        out["env"] = self.env.summary()
        return out
