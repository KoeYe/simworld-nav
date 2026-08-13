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
    TruncatedReply,
    ParsedAction,
    Spend,
    TurnLog,
    budget_exceeded,
    parse_reply,
)
from embodiedbench.agent.courier.frame_alias import FrameAliases
from embodiedbench.agent.courier.memory import CourierMemory
from embodiedbench.agent.courier.prompts import (
    FORMAT_ERROR_TEMPLATE,
    REJECTED_TEMPLATE,
    TRUNCATED_TEMPLATE,
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
        frame_aliases: FrameAliases | None = None,
    ):
        self.env = env
        self.city = city
        self.budgets = budgets or Budgets()
        self.with_images = with_images
        self.frames_seen = frame_aliases or FrameAliases()
        self.memory = CourierMemory()
        self.spend = Spend()
        self.run = CourierRun(spend=self.spend, budgets=self.budgets, memory=self.memory)
        # How many times in a row the identical call has been refused, and what
        # it was. Four, because three is a run a policy can plausibly be working
        # through -- of the measured runs, every one of length 2 or 3 was a
        # model trying something adjacent, and every one of length 5 or more was
        # a model repeating itself word for word.
        self._repeats = 0
        self._last_refused: str | None = None
        # The junction a walk started from, so it can be credited with the way
        # out that was taken rather than the one arrived at.
        self._leaving: str | None = None
        # The candidate rows the courier was last shown, so what it typed can
        # be resolved back to the row it meant before it is remembered. Memory
        # used to record the typed strings verbatim -- a bearing left out
        # because the name was unique, or given as "left" -- and then be
        # queried with the row's compass heading, so the very marker meant to
        # stop verbatim repeats could not fire on the repeats it was for.
        self._offered: list[dict[str, Any]] = []
        self.lamp_legs: dict[str, str] = {}
        self.feedback = ""
        self.allowed = list(env.allowed_tool_names())
        self.tools = [TOOLS_BY_NAME[name] for name in self.allowed]
        # Tools the session runs itself. The notebook belongs to the courier,
        # not to the city: nothing about the world changes when a line is
        # written, and putting a no-op on CourierEnv purely to satisfy the
        # lookup would put a lie in the environment's interface.
        self._own_tools: dict[str, Callable[..., Any]] = {"note": self._note}
        self.dispatch: dict[str, Callable[..., Any]] = {}
        for name in self.allowed:
            call = self._own_tools.get(name) or getattr(env, name, None)
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
        self._assert_prompt_only_names_callable_tools()
        self._record_arrival()
        # The slip is in the courier's pocket before the shift starts. Setting
        # the goal only after the first action meant turn 1 was always spent on
        # check_order() to learn what the job was -- a turn, 2 s, and a wholly
        # avoidable one, since a rider is handed the job with the bag.
        self._set_goal()

    def _assert_prompt_only_names_callable_tools(self) -> None:
        """The symmetry check, applied to the prose as well as the menu.

        The tool *table* was built from ``allowed`` and so could not disagree
        with the dispatch table. The paragraphs around it were free text, and
        they named ``follow_street(k, n)`` at block stride where it is not
        dispatchable -- so a policy following the instructions it had been given
        lost a turn to a format error. Anything shaped like a call in the
        rendered prompt has to be callable.
        """
        import re

        prompt = self.system_prompt()
        advertised = {
            name for name in TOOLS_BY_NAME
            if re.search(rf"\b{re.escape(name)}\s*\(", prompt)
        }
        unavailable = sorted(advertised - set(self.allowed))
        if unavailable:
            raise ValueError(
                f"the prompt advertises {unavailable}, which this environment "
                f"cannot execute (allowed: {sorted(self.allowed)})"
            )

    # ── what the courier is shown ────────────────────────────────────────────

    def system_prompt(self) -> str:
        # Taken from the environment, never configured separately. A prompt that
        # promises narrated lights to a courier whose lights are only in the
        # pictures is teaching a rule that does not hold, and the two would
        # drift apart the first time either was changed alone.
        return build_system_prompt(
            city=self.city, tools=self.tools,
            narration=getattr(self.env, "narration", "none"))

    def observe(self) -> Observation:
        rows = self.env.candidates()
        # ``render_candidates`` has always been able to print "(you have walked
        # this before)" and nothing has ever set the flag it reads, so the line
        # had never once appeared. Setting it here is the whole of the fix: the
        # fact belongs next to the choice it bears on, not in a list further up
        # the prompt that has to be cross-referenced against a menu.
        here = self.env.node_id
        for row in rows:
            street = row.get("street", "")
            # The lookup key is the row's own first-edge bearing -- the same
            # string the candidate line prints and ``_remember`` records. It
            # read ``reach_heading`` for a while, which on 5.6% of rows is a
            # different compass point than the recorder wrote, and on those
            # rows "(you have walked this before)" could never appear.
            heading = row.get("heading", "")
            row["seen"] = self.memory.has_taken(here, street, heading)
            row["refused"] = self.memory.refusal_at(here, street, heading)
        text = build_observation(
            memory=self.memory.render(),
            location=self.env.location_text(),
            clock=self.env.clock_text(),
            candidates=render_candidates(rows),
            photographs=(render_photographs(rows, phone_map=self.phone_map_on())
                         if self.with_images else ""),
            extra=(f"\n### what just happened\n{self.feedback}" if self.feedback else ""),
        )
        self._offered = rows
        return Observation(text=text, frames=self._frames(rows))

    def _frames(self, rows: list[dict[str, Any]]) -> list[Frame]:
        """The images, in the order their captions are listed.

        Street views first, then any pedestrian lights, because that is the order
        ``render_photographs`` writes the captions in and a model matching
        captions to images by position must not be misled by the harness.
        """
        if not self.with_images:
            return []
        # Renamed on the way out. The album names its files after what is in
        # them, which would let a policy read ``road_block`` off the path instead
        # of the picture. See ``frame_alias``.
        frames = [
            Frame(f"[{row['street']}, {row['heading']}] the view down it",
                  self.frames_seen.alias(row["image"]))
            for row in rows if row.get("image")
        ]
        # The caption key a lamp frame is found by, and the leg it governs.
        # A sender that drops frames needs this to tell the environment which
        # lamps actually arrived; without it the environment charges for a
        # light the model was never shown.
        self.lamp_legs = {
            f"{row['street']}, {row['heading']}": f"{self.env.node_id}|{row['node']}"
            for row in rows if row.get("signal_image")
        }
        frames += [
            Frame(f"[light: {row['street']}, {row['heading']}] the pedestrian "
                  "light for that crossing",
                  self.frames_seen.alias(row["signal_image"]))
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

    STUCK_REPEATS = 4

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
        except TruncatedReply as error:
            # The generation budget ran out, not the model's competence. It is
            # counted and answered, but it does not spend the three-strike
            # format budget: on the held-out set that budget was ending one
            # episode in six, every one at zero, on a fault of the harness's
            # own configuration.
            self.spend.truncated_replies += 1
            turn.status, turn.error = "truncated_reply", str(error)
            self.feedback = TRUNCATED_TEMPLATE.format(error=error)
            self.run.turns.append(turn)
            if budget_exceeded(self.spend, self.budgets) == "repeated_truncations":
                self._finish("repeated_truncations")
            return turn
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
        if action.unfenced:
            self.spend.unfenced_actions += 1
        turn.thought, turn.action = action.thought, action.render()
        turn.tool_kind = TOOLS_BY_NAME[action.tool].kind.value

        before = self.env.sim_seconds
        self._leaving = self.env.node_id
        outcome = self._execute(action)
        turn.sim_seconds = self.env.sim_seconds - before
        turn.reward = outcome.reward
        turn.status = "accepted" if outcome.ok else "rejected"
        if not outcome.ok:
            turn.error = outcome.code
        # A refusal carries the reason and then asks for the reasoning back.
        # REJECTED_TEMPLATE existed and was never wired to anything, so a
        # refused courier got the bare sentence and nothing else -- and on the
        # held-out set it answered by repeating the identical call with no
        # THOUGHT at all until the session ended stuck.
        self.feedback = (REJECTED_TEMPLATE.format(reason=outcome.message)
                         if not outcome.ok else outcome.message)

        self.spend.steps += 1
        self.spend.tool_calls += 1
        self.spend.sim_seconds += turn.sim_seconds
        self.run.total_reward += outcome.reward

        if outcome.moved:
            self._record_arrival()
        self._remember(action, outcome)
        turn.memory = self.memory.to_dict()
        self.run.turns.append(turn)

        if not outcome.ok and self._leaving is not None and action.args:
            # Against the junction it was refused at, so the marker appears on
            # the very line that will be offered again next turn.
            self.memory.refused(
                self._leaving,
                *self._street_as_offered(
                    str(action.args[0]),
                    str(action.args[1]) if len(action.args) > 1 else ""),
                outcome.code or "refused")
        if not outcome.ok and turn.action == self._last_refused:
            self._repeats += 1
        else:
            self._repeats = 0 if outcome.ok else 1
        self._last_refused = None if outcome.ok else turn.action

        if outcome.finished or self.env.shift_over:
            self._finish("delivered" if outcome.finished else "shift_over")
        elif self._repeats >= self.STUCK_REPEATS:
            # The same refused call, over and over, with the refusal unchanged.
            # This is not a courier working a problem: on 22 measured episodes
            # it was 47 turns -- 6.8% of every turn spent -- and one episode
            # made the identical refused call 28 times in a row while its own
            # reasoning said "the order cannot be collected". Nothing in the
            # remaining turns can differ, because nothing in the world has.
            #
            # It ends the session, not the city: the environment is not allowed
            # to disappear because the courier pressed the wrong button. And it
            # cannot be gamed, because a shift that stops early earns nothing
            # extra -- stopping is never better than carrying on.
            self._finish("stuck")
        else:
            stopped = budget_exceeded(self.spend, self.budgets)
            if stopped:
                self._finish(stopped)
        return turn

    def _note(self, text: str = ""):
        """Write a line in the notebook. Costs a turn, and not a second of clock.

        The tool has existed in the menu-building code since the beginning and
        has never been runnable -- it sat in ``UNIMPLEMENTED_TOOLS`` because no
        executor existed anywhere, so the one place a courier could put a fact
        it had worked out was unreachable. The skills refer to it by name.

        It takes a turn because a turn is a model call, and it does not move the
        clock because writing on your own hand does not take a minute. That
        split is deliberate: the courier can afford to think, but not for free.
        """
        from embodiedbench.runtime.city.courier_env import StepOutcome

        line = " ".join(str(text).split())
        if not line:
            return StepOutcome(
                ok=False, code="empty_note",
                message="A note needs something written on it.")
        self.memory.write(line)
        return StepOutcome(ok=True, code="noted",
                           message=f"Written down: {line}")

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

    def _street_as_offered(self, street: str, heading: str) -> tuple[str, str]:
        """The typed street and bearing, as the row the courier was shown.

        Memory must record the same key the next observation will look up:
        the row's name and first-edge compass. What the courier typed can be
        looser than that -- no bearing where the name was unique, "left" for
        the compass, a base name for a suffixed one -- and ``walk_to`` accepts
        all of it, so remembering the typed strings recorded keys that no
        lookup would ever present again.
        """
        from embodiedbench.runtime.city.street_names import match_street

        try:
            row = match_street(self._offered, street, heading or None)
            return str(row.get("street", street)), str(row.get("heading", heading))
        except Exception:  # noqa: BLE001 - unresolvable stays as typed
            return street, heading

    def _remember(self, action: ParsedAction, outcome: Any) -> None:
        self._set_goal()
        if outcome.ok and action.tool in ("walk_to", "follow_street"):
            # Recorded against the junction it was left by, which is where the
            # menu offering it will be shown again. ``leaving`` is captured
            # before the walk in ``step``; without it this would record the
            # junction arrived at, and mark the way *back* as already tried.
            if self._leaving is not None and action.args:
                street, heading = self._street_as_offered(
                    str(action.args[0]),
                    str(action.args[1]) if len(action.args) > 1 else "")
                self.memory.took(self._leaving, street, heading)
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
