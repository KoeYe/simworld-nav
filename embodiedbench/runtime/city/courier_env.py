"""A delivery runtime built on the compiled city, not on the engine's graph.

Every blocker the independent evaluation found traces to the same thing: the
simulation and the description of it were built from different data. The agent
was told a distance to one point and scored at another 12 m away. It was told a
bearing in one angle convention and shown a photograph in a second. It was
offered waypoints on a graph whose edges missed the road by a median of 11 m.
Patching those one at a time keeps the two descriptions in sync only until the
next one drifts.

So this runtime has exactly one source of truth -- the ``RoadNetwork`` the
conversion layer compiles -- and everything the agent reads is derived from it:

  where it stands        a node on the carriageway, on a named street
  where it can go        that node's neighbours, numbered
  the picture it sees    the street-view frame baked for that exact (node, neighbour)
  the address it wants   a door derived from a real building footprint
  the distance quoted    to the node the success check uses, not near it
  the bearing quoted     one convention, shared with the renderer

Because there is one source, a claim in the text cannot disagree with the world:
the number the agent is told to walk to is the number the arrival check tests.
That is also what makes the task deployable -- "walk to 42 Rue de Rivoli and
ring the bell" is an instruction a real courier robot could be given, and the
success condition is standing at a door, not being within tolerance of an
abstract point.

The vendored engine remains the reference implementation for order economics and
is not used here; the two are compared in the benchmark rather than stacked.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodiedbench.compiler.road_network import (
    Address,
    RoadNetwork,
    bearing_deg,
    build_road_network,
)
from embodiedbench.runtime.city.embodiment import Embodiment, Viewpoint
from embodiedbench.runtime.city.embodiment import get as embodiment_for
from embodiedbench.runtime.city.map_image import MapDrawing, render_map
from embodiedbench.runtime.city.street_names import (
    StreetAmbiguous,
    StreetNotHere,
    match_street,
    resolve_relative,
)
from embodiedbench.runtime.city.obstacles import (
    BLOCKED_SECONDS,
    ROAD_BLOCK,
    SLOW_PEDESTRIAN,
    ObstacleField,
    load_visibility as load_obstacle_visibility,
    obstacle_sites,
    site_key as obstacle_site_key,
)

COMPASS = ("north", "north-east", "east", "south-east",
           "south", "south-west", "west", "north-west")
# How a street leaving this junction lies relative to the way the courier is
# already facing. A compass bearing is what a map gives; "on your left" is what
# a person standing on the corner uses, and a courier following spoken
# directions ("turn left onto Rue X") needs the second to act on the first.
# Indexed the same way as COMPASS: 45-degree sectors, clockwise from straight on.
RELATIVE = ("straight ahead", "half right", "on your right", "sharp right",
            "behind you", "sharp left", "on your left", "half left")


class Difficulty:
    """How many jobs are live at once is the difficulty axis. The clock is not.

    The previous ladder claimed order count was the axis and gave each tier its
    own clock multiple -- 6.0, 4.5, 2.6, 1.9 -- and the measurement says the
    claim was false. Holding the multiple fixed and varying only the order
    count, the reference courier scores (25 seeds, delivered/issued):

        multiple   solo    pair   triple   shift
          1.9      52%     58%     63%      71%
          2.6      64%     78%     79%      92%
          4.5     100%     98%     99%     100%

    Order count does not make the task harder. It makes it *easier*, because a
    per-order clock averages over more orders and one bad leg stops deciding the
    episode. Every step of the published 100 / 100 / 79 / 72 ladder came from
    the hand-set multiples and nothing else -- a stopwatch wearing the costume
    of a task demand, which is exactly what the ladder was not supposed to be.

    What genuinely gets harder as a shift lengthens is holding several jobs at
    once. A courier with three live orders cannot walk them one at a time: their
    windows run concurrently, so serving them in the order they arrived means
    the last one is cold before it is collected. The work is to *sequence* --
    to batch stops that lie near each other, and to give up the order the
    dispatcher handed you. That is a real demand and it scales smoothly. With
    one uniform clock and only the depth varying (30 seeds, delivered/issued and
    on-time/issued):

        tier      orders  depth   reference        queue-aware optimum
        solo         1      1     100%   67% ot    100%   100% ot
        pair         2      1      98%   64% ot    100%   100% ot
        triple       3      2      92%   22% ot    100%    84% ot
        shift       10      3      64%    7% ot     97%    55% ot

    Every rung stays solvable -- perfect routing finishes 97-100% of all of them
    -- and every rung leaves room above the reference courier, in deliveries and
    in punctuality both. Raising the shift clock by any amount changes none of
    these numbers, which is the property the old ladder did not have.

    ``ENDLESS`` is the autonomous-courier tier: a fixed hour, a queue that never
    empties because orders are generated on demand rather than drawn from a
    finite list, and profit as the score. There is no order to run out of and no
    ratio to saturate, so a better policy always shows up as more money -- the
    reference courier takes 12.00 an hour and a perfect router 50.88, and
    doubling the hour doubles both.
    """

    SOLO = "solo"
    PAIR = "pair"
    TRIPLE = "triple"
    SHIFT = "shift"
    ENDLESS = "endless"
    ALL = (SOLO, PAIR, TRIPLE, SHIFT, ENDLESS)

    # tier: (orders in the shift, how many may be live at once)
    #
    # ``0`` orders means unbounded: the dispatcher keeps handing out work for as
    # long as the clock runs. Only ENDLESS uses it.
    #
    # The lower rungs are deliberately shallow: a tier the competent fail is a
    # broken tier, and solo/pair exist to prove an agent can read an address,
    # find a street and recognise a door at all. Depth rises from TRIPLE, so
    # what separates policies is scheduling rather than luck.
    SPEC: dict[str, tuple[int, int]] = {
        SOLO: (1, 1),
        PAIR: (2, 1),
        TRIPLE: (3, 2),
        SHIFT: (10, 3),
        ENDLESS: (0, 3),
    }
    ENDLESS_SECONDS = 3600.0

    @classmethod
    def order_count(cls, tier: str) -> int:
        return cls.spec(tier)[0]

    @classmethod
    def queue_depth(cls, tier: str) -> int:
        return cls.spec(tier)[1]

    @classmethod
    def spec(cls, tier: str) -> tuple[int, int]:
        if tier not in cls.SPEC:
            raise ValueError(f"unknown difficulty {tier!r}; expected {cls.ALL}")
        return cls.SPEC[tier]


class Condition:
    """How much the world tells the agent, as a difficulty ladder.

    The three rungs are not arbitrary knobs: they remove, in order, the crutches
    that let a text-only policy succeed without looking at anything.

    ``full``     signs, house numbers and a phone that gives bearing and range.
                 The reference courier solves this from text alone, which is why
                 it establishes a solvability floor and nothing more.
    ``no_phone`` the phone is gone. The agent must read street signs, remember
                 where it has been, and follow house numbers -- the way a courier
                 works in a district they do not know.
    ``visual``   **declared, not validated -- do not report scores on it.**
                 The intent was to move house numbers out of the text so the
                 photograph carried the only copy. Inspecting the renders killed
                 that: the CityCore facades have no legible door numbers, so the
                 condition is not vision-dependent, it is impossible. The
                 reference courier scores 0/20, which measures the absence of
                 information rather than the absence of perception.

                 What the renders *do* distinguish is storefronts -- the map
                 carries 18 restaurants and 11 stores with a ``poi_type``, and
                 their shopfronts are visible and signed. A sound visual rung
                 would ask the courier to find the restaurant by its frontage.
                 That needs a vision policy to establish it is solvable, so it
                 stays unvalidated until one has, rather than being shipped as a
                 hard condition and quietly inflating the difficulty range.

    Stating them this way keeps the benchmark honest about what it is measuring:
    a score on ``full`` is a claim about planning, a score on ``no_phone`` is a
    claim about search and memory, and no rung yet supports a claim about
    perception.
    """

    FULL = "full"
    NO_PHONE = "no_phone"
    VISUAL = "visual"
    ALL = (FULL, NO_PHONE, VISUAL)
    # The rungs whose solvability has been demonstrated. A benchmark run should
    # report these; VISUAL is present so the work is not lost, not so it can be
    # scored.
    # Round-2 measurement put the reference courier at 1/10 on NO_PHONE, not the
    # 30% an earlier and looser floor accepted. A rung nobody has solved cannot
    # be reported, so it joins VISUAL as declared-but-unvalidated until a policy
    # clears it.
    VALIDATED = (FULL,)


class Stride:
    """How far one ``walk_to`` carries. The same city at two resolutions.

    The compiled carriageway carries a waypoint every 18 m, which is the right
    spacing for a camera and the wrong one for a decision. Measured over 25
    recorded runs, a single delivery cost the reference courier between 75 and
    104 turns; 47% of them were ``walk_to`` calls pressing the same button down
    the same street, and 49% were phone lookups in between. A courier standing on
    a corner does not make eighteen decisions to walk one block. They make one.

    ``waypoint``  one call, one waypoint. Nothing between the courier and the
                  ground: every 18 m is a place to stop, look and change its
                  mind. This is what the trajectories, the albums and every
                  measurement so far were taken at, and it stays the default.
    ``block``     one call, one block -- along the named street until a choice
                  exists: a side turning, a fork, a dead end, a crossing with a
                  light the courier can read, a door it was sent to, or something
                  in the way. About 12 to 25 turns per delivery instead of 90.

    Neither is a simplification of the world. The same metres are walked, the
    same seconds spent, the same photographs shown, the same lights obeyed and
    the same obstacles hit. What differs is how often the courier is asked. They
    measure different things, which is why both are kept: the waypoint stride
    asks whether a policy can follow a street, the block stride whether it can
    plan a route, and running only the first is how a navigation benchmark ends
    up mostly measuring patience.
    """

    WAYPOINT = "waypoint"
    BLOCK = "block"
    ALL = (WAYPOINT, BLOCK)


# Standing this close to a door counts as being at it. A doorway is about a
# metre wide and a courier stops on the pavement outside, so a few metres is
# generous without being meaningless. It is quoted to the agent, not hidden.
ARRIVAL_TOLERANCE_CM = 800.0
WALK_SPEED_CM_S = 140.0          # a brisk walk, 1.4 m/s
# Signal timing follows the vendored DeliveryBench convention exactly
# (vlm_delivery/utils/traffic_lights.py signal_state_for_axis): the phase is one
# minute, and on odd minutes the south-north axis is red while east-west is
# green. Matching it rather than inventing a period is what lets the light
# frames the procgen maps already carry -- baked as yaw_000_red.png /
# yaw_000_green.png -- be read by this runtime unchanged, and keeps a score here
# comparable with one from the reference environment.
SIGNAL_PHASE_S = 60.0
# A light governs a crossing within this radius (their DEFAULT_CONTROL_RADIUS_CM).
SIGNAL_CONTROL_RADIUS_CM = 650.0
# Crossing against the light costs time as well as reward. Time is the honest
# unit -- the vendored default is 15 s -- because it makes the violation trade
# against the deadline the way it does for a real courier, instead of being a
# flat fee a policy can ignore.
#
# 15 s was the wrong number, and the direction of the error matters: the phase is
# 60 s, so waiting out a red costs a uniform 0-60 s, mean 30 s. At 15 s crossing
# on red was *strictly cheaper on the clock than obeying the light*, every time,
# and the only thing arguing the other way was a reward term ``summary()`` did
# not even report. A sighted policy that used its eyes was slower than a blind
# one. The charge has to exceed the expected wait or looking cannot pay.
#
# 45 s was still not enough, and the tier that showed it is the one the whole
# design points at. Crossing on red is only taken half the time -- the light is
# green the other half -- so the expected charge is half the penalty, while
# obeying the light costs the expected wait (half of the 60 s phase, so 30 s,
# halved again because half the crossings are green: 15 s) *plus a turn*. At 45 s
# the two are 22.5 s against 15 s and a turn, which is a rounding error, and on
# ENDLESS -- where the score is money against a fixed hour, so a turn spent
# waiting is a delivery not made -- the courier that read every lamp earned
# 4.75/h against 5.17/h for the one that read none. The mechanic the photographs
# exist for was actively unprofitable.
#
# Swept on 20 seeds, blind reference against the same policy allowed to read the
# lamp (delivered %, and profit per hour on ENDLESS):
#
#     penalty   SOLO          PAIR          ENDLESS
#       45 s    85 / 95       90 / 97.5     5.17 / 4.75   looking loses
#       75 s    80 / 95       85 / 97.5     4.15 / 4.75   looking wins everywhere
#      105 s    80 / 95       85 / 97.5     3.22 / 4.75
#
# 75 s is where the sign flips and the sighted courier is unchanged by it -- 95%,
# 97.5%, 4.75/h at every setting, because it never pays the charge. Only the
# courier that does not look is worse off, which is what a penalty for not
# looking is supposed to mean.
RED_CROSSING_PENALTY_S = 75.0
RED_CROSSING_PENALTY = 1.0
# Charging for the light requires the light to be *visible*. The album currently
# bakes one static frame per approach, so no frame shows the live phase, and the
# colour appears in no text either. Penalising it anyway made the score
# anti-correlated with success: the reference courier delivers 10/10 and scores
# -4.4 to -16.4, because 17 unavoidable violations at -1.0 swamp +1.6 of
# delivery credit. A metric that punishes information the environment withholds
# measures nothing, so the charge is off until time-matched red/green frames are
# served -- at which point this becomes True and the penalty is earned.
SIGNAL_FRAMES_AVAILABLE = False
# The same argument, one mechanic over: an obstacle is charged for exactly where
# the album can show it. ``obstacles.py`` holds the placement rule and the gate;
# what lives here is only how the runtime spends the two currencies on it.
OBSTACLE_FRAMES_AVAILABLE = False
# The sidecar that says *which* approaches actually show a lamp.
#
# Having a signal album is not the same as being able to see the light, and
# treating them as the same brought back the exact defect the flag above was
# added to kill. The Paris bake renders 347 signalised approaches in both phases,
# but the camera looks along the street the courier is about to take and the
# lamp is behind or beside it on most corners: comparing each red/green pair for
# lamp-coloured pixels finds a switching lamp on 55 of 347 approaches (15.9%),
# and a blind visual sample of twelve approaches found the light readable in at
# most two. Charging on all 347 is charging for an invisible light again, one
# layer down -- an oracle courier takes 43 violations a shift on routes where it
# could have seen and avoided six.
#
# So the album must *declare* what it shows. ``signal_visibility.json`` lists the
# approaches whose two frames differ in a lamp; anything absent from that list is
# an approach where the courier cannot see the light, and is not charged.
SIGNAL_VISIBILITY_FILE = "signal_visibility.json"
# A door number is legible from about this far along the pavement. Beyond it the
# courier is being told about a building it cannot see.
# A number is only worth printing if standing here counts as arriving. Reading
# doors 3x further than the arrival tolerance produced 76.2% false arrivals and a
# reproducible 14-turn livelock: the agent was told "numbers 6-8" and collect()
# refused every turn because door 7 was 18 m off. Tied to the tolerance so the
# two cannot drift apart again.
READABLE_NUMBER_CM = ARRIVAL_TOLERANCE_CM
# A delivery leg. 250 m each way is about four minutes' walking plus handling,
# so a ten-order shift fits an hour with room to make mistakes.
MAX_ORDER_WALK_CM = 25000.0
# ...except it did not, because only the *delivery* leg was bounded. The walk
# from the last drop-off to the next pickup was drawn without any limit and came
# out at a median 320 m and a maximum 1148 m, so the real leg -- approach plus
# delivery -- ran to a median 530 m. A shortest-path courier with perfect
# knowledge, no wrong turns and no time spent looking needed a mean of 76.6
# minutes (60.9-89.3 across seeds 0-19) to work ten orders, and delivered 6.95 of
# them inside the hour. Ten out of ten was not merely hard, it was arithmetically
# impossible on every seed, which makes ``delivered / 10`` a metric no policy can
# score well on and no policy can be compared by.
MAX_APPROACH_WALK_CM = MAX_ORDER_WALK_CM
# How much of the shift the generated orders are allowed to consume at optimal
# play, when a clock is *imposed from outside*. Below 1.0 so a competent courier
# finishes the list and a wandering one does not.
#
# It applies to nothing on the difficulty ladder, and saying so is the point:
# the tiers derive their clock from the list they drew, so cutting the list to
# fit that clock is circular. It silently did nothing on every tier anyway --
# ``_make_orders`` read ``self.shift_seconds`` before ``reset`` had computed it,
# so the budget was always ``None``. A constant whose effect is zero is worse
# than no constant, because the comment above it reads as a guarantee.
SHIFT_FILL = 0.85
# An order is never allowed to be a formality. Only the *delivery* leg had a
# floor, so 2% of chained orders had the next pickup at the very node the last
# drop-off used -- an approach of 0 m, an order that begins with the courier
# already standing at the door.
MIN_APPROACH_WALK_CM = 3000.0
# Two ends of every job, thirty seconds each: the time a courier spends at a
# door that is not spent walking. Named because the clock is derived from it and
# a number that sets the clock should not be a literal buried in three places.
HANDLING_SECONDS = 30.0
# One clock rule for the whole ladder: the time a courier who never puts a foot
# wrong would need for the whole list, times this.
#
# It is uniform on purpose. Per-tier multiples (6.0 / 4.5 / 2.6 / 1.9) were the
# entire published difficulty ladder -- see ``Difficulty`` -- and a stopwatch is
# not a task demand. With one multiple the tiers become genuinely comparable:
# the same proportional room everywhere, so a difference between tiers is a
# difference in the work. 4.5 is set from measurement, not taste: at 4.5 the
# reference courier finishes every order of the shallow tiers (100% and 98%),
# which is what "an easy tier must be near-perfect" requires, and the score is
# insensitive to the multiple thereafter -- 3.5 and 4.5 give the same ladder to
# within noise, because at that point the clock is no longer what binds.
# 4.5 was that number for a city with nothing in the way. Obstacles moved it, and
# the direction is the informative part: a barrier lengthens the *optimal* route
# by 1.21x, and this multiple scales with the optimum, so if the reference
# courier's overhead scaled the same way nothing would need to change. It does
# not. Its overhead is proportional to the distance it actually walks, and going
# round closures it walks 2.0x the optimum instead of 1.66x, so the same multiple
# bought it proportionally less room. Measured on 20 seeds with the obstacle
# album live:
#
#     multiple   solo (blind / sighted)   pair (blind / sighted)
#       4.5           65% / 65%               85% / 90%
#       6.75          85% / 95%               90% / 98%
#       9.0           85% / 95%               90% / 98%
#      13.5           85% / 95%               90% / 98%
#
# Two things are worth reading off that table. The easy tiers come back to
# near-perfect, which is what they are for. And past about 6.75 the clock stops
# being what decides anything at all -- doubling and trebling it again changes
# not one delivery, because what binds then is whether the courier can find the
# door, which is the demand the ladder is supposed to measure. 7.0 sits inside
# that flat region rather than on its edge.
#
# ---------------------------------------------------------------------------
# THAT TABLE DOES NOT REPRODUCE, and the half of it that fails is the half the
# choice of 7.0 rests on. Re-measured with this code -- same policy, same
# albums, 12 seeds, both strides, mean over solo and pair:
#
#     multiple   blind    sighted   gain   cells where sight wins
#       2.0      27.3%     36.9%    +9.6           4/4
#       2.6      39.4%     50.1%   +10.8           4/4
#       3.5      52.2%     63.4%   +11.2           4/4
#       4.5      66.0%     69.1%    +3.2           2/4
#       6.75     69.8%     72.9%    +3.1           2/4
#       9.0      69.8%     72.9%    +3.1           2/4
#
# The saturation claim holds exactly: 6.75 and 9.0 are identical to the
# delivery. What does not hold is "sight separates there". It separates by
# +3.1 points at 6.75 and by +11.2 at 3.5, and at 7.0 the sighted arm is level
# with or behind the blind one on half the cells -- perfect recognition removes
# every barrier collision and every red crossing and buys nothing, because at
# seven times optimal a courier can walk into every closure on the map and
# still finish.
#
# So the multiple is 3.5, chosen off the second table rather than the first.
# It is where the gain from looking is largest, and it is the largest multiple
# at which looking wins in every cell tested rather than half of them.
#
# What that costs, stated plainly because it is a real cost: the blind floor on
# solo and pair falls from about 70% to about 52%, so those tiers are no longer
# ones a text-only courier passes comfortably. The ``Difficulty`` docstring
# describes them as rungs that exist "to prove an agent can read an address,
# find a street and recognise a door at all", and at 3.5 a courier that cannot
# see fails about half of them -- which is the point: the half it fails are the
# ones with something in the way. A *sighted* agent still passes comfortably,
# and that is the property the ladder should have had all along.
#
# ``docs/review/tools/clock_sweep.py`` regenerates the second table, and
# ``docs/review/tools/reference_table.py`` regenerates the figures in
# docs/RUNNING.md. Both must be re-run if this number moves again.
TIME_BUDGET_MULTIPLE = 3.5
# When the dispatcher gives up on an order, as a multiple of the window it
# quoted. A job nobody can be bothered to finish has to stop being worth points,
# or "ignore the deadline" is a free strategy: before this, a delivery three
# hours late paid exactly what an on-time one did.
ORDER_EXPIRY_MULTIPLE = 3.0
# A late delivery still gets paid, but not in full. Half, because the two
# degenerate ends are both worse: pay full and the deadline is decorative, pay
# nothing and a courier running late is better off abandoning a bag it has
# already collected -- which is a strategy no dispatcher would design for.
LATE_FEE_FRACTION = 0.5
# What a refused action costs. It was zero on both currencies, and that made
# ``collect()`` a free unlimited rangefinder: standing anywhere, a policy could
# call it, be told "It is 140 m away", and pay neither a turn nor a second --
# the same information ``check_map`` charges a turn and five seconds for. An
# action the benchmark cannot see is an action outside the measurement.
#
# Charging for it was necessary and not sufficient, and the arithmetic says why:
# a block costs 13-26 s to walk, so at 5 s a refusal was *still strictly cheaper
# than moving*. The optimal endgame became walk-probe-walk -- read the sign of
# the change in the quoted distance and you have a gradient oracle for the door,
# needing no photograph, no door number and no street sign. Two independent
# reviewers found it and both used it to finish an episode. So the distance is
# gone from the refusal entirely: a refusal now reports arrival, which is a
# yes/no the courier could get by standing there, and nothing else. Distance to
# an address is what ``check_map`` sells.
REJECTED_ACTION_SECONDS = 5.0


def compass_of(bearing: float) -> str:
    return COMPASS[int((bearing % 360.0) / 45.0 + 0.5) % 8]


def _leading_number(numbers: str) -> int | None:
    """The first house number out of a rendered range like ``"12, 14, … 40"``."""
    head = numbers.split(",", 1)[0].strip()
    return int(head) if head.isdigit() else None


def relative_of(bearing: float, facing: float | None) -> str:
    """Where a bearing lies relative to the way the courier is facing."""
    if facing is None:
        return ""
    return RELATIVE[int(((bearing - facing) % 360.0) / 45.0 + 0.5) % 8]


def turn_word(from_bearing: float, to_bearing: float) -> str:
    """How a route describes the change of heading at a junction."""
    delta = (to_bearing - from_bearing) % 360.0
    if delta < 20.0 or delta > 340.0:
        return "continue"
    if delta < 70.0:
        return "bear right"
    if delta < 160.0:
        return "turn right"
    if delta < 200.0:
        return "turn back"
    if delta < 290.0:
        return "turn left"
    return "bear left"


def movement_axis(bearing: float) -> str:
    """Which crossing axis a heading belongs to."""
    return "north-south" if compass_of(bearing) in ("north", "south") else "east-west"


def signal_state_for_axis(seconds: float, axis: str) -> str:
    """Deterministic two-phase signal, matching the vendored controller.

    Reimplemented rather than imported so this runtime carries no import-time
    dependency on the vendored engine, and cross-checked against it by test:
    ``test_signal_phase_matches_the_vendored_controller`` fails if the two ever
    disagree. Copying the convention without pinning it would drift silently and
    make every baked light frame wrong by one phase.
    """
    minute = int(max(0.0, float(seconds)) // 60.0)
    south_north_red = bool(minute % 2 == 1)
    normalised = str(axis or "").strip().lower().replace("_", "-")
    if normalised in {"south-north", "north-south", "vertical", "sn", "ns"}:
        return "red" if south_north_red else "green"
    return "green" if south_north_red else "red"


def signal_state(node_id: str, bearing: float, sim_seconds: float) -> str:
    """The pedestrian light facing a courier about to cross, at this moment.

    The whole city switches together, as it does in the reference environment --
    an earlier version offset each junction by a hash of its id, which looked
    more realistic and made every baked frame disagree with the engine.

    Nothing about this reaches the observation text. The colour of a light is
    something you *see*, and making it readable in words would hand a text-only
    policy the one signal this environment has that genuinely requires looking.
    """
    return signal_state_for_axis(sim_seconds, movement_axis(bearing))


@dataclass
class Order:
    """One job, expressed entirely in addresses a person could read out."""

    index: int
    pickup: Address
    dropoff: Address
    fee: float
    deadline_s: float
    picked_up: bool = False
    delivered: bool = False
    delivered_at_s: float | None = None
    # When this job was handed over. A deadline measured from clock-in makes one
    # slow first order doom every later one -- the reference courier went 0/10
    # on-time for exactly that reason -- and no real dispatcher works that way.
    issued_at_s: float | None = None
    # Given up on: past ``ORDER_EXPIRY_MULTIPLE`` windows and taken back by the
    # dispatcher. It pays nothing and it cannot be delivered.
    expired: bool = False
    # What it actually paid. Full fee on time, ``LATE_FEE_FRACTION`` of it late,
    # nothing if it expired -- so the ledger and the deadline are one number
    # rather than two that can disagree.
    paid: float = 0.0

    def due_at(self, fallback_issue_s: float = 0.0) -> float:
        return (self.issued_at_s if self.issued_at_s is not None else fallback_issue_s) + self.deadline_s

    def expires_at(self, fallback_issue_s: float = 0.0) -> float:
        issued = self.issued_at_s if self.issued_at_s is not None else fallback_issue_s
        return issued + self.deadline_s * ORDER_EXPIRY_MULTIPLE

    def minutes_left(self, now_s: float) -> float:
        return (self.due_at() - now_s) / 60.0

    @property
    def live(self) -> bool:
        """Issued, not delivered, not given up on -- a job the courier still has."""
        return self.issued_at_s is not None and not self.delivered and not self.expired

    @property
    def target(self) -> Address:
        return self.dropoff if self.picked_up else self.pickup

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "pickup": self.pickup.text, "dropoff": self.dropoff.text,
            "fee": round(self.fee, 2), "deadline_s": round(self.deadline_s, 1),
            "picked_up": self.picked_up, "delivered": self.delivered,
            "expired": self.expired, "paid": round(self.paid, 2),
            "issued_at_s": (None if self.issued_at_s is None else round(self.issued_at_s, 1)),
        }


@dataclass
class StepOutcome:
    """What one tool call did."""

    ok: bool
    message: str = ""
    code: str = ""
    reward: float = 0.0
    sim_seconds: float = 0.0
    moved: bool = False
    finished: bool = False
    # Reported by the world, so a policy can account for its own effort without
    # computing distances from coordinates it should not have.
    walked_m: float = 0.0
    # A picture the tool produced, as SVG. Only the phone's map has one, and it
    # is deliberately not called an "image": the photographs come from the
    # world and this comes from the map, and a harness that put them in one list
    # would let a courier believe the phone can see the street.
    drawing: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class CourierEnv:
    """The delivery world, over a compiled road network."""

    def __init__(
        self,
        network: RoadNetwork,
        *,
        seed: int = 0,
        order_count: int = 1,
        album_root: Path | None = None,
        signal_album_root: Path | None = None,
        obstacle_album_root: Path | None = None,
        deadline_slack: float = 2.5,
        shift_seconds: float | None = None,
        condition: str = Condition.FULL,
        enforce_signals: bool | None = None,
        enforce_obstacles: bool | None = None,
        difficulty: str | None = None,
        queue_depth: int | None = None,
        stride: str = Stride.WAYPOINT,
        embodiment: str | Embodiment | None = None,
        pavement_album_root: Path | None = None,
        pavement_obstacle_album_root: Path | None = None,
        served_long_edge: float | None = None,
    ):
        # A tier sets the list, the queue and the clock; they are not
        # independent, and the tier is the only place they are chosen together.
        # An explicit depth still wins, because the claim "the depth is what
        # makes this hard" is only testable if the depth can be held at 1 with
        # everything else unchanged.
        if difficulty is not None:
            order_count, tier_depth = Difficulty.spec(difficulty)
            if queue_depth is None:
                queue_depth = tier_depth
            # The clock is derived in reset(), once the orders exist.
            shift_seconds = None
        if queue_depth is None:
            queue_depth = 1
        # What is doing the delivering. Speed, stamina, the cost of stopping and
        # -- the one that matters most for transfer -- which viewpoint's album it
        # is entitled to see. See ``embodiment.py``.
        self.embodiment = embodiment_for(embodiment)
        self.embodiment.require_defined()
        self.pavement_album_root = (
            Path(pavement_album_root) if pavement_album_root else None
        )
        self.pavement_obstacle_album_root = (
            Path(pavement_obstacle_album_root) if pavement_obstacle_album_root else None
        )
        self.difficulty = difficulty
        # How many jobs the dispatcher lets the courier hold at once. This is
        # the difficulty axis: their windows run concurrently, so a deep queue
        # has to be *sequenced* rather than worked in the order it arrived.
        self.queue_depth = max(1, int(queue_depth))
        # A tier of 0 orders means the dispatcher never runs out -- ENDLESS.
        self.unbounded = difficulty is not None and order_count == 0
        # Off unless the album can show the light. Overridable so the mechanic
        # stays testable before the frames land.
        # Charge for the light exactly when the light can be seen. Passing a
        # signal album is the evidence; without one the penalty would punish
        # information the environment withholds, which is what made the score
        # anti-correlated with delivery before this existed.
        if enforce_signals is None:
            enforce_signals = signal_album_root is not None or SIGNAL_FRAMES_AVAILABLE
        self.enforce_signals = bool(enforce_signals)
        if enforce_obstacles is None:
            enforce_obstacles = obstacle_album_root is not None or OBSTACLE_FRAMES_AVAILABLE
        self.enforce_obstacles = bool(enforce_obstacles)
        if condition not in Condition.ALL:
            raise ValueError(f"unknown condition {condition!r}; expected one of {Condition.ALL}")
        self.condition = condition
        if stride not in Stride.ALL:
            raise ValueError(f"unknown stride {stride!r}; expected one of {Stride.ALL}")
        self.stride = stride
        self.network = network
        self.seed = seed
        self.order_count = order_count
        # The album this body is entitled to. A person on foot is on the
        # pavement and a rider is in the carriageway, and showing either the
        # other one's frames trains a policy to recognise a world it will never
        # stand in. Only the carriageway album has been baked; until the
        # pavement bake lands, a walking courier is served carriageway frames and
        # ``viewpoint_served`` in the summary says so, so the debt is measurable
        # rather than silent.
        if (self.embodiment.viewpoint == Viewpoint.PAVEMENT
                and self.pavement_album_root is not None):
            album_root = self.pavement_album_root
            self.viewpoint_served = Viewpoint.PAVEMENT
            # The obstacle album has to move with it. Serving footway frames for
            # clear streets and centreline frames wherever an obstacle stands
            # makes the *viewpoint itself* announce the hazard: measured at 19.5%
            # of a walker's frames, every one of them an obstacle. That is the
            # filename leak again in a different disguise, so the two albums are
            # switched together or not at all.
            if pavement_obstacle_album_root is not None:
                obstacle_album_root = pavement_obstacle_album_root
            elif obstacle_album_root is not None:
                raise ValueError(
                    "a pavement album was given without a pavement obstacle "
                    "album: the obstacle frames would come from the carriageway "
                    "and the change of viewpoint alone would tell the policy an "
                    "obstacle is there. Pass pavement_obstacle_album_root, or "
                    "drop obstacle_album_root."
                )
        else:
            self.viewpoint_served = Viewpoint.CARRIAGEWAY
        self.viewpoint_matches_embodiment = (
            self.viewpoint_served == self.embodiment.viewpoint
        )
        self.album_root = Path(album_root) if album_root else None
        # Frames baked in both signal states, one pair per signalised approach.
        self.signal_album_root = Path(signal_album_root) if signal_album_root else None
        # Frames baked with something in the way, one per (approach, kind).
        self.obstacle_album_root = Path(obstacle_album_root) if obstacle_album_root else None
        self.deadline_slack = deadline_slack
        # A shift runs on a clock, not a step count. Ten deliveries in an hour is
        # a courier's day; a step cap measures how chatty the policy is instead.
        self.shift_seconds = shift_seconds

        self.streets = {s.index: s for s in network.streets}
        self.addresses_by_street: dict[str, list[Address]] = {}
        for address in network.addresses:
            self.addresses_by_street.setdefault(address.street_name, []).append(address)
        for values in self.addresses_by_street.values():
            values.sort(key=lambda a: a.number)

        self.node_id: str = ""
        self.arrived_from: str | None = None
        self.sim_seconds: float = 0.0
        self.orders: list[Order] = []
        self.earnings: float = 0.0
        self.finished = False
        self.red_crossings: int = 0
        # Which lamps the sender managed to put in front of the model this
        # turn. None means "no one is limiting them".
        self.signals_shown: set[str] | None = None
        self.waits_at_red: int = 0
        # Long-horizon is measured in both currencies: a policy can be cheap in
        # turns and slow on the clock, or the reverse, and only reporting one
        # hides half of what went wrong.
        self.turns: int = 0
        # How far the courier actually walked, against how far a perfect one
        # would have. Only the oracle used to count this, so route quality was
        # unmeasurable for every other policy -- and route quality is most of
        # what separates a good courier from a lucky one.
        self.walked_cm: float = 0.0
        self.optimal_seconds: float = 0.0
        self.stamina: float = float(self.embodiment.stamina or 0.0)
        self.rests: int = 0
        self.optimal_walk_cm: float = 0.0
        # Actions the world refused. A policy that spends a third of its turns
        # walking into walls is not the same as one that spends none, and the
        # score alone cannot tell them apart.
        self.rejected_actions: int = 0
        self.expired_count: int = 0
        # Walking into a barrier, and squeezing past a congested pavement. Both
        # are time a courier that read the photograph never spends, so they are
        # the two numbers that say whether looking paid.
        self.blocked_attempts: int = 0
        # Edges this courier has personally walked into and been turned back
        # from. Not the phone's knowledge -- the courier's own eyes.
        self.witnessed_blocks: set[tuple[str, str]] = set()
        self.slow_passages: int = 0
        # What is on the phone's screen. A map app does not switch off when you
        # put the phone in your pocket: the route it drew stays drawn and the
        # dot showing where you are keeps moving. Showing the map for one turn
        # made asking for it a tax -- playing an episode by hand, 45% of the
        # turns went on ``navigate()``, 37 lookups at 15 s each, nine minutes of
        # an hour spent standing still reading a phone.
        #
        # The *route* is frozen at the moment it was asked for and the *position*
        # is live. That is the whole design: walking is free to watch, but a
        # better route costs a call, so a courier that wanders off the blue line
        # can see that it has and must decide whether the answer is worth 15 s.
        self.screen_route: list[tuple[float, float]] = []
        self.screen_target: Address | None = None
        # The last door number read on the street the courier is on, so the
        # observation can say which way the numbers run rather than making the
        # policy remember across turns.
        self._last_numbers: tuple[str | None, int | None] = (None, None)
        # Nothing here records what the courier knows about barriers, and that
        # is the design. ``report_blocked`` used to let the courier tell its
        # phone a street was shut, and the phone would route round it -- which
        # is not a thing a rider does. You see a skip, you take the next street;
        # you do not open the map app and file a report. The phone is a survey
        # and stays permanently blind, so the route keeps pointing through the
        # barrier and the courier has to overrule it from what it can see. That
        # is what makes looking necessary rather than merely rewarded.

        self.signalised = network.signalised_nodes()
        # The long edge the harness actually sends, if it says. The album
        # certifies legibility after a resize to MODEL_LONG_EDGE_PX; a harness
        # that downscales further is not looking at the frame that was
        # certified, and the gate has to be asked again at the size the policy
        # gets. Silence means the album's own answer stands.
        self.served_long_edge = (
            float(served_long_edge) if served_long_edge else None)
        self.visible_signals = self._load_signal_visibility()
        # Where an obstacle can stand on this map. Seed-free and computed once:
        # it is a property of the road network, which is what lets one bake of
        # the album serve every episode of every seed.
        self._neighbours = {n: sorted(node.neighbours) for n, node in network.nodes.items()}
        self._obstacle_sites = obstacle_sites(self._neighbours, network.map_name)
        self._visible_obstacles = load_obstacle_visibility(self.obstacle_album_root)
        self.obstacles = ObstacleField()


    def _load_signal_visibility(self) -> set[str] | None:
        """Which approaches the album can actually show a lamp on.

        ``None`` means the album made no claim, and then nothing is charged --
        the same default as having no album at all. Silence is not consent: an
        album that does not say what it shows is an album that has not been
        checked, and the failure mode of guessing is a penalty on an invisible
        light, which is the defect this whole gate exists to prevent.
        """
        if self.signal_album_root is None:
            return None
        path = self.signal_album_root / SIGNAL_VISIBILITY_FILE
        if not path.exists():
            return None
        import json

        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        legible = {str(k) for k in data.get("legible", [])}
        legible &= self._readable_at_served_size(data, legible)
        return self._one_lamp_per_lamp(data, legible)

    def _one_lamp_per_lamp(self, data: dict, legible: set[str]) -> set[str]:
        """Stop charging one lamp several times over.

        The map has no lamp objects. Signalised junctions are derived from
        node degree and the bake puts one light mesh at each of them, so a
        junction with four ways out has four photographs of THE SAME LAMP,
        from the same camera, differing only in which phase is lit. The
        environment then gave each approach its own phase from its own bearing
        and charged each one separately -- one lamp treated as four.

        What that did to the courier is worse than the double-counting. Told
        "the lamp for Rue de la Paix" and "the lamp for Rue Cujas", it was
        handed two pictures with identical backgrounds and no way to tell
        which was which; the caption was the only thing distinguishing them.
        Measured on this album: 34 junctions show one lamp to three
        approaches, three show one to four, two show one to five, and only a
        single junction in the map has two genuinely different lamps. Of 130
        charged approaches about 61 were the same lamp counted again.

        So each group of approaches sharing a lamp keeps exactly one -- the
        view where the lamp is largest, which is the one a courier could
        actually read -- and the rest are treated as having no visible lamp at
        all: no frame, no charge. That is the same rule the visibility gate
        already applies, said about a lamp rather than about an album.
        """
        # An album that gives every approach its own lamp says so, and must
        # not be folded back together: composited lamps sit at the same place
        # in every frame, so their boxes coincide although the lamps are
        # genuinely separate. The rendered album's boxes coincide for the
        # opposite reason -- one lamp photographed repeatedly -- and only the
        # album knows which case it is.
        if data.get("lamps_are_per_approach"):
            return legible
        boxes = data.get("lamp_box")
        sizes = data.get("lamp_px")
        if not isinstance(boxes, dict):
            return legible

        def overlap(a: list, b: list) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            wide = max(0, min(ax1, bx1) - max(ax0, bx0))
            tall = max(0, min(ay1, by1) - max(ay0, by0))
            inter = wide * tall
            union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
            return inter / union if union > 0 else 0.0

        def lamp_pixels(key: str) -> int:
            row = (sizes or {}).get(key)
            return int(row[0]) if row else 0

        by_node: dict[str, list[str]] = {}
        for key in legible:
            by_node.setdefault(key.split("|")[0], []).append(key)

        kept: set[str] = set()
        for node, keys in by_node.items():
            groups: list[list[str]] = []
            for key in sorted(keys):
                box = boxes.get(key)
                if not box:
                    groups.append([key])
                    continue
                for group in groups:
                    other = boxes.get(group[0])
                    if other and overlap(box, other) > 0.5:
                        group.append(key)
                        break
                else:
                    groups.append([key])
            for group in groups:
                kept.add(max(group, key=lambda k: (lamp_pixels(k), k)))
        return kept

    # A 2x2 patch after the resize -- the album's own floor, restated here so
    # the runtime is not silently more permissive than the measurement was.
    SERVED_MIN_PIXELS = 4.0

    def _readable_at_served_size(self, data: dict, legible: set[str]) -> set[str]:
        """Of the certified approaches, those still readable at the served size.

        The album records each lamp's area in the frame as baked. Area falls
        with the square of the resize, so a lamp certified at 768 px can be
        under a 2x2 patch by the time a 320 px harness has finished with it --
        on the Paris kerb album that is 34 of 130 approaches. Charging those is
        charging for a light the policy was never sent enough pixels to see,
        which is the same defect the visibility gate exists to prevent, one
        stage further down the pipe.
        """
        sizes = data.get("lamp_px")
        if not self.served_long_edge:
            return legible
        if not isinstance(sizes, dict):
            # Album baked before lamp_px existed: the gate cannot run. Say so once.
            if not getattr(self, "_warned_no_lamp_px", False):
                self._warned_no_lamp_px = True
                import logging

                logging.getLogger(__name__).warning(
                    "served_long_edge=%s but this album has no lamp_px "
                    "metadata; the served-size visibility gate is OFF and "
                    "red-light charging follows the bake resolution. Re-bake "
                    "the album to get served-size gating.",
                    self.served_long_edge,
                )
            return legible
        readable = set()
        for key in legible:
            row = sizes.get(key)
            if not row:
                # Unmeasured. Keep it: the album certified it and this check is
                # a refinement, not a second gate with a different default.
                readable.add(key)
                continue
            px, width, height = row
            scale = min(1.0, self.served_long_edge / max(width, height))
            if px * scale * scale >= self.SERVED_MIN_PIXELS:
                readable.add(key)
        return readable

    def signal_is_visible(self, node_id: str, toward: str) -> bool:
        """Can the courier standing at ``node_id`` see the lamp for this crossing?

        Two gates, and the second exists because the first is not enough. The
        album says which crossings *have* a lamp it can show. The harness then
        decides how many pictures fit in one turn, and when it runs out it drops
        street views and their lamps together -- so a crossing the album can
        show is not necessarily a crossing the courier was shown. Charging on
        the album alone penalises a policy for a lamp that never arrived, which
        is the same defect the album gate was added to prevent, arriving one
        layer further out.

        ``signals_shown`` is set by whatever is doing the sending, each turn.
        Left as None it means "everything the album has", which is right for the
        evaluation harness that sends them all.
        """
        key = f"{node_id}|{toward}"
        if self.signals_shown is not None and key not in self.signals_shown:
            return False
        if self.visible_signals is None:
            return bool(SIGNAL_FRAMES_AVAILABLE)
        return key in self.visible_signals

    def show_only_these_signals(self, keys: "set[str] | None") -> None:
        """Declare which lamps actually reached the model this turn."""
        self.signals_shown = None if keys is None else set(keys)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        rng = random.Random(self.seed)
        # Spawn only where the courier has a real choice. A degree-1 node is a
        # cul-de-sac, and starting in one wastes turns on a decision that is not
        # a decision -- the engine's own spawn was such a node.
        candidates = sorted(
            n for n, node in self.network.nodes.items() if len(node.neighbours) >= 2
        ) or sorted(self.network.nodes)
        self.node_id = rng.choice(candidates)
        self.arrived_from = None
        self.sim_seconds = 0.0
        self.earnings = 0.0
        self.finished = False
        self.stamina = float(self.embodiment.stamina or 0.0)
        self.rests = 0
        self.red_crossings = 0
        self.signals_shown = None
        self.waits_at_red = 0
        self.walked_cm = 0.0
        self.rejected_actions = 0
        self.expired_count = 0
        self.turns = 0
        self.blocked_attempts = 0
        self.witnessed_blocks = set()
        self.slow_passages = 0
        self.screen_route = []
        self.screen_target = None
        self._last_numbers = (None, None)

        # Which sites are live this shift. Deterministic in (map, seed), so a
        # replay of a seed meets the same city; different every seed, so the
        # arrangement cannot be learned once and reused.
        self.obstacles = ObstacleField.generate(
            self._neighbours, self.network.map_name, self.seed,
            sites=self._obstacle_sites,
            visible=self._visible_obstacles if self.enforce_obstacles else None,
        )
        # Kept so ENDLESS can keep drawing after the initial list runs out. The
        # cursor is where the *last drawn* order ended, not where the courier
        # is, so a lazily drawn job chains onto the list exactly as a
        # pre-drawn one would and the two paths cannot diverge.
        self._order_rng = rng
        self._draw_cursor = self.node_id
        self._usable_addresses = [
            a for a in self.network.addresses if a.kerb_node in self.network.nodes
        ]
        self.orders = self._make_orders(rng)
        self.optimal_seconds, self.optimal_walk_cm = self._optimal_route()
        if self.difficulty is not None:
            if self.unbounded:
                self.shift_seconds = Difficulty.ENDLESS_SECONDS
            else:
                # One rule, every tier: the work this seed actually drew, times
                # the room a policy is allowed to waste. See TIME_BUDGET_MULTIPLE
                # for why the multiple is uniform.
                self.shift_seconds = self.optimal_seconds * TIME_BUDGET_MULTIPLE
        self._issue()

    def _optimal_route(self, orders: list[Order] | None = None) -> tuple[float, float]:
        """Seconds and centimetres for a courier who never puts a foot wrong.

        Serving the given list in the given order, chained from the spawn. Used
        two ways: over the whole drawn list it sizes the shift clock, and over
        the orders a run actually delivered it says how much further than
        necessary that run walked. A queue-aware policy can beat it by batching
        stops, which is the point -- it is a reference, not a bound.
        """
        cursor, seconds, walk = self.node_id, 0.0, 0.0
        for order in (self.orders if orders is None else orders):
            # Obstacle-aware, and that is not a detail. The clock every tier
            # derives from this number, so costing the shift on a route the
            # obstacles forbid would make the barriers a *stopwatch* penalty --
            # the tier would get harder because the deadline no longer fits,
            # which is precisely the failure the uniform TIME_BUDGET_MULTIPLE
            # was introduced to end. Priced here, a detour costs what a detour
            # costs and nothing more.
            approach = self.route_cost(cursor, order.pickup.kerb_node, obstacles=True)
            delivery = self.route_cost(order.pickup.kerb_node, order.dropoff.kerb_node,
                                       obstacles=True)
            for legs in (approach, delivery):
                if legs is None:
                    continue
                seconds += legs[0]
                walk += legs[1]
            seconds += 2.0 * HANDLING_SECONDS
            cursor = order.dropoff.kerb_node
        return seconds, walk

    def delivered_optimal(self) -> tuple[float, float]:
        """The perfect route through the stops this run actually served.

        Compared against what it really walked, this is the only route-quality
        number that survives a partial shift and an unbounded queue: a courier
        that delivers three of ten cannot be judged against the walk for ten,
        and ENDLESS has no fixed list to judge against at all.
        """
        done = sorted(
            (o for o in self.orders if o.delivered and o.delivered_at_s is not None),
            key=lambda o: o.delivered_at_s,
        )
        return self._optimal_route(done)

    def _draw_order(self, index: int, rng: random.Random,
                    budget_left: float | None = None) -> Order | None:
        """One job, chained onto the last one drawn.

        Chaining matters: drawing every pickup from the spawn makes each order an
        independent teleport-and-fetch, while a real shift is a sequence where
        the last drop-off is the next journey's start. It is also what keeps a
        deep queue in one district rather than scattered across the map, so the
        sequencing problem the tier poses is a real one and not a forced march.
        """
        usable = self._usable_addresses
        if len(usable) < 2:
            return None
        for _ in range(400):
            pickup, dropoff = rng.sample(usable, 2)
            if pickup.street_name == dropoff.street_name:
                continue
            walk = self.route_length_cm(pickup.kerb_node, dropoff.kerb_node)
            # Long enough to be a journey, short enough that ten fit in an
            # hour. Unbounded walks averaged 20 minutes a leg, which made a
            # ten-order shift a 197-minute queue nobody could finish.
            if walk is None or not (8000.0 <= walk <= MAX_ORDER_WALK_CM):
                continue
            start = self.route_length_cm(self._draw_cursor, pickup.kerb_node)
            # The approach leg counts against the shift exactly like the
            # delivery leg does, so it is bounded at both ends exactly like it:
            # a 0 m approach is a job that begins at the door.
            if start is None or not (MIN_APPROACH_WALK_CM <= start <= MAX_APPROACH_WALK_CM):
                continue
            # A deadline generous enough that a competent courier makes it
            # and a wandering one does not.
            # 2.5x the optimal walk plus a minute of handling. The multiple
            # has to cover perception, not just walking: an agent that has to
            # read signs and check a map to find an address spends real time
            # doing it, and at 1.6x the reference courier went 0/10 on time
            # while still delivering. At 3.0x nothing was ever late, which
            # measures nothing either.
            #
            # Priced on the city as it is *today*, obstacles included, for the
            # same reason the shift clock is. The two clocks were split for a
            # while and the split was the whole regression: the shift budget
            # followed the detours and the per-job window did not, so a job whose
            # only route ran round a barrier arrived inside the shift and outside
            # its own window. That is difficulty arriving as a tighter stopwatch,
            # which is exactly what the ladder is not allowed to do -- measured
            # on SOLO it took the reference courier from 100% to 70%, and the
            # failures were orders expiring with a third of the shift unspent.
            approach = self.route_cost(self._draw_cursor, pickup.kerb_node, obstacles=True)
            delivery = self.route_cost(pickup.kerb_node, dropoff.kerb_node, obstacles=True)
            if approach is None or delivery is None:
                continue
            walking_seconds = approach[0] + delivery[0]
            leg = walking_seconds * self.deadline_slack + 2.0 * HANDLING_SECONDS
            # What this job costs a courier who never puts a foot wrong: the
            # two legs at walking pace plus the handling at each end.
            optimal = walking_seconds + 2.0 * HANDLING_SECONDS
            # Draw again rather than give up: one long pair landing late in
            # the shift must not truncate a list that a shorter pair would
            # still fit. Giving up on the first over-budget draw cost four of
            # twenty seeds more than half their orders.
            if budget_left is not None and optimal > budget_left:
                continue
            self._draw_cursor = dropoff.kerb_node
            return Order(
                index=index, pickup=pickup, dropoff=dropoff,
                fee=self.fee_for(walk),
                # Per-job allowance, not a cumulative clock.
                deadline_s=leg,
            )
        return None

    @staticmethod
    def fee_for(walk_cm: float) -> float:
        """What a job pays, from the distance the parcel travels.

        A flat call-out plus a rate per metre, and the rate is what stops the
        courier from cherry-picking. Measured over the drawn distribution the
        marginal pay of an extra metre is 0.01, and an extra metre costs 0.71 s
        of walking; against the mean earning rate of about 0.016/s that makes a
        long job worth 0.0143/s at the margin against 0.0157/s on average --
        within 10%, so there is no length a profit-seeking courier should prefer
        on rate alone. ``test_no_order_length_is_a_free_lunch`` pins the band.
        """
        return round(3.0 + walk_cm / 100.0 * 0.01, 2)

    def _make_orders(self, rng: random.Random) -> list[Order]:
        """The shift's list, drawn up front. ENDLESS draws its as it goes."""
        orders: list[Order] = []
        # Only an externally imposed clock can bound the list; a derived one is
        # computed *from* the list, so cutting the list to fit it is circular.
        # Saying so is the point -- SHIFT_FILL silently did nothing on every
        # difficulty tier because ``reset`` had not set ``shift_seconds`` yet.
        budget = (self.shift_seconds * SHIFT_FILL
                  if self.shift_seconds is not None and self.difficulty is None else None)
        elapsed, cursor = 0.0, self.node_id
        # Jobs are numbered from 1, like everything else the courier is offered.
        # They used to start at 0 while streets started at 1, and nothing ever
        # showed the number -- the slip says "Job: collect from 8 Rue Bonaparte"
        # and the tool signature says "navigate(job: int = default)". A model
        # reading an interface whose every other index is 1-based writes
        # navigate(1) for its only job, and got "You are not carrying job 1. In
        # hand: 0." Measured on 40 held-out seeds: 23 refusals, 8 of them in one
        # episode, which is a fifth of that episode's forty turns spent on an
        # off-by-one in the interface rather than on the city.
        for index in range(1, self.order_count + 1):
            left = None if budget is None else budget - elapsed
            order = self._draw_order(index, rng, budget_left=left)
            if order is None:
                break
            orders.append(order)
            if budget is not None:
                # Two shortest paths an order, and only the externally clocked
                # case has anything to spend them on.
                elapsed += ((self.route_length_cm(cursor, order.pickup.kerb_node) or 0.0)
                            + (self.route_length_cm(order.pickup.kerb_node,
                                                    order.dropoff.kerb_node) or 0.0)
                            ) / WALK_SPEED_CM_S + 2.0 * HANDLING_SECONDS
            cursor = order.dropoff.kerb_node
        return orders

    # ── the live queue ───────────────────────────────────────────────────────

    def _expire_overdue(self) -> None:
        """Take back every order the dispatcher has given up on.

        Without this a deadline is decorative: a delivery three hours late paid
        exactly what an on-time one did, so "ignore the clock" cost nothing at
        all and the on-time count was a statistic rather than a stake.
        """
        for order in self.orders:
            if order.live and self.sim_seconds > order.expires_at():
                order.expired = True
                self.expired_count += 1

    def _issue(self) -> None:
        """Top the queue back up to its depth, drawing more work if need be.

        Sweeps first, so an order taken back is replaced in the same breath. A
        dispatcher that took an order back and gave nothing in return let a
        policy end its own episode by being slow: the reference courier was
        issued 8.1 of a ten-order shift and a random walker 3 of 10, because a
        queue that emptied through expiry was never refilled and the run simply
        stopped. Being bad at the job must not shorten it.
        """
        self._expire_overdue()
        live = sum(1 for o in self.orders if o.live)
        for order in self.orders:
            if live >= self.queue_depth:
                return
            if order.issued_at_s is None:
                order.issued_at_s = self.sim_seconds
                live += 1
        self._light_the_screen()
        while self.unbounded and live < self.queue_depth:
            order = self._draw_order(len(self.orders) + 1, self._order_rng)
            if order is None:
                return
            order.issued_at_s = self.sim_seconds
            self.orders.append(order)
            live += 1
        self._light_the_screen()

    def _light_the_screen(self) -> None:
        """Put the job in hand on the phone's map, without being asked.

        The direction to walk lives only on the map now -- nothing in any
        sentence the environment speaks says which way to go. That made the map
        load-bearing and left it behind a tool call: the courier had to spend a
        turn on navigate() before it had any direction at all, and on 40
        held-out episodes 25 never called it. Those 25 walked the whole shift
        with no source of direction whatsoever, which is not a hard task, it is
        an unanswerable one.

        A courier who has just been given a job is looking at it on their
        phone. So the screen starts lit, on the job in hand, and navigate()
        goes back to being what it is for: re-centring the route after the
        courier has moved, or pointing the phone at some other address.
        """
        if self.condition in (Condition.NO_PHONE, Condition.VISUAL):
            return
        # Read straight off the list rather than through active_order(), which
        # goes back through live_orders() and _issue() -- and this is called
        # from _issue. The first version recursed until the stack ran out.
        order = next((o for o in self.orders if o.live), None)
        if order is None or order.target is self.screen_target:
            return
        if order.target.kerb_node:
            # Computing the drawing is what stores the route on the screen.
            self.map_drawing(order.target)

    def live_orders(self) -> list[Order]:
        """The jobs in hand right now, oldest first.

        Sweeping and refilling here rather than only on hand-over is what makes
        the queue a property of the clock instead of a property of how often the
        policy happens to succeed.
        """
        self._issue()
        return [o for o in self.orders if o.live]

    # ── geometry ─────────────────────────────────────────────────────────────

    def position(self, node_id: str | None = None) -> tuple[float, float]:
        node = self.network.nodes[node_id or self.node_id]
        return (node.x_cm, node.y_cm)

    def street_of(self, node_id: str) -> str:
        return self.streets[self.network.nodes[node_id].street_index].name

    def edge_street(self, a: str, b: str) -> str:
        """The street an edge runs along.

        Naming a candidate after its *destination node* made 30.1% of edges
        report different names in the two directions -- walking A to B was "Rue
        Monge" and B back to A was "Rue Jacob" -- which contradicts the system
        prompt's own rule that a street keeps its name from one junction to the
        next. When both ends share a street, the edge runs along it; otherwise
        the step is a turn onto the neighbour's street, which is what a courier
        would say.
        """
        street_a = self.network.nodes[a].street_index
        street_b = self.network.nodes[b].street_index
        return self.streets[street_a if street_a == street_b else street_b].name

    def route_length_cm(self, start: str, goal: str) -> float | None:
        """Shortest walking distance, for the environment's own bookkeeping.

        Privileged: it sets deadlines and measures how far off optimal a run
        was. It is never placed in an observation, because handing the agent a
        shortest path would replace navigation with reading a number.

        Streets the courier has *reported* shut are avoided, and only those. The
        phone does not know a barrier is there until it is told, so this is the
        courier's own knowledge being used to route, not the environment's.
        """
        if start not in self.network.nodes or goal not in self.network.nodes:
            return None
        import heapq

        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                return cost
            if node in seen:
                continue
            seen.add(node)
            here = self.position(node)
            for neighbour in self.network.nodes[node].neighbours:
                if neighbour in seen:
                    continue
                heapq.heappush(
                    queue, (cost + math.dist(here, self.position(neighbour)), neighbour)
                )
        return None

    def route_cost(self, start: str, goal: str, *,
                   obstacles: bool = False) -> tuple[float, float] | None:
        """``(seconds, centimetres)`` for the best walk, optionally avoiding obstacles.

        Two separate currencies because with obstacles they stop agreeing: the
        quickest way past a congested pavement can be the longer way round. The
        search is therefore on *seconds* and the metres are carried along the
        path it picks, rather than the reverse.

        ``obstacles=False`` is the plain graph and is what the phone uses -- a
        map app does not know a street is shut. ``obstacles=True`` is what the
        environment uses to price a shift and to judge how much further than
        necessary a run walked; it is privileged and never observable.
        """
        if start not in self.network.nodes or goal not in self.network.nodes:
            return None
        import heapq

        best: dict[str, float] = {start: 0.0}
        walked: dict[str, float] = {start: 0.0}
        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                return cost, walked[node]
            if node in seen:
                continue
            seen.add(node)
            here = self.position(node)
            for neighbour in self.network.nodes[node].neighbours:
                if neighbour in seen:
                    continue
                if obstacles and self.obstacles.blocks(node, neighbour):
                    continue
                distance = math.dist(here, self.position(neighbour))
                step = cost + distance / WALK_SPEED_CM_S
                if obstacles:
                    step += self.obstacles.delay_seconds(node, neighbour)
                if step < best.get(neighbour, float("inf")):
                    best[neighbour] = step
                    walked[neighbour] = walked[node] + distance
                    heapq.heappush(queue, (step, neighbour))
        return None

    def route_nodes(self, start: str, goal: str) -> list[str] | None:
        """The shortest walk from ``start`` to ``goal``, as the nodes along it.

        Same search as ``route_length_cm``, keeping the predecessors. Privileged
        in the same way: it is the raw material the navigation tool turns into
        spoken directions, and it never reaches an observation as node ids.
        """
        if start not in self.network.nodes or goal not in self.network.nodes:
            return None
        import heapq

        # The predecessor has to be updated whenever a cheaper way in is found,
        # not fixed the first time a node is pushed. Keeping the first pusher
        # reconstructs a path made of edges the search never chose, and on a
        # dense grid that path can be longer than the distance quoted beside it.
        best: dict[str, float] = {start: 0.0}
        came: dict[str, str] = {}
        seen: set[str] = set()
        queue = [(0.0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if node == goal:
                path = [node]
                while path[-1] != start:
                    path.append(came[path[-1]])
                return list(reversed(path))
            if node in seen:
                continue
            seen.add(node)
            here = self.position(node)
            for neighbour in self.network.nodes[node].neighbours:
                if neighbour in seen:
                    continue
                step = cost + math.dist(here, self.position(neighbour))
                if step < best.get(neighbour, float("inf")):
                    best[neighbour] = step
                    came[neighbour] = node
                    heapq.heappush(queue, (step, neighbour))
        return None

    def route_legs(self, start: str, goal: str) -> list[dict[str, Any]] | None:
        """A route as a person would say it: one leg per street, with the turn.

        Consecutive edges on the same street are one instruction -- "keep going
        along Rue Monge for four junctions" -- because that is one decision for
        a courier, and because a turn-by-turn list of 30 identical 18 m hops is
        not directions, it is the graph.
        """
        path = self.route_nodes(start, goal)
        if path is None or len(path) < 2:
            return [] if path is not None else None
        legs: list[dict[str, Any]] = []
        # The turn at a junction is between the edge you arrive on and the edge
        # you leave on, so this tracks the *last* edge walked, not the first edge
        # of the previous leg. Streets curve, and on a leg of six 18 m hops the
        # two differ by enough to call a left turn a right one.
        last_bearing: float | None = None
        for a, b in zip(path, path[1:]):
            street = self.edge_street(a, b)
            bearing = bearing_deg(self.position(a), self.position(b))
            length = math.dist(self.position(a), self.position(b)) / 100.0
            # A junction the courier will be *asked about*, which is not the
            # same thing at both strides. Playing the environment by hand caught
            # this: the phone said "take Avenue des Rosiers north-east, 6
            # junctions, 108 m", one walk_to covered 18 m and stopped, and the
            # courier had no way to tell whether it had gone the wrong way or
            # simply been counted in different units. Under the block stride a
            # leg's count is the number of calls it takes.
            counts = (self.stride != Stride.BLOCK
                      or len(self._neighbours.get(b, ())) != 2)
            if legs and legs[-1]["street"] == street:
                legs[-1]["junctions"] += int(counts)
                legs[-1]["metres"] += length
                legs[-1]["end"] = b
            else:
                legs.append({
                    "street": street, "start": a, "end": b,
                    "junctions": max(1, int(counts)),
                    "metres": length, "bearing": bearing,
                    "heading": compass_of(bearing),
                    "turn": ("head" if last_bearing is None
                             else turn_word(last_bearing, bearing)),
                })
            last_bearing = bearing
        for leg in legs:
            # The heading a leg is *announced* by is the direction it goes
            # overall, not the direction of its first 5 m. On a curving boulevard
            # the two disagree: one leg was announced "west" from a 5 m stub and
            # then ran north-east for 144 m, so a courier who re-asked at the
            # fork was told to walk the opposite way from the instruction it was
            # already following. ``bearing`` stays the first edge, because that
            # is the step the courier takes now.
            leg["heading"] = compass_of(
                bearing_deg(self.position(leg["start"]), self.position(leg["end"]))
            )
        return legs

    def album_coverage(self) -> dict[str, Any]:
        """How much of the walkable network the album actually covers.

        The album is baked against a specific compiled network. Rebuilding the
        graph -- a weld-tolerance change was enough -- renames every node and the
        frames become orphans while every manifest row still says ``status: ok``.
        That happened once and went unnoticed until a policy reported 86% of its
        candidates had no picture. Coverage is measured, not assumed.
        """
        total = covered = signalled = 0
        for node_id, node in self.network.nodes.items():
            for neighbour in node.neighbours:
                total += 1
                if self._plain_frame(node_id, neighbour):
                    covered += 1
                if self.signal_album_root is not None and node_id in self.signalised:
                    path = (self.signal_album_root / "images" / node_id
                            / f"toward_{neighbour}_red.png")
                    if path.exists():
                        signalled += 1
        return {
            "directed_edges": total, "with_frame": covered,
            "fraction": round(covered / total, 4) if total else 0.0,
            "signalised_with_frame": signalled,
            "album_root": str(self.album_root) if self.album_root else None,
        }

    def facing(self) -> float | None:
        """The bearing the courier is facing: the way the last step brought it.

        ``None`` at the very start of a shift, when the courier has not walked
        anywhere yet and so has no back to its head. Every relative direction in
        the observation is derived from this one number, so it lives here rather
        than being recomputed by each caller.
        """
        if self.arrived_from is None or self.arrived_from not in self.network.nodes:
            return None
        return bearing_deg(self.position(self.arrived_from), self.position())

    def _raw_candidates(self) -> list[dict[str, Any]]:
        """Neighbours with geometry but no images, so frame_for cannot recurse."""
        here = self.position()
        facing = self.facing()
        rows: list[dict[str, Any]] = []
        for neighbour in sorted(self.network.nodes[self.node_id].neighbours):
            there = self.position(neighbour)
            rows.append({
                "node": neighbour,
                "street": self.edge_street(self.node_id, neighbour),
                "bearing": bearing_deg(here, there),
                "distance_m": math.dist(here, there) / 100.0,
                "back": neighbour == self.arrived_from,
            })
        rows.sort(key=lambda r: (r["bearing"], r["node"]))
        for number, row in enumerate(rows, start=1):
            row["k"] = number
            row["heading"] = compass_of(row["bearing"])
            # A courier turns left and right, not to 214 degrees. Compass alone
            # made every route instruction a two-step conversion the agent had to
            # do in its head from a bearing it was never given.
            row["relative"] = relative_of(row["bearing"], facing)
        return rows

    def _block_preview(self, first: str) -> tuple[float, int, str]:
        """How far ``walk_to`` will actually carry the courier down this street.

        The candidate line quoted ``distance_m``, the distance to the *next
        waypoint*, at both strides. At block stride that is not what the action
        does, and on a short stub into a bend it is not even the right direction:
        a reviewer took a candidate labelled "on your left (south-east) — next
        junction 7 m" and was carried 61 m north-west. Three numbers described
        one leg -- the row said 18 m, the route said "1 junction, 29 m", the
        outcome said "29 m, through 2 junctions" -- and a policy budgeting from
        the row was wrong every time.

        Walks the same stop rules as ``_run_street`` with no side effects.
        Obstacles are deliberately not consulted: a preview that shortened
        itself at a barrier would announce the barrier in the text, which is the
        one fact the photographs are supposed to hold alone.
        """
        street = self.edge_street(self.node_id, first)
        here, previous, node = self.node_id, self.node_id, first
        distance = math.dist(self.position(here), self.position(first)) / 100.0
        junctions = 1
        for _ in range(len(self.network.nodes)):
            if self._standing_at_a_door(node):
                break
            neighbours = sorted(self.network.nodes[node].neighbours)
            onward = [n for n in neighbours
                      if self.edge_street(node, n) == street and n != previous]
            if len(onward) != 1 or len(neighbours) > 2:
                break
            if node in self.signalised and self.signal_is_visible(node, onward[0]):
                break
            distance += math.dist(self.position(node), self.position(onward[0])) / 100.0
            previous, node = node, onward[0]
            junctions += 1
        return distance, junctions, node

    def candidates(self) -> list[dict[str, Any]]:
        """The numbered streets leaving this junction, with their pictures.

        Ordering is stable and geographic -- clockwise from north -- so the same
        junction always numbers the same way and the numbers mean something a
        person could re-derive. Ordering by distance, as an earlier version did,
        renumbered the same corner depending on where the agent came from.
        """
        rows = self._raw_candidates()
        for row in rows:
            row["image"] = self.frame_for(self.node_id, row["node"])
            row["signal_image"] = self.signal_frame_for(self.node_id, row["node"])
            row["blocked_seen"] = (self.node_id, row["node"]) in self.witnessed_blocks
            if self.stride == Stride.BLOCK:
                # What one call actually buys, so the courier can budget from the
                # number it is shown. ``distance_m`` stays as it was -- the step
                # to the next waypoint -- because the geometry and the reference
                # policies are written against it.
                reach, junctions, end = self._block_preview(row["node"])
                row["reach_m"] = reach
                row["reach_junctions"] = junctions
                row["reach_heading"] = compass_of(
                    bearing_deg(self.position(), self.position(end))
                ) if end != self.node_id else row["heading"]
        return rows

    def frame_for(self, node_id: str, toward: str) -> str | None:
        """The view from ``node_id`` looking down the street toward ``toward``.

        Always the street frame, and that is the whole point. This used to serve
        the *signal* frame at the 105 signalised junctions, and the signal bake
        aims the camera at the lamp rather than along the street: on 88 of those
        105 nodes every approach shares one yaw, so all of a junction's
        candidate photographs were the same picture, pointing somewhere the
        courier was not about to walk. The system prompt promises "the
        photograph labelled k is the view down that street", and at 30% of
        junctions it was not.

        The lamp is a second thing to look at, not a replacement for the first,
        so it is served by ``signal_frame_for`` and shown beside this one.

        An obstacle is different again: it is *on* the street being looked down,
        so it replaces the frame rather than being shown beside it. That
        substitution is the entire mechanism by which the obstacle exists for
        the agent -- there is no field, no sentence and no tool that says it is
        there, so a courier that does not compare this picture with the one it
        expected walks into it.
        """
        return self.obstacle_frame_for(node_id, toward) or self._plain_frame(node_id, toward)

    def block_chain(self, node_id: str, toward: str,
                    limit: int | None = None) -> list[tuple[str, str]]:
        """The (from, to) hops one ``walk_to`` covers at the current stride.

        One hop at waypoint stride. At block stride, the hops along the named
        street as far as the next corner -- the same walk ``_run_street`` takes,
        read rather than performed, so a question can be asked about the whole
        block before committing a turn to it.
        """
        if self.stride != Stride.BLOCK:
            return [(node_id, toward)]
        street = self.edge_street(node_id, toward)
        hops = [(node_id, toward)]
        previous, current = node_id, toward
        bound = limit if limit is not None else len(self.network.nodes)
        for _ in range(bound):
            neighbours = self._neighbours.get(current, [])
            if len(neighbours) != 2:
                break                       # a corner: the block ends here
            onward = [n for n in neighbours
                      if n != previous and self.edge_street(current, n) == street]
            if len(onward) != 1:
                break
            hops.append((current, onward[0]))
            previous, current = current, onward[0]
        return hops

    def obstacle_frame_for(self, node_id: str, toward: str) -> str | None:
        """The view down this street with what is standing in it, if anything is.

        At block stride the question is asked of the whole block, not of the
        first eighteen metres of it, and the difference is the difference
        between vision mattering and not. Measured on TRIPLE over six seeds, a
        courier that reads the frames avoids every barrier at waypoint stride
        (21 collisions to 0) and almost none at block stride (59 to 49) when
        only the first hop is consulted -- because the barrier it walks into is
        four hops down a street whose photograph showed a clear road.

        Serving the obstacle's own frame for the whole block is honest here
        because of what a block is: the contraction rule breaks a block at any
        bend over 35 degrees, at any change of street name and at every junction,
        so a barrier standing on it is standing in the courier's line of sight.
        The photograph is taken a few tens of metres further along than the
        courier's feet, and it shows the thing that is really there.
        """
        if self.obstacle_album_root is None:
            return None
        found: list[tuple[str, str, str]] = []
        for start, step in self.block_chain(node_id, toward):
            kind = self.obstacles.in_effect(start, step)
            if kind is None:
                continue
            path = (self.obstacle_album_root / "images" / start
                    / f"toward_{step}_{kind}.png")
            if path.exists():
                found.append((kind, str(path), step))
        if not found:
            return None
        # A block can hold more than one thing, and then which one is shown
        # decides whether looking was any use. Taking the nearest served the
        # stall on the first hop and hid the barrier four hops behind it, so a
        # courier that read the frame walked into the barrier anyway -- 11 of
        # the 11 remaining collisions at block stride were this exact shape, all
        # on one street. A barrier is what changes the decision, and on a block
        # straight to within 35 degrees it is in view past the furniture.
        blocking = next((f for f in found if f[0] == ROAD_BLOCK), None)
        return (blocking or found[0])[1]

    def signal_frame_for(self, node_id: str, toward: str) -> str | None:
        """The pedestrian lamp facing this crossing, in the phase now running.

        Only where the album can actually show it. An approach outside
        ``signal_visibility.json`` has a lamp somewhere off frame, and handing
        the agent a picture of a light it cannot see -- then charging it for
        crossing -- is the defect the visibility gate exists to prevent.
        """
        if self.signal_album_root is None or node_id not in self.signalised:
            return None
        if not self.signal_is_visible(node_id, toward):
            return None
        rows = {row["node"]: row for row in self._raw_candidates()}
        row = rows.get(toward)
        if row is None:
            return None
        state = signal_state(node_id, row["bearing"], self.sim_seconds)
        path = (self.signal_album_root / "images" / node_id
                / f"toward_{toward}_{state}.png")
        return str(path) if path.exists() else None

    def _plain_frame(self, node_id: str, toward: str) -> str | None:
        """The baked view from ``node_id`` looking at ``toward``.

        One frame per walkable direction, so the picture beside ``walk_to(k)``
        is the view down the street ``walk_to(k)`` takes. The earlier album
        rendered four fixed compass yaws, which put the median walkable street
        20 degrees off the centre of the frame it was attached to.
        """
        if self.album_root is None:
            return None
        path = self.album_root / "images" / node_id / f"toward_{toward}.png"
        return str(path) if path.exists() else None

    # ── what the agent is told ───────────────────────────────────────────────

    def house_numbers_near(self, node_id: str, limit: int = 10) -> str:
        """The numbers a courier could read off the doors *at this spot*.

        Only doors whose kerb is this node, or close enough to read from it.
        Taking the two spatially nearest regardless of range advertised numbers
        from a median 50 m and up to 196 m away, so a junction could read
        "outside number 6-8" while door 7 was 107 m down the road -- which is
        exactly how a policy following the "compare the number exactly" rule
        walked away from its delivery.
        """
        street = self.street_of(node_id)
        here = self.position(node_id)
        near = [
            a for a in self.addresses_by_street.get(street, [])
            if a.kerb_node == node_id or math.dist(here, a.kerb) <= READABLE_NUMBER_CM
        ]
        if not near:
            return ""
        numbers = sorted(a.number for a in near)
        # The span of what is actually here, not the span of the first four.
        #
        # ``[:limit]`` was applied before taking min and max, so a node with
        # doors 1, 2, 3, 5, 7, 9, 11, 13 -- all of them at that node, 0.0 m away
        # -- advertised "1-5" and denied the existence of six doors the courier
        # was standing in front of. 17 of the map's 477 addresses could not be
        # recognised at their own doorstep, and the ARRIVAL runbook tells the
        # courier to "compare the number and street to the slip, exactly", so a
        # policy that obeys the prompt walks away from a delivery it has already
        # reached. On seed 9 that cost the whole episode: the courier arrived at
        # 9 Rue Mouffetard on turn 5, read "1-5", left, and spent the remaining
        # 115 turns oscillating past the door for 0 deliveries.
        #
        # The doors are named, not summarised as a span. A span asserts that
        # every number inside it is here, and on this map that is false about
        # half the time: a corner showing "2-8" from doors 2 and 8 has doors 4
        # and 6 fifty metres down the road, so a courier told to compare the
        # number exactly reads a match and collects nothing. Measured over the
        # map, 458 of 953 advertised numbers were for a door out of reach, and
        # naming them instead takes that to 15. Only 8 of 366 junctions have
        # more doors than ``limit`` names, so the ellipsis is the rare case
        # rather than the usual one.
        if len(numbers) == 1:
            return str(numbers[0])
        if len(numbers) <= limit:
            return ", ".join(str(n) for n in numbers)
        return ", ".join(str(n) for n in numbers[:limit]) + f", … {numbers[-1]}"

    def location_text(self) -> str:
        street = self.street_of(self.node_id)
        numbers = self.house_numbers_near(self.node_id)
        text = f"You are on {street}"
        if numbers:
            noun = "number" if numbers.isdigit() else "numbers"
            text += f", outside {noun} {numbers}"
        text += "."

        # Say out loud whether this street is the one on the slip.
        #
        # This is not privileged information: it is string equality between two
        # things already printed a few lines apart, the job line and this one.
        # It is here because the comparison is the step policies skip. Measured
        # on Qwen3-VL-4B over 40 episodes: 42 collect() attempts, 32 of them
        # refused because it was not at the door -- 32 turns spent asking a
        # question the observation had already answered. Nothing here says
        # which way the address is; finding it is still the task.
        order = self.active_order()
        if order is not None:
            target = order.target
            if target.street_name == street:
                text += (f" This is the street on the slip; the slip says "
                         f"{target.number}.")
                trend = self._number_trend(street, numbers)
                if trend:
                    text += " " + trend
            else:
                text += f" The slip says {target.street_name}, which is not this street."
        return text

    def _number_trend(self, street: str, numbers: str) -> str:
        """Whether the doors counted up or down on the way here.

        This is the signal a person actually uses to find a door: not the
        number on this building, but which way the numbers are going. Two
        thirds of failed episodes never reach the pickup at all, and the
        observation was giving the courier a number with nothing to compare it
        against -- it had to hold the last one in its head across a turn, and
        across forty turns of context it did not.

        Facts only, no advice. It says the numbers rose or fell; it does not
        say to turn around. Which way to walk is still the decision under test.
        """
        here = _leading_number(numbers)
        last_street, last_number = self._last_numbers
        self._last_numbers = (street, here if here is not None else last_number)
        if here is None or last_street != street or last_number is None:
            return ""
        if here == last_number:
            return ""
        return ("The numbers rose as you walked here."
                if here > last_number else
                "The numbers fell as you walked here.")

    def clock_text(self) -> str:
        """Every job in hand and how long each has left.

        The whole queue, not just one job. Showing the focused order alone would
        hand the agent a deep queue and hide the thing that makes it deep: a
        courier cannot sequence windows it cannot see, and a tier whose demand
        is invisible in the observation measures luck.
        """
        live = self.live_orders()
        if not live:
            return ""
        if len(live) == 1:
            left = live[0].minutes_left(self.sim_seconds)
            if left < 0:
                return f"You are {abs(left):.0f} min past the deadline."
            return f"{left:.0f} min left before the deadline."
        lines = [f"You are carrying {len(live)} jobs. Their deadlines run at the same time:"]
        for order in live:
            left = order.minutes_left(self.sim_seconds)
            stage = ("deliver to " + order.dropoff.text if order.picked_up
                     else "collect from " + order.pickup.text)
            when = (f"{left:.0f} min left" if left >= 0 else f"{abs(left):.0f} min overdue")
            lines.append(f"  job {order.index}: {stage} — {when}")
        return "\n".join(lines)

    def active_order(self) -> Order | None:
        """The job the observation talks about by default.

        Whatever is already in the bag, otherwise the oldest job in the queue.
        Both halves matter. Collecting a parcel and then being told about a
        different address is how a policy ends up carrying a bag around the
        district; and defaulting to the *oldest* rather than the tightest keeps
        the focus from flipping under a policy every time the clock ticks -- a
        min-by-deadline default made the reference courier thrash between two
        targets and cost it half its deliveries.

        It is a default, not a constraint. ``collect`` and ``hand_over`` serve
        whichever live order the courier is actually standing at, so a policy
        that plans a better sequence is free to walk it, and that is precisely
        the capability the deeper tiers exist to measure.
        """
        live = self.live_orders()
        if not live:
            return None
        carried = [o for o in live if o.picked_up]
        return (carried or live)[0]

    @property
    def shift_over(self) -> bool:
        return (
            self.shift_seconds is not None and self.sim_seconds >= self.shift_seconds
        )

    @property
    def delivered_count(self) -> int:
        return sum(1 for o in self.orders if o.delivered)

    @property
    def issued_count(self) -> int:
        """Jobs the dispatcher actually handed over.

        Not ``len(self.orders)``. ENDLESS draws work on demand, and a list that
        keeps growing is not a denominator -- scoring ``delivered / 40`` against
        a queue nobody can finish reported a perfect courier at 39%.
        """
        return sum(1 for o in self.orders if o.issued_at_s is not None)

    @property
    def on_time_count(self) -> int:
        return sum(
            1 for o in self.orders
            if o.delivered and o.delivered_at_s is not None
            and o.delivered_at_s <= o.due_at()
        )

    @property
    def late_count(self) -> int:
        return self.delivered_count - self.on_time_count

    def target_address(self) -> Address | None:
        order = self.active_order()
        return order.target if order else None

    def distance_to_target_cm(self) -> float | None:
        """Straight-line distance to the *door the arrival check tests*.

        The evaluation found the previous runtime quoting distance to one point
        while validating arrival against another, with a gap over the tolerance
        in eight of ten seeds. Here both are this door.
        """
        target = self.target_address()
        if target is None:
            return None
        return math.dist(self.position(), target.kerb)

    # ── tools ────────────────────────────────────────────────────────────────

    def _charge(self, outcome: StepOutcome) -> StepOutcome:
        """Advance the clock by what the tool declared it cost.

        Every tool returned a ``sim_seconds`` and only walking and waiting ever
        applied it, so looking things up was free against the deadline and the
        optimal policy was to consult on every turn forever. Charging here, in
        one place, means a tool cannot declare a cost it does not pay.
        """
        self.turns += 1
        if not outcome.ok:
            # A looking or consulting tool that was refused is still a refused
            # action, and it is still a turn the courier did not spend walking.
            # The declared cost stands where there is one -- a phone with no
            # signal says so in a second -- but a refusal that declared nothing
            # pays the same floor as any other.
            self.rejected_actions += 1
            if outcome.sim_seconds <= 0.0:
                outcome.sim_seconds = REJECTED_ACTION_SECONDS
        self.sim_seconds += outcome.sim_seconds
        return outcome

    def _refuse(self, outcome: StepOutcome, seconds: float = REJECTED_ACTION_SECONDS) -> StepOutcome:
        """A refused action still happened, so it still costs.

        It cost nothing before -- not a turn, not a second -- and that made
        ``collect()`` an unlimited free rangefinder: its refusal states the exact
        distance to the door, which is what ``check_map`` charges a turn and five
        seconds to say. Walking up to a door and finding it is the wrong one is
        a thing a courier *does*, and both currencies this benchmark reports
        have to see it happen.
        """
        self.turns += 1
        self.rejected_actions += 1
        self.sim_seconds += seconds
        outcome.sim_seconds = seconds
        return outcome

    # ── the body ─────────────────────────────────────────────────────────────

    def travel_speed_cm_s(self) -> float:
        """How fast this body is moving *now*.

        A tired courier does not stop, it slows: a hard stop turns one bad
        estimate about stamina into an episode nobody can finish, and that is not
        what running out of energy does to a rider.
        """
        speed = self.embodiment.speed_cm_s or WALK_SPEED_CM_S
        if self.stamina <= 0.0:
            speed *= self.embodiment.tired_speed_fraction
        return speed

    def _spend_stamina(self, metres: float) -> None:
        """Stamina goes on distance, not on time.

        Charging by time would make the slowest body the most tired one, which
        is backwards -- a scooter covering the same ground in a third of the
        time has done less work, not more.
        """
        drain = self.embodiment.stamina_per_m or 0.0
        if drain:
            self.stamina = max(0.0, self.stamina - metres * drain)

    @property
    def tired(self) -> bool:
        return self.stamina <= 0.0

    def _standing_at_a_door(self, node_id: str | None = None) -> bool:
        """At the kerb of any live job -- not merely the one in hand.

        A walk that ran past a pickup because the courier happened to be
        carrying a different job would make the block stride worse than walking
        the same ground one waypoint at a time, which is the one thing it must
        never be.

        ``node_id`` lets ``_block_preview`` ask the question about a node the
        courier has not reached yet.
        """
        here = self.position(node_id)
        return any(
            math.dist(here, order.target.kerb) <= ARRIVAL_TOLERANCE_CM
            for order in self.live_orders()
        )

    def resolve_street(self, street: str, heading: str | None = None):
        """Which street leaving this junction the courier named.

        Returns ``(k, None)`` or ``(None, refusal)``. Streets are chosen by
        name and bearing rather than by a number the observation assigns,
        because the number is only stable within one junction: Rue de Grenelle
        is street 3 here, street 1 at the next corner and absent at the one
        after. A policy given numbers cannot carry a single fact about a
        street from one corner to the next -- "I have already tried that one"
        is not expressible. A name and a bearing are the same everywhere.

        The refusals are written to be acted on. Naming a street that is not
        here lists the ones that are; naming one that is here twice asks for
        the bearing and says which two are available.
        """
        rows = self.candidates()
        # "left"/"right" mean relative to facing, not west/east.
        heading = resolve_relative(heading, self.facing())
        try:
            row = match_street(rows, street, heading)
        except StreetNotHere:
            return None, self._refuse(StepOutcome(
                ok=False, code="no_such_street",
                message=(
                    f"There is no {street} leaving this junction. From here you "
                    f"can take: {self._street_menu(rows)}."
                ),
            ))
        except StreetAmbiguous as error:
            return None, self._refuse(StepOutcome(
                ok=False, code="which_way",
                message=(
                    f"{street} leaves this junction in more than one direction "
                    f"({' and '.join(error.headings)}). Say which: "
                    f'walk_to("{street}", "{error.headings[0]}").'
                ),
            ))
        return row["k"], None

    def street_at(self, k: int) -> tuple[str, str]:
        """The name and bearing of this junction's k-th street, clockwise from north.

        The courier no longer sees these numbers -- it names streets -- but the
        reference policies and the tests still need a way to say "the first
        street here" without knowing the map. Nothing the policy can reach
        calls this.
        """
        row = next(row for row in self.candidates() if row["k"] == k)
        return row["street"], row["heading"]

    def _street_menu(self, rows: list[dict[str, Any]] | None = None) -> str:
        """The streets here, as a courier would say them back."""
        rows = self.candidates() if rows is None else rows
        return ", ".join(f'"{row["street"]}" {row["heading"]}' for row in rows)

    def walk_to(self, street: str, heading: str | None = None) -> StepOutcome:
        """Take the named street. How far one call carries is the stride.

        Two resolutions of the same city, and the difference is only where the
        courier is asked to stop and choose. See ``Stride``.
        """
        k, refusal = self.resolve_street(street, heading)
        if refusal is not None:
            return refusal
        if self.stride == Stride.BLOCK:
            return self._run_street(k, steps=None, to_corner=True)
        return self._step_to(k)

    def _step_to(self, k: int) -> StepOutcome:
        """One waypoint along street ``k``: the atomic move, whatever the stride."""
        rows = {row["k"]: row for row in self.candidates()}
        if k not in rows:
            message = (f"That street does not leave this junction. From here "
                       f"you can take: {self._street_menu()}.")
            if len(rows) == 1:
                # Naming the legal street was not enough. On three of forty
                # episodes the courier stood at a dead end and asked for street
                # 2 twenty-five, twenty-six and thirty-three times in a row --
                # the whole episode -- because the only legal move went back the
                # way it came and it would not take it. The refusal now says
                # that going back is the move, not a mistake.
                only = rows[next(iter(rows))]
                message += (f' This is a dead end. walk_to("{only["street"]}", '
                            f'"{only["heading"]}") goes back the way you came, '
                            "and here that is the only way on: take it rather "
                            "than asking again.")
            return self._refuse(StepOutcome(
                ok=False, code="no_such_street", message=message,
            ))
        row = rows[k]
        # A barrier is found the way a courier finds one: by walking up to it.
        # The refusal names it, because someone standing at a barrier can see it
        # -- but by then the turn and the time are gone, and that gap is exactly
        # what the photograph was worth. Nothing before this moment mentions it.
        if self.obstacles.blocks(self.node_id, row["node"]):
            self.blocked_attempts += 1
            # The courier is standing at the barrier. The phone still does not
            # know -- a survey does not learn -- but a person who has just been
            # stopped by a barrier can still see it on the next turn, and
            # re-offering the street as though nothing happened is not realism,
            # it is the opposite. Measured on Qwen3-VL-4B: 96 of 153 refused
            # actions were way_blocked, and in 52 of 93 cases the very next
            # action was the same street again, because the menu was identical.
            self.witnessed_blocks.add((self.node_id, row["node"]))
            # Nothing is recorded. Whatever the courier now knows about this
            # street it knows the way a person does -- it was just standing at
            # the barrier -- and remembering it is the agent's job. The phone is
            # not told, because there is nobody to tell: the route is computed
            # from a survey, and a survey does not learn.
            return self._refuse(StepOutcome(
                ok=False, code="way_blocked",
                message=(
                    # "You walk back to the junction" read as "you are where you
                    # started", and at block stride that is flatly wrong: the
                    # metres already walked are kept and the courier is standing
                    # at the barrier, which the position header says and this
                    # sentence contradicted.
                    f"{row['street']} is blocked and you cannot get past. You are "
                    "at the last junction before it. You will have to go round."
                ),
            ), seconds=BLOCKED_SECONDS)
        penalty = 0.0
        if (self.enforce_signals and self.node_id in self.signalised
                and self.signal_is_visible(self.node_id, row["node"])):
            if signal_state(self.node_id, row["bearing"], self.sim_seconds) == "red":
                # Crossing anyway. Allowed, costed, and counted -- a courier who
                # never looks will do this about half the time.
                self.red_crossings += 1
                penalty = RED_CROSSING_PENALTY
                # Held up at the kerb, and the delay counts against the deadline.
                self.sim_seconds += RED_CROSSING_PENALTY_S
        self.turns += 1
        seconds = row["distance_m"] * 100.0 / self.travel_speed_cm_s()
        self._spend_stamina(row["distance_m"])
        # A congested pavement is passable, so this is not a refusal: it is the
        # same walk, slower. The message stays silent about why, because saying
        # "you were held up by the stand on the pavement" would put the obstacle
        # into the text and hand a blind policy a free map of the city's
        # obstructions after one lap. The courier sees the clock move; working
        # out what it was is what the photograph is for.
        delay = self.obstacles.delay_seconds(self.node_id, row["node"])
        if delay:
            self.slow_passages += 1
            seconds += delay
        self.arrived_from = self.node_id
        self.node_id = row["node"]
        self.sim_seconds += seconds
        self.walked_cm += row["distance_m"] * 100.0
        self._issue()
        crossed_on_red = penalty > 0.0
        return StepOutcome(
            ok=True, moved=True, sim_seconds=seconds, walked_m=row["distance_m"],
            reward=-penalty,
            message=(
                f"You walk {row['distance_m']:.0f} m {row['heading']} along {row['street']}."
                + (" You crossed against the pedestrian light." if crossed_on_red else "")
            ),
        )

    # How many junctions one ``follow_street`` may cover. Matches the macro
    # declared in ``skills.py``, which is where the argument for it is written.
    MAX_FOLLOW = 6

    def follow_street(self, street: str, heading: str | None = None,
                      n: int = MAX_FOLLOW) -> StepOutcome:
        """Take the named street and keep going straight, up to ``n`` junctions.

        The turn budget and the graph were sized against different worlds. A
        delivery leg is a median 530 m; an edge on the compiled carriageway is a
        median 18 m. So one order costs about 35 ``walk_to`` calls and a ten-order
        shift about 351 -- against a step budget of 120. Even a shortest-path
        oracle with no perception cost delivered 3.1 orders before the budget ran
        out, so the cap, not the courier, was setting the score.

        Walking four junctions down one street is one decision for a rider, not
        four, and ``skills.py`` already argued the case: a macro is admissible
        exactly when it is mechanical. This one never chooses a street -- the
        caller names it -- and it stops the moment the situation stops being
        obvious: at a fork, at a dead end, when the street changes name, and at
        the door it was sent to. It cannot walk a lost courier anywhere its
        caller could not have walked one step at a time, and it cannot walk it
        past the turn it should have taken.
        """
        k, refusal = self.resolve_street(street, heading)
        if refusal is not None:
            return refusal
        return self._run_street(k, steps=max(1, min(int(n), self.MAX_FOLLOW)),
                                to_corner=False)

    def _run_street(self, k: int, *, steps: int | None, to_corner: bool) -> StepOutcome:
        """Walk street ``k`` until something worth a decision happens.

        One body for both the macro and the block stride, because they stop for
        the same reasons and only disagree about two of them. ``steps`` bounds
        the macro's reach; ``None`` is the stride, which runs to the end of the
        block however long it is. ``to_corner`` adds the stride's extra rule:
        stop wherever a choice exists, not merely where this street stops going.

        That extra rule is the whole difference between the two resolutions. The
        macro is allowed to walk straight through a side turning, because its
        caller said "stay on this street"; the stride is not, because at block
        resolution a courier who is not offered the turning cannot take it.
        """
        rows = {row["k"]: row for row in self.candidates()}
        if k not in rows:
            return self._refuse(StepOutcome(
                ok=False, code="no_such_street",
                message=(
                    f"That street does not leave this junction. From here you "
                    f"can take: {self._street_menu()}."
                ),
            ))
        street = rows[k]["street"]
        before_turns = self.turns
        first = self._step_to(k)
        if not first.ok:
            return first
        walked, seconds, reward, taken, stop = first.walked_m, first.sim_seconds, first.reward, 1, ""
        # A block on any map is bounded by the graph, but a ring road with no
        # junction on it is not, and a stride with no bound would walk it for
        # ever. One step per node is past any real block and short of a hang.
        limit = steps if steps is not None else len(self.network.nodes)
        while taken < limit:
            if self._standing_at_a_door():
                stop = " You are at the address you were looking for."
                break
            here = self.candidates()
            onward = [
                row for row in here
                if row["street"] == street and row["node"] != self.arrived_from
            ]
            if not onward:
                stop = (" The street ends here." if len(here) <= 1
                        else f" {street} does not go on from here.")
                break
            if len(onward) > 1:
                stop = f" {street} forks here."
                break
            # A crossing whose light the courier can see is a decision, so the
            # macro hands control back rather than walking through it. Without
            # this the macro quietly took the red lights its caller was being
            # charged for and never got to look at -- reintroducing, inside one
            # tool, exactly the "penalised for something you could not observe"
            # defect the visibility gate exists to remove.
            if (self.node_id in self.signalised
                    and self.signal_is_visible(self.node_id, onward[0]["node"])):
                stop = " There is a pedestrian light at this crossing."
                break
            if to_corner and len(here) > 2:
                # A side turning. The macro may pass it; the stride may not, or
                # the courier is never offered a turn it was standing on.
                stop = f" A street leaves {street} here."
                break
            # Obstacles are deliberately *not* handled here, and the reason is a
            # leak rather than an oversight. Every other early stop announces
            # itself -- a fork, a dead end, a light -- so an unexplained one
            # would tell a policy that never looks at anything that there is
            # something in the road ahead, which is the one fact the pictures
            # are supposed to hold alone. Instead the walk simply runs into it
            # below and reports it from the kerb, the way it happens.
            outcome = self._step_to(onward[0]["k"])
            if not outcome.ok:
                # Whatever stopped it is reported in its own words, from where
                # the courier now stands. A macro that ended silently would be
                # the leak described above.
                stop = f" {outcome.message}"
                break
            walked += outcome.walked_m
            seconds += outcome.sim_seconds
            reward += outcome.reward
            taken += 1
        # One decision, one turn. Each inner ``walk_to`` charged a turn of its
        # own, so a macro whose entire argument is that four junctions down one
        # street is *one* decision for a rider was billed as four -- which made
        # the turn budget refuse to reward the thing it was introduced to allow.
        # Time is untouched: the walking still takes exactly as long.
        self.turns = before_turns + 1
        return StepOutcome(
            ok=True, moved=True, sim_seconds=seconds, walked_m=walked, reward=reward,
            message=(
                f"You walk {walked:.0f} m along {street}, through {taken} junction"
                f"{'' if taken == 1 else 's'}.{stop}"
            ),
        )

    def _look_impl(self, street: str, heading: str | None = None) -> StepOutcome:
        """The door numbers down street k, and nothing the turn already said.

        It used to open with the street's name, compass heading and distance to
        the next junction -- which is the candidate line for k, word for word,
        already on screen. Two of its three sentences could not change what the
        courier knew, so the courier's only reason to spend a turn here was the
        third, and it was buried at the end behind the restatement.

        What is left is the one thing this junction's text does not carry: how
        the numbers run *down a street the courier is not on*. That is what
        chooses a direction along a road when no phone will, and it is worth two
        seconds precisely because it is not free every turn for every street.
        """
        k, refusal = self.resolve_street(street, heading)
        if refusal is not None:
            return refusal
        row = next(row for row in self.candidates() if row["k"] == k)
        if self.condition == Condition.VISUAL:
            # The numbers are meant to be on the doors under this condition. They
            # are not legible in these renders -- see ``Condition`` -- so this
            # says what it can see rather than pretending.
            return StepOutcome(
                ok=True, sim_seconds=2.0,
                message=(
                    f"You cannot make out door numbers down {row['street']} from "
                    "here. You would have to walk it."
                ),
            )
        numbers = self.house_numbers_near(row["node"])
        here = self.house_numbers_near(self.node_id)
        if not numbers:
            return StepOutcome(
                ok=True, sim_seconds=2.0,
                message=f"No door numbers are visible down {row['street']}.",
            )
        way = ""
        # Which way the numbers run is the whole reason to look, so say it rather
        # than leaving two number ranges for the agent to difference.
        first, second = _leading_number(here), _leading_number(numbers)
        if first is not None and second is not None and first != second:
            way = " climbing" if second > first else " falling"
        return StepOutcome(
            ok=True, sim_seconds=2.0,
            message=f"Down {row['street']} the doors read {numbers}{way}.",
        )

    def _check_order_impl(self) -> StepOutcome:
        """The whole slip stack, not one slip.

        A courier holding three jobs reads all three. Reporting only the focused
        one hid the entire demand of the deeper tiers -- which job to serve next
        is the decision being scored, and it cannot be made from a description
        of one job.
        """
        live = self.live_orders()
        if not live:
            return StepOutcome(ok=True, sim_seconds=1.0, message="You have no job in hand.")
        lines = []
        for order in live:
            stage = "deliver to" if order.picked_up else "collect from"
            lines.append(
                f"Job {order.index}: {stage} {order.target.text}. "
                f"Pickup {order.pickup.text}, dropoff {order.dropoff.text}. "
                f"Fee {order.fee:.2f}. "
                f"{order.minutes_left(self.sim_seconds):.0f} min left."
            )
        return StepOutcome(ok=True, sim_seconds=1.0 + 1.0 * len(live),
                           message="\n".join(lines))

    def _check_map_impl(self, address: str) -> StepOutcome:
        """Phone lookup: which street, how far, roughly which way.

        A direction and a distance, which is what a map app gives. Not a route:
        turn-by-turn directions would make the phone the navigator.
        """
        if self.condition in (Condition.NO_PHONE, Condition.VISUAL):
            return StepOutcome(
                ok=False, code="no_phone", sim_seconds=1.0,
                message="Your phone has no signal here. You will have to find it by the streets.",
            )
        match = self._find_address(address)
        if match is None:
            known = sorted(self.addresses_by_street)[:4]
            return StepOutcome(
                ok=False, code="unknown_address", sim_seconds=5.0,
                message=(
                    f"Your phone cannot find {address!r}. Streets you know of include "
                    f"{', '.join(known)}."
                ),
            )
        here = self.position()
        # Walking distance, not crow-flies. Straight-line range was actively
        # misleading: a six-turn trap had it improving 140 -> 105 m while the
        # route the courier would have to walk worsened 148 -> 238 m, so the
        # agent was rewarded for approaching a wall. A map app quotes route
        # distance, and so does this.
        route = self.route_length_cm(self.node_id, match.kerb_node) if match.kerb_node else None
        distance = (route if route is not None else math.dist(here, match.kerb)) / 100.0
        # Straight-line, and now *said* to be straight-line.
        #
        # The two halves of this sentence are measured in different frames and
        # always were: the distance is along the road and the bearing is as the
        # crow flies. Over 1495 sampled lookups the straight-line bearing and the
        # direction the route actually leaves in differ by a median of 46
        # degrees, by more than 90 on 22.7%, and by more than 135 -- the pin
        # lying broadly behind the courier -- on 8.4%. Only 27% named the compass
        # point the courier should walk in.
        #
        # That is not a bug in the bearing, it is a pin: a map app shows you
        # where a place *is*, and working out which street gets you there is the
        # job. It became a bug because the sentence read like an instruction. On
        # pair seed 20 -- the one easy-tier failure in 40 seeds -- the phone said
        # "to the east" for a dropoff whose route leaves south-west, and a
        # courier following the bearing walked away from the door for the rest of
        # the shift.
        #
        # Quoting the first hop of the route instead was tried and rejected: it
        # is what ``navigate()`` is for, it lifts the reference courier from 53%
        # to 70% on SHIFT by doing the navigation for it, and it would leave the
        # benchmark with two tools that both route. So the wording carries the
        # frame instead, and the courier is told which of the two numbers is a
        # direction to walk in -- neither.
        # No bearing in the words. A phone shows you where a place is by
        # drawing it, and every direction this benchmark spoke aloud was a
        # direction the policy could act on without looking at anything --
        # which made choosing a street a text problem and the photographs
        # decoration. The bearing is on the map, where a person reads it.
        return StepOutcome(
            ok=True, sim_seconds=5.0,
            message=(
                f"{match.text} is on {match.street_name}, about {distance:.0f} m "
                f"away on foot. Your phone is showing you where it is."
            ),
        )

    # How many legs of the route the phone reads out before it stops. A map app
    # shows the next few turns, not the whole itinerary, and a courier who is
    # told fourteen turns will not remember the fourteenth. Asking again is a
    # turn and 15 s, which is the cost of not paying attention.
    NAVIGATE_LEGS_SHOWN = 5
    # What a phone lookup plus reading the route off the screen costs a courier
    # standing on the pavement. Set above ``check_map``'s 5 s deliberately: the
    # route is worth more, and a policy that calls it every turn instead of
    # walking should lose the race to one that calls it once a leg.
    NAVIGATE_SECONDS = 15.0

    def _navigate_impl(self, where: str | None = None) -> StepOutcome:
        """Put a route to an address on the phone's screen.

        Takes the address, the way a person types one in:
        ``navigate("13 Avenue Dauphine")``. It used to take a job number, which
        is a thing the dispatcher knows and a phone does not, and which the
        courier had to be told separately. An address is on the slip in front
        of it. A bare ``navigate()`` still routes to the job in hand, and a
        number still selects among several jobs, because at the deeper tiers
        sequencing is the task and "route to my second job" is a real thing to
        want.

        This is the phone's map app, and it is deliberately the *only* thing in
        the environment that will tell a courier which way to go -- and it now
        does that by drawing rather than by speaking. Everything a rider does
        with their eyes stays with their eyes: it says nothing about the
        pedestrian light at the next crossing and nothing about what is in the
        way.

        A route is not a solution. The courier still has to read it off the
        screen, execute it, watch the crossings, and recognise the door.
        """
        # Addresses only. The signature says text and the implementation used
        # to also take a job number, which is the shape of mismatch this
        # benchmark keeps finding in itself: a contract stated in one place and
        # quietly widened in another. Selecting among several jobs is still
        # expressible, because each one has its own address on the slip.
        # Condition gate first, before the address branch's early returns.
        if self.condition in (Condition.NO_PHONE, Condition.VISUAL):
            return StepOutcome(
                ok=False, code="no_phone", sim_seconds=1.0,
                message="Your phone has no signal here. You will have to find it by the streets.",
            )
        job: int | None = None
        if isinstance(where, str) and where.strip():
            match = self._find_address(where)
            if match is None:
                known = sorted(self.addresses_by_street)[:4]
                return self._refuse(StepOutcome(
                    ok=False, code="unknown_address", sim_seconds=5.0,
                    message=(
                        f"Your phone cannot find {where}. Streets it knows "
                        f"include {', '.join(known)}."
                    ),
                ))
            order = next(
                (o for o in self.live_orders()
                 if o.target.text.strip().lower() == match.text.strip().lower()),
                None)
            if order is None:
                # A real map app routes anywhere; it does not check your job
                # list first. The screen goes to the address asked for.
                return self._route_to(match)
            job = order.index
        live = self.live_orders()
        if not live:
            return StepOutcome(ok=True, sim_seconds=1.0, message="You have no job in hand.")
        if job is None:
            order = self.active_order()
        else:
            # Which job to route to is the courier's decision, not the phone's.
            # With several live orders the sequencing *is* the task at the deeper
            # tiers, and a navigation tool that only ever routes to the oldest
            # job would quietly make that decision on the agent's behalf.
            order = next((o for o in live if o.index == int(job)), None)
            if order is None:
                return StepOutcome(
                    ok=False, code="no_such_job", sim_seconds=1.0,
                    message=(
                        f"You are not carrying job {job}. In hand: "
                        f"{', '.join('job ' + str(o.index) for o in live)}."
                    ),
                )
        if order is None:
            return StepOutcome(ok=True, sim_seconds=1.0, message="You have no job in hand.")
        return self._route_to(order.target)

    def _route_to(self, target: Address) -> StepOutcome:
        """Put a route to one address on the screen, whoever asked for it."""
        gap = math.dist(self.position(), target.kerb)
        if gap <= ARRIVAL_TOLERANCE_CM:
            return StepOutcome(
                ok=True, sim_seconds=self.NAVIGATE_SECONDS,
                message=f"You have arrived: {target.text} is right here.",
            )
        legs = self.route_legs(self.node_id, target.kerb_node) if target.kerb_node else None
        if not legs:
            return StepOutcome(
                ok=False, code="no_route", sim_seconds=self.NAVIGATE_SECONDS,
                message=(
                    f"Your phone cannot plot a route to {target.text} from here. "
                    "Walk to a bigger street and ask again."
                ),
            )
        metres = sum(leg["metres"] for leg in legs)
        minutes = metres * 100.0 / WALK_SPEED_CM_S / 60.0
        # The route is drawn, not dictated. Every leg used to be spelled out
        # -- "Take Rue de Grenelle, east, 1 junction, 18 m" -- and a courier
        # holding that text never had to look at anything: the street to take
        # was named, the bearing was named, and the photographs and the map
        # were both decoration. Choosing a street was a reading exercise.
        #
        # What the phone says now is what a phone says when you glance at it
        # without stopping: how far, how long, and that it is on the screen.
        # Which way to go is on the map, which is a picture. The names of the
        # streets are on the corner, which is text. Whether the way is open,
        # and whether the light is red, are in the photographs. No one of them
        # is enough.
        lines = [
            f"Route to {target.text} — {metres:.0f} m, about {minutes:.0f} min "
            f"on foot, {len(legs)} street{'' if len(legs) == 1 else 's'} to walk.",
            "  Your phone is showing the route. Read it off the map: the line "
            "runs from where you are to where you are going.",
        ]
        return StepOutcome(
            ok=True, sim_seconds=self.NAVIGATE_SECONDS, message="\n".join(lines),
            drawing=self.map_drawing(target, legs).svg,
        )

    def map_drawing(self, target: Address | None = None,
                    legs: list[dict[str, Any]] | None = None) -> MapDrawing:
        """The picture on the phone's screen: streets, the route, the pin, you.

        Drawn from the compiled network, so it says what a survey knows and
        nothing else. It cannot show a light, a barrier or a shopfront -- not
        because that would be hard, but because the phone cannot see the street,
        and that separation is the reason ``report_blocked`` exists at all. The
        Nothing on it came from the courier's eyes. There is no way to put a
        barrier on this map, because there is no way to tell the map about one.

        With no ``target``, this is the screen as it stands: the last route the
        courier asked for, still drawn where it was drawn, with the courier's
        current position and heading on it. Pass a target to compute a fresh
        route -- which is what ``navigate`` does, and what it charges for.
        """
        route: list[tuple[float, float]] = []
        if target is None:
            target, route = self.screen_target, list(self.screen_route)
        elif target.kerb_node:
            path = self.route_nodes(self.node_id, target.kerb_node) or []
            route = [self.position(node) for node in path]
            self.screen_target, self.screen_route = target, list(route)
        return render_map(
            self.network,
            here=self.position(),
            facing_deg=self.facing(),
            route=route,
            destination=(target.kerb if target is not None else None),
            destination_label=(target.text if target is not None else ""),
            here_label="you are here",
        )

    def _find_address(self, text: str) -> Address | None:
        wanted = " ".join(str(text).split()).lower()
        for address in self.network.addresses:
            if address.text.lower() == wanted:
                return address
        for address in self.network.addresses:
            if wanted and wanted in address.text.lower():
                return address
        return None

    def _order_at_hand(self, collected: bool) -> Order | None:
        """A live order whose next stop is the door the courier is standing at.

        Serving *whichever* job is here, rather than only the focused one, is
        what makes a queue a queue: the courier's route is its plan, and a plan
        that batches two nearby stops has to be executable without a tool for
        saying so. Nearest first, so two doors inside the tolerance resolve the
        way a person would resolve them.
        """
        here = self.position()
        candidates = [
            (math.dist(here, (o.dropoff if collected else o.pickup).kerb), o.index, o)
            for o in self.live_orders() if o.picked_up == collected
        ]
        candidates = [c for c in candidates if c[0] <= ARRIVAL_TOLERANCE_CM]
        return min(candidates)[2] if candidates else None

    def _nearest_live(self, *, collected: bool) -> Order | None:
        """The live job whose door is closest, for a refusal that names it.

        Falling back to ``active_order`` meant a courier standing one junction
        short of job 1's pickup was told "you are not at 19 Boulevard du Temple"
        -- job 0's address, 71 m the other way. The message named a place the
        courier was not going and said nothing about the one it was, which is
        the opposite of what a refusal is for.
        """
        live = [o for o in self.live_orders() if o.picked_up == collected]
        if not live:
            return None
        here = self.position()
        return min(live, key=lambda o: math.dist(
            here, (o.dropoff if collected else o.pickup).kerb))

    def collect(self) -> StepOutcome:
        order = (self._order_at_hand(collected=False) or self._nearest_live(collected=False)
                 or self.active_order())
        if order is None:
            return self._refuse(StepOutcome(
                ok=False, code="no_order", message="You have no job in hand."))
        if order.picked_up:
            return self._refuse(StepOutcome(
                ok=False, code="already_collected",
                message="You already have this order."))
        gap = math.dist(self.position(), order.pickup.kerb)
        if gap > ARRIVAL_TOLERANCE_CM:
            return self._refuse(StepOutcome(
                ok=False, code="not_at_pickup",
                message=f"You are not standing at {order.pickup.text}.",
            ))
        self.turns += 1
        order.picked_up = True
        # Handling time is charged here, like walking charges its own. Both
        # ``collect`` and ``hand_over`` declared 30 s and neither advanced the
        # clock, so 60 s a delivery -- 10 minutes over a ten-order shift -- was
        # free, and ``_make_orders`` had already budgeted for it when it sized
        # the deadlines. A tool must not declare a cost it does not pay; that is
        # what ``_charge`` exists to guarantee, and these two bypass it.
        # Handling, plus whatever this body costs to stop with: nothing on
        # foot, a stand and a lock on a scooter, somewhere to leave a car.
        # Without that a car strictly dominates and the vehicle is not a
        # choice.
        self.sim_seconds += HANDLING_SECONDS + self.embodiment.stop_overhead_s
        self._issue()
        return StepOutcome(ok=True, sim_seconds=HANDLING_SECONDS, reward=0.1,
                           message=f"You collect the order from {order.pickup.text}.")

    def hand_over(self) -> StepOutcome:
        order = (self._order_at_hand(collected=True) or self._nearest_live(collected=True)
                 or self.active_order())
        if order is None:
            return self._refuse(StepOutcome(
                ok=False, code="no_order", message="You have no job in hand."))
        if not order.picked_up:
            return self._refuse(StepOutcome(
                ok=False, code="not_collected",
                message=f"You have not collected it yet, from {order.pickup.text}."))
        gap = math.dist(self.position(), order.dropoff.kerb)
        if gap > ARRIVAL_TOLERANCE_CM:
            return self._refuse(StepOutcome(
                ok=False, code="not_at_dropoff",
                message=f"You are not standing at {order.dropoff.text}.",
            ))
        self.turns += 1
        # The handover takes 30 s and the customer is not served until it is
        # done, so the clock moves before lateness is judged. See ``collect``.
        # Handling, plus whatever this body costs to stop with: nothing on
        # foot, a stand and a lock on a scooter, somewhere to leave a car.
        # Without that a car strictly dominates and the vehicle is not a
        # choice.
        self.sim_seconds += HANDLING_SECONDS + self.embodiment.stop_overhead_s
        order.delivered = True
        order.delivered_at_s = self.sim_seconds
        on_time = self.sim_seconds <= order.due_at()
        # A late parcel is still a parcel, and it is still worth something --
        # but not what an on-time one is worth. Paying the full fee whatever the
        # clock said made the deadline decorative under an objective that is
        # explicitly "maximise profit": a policy could ignore every window and
        # lose nothing it was scored on.
        order.paid = order.fee if on_time else round(order.fee * LATE_FEE_FRACTION, 2)
        self.earnings += order.paid
        # Take another job the moment a hand is free, so a queue that is meant
        # to be ``queue_depth`` deep actually stays that deep.
        self._issue()
        self.finished = not self.live_orders() or self.shift_over
        return StepOutcome(
            ok=True, sim_seconds=HANDLING_SECONDS,
            reward=1.0 + (0.5 if on_time else -0.5),
            finished=self.finished,
            message=(
                f"You hand the order to the customer at {order.dropoff.text}"
                f"{'' if on_time else ', late'}. You are paid {order.paid:.2f}."
            ),
        )

    WAIT_SECONDS = 15.0

    def wait(self) -> StepOutcome:
        """Wait where you are -- at a crossing, for the light.

        At a signalised junction this waits *for the phase*, not for a fixed 15
        seconds. A flat 15 s against a 60 s phase meant one ``wait()`` usually
        left the light exactly as red as it was, so obeying a light cost between
        one and four turns and the courier could not tell in advance which. That
        made the only tool the photographs exist to support unusable in practice:
        a policy that could read the lamp still had no reliable way to act on it.
        Waiting out the phase is one turn, always works, and costs the time it
        really costs -- 0 to 60 s, 30 s on average.
        """
        if self.node_id in self.signalised:
            self.waits_at_red += 1
            phase_end = (math.floor(self.sim_seconds / SIGNAL_PHASE_S) + 1) * SIGNAL_PHASE_S
            seconds = max(phase_end - self.sim_seconds, 1.0)
        else:
            seconds = self.WAIT_SECONDS
        self.turns += 1
        self.sim_seconds += seconds
        self._issue()
        return StepOutcome(ok=True, sim_seconds=seconds,
                           message=f"You wait {seconds:.0f} s.")

    REST_SECONDS = 60.0

    def rest(self) -> StepOutcome:
        """Spend a minute to get energy back.

        The counterpart to the stamina drain, and without it the drain is not a
        resource but a decay: an hour-long shift simply got slower and there was
        nothing for the courier to decide. With it, a small tank and a fast drain
        cost *turns* -- which is what makes one body different from another in
        the only currency the benchmark reports.

        A full tank's worth is recovered per rest, so one call is always enough,
        for the same reason ``wait()`` sees the whole phase out: a tool the agent
        has to guess the repeat count of is a tool it cannot plan with.
        """
        capacity = float(self.embodiment.stamina or 0.0)
        # Refuse on the same condition the tool is gated on, or a body that is
        # not offered a rest can still take one by guessing the name.
        if not (capacity and self.embodiment.stamina_per_m):
            return self._refuse(StepOutcome(
                ok=False, code="never_tires",
                message="You are not the one doing the work; there is nothing to rest.",
            ))
        before = self.stamina
        self.stamina = capacity
        self.turns += 1
        self.sim_seconds += self.REST_SECONDS
        self.rests += 1
        self._issue()
        return StepOutcome(
            ok=True, sim_seconds=self.REST_SECONDS,
            message=(
                f"You stop for a minute. You feel ready again."
                if before < capacity else
                "You stop for a minute, though you were not tired."
            ),
        )

    def light_here(self, k: int) -> str | None:
        """Ground truth for the light facing candidate ``k``.

        Privileged: it scores the run and picks which baked frame to serve. It is
        never rendered into the observation text.
        """
        if self.node_id not in self.signalised:
            return None
        rows = {row["k"]: row for row in self.candidates()}
        if k not in rows:
            return None
        return signal_state(self.node_id, rows[k]["bearing"], self.sim_seconds)

    def allowed_tool_names(self) -> list[str]:
        """Tool names this condition permits, for building the prompt.

        The condition gated ``check_map`` at call time but never reached the tool
        menu, so a no-phone episode still opened by telling the courier to use a
        phone it did not have -- and the policy duly wasted its first turn on it.
        """
        from embodiedbench.agent.courier.tools import available_tools

        env_actions = ["VIEW_ORDERS", "ACCEPT_ORDER", "PICKUP", "DROP_OFF", "WAIT", "MOVE_TO"]
        # A body that never tires must not be offered a rest: a tool that
        # can only be refused is a turn the agent is invited to lose.
        if self.embodiment.stamina and self.embodiment.stamina_per_m:
            env_actions.append("REST")
        allow_consult = self.condition == Condition.FULL
        if self.stride == Stride.BLOCK:
            # ``follow_street`` exists to spend one turn on several waypoints of
            # one street. Under this stride ``walk_to`` already does that, and
            # offering both would put two names on one action -- the same defect
            # the tool audit retired ``read_sign`` for, arrived at from the other
            # direction.
            return [
                t.name for t in available_tools(env_actions, allow_consult=allow_consult)
                if t.name != "follow_street"
            ]
        # ``note`` is back, and this time it runs. It was removed because it
        # was advertised with no executor anywhere -- the defect the
        # UNIMPLEMENTED_TOOLS split exists to make impossible. CourierSession
        # now executes it itself, because a notebook is the courier's and not
        # the city's, so the name is dispatchable again.
        names = [t.name for t in available_tools(env_actions, allow_consult=allow_consult)]
        if allow_consult and "note" not in names:
            names.append("note")
        return names

    def summary(self) -> dict[str, Any]:
        """Everything needed to judge a run, including how it was judged.

        Three things were missing and each hid a different failure. Route
        quality: only the oracle counted the metres it walked, so a policy that
        delivered by wandering twice as far scored the same as one that went
        straight there. Wasted effort: refused actions were invisible, so a
        policy spending a third of its turns walking into walls read as clean.
        And rate: ENDLESS is scored on money against a fixed hour, and money per
        hour is the objective itself -- reporting the total without the clock it
        was earned against makes two runs of different lengths look comparable.
        """
        hours = self.sim_seconds / 3600.0 if self.sim_seconds > 0 else 0.0
        walked_m = self.walked_cm / 100.0
        done_seconds, done_walk_cm = self.delivered_optimal()
        return {
            "node": self.node_id, "street": self.street_of(self.node_id) if self.node_id else "",
            "sim_seconds": round(self.sim_seconds, 1), "earnings": round(self.earnings, 2),
            # The score for ENDLESS, and an honest ledger everywhere else: what
            # the shift actually paid, after late jobs are discounted and
            # abandoned ones pay nothing.
            "profit": round(self.earnings, 2),
            "orders": [o.to_dict() for o in self.orders],
            # The body, and whether it was shown its own viewpoint. A run where
            # ``viewpoint_matches_embodiment`` is false is still a run, but it is
            # not evidence about a policy that has to work from a pavement.
            "embodiment": self.embodiment.name,
            "viewpoint_expected": self.embodiment.viewpoint,
            "viewpoint_served": self.viewpoint_served,
            "viewpoint_matches_embodiment": self.viewpoint_matches_embodiment,
            "rests": self.rests,
            "stamina_left": round(self.stamina, 2),
            "stamina_spent": round(
                float(self.embodiment.stamina or 0.0) - self.stamina, 2),
            "tired": self.tired,
            "difficulty": self.difficulty, "turns": self.turns,
            "queue_depth": self.queue_depth,
            "minutes_per_delivery": (
                round(self.sim_seconds / 60.0 / self.delivered_count, 2)
                if self.delivered_count else None
            ),
            "turns_per_delivery": (
                round(self.turns / self.delivered_count, 1)
                if self.delivered_count else None
            ),
            "delivered": self.delivered_count, "on_time": self.on_time_count,
            "late": self.late_count, "expired": self.expired_count,
            "orders_issued": self.issued_count,
            "rejected_actions": self.rejected_actions,
            "walked_m": round(walked_m, 1),
            # Optimal for the deliveries *actually completed*, not for the shift
            # as issued -- so it is only a yardstick once something has arrived.
            # It read 0.0 mid-episode, which looks like a measured bound of zero
            # and made ``walked_m vs optimal_walk_m`` nonsense on any live
            # dashboard. ``None`` says "not yet defined", which is the truth.
            "optimal_walk_m": (round(done_walk_cm / 100.0, 1)
                               if done_walk_cm else None),
            # Metres walked against metres a perfect courier would have walked
            # for the very same deliveries. 1.0 is a straight line to every
            # door; 2.0 means half the shift was spent lost. Below 1.0 is legal
            # and informative -- it is a queue served in a better order than the
            # one it was drawn in.
            "walk_ratio": (round(walked_m / (done_walk_cm / 100.0), 2)
                           if done_walk_cm else None),
            "optimal_seconds": round(done_seconds, 1),
            "time_ratio": (round(self.sim_seconds / done_seconds, 2)
                           if done_seconds else None),
            "shift_optimal_seconds": round(self.optimal_seconds, 1),
            "deliveries_per_hour": (round(self.delivered_count / hours, 2) if hours else None),
            "earnings_per_hour": (round(self.earnings / hours, 2) if hours else None),
            "shift_seconds": self.shift_seconds, "shift_over": self.shift_over,
            "red_crossings": self.red_crossings, "waits_at_red": self.waits_at_red,
            "signalised_junctions": len(self.signalised),
            # The obstacle ledger. ``blocked_attempts`` is the number a policy
            # that looks at its photographs drives to zero and one that does not
            # cannot: every one of them is a wasted turn and 45 s that the same
            # route, chosen from the same information, would not have cost.
            "blocked_attempts": self.blocked_attempts,
            "slow_passages": self.slow_passages,
            "obstacles": self.obstacles.counts(),
            "finished": self.finished,
        }


def load_city(map_dir: Path, **kwargs: Any) -> CourierEnv:
    """Compile a map and open a courier world on it."""
    network = build_road_network(Path(map_dir), map_name=Path(map_dir).name)
    return CourierEnv(network, **kwargs)


def _charged(name: str):
    """Public tool that charges its own declared time."""

    def call(self, *args, **kwargs):
        return self._charge(getattr(self, f"_{name}_impl")(*args, **kwargs))

    call.__name__ = name
    return call


for _name in ("look", "check_order", "check_map", "navigate"):
    setattr(CourierEnv, _name, _charged(_name))
