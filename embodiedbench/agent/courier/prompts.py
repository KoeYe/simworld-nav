"""Prompt templates for the courier harness. Data, not code.

Kept as templates for the same reason mini-SWE-agent keeps its prompts in YAML:
the wording is the part most often changed and least often tested, so it must be
inspectable and diffable without reading control flow. Every field the templates
interpolate is listed in ``REQUIRED_FIELDS`` and checked, so a renamed field
fails loudly instead of rendering the literal ``{street}`` into the model's
context.

The system prompt is built from the environment's own tool set, so a tool this
map cannot execute is never described. That is not tidiness: the first Paris
observation told the agent to use ``MOVE(direction=...)`` on a map where MOVE was
disabled, and a policy that obeyed it was rejected on every single turn.
"""

from __future__ import annotations

from embodiedbench.agent.courier.skills import render_procedures
from embodiedbench.agent.courier.tools import Tool, render_tool_menu

SYSTEM_TEMPLATE = """You are a delivery courier working on foot in {city}. You collect parcels and
hand them to customers at street addresses, against a clock.

Each turn you are shown your notes, where you are standing, the streets leaving
this junction by name and bearing, and a photograph looking down them. Where the
next junction is metres off, or the street turns, the photograph is mostly the
building opposite — a view that does not reach, not an empty street.

{sources}

{steps}

READING THE MAP. It is a picture of the streets around you, north up. On it:

  - a BLUE LINE is your route, from where you are to where you are going;
  - a THICK BLUE ARROW leaves your position along the first stretch of that
    route — it points at the street you want next;
  - a small circle labelled "you are here" is you;
  - a red pin is the address you are heading for;
  - the streets are named on the map, written along each street.

The surest way to use it is by NAME, not by angle — but only ever a name that
is ALSO IN THE LIST above. Do it in this order:

  1. read the names the blue line runs along;
  2. go down the list of streets leaving this junction and find one of those
     names in it;
  3. walk that one, spelling it exactly as THE LIST spells it.

If none of the names on the line is in the list, you have not reached any of
them yet. Do not type a name off the map that is not in the list: it will be
refused, the turn is gone and nothing has moved. Take the street in the list
that runs most nearly the way the arrow points instead, and read the map again
from the next corner.

Use the arrow and the compass as a check on the name you picked, not instead
of it.

The route is walked one junction at a time. The line on the map crosses several
streets; you can only ever take one that leaves the corner you are on. When the
street you had in mind is not in the list, THE MAP IS NOT WRONG AND NEITHER IS
THE LIST — you simply have not reached that street yet. Take whichever street
here goes most nearly the right way, and look again from there.

Repeating a call that was just refused will be refused again for the same
reason. Nothing about the world changed in between. Read what the refusal
listed, and choose from that.

{hazards}

{tool_menu}

How the streets work here:
  - You take a street by naming it and the way you are going:
    walk_to("Rue de Grenelle", "east"). Both are written on the line above it,
    and the photograph captioned with the same two words is the view down it.
  - The same street name usually leaves a junction twice, once each way. They
    are two directions along one street, not two streets, which is why the
    bearing is part of naming one.
  - A street name means the same thing everywhere in the city. If you walked
    "Rue de Grenelle" going east an hour ago, that is the same street you are
    looking at now — so a street you have already tried is a street you can
    recognise and rule out.
  - Each street also says where it lies relative to the way you face, so
    "turn left onto Rue X" is something you can act on.
  - House numbers run in order along a street, odd one side and even the other;
    if they are falling and you want a higher one, turn around.
  - A street keeps its name from junction to junction, so a turn shows a new
    name. A new name means you have left the street you were on.

What you have been trained to do:
{procedures}

HOW TO REPLY — exactly this shape, every turn:

THOUGHT: one line saying what you read and what you concluded.
```
{reply_example}
```

  - {call_count_rule}
  - The name must be one of the tools listed above.
  - Street names and bearings go in double quotes, spelled as they are written
    in the list: {number_example}.
  - Whole numbers go bare.
  - A tool with no arguments still needs its brackets: {no_arg_example}
  - Keep the THOUGHT to one line. A reply that runs too long is cut off before
    it reaches the action, and a cut-off reply loses the turn."""

THREE_STEPS = """SO EVERY MOVE IS THE SAME THREE STEPS:
  1. Look at the map. Which way does the line leave you — which compass point?
  2. Look at the list of streets here. Which one goes that way? Take the one
     whose bearing is nearest the line, even if its name is not one you were
     expecting; street names change from junction to junction and the route
     runs through several of them.
  3. Look at that street's photograph. Red light, or blocked? Then wait() or
     take a different street. Otherwise walk it."""

HAZARD_RULES = """LOOK AT THE PHOTOGRAPHS BEFORE YOU ACT. The text will never tell you the colour
of a pedestrian light, what is standing in your way, or what a shopfront says.
Those are in the pictures and nowhere else. Where a crossing has a pedestrian
light you can see, a separate photograph of it is shown, captioned
[light: street name, bearing], and that lamp — not any light in the street views,
which are older photographs — is the one governing your crossing.
Crossing while it is red costs you time and counts against you; wait() sees the
phase out.

Streets get blocked and streets get congested. Roadworks, a barrier or a skip
can shut a street completely — you cannot walk it at all, and finding that out
by trying costs you a turn and about three quarters of a minute before you are
back where you started. Furniture crowding the pavement does not stop you but
slows you down by about the same. Which streets, and where, changes from shift
to shift, and is written in no list, no order slip and no route: look down each
street's photograph before you take it.{blocked_advice}"""
# How many calls a turn may carry. One sentence, held in a constant rather than
# written into the template, because the chunked session changes this rule and
# nothing else about the prompt: a second copy of the whole system prompt with
# one paragraph different is how a training prompt and an evaluation prompt
# drift apart without anyone deciding they should.
ONE_CALL_RULE = ("Exactly one fenced block, containing exactly one call, and "
                 "nothing after it.")

# The multi-call rule, for K > 1. It states the cost as well as the permission:
# the calls run in order and the run stops at the first refusal, so a chunk is
# a bet that every call after the first will still make sense. A model told it
# may issue K calls and not told what a refusal does to the rest will chain
# optimistically and lose the turn.
CHUNK_CALL_RULE = """Exactly one fenced block. It may hold UP TO {calls} calls, one per line,
    and nothing after it. They are carried out IN ORDER, and the turn STOPS at
    the first one that is refused -- everything you wrote after it is thrown
    away and never happens. So a wrong early call wastes the whole rest of the
    turn: chain calls only where you already know each one will be accepted,
    and write just one when you are not sure. More than {calls} calls is
    refused outright, and nothing in the reply is carried out."""

# What a chunked turn gets back: every call it made, in order, with what the
# world said to each. A single summarising sentence would lose exactly the
# thing the courier needs -- which call it was that stopped the turn.
CHUNK_FEEDBACK_TEMPLATE = """You made {count} calls. In order, this is what happened:

{lines}
{tail}"""

OBSERVATION_TEMPLATE = """{memory}

### where you are
{location}
{clock}

### streets leaving this junction
{candidates}
(take one with walk_to("street name", "bearing") — the name and the bearing
 exactly as they are written above; nothing else is a street)

### photographs
{photographs}
{extra}"""


def render_photographs(rows: list[dict], *, phone_map: bool = False) -> str:
    """The caption list for the images attached to this turn.

    The images arrive as an ordered list beside the text, and a model that
    cannot tell which picture is which street will read them in whatever order
    it likes. So every frame gets a caption here, in the same order the frames
    are attached, and the caption names the street it belongs to -- in the
    same words ``walk_to`` takes, so reading the picture and acting on it do
    not require a translation step.

    The phone's map, when there is one, is captioned last and captioned as a
    *drawing*. It is the one picture in the list that did not come through the
    courier's eyes, and a caption that let it pass for a photograph would be the
    harness telling the courier its phone can see the street.
    """
    lines: list[str] = []
    # One caption per frame, naming the street the way walk_to takes it. This
    # used to be a single index list ("[1], [2] — the view down each of those
    # streets, in that order") because a per-street caption repeated the
    # candidate line above it word for word. With the streets named rather
    # than numbered there is no index to carry the ordering contract, so the
    # caption carries it by naming, which also survives the image budget
    # dropping some of them.
    for row in rows:
        if row.get("image"):
            lines.append(f'  [{row["street"]}, {row["heading"]}] '
                         "the view down it from here")
    for row in rows:
        if row.get("signal_image"):
            lines.append(f'  [light: {row["street"]}, {row["heading"]}] '
                         "the pedestrian light for that crossing")
    if phone_map:
        lines.append(
            "  [map] your phone's map — a drawing, not a photograph: it has the "
            "streets and your route on it and cannot see anything in them"
        )
    if not lines:
        return "  (no photographs here)"
    return "\n".join(lines)

FORMAT_ERROR_TEMPLATE = """Your last reply could not be read as an action.

{error}

Reply with a short THOUGHT, then exactly one action in a fenced block:
```
walk_to(3)
```"""

TRUNCATED_TEMPLATE = """{error}

Your reply is cut off when it gets too long, and a cut-off reply loses the
turn. Lead with the action and keep the reasoning to a single line."""

# A refusal that only says no gets repeated at. Measured on 40 held-out
# episodes: after being refused, the model stopped reasoning entirely -- four
# turns in a row of a bare `walk_to("Rue de Mazarine", "north-west")` with no
# THOUGHT at all, the same call each time, until the session ended stuck. It
# had not thought the wrong thing, it had stopped thinking.
#
# So the refusal asks for the reasoning back, in the order the move is made,
# and asks for it in words before the call. It gives no answer away: the
# bearing has to be read off the map, and which street matches it is still the
# decision under test.
REJECTED_TEMPLATE = """That did not work: {reason}

You are still where you were, and nothing about the junction has changed, so
the same call will fail the same way. Work it out again in your THOUGHT before
you act:
  1. Which way does the route line leave you on the map — which compass point?
  2. Of the streets listed above, which one goes most nearly that way?
  3. Is that street's photograph clear — light green, nothing across the road?"""

REQUIRED_FIELDS = {
    "system": {"city", "tool_menu", "procedures", "blocked_advice"},
    "observation": {"memory", "location", "clock", "candidates", "photographs", "extra"},
}


def build_system_prompt(*, city: str, tools: list[Tool],
                        narration: str = "none",
                        action_chunk: int = 1) -> str:
    """Compose the system prompt from the tools this environment really has.

    ``action_chunk`` is how many calls one turn may carry. At 1 -- the default,
    and every condition that existed before chunking -- this renders exactly
    the bytes it always did; above 1 the reply rule is replaced (never
    appended to, never duplicated) by the multi-call one, because a prompt that
    said "exactly one call" *and* "up to three calls" would be a contradiction
    the policy has to guess its way out of.
    """
    # Macros are described in skills.py but no executor dispatches them, and
    # parse_reply rightly rejects a name it cannot run -- three of those in a row
    # truncates the episode. Advertising a tool that does not exist is the same
    # defect as advertising a disabled one, so they stay out of the prompt until
    # something can run them.
    names = {t.name for t in tools}
    # Advice that names a tool this condition has taken away is the same defect
    # as a menu that does: under ``no_phone`` there is no phone to tell, so the
    # sentence about telling it must go with the tool.
    # Your phone cannot see, there is nobody to tell, and it will not learn.
    # A courier that keeps taking the street the route names will keep walking
    # into the same barrier for the rest of the shift.
    blocked_advice = (
        " Your phone routes on a map. A map does not know about a skip, a"
        " barrier or roadworks, there is no way to tell it, and it will go on"
        " sending you down a street that is shut every time you ask. When the"
        " picture and the route disagree, believe the picture: take another"
        " street yourself, get past, and ask again from there."
        if "navigate" in names else
        " If a street is shut, remember it and go round; nothing will remind you."
    )
    # Even the formatting examples have to come from the live tool set. This line
    # read "walk_to(2), follow_street(2, 4)" at every stride, so the block-stride
    # prompt demonstrated the syntax of a tool the runtime would refuse.
    # One call only: the fence it lands in says "exactly one call", and a
    # second comma-joined call made the shown example a reply the parser rejects.
    number_example = 'walk_to("Rue de Grenelle", "east")'
    # Likewise the no-argument example: it named check_order(), which no_phone
    # takes away, so that condition's prompt demonstrated a tool it had removed.
    no_arg = next((t.name for t in tools if not t.params), "")
    no_arg_example = f"{no_arg}()" if no_arg else "a call with empty brackets"
    # Each setting is told the truth about itself and nothing else. A prompt
    # that keeps telling a narrated courier the colour is only in the picture
    # is teaching a rule that does not hold, and one that leaves the sentence
    # out of the visual setting removes the only warning that it does.
    if narration == "all":
        sources = (
            "EVERYTHING YOU NEED IS IN THE WORDS. The list of streets below\n"
            "carries all of it: which street the route takes, marked *** THE ROUTE\n"
            "GOES THIS WAY ***; whether a pedestrian light is red or green; and\n"
            "whether a street is blocked. The photographs are there to look at and\n"
            "you are not required to read anything out of them.")
    elif narration == "route":
        sources = (
            "WHERE EACH THING COMES FROM. Two sources, and you need both.\n\n"
            "  - THE LIST OF STREETS tells you which way to go. The street the\n"
            "    route takes is marked *** THE ROUTE GOES THIS WAY ***. You do not\n"
            "    have to work the direction out of the map.\n"
            "  - THE PHOTOGRAPHS are the only place a red light or a barrier\n"
            "    appears. Nothing in the words will ever tell you the colour of a\n"
            "    light or that a street is shut. Look before you commit.")
    else:
        sources = (
            "WHERE EACH THING YOU NEED COMES FROM. Three sources, and no one of "
            "them is\nenough to reach a door.\n\n"
            "  The map on your phone tells you WHICH WAY. It draws a line from "
            "where you are\n  to where you are going, and the streets are named "
            "on it.\n\n"
            "  The junction you are standing at tells you WHAT THE STREETS ARE "
            "CALLED. Only\n  the streets in that list exist for you this turn. A "
            "street anywhere else in\n  the city — including one further along "
            "your route — cannot be walked from\n  here, however clearly the "
            "line passes through it.\n\n"
            "  The photographs tell you WHETHER YOU CAN GO. Whether the "
            "pedestrian light is\n  red, whether a barrier is across the road, "
            "whether the pavement is choked:\n  none of that is in any text, "
            "here or anywhere.")
    # The three steps and the hazard rules were written for the visual world
    # and stayed put when only {sources} was swapped, so the narrated prompt
    # said "you are not required to read anything out of them" and then told
    # the courier six more times that the colour is in no text. A prompt that
    # contradicts itself teaches nothing; worse, it teaches the courier to
    # spend turns looking for a fact the words already gave it.
    if narration == "all":
        steps = (
            "SO EVERY MOVE IS THE SAME TWO STEPS:\n"
            "  1. Find the street in the list marked *** THE ROUTE GOES THIS "
            "WAY ***.\n"
            "  2. If its line says the pedestrian light is RED, wait(). If its "
            "line says\n     BLOCKED, that street is shut — take another and "
            "the marker will move.\n     Otherwise walk it.")
        hazards = (
            "Streets get blocked and streets get congested, and the line for "
            "that street\nsays so: BLOCKED means you cannot walk it at all. A "
            "crowded pavement is not\ncalled out and only costs you a little "
            "time. Crossing on a stated RED costs you\ntime and counts against "
            "you; wait() sees the phase out, and one wait is enough.")
    elif narration == "route":
        steps = (
            "SO EVERY MOVE IS THE SAME TWO STEPS:\n"
            "  1. Find the street in the list marked *** THE ROUTE GOES THIS "
            "WAY ***.\n"
            "  2. Look at that street's photograph. Red light, or blocked? Then "
            "wait() or\n     take a different street. Otherwise walk it.")
        hazards = HAZARD_RULES.format(blocked_advice=blocked_advice)
    else:
        steps = THREE_STEPS
        hazards = HAZARD_RULES.format(blocked_advice=blocked_advice)

    # The worked example is the rule in miniature, so it has to carry the same
    # number of calls the rule permits: a chunked prompt whose only example is
    # a single call teaches the shape it is trying to move away from. The
    # second line is another walk for the same reason the first one is -- it
    # comes out of the live tool set, not out of a tool this map may not have.
    chunk = max(1, int(action_chunk))
    call_count_rule = (ONE_CALL_RULE if chunk == 1
                       else CHUNK_CALL_RULE.format(calls=chunk))
    reply_example = (number_example if chunk == 1 else
                     f'{number_example}\nwalk_to("Rue du Bac", "north")')
    return SYSTEM_TEMPLATE.format(
        city=city,
        sources=sources,
        steps=steps,
        hazards=hazards,
        tool_menu=render_tool_menu(tools),
        procedures=render_procedures(available=names, narration=narration),
        blocked_advice=blocked_advice,
        number_example=number_example,
        no_arg_example=no_arg_example,
        call_count_rule=call_count_rule,
        reply_example=reply_example,
    )


def build_observation(
    *,
    memory: str,
    location: str,
    clock: str = "",
    candidates: str,
    photographs: str = "",
    extra: str = "",
) -> str:
    """Compose one turn's observation."""
    return OBSERVATION_TEMPLATE.format(
        memory=memory, location=location, clock=clock,
        candidates=candidates or "There is no way on from here.",
        photographs=photographs or "  (no photographs here)",
        extra=extra,
    ).rstrip() + "\n"


def render_candidates(rows: list[dict]) -> str:
    """The streets leaving this junction, named as a courier would name them.

    Each line carries what a rider reads off a corner: the street's name, how
    far the next junction is, which way it heads, and the house numbers that
    way. It deliberately does not say which one is correct -- that is the
    decision under test.

    The lines used to be numbered, and the courier chose by number. The number
    is stable within one junction and meaningless across junctions, so nothing
    the policy learned about a street survived walking to the next corner. A
    name and a bearing are the same everywhere, which is what makes "I have
    already tried that one" a thought the policy can have.
    """
    if not rows:
        return "There is no way on from here."
    lines = []
    for row in rows:
        parts = [f'  "{row["street"]}"']
        # Relative first, compass second. A courier on a corner decides in left
        # and right; the compass is what the phone speaks, and both are needed to
        # act on a route instruction, but only one of them is what the body does.
        # Where the block stride knows how far one call carries, the line says
        # that, and the compass is the direction the *block* runs rather than its
        # first few metres. Quoting the next waypoint at block stride described a
        # different action from the one the number was attached to: a 7 m stub
        # labelled south-east carried a reviewer 61 m north-west.
        heading = row.get("reach_heading") or row.get("heading", "")
        # The bearing is the second half of the street's name here: it is what
        # walk_to needs, so it is printed in the words walk_to takes rather
        # than only as scenery. The relative direction stays alongside it,
        # because a courier on a corner thinks in left and right while the
        # phone speaks compass, and acting on a route needs both.
        if heading:
            parts.append(f"going {heading}"
                         + (f", {row['relative']}" if row.get("relative") else ""))
        elif row.get("relative"):
            parts.append(str(row["relative"]))
        if row.get("reach_m") is not None:
            junctions = row.get("reach_junctions", 1)
            parts.append(
                f"{row['reach_m']:.0f} m on, {junctions} junction"
                f"{'' if junctions == 1 else 's'}, to the next choice"
            )
        elif row.get("distance_m") is not None:
            parts.append(f"next junction {row['distance_m']:.0f} m")
        if row.get("numbers"):
            parts.append(f"numbers {row['numbers']}")
        if row.get("blocked_seen"):
            # What the courier saw with its own eyes last time it tried. The
            # phone is still not told; this is memory, not routing. Without it
            # the menu after a refusal is byte-identical to the menu before,
            # and a policy re-picks the barrier -- 52 of 93 times, measured.
            parts.append("(BLOCKED — you tried this and could not get past)")
        if row.get("on_route"):
            # Under the narrated settings this is the whole of the direction
            # information, so it is stated plainly rather than hinted at.
            parts.append("*** THE ROUTE GOES THIS WAY ***")
        if row.get("told_signal") == "red":
            parts.append("pedestrian light: RED")
        elif row.get("told_signal") == "green":
            parts.append("pedestrian light: green")
        if row.get("told_blocked"):
            parts.append("BLOCKED — there is a barrier across it")
        if row.get("refused"):
            # The strongest place to put a refusal is the line being chosen
            # from. Told only in prose, it was ignored: the identical call was
            # made again immediately in 63 of 128 attempts.
            parts.append(f"(REFUSED ALREADY — {row['refused']})")
        if row.get("seen"):
            parts.append("(you have walked this before)")
        lines.append(" — ".join(parts))

    body = "\n".join(lines)
    if len(rows) == 1:
        # A dead end reads as an ordinary one-line menu, and a policy used to
        # two or three choices asks for street 2. Twenty of twenty-five
        # no_such_street refusals were exactly that.
        body += "\nThis is a dead end: street 1 is the only way on."
    return body
