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
this junction numbered 1..n, and one photograph per numbered street: what you
see looking that way. Where the next junction is metres off, or the street
turns, that is mostly the building opposite — a view that does not reach, not an
empty street.

LOOK AT THE PHOTOGRAPHS BEFORE YOU ACT. The text will never tell you the colour
of a pedestrian light, what is standing in your way, or what a shopfront says.
Those are in the pictures and nowhere else. Where a crossing has a pedestrian
light you can see, a separate photograph of it is shown, labelled [light k], and
that lamp — not any light in the street views, which are older photographs — is
the one governing your crossing. Crossing while it is red costs you time and
counts against you; wait() sees the phase out.

Streets get blocked and streets get congested. Roadworks, a barrier or a skip
can shut a street completely — you cannot walk it at all, and finding that out
by trying costs you a turn and about three quarters of a minute before you are
back where you started. Furniture crowding the pavement does not stop you but
slows you down by about the same. Which streets, and where, changes from shift
to shift, and is written in no list, no order slip and no route: look down each
street's photograph before you take it.{blocked_advice}

{tool_menu}

How the streets work here:
  - walk_to(k) takes the street numbered k, and photograph k is the view down it.
  - Each street is listed both by compass heading and by where it lies relative to
    the way you face, so "turn left onto Rue X" is something you can act on.
  - House numbers run in order along a street, odd one side and even the other; if
    they are falling and you want a higher one, turn around.
  - A street keeps its name from junction to junction, so a turn shows a new name.
  - The same street name can leave a junction twice — once each way. They are
    different directions along one street, not two streets.

What you have been trained to do:
{procedures}

HOW TO REPLY — exactly this shape, every turn:

THOUGHT: one or two lines saying what you read and what you concluded.
```
walk_to(2)
```

  - Exactly one fenced block, containing exactly one call, and nothing after it.
  - The name must be one of the tools listed above.
  - Whole numbers go bare: {number_example}.
  - Any text argument goes in double quotes.
  - A tool with no arguments still needs its brackets: {no_arg_example}"""

OBSERVATION_TEMPLATE = """{memory}

### where you are
{location}
{clock}

### streets leaving this junction
{candidates}
(choose by the number at the start of a line; nothing else is a street)

### photographs
{photographs}
{extra}"""


def render_photographs(rows: list[dict], *, phone_map: bool = False) -> str:
    """The caption list for the images attached to this turn.

    The images arrive as an ordered list beside the text, and a model that
    cannot tell which picture is which street will read them in whatever order
    it likes. So every frame gets a caption here, in the same order the frames
    are attached, and the caption names the street number it belongs to.

    The phone's map, when there is one, is captioned last and captioned as a
    *drawing*. It is the one picture in the list that did not come through the
    courier's eyes, and a caption that let it pass for a photograph would be the
    harness telling the courier its phone can see the street.
    """
    lines: list[str] = []
    # One line for all the street views instead of one line each. The per-street
    # caption repeated the candidate line above it almost word for word -- same
    # index, same street, same relative bearing -- so a four-way junction spent
    # eight lines saying what four could. What the caption block has to preserve
    # is the *ordering contract*: which attached image is which street. That is
    # an index list, not a sentence per image.
    views = [row["k"] for row in rows if row.get("image")]
    if views:
        numbers = ", ".join(f"[{k}]" for k in views)
        lines.append(
            f"  {numbers} — the view down each of those streets, in that order"
        )
    for row in rows:
        if row.get("signal_image"):
            lines.append(
                f"  [light {row['k']}] the pedestrian light for street {row['k']}"
            )
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

REJECTED_TEMPLATE = """That did not work: {reason}

You are still where you were. Choose a different action."""

REQUIRED_FIELDS = {
    "system": {"city", "tool_menu", "procedures", "blocked_advice"},
    "observation": {"memory", "location", "clock", "candidates", "photographs", "extra"},
}


def build_system_prompt(*, city: str, tools: list[Tool]) -> str:
    """Compose the system prompt from the tools this environment really has."""
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
    number_example = "walk_to(2)"
    if "follow_street" in names:
        number_example += ", follow_street(2, 4)"
    # Likewise the no-argument example: it named check_order(), which no_phone
    # takes away, so that condition's prompt demonstrated a tool it had removed.
    no_arg = next((t.name for t in tools if not t.params), "")
    no_arg_example = f"{no_arg}()" if no_arg else "a call with empty brackets"
    return SYSTEM_TEMPLATE.format(
        city=city,
        tool_menu=render_tool_menu(tools),
        procedures=render_procedures(available=names),
        blocked_advice=blocked_advice,
        number_example=number_example,
        no_arg_example=no_arg_example,
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
    """The numbered street list.

    Each line carries what a rider reads off a corner: the street's name, how far
    the next junction is, which way it heads, and the house numbers that way. It
    deliberately does not say which one is correct -- that is the decision under
    test.
    """
    if not rows:
        return "There is no way on from here."
    lines = []
    for row in rows:
        parts = [f"  {row['k']}. {row['street']}"]
        # Relative first, compass second. A courier on a corner decides in left
        # and right; the compass is what the phone speaks, and both are needed to
        # act on a route instruction, but only one of them is what the body does.
        # Where the block stride knows how far one call carries, the line says
        # that, and the compass is the direction the *block* runs rather than its
        # first few metres. Quoting the next waypoint at block stride described a
        # different action from the one the number was attached to: a 7 m stub
        # labelled south-east carried a reviewer 61 m north-west.
        heading = row.get("reach_heading") or row.get("heading", "")
        if row.get("relative"):
            parts.append(f"{row['relative']} ({heading})")
        elif heading:
            parts.append(f"heading {heading}")
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
