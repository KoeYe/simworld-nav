"""Draw the phone's map: streets, a route on them, and where you are standing.

``navigate()`` spoke its directions and showed nothing, which is not what a
phone does. A map app draws you a picture — the streets around you, the line you
are meant to follow, a pin on the destination, your own position with the
direction you are facing — and the picture is most of why the app is useful. A
courier glances at it and knows whether the next turn is the first or the third,
which a spoken list of five legs does not tell you.

Everything here is drawn from the compiled road network and nothing else, so it
works on any map the compiler can build and needs no renderer, no engine and no
assets. That also fixes what it is: **a map, not a window**. It shows what a
survey knows — geometry, names, the route — and it cannot show a light, a
barrier, a skip or a shopfront, because a map app cannot see the street. That
line is the whole design of this environment's tool set and this image must not
cross it. What the courier sees out of its eyes stays in the photographs.

Rendered as SVG on purpose. It is text, so it costs nothing to store beside a
trajectory and can be diffed; it scales without going soft when a vision model
resizes it; and it needs no image library on the path that produces it. Callers
that want pixels rasterise once at the edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

# How much of the city the phone shows. A map app frames the route with a
# margin, and clamps how far it will zoom out so a long route stays a route
# rather than a hairline across the whole district.
MARGIN_FRACTION = 0.12
MIN_SPAN_CM = 12000.0     # never zoom in past ~120 m across: context matters
MAX_SPAN_CM = 90000.0     # never zoom out past ~900 m: the line must stay readable
# Drawing size. 4:3 to match the photographs, so a model resizing both sees them
# at the same scale.
WIDTH_PX, HEIGHT_PX = 720, 540
# Streets near the route are drawn; the rest of the city is not, or a dense map
# reads as noise. Measured in multiples of the framed span.
CONTEXT_PAD = 0.25


def _round_scale(span_cm: float) -> tuple[float, str]:
    """A scale bar length that is a round number of metres, and its label."""
    target = span_cm * 0.25 / 100.0
    for step in (10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if target <= step:
            return step * 100.0, f"{step} m"
    return 100000.0, "1 km"


@dataclass
class MapView:
    """The window on the city this drawing covers, in map centimetres."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    width_px: int = WIDTH_PX
    height_px: int = HEIGHT_PX

    @property
    def span_cm(self) -> float:
        return max(self.max_x - self.min_x, self.max_y - self.min_y, 1.0)

    def to_px(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map centimetres to drawing pixels, with north up.

        **This map's north is +x and its east is +y.** Not a convention anyone
        would choose, but it is the one ``bearing_deg`` and ``compass_of``
        agree on -- ``bearing_deg((0,0),(1,0))`` is 0 degrees and
        ``compass_of(0)`` is "north" -- and the whole environment speaks it, so
        the drawing has to as well.

        The first version of this assumed the ordinary +y-is-north and flipped y
        accordingly, which drew the city rotated 90 degrees under a compass rose
        pointing the wrong way: every heading in the spoken route disagreed with
        the picture beside it. It survived its own unit test because the test was
        written from the same assumption. The test now derives the convention
        from ``bearing_deg`` instead of restating it.

        So: north (+x) goes up the drawing, east (+y) goes right.
        """
        scale = self.scale_px_per_cm()
        centre_x = (self.min_x + self.max_x) / 2.0
        centre_y = (self.min_y + self.max_y) / 2.0
        return (self.width_px / 2.0 + (point[1] - centre_y) * scale,
                self.height_px / 2.0 - (point[0] - centre_x) * scale)

    def scale_px_per_cm(self) -> float:
        """Pixels per map centimetre. One scale for both axes -- see ``to_px``:
        the drawing's width spans the map's y and its height spans the map's x.
        """
        return min(self.width_px / max(self.max_y - self.min_y, 1.0),
                   self.height_px / max(self.max_x - self.min_x, 1.0))


def frame_view(points: Iterable[tuple[float, float]], *,
               width_px: int = WIDTH_PX, height_px: int = HEIGHT_PX) -> MapView:
    """The window that holds every given point, with a margin, squared up."""
    points = list(points)
    if not points:
        return MapView(-MIN_SPAN_CM / 2, -MIN_SPAN_CM / 2,
                       MIN_SPAN_CM / 2, MIN_SPAN_CM / 2, width_px, height_px)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    centre_x, centre_y = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    # One span for both axes, then fitted to the drawing's aspect. Framing each
    # axis separately stretches the city, and a stretched map turns a right
    # angle into something the courier cannot match to the corner it is on.
    # The drawing is wider than it is tall and its width spans the map's y, so
    # the y extent is the one that gets the extra room.
    span = max(max(ys) - min(ys), (max(xs) - min(xs)) * width_px / height_px)
    span = max(MIN_SPAN_CM, min(MAX_SPAN_CM, span * (1.0 + 2 * MARGIN_FRACTION)))
    half_y = span / 2.0
    half_x = span / 2.0 * height_px / width_px
    return MapView(centre_x - half_x, centre_y - half_y,
                   centre_x + half_x, centre_y + half_y, width_px, height_px)


@dataclass
class MapDrawing:
    """One rendered map, and what went into it."""

    svg: str
    view: MapView
    streets_drawn: int = 0
    route_metres: float = 0.0
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "svg": self.svg, "streets_drawn": self.streets_drawn,
            "route_metres": round(self.route_metres, 1),
            "labels": list(self.labels),
            "span_m": round(self.view.span_cm / 100.0, 1),
        }


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def _halo_text(x: float, y: float, text: str, cls: str = "ui",
               anchor: str = "middle", rotate: float | None = None) -> str:
    """A label drawn twice: once as a thick background-coloured outline, once
    filled on top.

    The one-element version used ``paint-order="stroke"``, which is the tidy way
    to say it and is not honoured by every rasteriser -- cairosvg paints the
    stroke last, so every label on the first map came out as a smear of
    background colour. Two elements is uglier and works everywhere, and a name
    a courier cannot read is not a label.
    """
    turn = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate is not None else ""
    common = f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{turn}'
    body = _esc(text)
    return (f'<text class="{cls} halo" {common}>{body}</text>'
            f'<text class="{cls}" {common}>{body}</text>')


def _label_anchor(polyline: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Where to write a street's name, and at what angle, in drawing pixels.

    The middle of its longest drawn run, rotated to lie along it -- which is how
    a map labels a street and why the name is readable without a legend.
    """
    best = (0.0, polyline[0], polyline[0])
    for a, b in zip(polyline, polyline[1:]):
        length = math.dist(a, b)
        if length > best[0]:
            best = (length, a, b)
    _, a, b = best
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    # Never upside down: a name at 170 degrees reads as mirror writing.
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, angle)


def render_map(
    network: Any,
    *,
    here: tuple[float, float],
    facing_deg: float | None = None,
    route: list[tuple[float, float]] | None = None,
    destination: tuple[float, float] | None = None,
    destination_label: str = "",
    here_label: str = "you are here",
    blocked: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    width_px: int = WIDTH_PX,
    height_px: int = HEIGHT_PX,
) -> MapDrawing:
    """Draw the map a courier would be looking at.

    ``route`` and ``destination`` are optional: with neither, this is the "where
    am I" view a courier gets from opening the app without asking for anything.

    ``blocked`` is kept for callers that want to draw a street as shut. The
    courier environment never passes it: a rider does not file a report with the
    map app, so the phone is never told about a barrier and cannot draw one. What
    the map shows is the survey, and going round is worked out from the
    photographs.
    """
    route = route or []
    interest = [here] + list(route)
    if destination is not None:
        interest.append(destination)
    view = frame_view(interest, width_px=width_px, height_px=height_px)
    pad = view.span_cm * CONTEXT_PAD

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" '
        f'width="{width_px}" height="{height_px}" role="img" '
        f'aria-label="map of the streets around the courier">',
        '<defs><style>'
        '.bg{fill:#e8e5dd}.blk{fill:#d8d4c9;stroke:#cfcabd;stroke-width:.6}'
        '.st{stroke:#ffffff;stroke-linecap:round;stroke-linejoin:round;fill:none}'
        '.nm{font:500 9px ui-sans-serif,sans-serif;fill:#8a8578}'
        '.rt{stroke:#2f6ee0;stroke-width:6;stroke-linecap:round;stroke-linejoin:round;fill:none;opacity:.92}'
        '.rtc{stroke:#ffffff;stroke-width:9;stroke-linecap:round;stroke-linejoin:round;fill:none}'
        '.shut{stroke:#c8342f;stroke-width:4;stroke-linecap:round;fill:none}'
        '.ui{font:600 11px ui-sans-serif,sans-serif;fill:#3a3730}'
        '.pin{fill:#c8342f}.me{fill:#2f6ee0}'
        '.halo{stroke:#e8e5dd;stroke-width:3.4;stroke-linejoin:round;fill:none}'
        '</style></defs>',
        f'<rect class="bg" width="{width_px}" height="{height_px}"/>',
    ]

    # ── the blocks between the streets ───────────────────────────────────────
    # A map is mostly this. Without them the drawing is white lines on a flat
    # ground and gives a courier nothing to match against what it can see; with
    # them, the shape of a corner on the screen is the shape of the corner it is
    # standing on. Drawn first so everything else sits on top.
    for building in getattr(network, "buildings", []):
        min_x, min_y, max_x, max_y = building.box
        if (max_x < view.min_x - pad or min_x > view.max_x + pad
                or max_y < view.min_y - pad or min_y > view.max_y + pad):
            continue
        # Top-left on screen is (max north, min east) = (max_x, min_y).
        x0, y0 = view.to_px((max_x, min_y))
        x1, y1 = view.to_px((min_x, max_y))
        if x1 - x0 < 1.5 or y1 - y0 < 1.5:
            continue
        parts.append(f'<rect class="blk" x="{x0:.1f}" y="{y0:.1f}" '
                     f'width="{x1-x0:.1f}" height="{y1-y0:.1f}" rx="1"/>')

    # ── the streets ──────────────────────────────────────────────────────────
    drawn = 0
    labelled: set[str] = set()
    labels: list[str] = []
    for street in getattr(network, "streets", []):
        # Kept or dropped whole, by whether the street's extent touches the
        # window -- not by which of its *vertices* land inside it. The source
        # splines carry 120 vertices between 59 streets, so a street can cross
        # the whole drawing with both its vertices outside: filtering by vertex
        # deleted exactly those, and the map came out with the route running
        # across blank ground while every street it actually follows was
        # missing. The SVG viewBox clips the overhang for free.
        if len(street.polyline) < 2:
            continue
        xs = [p[0] for p in street.polyline]
        ys = [p[1] for p in street.polyline]
        if (max(xs) < view.min_x - pad or min(xs) > view.max_x + pad
                or max(ys) < view.min_y - pad or min(ys) > view.max_y + pad):
            continue
        points = list(street.polyline)
        pixels = [view.to_px(p) for p in points]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(pixels))
        width = max(3.0, min(14.0, street.width_cm * view.scale_px_per_cm()))
        parts.append(f'<path class="st" style="stroke-width:{width:.1f}" d="{path}"/>')
        drawn += 1
        if street.name not in labelled and len(pixels) >= 2:
            # Anchored on the longest run that is actually on screen, so a
            # street entering the corner of the drawing is still named.
            visible = [q for q in pixels
                       if -20 < q[0] < width_px + 20 and -20 < q[1] < height_px + 20]
            x, y, angle = _label_anchor(visible if len(visible) >= 2 else pixels)
            if 0 < x < width_px and 0 < y < height_px:
                labelled.add(street.name)
                labels.append(street.name)
                parts.append(_halo_text(x, y, street.name, cls="nm", rotate=angle))

    # ── what the courier has told the phone is shut ──────────────────────────
    for a, b in (blocked or []):
        ax, ay = view.to_px(a)
        bx, by = view.to_px(b)
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        parts.append(f'<line class="shut" x1="{mx-6:.1f}" y1="{my-6:.1f}" '
                     f'x2="{mx+6:.1f}" y2="{my+6:.1f}"/>')
        parts.append(f'<line class="shut" x1="{mx-6:.1f}" y1="{my+6:.1f}" '
                     f'x2="{mx+6:.1f}" y2="{my-6:.1f}"/>')

    # ── the route ────────────────────────────────────────────────────────────
    metres = 0.0
    if len(route) >= 2:
        metres = sum(math.dist(a, b) for a, b in zip(route, route[1:])) / 100.0
        pixels = [view.to_px(p) for p in route]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(pixels))
        parts.append(f'<path class="rtc" d="{path}"/>')
        parts.append(f'<path class="rt" d="{path}"/>')

    # ── the destination ──────────────────────────────────────────────────────
    if destination is not None:
        x, y = view.to_px(destination)
        parts.append(
            f'<path class="pin" d="M{x:.1f},{y:.1f} l-6,-9 a7,7 0 1,1 12,0 z"/>'
            f'<circle cx="{x:.1f}" cy="{y-11:.1f}" r="2.6" fill="#eceae4"/>')
        if destination_label:
            parts.append(_halo_text(x, y - 20, destination_label))

    # ── the courier ──────────────────────────────────────────────────────────
    x, y = view.to_px(here)
    if facing_deg is None:
        parts.append(f'<circle class="me" cx="{x:.1f}" cy="{y:.1f}" r="6"/>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="#ffffff"/>')
    else:
        # An arrowhead pointing the way the courier faces. The glyph is drawn
        # pointing right, bearing 0 is north and north is up, so a bearing turns
        # into a screen angle by subtracting the quarter turn between them.
        angle = facing_deg - 90.0
        parts.append(
            f'<g transform="translate({x:.1f},{y:.1f}) rotate({angle:.1f})">'
            f'<circle class="me" r="9" opacity="0.25"/>'
            f'<path class="me" d="M11,0 L-6,-6.5 L-3,0 L-6,6.5 Z"/></g>')
    if here_label:
        # Offset below the marker, because the street name it sits on is drawn
        # along the street and the two collided.
        parts.append(_halo_text(x, y + 22, here_label))

    # ── scale bar and north ──────────────────────────────────────────────────
    bar_cm, bar_text = _round_scale(view.span_cm)
    bar_px = bar_cm * view.scale_px_per_cm()
    bx, by = 14.0, height_px - 16.0
    parts.append(f'<line x1="{bx}" y1="{by}" x2="{bx+bar_px:.1f}" y2="{by}" '
                 f'stroke="#3a3730" stroke-width="2"/>'
                 f'<line x1="{bx}" y1="{by-4}" x2="{bx}" y2="{by+4}" '
                 f'stroke="#3a3730" stroke-width="2"/>'
                 f'<line x1="{bx+bar_px:.1f}" y1="{by-4}" x2="{bx+bar_px:.1f}" y2="{by+4}" '
                 f'stroke="#3a3730" stroke-width="2"/>'
                 + _halo_text(bx, by - 8, bar_text, anchor="start"))
    nx, ny = width_px - 22.0, 26.0
    parts.append(f'<g transform="translate({nx},{ny})">'
                 f'<path d="M0,-11 L5,7 L0,3 L-5,7 Z" fill="#3a3730"/>'
                 + _halo_text(0, 20, "N") + '</g>')
    if metres:
        parts.append(_halo_text(14, 22, f"route {metres:.0f} m", anchor="start"))
    parts.append("</svg>")

    return MapDrawing(svg="".join(parts), view=view, streets_drawn=drawn,
                      route_metres=metres, labels=labels)
