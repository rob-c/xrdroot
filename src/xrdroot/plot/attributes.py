"""ROOT's drawing attributes, read off an object and turned into a :class:`~.model.Look`.

Everything ROOT draws inherits ``TAttLine``, ``TAttFill`` and ``TAttMarker``:
a colour index, a style number and a size each. They are written with the
object, so a histogram styled in a macro and saved comes back styled. This
reads them wherever in the members they are inherited, translates the
numbers - colour ``2`` is red, marker ``20`` a full circle, fill ``3004``
hatched - and then lets the caller's own keywords win over every one.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from .colors import color
from .model import Frame, Look

__all__ = [
    "FRAME_WORDS",
    "LOOK_WORDS",
    "MARKERS",
    "found",
    "frame",
    "look",
    "restyled",
    "split_style",
]

#: ROOT's marker styles, as a shape and whether it is drawn hollow. What is
#: not here - a style some later ROOT added - is drawn as a full circle.
MARKERS: dict[int, tuple[str, bool]] = {
    1: ("point", False), 2: ("plus", False), 3: ("asterisk", False), 4: ("circle", True),
    5: ("x", False), 6: ("point", False), 7: ("point", False), 8: ("circle", False),
    20: ("circle", False), 21: ("square", False), 22: ("triangle-up", False),
    23: ("triangle-down", False), 24: ("circle", True), 25: ("square", True),
    26: ("triangle-up", True), 27: ("diamond", True), 28: ("cross", True),
    29: ("star", False), 30: ("star", True), 31: ("asterisk", False),
    32: ("triangle-down", True), 33: ("diamond", False), 34: ("cross", False),
    35: ("diamond", True), 36: ("square", True), 37: ("triangle-up", True),
    38: ("circle", True), 39: ("triangle-up", False), 40: ("x", True), 41: ("x", False),
    42: ("star", True), 43: ("star", False), 44: ("cross", True), 45: ("cross", False),
    46: ("x", True), 47: ("x", False), 48: ("square", False), 49: ("square", True),
}  # fmt: skip

#: The shapes a marker can be asked for by, and the spellings matplotlib
#: users reach for first, which mean the same shapes.
SHAPES = {
    "point": "point", ".": "point", "plus": "plus", "+": "plus", "asterisk": "asterisk",
    "circle": "circle", "o": "circle", "x": "x", "square": "square", "s": "square",
    "triangle-up": "triangle-up", "^": "triangle-up", "triangle-down": "triangle-down",
    "v": "triangle-down", "diamond": "diamond", "D": "diamond", "d": "diamond",
    "cross": "cross", "P": "cross", "star": "star", "*": "star",
}  # fmt: skip

#: ROOT's line styles: solid, dashed, dotted, dash-dotted, and the longer
#: dashes after them drawn as the nearest of those four.
DASHES = {1: "solid", 2: "dashed", 3: "dotted", 4: "dashdot", 5: "dashdot", 6: "dashdot",
          7: "dashed", 8: "dashdot", 9: "dashed", 10: "dashdot"}  # fmt: skip

#: The dashes a line can be asked for by, in words or matplotlib's marks.
DASH_WORDS = {"solid": "solid", "-": "solid", "dashed": "dashed", "--": "dashed",
              "dotted": "dotted", ":": "dotted", "dashdot": "dashdot", "-.": "dashdot"}  # fmt: skip

#: ROOT's hatched fill styles, ``3000`` to ``3999``, as the nearest hatch.
HATCHES = {3001: ".", 3002: ".", 3003: ".", 3004: "/", 3005: "\\", 3006: "|", 3007: "-",
           3013: "x", 3144: "x", 3244: "x", 3345: "/", 3354: "\\", 3444: "x"}  # fmt: skip

#: The keywords that restyle the layers, against the :class:`~.model.Look` field each sets.
LOOK_WORDS = {
    "color": "color", "linewidth": "width", "lw": "width", "linestyle": "dash",
    "ls": "dash", "fill": "fill", "alpha": "alpha", "hatch": "hatch", "marker": "marker",
    "markersize": "marker_size", "ms": "marker_size", "markercolor": "marker_color",
    "label": "label",
}  # fmt: skip

#: The keywords that set the frame rather than the marks in it.
FRAME_WORDS = frozenset(Frame._fields)

#: What ``fill=True`` fills with when nothing gives a colour: the line's own.
SOLID = 1001


def found(members: Any, key: str) -> dict[str, Any] | None:
    """The dictionary under ``key`` nearest the top of ``members``, however inherited.

    Breadth first, so a ``TEfficiency``'s own ``TAttLine`` is found before
    the one inside a histogram it holds.
    """
    queue: deque[Any] = deque([members])
    while queue:
        here = queue.popleft()
        if not isinstance(here, Mapping):
            continue
        held = here.get(key)
        if isinstance(held, Mapping):
            return dict(held)
        queue.extend(value for value in here.values() if isinstance(value, Mapping))
    return None


def _marker(style: int) -> tuple[str, bool]:
    return MARKERS.get(int(style), ("circle", False))


def _fill(fill_color: Any, style: int) -> tuple[str | None, float, str | None]:
    """ROOT's fill: hollow for style ``0`` or colour ``0``, hatched, translucent or solid."""
    if int(style) == 0 or int(fill_color) == 0:
        return None, 1.0, None
    if 3000 <= style < 4000:
        return color(fill_color), 1.0, HATCHES.get(int(style), "/")
    if 4000 <= style <= 4100:
        return color(fill_color), (style - 4000) / 100, None
    return color(fill_color), 1.0, None


def look(members: Any, markers: bool = False) -> Look:
    """The :class:`~.model.Look` an object's members say it is drawn in.

    ``markers`` asks for its marker even when it is ROOT's default dot,
    which is how points are drawn when a caller asks for them.
    """
    line = found(members, "TAttLine") or {}
    fill = found(members, "TAttFill") or {}
    marker = found(members, "TAttMarker") or {}
    shape, hollow = _marker(marker.get("fMarkerStyle", 1))
    filled, alpha, hatch = _fill(fill.get("fFillColor", 0), fill.get("fFillStyle", SOLID))
    return Look(
        color=color(line.get("fLineColor", 1)),
        width=float(line.get("fLineWidth", 1) or 1),
        dash=DASHES.get(int(line.get("fLineStyle", 1)), "solid"),
        fill=filled,
        alpha=alpha,
        hatch=hatch,
        marker=shape if markers else None,
        hollow=hollow,
        marker_color=color(marker.get("fMarkerColor", 1)),
        marker_size=float(marker.get("fMarkerSize", 1.0)),
    )


def _shape(value: Any) -> tuple[str, bool | None]:
    """A marker asked for: a ROOT style number, a shape's name, or matplotlib's mark."""
    if isinstance(value, int) and not isinstance(value, bool):
        return _marker(value)
    if value not in SHAPES:
        raise ValueError(
            f"marker={value!r} is not a marker drawn here: give a ROOT marker style number, "
            f"such as 20, or one of {', '.join(sorted(set(SHAPES.values())))}"
        )
    return SHAPES[value], None


def _dash(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return DASHES.get(value, "solid")
    if value not in DASH_WORDS:
        raise ValueError(
            f"linestyle={value!r} is not a line drawn here: give a ROOT line style number, "
            f"or one of solid, dashed, dotted and dashdot"
        )
    return DASH_WORDS[value]


def _filled(value: Any, base: Look) -> str | None:
    """``fill=True`` fills in the line's colour, ``False`` not at all, anything else is a colour."""
    if value is True:
        return base.fill or base.color
    if value is False:
        return None
    return color(value)


def restyled(base: Look, words: Mapping[str, Any]) -> Look:
    """``base`` with whatever the caller's keywords say instead.

    ``color`` is the line and the markers both, as it is in matplotlib;
    ``markercolor`` is the markers alone.
    """
    changes: dict[str, Any] = {}
    for word, value in words.items():
        field = LOOK_WORDS[word]
        changes.update(_change(field, value, base))
    if "color" in words and "markercolor" not in words:
        changes["marker_color"] = changes["color"]
    return base._replace(**changes)


def _change(field: str, value: Any, base: Look) -> dict[str, Any]:
    """The fields one keyword changes: a marker is its shape and whether it is hollow."""
    if field == "marker":
        shape, hollow = _shape(value)
        return {"marker": shape} if hollow is None else {"marker": shape, "hollow": hollow}
    converters: dict[str, Callable[[Any], Any]] = {
        "color": color,
        "marker_color": color,
        "dash": _dash,
        "fill": lambda given: _filled(given, base),
        "width": float,
        "alpha": float,
        "marker_size": float,
    }
    convert = converters.get(field)
    return {field: convert(value) if convert else value}


def split_style(style: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The caller's keywords sorted three ways: the marks', the frame's, and the library's own."""
    marks = {word: value for word, value in style.items() if word in LOOK_WORDS}
    framing = {word: value for word, value in style.items() if word in FRAME_WORDS}
    native = {
        word: value
        for word, value in style.items()
        if word not in LOOK_WORDS and word not in FRAME_WORDS
    }
    return marks, framing, native


def frame(base: Frame, words: Mapping[str, Any]) -> Frame:
    """``base`` with the caller's ``title``, ``xlabel``, ``logy`` and the rest instead."""
    return base._replace(**dict(words))
