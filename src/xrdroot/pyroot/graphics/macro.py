"""A pad saved under a ``.C`` name, as the macro ROOT writes: the calls that draw it again.

ROOT saves no picture for that name. Every drawing class has a
``SavePrimitive`` writing the C++ that rebuilds it, so what runs the file - the
test's ``.x``, a user's - makes the pad again rather than looking at it. This
is that idea over the members :mod:`.drawn` keeps: the arguments its
constructor takes, by member name, then a setter for each attribute it carries,
then ``Draw`` with the option it was drawn with.

A class whose constructor this does not spell out is refused by name instead of
written some other way: a macro that rebuilds an object differently is not the
pad that was saved, and a wrong picture is worse than none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .drawn import GROUPS, Drawn

__all__ = ["CORNERS", "CTORS", "write"]

#: The members a line's or a box's constructor takes, in its order.
CORNERS = ("fX1", "fY1", "fX2", "fY2")
#: Each class written, by the members its ROOT constructor takes in order.
CTORS: dict[str, tuple[str, ...]] = {
    "TLine": CORNERS, "TBox": CORNERS, "TWbox": CORNERS,
    "TText": ("fX", "fY", "fTitle"), "TLatex": ("fX", "fY", "fTitle"),
    "TMarker": ("fX", "fY", "fMarkerStyle"),
}  # fmt: skip
#: A canvas this cannot measure is written at ROOT's default window, 700 by 500.
DEFAULT_WINDOW = (700, 500)


def _literal(value: Any) -> str:
    """``value`` as C++ writes it: a number, a bool, or a string with its quotes escaped."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return str(value)


def _variable(index: int, classname: str) -> str:
    """The macro's variable for the ``index``-th object: a C name, and one no other holds."""
    letters = "".join(letter for letter in classname if letter.isalnum())
    return f"{letters.lower()}{index}"


def _function(filename: str) -> str:
    """The macro's function: the file's name without its suffix, as ``.x`` finds it again."""
    stem = Path(filename).stem or "pad"
    return "".join(letter if letter.isalnum() else "_" for letter in stem)


def _classname(obj: Any) -> str:
    """What ``obj`` is: a drawing class written here, or a wrapper of one, which says its name."""
    return obj.classname if isinstance(obj, Drawn) else str(obj.ClassName())


def _object(obj: Any, index: int) -> list[str]:
    """The lines that make ``obj``, as ROOT's ``SavePrimitive`` of it would."""
    classname = _classname(obj)
    if classname not in CTORS:
        from ...errors import UnsupportedFeatureError

        known = ", ".join(sorted(CTORS))
        raise UnsupportedFeatureError(
            f"A pad drawing a {classname} is not yet written as the C macro the '.C' name asks "
            f"for: xrdroot writes the calls that draw a pad again for {known}, and knows no "
            "constructor for this one. Save the pad as a picture (.png, .pdf, .ps, .svg) "
            "instead."
        )
    variable = _variable(index, classname)
    arguments = ", ".join(_literal(obj.members[member]) for member in CTORS[classname])
    lines = [f"   {classname} *{variable} = new {classname}({arguments});"]
    for group in obj.groups:
        for setter, (member, _style, _kind) in GROUPS[group].items():
            lines.append(f"   {variable}->Set{setter}({_literal(obj.members[member])});")
    return lines


def _canvas(pad: Any) -> str:
    """The canvas at ``pad``'s place and size, whose name and title are ``pad``'s."""
    members = pad.GetCanvas().members
    across, down = (int(members.get(key, default)) for key, default in zip(
        ("fWindowWidth", "fWindowHeight"), DEFAULT_WINDOW, strict=True,
    ))  # fmt: skip
    left, up = (int(members.get(key, 0)) for key in ("fWindowTopX", "fWindowTopY"))
    given = ", ".join(_literal(members[key]) for key in ("fName", "fTitle"))
    return f"   TCanvas *canvas = new TCanvas({given}, {left}, {up}, {across}, {down});"


def write(pad: Any, filename: str) -> None:
    """Write ``pad`` as the macro ``filename`` names: run it and the pad is drawn again."""
    lines = [
        f"// A macro generated from the pad {pad.members['fName']} by xrdroot.",
        "",
        f"void {_function(filename)}()",
        "{",
        _canvas(pad),
    ]
    for index, (obj, option) in enumerate(pad.primitives, 1):
        lines += _object(obj, index)
        lines.append(f"   {_variable(index, _classname(obj))}->Draw({_literal(option)});")
    Path(filename).write_text("\n".join([*lines, "}", ""]))
