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

__all__ = ["CORNERS", "CTORS", "EXTRAS", "PAD", "POINTS", "ROUNDED", "write"]

#: The members a line's or a box's constructor takes, in its order.
CORNERS = ("fX1", "fY1", "fX2", "fY2")
#: The members a round shape's constructor takes, in ROOT's order: its centre, its two
#: radii, and the angles it is cut between.
ROUNDED = ("fX1", "fY1", "fR1", "fR2", "fPhimin", "fPhimax")
#: Each class written, by the members its ROOT constructor takes in order.
CTORS: dict[str, tuple[str, ...]] = {
    "TLine": CORNERS, "TBox": CORNERS, "TWbox": CORNERS,
    "TText": ("fX", "fY", "fTitle"), "TLatex": ("fX", "fY", "fTitle"),
    "TMarker": ("fX", "fY", "fMarkerStyle"),
    "TArrow": (*CORNERS, "fArrowSize", "fOption"),
    "TPave": (*CORNERS, "fBorderSize", "fOption"),
    "TPaveLabel": (*CORNERS, "fLabel", "fOption"),
    "TEllipse": (*ROUNDED, "fTheta"),
    "TArc": ("fX1", "fY1", "fR1", "fPhimin", "fPhimax"),
    "TCrown": ROUNDED,
}  # fmt: skip
#: A member only a ``Set`` of its own reaches, which ROOT's macro writes as that ``Set``.
EXTRAS: dict[str, tuple[tuple[str, str], ...]] = {"TArrow": (("Angle", "fAngle"),)}
#: A ``Set`` ROOT's macro writes that takes more than one member: a pie's circle is its
#: centre and its radius together, and its formats are what its labels are made of.
CALLS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "TPie": (
        ("Circle", ("fX", "fY", "fRadius")),
        ("ValueFormat", ("fValueFormat",)),
        ("LabelFormat", ("fLabelFormat",)),
        ("PercentFormat", ("fPercentFormat",)),
        ("FractionFormat", ("fFractionFormat",)),
        ("LabelsOffset", ("fLabelsOffset",)),
        ("AngularOffset", ("fAngularOffset",)),
        ("Height", ("fHeight",)),
        ("Angle3D", ("fAngle3D",)),
    ),
}  # fmt: skip
#: Each class made from its points, by the members holding its count, its xs and its ys.
#: ROOT's constructor for these takes the arrays, which a macro spells out by setting
#: every point in turn: an array literal of a length only the pad knows is not the same
#: thing to read back, and a point set by name is the one the pad drew.
POINTS: dict[str, tuple[str, str, str]] = {
    "TPolyLine": ("fN", "fX", "fY"), "TPolyMarker": ("fN", "fX", "fY"),
}  # fmt: skip
#: Each class made by its count and then given its pieces one at a time, by the member
#: holding them and the setters a piece takes: a pie's constructor is told how many slices
#: it has, and each slice's title, value, offset and fill are then set on it by number.
#: ROOT's macro writes them through ``GetSlice(i)->Set…``, which is a slice's own setter.
PIECES: dict[str, tuple[str, tuple[str, ...]]] = {
    "TPie": (
        "fPieSlices",
        ("Title", "Value", "RadiusOffset", "FillColor", "FillStyle",
         "LineColor", "LineStyle", "LineWidth"),
    ),
}  # fmt: skip
#: The class of a pad inside the canvas, which a macro makes and then draws into.
PAD = "TPad"
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


def _points(obj: Any, classname: str, variable: str) -> list[str]:
    """The lines that make a polyline or a polymarker of ``obj``'s points, one point at a time."""
    count, xs, ys = POINTS[classname]
    x, y = obj.members[xs], obj.members[ys]
    lines = [f"   {classname} *{variable} = new {classname}({int(obj.members[count])});"]
    for at in range(min(int(obj.members[count]), len(x), len(y))):
        right = f"{_literal(float(x[at]))}, {_literal(float(y[at]))}"
        lines.append(f"   {variable}->SetPoint({at}, {right});")
    return lines


def _pieces(obj: Any, classname: str, variable: str) -> list[str]:
    """The lines that make ``obj`` and give each of its pieces what the pad gave it."""
    member, setters = PIECES[classname]
    pieces = obj.members[member]
    given = f"{_literal(obj.members['fName'])}, {_literal(obj.members['fTitle'])}"
    lines = [f"   {classname} *{variable} = new {classname}({given}, {len(pieces)});"]
    lines += _setters(obj, classname, variable)
    for at, piece in enumerate(pieces):
        lines += [
            f"   {variable}->GetSlice({at})->Set{setter}("
            f"{_literal(piece.members[f'f{setter}'])});"
            for setter in setters
        ]
    return lines


def _made(obj: Any, classname: str, variable: str) -> list[str]:
    """The one line that makes ``obj``: by its constructor, its points, or its place in the pad."""
    if classname in POINTS:
        return _points(obj, classname, variable)
    if classname == PAD:
        return _subpad(obj, variable)
    arguments = ", ".join(_literal(obj.members[member]) for member in CTORS[classname])
    return [f"   {classname} *{variable} = new {classname}({arguments});"]


def _unknown(classname: str) -> None:
    """Refuse a class this has no constructor for, by name, and say what to save instead."""
    from ...errors import UnsupportedFeatureError

    known = ", ".join(sorted({*CTORS, *POINTS, *PIECES, PAD}))
    raise UnsupportedFeatureError(
        f"A pad drawing a {classname} is not yet written as the C macro the '.C' name asks "
        f"for: xrdroot writes the calls that draw a pad again for {known}, and knows no "
        "constructor for this one. Save the pad as a picture (.png, .pdf, .ps, .svg) "
        "instead."
    )


def _setters(obj: Any, classname: str, variable: str) -> list[str]:
    """A ``Set`` for each attribute ``obj`` carries, its groups' first, then those of its own."""
    lines = [
        f"   {variable}->Set{setter}({_literal(obj.members[member])});"
        for group in obj.groups
        for setter, (member, _style, _kind) in GROUPS[group].items()
    ]
    lines += [
        f"   {variable}->Set{setter}({_literal(obj.members[member])});"
        for setter, member in EXTRAS.get(classname, ())
    ]
    return lines + [
        f"   {variable}->Set{setter}({', '.join(_literal(obj.members[one]) for one in members)});"
        for setter, members in CALLS.get(classname, ())
    ]


def _object(obj: Any, index: int) -> list[str]:
    """The lines that make ``obj``, as ROOT's ``SavePrimitive`` of it would."""
    classname = _classname(obj)
    if classname not in CTORS and classname not in POINTS and classname != PAD:
        if classname not in PIECES:
            _unknown(classname)
    variable = _variable(index, classname)
    if classname in PIECES:
        return _pieces(obj, classname, variable)
    return _made(obj, classname, variable) + _setters(obj, classname, variable)


def _subpad(pad: Any, variable: str) -> list[str]:
    """A pad inside the canvas: where it sits in fractions of its mother, which a macro places."""
    members = pad.members
    left, bottom = float(members["fXlowNDC"]), float(members["fYlowNDC"])
    given = ", ".join(_literal(value) for value in (
        members["fName"], members["fTitle"], left, bottom,
        left + float(members["fWNDC"]), bottom + float(members["fHNDC"]),
    ))  # fmt: skip
    return [f"   TPad *{variable} = new TPad({given});"]


def _canvas(pad: Any) -> str:
    """The canvas at ``pad``'s place and size, whose name and title are ``pad``'s."""
    members = pad.GetCanvas().members
    across, down = (int(members.get(key, default)) for key, default in zip(
        ("fWindowWidth", "fWindowHeight"), DEFAULT_WINDOW, strict=True,
    ))  # fmt: skip
    left, up = (int(members.get(key, 0)) for key in ("fWindowTopX", "fWindowTopY"))
    given = ", ".join(_literal(members[key]) for key in ("fName", "fTitle"))
    return f"   TCanvas *canvas = new TCanvas({given}, {left}, {up}, {across}, {down});"


def _contents(pad: Any, counted: list[int]) -> list[str]:
    """Everything ``pad`` holds, drawn: a pad inside it is entered, drawn into, and left again.

    ``counted`` is the one counter of the macro's variables, carried down into the pads
    drawn in this one, so that no two objects of the macro share a name.
    """
    lines: list[str] = []
    for obj, option in pad.primitives:
        counted[0] += 1
        classname, variable = _classname(obj), _variable(counted[0], _classname(obj))
        lines += _object(obj, counted[0])
        lines.append(f"   {variable}->Draw({_literal(option)});")
        if classname == PAD:
            lines.append(f"   {variable}->cd();")
            lines += _contents(obj, counted)
            lines.append("   canvas->cd();")
    return lines


def write(pad: Any, filename: str) -> None:
    """Write ``pad`` as the macro ``filename`` names: run it and the pad is drawn again."""
    lines = [
        f"// A macro generated from the pad {pad.members['fName']} by xrdroot.",
        "",
        f"void {_function(filename)}()",
        "{",
        _canvas(pad),
        *_contents(pad, [0]),
    ]
    Path(filename).write_text("\n".join([*lines, "}", ""]))
