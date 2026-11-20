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
    "TMathText": ("fX", "fY", "fTitle"),
    "TMarker": ("fX", "fY", "fMarkerStyle"),
    "TArrow": (*CORNERS, "fArrowSize", "fOption"),
    "TPave": (*CORNERS, "fBorderSize", "fOption"),
    "TPaveLabel": (*CORNERS, "fLabel", "fOption"),
    "TEllipse": (*ROUNDED, "fTheta"),
    "TArc": ("fX1", "fY1", "fR1", "fPhimin", "fPhimax"),
    "TCrown": ROUNDED,
    "TPaveText": (*CORNERS, "fOption"),
    #: An axis drawn on a pad: where it runs, the range it says, its divisions and its options.
    "TGaxis": (*CORNERS, "fWmin", "fWmax", "fNdiv", "fChopt"),
}  # fmt: skip
#: A member only a ``Set`` of its own reaches, which ROOT's macro writes as that ``Set``.
EXTRAS: dict[str, tuple[tuple[str, str], ...]] = {
    "TArrow": (("Angle", "fAngle"),), "TPaveText": (("Label", "fLabel"),),
    #: An axis carries its labels' and its title's size, font, colour and offset of its own,
    #: no attribute group's: they are the ``Set`` ROOT's macro writes for each.
    "TGaxis": (
        ("LabelSize", "fLabelSize"), ("LabelFont", "fLabelFont"),
        ("LabelColor", "fLabelColor"), ("LabelOffset", "fLabelOffset"),
        ("TitleSize", "fTitleSize"), ("TitleOffset", "fTitleOffset"),
        ("TickSize", "fTickSize"),
    ),
}  # fmt: skip
#: Each class that holds lines and adds them back one at a time, by the member holding them:
#: a pave of text is made empty and its lines are added to it, as ROOT's own macro adds them.
LINES: dict[str, str] = {"TPaveText": "fLines"}
#: The ``Add`` each kind of line is added with: a dividing line by its corners, a line of text
#: by the text it says and, if it was placed in the pave's fractions, by where.
ADDS = {"TLine": "AddLine", "TText": "AddText", "TLatex": "AddText"}
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
#: Each graph written from its public API: a graph keeps no members this can read by name,
#: so it is made by its count and its points set one at a time through ``SetPoint``, which
#: is how ROOT's own ``SavePrimitive`` writes the arrays its constructor was given.
GRAPHS: dict[str, tuple[str, str]] = {"TGraph": ("GetN", "SetPoint")}
#: Each graph whose points carry errors, by the calls ``SetPointError`` takes them from: a
#: symmetric error pair, or the four an asymmetric one has.
ERRORS: dict[str, tuple[str, ...]] = {
    "TGraphErrors": ("GetErrorX", "GetErrorY"),
    "TGraphAsymmErrors": ("GetErrorXlow", "GetErrorXhigh", "GetErrorYlow", "GetErrorYhigh"),
}  # fmt: skip
#: Each one-dimensional histogram written from its public API, by the axis its binning is:
#: a macro makes it from its bin count and its edges, then gives each bin its content.
HISTS = frozenset({"TH1C", "TH1S", "TH1I", "TH1F", "TH1D", "TH1L"})
#: The attribute groups a graph or a histogram is written with, in the order ROOT's macro
#: sets them: neither carries text of its own, its title being the pad's.
API_GROUPS = ("line", "fill", "marker")
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
#: Each class that holds other drawn objects, by the call that asks it for them.
HOLDS: dict[str, str] = {"TMultiGraph": "GetListOfGraphs"}
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


def _graph(obj: Any, classname: str, variable: str) -> list[str]:
    """A graph made by its count and then given each of its points, and their errors, by name."""
    count, points = GRAPHS.get(classname, ("GetN", "SetPoint"))
    total, xs, ys = int(getattr(obj, count)()), obj.GetX(), obj.GetY()
    lines = [f"   {classname} *{variable} = new {classname}({total});"]
    for at in range(min(total, len(xs), len(ys))):
        right = f"{_literal(float(xs[at]))}, {_literal(float(ys[at]))}"
        lines.append(f"   {variable}->{points}({at}, {right});")
        lines += _point_errors(obj, classname, variable, at)
    return lines


def _point_errors(obj: Any, classname: str, variable: str, at: int) -> list[str]:
    """The errors of ``obj``'s ``at``-th point, for the graph classes that carry any."""
    asked = ERRORS.get(classname)
    if not asked:
        return []
    given = ", ".join(_literal(float(getattr(obj, one)(at))) for one in asked)
    return [f"   {variable}->SetPointError({at}, {given});"]


def _api_setters(obj: Any, variable: str) -> list[str]:
    """A ``Set`` for each attribute a graph has, read back with the ``Get`` of the same name.

    A graph is not a :class:`.drawn.Drawn`, so its members are only reachable through the
    Set and Get ROOT gives it: what the pad drew is what those Get return.
    """
    lines = [
        f"   {variable}->Set{setter}({_literal(getattr(obj, f'Get{setter}')())});"
        for group in API_GROUPS
        for setter in GROUPS[group]
        if hasattr(obj, f"Get{setter}")
    ]
    if str(obj.GetTitle()):
        lines.append(f"   {variable}->SetTitle({_literal(str(obj.GetTitle()))});")
    return lines + [
        f"   {variable}->Get{axis}axis()->SetTitle({_literal(title)});"
        for axis, title in _axis_titles(obj)
    ]


def _axis_titles(obj: Any) -> list[tuple[str, str]]:
    """The axes of ``obj`` that a title was given to, and the title each draws."""
    return [
        (axis, str(getattr(obj, f"Get{axis}axis")().GetTitle()))
        for axis in "XY"
        if str(getattr(obj, f"Get{axis}axis")().GetTitle())
    ]


def _hist(obj: Any, classname: str, variable: str) -> list[str]:
    """A histogram made from its binning and then given every bin's content, and its error."""
    axis = obj.GetXaxis()
    given = ", ".join([
        _literal(str(obj.GetName())), _literal(str(obj.GetTitle())),
        str(int(obj.GetNbinsX())), _literal(float(axis.GetXmin())),
        _literal(float(axis.GetXmax())),
    ])
    lines = [f"   {classname} *{variable} = new {classname}({given});"]
    for at in range(1, int(obj.GetNbinsX()) + 1):
        lines.append(
            f"   {variable}->SetBinContent({at}, {_literal(float(obj.GetBinContent(at)))});"
        )
        error = float(obj.GetBinError(at))
        if error:
            lines.append(f"   {variable}->SetBinError({at}, {_literal(error)});")
    return lines


def _line_arguments(line: Any, kind: str) -> str:
    """The arguments a pave's line is added back with: its corners, or its text where it was put.

    A line of text no ``AddText(x, y, text)`` placed - ROOT's own macro's test, both
    coordinates zero - is the one the pave stacks from its top, so only its text is given.
    """
    members = line.members
    if kind == "TLine":
        return ", ".join(_literal(float(members[one])) for one in CORNERS)
    x, y = float(members["fX"]), float(members["fY"])
    said = _literal(str(members["fTitle"]))
    return ", ".join([_literal(x), _literal(y), said]) if x or y else said


def _added_lines(obj: Any, classname: str, variable: str) -> list[str]:
    """Each line ``obj`` holds, added back to it and then given what the pad gave it."""
    lines: list[str] = []
    for at, line in enumerate(obj.members[LINES[classname]]):
        kind = _classname(line)
        named = f"{variable}_{'line' if kind == 'TLine' else 'text'}{at}"
        given = _line_arguments(line, kind)
        lines.append(f"   {kind} *{named} = {variable}->{ADDS[kind]}({given});")
        lines += _setters(line, kind, named)
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

    known = ", ".join(sorted({*CTORS, *POINTS, *PIECES, *GRAPHS, *ERRORS, *HOLDS, *HISTS, PAD}))
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


def _known(classname: str) -> bool:
    """Whether this writes a ``classname`` at all: by a constructor, or by its public API."""
    return classname in {*CTORS, *POINTS, *PIECES, *GRAPHS, *ERRORS, PAD, *HOLDS, *HISTS}


def _object(obj: Any, counted: list[int]) -> list[str]:
    """The lines that make ``obj``, as ROOT's ``SavePrimitive`` of it would.

    ``counted`` is the macro's one counter of variables, so an object held by another - a
    graph of a multigraph - takes the next name rather than one already spoken for.
    """
    classname = _classname(obj)
    if not _known(classname):
        _unknown(classname)
    variable = _variable(counted[0], classname)
    if classname in PIECES:
        return _pieces(obj, classname, variable)
    if classname in HOLDS:
        return _holder(obj, classname, variable, counted)
    if classname in GRAPHS or classname in ERRORS:
        return _graph(obj, classname, variable) + _api_setters(obj, variable)
    if classname in HISTS:
        return _hist(obj, classname, variable) + _api_setters(obj, variable)
    lines = _made(obj, classname, variable) + _setters(obj, classname, variable)
    if classname in LINES:
        lines += _added_lines(obj, classname, variable)
    return lines


def _holder(obj: Any, classname: str, variable: str, counted: list[int]) -> list[str]:
    """Something that holds other drawn objects - a multigraph - made and then given them."""
    lines = [f"   {classname} *{variable} = new {classname}();", *_api_setters(obj, variable)]
    for held in getattr(obj, HOLDS[classname])():
        counted[0] += 1
        child = _variable(counted[0], _classname(held))
        lines += [*_object(held, counted), f"   {variable}->Add({child});"]
    return lines


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
        lines += _object(obj, counted)
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
