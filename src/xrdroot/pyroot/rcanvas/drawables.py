"""The primitives a ROOT 7 pad holds: lines, boxes, text, markers, paves, axes, ROOT 6 objects.

Each is a group of attributes (:mod:`.attrs`) with its geometry among
them - ``p1`` and ``p2`` for a line or a box, ``pos`` for text - and the
members every drawable has: an ``id`` and a CSS class a style can pick it
out by, and whether it is drawn on the frame and clipped to it.
"""

from __future__ import annotations

from typing import Any

from .attrs import Attrs, frozen
from .axes import RAttrAxis
from .kinds import RAttrBorder, RAttrFill, RAttrLine, RAttrMarker, RAttrText
from .lengths import RPadLength, RPadPos, length, position

__all__ = ["RLine", "RBox", "RText", "RMarker", "RPave", "RPaveText", "RFrameTitle",
           "RAxisDrawable", "TObjectDrawable", "RFont", "Primitive"]  # fmt: skip


class Primitive(Attrs):
    """What every drawable has: an id, a CSS class, and its place on or off the frame."""

    MEMBERS = frozen({"id": lambda: "", "cssClass": lambda: "", "onFrame": lambda: False,
                      "clipping": lambda: False})  # fmt: skip

    #: The CSS type a style picks every drawable of this class out by.
    CSS_TYPE = ""

    def SetId(self, id: Any) -> Primitive:
        self.id = str(id)
        return self

    def GetId(self) -> str:
        return str(self.id)

    def SetCssClass(self, name: Any) -> Primitive:
        self.cssClass = str(name)
        return self

    def GetCssClass(self) -> str:
        return str(self.cssClass)

    def SetOnFrame(self, on: Any = True) -> Primitive:
        self.onFrame = bool(on)
        return self

    def SetClipping(self, on: Any = True) -> Primitive:
        self.clipping = bool(on)
        return self


class _Between(Primitive):
    """A primitive between two points, ``p1`` and ``p2``."""

    MEMBERS = frozen({"p1": RPadPos, "p2": RPadPos})

    def __init__(self, p1: Any = None, p2: Any = None) -> None:
        super().__init__()
        self.p1 = RPadPos() if p1 is None else position(p1)
        self.p2 = RPadPos() if p2 is None else position(p2)

    def SetP1(self, p1: Any) -> _Between:
        self.p1 = position(p1)
        return self

    def SetP2(self, p2: Any) -> _Between:
        self.p2 = position(p2)
        return self

    def GetP1(self) -> RPadPos:
        return self.p1

    def GetP2(self) -> RPadPos:
        return self.p2


class RLine(_Between):
    """``RLine``: a line from ``p1`` to ``p2``."""

    MEMBERS = frozen({"line": RAttrLine})
    CSS_TYPE = "line"


class RBox(_Between):
    """``RBox``: a box with corners ``p1`` and ``p2``, its border and its fill."""

    MEMBERS = frozen({"border": RAttrBorder, "fill": RAttrFill})
    CSS_TYPE = "box"


class RText(Primitive):
    """``RText``: a string, at ``pos``, drawn with the attributes of ``text``."""

    MEMBERS = frozen({"pos": RPadPos, "content": lambda: "", "text": RAttrText})
    CSS_TYPE = "text"

    def __init__(self, pos: Any = None, content: Any = None) -> None:
        super().__init__()
        if content is None and isinstance(pos, str):  # RText("label"): at the pad's corner
            pos, content = None, pos
        self.pos = RPadPos() if pos is None else position(pos)
        self.content = "" if content is None else str(content)

    def SetText(self, content: Any) -> RText:
        self.content = str(content)
        return self

    def GetText(self) -> str:
        return str(self.content)

    def SetPos(self, pos: Any) -> RText:
        self.pos = position(pos)
        return self

    def GetPos(self) -> RPadPos:
        return self.pos


class RMarker(Primitive):
    """``RMarker``: a marker at ``p``."""

    MEMBERS = frozen({"p": RPadPos, "marker": RAttrMarker})
    CSS_TYPE = "marker"

    def __init__(self, p: Any = None) -> None:
        super().__init__()
        self.p = RPadPos() if p is None else position(p)

    def SetP(self, p: Any) -> RMarker:
        self.p = position(p)
        return self

    def GetP(self) -> RPadPos:
        return self.p


class RPave(Primitive):
    """``RPave``: a box placed by the corner of the frame it sits in, and its size."""

    MEMBERS = frozen({
        "border": RAttrBorder, "fill": RAttrFill, "corner": lambda: 2,
        "offsetX": lambda: RPadLength(0.02), "offsetY": lambda: RPadLength(0.02),
        "width": lambda: RPadLength(0.4), "height": lambda: RPadLength(0.2),
    })  # fmt: skip
    CSS_TYPE = "pave"
    kTopLeft, kTopRight, kBottomLeft, kBottomRight = 1, 2, 3, 4


class RPaveText(RPave):
    """``RPaveText``: a pave of lines of text."""

    MEMBERS = frozen({"text": RAttrText, "lines": list})
    CSS_TYPE = "pavetext"

    def AddLine(self, line: Any) -> None:
        self.lines.append(str(line))

    def GetLine(self, index: Any) -> str:
        return str(self.lines[int(index)])

    def GetNumLines(self) -> int:
        return len(self.lines)

    def ClearLines(self) -> None:
        self.lines.clear()


class RFrameTitle(Primitive):
    """``RFrameTitle``: the title above a pad's frame, how far above it and how tall."""

    MEMBERS = frozen({"content": lambda: "", "margin": lambda: RPadLength(0.02),
                      "height": lambda: RPadLength(0.05), "text": RAttrText})  # fmt: skip
    CSS_TYPE = "title"

    def __init__(self, content: Any = "") -> None:
        super().__init__()
        self.content = str(content)

    def SetText(self, content: Any) -> RFrameTitle:
        self.content = str(content)
        return self

    def GetText(self) -> str:
        return str(self.content)


class RAxisDrawable(Primitive):
    """``RAxisDrawable``: an axis on its own - where it starts, which way, how long."""

    MEMBERS = frozen({"pos": RPadPos, "vertical": lambda: False, "length": RPadLength,
                      "labels": list, "axis": RAttrAxis})  # fmt: skip
    CSS_TYPE = "axis"

    def __init__(self, pos: Any = None, vertical: Any = False, size: Any = 0.0) -> None:
        super().__init__()
        self.pos = RPadPos() if pos is None else position(pos)
        self.vertical, self.length = bool(vertical), length(size)

    def SetLabels(self, labels: Any) -> RAxisDrawable:
        """``SetLabels``: the axis is of these labels, one per unit, not of numbers."""
        self.labels = [str(label) for label in labels]
        return self


class TObjectDrawable(Primitive):
    """``TObjectDrawable``: a ROOT 6 object - a histogram, a graph - drawn on a ROOT 7 pad;
    or, by kind alone, the style, colours or palette ROOT 6 draws with."""

    MEMBERS = frozen({"object": lambda: None, "option": lambda: "", "kind": lambda: 1,
                      "line": RAttrLine, "fill": RAttrFill, "marker": RAttrMarker,
                      "text": RAttrText})  # fmt: skip
    kNone, kObject, kColors, kStyle, kPalette = 0, 1, 2, 3, 4

    def __init__(self, obj: Any = None, option: Any = "") -> None:
        super().__init__()
        if isinstance(obj, int):
            self.kind = obj
        else:
            self.object, self.option = obj, str(option)

    @property
    def CSS_TYPE(self) -> str:  # type: ignore[override]
        """A drawn object is picked out by its class, in lower case: ``tgraph``, ``th1d``."""
        return type(self.object).__name__.lower() if self.object is not None else ""

    def GetObject(self) -> Any:
        return self.object

    def GetOption(self) -> str:
        return str(self.option)


class RFont(Primitive):
    """``RFont``: a font the pad's browser loads by name, from a file or a URL."""

    MEMBERS = frozen({"family": lambda: "", "src": lambda: ""})

    def __init__(self, family: Any = "", src: Any = "") -> None:
        super().__init__()
        self.family, self.src = str(family), str(src)
