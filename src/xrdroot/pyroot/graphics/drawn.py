"""What every drawing class here is: ROOT's members by name, and their Set and Get.

A drawing class is its members - ``fX1``, ``fLineColor``, ``fTextSize`` -
kept by name in :attr:`Drawn.members`, which is what a saved one is read
as, so a pad hands its primitives to :mod:`xrdroot.canvas` exactly as
though ROOT had written them. ``TAttLine``, ``TAttFill``, ``TAttMarker`` and
``TAttText`` are :data:`GROUPS`: a class says which it has, and
``SetLineColor``/``GetLineColor`` and the rest are made from the table,
their defaults taken from ``gStyle`` when the object is made, as ROOT's
constructors take them.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, ClassVar

from ..core import draw_hook
from .style import gStyle

__all__ = ["GROUPS", "NDC_BIT", "Drawn"]

#: ``SetNDC``'s bit of ``fBits``.
NDC_BIT = 1 << 14
#: The bits every object on the heap has, as every drawn one is.
HEAP = 0x03000000

#: Each attribute group: its methods' names, the member each sets, the
#: ``gStyle`` field its default comes from, and the type it is kept as.
GROUPS: dict[str, dict[str, tuple[str, str, type]]] = {
    "line": {
        "LineColor": ("fLineColor", "LineColor", int),
        "LineStyle": ("fLineStyle", "LineStyle", int),
        "LineWidth": ("fLineWidth", "LineWidth", int),
    },
    "fill": {
        "FillColor": ("fFillColor", "FillColor", int),
        "FillStyle": ("fFillStyle", "FillStyle", int),
    },
    "marker": {
        "MarkerColor": ("fMarkerColor", "MarkerColor", int),
        "MarkerStyle": ("fMarkerStyle", "MarkerStyle", int),
        "MarkerSize": ("fMarkerSize", "MarkerSize", float),
    },
    "text": {
        "TextColor": ("fTextColor", "TextColor", int),
        "TextFont": ("fTextFont", "TextFont", int),
        "TextSize": ("fTextSize", "TextSize", float),
        "TextAlign": ("fTextAlign", "TextAlign", int),
        "TextAngle": ("fTextAngle", "TextAngle", float),
    },
}


def _table(groups: tuple[str, ...]) -> dict[str, tuple[str, str, type]]:
    found: dict[str, tuple[str, str, type]] = {}
    for group in groups:
        found.update(GROUPS[group])
    return found


class Drawn:
    """One of ROOT's drawing classes, kept as the members ROOT would write."""

    #: The class this stands for.
    classname = "TObject"
    #: The attribute groups it has.
    groups: ClassVar[tuple[str, ...]] = ()
    #: Members of its own, beyond those groups, and what they start as.
    defaults: ClassVar[dict[str, Any]] = {}
    #: Members with a Set and a Get of their own: ``"X1"`` is ``SetX1`` of ``fX1``.
    fields: ClassVar[dict[str, type]] = {}

    def __init__(self, **members: Any) -> None:
        start: dict[str, Any] = {"fName": "", "fTitle": "", "fBits": HEAP, "fUniqueID": 0}
        for member, field, kind in _table(self.groups).values():
            start[member] = kind(getattr(gStyle, f"Get{field}")())
        start.update(copy.deepcopy(self.defaults))
        start.update(members)
        #: Every member, by the name ROOT gives it.
        self.members = start

    # -- the attribute groups' Set and Get ---------------------------------------

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith(("Set", "Get")) and "members" in self.__dict__:
            found = _table(type(self).groups).get(name[3:])
            if found is None and name[3:] in type(self).fields:
                found = (f"f{name[3:]}", "", type(self).fields[name[3:]])
            if found is not None:
                return self._accessor(name[:3], found)
        raise AttributeError(f"ROOT's {self.classname} has {name}; xrdroot.pyroot's does not yet")

    def _accessor(self, verb: str, found: tuple[str, str, type]) -> Callable[..., Any]:
        member, _field, kind = found
        if verb == "Get":
            return lambda: self.members[member]

        def setter(value: Any) -> None:
            self.members[member] = kind(value)

        return setter

    def SetFillColorAlpha(self, color: int, alpha: float) -> None:
        """A fill in colour ``color`` at opacity ``alpha``, as a colour of its own."""
        from .colors import TColor

        self.members["fFillColor"] = TColor.GetColorTransparent(color, alpha)

    def SetLineColorAlpha(self, color: int, alpha: float) -> None:
        from .colors import TColor

        self.members["fLineColor"] = TColor.GetColorTransparent(color, alpha)

    # -- TObject --------------------------------------------------------------

    def GetName(self) -> str:
        return str(self.members.get("fName", ""))

    def SetName(self, name: str) -> None:
        self.members["fName"] = str(name)

    def GetTitle(self) -> str:
        return str(self.members.get("fTitle", ""))

    def SetTitle(self, title: str = "") -> None:
        self.members["fTitle"] = str(title)

    def ClassName(self) -> str:
        return self.classname

    def InheritsFrom(self, classname: Any) -> bool:
        wanted = classname if isinstance(classname, str) else getattr(classname, "classname", "")
        return any(getattr(kind, "classname", "") == wanted for kind in type(self).__mro__)

    def SetBit(self, bit: int, on: bool = True) -> None:
        bits = int(self.members.get("fBits", 0))
        self.members["fBits"] = bits | bit if on else bits & ~bit

    def ResetBit(self, bit: int) -> None:
        self.SetBit(bit, False)

    def TestBit(self, bit: int) -> bool:
        return bool(int(self.members.get("fBits", 0)) & bit)

    def SetUniqueID(self, uid: int) -> None:
        self.members["fUniqueID"] = int(uid)

    def GetUniqueID(self) -> int:
        return int(self.members.get("fUniqueID", 0))

    def SetNDC(self, isNDC: bool = True) -> None:
        """Place this in fractions of the pad rather than the units of its axes."""
        self.SetBit(NDC_BIT, bool(isNDC))

    def GetNDC(self) -> bool:
        return self.TestBit(NDC_BIT)

    def Draw(self, option: str = "") -> None:
        """Add this to the current pad, drawn with ``option``."""
        draw_hook(self, option)

    def Clone(self, newname: str = "") -> Any:
        made = copy.copy(self)
        made.members = copy.deepcopy(self.members)
        if newname:
            made.members["fName"] = newname
        return made

    def DrawClone(self, option: str = "") -> Any:
        made = self.Clone()
        made.Draw(option)
        return made

    def Paint(self, option: str = "") -> None:
        """Nothing: a pad paints what it holds when it is saved."""

    def __repr__(self) -> str:
        name = self.GetName()
        return f"<{self.classname} {name!r}>" if name else f"<{self.classname}>"
