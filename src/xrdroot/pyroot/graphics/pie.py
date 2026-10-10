"""``TPie`` and ``TPieSlice``: a pie chart, each slice a value's share of the whole.

A pie is a circle at ``(fX, fY)`` of radius ``fRadius`` in the pad's
coordinates, cut from ``fAngularOffset`` degrees anticlockwise into a slice
per value, each with its fill, its line, a title the label is made of by
``fLabelFormat`` (``%txt`` the title, ``%val`` the value, ``%frac`` its
fraction and ``%perc`` its percentage), and an offset out from the centre.
How a pad paints one is :mod:`xrdroot.canvas.pie`'s business.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .drawn import Drawn

__all__ = ["TPie", "TPieSlice"]


class TPieSlice(Drawn):
    """``TPieSlice``: one slice's value, how far out it sits, and its fill and line."""

    classname = "TPieSlice"
    groups: ClassVar[tuple[str, ...]] = ("fill", "line")
    fields: ClassVar[dict[str, type]] = {"Value": float, "RadiusOffset": float}

    def __init__(
        self, name: str = "", title: str = "", pie: Any = None, value: float = 0.0
    ) -> None:
        super().__init__(fName=str(name), fTitle=str(title), fValue=float(value),
                         fRadiusOffset=0.0)  # fmt: skip
        self._pie = pie


class TPie(Drawn):
    """``TPie(name, title, n[, values[, colors[, labels]]])``, or ``TPie(histogram)``."""

    classname = "TPie"
    groups: ClassVar[tuple[str, ...]] = ("text",)
    fields: ClassVar[dict[str, type]] = {
        "X": float, "Y": float, "Radius": float, "AngularOffset": float, "LabelsOffset": float,
        "LabelFormat": str, "ValueFormat": str, "FractionFormat": str, "PercentFormat": str,
        "Height": float, "Angle3D": float,
    }  # fmt: skip

    def __init__(self, name: Any = "", title: str = "", npoints: int = 0, values: Any = None,
                 colors: Any = None, labels: Any = None) -> None:  # fmt: skip
        super().__init__(
            fX=0.5, fY=0.5, fRadius=0.4, fAngularOffset=0.0, fLabelsOffset=0.1,
            fLabelFormat="%txt", fValueFormat="%4.2f", fFractionFormat="%3.2f",
            fPercentFormat="%3.1f", fHeight=0.08, fAngle3D=30.0, fPieSlices=[],
        )  # fmt: skip
        if hasattr(name, "GetNbinsX"):  # a histogram: a slice per bin, titled by its label
            self._from_histogram(name)
            return
        self.members.update(fName=str(name), fTitle=str(title))
        self._make(int(npoints), values, colors, labels)

    def _from_histogram(self, h: Any) -> None:
        bins = range(1, h.GetNbinsX() + 1)
        self.members.update(fName=h.GetName(), fTitle=h.GetTitle())
        self._make(len(bins), [h.GetBinContent(b) for b in bins], None,
                   [h.GetXaxis().GetBinLabel(b) for b in bins])  # fmt: skip

    def _make(self, count: int, values: Any, colors: Any, labels: Any) -> None:
        """``Init``: a slice per value, named after the pie, coloured and labelled as given."""
        slices = []
        for at in range(count):
            name = f"{self.GetName()}_slice_{at}"
            made = TPieSlice(name, name, self, values[at] if values is not None else 0.0)
            if colors is not None:
                made.SetFillColor(int(colors[at]))
            if labels is not None and labels[at]:
                made.SetTitle(str(labels[at]))
            slices.append(made)
        self.members["fPieSlices"] = slices

    # -- the slices ----------------------------------------------------------------------------

    def GetEntries(self) -> int:
        return len(self.members["fPieSlices"])

    def GetSlice(self, i: int) -> TPieSlice:
        return self.members["fPieSlices"][int(i)]  # type: ignore[no-any-return]

    def SetEntryVal(self, i: int, value: float) -> None:
        self.GetSlice(i).SetValue(value)

    def GetEntryVal(self, i: int) -> float:
        return float(self.GetSlice(i).GetValue())

    def SetEntryLabel(self, i: int, text: str = "Slice") -> None:
        self.GetSlice(i).SetTitle(text)

    def GetEntryLabel(self, i: int) -> str:
        return self.GetSlice(i).GetTitle()

    def SetEntryRadiusOffset(self, i: int, offset: float) -> None:
        self.GetSlice(i).SetRadiusOffset(offset)

    def GetEntryRadiusOffset(self, i: int) -> float:
        return float(self.GetSlice(i).GetRadiusOffset())

    def _entry(self, name: str) -> Any:
        """``SetEntryFillColor`` and its kin: the slice's own setter, by its attribute's name."""
        return lambda i, value: getattr(self.GetSlice(i), f"Set{name}")(value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("SetEntry") and name not in ("SetEntryVal", "SetEntryLabel"):
            return self._entry(name[8:])
        if name.startswith("GetEntry") and name not in ("GetEntryVal", "GetEntryLabel"):
            return lambda i: getattr(self.GetSlice(i), f"Get{name[8:]}")()
        return super().__getattr__(name)

    def SetFillColors(self, colors: Any) -> None:
        for at, piece in enumerate(self.members["fPieSlices"]):
            piece.SetFillColor(int(colors[at]))

    def SetLabels(self, labels: Any) -> None:
        for at, piece in enumerate(self.members["fPieSlices"]):
            piece.SetTitle(str(labels[at]))

    def SetCircle(self, x: float = 0.5, y: float = 0.5, radius: float = 0.4) -> None:
        self.members.update(fX=float(x), fY=float(y), fRadius=float(radius))

    def Draw(self, option: str = "l") -> None:
        """``Draw``: onto the current pad - ``R`` labels along the radius, ``T`` along the
        tangent, ``SC`` in the slice's colour, ``NOL`` no labels; ``3D`` is painted flat."""
        super().Draw(option)

    def MakeLegend(self, x1: float = 0.65, y1: float = 0.65, x2: float = 0.95, y2: float = 0.95,
                   header: str = "") -> Any:  # fmt: skip
        """``MakeLegend``: a legend with an entry per slice, filled as the slice is."""
        from .legend import TLegend

        made = TLegend(x1, y1, x2, y2, header)
        for piece in self.members["fPieSlices"]:
            made.AddEntry(piece, piece.GetTitle(), "f")
        self._legend = made
        return made

    def GetLegend(self) -> Any:
        return self.__dict__.get("_legend")
