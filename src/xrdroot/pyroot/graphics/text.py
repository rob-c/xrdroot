"""Text, in ROOT's ``#`` mathematics or as it is, and axes drawn anywhere.

``TLatex`` and ``TText`` are a string at ``(x, y)`` - in the pad's axes'
units, or its fractions once ``SetNDC`` - and ``DrawLatex``/``DrawText``
draw a copy at a new place with a new string. ``TMathText`` takes real
LaTeX, which is drawn by matplotlib's mathtext as it is. ``TGaxis`` is a
graduated line of its own scale.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .drawn import NDC_BIT, Drawn

__all__ = ["TGaxis", "TLatex", "TMathText", "TText"]


class TText(Drawn):
    """A string at ``(x, y)``, drawn as it is."""

    classname = "TText"
    groups: ClassVar[tuple[str, ...]] = ("text",)
    fields: ClassVar[dict[str, type]] = {"X": float, "Y": float}

    def __init__(self, x: float = 0.0, y: float = 0.0, text: str = "") -> None:
        super().__init__(fX=float(x), fY=float(y), fTitle=str(text))

    def SetText(self, x: float, y: float, text: str) -> None:
        self.members.update(fX=float(x), fY=float(y), fTitle=str(text))

    def _draw_at(self, x: float, y: float, text: str, ndc: bool) -> Any:
        made = self.Clone()
        made.SetText(x, y, text)
        made.SetBit(NDC_BIT, ndc)
        made.Draw()
        return made

    def _draw_text(self, x: float, y: float, text: str, ndc: bool) -> Any:
        """``TText::DrawText``: a new ``TText`` - never a formula - in this one's text
        attributes."""
        made = TText(x, y, text)
        own = {name: value for name, value in self.members.items() if name.startswith("fText")}
        made.members.update(own)
        made.SetBit(NDC_BIT, ndc)
        made.Draw()
        return made

    def DrawText(self, x: float, y: float, text: str) -> Any:
        return self._draw_text(x, y, text, False)

    def DrawTextNDC(self, x: float, y: float, text: str) -> Any:
        return self._draw_text(x, y, text, True)


class TLatex(TText):
    """A string at ``(x, y)``, with ROOT's ``#`` mathematics drawn."""

    classname = "TLatex"
    groups: ClassVar[tuple[str, ...]] = ("text", "line")
    defaults: ClassVar[dict[str, Any]] = {"fLineWidth": 2}

    def DrawLatex(self, x: float, y: float, text: str) -> Any:
        return self._draw_at(x, y, text, False)

    def DrawLatexNDC(self, x: float, y: float, text: str) -> Any:
        return self._draw_at(x, y, text, True)


class TMathText(TText):
    """A string at ``(x, y)`` in LaTeX's own mathematics, ``\\alpha`` and all."""

    classname = "TMathText"

    def DrawMathText(self, x: float, y: float, text: str) -> Any:
        return self._draw_at(x, y, text, False)


class TGaxis(Drawn):
    """An axis from ``(xmin, ymin)`` to ``(xmax, ymax)``, graduated ``wmin`` to ``wmax``.

    A function's name in place of ``wmin`` and ``wmax`` graduates it over
    that function's range, if it is one this session has made.
    """

    classname = "TGaxis"
    groups: ClassVar[tuple[str, ...]] = ("line", "text")
    fields: ClassVar[dict[str, type]] = {
        "Wmin": float, "Wmax": float, "LabelSize": float, "LabelFont": int,
        "LabelColor": int, "LabelOffset": float, "TitleSize": float, "TitleOffset": float,
        "TickSize": float, "Ndiv": int, "Chopt": str, "MaxDigits": int, "GridLength": float,
    }  # fmt: skip

    def __init__(
        self, xmin: float = 0.0, ymin: float = 0.0, xmax: float = 1.0, ymax: float = 0.0,
        *scale: Any,
    ) -> None:  # fmt: skip
        by_function = bool(scale) and isinstance(scale[0], str)
        rest = list(scale[1:] if by_function else scale[2:])
        ndiv, chopt, gridlength = (rest + [510, "", 0.0][len(rest) :])[:3]
        super().__init__(
            fX1=float(xmin), fY1=float(ymin), fX2=float(xmax), fY2=float(ymax),
            fNdiv=int(ndiv), fChopt=str(chopt), fGridLength=float(gridlength),
            fLabelSize=0.04, fLabelFont=62, fLabelColor=1, fLabelOffset=0.005,
            fTitleSize=0.04, fTitleOffset=1.0, fTickSize=0.03, fMaxDigits=5,
            fFunctionName="", fWmin=0.0, fWmax=1.0,
        )  # fmt: skip
        if by_function:
            self._graduate_by(scale[0])
        elif scale:
            self.members.update(fWmin=float(scale[0]), fWmax=float(scale[1]))

    def _graduate_by(self, name: str) -> None:
        """The scale of the function named ``name``: its range."""
        from .pads import find_anywhere

        function = find_anywhere(name)
        low, high = getattr(getattr(function, "_xrd", function), "range", (0.0, 1.0))[:2]
        self.members.update(fFunctionName=name, fWmin=float(low), fWmax=float(high))

    def SetNdivisions(self, ndiv: int, optim: bool = True) -> None:
        del optim
        self.members["fNdiv"] = int(ndiv)

    def SetOption(self, chopt: str = "") -> None:
        self.members["fChopt"] = str(chopt)

    def CenterTitle(self, center: bool = True) -> None:
        chopt = str(self.members["fChopt"]).replace("C", "")
        self.members["fChopt"] = chopt + ("C" if center else "")

    def DrawAxis(
        self, xmin: float, ymin: float, xmax: float, ymax: float,
        wmin: float, wmax: float, ndiv: int = 510, chopt: str = "", gridlength: float = 0.0,
    ) -> Any:  # fmt: skip
        made = self.Clone()
        made.members.update(
            fX1=xmin, fY1=ymin, fX2=xmax, fY2=ymax, fWmin=wmin, fWmax=wmax,
            fNdiv=ndiv, fChopt=chopt, fGridLength=gridlength,
        )  # fmt: skip
        made.Draw()
        return made
