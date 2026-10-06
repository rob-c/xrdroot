"""An axis's attributes: its range and scale, its title, ticks, labels and how its line ends.

``RAttrAxis`` is what a frame's ``x`` and ``y`` and an ``RAxisDrawable``'s
``axis`` are: ``min`` and ``max``, a ``log`` base or a ``symlog`` constant,
and groups for the parts drawn. Its title takes the text it shows by
assignment - ``axis.title = "x"`` - and is placed by ``SetCenter()``.
"""

from __future__ import annotations

from typing import Any

from .attrs import Attrs, frozen
from .kinds import RAttrLine, RAttrText, _black
from .lengths import RPadLength

__all__ = ["RAttrAxis", "RAttrAxisTitle", "RAttrAxisTicks", "RAttrAxisLabels", "RAttrLineEnding"]


class RAttrAxisTitle(RAttrText):
    """An axis's title: its text, where along the axis it is, and how far off."""

    MEMBERS = frozen({"value": lambda: "", "position": lambda: "right", "offset": RPadLength})

    def _assign(self, value: Any) -> None:
        self.value = str(value)

    def SetCenter(self) -> RAttrAxisTitle:
        self.position = "center"
        return self

    def SetLeft(self) -> RAttrAxisTitle:
        self.position = "left"
        return self

    def SetRight(self) -> RAttrAxisTitle:
        self.position = "right"
        return self


class RAttrAxisLabels(RAttrText):
    """An axis's labels: how far off, centred between ticks or not, shown or not."""

    MEMBERS = frozen({"offset": RPadLength, "center": lambda: False, "hide": lambda: False})


class RAttrAxisTicks(Attrs):
    """An axis's ticks: which side they are on, how long, how thick, what colour."""

    MEMBERS = frozen({"side": lambda: "normal", "size": lambda: RPadLength(0.02),
                      "width": lambda: 1.0, "color": _black})  # fmt: skip

    def SetNormal(self) -> RAttrAxisTicks:
        self.side = "normal"
        return self

    def SetInvert(self) -> RAttrAxisTicks:
        self.side = "invert"
        return self

    def SetBoth(self) -> RAttrAxisTicks:
        self.side = "both"
        return self


class RAttrLineEnding(Attrs):
    """How an axis's line ends: plainly, or in an arrow, a circle, a square or a diamond."""

    MEMBERS = frozen({"style": lambda: "", "size": lambda: RPadLength(0.02)})

    def _ends(self, style: str) -> RAttrLineEnding:
        self.style = style
        return self

    def SetNone(self) -> RAttrLineEnding:
        return self._ends("")

    def SetArrow(self) -> RAttrLineEnding:
        return self._ends("arrow")

    def SetCircle(self) -> RAttrLineEnding:
        return self._ends("circle")

    def SetSquare(self) -> RAttrLineEnding:
        return self._ends("square")

    def SetDiamond(self) -> RAttrLineEnding:
        return self._ends("diamond")


class RAttrAxis(Attrs):
    """``RAttrAxis``: an axis's range, its scale, and the groups of its drawn parts."""

    MEMBERS = frozen({
        "min": lambda: 0.0, "max": lambda: 0.0, "zoomMin": lambda: 0.0, "zoomMax": lambda: 0.0,
        "log": lambda: 0.0, "symlog": lambda: 0.0, "reverse": lambda: False,
        "time": lambda: False, "timeFormat": lambda: "", "timeOffset": lambda: 0.0,
        "line": RAttrLine, "ending": RAttrLineEnding, "ticks": RAttrAxisTicks,
        "labels": RAttrAxisLabels, "title": RAttrAxisTitle,
    })  # fmt: skip

    def SetMinMax(self, low: Any, high: Any) -> RAttrAxis:
        self.min, self.max = float(low), float(high)
        return self

    def SetTimeDisplay(self, fmt: Any = "", offset: Any = -1.0) -> RAttrAxis:
        """``SetTimeDisplay(format)``: the labels are times, written as ``strftime`` writes."""
        self.time, self.timeFormat, self.timeOffset = True, str(fmt), float(offset)
        return self
