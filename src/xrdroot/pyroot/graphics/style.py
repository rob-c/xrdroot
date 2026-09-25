"""``TStyle`` and ``gStyle``: the defaults everything drawn from now on takes.

A style is a long list of numbers - where the stats box goes and what it
says, how big an axis's labels are, what colour a pad is - each with a
``Set`` and a ``Get`` of the same name. Here the numbers are
:data:`FIELDS` and :data:`AXIS_FIELDS`, and the pairs are made from them,
so ``gStyle.SetStatX(0.9)`` and ``gStyle.GetStatX()`` exist for every one
ROOT has without a method written out for each.

``gStyle`` is always the current style: ``gROOT->SetStyle("Plain")`` makes
the style of that name current, and every name ROOT ships - ``Modern``, its
default since 5.30, ``Plain``, ``Classic``, ``Bold``, ``Video``, ``Pub``,
``ATLAS`` and ``BELLE2`` - is here, as ROOT's ``TStyle::BuildStyles`` makes
them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import colors
from .styledata import AXIS_FIELDS, FIELDS, STYLES, opt_stat

__all__ = ["TStyle", "gStyle", "set_style", "get_style"]

#: The styles there are, by name, made the first time each is asked for.
_MADE: dict[str, TStyle] = {}


def _axes(axis: str) -> list[str]:
    """Which axes ``"xyz"`` names; none of them means the pad's title."""
    named = [letter for letter in "XYZ" if letter in axis.upper()]
    return named or [""]


class TStyle:
    """A named set of defaults for drawing; ``gStyle`` is the current one."""

    def __init__(self, name: str = "Modern", title: str = "") -> None:
        self._name, self._title = name, title or f"{name} style"
        self.values: dict[str, Any] = dict(FIELDS)
        self.axes: dict[str, dict[str, Any]] = {
            axis: dict(AXIS_FIELDS) for axis in ("X", "Y", "Z", "")
        }
        self.axes[""]["TitleSize"] = 0.05  # the pad's title, bigger than an axis's
        self._palette: list[int] | None = None
        for field, value in STYLES.get(name, {}).items():
            self._apply(field, value)
        _MADE.setdefault(name, self)

    def _apply(self, field: str, value: Any) -> None:
        """One setting of a named style: ``"LabelSize:XYZ"`` for an axis's, else a field."""
        name, _colon, axis = field.partition(":")
        if _colon:
            for letter in _axes(axis) if axis else [""]:
                self.axes[letter][name] = value
        else:
            self.values[name] = value

    # -- the Set and Get of every field ----------------------------------------

    def __getattr__(self, name: str) -> Callable[..., Any]:
        verb, field = name[:3], name[3:]
        if verb in ("Set", "Get") and field in FIELDS:
            return self._setter(field) if verb == "Set" else lambda: self.values[field]
        if verb in ("Set", "Get") and field in AXIS_FIELDS:
            return self._axis_setter(field) if verb == "Set" else self._axis_getter(field)
        raise AttributeError(f"ROOT's TStyle has {name}; xrdroot.pyroot's does not yet")

    def _setter(self, field: str) -> Callable[..., None]:
        default = FIELDS[field]

        def setter(value: Any = default, *_rest: Any) -> None:
            self.values[field] = type(default)(value)

        return setter

    def _axis_setter(self, field: str) -> Callable[..., None]:
        def setter(value: Any, axis: str = "X") -> None:
            for letter in _axes(axis):
                self.axes[letter][field] = value

        return setter

    def _axis_getter(self, field: str) -> Callable[..., Any]:
        def getter(axis: str = "X") -> Any:
            return self.axes[_axes(axis)[0]][field]

        return getter

    # -- the ones that are more than a number ----------------------------------

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetOptStat(self, mode: Any = 1) -> None:
        """What the stats box says: digits, or ROOT's letters ``"nemruo"``."""
        self.values["OptStat"] = opt_stat(mode)

    def SetOptFit(self, mode: int = 1) -> None:
        self.values["OptFit"] = int(mode)

    def SetTitleFontSize(self, size: float = 0.0) -> None:
        self.axes[""]["TitleSize"] = float(size)

    def GetTitleFontSize(self) -> float:
        return float(self.axes[""]["TitleSize"])

    def SetPadTickX(self, tick: int) -> None:
        self.values["PadTickX"] = int(tick)

    def SetPalette(self, ncolors: Any = 57, colors_: Any = None, alpha: float = 1.0) -> None:
        """``kBird`` and ROOT's other palettes by number, or colours of one's own."""
        del alpha
        if colors_ is not None and int(ncolors) > 0:
            self._palette = [int(c) for c in list(colors_)[: int(ncolors)]]
        else:
            self._palette = colors.palette_indices(int(ncolors))

    def palette(self) -> list[int]:
        """The palette's colour indices, lowest first; ``kBird``'s when none was set."""
        return _bird() if self._palette is None else list(self._palette)

    def custom_palette(self) -> list[int] | None:
        """The palette set, or ``None`` for the default ``kBird``."""
        return None if self._palette is None else list(self._palette)

    def GetNumberOfColors(self) -> int:
        return len(self.palette())

    def GetColorPalette(self, i: int) -> int:
        shades = self.palette()
        return shades[int(i) % len(shades)]

    def cd(self) -> None:
        """Make this the current style, ``gStyle``."""
        _CURRENT[0] = self

    def Copy(self, other: TStyle) -> None:
        other.values = dict(self.values)
        other.axes = {axis: dict(held) for axis, held in self.axes.items()}
        other._palette = self._palette

    def Reset(self, _option: str = "") -> None:
        fresh = TStyle.__new__(TStyle)
        TStyle.__init__(fresh, self._name)
        fresh.Copy(self)

    def __repr__(self) -> str:
        return f"<TStyle {self._name!r}>"


def _bird() -> list[int]:
    """``kBird``, laid as colours of this session the first time it is asked for by index."""
    from ...plot.colors import PALETTES

    if 57 not in colors.LAID:
        colors.LAID[57] = [
            colors.TColor(-1, *colors._hex_rgb(shade)).GetNumber() for shade in PALETTES["bird"]
        ]
    return list(colors.LAID[57])


def get_style(name: str) -> TStyle | None:
    """``gROOT->GetStyle(name)``: a style ROOT ships or one made here, or ``None``."""
    if name not in _MADE and name in STYLES:
        TStyle(name)
    return _MADE.get(name)


def set_style(name: str = "Modern") -> None:
    """``gROOT->SetStyle(name)``: make the style of that name current."""
    style = get_style(name)
    if style is None:
        raise ValueError(
            f"there is no style {name!r}; ROOT ships {', '.join(sorted(STYLES))}, "
            f"and a TStyle made with a name of its own is found by it"
        )
    style.cd()


class _Current:
    """``gStyle``: whichever style is current, whenever it is asked."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_CURRENT[0], name)

    def __repr__(self) -> str:
        return repr(_CURRENT[0])


#: The current style, which ``gStyle`` stands for.
_CURRENT: list[TStyle] = [TStyle("Modern")]
gStyle: Any = _Current()
