"""``RooPlot``: a frame of one variable, and the data and curves drawn on it.

``x.frame()`` is an empty ``TH1D`` of the variable's range - its bins the
variable's, its title ``A RooPlot of "x"`` - that only draws its axes, and a
list of what was plotted on it, each with its draw option. Each addition
raises the frame's maximum to what it holds plus 5%, and the first data
added fixes how many events, in bins how wide, curves are scaled to.

Drawing a frame draws the histogram with ``FUNC`` - its axes and nothing
else - then each item, then the axes again, exactly as ``RooPlot::Draw``;
what "draw" means is the graphics layer's, installed with :func:`set_drawer`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...hist import Histogram
from ..cmdargs import commands
from ..messages import INFO, log
from ..printing import RooPrintable, address, g

__all__ = ["AXIS", "Axis", "RooPlot", "make_frame", "set_axis", "set_drawer"]

#: ``TH1::kNoStats`` and ``TH1::kNoTitle``' neighbour: a frame never has a stats box.
NO_STATS = 1 << 9
#: What drawing an object does: the graphics layer's hook, remembering by default.
DRAWN: list[tuple[Any, str]] = []
_DRAWER: list[Callable[[Any, str], Any]] = [lambda obj, option: DRAWN.append((obj, option))]


def set_drawer(fn: Callable[[Any, str], Any]) -> None:
    """Make ``fn(obj, option)`` what drawing a frame's parts does."""
    _DRAWER[0] = fn


class Axis:
    """An axis of the frame, over its members: the little of ``TAxis`` a macro sets on one."""

    def __init__(self, row: dict[str, Any], owner: Any = None) -> None:
        self._row = row

    def SetTitle(self, title: str = "") -> None:
        self._row["TNamed"]["fTitle"] = str(title)

    def GetTitle(self) -> str:
        return str(self._row["TNamed"]["fTitle"])

    def __getattr__(self, name: str) -> Any:
        member = "f" + name[3:]
        attributes = self.__dict__["_row"]["TAttAxis"]
        if name.startswith("Set") and member in attributes:
            return lambda value, *rest: attributes.__setitem__(member, type(attributes[member])(value))
        if name.startswith("Get") and member in attributes:
            return lambda: attributes[member]
        raise AttributeError(f"a RooPlot's axis has no {name} here")


#: What stands for an axis of a frame: the pyroot layer puts its ``TAxis`` here.
AXIS: list[Callable[[dict[str, Any], Any], Any]] = [Axis]


def set_axis(fn: Callable[[dict[str, Any], Any], Any]) -> None:
    AXIS[0] = fn


class RooPlot(RooPrintable):
    """A frame: its axes' histogram, and what was plotted on it."""

    def __init__(self, var: Any, low: float, high: float, bins: int) -> None:
        self.var = var
        self.hist = Histogram.book(f"frame_{var.GetName()}_{address(self)[2:]}", (int(bins), low, high))
        th1 = self.hist.members["TH1"]
        th1["fBits"] = NO_STATS
        th1["fXaxis"]["TNamed"]["fTitle"] = var.getTitle(True)
        self.SetTitle(f'A RooPlot of "{var.getTitle()}"')
        self.items: list[tuple[Any, str, bool]] = []
        self.norm_bin_width = (high - low) / bins
        self.norm_events = 0.0
        self.norm_obj: Any = None
        self.norm_vars: list[Any] | None = None
        self.pad_factor = 0.05
        self._name = self.hist.name

    # -- TObject and TH1 ----------------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def GetTitle(self) -> str:
        return str(self.hist.members["TH1"]["TNamed"]["fTitle"])

    def SetTitle(self, title: str) -> None:
        self.hist.members["TH1"]["TNamed"]["fTitle"] = str(title)

    def ClassName(self) -> str:
        return "RooPlot"

    def GetMaximum(self) -> float:
        found = float(self.hist.members["TH1"]["fMaximum"])
        return found if found != -1111.0 else float(max(self.hist.values().max(), 0.0))

    def GetMinimum(self) -> float:
        found = float(self.hist.members["TH1"]["fMinimum"])
        return found if found != -1111.0 else float(min(self.hist.values().min(), 0.0))

    def SetMaximum(self, value: float = -1111.0) -> None:
        self.hist.members["TH1"]["fMaximum"] = float(value)

    def SetMinimum(self, value: float = -1111.0) -> None:
        self.hist.members["TH1"]["fMinimum"] = float(value)

    def GetNbinsX(self) -> int:
        return int(self.hist.members["TH1"]["fXaxis"]["fNbins"])

    def GetXmin(self) -> float:
        return float(self.hist.members["TH1"]["fXaxis"]["fXmin"])

    def GetXmax(self) -> float:
        return float(self.hist.members["TH1"]["fXaxis"]["fXmax"])

    def axis(self, which: str) -> dict[str, Any]:
        """The members of the ``"x"`` or ``"y"`` axis, which the pyroot layer's ``TAxis`` edits."""
        return self.hist.members["TH1"][f"f{which.upper()}axis"]  # type: ignore[no-any-return]

    def GetXaxis(self) -> Any:
        return AXIS[0](self.axis("x"), self.hist)

    def GetYaxis(self) -> Any:
        return AXIS[0](self.axis("y"), self.hist)

    def SetXTitle(self, title: str) -> None:
        self.axis("x")["TNamed"]["fTitle"] = str(title)

    def SetYTitle(self, title: str) -> None:
        self.axis("y")["TNamed"]["fTitle"] = str(title)

    def getPlotVar(self) -> Any:
        return self.var

    def getFitRangeNEvt(self, low: Any = None, high: Any = None) -> float:
        if low is None or self.norm_obj is None:
            return self.norm_events
        return self.norm_events * self.norm_obj.events_between(low, high) / self.norm_obj.fit_range_events()

    def getFitRangeBinW(self) -> float:
        return self.norm_bin_width

    def getPadFactor(self) -> float:
        return self.pad_factor

    def setPadFactor(self, factor: float) -> None:
        self.pad_factor = float(factor)

    # -- adding -------------------------------------------------------------------

    def update_y_axis(self, ymin: float, ymax: float, label: str) -> None:
        """``updateYAxis``: room above for what was added, and its label if the axis has none."""
        if self.GetMinimum() == 0 and ymin > 0:
            ymin = 0.0
        pad = self.pad_factor * (ymax - ymin)
        ymax += pad
        if ymin < 0:
            ymin -= pad
        if self.GetMaximum() < ymax:
            self.SetMaximum(ymax)
            self.hist.members["TH1"]["fArray"] = _with_first(self.hist, ymax)
        if self.GetMinimum() > ymin:
            self.SetMinimum(ymin)
        if not self.axis("y")["TNamed"]["fTitle"]:
            self.axis("y")["TNamed"]["fTitle"] = label

    def add_plotable(self, item: Any, option: str, invisible: bool = False,
                     refresh: bool = True) -> None:  # fmt: skip
        """``addPlotable``: a data histogram or a curve, and what it does to the frame."""
        ys, lows, highs = item.y, *_bars(item)
        self.update_y_axis(float((ys - lows).min()), float((ys + highs).max()), item.y_label)
        if hasattr(item, "fit_range_events"):
            self._normalise_to(item, refresh)
        self.items.append((item, option, invisible))

    def _normalise_to(self, item: Any, refresh: bool) -> None:
        """``updateFitRangeNorm``: the first data - or new data, if asked - set the curves' scale."""
        events = item.fit_range_events()
        if self.norm_events != 0:
            if not refresh:
                return
            factor = self.norm_bin_width / item.fit_range_bin_width()
            if abs(events / factor - self.norm_events) > 1e-6:
                log(self, INFO, "Plotting", f"RooPlot::updateFitRangeNorm: New event count of "
                    f"{g(events / factor)} will supersede previous event count of "
                    f"{g(self.norm_events)} for normalization of PDF projections")  # fmt: skip
            self.norm_events = events / factor
        else:
            self.norm_events = events
            if item.fit_range_bin_width():
                self.norm_bin_width = item.fit_range_bin_width()
        self.norm_obj = item

    def addObject(self, obj: Any, option: str = "", invisible: bool = False) -> None:
        self.items.append((obj, option, bool(invisible)))

    addTH1 = addObject

    def update_norm_vars(self, variables: Any) -> None:
        if self.norm_vars is None:
            self.norm_vars = list(variables)

    def findObject(self, name: Any = None, kind: Any = None) -> Any:
        """The item called ``name`` - or the last item, for none - optionally of a kind."""
        found = [obj for obj, _, _ in self.items if _matches(obj, name, kind)]
        return found[-1] if found else None

    def getObject(self, index: int) -> Any:
        return self.items[index][0]

    def numItems(self) -> int:
        return len(self.items)

    def nameOf(self, index: int) -> str:
        return str(self.items[index][0].GetName())

    def getDrawOptions(self, name: str) -> str:
        return next((option for obj, option, _ in self.items if obj.GetName() == name), "")

    def setDrawOptions(self, name: str, options: str) -> bool:
        for index, (obj, _, invisible) in enumerate(self.items):
            if obj.GetName() == name:
                self.items[index] = (obj, str(options), invisible)
                return True
        return False

    def setInvisible(self, name: str, flag: bool = True) -> None:
        for index, (obj, option, _) in enumerate(self.items):
            if obj.GetName() == name:
                self.items[index] = (obj, option, bool(flag))

    def remove(self, name: Any = None, deleteToo: bool = True) -> None:
        target = self.findObject(name)
        self.items = [item for item in self.items if item[0] is not target]

    # -- drawing ------------------------------------------------------------------

    def Draw(self, option: str = "") -> None:
        """``RooPlot::Draw``: the axes, every visible item, the axes again."""
        same = "same" in str(option).lower()
        _DRAWER[0](self.hist, "FUNCSAME" if same else "FUNC")
        for obj, opt, invisible in self.items:
            if not invisible:
                _DRAWER[0](obj, opt or ("LP" if hasattr(obj, "x") else ""))
        _DRAWER[0](self.hist, "AXISSAME")

    def printName(self) -> str:
        return self._name

    def printTitle(self) -> str:
        return self.GetTitle()

    def printClassName(self) -> str:
        return "RooPlot"

    def printValue(self) -> str:
        return f"(" + ",".join(obj.GetName() for obj, _, _ in self.items) + ")"


def _with_first(hist: Histogram, value: float) -> Any:
    values = hist.members["TH1"].get("fArray")
    if values is None:
        return values
    values = values.copy()
    values[1] = value
    return values


def _bars(item: Any) -> tuple[Any, Any]:
    errors = getattr(item, "errors", None)
    if errors is None:
        return 0.0 * item.y, 0.0 * item.y
    return errors()  # type: ignore[no-any-return]


def _matches(obj: Any, name: Any, kind: Any) -> bool:
    if name and obj.GetName() != str(name):
        return False
    return kind is None or obj.ClassName() == getattr(kind, "__name__", str(kind))


def make_frame(var: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> RooPlot:
    """``var.frame(...)``: ``frame(nbins)``, ``frame(low, high, nbins)``, or options."""
    numbers = [a for a in args if isinstance(a, (int, float))]
    options = commands([a for a in args if not isinstance(a, (int, float))], kwargs)
    low, high = var.getMin(), var.getMax()
    bins = var.getBins()
    if len(numbers) == 1:
        bins = int(numbers[0])
    elif len(numbers) >= 2:
        low, high = float(numbers[0]), float(numbers[1])
        bins = int(numbers[2]) if len(numbers) > 2 else bins
    if "Range" in options:
        rng = options.args("Range")
        low, high = (var.getMin(rng[0]), var.getMax(rng[0])) if isinstance(rng[0], str) else rng[:2]
    bins = int(options.get("Bins", 0, bins))
    frame = RooPlot(var, float(low), float(high), bins)
    if "Title" in options:
        frame.SetTitle(str(options.get("Title")))
    if "Name" in options:
        frame.SetName(str(options.get("Name")))
    return frame
