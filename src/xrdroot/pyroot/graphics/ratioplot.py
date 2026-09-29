"""``TRatioPlot``: two histograms, or one and its fit, over the ratio, difference or residuals.

    >>> rp = ROOT.TRatioPlot(h1, h2)          # h1 / h2, as TGraphAsymmErrors::Divide "pois"
    ... # doctest: +SKIP
    >>> rp.Draw(); rp.GetLowerRefGraph().SetMinimum(0.5)                 # doctest: +SKIP

ROOT's ratio plot is a helper that lays out pads: an upper pad for the
histograms, a lower one - below ``SetSplitFraction``, 0.3 of the height -
for what :mod:`.ratiocalc` works out, and a transparent pad over both
holding the axes :mod:`.ratioaxes` draws, so the two plots share one x axis.
It is built as ROOT builds it, on the live pads: ``GetUpperPad()`` is a
``TPad`` to draw a legend in, ``GetLowerRefGraph()`` the graph whose range
the lower frame takes, and the canvas saved draws it as ROOT's does. The
ratio plot itself is one of the parent pad's primitives, and when the
canvas is drawn it places its axes afresh, as ``TRatioPlot::Paint`` does.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ...hist import Histogram
from ..core.colors import named
from ..core.graphs import TGraph
from ..core.objects import TObject
from ..core.stacks import THStack
from . import ratioaxes, ratiocalc
from .pads import TPad, current
from .ratioaxes import f32
from .shapes import TLine
from .style import gStyle

__all__ = ["TRatioPlot"]

#: ``kYellow`` and ``kGreen``: the one- and two-sigma bands' colours.
YELLOW, GREEN = 400, 416
#: What a ratio plot's pads start with: the upper pad's top and bottom margins, the lower
#: pad's, the gap round the pads (``fInsetWidth``) and where the pads meet.
UP_TOP, UP_BOTTOM, LOW_TOP, LOW_BOTTOM = 0.1, 0.05, 0.05, 0.3
INSET, SPLIT = 0.0025, 0.3
#: The one- and two-sigma confidence levels of the fit's bands.
CL1, CL2 = 0.6827, 0.9545
#: ``TObject::kCannotPick``: the top pad lets clicks through to the pads under it.
CANNOT_PICK = 1 << 6


def _is_stack(obj: Any) -> bool:
    return isinstance(obj, THStack)


def _colour(value: Any) -> int:
    return named(value) if isinstance(value, str) else int(value)


def _is_histogram(obj: Any) -> bool:
    return isinstance(getattr(obj, "_xrd", None), Histogram) and hasattr(obj, "GetXaxis")


def _summed(stack: Any) -> Any:
    """The stack's histograms added into a copy of the first, as ROOT's constructors add them."""
    hists = stack.GetHists()
    made = hists.At(0).Clone()
    made.SetDirectory(None)
    made.Reset()
    for i in range(hists.GetSize()):
        made.Add(hists.At(i))
    return made


def _one(obj: Any) -> Any:
    """A histogram as itself, a stack as the sum of its histograms."""
    return _summed(obj) if _is_stack(obj) else obj


def _refused(first: Any, second: Any) -> str:
    """ROOT's warning for what two constructor arguments lack, or nothing if they will do."""
    stacked = [obj for obj in (first, second) if _is_stack(obj)]
    if first is None or second is None:
        return "Need a histogram and a stack" if stacked else "Need two histograms."
    if any(stack.GetHists().GetSize() == 0 for stack in stacked):
        return "Stack does not have histograms"
    return ""


def _clone_axis(axis: Any) -> Any:
    """A ``TAxis`` of its own, as ROOT's ``Clone`` of one stands apart from its histogram."""
    return type(axis)._of(copy.deepcopy(axis._row), None)


def _mode_of(option: str, rp: Any) -> tuple[int, str]:
    """``Init``'s reading of the option: the calculation, and what is left to pass on."""
    if "divsym" in option:
        return ratiocalc.DIVIDE_HIST, option.replace("divsym", "")
    if "diffsig" in option:
        option = _error_mode_of(option.replace("diffsig", ""), rp)
        return ratiocalc.SIGNIFICANCE, option
    if "diff" in option:
        return ratiocalc.DIFFERENCE, option.replace("diff", "")
    return ratiocalc.DIVIDE, option


def _error_mode_of(option: str, rp: Any) -> str:
    """``errasym`` and ``errfunc``: which error a significance or residual is over."""
    if "errasym" in option:
        rp._error_mode, option = ratiocalc.ASYMMETRIC, option.replace("errasym", "")
    if "errfunc" in option:
        rp._error_mode, option = ratiocalc.FUNCTION, option.replace("errfunc", "")
    return option


class TRatioPlot(TObject):
    """ROOT's ``TRatioPlot``: made from two histograms, a stack and a histogram, or a fitted one."""

    CLASS_TITLE = "A ratio of histograms"

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self._defaults()
        if not args:
            return
        second = args[1] if len(args) > 1 else ""
        if second is None or isinstance(second, str):
            self._fitted(args[0], str(second or ""), args[2] if len(args) > 2 else None)
        else:
            self._two(args[0], second, str(args[2]) if len(args) > 2 else "pois")

    def _defaults(self) -> None:
        """What ROOT's default constructor and the header's initialisers leave in each member."""
        self._parent_pad: Any = None
        self._upper_pad: Any = None
        self._lower_pad: Any = None
        self._top_pad: Any = None
        self._h1: Any = None
        self._h2: Any = None
        self._proxy: Any = None
        self._proxy_stack = False
        self._mode = 0
        self._error_mode = ratiocalc.SYMMETRIC
        self._option = ""
        self._h1_draw_opt, self._h2_draw_opt, self._graph_draw_opt = "", "", ""
        self._fit_draw_opt = ""
        self._split_fraction = f32(SPLIT)
        self._ratio_graph: Any = None
        self._ci1: Any = None
        self._ci2: Any = None
        self._ci1_color, self._ci2_color = YELLOW, GREEN
        self._show_confint = True
        self._show_gridlines = True
        self._cl1, self._cl2 = CL1, CL2
        self._c1, self._c2 = 1.0, 1.0
        self._fit_result: Any = None
        self._shared_x: Any = None
        self._up_y: Any = None
        self._low_y: Any = None
        self._axes: dict[str, Any] = {}
        self._mirrors: dict[str, Any] = {}
        self._gridlines: list[Any] = []
        self._gridline_positions: list[float] = []
        self._hide_label_mode = ratioaxes.HIDE_LOW
        self._up_top, self._up_bottom = f32(UP_TOP), f32(UP_BOTTOM)
        self._low_top, self._low_bottom = f32(LOW_TOP), f32(LOW_BOTTOM)
        self._left = f32(gStyle.GetPadLeftMargin())
        self._right = f32(gStyle.GetPadRightMargin())
        self._inset_width = INSET

    # -- making one ------------------------------------------------------------------------

    def _two(self, first: Any, second: Any, option: str) -> None:
        """``TRatioPlot(h1, h2)``, ``(stack, h2)`` or ``(h1, stack)``: a ratio or difference."""
        refused = _refused(first, second)
        if refused:
            self.Warning("TRatioPlot", refused)
            return
        self._proxy = first
        self._proxy_stack = _is_stack(first)
        self._init(_one(first), _one(second), option)

    def _init(self, h1: Any, h2: Any, option: str) -> None:
        """``Init``: the pads, the mode the option asks for, and the lower plot."""
        self._h1, self._h2 = h1, h2
        self._setup_pads()
        self._mode, self._option = _mode_of(option, self)
        self._h1_draw_opt, self._h2_draw_opt, self._graph_draw_opt = "hist", "E", "AP"
        ratiocalc.build(self)
        self._copy_axes()

    def _fitted(self, h1: Any, option: str, fitres: Any) -> None:
        """``TRatioPlot(h1, option, fitres)``: a fitted histogram and its fit's residuals."""
        self._h1 = h1
        if h1 is None:
            self.Warning("TRatioPlot", "Need a histogram.")
            return
        if h1.GetListOfFunctions().GetSize() < 1:
            self.Warning(
                "TRatioPlot", "Histogram given needs to have a (fit) function associated with it"
            )
            return
        self._proxy, self._fit_result = h1, fitres
        self._mode = ratiocalc.FIT_RESIDUAL
        self._option = _error_mode_of(option, self)
        if not ratiocalc.build(self):
            return
        self._h1_draw_opt = "E" if h1.GetSumw2N() > 0 else "hist"
        self._graph_draw_opt = "LX"
        self._copy_axes()
        self._setup_pads()

    def _copy_axes(self) -> None:
        """The axes the ranges and divisions are kept on, apart from the objects drawn."""
        self._shared_x = _clone_axis(self._h1.GetXaxis())
        self._up_y = _clone_axis(self._h1.GetYaxis())
        self._low_y = _clone_axis(self._ratio_graph.GetYaxis())

    def _setup_pads(self) -> None:
        """``SetupPads``: the upper and lower pads, split at ``fSplitFraction``, and the top one."""
        parent = current()
        if parent is None:
            self.Error("SetupPads", "need to create a canvas first")
            return
        pm = self._inset_width
        f = parent.GetHNDC() / parent.GetWNDC()
        self._upper_pad = TPad(
            "upper_pad", "", pm * f, self._split_fraction, 1.0 - pm * f, 1.0 - pm
        )
        self._lower_pad = TPad("lower_pad", "", pm * f, pm, 1.0 - pm * f, self._split_fraction)
        self._set_pad_margins()
        self._top_pad = TPad("top_pad", "", pm * f, pm, 1 - pm * f, 1 - pm)
        self._top_pad.SetBit(CANNOT_PICK)

    def _set_pad_margins(self) -> None:
        """``SetPadMargins``: each pad's margins from the ratio plot's."""
        for pad, top, bottom in ((self._upper_pad, self._up_top, self._up_bottom),
                                 (self._lower_pad, self._low_top, self._low_bottom)):  # fmt: skip
            if pad is not None:
                pad.SetMargin(self._left, self._right, bottom, top)

    # -- drawing ---------------------------------------------------------------------------

    def _draw_options(self, option: str) -> None:
        """``Draw``'s option: grid lines, bands, and which crowded label to hide."""
        if "nogrid" in option:
            self._show_gridlines, option = False, option.replace("nogrid", "")
        elif "grid" in option:
            self._show_gridlines, option = True, option.replace("grid", "")
        if "noconfint" in option:
            self._show_confint = False
        elif "confint" in option:
            self._show_confint = True
        modes = (("fhideup", ratioaxes.FORCE_HIDE_UP), ("fhidelow", ratioaxes.FORCE_HIDE_LOW),
                 ("hideup", ratioaxes.HIDE_UP), ("hidelow", ratioaxes.HIDE_LOW),
                 ("nohide", ratioaxes.NO_HIDE))  # fmt: skip
        self._hide_label_mode = next(
            (mode for word, mode in modes if word in option), ratioaxes.HIDE_LOW
        )

    def Draw(self, option: str = "") -> None:
        """``Draw``: the pads in the current pad, the plots in them, and the axes over both."""
        self._draw_options(str(option))
        parent = current()
        if parent is None:
            self.Error("Draw", "need to create a canvas first")
            return
        self._parent_pad = parent
        parent.add(self, "")
        for pad in (self._upper_pad, self._lower_pad):
            pad.SetLogx(parent.GetLogx())
            pad.SetGridx(parent.GetGridx())
            pad.SetGridy(parent.GetGridy())
        self._upper_pad.SetLogy(parent.GetLogy())
        self._upper_pad.Draw()
        self._lower_pad.Draw()
        self._top_pad.SetFillStyle(0)
        self._top_pad.Draw()
        self._upper_pad.cd()
        self._ci1.SetFillColor(self._ci1_color)
        self._ci2.SetFillColor(self._ci2_color)
        drawn = self._draw_fit() if self._mode == ratiocalc.FIT_RESIDUAL else self._draw_ratio()
        if drawn:
            self._sync_axes_ranges()
            ratioaxes.create(self)
            self._create_gridlines()
        parent.cd()

    def _draw_fit(self) -> bool:
        """The histogram and its fit above, the residuals and bands below."""
        functions = self._h1.GetListOfFunctions()
        found = [functions.At(i) for i in range(functions.GetSize())]
        fits = [f for f in found if f.InheritsFrom("TF1")]
        if not fits:
            self.Error("Draw", "h1 does not have a fit function")
            return False
        self._h1.Draw("A" + self._h1_draw_opt)
        fits[-1].Draw(self._fit_draw_opt + "same")
        self._lower_pad.cd()
        if self._show_confint:
            self._ci2.Draw("IA3")
            self._ci1.Draw("3")
            self._ratio_graph.Draw(self._graph_draw_opt + "SAME")
        else:
            self._ratio_graph.Draw("IA" + self._graph_draw_opt + "SAME")
        return True

    def _draw_ratio(self) -> bool:
        """The two histograms above - the first as its stack, if it was one - the ratio below."""
        self._proxy.Draw("A" + self._h1_draw_opt)
        self._h2.Draw("A" + self._h2_draw_opt + "same")
        self._lower_pad.cd()
        self._ratio_graph.Draw("IA" + self._graph_draw_opt)
        return True

    def _sync_axes_ranges(self) -> None:
        """``SyncAxesRanges``: both pads' x axes over the shared axis's range."""
        first, last = self._shared_range()
        ref = self.GetLowerRefXaxis()
        ref.SetLimits(first, last)
        ref.SetRangeUser(first, last)
        self.GetUpperRefXaxis().SetRangeUser(first, last)

    def _shared_range(self) -> tuple[float, float]:
        shared = self._shared_x
        return shared.GetBinLowEdge(shared.GetFirst()), shared.GetBinUpEdge(shared.GetLast())

    def _create_gridlines(self) -> None:
        """``CreateGridlines``: a dashed line in the lower pad for each reference position."""
        if not self._show_gridlines:
            return
        while len(self._gridlines) < len(self._gridline_positions):
            line = TLine(0, 0, 0, 0)
            line.SetLineStyle(2)
            self._lower_pad.add(line, "")
            self._gridlines.append(line)
        self._update_gridlines()

    def _update_gridlines(self) -> None:
        """``UpdateGridlines``: each line across the frame at its height, or nowhere if off it."""
        first, last = self._shared_range()
        low, high = self._lower_pad.GetUymin(), self._lower_pad.GetUymax()
        for i, line in enumerate(self._gridlines):
            y = self._gridline_positions[i] if i < len(self._gridline_positions) else None
            if y is not None and low <= y <= high:
                ends = (first, y, last, y)
            else:
                ends = (first, low, first, low)
            for setter, value in zip((line.SetX1, line.SetY1, line.SetX2, line.SetY2), ends):
                setter(value)

    # -- what it is made of -------------------------------------------------------------------

    def GetLowerRefGraph(self) -> Any:
        """The first graph in the lower pad: the one whose range is the lower frame's."""
        if self._lower_pad is None:
            self.Error("GetLowerRefGraph", "Lower pad has not been defined")
            return None
        drawn = [obj for obj, _ in self._lower_pad.primitives]
        if not drawn:
            self.Error("GetLowerRefGraph", "Lower pad does not have primitives")
            return None
        found = next((obj for obj in drawn if isinstance(obj, TGraph)), None)
        if found is None:
            self.Error("GetLowerRefGraph", "Did not find graph in list")
        return found

    def GetUpperRefObject(self) -> Any:
        """The first histogram or stack in the upper pad: whose axes are the upper frame's."""
        drawn = [obj for obj, _ in self._upper_pad.primitives]
        found = next((obj for obj in drawn if _is_stack(obj) or _is_histogram(obj)), None)
        if found is None:
            self.Error("GetUpperRefObject", "No upper ref object of TH1 or THStack type found")
        return found

    def GetUpperRefXaxis(self) -> Any:
        found = self.GetUpperRefObject()
        return found.GetXaxis() if found is not None else None

    def GetUpperRefYaxis(self) -> Any:
        found = self.GetUpperRefObject()
        return found.GetYaxis() if found is not None else None

    def GetLowerRefXaxis(self) -> Any:
        found = self.GetLowerRefGraph()
        return found.GetXaxis() if found is not None else None

    def GetLowerRefYaxis(self) -> Any:
        found = self.GetLowerRefGraph()
        return found.GetYaxis() if found is not None else None

    def GetXaxis(self) -> Any:
        return self._shared_x

    def GetUpYaxis(self) -> Any:
        return self._up_y

    def GetLowYaxis(self) -> Any:
        return self._low_y

    def GetCalculationOutputGraph(self) -> Any:
        return self._ratio_graph

    def GetConfidenceInterval1(self) -> Any:
        return self._ci1

    def GetConfidenceInterval2(self) -> Any:
        return self._ci2

    def GetUpperPad(self) -> Any:
        return self._upper_pad

    def GetLowerPad(self) -> Any:
        return self._lower_pad

    # -- how it looks ----------------------------------------------------------------------------

    def SetH1DrawOpt(self, opt: str) -> None:
        self._h1_draw_opt = str(opt)

    def SetH2DrawOpt(self, opt: str) -> None:
        """``SetH2DrawOpt``: ``same`` taken out, as the second histogram is always drawn so."""
        self._h2_draw_opt = str(opt).replace("same", "").replace("SAME", "")

    def SetGraphDrawOpt(self, opt: str) -> None:
        self._graph_draw_opt = str(opt)

    def SetFitDrawOpt(self, opt: str) -> None:
        self._fit_draw_opt = str(opt)

    def _margin(self, member: str, margin: float) -> None:
        setattr(self, member, f32(margin))
        self._set_pad_margins()

    def SetUpTopMargin(self, margin: float) -> None:
        self._margin("_up_top", margin)

    def SetUpBottomMargin(self, margin: float) -> None:
        self._margin("_up_bottom", margin)

    def SetLowTopMargin(self, margin: float) -> None:
        self._margin("_low_top", margin)

    def SetLowBottomMargin(self, margin: float) -> None:
        self._margin("_low_bottom", margin)

    def SetLeftMargin(self, margin: float) -> None:
        self._margin("_left", margin)

    def SetRightMargin(self, margin: float) -> None:
        self._margin("_right", margin)

    def SetSeparationMargin(self, margin: float) -> None:
        """``SetSeparationMargin``: the gap between the frames, half from each pad's margin."""
        sf = np.float32(self._split_fraction)
        self._up_bottom = f32(margin / 2.0 / float(np.float32(1) - sf))
        self._low_top = f32(margin / 2.0 / float(sf))
        self._set_pad_margins()

    def GetSeparationMargin(self) -> float:
        """The gap between the frames, as a fraction of the whole, in ROOT's single precision."""
        sf = np.float32(self._split_fraction)
        up = np.float32(self._up_bottom) * (np.float32(1) - sf)
        down = np.float32(self._low_top) * sf
        return float(np.float32(up + down))

    def SetSplitFraction(self, sf: float) -> None:
        """``SetSplitFraction``: where the pads meet, once the ratio plot is drawn."""
        if self._parent_pad is None:
            self.Warning("SetSplitFraction", "Can only be used after TRatioPlot has been drawn.")
            return
        if sf < 0.0001 or sf > 0.9999:
            self.Warning("SetSplitFraction", "Value %f is out of allowed range", float(sf))
            return
        self._split_fraction = f32(sf)
        pm, f = self._inset_width, self._aspect()
        self._upper_pad.SetPad(pm * f, self._split_fraction, 1.0 - pm * f, 1.0 - pm)
        self._lower_pad.SetPad(pm * f, pm, 1.0 - pm * f, self._split_fraction)

    def _aspect(self) -> float:
        return float(self._parent_pad.GetHNDC() / self._parent_pad.GetWNDC())

    def SetInsetWidth(self, width: float) -> None:
        """``SetInsetWidth``: the gap round the pads, once the ratio plot is drawn."""
        if self._parent_pad is None:
            self.Warning("SetInsetWidth", "Can only be used after TRatioPlot has been drawn.")
            return
        self._inset_width = float(width)
        self.SetSplitFraction(self._split_fraction)
        pm, f = self._inset_width, self._aspect()
        self._top_pad.SetPad(pm * f, pm, 1 - pm * f, 1 - pm)

    def SetConfidenceLevels(self, c1: float, c2: float) -> None:
        """``SetConfidenceLevels``: the bands' levels, and the lower plot worked out again."""
        self._cl1, self._cl2 = float(c1), float(c2)
        ratiocalc.build(self)

    def SetGridlines(self, gridlines: Any, numGridlines: int | None = None) -> None:
        """``SetGridlines``: where the reference lines go - a vector, or an array and its length."""
        values = list(gridlines)
        if numGridlines is not None:
            values = values[: int(numGridlines)]
        self._gridline_positions = [float(y) for y in values]

    def SetConfidenceIntervalColors(self, ci1: Any = YELLOW, ci2: Any = GREEN) -> None:
        """The bands' colours: numbers, or names like ``"kBlue"``, as ``TColorNumber`` takes."""
        self._ci1_color, self._ci2_color = _colour(ci1), _colour(ci2)

    def SetC1(self, c1: float) -> None:
        self._c1 = float(c1)

    def SetC2(self, c2: float) -> None:
        self._c2 = float(c2)

    def SetFitResult(self, fitres: Any) -> None:
        self._fit_result = fitres

    def paint_pad(self) -> None:
        """``Paint``: the axes and grid lines placed afresh, as the parent pad is drawn."""
        if self._axes:
            ratioaxes.update(self)
            self._update_gridlines()
