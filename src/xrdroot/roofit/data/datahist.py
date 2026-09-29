"""``RooDataHist``: binned data - a weight per bin, kept as one event at each bin's centre.

``RooDataHist("dh", "dh", x, Import(h))`` takes a histogram's bins - the
variable's range widened to the nearest bin boundaries, as
``RooDataHist::adjustBinning`` widens it - and ``RooDataHist("dh", "dh",
x, data)`` bins a dataset in the variable's binning. Either way it is a
dataset of one event per bin, at the bin's centre, weighing the bin's
content, with the squared errors kept beside: so a likelihood, a plot or a
sum over it is the same code as for unbinned data, as it is in RooFit.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..binning import RooBinning, RooUniformBinning
from ..cmdargs import RooCmdArg, commands
from ..collections import as_list
from ..messages import INFO, log
from ..printing import g
from .store import RooAbsData

__all__ = ["RooDataHist"]


def _histogram_of(obj: Any) -> Any:
    """The :class:`xrdroot.Histogram` a pyroot ``TH1`` - or anything with ``_xrd`` - stands for."""
    return getattr(obj, "_xrd", obj)


def _arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], list[Any], Any]:
    """The variables, the datasets to fill from, and the options, of a constructor's arguments."""
    variables = [a for a in args if not isinstance(a, (RooCmdArg, RooAbsData))]
    sources = [a for a in args if isinstance(a, RooAbsData)]
    given = [a for a in args if isinstance(a, RooCmdArg)]
    hists = [a for a in variables[1:] if hasattr(a, "_xrd")]  # (name, title, vars, TH1*)
    options = commands([*given, RooCmdArg("Import", hists[0])] if hists else given, kwargs)
    return (as_list(variables[0]) if variables else []), sources, options


class RooDataHist(RooAbsData):
    """Binned data over one or more variables."""

    def __init__(self, name: Any = "", title: Any = "", *args: Any, **kwargs: Any) -> None:
        chosen, sources, options = _arguments(args, kwargs)
        super().__init__(name, title, chosen)
        self._weights = np.zeros(0)
        self._sumw2 = np.zeros(0)
        histogram = options.get("Import")
        if histogram is not None and not isinstance(histogram, RooAbsData):
            self._import(chosen, _histogram_of(histogram), bool(options.get("Import", 1, False)))
            return
        self._empty()
        source = sources[0] if sources else histogram
        if source is not None:
            self.add_data(source)

    # -- bins ---------------------------------------------------------------------

    def _grid_weights(self) -> np.ndarray[Any, Any]:
        """The bins' weights: a binned dataset always has them."""
        assert self._weights is not None
        return self._weights

    def _edges(self) -> list[np.ndarray[Any, Any]]:
        return [one.getBinning().array() for one in self._vars]

    def _empty(self) -> None:
        """A bin for every combination of the variables' bins, each weighing nothing."""
        centres = [0.5 * (e[1:] + e[:-1]) for e in self._edges()]
        grid = list(itertools.product(*centres)) or [()]
        for index, one in enumerate(self._vars):
            self._columns[one.GetName()] = np.array(
                [point[index] for point in grid], dtype=np.float64
            )
        self._weights = np.zeros(len(grid))
        self._sumw2 = np.zeros(len(grid))
        for one, centre in zip(self._vars, grid[-1]):  # RooFit's variables end at the last bin
            one.setVal(float(centre))

    def _bin_of(self, columns: dict[str, Any]) -> np.ndarray[Any, Any]:
        """The flat bin number of each event, or -1 for one outside every range."""
        flat = np.zeros(len(next(iter(columns.values()))), dtype=np.int64)
        valid = np.ones(len(flat), dtype=bool)
        for one, edges in zip(self._vars, self._edges()):
            values = np.asarray(columns[one.GetName()], dtype=np.float64)
            found = np.searchsorted(edges, values, side="right") - 1
            found = np.where(values == edges[-1], len(edges) - 2, found)
            valid &= (found >= 0) & (found < len(edges) - 1)
            flat = flat * (len(edges) - 1) + np.clip(found, 0, len(edges) - 2)
        return np.where(valid, flat, -1)

    def add_data(self, data: RooAbsData) -> None:
        """Fill in a dataset's events, each weighing its weight."""
        columns = {one.GetName(): data.column(one.GetName()) for one in self._vars}
        bins = self._bin_of(columns)
        keep = bins >= 0
        weights = data.weights()[keep]
        size = len(self._grid_weights())
        self._weights = self._grid_weights() + np.bincount(bins[keep], weights, size)
        self._sumw2 = self._sumw2 + np.bincount(bins[keep], weights**2, size)

    def _import(self, chosen: list[Any], histogram: Any, density: bool) -> None:
        """``importTH1``: the histogram's bins inside the variables' ranges, the ranges widened to
        them."""
        offsets = [self._adjust(var, histogram.axes[i]) for i, var in enumerate(chosen)]
        self._empty()
        values, errors = histogram.values(), histogram.errors()
        window = tuple(slice(o, o + one.getBins()) for o, one in zip(offsets, self._vars))
        self._weights = np.asarray(values[window], dtype=np.float64).reshape(-1).copy()
        self._sumw2 = np.asarray(errors[window], dtype=np.float64).reshape(-1) ** 2
        if density:
            volume = self.binVolumes()
            self._weights, self._sumw2 = self._weights * volume, self._sumw2 * volume

    def _adjust(self, var: Any, axis: Any) -> int:
        """``_adjustBinning``: the variable's range moved out to the axis's bin edges; its first
        bin."""
        edges = np.asarray(axis.edges(), dtype=np.float64)
        low, high = var.getMin(), var.getMax()
        tolerance = 1e-6 * (edges[-1] - edges[0]) / (len(edges) - 1)
        first = max(int(np.searchsorted(edges, low + tolerance, side="right")) - 1, 0)
        last = min(int(np.searchsorted(edges, high - tolerance, side="right")) - 1, len(edges) - 2)
        uniform = np.allclose(np.diff(edges), edges[1] - edges[0])
        made: Any = (
            RooUniformBinning(edges[first], edges[last + 1], last - first + 1)
            if uniform
            else _binning(edges[first : last + 2])
        )
        if uniform:
            var.setRange(float(edges[first]), float(edges[last + 1]))
        else:
            var.setBinning(made)
        if abs(edges[first] - low) > tolerance or abs(edges[last + 1] - high) > tolerance:
            log(
                self,
                INFO,
                "DataHandling",
                f"RooDataHist::adjustBinning({self._name}): fit range of "
                f"variable {var.GetName()} expanded to nearest bin boundaries: "
                f"[{g(low)},{g(high)}] --> "
                f"[{g(edges[first])},{g(edges[last + 1])}]",
            )
        self._vars.find(var.GetName()).setBinning(made)
        return first

    # -- what it holds ------------------------------------------------------------

    def binVolumes(self) -> np.ndarray[Any, Any]:
        widths = [np.diff(e) for e in self._edges()]
        grid = list(itertools.product(*widths)) or [()]
        return np.array([float(np.prod(point)) for point in grid])

    def weight(self, *args: Any) -> float:
        if args:
            bins = self._bin_of(
                {one.GetName(): [args[0].find(one.GetName()).getVal()] for one in self._vars}
            )
            return float(self._grid_weights()[bins[0]]) if bins[0] >= 0 else 0.0
        return super().weight()

    def numEntries(self) -> int:
        return len(self._weights) if self._weights is not None else 0

    def isWeighted(self) -> bool:
        return True

    def isNonPoissonWeighted(self) -> bool:
        w = self.weights()
        return bool(np.any((w != np.floor(w)) | (w < 0)))

    def default_binning(self, var: Any) -> Any:
        """``plotOnImpl``: a frame shows binned data in the data's own bins."""
        found = self._vars.find(var.GetName())
        return None if found is None else found.getBinning()

    def binnedClone(self, name: Any = None, title: Any = None) -> RooDataHist:
        return self

    def sum(self, correctForBinSize: bool = False, inverseCorr: bool = False) -> float:
        if not correctForBinSize:
            return float(np.sum(self._weights))
        volume = self.binVolumes()
        return float(np.sum(self._weights / volume if inverseCorr else self._weights * volume))

    def printValue(self) -> str:
        return f"{self.numEntries()} bins ({g(self.sumEntries())} weights)"

    def printArgs(self) -> str:
        return "[" + ",".join(one.GetName() for one in self._vars) + "]"

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        """The store's lines, then the binned dataset's own: its bins, their weight, its
        variables - listed straight after ``Observables:``, as RooFit leaves them."""
        from ..printing import kExtras, kName, kTitle, kValue, kVerbose

        text = super().printMultiline(contents, verbose, indent)
        text += f"{indent}Binned Dataset {self._name} ({self._title})\n"
        text += f"{indent}  Contains {self.numEntries()} bins with a total weight of "
        text += f"{g(self.sumEntries())}\n"
        if not verbose:
            return text + f"{indent}  Observables {self._vars.printValue()}\n"
        listed = self._vars.printStream(kName | kValue | kExtras | kTitle, kVerbose, indent + "  ")
        return text + f"{indent}  Observables: " + listed


def _binning(edges: np.ndarray[Any, Any]) -> RooBinning:
    made = RooBinning(float(edges[0]), float(edges[-1]))
    for edge in edges:
        made.addBoundary(float(edge))
    return made
