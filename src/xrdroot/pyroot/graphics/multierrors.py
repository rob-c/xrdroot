"""``TGraphMultiErrors``: points whose y errors come in a layer per kind of uncertainty.

ROOT keeps one layer of y errors for each source - a statistical one and a
systematic one - each with a line and a fill of its own, and adds the layers
up as the graph was told to: the first alone, the square sum of all of them,
or their absolute sum. What they add up to is what ``GetErrorY`` answers, and
it is ``xrdroot.graphmath`` that works it out, the same arithmetic the reading
side has always done - so a graph written here and read back sums its layers
exactly as one ROOT wrote does.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from ...graph import Graph
from ..core.graphs import TGraph
from ..core.objects import ATTRIBUTES, TAttFill, TAttLine
from ..core.wrapping import adopt, register

__all__ = ["TGraphMultiErrors"]

#: ``TGraphMultiErrors::ESummationModes``: how the layers of y errors add up.
ONLY_FIRST, SQUARE_SUM, ABS_SUM = 0, 1, 2


def _cut(values: Any, count: int) -> np.ndarray[Any, Any]:
    """``values`` cut or padded with zeros to ``count``, as ROOT's arrays grow and shrink."""
    made = np.zeros(count) if values is None else np.asarray(values, dtype=np.float64).reshape(-1)
    return np.concatenate([made[:count], np.zeros(max(count - len(made), 0))])


def _attribute_groups() -> dict[str, dict[str, Any]]:
    """One layer's line and fill, as ROOT's default constructors give them."""
    return {group: dict(ATTRIBUTES[group]) for group in ("TAttLine", "TAttFill")}


class _Layer(TAttLine, TAttFill):
    """One layer's line and fill, which ``GetAttLine(e)`` and ``GetAttFill(e)`` hand out.

    What it holds is the graph's own dictionary for that layer, so a colour set
    through it is the colour the layer is written with.
    """

    def __init__(self, held: dict[str, dict[str, Any]]) -> None:
        self._held = held

    def _attribute_holder(self) -> dict[str, dict[str, Any]]:
        return self._held


class _WrittenObject:
    """What a graph this session wrote is read back from, under the key it was written to.

    ``TFile::Get`` in ROOT reads the record the file has just been given, and the record's
    own reader here is made only when a file is opened to read - so what answers is the
    graph the record was made from, exactly as the kit already does for a class a program
    declared. A file reopened from disk is read by its own bytes, as it always is.
    """

    def __init__(self, written: Any) -> None:
        self._written = written

    def read(self) -> Any:
        return self._written


def _attach(directory: Any, name: str, xrd: Any) -> None:
    """Keep what was written beside the key it went in under, for a file still open to find.

    A key that is not there is nothing to attach to, which is what a directory the object
    was kept in none the worse for - ``Get`` then answers as it would for any other class.
    """
    key = directory.GetKey(name, 9999)
    if key is not None:
        key._key = _WrittenObject(xrd)


class TGraphMultiErrors(TGraph):
    """``TGraphMultiErrors(n, x, y, exl, exh, eyl, eyh[, mode])``: a graph of layers.

        >>> g = ROOT.TGraphMultiErrors(2, [1, 2], [3, 4])    # doctest: +SKIP
        >>> g.AddYError(2, [0.1, 0.1], [0.2, 0.2])           # doctest: +SKIP
        >>> g.SetSumErrorsMode(ROOT.TGraphMultiErrors.kSquareSum)  # doctest: +SKIP
    """

    CLASS_TITLE = "A graph with multiple y error dimensions"
    kOnlyFirst = ONLY_FIRST
    kSquareSum = SQUARE_SUM
    kAbsSum = ABS_SUM

    def __init__(self, *args: Any) -> None:
        self._layers: list[list[np.ndarray[Any, Any]]] = []
        self._ex: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] = (
            np.zeros(0),
            np.zeros(0),
        )
        self._mode = ONLY_FIRST
        self._layer_atts: list[dict[str, dict[str, Any]]] = []
        super().__init__(*args)

    # -- ROOT's constructors ----------------------------------------------------------------

    def _construct(self, args: tuple[Any, ...]) -> None:
        """``(n)``, ``(n, x, y)`` or ``(n, x, y, exl, exh, eyl, eyh[, mode])`` - one layer."""
        if not args:
            return
        given = (len(args[0]), *args) if np.ndim(args[0]) > 0 else args
        count = int(given[0])
        rest = [*given[1:], *([None] * 8)]
        self._points = (_cut(rest[0], count), _cut(rest[1], count))
        self._ex = (_cut(rest[2], count), _cut(rest[3], count))
        self._append_layer(rest[4], rest[5], count)
        self._mode = int(rest[7]) if rest[7] is not None else ONLY_FIRST

    def _adopted(self, xrd: Any) -> None:
        """A ``TGraphMultiErrors`` read from a file: its layers, sum mode and per-layer look."""
        TGraph._adopted(self, xrd)
        self._layers = [[np.array(low), np.array(high)] for low, high in xrd.layers]
        low, high = xrd.xerr or (np.zeros(len(xrd)), np.zeros(len(xrd)))
        self._ex = (np.array(low), np.array(high))
        self._mode = xrd.sum_errors_mode
        self._layer_atts = self._holds_attributes(xrd)
        self._cached = None  # built again from the layers, so a change to one is written

    def _holds_attributes(self, xrd: Any) -> list[dict[str, dict[str, Any]]]:
        """Each layer's line and fill, from the file's own where it wrote them."""
        filled = list(xrd.members.get("fAttFill") or ())
        lined = list(xrd.members.get("fAttLine") or ())
        held = []
        for at in range(len(self._layers)):
            layer = _attribute_groups()
            for group, source in (("TAttFill", filled), ("TAttLine", lined)):
                if at < len(source):
                    layer[group].update(dict(source[at] or {}))
            held.append(layer)
        return held

    def _built(self) -> Any:
        """The xrdroot graph of the points and layers, as the writer lays them out."""
        made = Graph.from_layers(
            self.GetName(),
            *self._points,
            title=self.GetTitle(),
            xerr=self._ex,
            layers=self._layers,
            mode=self._mode,
            fills=[held["TAttFill"] for held in self._layer_atts],
            lines=[held["TAttLine"] for held in self._layer_atts],
        )
        made._core["fFunctions"] = self._functions
        made._core["fMinimum"], made._core["fMaximum"] = self._extremes
        return made

    def Write(self, name: Any = None, option: int = 0, bufsize: int = 0) -> int:
        """``Write``: the graph into the open file, and readable back out of it at once.

        A program that writes a graph and asks the same file for it - as ROOT's own test of
        this class does - gets it: the key it went in under is handed what the record was
        made from, so ``Get`` answers with it rather than with nothing.
        """
        from ..core.directories import current_directory

        directory = current_directory()
        count = int(directory.WriteTObject(self, name, option))
        _attach(directory, str(name) if name else self.GetName(), self._xrd)
        return count

    # -- the layers -------------------------------------------------------------------------

    def GetNYErrors(self) -> int:
        """``GetNYErrors``: how many layers of y errors the graph keeps."""
        return len(self._layers)

    def _append_layer(self, low: Any, high: Any, count: int) -> None:
        """One more layer of bars, with a line and a fill of its own."""
        self._layers.append([_cut(low, count), _cut(high, count)])
        self._layer_atts.append(_attribute_groups())
        self._changed()

    def AddYError(self, count: Any = 0, low: Any = None, high: Any = None) -> None:
        """``AddYError(np, eyl, eyh)``: another kind of y error, over the points there are."""
        self._append_layer(low, high, self.GetN() or int(count))

    def DeleteYError(self, layer: int) -> None:
        """``DeleteYError(e)``: one layer fewer - and the last one is never deleted."""
        if self.GetNYErrors() <= 1 or not 0 <= layer < self.GetNYErrors():
            return
        for kept in (self._layers, self._layer_atts):
            del kept[layer]
        self._changed()

    def SetSumErrorsMode(self, mode: int) -> None:
        """``SetSumErrorsMode(mode)``: how the layers add up - ``kOnlyFirst`` by default."""
        self._mode = int(mode)
        self._changed()

    def GetSumErrorsMode(self) -> int:
        """``GetSumErrorsMode``: which of ROOT's three ways this graph adds its layers up."""
        return self._mode

    def GetErrorY(self, i: int, e: int | None = None) -> float:
        """``GetErrorY(i)`` - the layers summed - or ``GetErrorY(i, e)`` for one layer of them."""
        if e is None:
            return float(self._xrd.error_y(int(i)))
        if not 0 <= e < self.GetNYErrors() or not 0 <= i < self.GetN():
            return -1.0
        low, high = self._layers[e]
        return float(np.sqrt((low[i] * low[i] + high[i] * high[i]) / 2.0))

    def GetAttLine(self, e: int) -> Any:
        """``GetAttLine(e)``: the line the ``e``-th layer's bars are drawn with, or ``None``."""
        return self._layer(e)

    def GetAttFill(self, e: int) -> Any:
        """``GetAttFill(e)``: the fill the ``e``-th layer's bars are drawn with, or ``None``."""
        return self._layer(e)

    def _layer(self, e: int) -> Any:
        """The line and fill of one layer, if the graph keeps that many."""
        if not 0 <= e < len(self._layer_atts):
            return None
        return _Layer(self._layer_atts[e])

    # -- the points -------------------------------------------------------------------------

    def Set(self, n: int) -> None:
        """``Set(n)``: ``n`` points, every layer of bars grown or cut to them."""
        count = int(n)
        TGraph.Set(self, count)
        self._ex = (_cut(self._ex[0], count), _cut(self._ex[1], count))
        self._layers = [[_cut(low, count), _cut(high, count)] for low, high in self._layers]

    def SetPointEX(self, i: int, low: float, high: float) -> None:
        """``SetPointEX(i, exl, exh)``: point ``i``'s x bars, which are not summed."""
        if i < 0:
            return
        if i >= self.GetN():
            self.Set(i + 1)
        self._ex[0][i], self._ex[1][i] = float(low), float(high)
        self._changed()

    def SetPointEY(self, i: int, e: int, low: float, high: float) -> None:
        """``SetPointEY(i, e, eyl, eyh)``: point ``i``'s bars in layer ``e``."""
        if i < 0 or not 0 <= e < self.GetNYErrors():
            return
        if i >= self.GetN():
            self.Set(i + 1)
        self._layers[e][0][i], self._layers[e][1][i] = float(low), float(high)
        self._changed()

    def GetErrorXlow(self, i: int) -> float:
        return -1.0 if not 0 <= i < self.GetN() else float(self._ex[0][i])

    def GetErrorXhigh(self, i: int) -> float:
        return -1.0 if not 0 <= i < self.GetN() else float(self._ex[1][i])

    def GetErrorX(self, i: int) -> float:
        """``GetErrorX(i)``: ROOT's mean of the square sum of the two sides of the x bars."""
        if not 0 <= i < self.GetN():
            return -1.0
        low, high = float(self._ex[0][i]), float(self._ex[1][i])
        return float(np.sqrt((low * low + high * high) / 2.0))


register(TGraphMultiErrors.__name__, factory=partial(adopt, TGraphMultiErrors))
