"""``TSpectrum2``: ROOT's two-dimensional spectrum processing, over :mod:`xrdroot.spectrum`.

A spectrum is ``Double_t **``, ``source[i][j]`` with ``i`` along x: a list
of rows - what a translated macro's ``new Double_t *[n]`` or a vector's
``data()`` is - or a two-dimensional NumPy array, filled in place.
``Search`` and ``Background`` of a ``TH2`` work on its bins, as ROOT's do.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ...spectrum.background2 import background2
from ...spectrum.gold2 import Response2, gold2
from ...spectrum.markov2 import smooth_markov2
from ...spectrum.search2 import search_high_res2
from .arrays import matrix_in, matrix_out
from .common import Peaks, option_flags

__all__ = ["TSpectrum2"]


def _histogram(value: Any) -> bool:
    return hasattr(value, "GetDimension")


class TSpectrum2(Peaks):
    """``TSpectrum2``: background, smoothing, deconvolution and peaks of two-dimensional spectra."""

    kBackSuccessiveFiltering: ClassVar[int] = 0
    kBackOneStepFiltering: ClassVar[int] = 1

    def ClassName(self) -> str:
        return "TSpectrum2"

    def Background(self, *args: Any) -> Any:
        """``Background(h, nIterX=20, nIterY=20, option="")`` - a histogram of the background -
        or ``Background(spectrum, ssizex, ssizey, numberIterationsX, numberIterationsY,
        direction, filterType)``, the background left in ``spectrum``."""
        if args and (args[0] is None or _histogram(args[0])):
            return self._background_of(*args)
        return self._background(*args)

    def _background(
        self, spectrum: Any, ssizex: int, ssizey: int, numberIterationsX: int,
        numberIterationsY: int, direction: int, filterType: int,
    ) -> str | None:  # fmt: skip
        found = background2(
            _plane(spectrum, ssizex, ssizey), int(numberIterationsX), int(numberIterationsY),
            direction != self.kBackIncreasingWindow, filterType == self.kBackOneStepFiltering,
        )  # fmt: skip
        return _answer(spectrum, found)

    def SmoothMarkov(self, source: Any, ssizex: int, ssizey: int, averWindow: int) -> str | None:
        """``SmoothMarkov``: ``source`` smoothed by a Markov chain over ``averWindow`` channels."""
        return _answer(source, smooth_markov2(_plane(source, ssizex, ssizey), int(averWindow)))

    def Deconvolution(
        self, source: Any, resp: Any, ssizex: int, ssizey: int,
        numberIterations: int, numberRepetitions: int, boost: float,
    ) -> str | None:  # fmt: skip
        """``Deconvolution``: ``source`` deconvolved by ``resp`` with Gold's method, boosted."""
        checks = (
            (int(ssizex) <= 0 or int(ssizey) <= 0, "Wrong parameters"),
            (int(numberIterations) <= 0, "Number of iterations must be positive"),
            (int(numberRepetitions) <= 0, "Number of repetitions must be positive"),
        )
        refused = next((message for failed, message in checks if failed), None)
        if refused is not None:
            return refused
        values = _plane(resp, ssizex, ssizey)
        response = Response2(values)
        if response.lengths[0] == -1:
            return "Zero response data"
        found = gold2(_plane(source, ssizex, ssizey), response, values, int(numberIterations),
                      int(numberRepetitions), float(boost))  # fmt: skip
        return _answer(source, found)

    def SearchHighRes(
        self, source: Any, dest: Any, ssizex: int, ssizey: int, sigma: float, threshold: float,
        backgroundRemove: bool, deconIterations: int, markov: bool, averWindow: int,
    ) -> int:  # fmt: skip
        """``SearchHighRes``: the number of peaks found, their positions in channels in
        ``GetPositionX`` and ``GetPositionY``, and the deconvolved spectrum in ``dest``."""
        found = search_high_res2(
            _plane(source, ssizex, ssizey), float(sigma), float(threshold),
            bool(backgroundRemove), int(deconIterations), bool(markov), int(averWindow),
            self.fMaxPeaks,
        )  # fmt: skip
        if isinstance(found, str):
            self.Error("SearchHighRes", found)
            return 0
        peaks, ys = found
        if peaks.deconvolved is None:
            return 0
        matrix_out(dest, peaks.deconvolved)
        self._found(peaks.positions)
        self.fPositionY[: len(ys)] = ys
        return self.fNPeaks

    def _background_of(self, h: Any, nIterX: int = 20, nIterY: int = 20, option: str = "") -> Any:
        if h is None:
            return None
        if h.GetDimension() != 2:
            self.Error("Background", "Only implemented for 2-d histograms")
            return None
        opt = str(option).lower()
        xs, ys = _span(h.GetXaxis()), _span(h.GetYaxis())
        source = [[h.GetBinContent(i, j) for j in ys] for i in xs]
        self._background(source, len(xs), len(ys), nIterX, nIterY,
                         int("backincreasingwindow" not in opt),
                         int("backonestepfiltering" in opt))  # fmt: skip
        name = f"{h.GetName()}_background"
        hb = h.Clone(name)
        hb.Reset()
        hb.GetListOfFunctions().Delete()
        for row, i in zip(source, xs, strict=False):
            for value, j in zip(row, ys, strict=False):
                hb.SetBinContent(i, j, value)
        hb.SetEntries(len(xs) * len(ys))
        if "same" in opt:
            from .tspectrum import _undraw

            _undraw(name)
            hb.Draw("same")
        return hb

    def Search(self, hin: Any, sigma: float = 2, option: str = "", threshold: float = 0.05) -> int:
        """``Search``: the peaks of ``hin`` - their bin centres in ``GetPositionX`` and
        ``GetPositionY`` - marked on it, and it drawn with ``option``, unless ``goff``."""
        if hin is None:
            return 0
        if hin.GetDimension() != 2:
            self.Error("Search", "Must be a 2-d histogram")
            return 0
        opt, (quiet, plain) = option_flags(option, "nobackground", "nomarkov")
        sizex, sizey = hin.GetXaxis().GetNbins(), hin.GetYaxis().GetNbins()
        source = [[hin.GetBinContent(i + 1, j + 1) for j in range(sizey)] for i in range(sizex)]
        npeaks = self.SearchHighRes(source, np.zeros((sizex, sizey)), sizex, sizey, sigma,
                                    100 * threshold, not quiet, self.fgIterations, not plain,
                                    self.fgAverageWindow)  # fmt: skip
        for i in range(npeaks):
            binx, biny = 1 + int(self.fPositionX[i] + 0.5), 1 + int(self.fPositionY[i] + 0.5)
            self.fPositionX[i] = hin.GetXaxis().GetBinCenter(binx)
            self.fPositionY[i] = hin.GetYaxis().GetBinCenter(biny)
        if "goff" in opt or not npeaks:
            return npeaks
        self._mark(hin, npeaks)
        hin.Draw(option)
        return npeaks

    @staticmethod
    def StaticSearch(
        hist: Any, sigma: float = 2, option: str = "goff", threshold: float = 0.05
    ) -> int:
        """``TSpectrum2::StaticSearch``: ``Search`` by a spectrum made for it."""
        return TSpectrum2().Search(hist, sigma, option, threshold)

    @staticmethod
    def StaticBackground(hist: Any, nIterX: int = 20, nIterY: int = 20, option: str = "") -> Any:
        """``TSpectrum2::StaticBackground``: ``Background`` by a spectrum made for it."""
        return TSpectrum2().Background(hist, nIterX, nIterY, option)


def _span(axis: Any) -> list[int]:
    """The bins of ``axis``'s range, as ``GetFirst`` to ``GetLast``."""
    return list(range(axis.GetFirst(), axis.GetLast() + 1))


def _plane(source: Any, sizex: int, sizey: int) -> np.ndarray[Any, Any]:
    """``source[i][j]`` for ``i < sizex``, ``j < sizey`` - none for sizes that are not positive."""
    if int(sizex) <= 0 or int(sizey) <= 0:
        return np.zeros((max(int(sizex), 0), max(int(sizey), 0)))
    return matrix_in(source, sizex, sizey)


def _answer(target: Any, found: Any) -> str | None:
    """ROOT's message for a refusal; else ``found`` - if there is one - written into ``target``."""
    if isinstance(found, str):
        return found
    if found is not None:
        matrix_out(target, found)
    return None
