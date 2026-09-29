"""``TSpectrum``: ROOT's one-dimensional spectrum processing, over :mod:`xrdroot.spectrum`.

    >>> import numpy as np
    >>> s = TSpectrum()
    >>> source = np.zeros(64); source[20] = source[40] = 100.0
    >>> s.SearchHighRes(source, np.zeros(64), 64, 2, 10, False, 3, False, 3)
    2

``Background``, ``SmoothMarkov``, ``Deconvolution``, ``DeconvolutionRL``,
``Unfolding`` and ``SearchHighRes`` take a spectrum as ROOT does - an array
and its size - and leave their answer in it; the ones that refuse their
arguments hand back ROOT's message, and ``None`` otherwise, as ROOT's
``const char *`` is ``nullptr``. ``Search`` and ``Background`` of a
histogram work on the bins of its axis range, as ROOT's do.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...spectrum.background import background
from ...spectrum.gold import gold, response_of
from ...spectrum.lucy import lucy
from ...spectrum.markov import smooth_markov
from ...spectrum.search import search_high_res
from ...spectrum.unfolding import unfold
from .arrays import matrix_in, vector_in, vector_out
from .common import Peaks, option_flags

__all__ = ["TSpectrum"]

#: ``kBackOrder2`` to ``kBackOrder8``, and the filter order each stands for.
ORDERS = {0: 2, 1: 4, 2: 6, 3: 8}
#: The largest ``sigma`` ``Search`` picks for itself, when it is given less than 1.
WIDEST_SIGMA = 8


def _histogram(value: Any) -> bool:
    """Is ``value`` a histogram - what the first of ``Background``'s two forms takes?"""
    return hasattr(value, "GetDimension")


class TSpectrum(Peaks):
    """``TSpectrum``: background, smoothing, deconvolution and peaks of one-dimensional spectra."""

    kBackOrder2: ClassVar[int] = 0
    kBackOrder4: ClassVar[int] = 1
    kBackOrder6: ClassVar[int] = 2
    kBackOrder8: ClassVar[int] = 3
    kBackSmoothing3: ClassVar[int] = 3
    kBackSmoothing5: ClassVar[int] = 5
    kBackSmoothing7: ClassVar[int] = 7
    kBackSmoothing9: ClassVar[int] = 9
    kBackSmoothing11: ClassVar[int] = 11
    kBackSmoothing13: ClassVar[int] = 13
    kBackSmoothing15: ClassVar[int] = 15

    def __init__(self, maxpositions: int = 100, resolution: float = 1) -> None:
        super().__init__(max(int(maxpositions), 1), resolution)

    def ClassName(self) -> str:
        return "TSpectrum"

    # -- on arrays ---------------------------------------------------------------------

    def Background(self, *args: Any) -> Any:
        """``Background(h, nIter=20, option="")`` - a histogram of the background - or
        ``Background(spectrum, ssize, numberIterations, direction, filterOrder, smoothing,
        smoothWindow, compton)``, the background left in ``spectrum``."""
        if args and (args[0] is None or _histogram(args[0])):
            return self._background_of(*args)
        return self._background(*args)

    def _background(
        self, spectrum: Any, ssize: int, numberIterations: int, direction: int,
        filterOrder: int, smoothing: bool, smoothWindow: int, compton: bool,
    ) -> str | None:  # fmt: skip
        found = background(
            vector_in(spectrum, ssize), int(numberIterations),
            direction != self.kBackIncreasingWindow, ORDERS.get(int(filterOrder), 0),
            bool(smoothing), int(smoothWindow), bool(compton),
        )  # fmt: skip
        return _answer(spectrum, found)

    def SmoothMarkov(self, source: Any, ssize: int, averWindow: int) -> str | None:
        """``SmoothMarkov``: ``source`` smoothed by a Markov chain over ``averWindow`` channels."""
        return _answer(source, smooth_markov(vector_in(source, ssize), int(averWindow)))

    def Deconvolution(
        self, source: Any, response: Any, ssize: int,
        numberIterations: int, numberRepetitions: int, boost: float,
    ) -> str | None:  # fmt: skip
        """``Deconvolution``: ``source`` deconvolved by ``response`` with Gold's method, boosted."""
        return self._deconvolve(gold, source, response, ssize, numberIterations,
                                numberRepetitions, boost)  # fmt: skip

    def DeconvolutionRL(
        self, source: Any, response: Any, ssize: int,
        numberIterations: int, numberRepetitions: int, boost: float,
    ) -> str | None:  # fmt: skip
        """``DeconvolutionRL``: ``source`` deconvolved by ``response`` with Richardson-Lucy."""
        return self._deconvolve(lucy, source, response, ssize, numberIterations,
                                numberRepetitions, boost)  # fmt: skip

    def _deconvolve(
        self, method: Any, source: Any, response: Any, ssize: int,
        iterations: int, repetitions: int, boost: float,
    ) -> str | None:  # fmt: skip
        values = vector_in(response, ssize)
        if int(ssize) <= 0 or int(repetitions) <= 0:
            return "Wrong Parameters"
        found = response_of(values)
        if isinstance(found, str):
            return found
        out = method(vector_in(source, ssize), found, values, int(iterations), int(repetitions),
                     float(boost))  # fmt: skip
        return _answer(source, out)

    def Unfolding(
        self, source: Any, respMatrix: Any, ssizex: int, ssizey: int,
        numberIterations: int, numberRepetitions: int, boost: float,
    ) -> str | None:  # fmt: skip
        """``Unfolding``: ``source`` unfolded through ``respMatrix`` - ``respMatrix[j]`` the
        response of channel ``j`` of the answer - with Gold's method, boosted."""
        if int(ssizex) <= 0 or int(ssizey) <= 0:
            return "Wrong Parameters"
        matrix = matrix_in(respMatrix, ssizey, ssizex)
        found = unfold(vector_in(source, ssizex), matrix, int(numberIterations),
                       int(numberRepetitions), float(boost))  # fmt: skip
        return _answer(source, found)

    def SearchHighRes(
        self, source: Any, destVector: Any, ssize: int, sigma: float, threshold: float,
        backgroundRemove: bool, deconIterations: int, markov: bool, averWindow: int,
    ) -> int:  # fmt: skip
        """``SearchHighRes``: the number of peaks found, their positions in channels in
        ``GetPositionX``, and the deconvolved spectrum in ``destVector``."""
        found = search_high_res(
            vector_in(source, ssize), float(sigma), float(threshold), bool(backgroundRemove),
            int(deconIterations), bool(markov), int(averWindow), self.fMaxPeaks,
        )  # fmt: skip
        if isinstance(found, str):
            self.Error("SearchHighRes", found)
            return 0
        if found.deconvolved is None:
            return 0
        vector_out(destVector, found.deconvolved)
        self._found(found.positions)
        if found.full:
            self.Warning("SearchHighRes", "Peak buffer full")
        return self.fNPeaks

    Search1HighRes = SearchHighRes

    # -- on histograms -----------------------------------------------------------------

    def _background_of(self, h: Any, nIter: int = 20, option: str = "") -> Any:
        if h is None:
            return None
        if h.GetDimension() != 1:
            self.Error("Background", "Only implemented for 1-d histograms")
            return None
        opt = str(option).lower()
        first, last = h.GetXaxis().GetFirst(), h.GetXaxis().GetLast()
        size = last - first + 1
        source = [h.GetBinContent(i + first) for i in range(size)]
        order = next((o for o, name in ((3, "backorder8"), (2, "backorder6"), (1, "backorder4"))
                      if name in opt), 0)  # fmt: skip
        window = next((w for w in (15, 13, 11, 9, 7, 5) if f"backsmoothing{w}" in opt), 3)
        direction = int("backincreasingwindow" not in opt)
        self._background(source, size, nIter, direction, order, "nosmoothing" not in opt,
                         window, "compton" in opt)  # fmt: skip
        return _background_histogram(h, source, first, "same" in opt)

    def Search(self, hin: Any, sigma: float = 2, option: str = "", threshold: float = 0.05) -> int:
        """``Search``: the peaks of ``hin`` - their bin centres in ``GetPositionX``, their contents
        in ``GetPositionY`` - marked on it, and it drawn, unless ``goff`` or ``nodraw``."""
        if hin is None:
            return 0
        if hin.GetDimension() > 2:
            self.Error("Search", "Only implemented for 1-d and 2-d histograms")
            return 0
        if threshold <= 0 or threshold >= 1:
            self.Warning("Search", "threshold must 0<threshold<1, threshold=0.05 assumed")
            threshold = 0.05
        opt, (quiet, plain, hidden) = option_flags(option, "nobackground", "nomarkov", "nodraw")
        if hin.GetDimension() != 1:
            return 0
        first, last = hin.GetXaxis().GetFirst(), hin.GetXaxis().GetLast()
        size = last - first + 1
        if sigma < 1:
            sigma = min(max(size // self.fMaxPeaks, 1), WIDEST_SIGMA)
        source = [hin.GetBinContent(i + first) for i in range(size)]
        npeaks = self.SearchHighRes(source, [0.0] * size, size, sigma, 100 * threshold, not quiet,
                                    self.fgIterations, not plain, self.fgAverageWindow)  # fmt: skip
        for i in range(npeaks):
            at = first + int(self.fPositionX[i] + 0.5)
            self.fPositionX[i], self.fPositionY[i] = hin.GetBinCenter(at), hin.GetBinContent(at)
        if "goff" in opt or not npeaks:
            return npeaks
        self._mark(hin, npeaks)
        if not hidden:
            hin.Draw(opt.replace(" ", "").replace(",", ""))
        return npeaks

    @staticmethod
    def StaticSearch(
        hist: Any, sigma: float = 2, option: str = "goff", threshold: float = 0.05
    ) -> int:
        """``TSpectrum::StaticSearch``: ``Search`` by a spectrum made for it."""
        return TSpectrum().Search(hist, sigma, option, threshold)

    @staticmethod
    def StaticBackground(hist: Any, niter: int = 20, option: str = "") -> Any:
        """``TSpectrum::StaticBackground``: ``Background`` by a spectrum made for it."""
        return TSpectrum().Background(hist, niter, option)


def _answer(target: Any, found: Any) -> str | None:
    """ROOT's message for a refusal; else ``found`` - if there is one - written into ``target``."""
    if isinstance(found, str):
        return found
    if found is not None:
        vector_out(target, found)
    return None


def _background_histogram(h: Any, values: list[float], first: int, same: bool) -> Any:
    """``h``'s clone ``<name>_background``, red, of ``values`` from bin ``first`` - drawn over
    the pad with ``same``, in place of one drawn there before."""
    name = f"{h.GetName()}_background"
    hb = h.Clone(name)
    hb.Reset()
    hb.GetListOfFunctions().Delete()
    hb.SetLineColor(2)
    for i, value in enumerate(values):
        hb.SetBinContent(i + first, value)
    hb.SetEntries(len(values))
    if same:
        _undraw(name)
        hb.Draw("same")
    return hb


def _undraw(name: str) -> None:
    """Take what the current pad draws called ``name`` off it, as ``delete`` of it would."""
    from ..graphics.pads import current

    pad = current()
    if pad is not None:
        drawn = pad.GetPrimitive(name)
        pad.primitives = [(obj, how) for obj, how in pad.primitives if obj is not drawn]
