"""``RooFFTConvPdf``: two densities convolved numerically, by FFT, over one observable.

``RooFFTConvPdf("lxg", "landau (X) gauss", t, landau, gauss)`` is the
density of the sum of a Landau and a Gaussian variable. RooFit samples both
on the observable's "cache" binning, convolves the samples by FFT - so the
observable is taken to be cyclic, and a buffer either side keeps the ends
apart - and keeps the result as a histogram, interpolated to second order
by default (:mod:`.fftcache`). A cache is made for each set of observables
the density is asked to be normalised over, the first time it is asked -
and by each copy RooFit makes of the density to fit, plot or generate
with, each announcing itself (:mod:`..copies`); the numbers are the same
for every copy.

With a function of the observable - ``RooFFTConvPdf(name, title, psif,
psi, pdf1, pdf2)``, ``psif = acos(cpsi)`` - the convolution is over ``psi``
and the density is read at ``psif``: a density of ``cpsi``, normalised
numerically since the histogram's sum is not its integral over ``cpsi``.

Its events are the sum of an event of each density when both draw their
own - the convolution variable's range opened for them, the sum kept only
inside it - and are sampled numerically otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..binning import RooUniformBinning
from ..collections import as_list
from ..integration import numeric
from ..messages import ERROR, INFO, log
from ..pdf import RooAbsPdf, normalized
from ..printing import address
from ..real import Context
from .. import copies
from .fftcache import EXTEND, FLAT, MIRROR, FFTCache

__all__ = ["RooFFTConvPdf"]

#: The most FFT bins RooFit aims for: with the default buffer, 930 bins make 1024.
FFT_BINS = 1024


def said_numeric(conv: Any, names: list[str], label: str, rng: Any = None) -> None:
    """``RooRealIntegral::init``'s line for an integral over ``names`` that is taken numerically."""
    method = "RooIntegrator1D" if len(names) == 1 else "RooAdaptiveIntegratorND"
    if len(names) == 1 and any(np.isinf(conv.bounds(names[0], rng))):
        method = "RooImproperIntegrator1D"
    log(conv, INFO, "NumericIntegration", f"RooRealIntegral::init({label}) using numeric integrator "
        f"{method} to calculate Int({','.join(names)})")  # fmt: skip


class _Caches:
    """``RooObjCacheManager``: one copy's caches, and the normalisation sets it has slots for.

    A copy starts with the slots of the density it was made from and none
    of its caches; a cache's code is the number of its slot.
    """

    def __init__(self, conv: Any, slots: list[frozenset[str]], purpose: str) -> None:
        self.conv = conv
        self.slots = list(slots)
        self.purpose = purpose
        self.made: dict[frozenset[str], FFTCache] = {}
        self.said_numeric = False

    def cache(self, nset: frozenset[str]) -> FFTCache:
        found = self.made.get(nset)
        return found if found is not None else self._make(nset)

    def _make(self, nset: frozenset[str]) -> FFTCache:
        """``getCache``: make and fill the cache, and say so - with RooFit's name for its density."""
        conv = self.conv
        made = FFTCache(conv, conv.actual_observables(nset), nset)
        made.weights({})
        if nset not in self.slots:
            self.slots.append(nset)
        self.made[nset] = made
        ordered = conv.ordered(nset)
        name = conv.cache_name(made.hist_obs, ordered)
        log(conv, INFO, "Caching", f"RooAbsCachedPdf::getCache({conv.GetName()}) creating new cache "
            f"{address(made)} with pdf {name} for nset ({','.join(ordered)}) with code {self.slots.index(nset)}")  # fmt: skip
        if nset and self.purpose != "fit" and not made.analytic_over(nset):
            said_numeric(conv, ordered, f"{name}_Int[{','.join(ordered)}]")
        return made

    def sterilize(self) -> None:
        self.made.clear()


class RooFFTConvPdf(RooAbsPdf):
    """The convolution of two densities of one observable, computed by FFT."""

    #: ``RooFFTConvPdf::BufStrat``: how the buffers either side of the range are filled.
    Extend, Mirror, Flat = EXTEND, MIRROR, FLAT

    def __init__(self, name: Any, title: Any, *args: Any) -> None:
        super().__init__(name, title)
        form = 0 if isinstance(args[1], RooAbsPdf) else 1
        xprime = args[0] if form else None
        convVar, pdf1, pdf2 = args[form:form + 3]
        self._order = int(args[form + 3]) if len(args) > form + 3 else 2
        self.x = self._proxy("!x", convVar)
        self.xprime = self._proxy("!xprime", xprime) if xprime is not None else None
        self.pdf1 = self._proxy("!pdf1", pdf1)
        self.pdf2 = self._proxy("!pdf2", pdf2)
        self._fraction = 0.1
        self._strategy = EXTEND
        self._shift2 = (convVar.getMax("cache") + convVar.getMin("cache")) / 2
        self._cache_obs: list[Any] = []
        self._prepare_binning()
        self._caches = _Caches(self, [], "original")

    def _prepare_binning(self) -> None:
        """``prepareFFTBinning``: give the observable a "cache" binning of 930 bins if it has none."""
        x = self.x
        if x.hasBinning("cache"):
            return
        binning = x.getBinning()
        optimal = int(FFT_BINS / (1.0 + self._fraction))
        if binning.numBins() < optimal and binning.isUniform():
            log(self, INFO, "Caching", f"Changing internal binning of variable '{x.GetName()}' in FFT "
                f"'{self.GetName()}' from {binning.numBins()} to {optimal} to improve the precision of the "
                "numerical FFT. This can be done manually by setting an additional binning named 'cache'.")  # fmt: skip
            x.setBinning(RooUniformBinning(binning.lowBound(), binning.highBound(), optimal), "cache")
        else:
            log(self, ERROR, "Caching", f"The internal binning of variable {x.GetName()} is not uniform. "
                "The numerical FFT will likely yield wrong results.")  # fmt: skip
            x.setBinning(binning, "cache")

    # -- settings -----------------------------------------------------------------

    def setBufferFraction(self, frac: float) -> None:
        if frac < 0:
            log(self, ERROR, "InputArguments", f"RooFFTConvPdf::setBufferFraction({self.GetName()}) fraction "
                "should be greater than or equal to zero")  # fmt: skip
            return
        self._fraction = float(frac)
        self._caches.sterilize()

    def getBufferFraction(self) -> float:
        return self._fraction

    bufferFraction = getBufferFraction

    def setBufferStrategy(self, bs: int) -> None:
        self._strategy = int(bs)
        self._caches.sterilize()

    def bufferStrategy(self) -> int:
        return self._strategy

    def setInterpolationOrder(self, order: int) -> None:
        self._order = int(order)

    def getInterpolationOrder(self) -> int:
        return self._order

    def setCacheObservables(self, obs: Any) -> None:
        self._cache_obs = list(as_list(obs))
        self._caches.sterilize()

    def cacheObservables(self) -> Any:
        from ..collections import RooArgSet

        return RooArgSet(self._cache_obs)

    def printMetaArgs(self) -> str:
        x = self.x.GetName()
        return f"{self.pdf1.GetName()}({x}) (*) {self.pdf2.GetName()}({x}) "

    # -- which caches -------------------------------------------------------------

    def _observables_of(self, nset: frozenset[str]) -> list[Any]:
        """``getObservables(nset)`` of both densities, in the order of their variables."""
        found: list[Any] = []
        for pdf in (self.pdf1, self.pdf2):
            for one in pdf.leaves():
                if one.GetName() in nset and all(one is not seen for seen in found):
                    found.append(one)
        return found

    def actual_observables(self, nset: frozenset[str]) -> list[Any]:
        """``actualObservables``: what a cache for ``nset`` holds - the convolution observable,
        with the categories of ``nset`` and the observables named by ``setCacheObservables``
        if it is in ``nset``, or with every observable of ``nset`` otherwise."""
        found = self._observables_of(nset)
        cached = {one.GetName() for one in self._cache_obs}
        if self.x.GetName() in nset or cached:
            found = [one for one in found if hasattr(one, "lookupIndex") or one.GetName() in cached]
        for one in [self.x, *(self._cache_obs if self.x.GetName() in nset else [])]:
            if all(one.GetName() != seen.GetName() for seen in found):
                found.append(one)
        return found

    def pdf_observables(self, hist_obs: list[Any]) -> list[Any]:
        """``pdfObservable``: each histogram observable as the density reads it - ``xprime`` for ``x``."""
        mine = self.x.GetName()
        return [self.xprime if self.xprime is not None and one.GetName() == mine else one for one in hist_obs]

    def pdf_observable_names(self, names: list[str]) -> set[str]:
        """The variables the density's observables are made of, for the histogram of ``names``."""
        found: set[str] = set()
        for obs in self.pdf_observables([self.variable(name) for name in names]):
            found |= {obs.GetName()} if obs.isFundamental() else obs.dependents()
        return found

    def ordered(self, nset: frozenset[str]) -> list[str]:
        """``nset``'s names as RooFit keeps the set: the first density's variables first."""
        order = [one.GetName() for one in (*self.pdf1.leaves(), *self.pdf2.leaves(), *self.leaves())]
        return sorted(nset, key=lambda name: order.index(name) if name in order else len(order))

    def cache_name(self, hist_obs: list[Any], ordered: list[str]) -> str:
        """``lx_CONV_gauss_CACHE_Obs[t]_NORM_t``: RooFit's name for a cache's density."""
        final: list[str] = []
        for obs in self.pdf_observables(hist_obs):
            names = [obs.GetName()] if obs.isFundamental() else sorted(obs.dependents())
            final += [name for name in names if name not in final]
        name = f"{self.pdf1.GetName()}_CONV_{self.pdf2.GetName()}_CACHE_Obs[{','.join(final)}]"
        return name + ("_NORM_" + "_".join(ordered) if ordered else "")

    def _active(self) -> _Caches:
        found = copies.state_of(self)
        return found if found is not None else self._caches

    def copy_for(self, purpose: str, nset: frozenset[str]) -> _Caches:
        """The caches of a copy made to ``purpose``: a fit's and a generator's make the one they
        normalise with at once, a plot's when it first needs one."""
        made = _Caches(self, self._caches.slots, purpose)
        names = frozenset(nset) & self.dependents()
        if purpose in ("fit", "generate") and names:
            made.cache(names)
        return made

    # -- values and integrals -----------------------------------------------------

    def compute(self, ctx: Context) -> Any:
        """``getVal()``: the histogram of the cache for no normalisation set, as it reads."""
        return self._active().cache(frozenset()).raw(ctx)

    def value(self, ctx: Context, nset: Any = None, rng: Any = None) -> Any:
        """The cache for ``nset``, over its integral: RooFit's cached density normalised."""
        names = frozenset(nset or ()) & self.dependents()
        if not names:
            return self.compute(ctx)
        caches = self._active()
        cache = caches.cache(names)
        return normalized(cache.raw(ctx), self._norm(caches, cache, names, ctx, rng))

    def norm(self, ctx: Context, nset: Any, rng: Any = None) -> Any:
        names = frozenset(nset or ()) & self.dependents()
        if not names:
            return self.compute(ctx)
        caches = self._active()
        return self._norm(caches, caches.cache(names), names, ctx, rng)

    def _norm(self, caches: _Caches, cache: FFTCache, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        """The integral over ``names`` a fit or the cache's own density divides by.

        A sum of weights when the histogram's observables are the density's;
        otherwise a numerical integral - in a fit, of the density itself,
        whose value is that of a cache for no normalisation set, which the
        fit's copy then makes.
        """
        if cache.analytic_over(names):
            return cache.summed(names, ctx)
        if caches.purpose == "fit":
            if not caches.said_numeric:
                caches.said_numeric = True
                said_numeric(self, sorted(names), f"{self.GetName()}_Int[{','.join(sorted(names))}]", rng)
            cache = caches.cache(frozenset())
        order = [one.GetName() for one in self.leaves() if one.GetName() in names]
        return numeric(self, order, cache.raw, ctx, rng)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        """``forceAnalyticalInt``: every observable - the cache's density integrates itself."""
        return frozenset(names) & self.dependents()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        """The integral of the cache for ``names``: its sum of weights, or numerically if not that."""
        names = frozenset(names)
        cache = self._active().cache(names)
        if not rng and cache.analytic_over(names):
            return cache.summed(names, ctx)
        order = [one.GetName() for one in self.leaves() if one.GetName() in names]
        return numeric(self, order, cache.raw, ctx, rng)

    # -- generating ---------------------------------------------------------------

    def gen_context(self, names: frozenset[str]) -> Any:
        """``genContext``: the sum of the two densities' own events if both draw them, else sampling."""
        from .fftgen import ConvolutionContext, SampledContext

        if names == frozenset([self.x.GetName()]) and all(self._draws(pdf) for pdf in (self.pdf1, self.pdf2)):
            return ConvolutionContext(self, names)
        return SampledContext(self, names)

    def _draws(self, pdf: Any) -> bool:
        """``getGenerator`` and ``isDirectGenSafe``: whether ``pdf`` draws the observable itself, safely."""
        code = getattr(pdf, "generator_code", None)
        mine = self.x.GetName()
        if code is None or not code(frozenset([mine])) or pdf.findServer(mine) is None:
            return False
        return not any(server.GetName() != mine and mine in server.dependents() for server in pdf.servers())
