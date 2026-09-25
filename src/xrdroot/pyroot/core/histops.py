"""What is done with a histogram: arithmetic, rebinning, projecting, fitting, drawing from it.

Each is ROOT's method over xrdroot's: ``Add``, ``Scale``, ``Divide`` change
the histogram in place, ``Rebin(n)`` does too unless given a new name, and a
projection is a new histogram, kept in the current directory under ROOT's
name for it - ``h_px``, ``h_pfx`` - replacing without a word one of that
name, as ROOT reuses it. ``Fit`` hands back a ``TFitResultPtr``; the draws
of ``FillRandom`` and ``GetRandom`` are ``gRandom``'s.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .directories import current_directory
from .randoms import current_generator
from .refs import store
from .wrapping import unwrap, wrap

__all__: list[str] = []


def _kept(made: Any) -> Any:
    """A new histogram wrapped, and kept in the current directory as ROOT keeps it."""
    wrapped = wrap(made)
    here = current_directory()
    old = here.FindObject(wrapped.GetName())
    if old is not None and old is not wrapped:
        here.Remove(old)
    if wrapped._add_directory[0]:
        here.Append(wrapped)
    return wrapped


def _range(first: int, last: int) -> Any:
    """ROOT's ``(first, last)`` bin arguments: all of them, flow included, when ``last < first``."""
    return None if last < first else (int(first), int(last))


def _cumulated(
    target: Any, source: Any, widths: list[int], cut: tuple[slice, ...], forward: bool
) -> None:
    """``source``'s running sum over the block ``cut`` - every axis at once - into ``target``."""
    shape = tuple(reversed(widths))
    block = np.asarray(source, dtype=np.float64).reshape(shape)[tuple(reversed(cut))]
    for axis in range(block.ndim):
        block = (
            np.flip(np.cumsum(np.flip(block, axis), axis), axis)
            if not forward
            else np.cumsum(block, axis)
        )
    view = np.asarray(target).reshape(shape)
    view[tuple(reversed(cut))] = block


class Operations:
    """Arithmetic, reshaping, fitting and random numbers for a ``TH1``."""

    _xrd: Any
    _core: Any
    _replace: Any
    GetName: Any
    SetName: Any

    # -- copies ---------------------------------------------------------------------------

    def Clone(self, newname: str = "") -> Any:
        """``Clone``: the same histogram, sharing nothing, kept in the current directory."""
        made = wrap(self._xrd.copy(str(newname) if newname else None))
        if made._add_directory[0]:
            current_directory().Append(made, True)
        return made

    def Copy(self, obj: Any) -> None:
        obj._replace(self._xrd.copy())

    def DrawCopy(self, option: str = "", name_postfix: str = "_copy") -> Any:
        """``DrawCopy``: draw a copy, named with ``_copy`` after, kept in no directory."""
        made = wrap(self._xrd.copy(f"{self.GetName()}{name_postfix}"))
        made.Draw(option)
        return made

    def DrawNormalized(self, option: str = "", norm: float = 1.0) -> Any:
        """``DrawNormalized``: draw a copy scaled so its bins add to ``norm``."""
        made = wrap(self._xrd.copy())
        total = self._xrd.sum()
        if total:
            made.Scale(norm / total)
        made.Draw(option)
        return made

    # -- arithmetic ---------------------------------------------------------------------

    def Add(self, h1: Any, *rest: Any) -> bool:
        """``Add(h1, c1)``, or ``Add(h1, h2, c1, c2)``: this becomes ``c1*h1 + c2*h2``."""
        if rest and hasattr(rest[0], "_xrd"):
            c1, c2 = (
                (float(rest[1]) if len(rest) > 1 else 1.0),
                (float(rest[2]) if len(rest) > 2 else 1.0),
            )
            self._xrd.reset()
            self._xrd.add(unwrap(h1), c1)
            self._xrd.add(unwrap(rest[0]), c2)
            return True
        if not hasattr(h1, "_xrd"):
            return self._add_function(h1, float(rest[0]) if rest else 1.0)
        self._xrd.add(unwrap(h1), float(rest[0]) if rest else 1.0)
        return True

    def _add_function(self, f1: Any, c1: float) -> bool:
        """``Add(TF1, c1)``: each bin gains ``c1`` times the function at its centre."""
        axis = self._xrd.axes[0]
        centres = axis.root_centers()
        cells = self._xrd._cells()
        cells[:] = cells + c1 * np.asarray(f1.Eval(centres) if hasattr(f1, "Eval") else f1(centres))
        self._core()["fTsumw"] = 0.0
        return True

    def Scale(self, c1: float = 1.0, option: str = "") -> None:
        """``Scale(c1[, "width"])``: in place; ``"width"`` divides by each bin's width too."""
        self._xrd.scale(float(c1), "width" in str(option).lower())

    def Multiply(
        self, h1: Any, h2: Any = None, c1: float = 1.0, c2: float = 1.0, option: str = ""
    ) -> bool:
        if h2 is None:
            self._xrd.multiply(unwrap(h1))
            return True
        product = unwrap(h1).copy()
        product.multiply(unwrap(h2))
        product.scale(float(c1) * float(c2))
        self._replace_contents(product)
        return True

    def Divide(
        self, h1: Any, h2: Any = None, c1: float = 1.0, c2: float = 1.0, option: str = ""
    ) -> bool:
        """``Divide(h1)``, or ``Divide(h1, h2, c1, c2, "B")``: this becomes ``c1*h1 / c2*h2``."""
        if h2 is None:
            self._xrd.divide(unwrap(h1))
            return True
        quotient = unwrap(h1).copy()
        quotient.scale(float(c1))
        below = unwrap(h2).copy()
        below.scale(float(c2))
        quotient.divide(below, "B" in str(option).upper())
        self._replace_contents(quotient)
        return True

    def _replace_contents(self, other: Any) -> None:
        """Take ``other``'s bins, squares and sums, keeping this histogram's name and look."""
        self._xrd.reset()
        self._xrd.add(other)

    def __add__(self, other: Any) -> Any:
        return wrap(self._xrd + unwrap(other))

    def __sub__(self, other: Any) -> Any:
        return wrap(self._xrd - unwrap(other))

    def __mul__(self, other: Any) -> Any:
        return wrap(self._xrd * unwrap(other))

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        return wrap(self._xrd / unwrap(other))

    def __iadd__(self, other: Any) -> Any:
        self._xrd += unwrap(other)
        return self

    def __isub__(self, other: Any) -> Any:
        self._xrd -= unwrap(other)
        return self

    def __imul__(self, other: Any) -> Any:
        self._xrd *= unwrap(other)
        return self

    def __itruediv__(self, other: Any) -> Any:
        self._xrd /= unwrap(other)
        return self

    # -- reshaping ----------------------------------------------------------------------------

    def Rebin(self, ngroup: int = 2, newname: str = "", xbins: Any = None) -> Any:
        """``Rebin(n)``: in place; ``Rebin(n, "name")``: a new one; ``xbins``: onto those edges."""
        group: Any = (
            list(np.asarray(xbins, dtype=np.float64)[: int(ngroup) + 1])
            if xbins is not None
            else int(ngroup)
        )
        made = self._xrd.rebin(group, name=str(newname) if newname else None)
        if newname:
            return _kept(made)
        self._replace(made)
        return self

    def RebinX(self, ngroup: int = 2, newname: str = "") -> Any:
        return self.Rebin(ngroup, newname)

    def Rebin2D(self, nxgroup: int = 2, nygroup: int = 2, newname: str = "") -> Any:
        made = self._xrd.rebin(int(nxgroup), int(nygroup), name=str(newname) if newname else None)
        if newname:
            return _kept(made)
        self._replace(made)
        return self

    def RebinY(self, ngroup: int = 2, newname: str = "") -> Any:
        return self.Rebin2D(1, ngroup, newname)

    def ProjectionX(
        self, name: str = "_px", firstybin: int = 0, lastybin: int = -1, option: str = ""
    ) -> Any:
        """``ProjectionX``: y summed over bins ``firstybin`` to ``lastybin``, all by default."""
        called = f"{self.GetName()}{name}" if name == "_px" else str(name)
        return _kept(self._xrd.projection_x(called, _range(firstybin, lastybin)))

    def ProjectionY(
        self, name: str = "_py", firstxbin: int = 0, lastxbin: int = -1, option: str = ""
    ) -> Any:
        called = f"{self.GetName()}{name}" if name == "_py" else str(name)
        return _kept(self._xrd.projection_y(called, _range(firstxbin, lastxbin)))

    def Project3D(self, option: str = "x") -> Any:
        """``Project3D("xy")``: the axes named kept - in ROOT's order, the first letter along y."""
        axes = "".join(letter for letter in str(option).lower() if letter in "xyz")
        ordered = axes[::-1] if len(axes) == 2 else axes
        return _kept(self._xrd.projection(ordered, f"{self.GetName()}_{axes}"))

    def ProfileX(
        self, name: str = "_pfx", firstybin: int = 1, lastybin: int = -1, option: str = ""
    ) -> Any:
        called = f"{self.GetName()}{name}" if name == "_pfx" else str(name)
        return _kept(self._xrd.profile_x(called, _range(firstybin, lastybin)))

    def ProfileY(
        self, name: str = "_pfy", firstxbin: int = 1, lastxbin: int = -1, option: str = ""
    ) -> Any:
        called = f"{self.GetName()}{name}" if name == "_pfy" else str(name)
        return _kept(self._xrd.profile_y(called, _range(firstxbin, lastxbin)))

    def GetCumulative(self, forward: bool = True, suffix: str = "_cumulative") -> Any:
        """``GetCumulative``: each bin in the range drawn, the sum of those before it - or after."""
        made = self._xrd.copy(f"{self.GetName()}{suffix}")
        made.reset()
        cut = tuple(slice(axis.GetFirst(), axis.GetLast() + 1) for axis in self._axes())
        _cumulated(made._cells(), self._xrd._bins, self._widths(), cut, forward)
        squares = self._xrd._sumw2()
        if squares is not None:
            _cumulated(made._ensure_sumw2(), squares, self._widths(), cut, forward)
        made._core["fEntries"] = float(np.prod([s.stop - s.start for s in cut]))
        made._core["fTsumw"] = 0.0
        return _kept(made)

    def Smooth(self, ntimes: int = 1, option: str = "") -> None:
        self._xrd.smooth(int(ntimes))

    def GetQuantiles(self, nprobSum: int, q: Any, probSum: Any = None) -> int:
        """``GetQuantiles(n, q, probSum)``: the quantiles at ``probSum``, into ``q``."""
        probabilities = (
            None if probSum is None else np.asarray(probSum, dtype=np.float64)[:nprobSum]
        )
        found = self._xrd.quantiles(probabilities)
        for at, value in enumerate(found[:nprobSum]):
            q[at] = float(value)
        return int(min(len(found), nprobSum))

    # -- comparing -----------------------------------------------------------------------

    def KolmogorovTest(self, h2: Any, option: str = "") -> float:
        return float(self._xrd.kolmogorov_test(unwrap(h2), str(option), rng=current_generator()))

    def Chi2Test(self, h2: Any, option: str = "UU", res: Any = None) -> float:
        """``Chi2Test``: the p-value - or with ``CHI2`` the chi-square - residuals into ``res``."""
        if res is not None:
            full = self._xrd.chi2_test_full(unwrap(h2), str(option))
            for at, value in enumerate(np.asarray(full.residuals).ravel()):
                res[at] = float(value)
        return float(self._xrd.chi2_test(unwrap(h2), str(option)))

    # -- fitting ------------------------------------------------------------------------------

    def Fit(
        self, f1: Any, option: str = "", goption: str = "", xxmin: float = 0.0, xxmax: float = 0.0
    ) -> Any:
        """``Fit(f, option, goption, xmin, xmax)``: a ``TFitResultPtr``, and ``f`` fitted."""
        from .fits import fit

        return fit(self, f1, option, goption, (xxmin, xxmax))

    def GetFunction(self, name: Any) -> Any:
        """``GetFunction``: a function hung on the histogram - a fit - by name, or ``None``."""
        found = [item for item in self._xrd.functions if getattr(item, "name", None) == str(name)]
        return wrap(found[0]) if found else None

    def GetListOfFunctions(self) -> Any:
        from .collections import TList

        made = TList()
        for item in self._xrd.functions:
            made.Add(wrap(item))
        return made

    # -- random numbers ------------------------------------------------------------------

    def FillRandom(self, fname: Any, ntimes: int = 5000, rng: Any = None) -> None:
        """``FillRandom``: ``ntimes`` draws from a function - by name or itself - or a histogram."""
        source = unwrap(fname)
        if isinstance(fname, str):
            from .troot import gROOT

            named = gROOT.GetListOfFunctions().FindObject(fname)
            source = unwrap(named) if named is not None else fname
        generator = unwrap(rng) if rng is not None else current_generator()
        self._xrd.fill_random(source, int(ntimes), rng=generator)

    def ComputeIntegral(self, onlyPositive: bool = False) -> float:
        """``ComputeIntegral``: the running sum of the bins, x fastest, kept normalised."""
        values = np.asarray(self._xrd.values(), dtype=np.float64).ravel(order="F")
        running = np.concatenate([[0.0], np.cumsum(values)])
        total = float(running[-1])
        self._integral = running / total if total else running
        return total

    def GetIntegral(self) -> np.ndarray[Any, Any]:
        self.ComputeIntegral()
        return self._integral

    def GetRandom(self, rng: Any = None) -> float:
        """``GetRandom``: a number drawn from the histogram, as ROOT draws it from its bins."""
        if self.ComputeIntegral(True) == 0:
            return 0.0
        generator = unwrap(rng) if rng is not None else current_generator()
        r1 = float(generator.rndm())
        return self._position(0, r1)

    def _position(self, at: int, r1: float) -> float:
        """Where in its bin a draw of ``r1`` lands, by the running sum, along axis ``at``."""
        integral = self._integral
        nbins = len(integral) - 1
        ibin = int(np.searchsorted(integral[:nbins], r1, side="right")) - 1
        axis = self.GetXaxis()  # type: ignore[attr-defined]
        x = float(axis.GetBinLowEdge(ibin + 1))
        if r1 > integral[ibin]:
            x += (
                axis.GetBinWidth(ibin + 1)
                * (r1 - integral[ibin])
                / (integral[ibin + 1] - integral[ibin])
            )
        return x

    def GetRandom2(self, x: Any = None, y: Any = None, rng: Any = None) -> tuple[float, float]:
        """``GetRandom2(x, y)``: a point drawn from a two-dimensional histogram."""
        if self.ComputeIntegral(True) == 0:
            store(x, 0.0)
            store(y, 0.0)
            return 0.0, 0.0
        generator = unwrap(rng) if rng is not None else current_generator()
        r1 = float(generator.rndm())
        integral, nx = self._integral, self._xrd.axes[0].nbins
        ibin = int(np.searchsorted(integral[: len(integral) - 1], r1, side="right")) - 1
        biny, binx = divmod(ibin, nx)
        xaxis, yaxis = self.GetXaxis(), self.GetYaxis()  # type: ignore[attr-defined]
        px = float(xaxis.GetBinLowEdge(binx + 1))
        if r1 > integral[ibin]:
            px += (
                xaxis.GetBinWidth(binx + 1)
                * (r1 - integral[ibin])
                / (integral[ibin + 1] - integral[ibin])
            )
        py = float(yaxis.GetBinLowEdge(biny + 1)) + yaxis.GetBinWidth(biny + 1) * float(
            generator.rndm()
        )
        store(x, px)
        store(y, py)
        return px, py
