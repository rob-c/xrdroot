"""Profiles: histograms whose bins hold the mean of something, not a count.

A ``TProfile`` is a ``TH1D`` underneath, and fills the same three arrays one
does - only what goes in them means something else. The bin contents are the
sum of ``w*y`` over every fill that landed in the bin, the squared weights
are the sum of ``w*y*y``, and one more array, ``fBinEntries``, holds the sum
of the weights themselves. The mean is the first divided by the last, and the
error on it is whichever of four things ROOT was told to call the error.

None of that is stored as a mean, so a profile read as a plain histogram
would plot sums of ``y`` and be wrong without looking wrong. This reads it as
what it is: ``values()`` are the means, ``kind`` is ``MEAN`` in the words of
the plotting protocol, and ``to_hist()`` gives a ``hist.Hist`` with ``Mean``
storage, which keeps the same sums the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import UnsupportedFeatureError
from .hist import Histogram

__all__ = ["PROFILES", "ERROR_MODES", "Profile"]

#: The profile classes, against how many axes each is binned along.
PROFILES = {"TProfile": 1, "TProfile2D": 2, "TProfile3D": 3}

#: ROOT's ``EErrorType``, by the letter its ``SetErrorOption`` takes: the
#: error on the mean, the spread, the spread of integers, and the error of
#: a fill weighted by one over its own variance.
ERROR_MODES = {0: "", 1: "s", 2: "i", 3: "g"}


def _divided(top: np.ndarray[Any, Any], bottom: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """``top / bottom`` wherever ``bottom`` is not zero, and zero where it is."""
    return np.divide(top, bottom, out=np.zeros_like(top), where=bottom != 0)


class Profile(Histogram):
    """A ``TProfile``, ``TProfile2D`` or ``TProfile3D``: the mean of ``y`` per bin.

        >>> profile = f["pz_vs_px"]                  # doctest: +SKIP
        >>> profile.values()[:3], profile.errors()[:3]

    Everything :class:`~.hist.Histogram` does, it does, shaped and indexed
    the same way; what the numbers are is what differs. :meth:`values` is
    the mean in each bin, :meth:`bin_entries` the sum of the weights that
    went into it, :meth:`counts` the effective number of entries, and
    :meth:`errors` whatever the profile was told to call its error -
    :attr:`error_mode` says which, and each method takes another.
    """

    __slots__ = ("_weights", "_weight_squares", "error_mode")

    #: The bins hold means, which is what the plotting protocol calls ``MEAN``.
    kind = "MEAN"

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        super().__init__(classname, members)
        cells = len(self._bins)
        #: The sum of the weights of the fills in each bin.
        self._weights = np.asarray(members.get("fBinEntries", ()), dtype=np.float64)
        if len(self._weights) < cells:
            raise UnsupportedFeatureError(
                f"a {classname} of {cells} bins keeps the weights of only "
                f"{len(self._weights)} of them, so what the mean of each bin is cannot "
                f"be worked out"
            )
        squares = np.asarray(members.get("fBinSumw2", ()), dtype=np.float64)
        #: The sum of the squared weights per bin, which only a profile filled
        #: with weights keeps; ``None`` for one filled without them.
        self._weight_squares = squares if len(squares) == cells else None
        #: How the error on a bin is worked out unless a method is told
        #: otherwise: ``""`` for the error on the mean, ``"s"`` for the spread,
        #: ``"i"`` for the spread of integers and ``"g"`` for Gaussian weights.
        self.error_mode = ERROR_MODES.get(int(members.get("fErrorMode", 0)), "")

    def _dimensions(self) -> int:
        return PROFILES[self.classname]

    @property
    def weighted(self) -> bool:
        """Was it filled with weights, so that it keeps their squares as well?"""
        return self._weight_squares is not None

    def sums(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The sum of ``w*y`` in each bin, which is what the file stores as contents."""
        return self._shaped(np.asarray(self._bins, dtype=np.float64), flow)

    def bin_entries(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The sum of the weights in each bin - ``GetBinEntries`` in ROOT."""
        return self._shaped(self._weights, flow)

    def values(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The mean in each bin, and zero in a bin nothing was filled into."""
        return _divided(self.sums(flow), self.bin_entries(flow))

    def counts(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The effective number of entries: ``(sum w)**2 / sum w**2`` per bin.

        Without weights that is the number of fills, which is what
        :meth:`bin_entries` already holds.
        """
        weights = self.bin_entries(flow)
        if self._weight_squares is None:
            return weights
        return _divided(weights * weights, self._shaped(self._weight_squares, flow))

    def errors(self, flow: bool = False, error_mode: str | None = None) -> np.ndarray[Any, Any]:
        """The error on each bin's mean, the way ``TProfile::GetBinError`` gives it.

        ``error_mode`` is one of ROOT's letters and :attr:`error_mode` unless
        said otherwise: ``""`` is the standard error on the mean, the spread
        divided by the root of :meth:`counts`; ``"s"`` is the spread itself;
        ``"i"`` is the error on the mean with a floor of ``1/sqrt(12)`` for
        a bin of identical integers; and ``"g"`` is one over the root of the
        sum of the weights, for fills weighted by one over their variance.
        An empty bin has an error of zero whichever is asked for.
        """
        mode = self.error_mode if error_mode is None else error_mode.lower()
        if mode not in ERROR_MODES.values():
            raise ValueError(
                f"error_mode={error_mode!r} is not one of ROOT's: '' for the error on the "
                f"mean, 's' for the spread, 'i' for integers and 'g' for Gaussian weights"
            )
        weights = self.bin_entries(flow)
        if mode == "g":
            return _divided(np.ones_like(weights), np.sqrt(np.abs(weights)))
        spread = self.spread(flow)
        if mode == "s":
            return spread
        if mode == "i":
            spread = np.where(spread != 0, spread, np.sqrt(1 / 12))
        return np.where(weights != 0, _divided(spread, np.sqrt(self.counts(flow))), 0.0)

    def spread(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The standard deviation of what was filled into each bin, zero in an empty one."""
        weights = self.bin_entries(flow)
        squares = _divided(self._shaped(self._sumw2(), flow), weights)
        return np.where(weights != 0, np.sqrt(np.abs(squares - self.values(flow) ** 2)), 0.0)

    def variances(self, flow: bool = False, error_mode: str | None = None) -> np.ndarray[Any, Any]:
        """The square of :meth:`errors`, which is what the plotting protocol asks for."""
        return self.errors(flow, error_mode) ** 2

    def _sumw2(self) -> np.ndarray[Any, Any]:
        """The sum of ``w*y*y`` per bin, which a profile always keeps."""
        squares = np.asarray(self._core["fSumw2"], dtype=np.float64)
        if len(squares) < len(self._bins):
            raise UnsupportedFeatureError(
                f"this {self.classname} keeps no sum of the squares of what was filled, so "
                f"it has no spread to give an error from"
            )
        return squares

    def sum(self, flow: bool = False) -> float:
        raise UnsupportedFeatureError(
            "the bins of a profile are means, and a sum of means is not a thing; "
            "bin_entries() has how much went into each bin, and sums() what it added up to"
        )

    def density(self, flow: bool = False) -> np.ndarray[Any, Any]:
        raise UnsupportedFeatureError(
            "the bins of a profile are means rather than counts, so they have no density"
        )

    def to_hist(self) -> Any:
        """The same profile as a :class:`hist.Hist` of ``Mean`` storage, flow and all.

        A weighted profile becomes ``WeightedMean`` storage instead. Either
        keeps what ROOT keeps - the weight in each bin, the mean, and the sum
        of squared distances from it - so the means come across exactly.
        """
        try:
            import hist
        except ImportError:
            raise UnsupportedFeatureError(
                "turning this into a hist.Hist needs the hist package: pip install hist "
                "- or use .values() and .errors(), which need nothing more"
            ) from None
        made = [
            hist.axis.Variable(axis.edges(), name=axis.name, label=axis.title) for axis in self.axes
        ]
        weighted = self._weight_squares is not None
        storage = hist.storage.WeightedMean() if weighted else hist.storage.Mean()
        out = hist.Hist(*made, storage=storage, name=self.name, label=self.title)
        view = out.view(flow=True)
        weights, means = self.bin_entries(True), self.values(True)
        deltas = np.maximum(self._shaped(self._sumw2(), True) - weights * means**2, 0.0)
        if self._weight_squares is not None:
            view["sum_of_weights"] = weights
            view["sum_of_weights_squared"] = self._shaped(self._weight_squares, True)
            view["value"] = means
            view["_sum_of_weighted_deltas_squared"] = deltas
        else:
            view["count"] = weights
            view["value"] = means
            view["_sum_of_deltas_squared"] = deltas
        return out
