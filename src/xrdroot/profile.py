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

A profile is booked and filled the way ROOT's is, too - :meth:`Profile.book`
and :meth:`Profile.fill` - with the same bookkeeping, and merges with others
as ``hadd`` merges them. The rest of a histogram's arithmetic is refused: on
means it is a different arithmetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import arithmetic, filling
from .booking import profile_members
from .errors import UnsupportedFeatureError
from .hist import Histogram
from .moments import PROFILE_NAMES

__all__ = ["PROFILES", "ERROR_MODES", "Profile"]

#: The profile classes, against how many axes each is binned along.
PROFILES = {"TProfile": 1, "TProfile2D": 2, "TProfile3D": 3}

#: ROOT's ``EErrorType``, by the letter its ``SetErrorOption`` takes: the
#: error on the mean, the spread, the spread of integers, and the error of
#: a fill weighted by one over its own variance.
ERROR_MODES = {0: "", 1: "s", 2: "i", 3: "g"}

#: The letter of what each profile averages, which names its moments and range.
VALUES = {1: "y", 2: "z", 3: "t"}


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

    __slots__ = ("error_mode",)

    #: The bins hold means, which is what the plotting protocol calls ``MEAN``.
    kind = "MEAN"

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        super().__init__(classname, members)
        cells = len(self._bins)
        weights = members.get("fBinEntries", ())
        if len(weights) < cells:
            raise UnsupportedFeatureError(
                f"a {classname} of {cells} bins keeps the weights of only "
                f"{len(weights)} of them, so what the mean of each bin is cannot "
                f"be worked out"
            )
        #: How the error on a bin is worked out unless a method is told
        #: otherwise: ``""`` for the error on the mean, ``"s"`` for the spread,
        #: ``"i"`` for the spread of integers and ``"g"`` for Gaussian weights.
        self.error_mode = ERROR_MODES.get(int(members.get("fErrorMode", 0)), "")

    def _dimensions(self) -> int:
        return PROFILES[self.classname]

    @classmethod
    def book(  # type: ignore[override]  # a profile's storage is always doubles
        cls,
        name: str,
        *axes: Any,
        title: str = "",
        error_option: str = "",
        value_range: Any = None,
        labels: Any = None,
    ) -> Profile:
        """An empty profile of one, two or three axes, as ROOT's constructors make one.

            >>> p = Profile.book("p", (100, -4, 4))               # TProfile("p", "", 100, -4, 4)
            >>> p = Profile.book("p", (100, -4, 4), value_range=(0, 20), error_option="s")

        The axes are given as :meth:`Histogram.book` takes them.
        ``error_option`` is what ``SetErrorOption`` takes - ``""``, ``"s"``,
        ``"i"`` or ``"g"`` - and ``value_range`` the low and high limit of
        what is averaged, outside which a fill is dropped, as ROOT's
        ``TProfile(..., ylow, yup)`` drops it.
        """
        classname, members = profile_members(name, axes, title, error_option, value_range, labels)
        return cls(classname, members)

    def fill(
        self, x: Any, y: Any = None, z: Any = None, t: Any = None, *, weight: Any = None
    ) -> None:
        """``Fill``: the coordinates of each entry, then the value averaged.

            >>> p.fill(px, pz)                     # a TProfile: x, then the value  # doctest: +SKIP
            >>> p2.fill(px, py, pz, weight=w)      # a TProfile2D                  # doctest: +SKIP

        As ROOT keeps it: the contents take ``w*value``, the squares
        ``w*value*value`` and the bin entries ``w``; the squares of the
        weights start being kept at the first weight that is not one, and a
        value outside the booked range, or a NaN when there is one, is not
        filled at all.
        """
        given = [value for value in (x, y, z, t) if value is not None]
        if len(given) != len(self.axes) + 1:
            raise ValueError(
                f"{self.name!r} is binned along {len(self.axes)} axes, and filling it takes a "
                f"coordinate for each and then the value averaged: {len(self.axes) + 1} "
                f"arrays or numbers, not {len(given)}"
            )
        coordinates, weights = filling.arrays(given, weight)
        filling.fill_profile(self, coordinates[:-1], coordinates[-1], weights)

    # -- the state a profile keeps beyond a histogram's ------------------------

    def _bin_weights(self) -> np.ndarray[Any, Any]:
        """The sum of the weights per bin, as an array to change in place."""
        found = self._writable(self.members, "fBinEntries")
        assert found is not None  # checked when the profile was made
        return found

    def _bin_sumw2(self) -> np.ndarray[Any, Any] | None:
        """The sum of squared weights per bin, which only a weighted profile keeps."""
        return self._writable(self.members, "fBinSumw2")

    def _ensure_sumw2(self) -> np.ndarray[Any, Any]:
        """``TProfile::Sumw2``: start keeping the squared weights, from the weights so far."""
        squares = self._bin_sumw2()
        if squares is None:
            squares = self._bin_weights().copy()
            self.members["fBinSumw2"] = squares
        return squares

    def _drop_sumw2(self) -> None:
        """``TProfile::Sumw2(false)``: forget the squared weights, never the squared values."""
        self.members["fBinSumw2"] = np.zeros(0)

    def _value_squares(self) -> np.ndarray[Any, Any]:
        """The sum of ``w*y*y`` per bin, which a profile always keeps."""
        squares = self._sumw2()
        if squares is None:
            raise UnsupportedFeatureError(
                f"this {self.classname} keeps no sum of the squares of what was filled, so "
                f"it has no spread to give an error from"
            )
        return squares

    def _value_letter(self) -> str:
        return VALUES[len(self.axes)]

    def _value_range(self) -> tuple[float, float]:
        """The range what is averaged must fall in, or two equal numbers for none."""
        bound = self._value_letter().upper()
        return float(self.members.get(f"f{bound}min", 0.0)), float(
            self.members.get(f"f{bound}max", 0.0)
        )

    def _moment_names(self) -> tuple[str, ...]:
        return PROFILE_NAMES[len(self.axes)]

    def _from_bins(self, total_weight: float) -> bool:
        return total_weight == 0

    def _moment_axes(self) -> int:
        return len(self.axes) + 1

    def _bin_terms(self) -> dict[str, np.ndarray[Any, Any]]:
        """``TProfile::GetStats`` from the bins: the weights, and the sums they average."""
        weights = self._flat_inner(self.bin_entries(True))
        squares = self._bin_sumw2()
        terms = dict(filling.axis_terms(weights, self._centre_grid()))
        if squares is not None:
            terms["fTsumw2"] = self._flat_inner(self._shaped(squares, True))
        else:
            terms["fTsumw2"] = weights
        letter = self._value_letter()
        terms[f"fTsumw{letter}"] = self._flat_inner(self.sums(True))
        terms[f"fTsumw{letter}2"] = self._flat_inner(self._shaped(self._value_squares(), True))
        return terms

    def _per_cell(self) -> list[np.ndarray[Any, Any]]:
        squares = self._bin_sumw2()
        extra = [] if squares is None else [squares]
        return [self._value_squares(), self._bin_weights(), *extra]

    def _merge_cells(self, other: Histogram) -> None:
        """What ``TProfile::Merge`` does to the bins for each profile merged in."""
        assert isinstance(other, Profile)  # a merge checks the kinds match
        theirs = other._bin_sumw2()
        if theirs is not None:
            self._ensure_sumw2()
        self._cells()[:] += other._bins
        self._value_squares()[:] += other._value_squares()
        squares = self._bin_sumw2()
        if squares is not None:
            squares += theirs if theirs is not None else other._bin_weights()
        self._bin_weights()[:] += other._bin_weights()

    @classmethod
    def merge(cls, histograms: Any) -> Profile:
        """``TProfile::Merge``, as ``hadd`` does it: every sum each profile keeps, added up."""
        made = arithmetic.merge(histograms)
        assert isinstance(made, Profile)
        return made

    # -- what the bins mean ------------------------------------------------------

    @property
    def weighted(self) -> bool:
        """Was it filled with weights, so that it keeps their squares as well?"""
        return self._bin_sumw2() is not None

    def sums(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The sum of ``w*y`` in each bin, which is what the file stores as contents."""
        return self._shaped(np.asarray(self._bins, dtype=np.float64), flow)

    def bin_entries(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The sum of the weights in each bin - ``GetBinEntries`` in ROOT."""
        return self._shaped(np.asarray(self.members["fBinEntries"], dtype=np.float64), flow)

    def values(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The mean in each bin, and zero in a bin nothing was filled into."""
        return _divided(self.sums(flow), self.bin_entries(flow))

    def counts(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The effective number of entries: ``(sum w)**2 / sum w**2`` per bin.

        Without weights that is the number of fills, which is what
        :meth:`bin_entries` already holds.
        """
        weights = self.bin_entries(flow)
        squares = self._bin_sumw2()
        if squares is None:
            return weights
        return _divided(weights * weights, self._shaped(squares, flow))

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
        squares = _divided(self._shaped(self._value_squares(), flow), weights)
        return np.where(weights != 0, np.sqrt(np.abs(squares - self.values(flow) ** 2)), 0.0)

    def variances(self, flow: bool = False, error_mode: str | None = None) -> np.ndarray[Any, Any]:
        """The square of :meth:`errors`, which is what the plotting protocol asks for."""
        return self.errors(flow, error_mode) ** 2

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
        squares = self._bin_sumw2()
        storage = hist.storage.WeightedMean() if squares is not None else hist.storage.Mean()
        out = hist.Hist(*made, storage=storage, name=self.name, label=self.title)
        view = out.view(flow=True)
        weights, means = self.bin_entries(True), self.values(True)
        deltas = np.maximum(self._shaped(self._value_squares(), True) - weights * means**2, 0.0)
        if squares is not None:
            view["sum_of_weights"] = weights
            view["sum_of_weights_squared"] = self._shaped(squares, True)
            view["value"] = means
            view["_sum_of_weighted_deltas_squared"] = deltas
        else:
            view["count"] = weights
            view["value"] = means
            view["_sum_of_deltas_squared"] = deltas
        return out
