"""``TProfile``, ``TProfile2D``, ``TProfile3D``: the mean of a value in each bin.

A profile is a histogram whose bin content is a mean, so ``GetBinContent``
is ``sum(w*y) / sum(w)`` and ``GetBinError`` the error of that mean - or,
after ``SetErrorOption("s")``, the spread - as :class:`xrdroot.Profile`
computes them. The constructors are ROOT's: the axes, then optionally the
range of values averaged, then the error option.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np

from ...booking import ERROR_OPTIONS
from ...filling import one_bin
from ...profile import ERROR_MODES
from .histcore import axis_specs
from .hists import TH1, TH2, TH3
from .wrapping import adopt, register, wrap

__all__ = ["TProfile", "TProfile2D", "TProfile3D"]


def _value_range(rest: tuple[Any, ...]) -> tuple[Any, str]:
    """The value range and error option after the axes: ``(ylow, yup[, opt])`` or ``(opt)``."""
    numbers = [value for value in rest if not isinstance(value, str)]
    option = next((value for value in rest if isinstance(value, str)), "")
    found = (float(numbers[0]), float(numbers[1])) if len(numbers) >= 2 else None
    return found, option


def _fill_target(profile: Any) -> tuple[Any, ...]:
    """What a loop of fills adds to, read out of the profile's members once.

    A profile filled an entry at a time - ROOT's own way, and every tutorial's - is
    filled a million times over, and finding each of these in the members for every
    entry costs about twice what the entry's arithmetic does. The layout is ``axes,
    widths, nbins, members, core, home, key, cells, squares, weights, the squared
    weights as the members keep them, the squared weights as a fill uses them, the
    moments, the value range's two member names, and the identities
    :func:`_still_ours` compares``. Nothing here is a copy: every array is the member
    itself, so a fill through the target is a fill of the profile.
    """
    letter = profile._value_letter()
    homes = profile._moment_homes()
    named = (f"fTsumw{letter}", f"fTsumw{letter}2")
    terms = profile._moment_terms() + tuple((homes[name], name) for name in named)
    kept = profile.members.get("fBinSumw2")
    return (profile.axes, profile._widths, tuple(axis.nbins for axis in profile.axes),
            profile.members, profile._core, profile._home, profile._key,
            profile._cells(), profile._value_squares(), profile._bin_weights(), kept,
            profile._bin_sumw2(), terms, f"f{letter.upper()}min", f"f{letter.upper()}max",
            (id(profile.axes), id(profile._home.get(profile._key)),
             id(profile._core.get("fSumw2")), id(profile.members.get("fBinEntries")),
             id(kept)))  # fmt: skip


def _still_ours(target: tuple[Any, ...], profile: Any) -> bool:
    """Does the profile still keep the very arrays and moments this target was read from?

    A fill written into an array the profile has let go of is a fill lost, so a profile
    that replaced one - reading replaced the bins of a file written in another byte
    order, and ``Sumw2`` and the first weight that is not one start or stop keeping the
    squared weights - is met by a target made afresh. The identities are in one tuple,
    compared in one go, which costs a fraction of the fill it guards.
    """
    home, key = target[5], target[6]
    members, core = target[3], target[4]
    now = (id(profile.axes), id(home.get(key)), id(core.get("fSumw2")),
           id(members.get("fBinEntries")), id(members.get("fBinSumw2")))  # fmt: skip
    return bool(now == target[15])


def _fast_fill(profile: Any, target: tuple[Any, ...], coordinates: Sequence[float],
               value: float, weight: float) -> int:  # fmt: skip
    """``TProfile::Fill`` of one entry, added where :func:`~.filling.fill_profile_one` adds it.

    The same sums in the same order - the bin's value, its square, its squared weights,
    its weights, then the moments, only what fell inside every axis - so a profile
    filled through here holds the doubles one filled through the engine holds.
    """
    found, cell = _fast_place(target, coordinates)
    members, core, nbins = target[3], target[4], target[2]
    cells, squares, weights, sq2, terms = target[7], target[8], target[9], target[11], target[12]
    low, high = float(members.get(target[13], 0.0)), float(members.get(target[14], 0.0))
    if not (low == high or low <= value <= high):
        return cell  # outside the range of values averaged: counted nowhere, as ROOT does
    counts = weight * value
    squared = counts * value
    core["fEntries"] = float(core["fEntries"]) + 1
    cells[cell] += cells.dtype.type(counts)  # a float bin adds in its own precision
    squares[cell] += squared
    _fast_squares(profile, sq2, cell, weight)
    weights[cell] += weight
    if all(1 <= found[at] <= nbins[at] for at in range(len(nbins))):
        _fast_moments(terms, weight, coordinates, counts, squared)
    return cell


def _fast_place(target: tuple[Any, ...], coordinates: Sequence[float]) -> tuple[list[int], int]:
    """Which bin of each axis the entry falls in, and ROOT's global bin number of them."""
    axes, widths = target[0], target[1]
    found = [one_bin(axis, x) for axis, x in zip(axes, coordinates, strict=False)]
    cell = found[-1]
    for at in range(len(found) - 2, -1, -1):
        cell = cell * widths[at] + found[at]
    return found, cell


def _fast_squares(profile: Any, sq2: Any, cell: int, weight: float) -> None:
    """The sum of squared weights one bin takes, which a weight other than one starts."""
    if sq2 is None and weight != 1.0:
        sq2 = profile._ensure_sumw2()  # the first weight that is not one starts them
    if sq2 is not None:
        sq2[cell] += weight * weight


def _fast_moments(terms: Sequence[tuple[dict[str, Any], str]], weight: float,
                  coordinates: Sequence[float], counts: float, squared: float) -> None:  # fmt: skip
    """:func:`~.filling.one_terms` added to the moments each belongs to, in its order.

    ``w*x`` is formed before it meets the second ``x``, as ``one_terms`` forms it, so
    each sum takes the double ROOT's own loop takes.
    """
    values = [weight, weight * weight]
    for x in coordinates:
        values += [weight * x, weight * x * x]
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            values.append(weight * coordinates[first] * coordinates[second])
    values += [counts, squared]
    for (home, name), value in zip(terms, values, strict=True):
        home[name] = home[name] + value


class _Profiled:
    """What the three profiles share over their histogram's methods."""

    _xrd: Any
    DIM: int

    def _booked(self, args: tuple[Any, ...]) -> Any:
        """The profile ROOT's constructor arguments describe, drawn in ``gStyle``'s attributes.

        A ``TProfile`` is a ``TH1`` and is built by ``TH1::Build`` as one is, so its
        line colour is ``gStyle``'s ``HistLineColor`` - 602 - and not the ``1`` a
        ``TAttLine`` starts with. That colour is in the bytes a file keeps.
        """
        from ...profile import Profile
        from .histcore import _in_style

        if not args:
            return _in_style(Profile.book("", *[(1, 0.0, 1.0)] * self.DIM))
        specs, rest = axis_specs(args[2:], self.DIM)
        values, option = _value_range(rest)
        made = Profile.book(str(args[0]), *specs, title=str(args[1]),
                            error_option=option.lower(), value_range=values)  # fmt: skip
        return _in_style(made)

    def ClassName(self) -> str:
        return str(self._xrd.classname)

    def Fill(self, *args: Any) -> int:
        """``Fill(x[, y[, z]], value[, w])``: a value averaged in the bin of the coordinates."""
        count = self.DIM + 1
        given = [float(value) for value in args[:count]]
        weight = float(args[count]) if len(args) > count else 1.0
        target = self.__dict__.get("_fill_target")
        if target is None or not _still_ours(target, self._xrd):
            target = self.__dict__["_fill_target"] = _fill_target(self._xrd)
        return _fast_fill(self._xrd, target, given[:-1], given[-1], weight)

    def FillN(self, ntimes: int, x: Any, y: Any, w: Any = None, stride: int = 1) -> None:
        count = int(ntimes)
        weights = None if w is None else np.asarray(w, dtype=np.float64)[:count]
        xs = np.asarray(x, dtype=np.float64)[:count]
        self._xrd.fill(xs, np.asarray(y, dtype=np.float64)[:count], weight=weights)

    def _contents(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.values(flow=True)).ravel(order="F")

    def _one_content(self, bin: int) -> float:
        """A profile's bin is a mean, which is an array of its own: one bin's content comes
        out of it as it always did."""
        cells = self._contents()
        return float(cells[bin]) if 0 <= bin < len(cells) else 0.0

    def _one_error(self, bin: int) -> float:
        """A profile's error is its spread, which is an array of its own: one bin's comes
        out of it as it always did."""
        errors = self._errors()
        return float(errors[bin]) if 0 <= bin < len(errors) else 0.0

    def _errors(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.errors(flow=True)).ravel(order="F")

    def GetBinEntries(self, bin: int) -> float:
        return float(np.asarray(self._xrd.members["fBinEntries"])[int(bin)])

    def GetBinEffectiveEntries(self, bin: int) -> float:
        return float(np.asarray(self._xrd.counts(flow=True)).ravel(order="F")[int(bin)])

    def GetSumOfWeights(self) -> float:
        """``GetSumOfWeights``: of a profile, the sum of its bins' means, as ROOT adds them up."""
        return float(np.sum(self._xrd.values()))

    def SetErrorOption(self, option: str = "") -> None:
        """``SetErrorOption``: ``""`` the mean's error, ``"s"`` the spread, ``"i"``, ``"g"``."""
        chosen = str(option).lower()
        self._xrd.members["fErrorMode"] = ERROR_OPTIONS[chosen]
        self._xrd.error_mode = ERROR_MODES[ERROR_OPTIONS[chosen]]

    def GetErrorOption(self) -> str:
        return str(self._xrd.error_mode)

    def ProjectionX(self, name: str = "_px", option: str = "e") -> Any:
        """``ProjectionX``: a ``TH1D`` of the means - with ``"b"``, the entries - as ROOT's."""
        from ...hist import Histogram

        axis = self._xrd.axes[0]
        called = f"{self.GetName()}{name}" if name == "_px" else str(name)  # type: ignore[attr-defined]
        chosen = str(option).lower()
        values = self._xrd.bin_entries() if "b" in chosen else self._xrd.values()
        errors = None if "b" in chosen else self._xrd.errors()
        made = Histogram.new(called, axis.edges(), values, title=self.GetTitle(), errors=errors)  # type: ignore[attr-defined]
        made.members["TH1"]["fEntries"] = float(self.GetEntries())  # type: ignore[attr-defined]
        return wrap(made)

    def ProjectionXY(self, name: Any = "_pxy", option: Any = "") -> Any:
        """``ProjectionXY``: a ``TH2D`` whose bin is the profile's mean there.

        ROOT copies both axes whole - their binning, their ranges and their bin labels - and
        puts the profile's own content, or with ``"b"`` its entries, in the same bin.
        """
        from ...hist import Histogram

        called = f"{self.GetName()}{name}" if str(name) == "_pxy" else str(name)  # type: ignore[attr-defined]
        chosen = str(option).lower()
        axes = [axis.edges() for axis in self._xrd.axes]
        values = self._xrd.bin_entries() if "b" in chosen else self._xrd.values()
        errors = None if "b" in chosen else self._xrd.errors()
        made = Histogram.new(called, axes, values, title=self.GetTitle(), errors=errors)  # type: ignore[attr-defined]
        core = made.members["TH2"]["TH1"]
        for letter, axis in zip("XY", self._xrd.axes, strict=False):
            if axis.labels:
                core[f"f{letter}axis"]["_labels"] = {str(b): v for b, v in axis.labels.items()}
        core["fEntries"] = float(self.GetEntries())  # type: ignore[attr-defined]
        return wrap(made)

    def GetYmin(self) -> float:
        return float(self._xrd.members.get("fYmin", 0.0))

    def GetYmax(self) -> float:
        return float(self._xrd.members.get("fYmax", 0.0))


class TProfile(_Profiled, TH1):  # type: ignore[misc]  # ProjectionX takes a profile's arguments
    """``TProfile``: the mean of y in bins of x."""

    CLASS_TITLE = "Profile histogram class"
    DIM = 1
    KIND = "D"


class TProfile2D(_Profiled, TH2):  # type: ignore[misc]  # ProjectionX takes a profile's arguments
    """``TProfile2D``: the mean of z in bins of x and y."""

    CLASS_TITLE = "Profile2D histogram class"
    DIM = 2
    KIND = "D"

    def GetNumberOfBins(self) -> float:
        """``GetNumberOfBins``: how many cells ``fBinEntries`` holds, the ends with them.

        A ``TProfile2D`` answers this one as ``fBinEntries.GetSize()`` - a ``Double_t``,
        ``(nx + 2) * (ny + 2)`` - and not as the ``GetNbinsX`` a ``TH2`` answers, so a
        loop over the bins it counts runs over the flow bins as well.
        """
        return float(np.asarray(self._xrd.members["fBinEntries"]).size)


class TProfile3D(_Profiled, TH3):  # type: ignore[misc]  # ProjectionX takes a profile's arguments
    """``TProfile3D``: the mean of t in bins of x, y and z."""

    CLASS_TITLE = "Profile3D histogram class"
    DIM = 3
    KIND = "D"


for _cls in (TProfile, TProfile2D, TProfile3D):
    register(_cls.__name__, factory=partial(adopt, _cls))
