"""What ``Histo1D`` and its kin are told to book: ROOT's models, as tuples or as objects.

ROOT's ``Histo1D({"h", "title", 100, 0., 1.}, "x")`` books from a
``TH1DModel``: a name, a title and the binning, in the order the ``TH1D``
constructor takes them. The same tuple does here, in Python's brackets:

    >>> df.Histo1D(("h", "p_{T}", 100, 0.0, 200.0), "pt")      # doctest: +SKIP
    >>> df.Histo2D(("m", "", 50, -2.5, 2.5, [0, 10, 50]), "eta", "pt")   # doctest: +SKIP

Each axis is a number of bins and then either the low and high edge or every
edge, as a list; a profile's binning is followed by the range of values it
averages and its error option, as ``TProfile``'s constructor has them. A
tuple may leave out the name and title and start from the binning, and then
takes the column's name. A :class:`~xrdroot.Histogram` or
:class:`~xrdroot.Profile` already booked is a model too: its binning, kind
and title are used, and whatever it holds is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..hist import Histogram
from ..profile import Profile

__all__ = ["model_of", "is_model"]


def is_model(given: Any) -> bool:
    """Is this a model rather than a column's name?"""
    return isinstance(given, (tuple, Histogram))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _axis(parts: list[Any], at: int, what: str) -> tuple[Any, int]:
    """One axis from the parts of a model: ``(n, low, high)`` or ``(n, edges)``."""
    if at < len(parts) and isinstance(parts[at], (list, np.ndarray)):
        return list(parts[at]), at + 1  # edges on their own
    if at + 1 < len(parts) and isinstance(parts[at + 1], (list, tuple, np.ndarray)):
        return list(parts[at + 1]), at + 2
    if at + 2 < len(parts) and all(_number(part) for part in parts[at : at + 3]):
        return (int(parts[at]), float(parts[at + 1]), float(parts[at + 2])), at + 3
    raise ValueError(
        f"{what} takes, for each axis, a number of bins then its low and high edge, or a "
        f"number of bins then a list of every edge; {tuple(parts)!r} does not say that"
    )


def _named(parts: list[Any], column: str) -> tuple[str, str, list[Any]]:
    if parts and isinstance(parts[0], str):
        title = parts[1] if len(parts) > 1 and isinstance(parts[1], str) else ""
        rest = parts[2:] if len(parts) > 1 and isinstance(parts[1], str) else parts[1:]
        return parts[0], title, rest
    return column, column, parts


def _profile_extras(rest: list[Any], what: str) -> tuple[Any, str]:
    """What a profile's model has after its axes: the range averaged over, and the option."""
    option = rest.pop() if rest and isinstance(rest[-1], str) else ""
    if not rest:
        return None, option
    if len(rest) == 2 and all(_number(part) for part in rest):
        low, high = float(rest[0]), float(rest[1])
        return (None if low == high == 0 else (low, high)), option
    raise ValueError(
        f"{what}'s model ends in the low and high value it averages and its error option, "
        f"and {rest!r} is neither"
    )


def model_of(given: Any, dimensions: int, profile: bool, column: str, what: str) -> Any:
    """A booked, empty histogram or profile from a model, as ``what`` is told to book it."""
    if isinstance(given, Histogram):
        return _template(given, dimensions, profile, what)
    name, title, parts = _named(list(given), column)
    axes, at = [], 0
    for _ in range(dimensions):
        axis, at = _axis(parts, at, what)
        axes.append(axis)
    rest = parts[at:]
    if profile:
        value_range, option = _profile_extras(rest, what)
        return Profile.book(name, *axes, title=title, error_option=option, value_range=value_range)
    if rest:
        raise ValueError(f"{what}'s model has {rest!r} left over after its {dimensions} axes")
    return Histogram.book(name, *axes, title=title)


def _template(given: Histogram, dimensions: int, profile: bool, what: str) -> Any:
    if len(given.axes) != dimensions or isinstance(given, Profile) != profile:
        kind = "profile" if profile else "histogram"
        raise ValueError(
            f"{what} books a {kind} of {dimensions} axes, and was given {given.classname} "
            f"{given.name!r} as its model"
        )
    made = given.copy()
    made.reset()
    return made


def split_arguments(
    args: Sequence[Any], model: Any, count: int, what: str
) -> tuple[Any, list[str], str | None]:
    """A model, the columns filled, and a weight, from the ways ROOT's methods are called."""
    rest = list(args)
    if model is None and rest and is_model(rest[0]):
        model = rest.pop(0)
    if len(rest) not in (count, count + 1) or not all(isinstance(each, str) for each in rest):
        raise TypeError(
            f"{what} takes a model, then {count} column name{'' if count == 1 else 's'} "
            f"and optionally a weight's; it was given {tuple(args)!r}"
        )
    weight = rest[count] if len(rest) > count else None
    return model, rest[:count], weight
