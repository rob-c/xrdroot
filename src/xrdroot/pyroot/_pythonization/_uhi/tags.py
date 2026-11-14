"""``ROOT.uhi``'s tags: where a bin is, and what a slice's step does with the bins it takes.

ROOT's UHI numbers bins from zero, the underflow and overflow apart, and
finds one with a tag called with the histogram and the axis' position in it:
:func:`_underflow` and :func:`_overflow` give ROOT's own bin numbers for the
flow, :class:`_loc` the bin a coordinate falls in, ``len`` the bin past the
last. A slice's step is :class:`_rebin`, which merges bins, or :func:`_sum`,
which sums an axis away - the projection ROOT would make, or the integral
when no axis is left.
"""

from __future__ import annotations

from typing import Any

__all__: list[str] = []


def _get_axis(hist: Any, axis: int) -> Any:
    """The ``TAxis`` at that position: 0 is x, 1 is y, 2 is z."""
    return getattr(hist, f"Get{'XYZ'[axis]}axis")()


def _get_axis_len(hist: Any, axis: int, flow: bool = False) -> int:
    """How many bins the axis has, the two flow bins counted when asked."""
    return int(_get_axis(hist, axis).GetNbins()) + (2 if flow else 0)


def _underflow(hist: Any, axis: int) -> int:
    """``ROOT.uhi.underflow``: ROOT's number for the underflow, which is always 0."""
    return 0


def _overflow(hist: Any, axis: int) -> int:
    """``ROOT.uhi.overflow``: ROOT's number for the overflow, one past the last bin."""
    return int(_get_axis(hist, axis).GetNbins()) + 1


class _loc:
    """``ROOT.uhi.loc(x)``: the bin ``x`` falls in, moved by whole bins with ``+`` and ``-``.

    ``h[loc(b) + 1]`` is the bin above the one holding ``b``.
    """

    def __init__(self, value: float, offset: int = 0) -> None:
        self.value = value
        self.offset = offset

    def __add__(self, other: int) -> _loc:
        if not isinstance(other, int):
            raise TypeError(
                f"Unsupported type for addition: {type(other).__name__}. Expected an integer."
            )
        return _loc(self.value, self.offset + other)

    def __sub__(self, other: int) -> _loc:
        if not isinstance(other, int):
            raise TypeError(
                f"Unsupported type for substraction: {type(other).__name__}. Expected an integer."
            )
        return _loc(self.value, self.offset - other)

    def __call__(self, hist: Any, axis: int) -> int:
        return int(_get_axis(hist, axis).FindBin(self.value)) + self.offset

    def __repr__(self) -> str:
        return f"loc({self.value!r})" + (f" + {self.offset}" if self.offset else "")


class _rebin:
    """``ROOT.uhi.rebin(n)``: a slice's step merging every ``n`` bins into one.

    ``h[::rebin(2)]`` is ``h.Rebin(2)`` made as a new histogram of the same name.
    """

    def __init__(self, ngroup: int) -> None:
        self.ngroup = int(ngroup)

    def __repr__(self) -> str:
        return f"rebin({self.ngroup})"


def _rebinned(hist: Any, groups: list[int]) -> Any:
    """The histogram with each axis' bins merged by its group: ``Rebin``, ``Rebin2D`` or
    ``Rebin3D``, as a new histogram under the old one's name."""
    method = {1: "Rebin", 2: "Rebin2D", 3: "Rebin3D"}[hist.GetDimension()]
    return getattr(hist, method)(*groups, newname=hist.GetName())


#: The axes ``Project3D`` keeps once those named are summed away, in its option's order.
_PROJECT3D = {(0,): "zy", (1,): "zx", (2,): "yx", (0, 1): "z", (0, 2): "y", (1, 2): "x"}


def _sum(hist: Any, axis: Any, args: Any = None) -> Any:
    """``ROOT.uhi.sum``: the axes named summed away.

    Of a 1-D histogram that is its integral over the bins ``args`` gives; of
    a 2-D or 3-D one, the projection onto the axes left - or, every axis
    summed, the integral of the bins inside the flow.
    """
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    dim = hist.GetDimension()
    if axes == tuple(range(dim)):
        return hist.Integral(*(args or ()))
    if dim == 2 and axes in ((0,), (1,)):
        return hist.ProjectionY() if axes == (0,) else hist.ProjectionX()
    if dim == 3 and axes in _PROJECT3D:
        return hist.Project3D(_PROJECT3D[axes])
    raise ValueError(f"Invalid axis {axis} for {dim}D histogram")
