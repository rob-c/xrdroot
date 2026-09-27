"""``createHistogram``: a dataset, or a function, as a ROOT histogram of one, two or three
variables.

``data.createHistogram("h", x, Binning(20), YVar(y, Binning(10)))`` - or
``data.createHistogram("x,y", Binning(20), Binning(20))`` - fills an
:class:`xrdroot.Histogram` with the events' weights; ``pdf.createHistogram``
fills one with the density normalised over the histogram's variables,
times each bin's volume, as ``RooAbsReal::fillHistogram`` scales it. What
the histogram is handed back as is the kit's: :func:`set_wrapper` installs
the pyroot ``TH1`` that stands for it.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from ..hist import Histogram
from .cmdargs import RooCmdArg, commands
from .collections import as_list

__all__ = ["data_histogram", "function_histogram", "set_wrapper"]

#: What a made histogram is handed back as: the pyroot ``TH1`` when the kit has one.
WRAP: list[Callable[[Any], Any]] = [lambda made: made]


def set_wrapper(fn: Callable[[Any], Any]) -> None:
    WRAP[0] = fn


def _binning(var: Any, command: Any) -> np.ndarray[Any, Any]:
    """The edges a ``Binning(...)`` - or the variable's own binning - gives."""
    if command is None:
        return var.getBinning().array()
    first = command.value(0)
    if hasattr(first, "array"):
        return first.array()
    if isinstance(first, str):
        return var.getBinning(first).array()
    low, high = (
        (command.value(1), command.value(2))
        if command.value(2) is not None
        else (var.getMin(), var.getMax())
    )
    return np.linspace(float(low), float(high), int(first) + 1)


def _axes(
    first: Any, args: tuple[Any, ...], kwargs: dict[str, Any], known: Any
) -> list[tuple[Any, Any]]:
    """Each variable and its binning command: ``x, Binning, YVar(y, Binning)``, or ``"x,y"`` and
    bins."""
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    binnings = options.every("Binning")
    if isinstance(first, str):
        names = [one for one in first.split(",") if one]
        padded = binnings + [None] * len(names)
        return [(known(name), padded[i]) for i, name in enumerate(names)]
    return [(first, binnings[0] if binnings else None), *_extra_axes(options)]


def _extra_axes(options: Any) -> list[tuple[Any, Any]]:
    """``YVar(y, Binning(...))`` and ``ZVar(z, ...)``: the further variables and their binnings."""
    found = []
    for axis in ("YVar", "ZVar"):
        if axis in options:
            extra = options.get(axis, 1)
            found.append((options.get(axis), extra if isinstance(extra, RooCmdArg) else None))
    return found


def _book(name: str, axes: list[tuple[Any, Any]]) -> Histogram:
    """``RooAbsRealLValue::createHistogram``'s histogram: ``h__x_y``, "Histogram of h__x_y"."""
    edges = [_binning(var, command) for var, command in axes]
    full = name + "_" + "".join("_" + var.GetName() for var, _ in axes)
    return Histogram.book(full, *[list(e) for e in edges], title=f"Histogram of {full}")


def _own_name(first: Any, args: tuple[Any, ...]) -> Any:
    """``createHistogram("x")`` alone names the histogram after the variable; else the data does."""
    return str(first) if isinstance(first, str) and "," not in first and not args else None


def _grid(edges: list[Any]) -> tuple[Any, Any]:
    """Every bin's centre, the first variable slowest, and every bin's volume."""
    centres = [0.5 * (e[1:] + e[:-1]) for e in edges]
    grid = np.array(list(itertools.product(*centres)))
    volumes = np.prod(np.array(list(itertools.product(*[np.diff(e) for e in edges]))), axis=1)
    return grid, volumes


def data_histogram(data: Any, first: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``RooAbsData::createHistogram``."""
    name = _own_name(first, args)
    axes = _axes(first, tuple(a for a in args if not isinstance(a, str)), kwargs, data.variable)
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    made = _book(name or data.GetName(), axes)
    keep = data.mask(options.get("Cut"), options.get("CutRange"))
    columns = [data.column(var.GetName())[keep] for var, _ in axes]
    made.fill(*columns, weight=data.weights()[keep])
    return WRAP[0](made)


def _by_names(func: Any, names: str, counts: tuple[Any, ...]) -> tuple[str, Any, tuple[Any, ...]]:
    """``createHistogram("x,y", nx, ny)``: the function's name, ``x``, and ``Binning(nx)``,
    ``YVar(y, Binning(ny))`` - ROOT's own translation."""
    variables = [func.variable(one) for one in re.split("[,:]", names) if one]
    bins = [int(one) for one in counts if one is not None] + [0, 0, 0]
    made: list[Any] = [RooCmdArg("Binning", bins[0])] if bins[0] > 0 else []
    for axis, var, count in zip(("YVar", "ZVar"), variables[1:], bins[1:]):
        made.append(RooCmdArg(axis, var, *([RooCmdArg("Binning", count)] if count > 0 else [])))
    return func.GetName(), variables[0], tuple(made)


def function_histogram(
    func: Any, name: str, first: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """``RooAbsReal::createHistogram``: the function at each bin's centre, a density times the
    volume."""
    if not hasattr(first, "GetName"):  # createHistogram("x,y", 50, 50): names and bin counts
        name, first, args = _by_names(func, str(name), (first, *args))
    axes = _axes(first, args, kwargs, func.variable)
    made = _book(str(name), axes)
    edges = [_binning(var, command) for var, command in axes]
    grid, volumes = _grid(edges)
    ctx = {var.GetName(): grid[:, i] for i, (var, _) in enumerate(axes)}
    nset = frozenset(ctx)
    values = _values(func, ctx, nset, volumes)
    made.fill(*[grid[:, i] for i in range(len(axes))], weight=values)
    return WRAP[0](made)


def _values(func: Any, ctx: dict[str, Any], nset: frozenset[str], volumes: Any) -> Any:
    """Each bin's content: a density's normalised value times the bin's volume, or the value."""
    if not hasattr(func, "canBeExtended"):
        return np.broadcast_to(np.asarray(func.compute(ctx), dtype=np.float64), (len(volumes),))
    from .integration import announce

    for _ in range(2):  # the projection's normalisation, and its clone's: RooFit makes both
        announce(func, nset)
    values = np.asarray(func.value(ctx, nset), dtype=np.float64)
    return np.broadcast_to(values, (len(volumes),)) * volumes


def names_of(items: Any) -> list[str]:
    return [one.GetName() for one in as_list(items)]
