"""``createHistogram``: a dataset, or a function, as a ROOT histogram of one, two or three variables.

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
    low, high = (command.value(1), command.value(2)) if command.value(2) is not None else (var.getMin(), var.getMax())
    return np.linspace(float(low), float(high), int(first) + 1)


def _axes(first: Any, args: tuple[Any, ...], kwargs: dict[str, Any], known: Any) -> list[tuple[Any, Any]]:
    """Each variable and its binning command: ``x, Binning, YVar(y, Binning)``, or ``"x,y"`` and bins."""
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    binnings = options.every("Binning")
    if isinstance(first, str):
        names = [one for one in first.split(",") if one]
        return [(known(name), binnings[i] if i < len(binnings) else None) for i, name in enumerate(names)]
    found = [(first, binnings[0] if binnings else None)]
    for axis in ("YVar", "ZVar"):
        if axis in options:
            extra = options.get(axis, 1)
            found.append((options.get(axis), extra if isinstance(extra, RooCmdArg) else None))
    return found


def _book(name: str, axes: list[tuple[Any, Any]]) -> Histogram:
    edges = [_binning(var, command) for var, command in axes]
    return Histogram.book(name, *[list(e) for e in edges], title=name)


def data_histogram(data: Any, first: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``RooAbsData::createHistogram``."""
    name = str(first) if isinstance(first, str) and "," not in first and not args else None
    axes = _axes(first, tuple(a for a in args if not isinstance(a, str)), kwargs, data.variable)
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    made = _book(name or data.GetName(), axes)
    keep = data.mask(options.get("Cut"), options.get("CutRange"))
    columns = [data.column(var.GetName())[keep] for var, _ in axes]
    made.fill(*columns, weight=data.weights()[keep])
    return WRAP[0](made)


def function_histogram(func: Any, name: str, first: Any, args: tuple[Any, ...],
                       kwargs: dict[str, Any]) -> Any:  # fmt: skip
    """``RooAbsReal::createHistogram``: the function at each bin's centre, a density times the volume."""
    axes = _axes(first, args, kwargs, func.variable)
    made = _book(str(name), axes)
    edges = [_binning(var, command) for var, command in axes]
    centres = [0.5 * (e[1:] + e[:-1]) for e in edges]
    grid = np.array(list(itertools.product(*centres)))
    ctx = {var.GetName(): grid[:, i] for i, (var, _) in enumerate(axes)}
    nset = frozenset(var.GetName() for var, _ in axes)
    volumes = np.prod(np.array(list(itertools.product(*[np.diff(e) for e in edges]))), axis=1)
    is_pdf = hasattr(func, "canBeExtended")
    values = np.asarray(func.value(ctx, nset) if is_pdf else func.compute(ctx), dtype=np.float64)
    values = np.broadcast_to(values, (len(grid),)) * (volumes if is_pdf else 1.0)
    made.fill(*[grid[:, i] for i in range(len(axes))], weight=values)
    return WRAP[0](made)


def names_of(items: Any) -> list[str]:
    return [one.GetName() for one in as_list(items)]
