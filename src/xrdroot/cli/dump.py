"""``xrdroot dump``: every object in a file, as text - go-hep's ``root-dump``.

    $ xrdroot dump tests/data/simple.root
    >>> file[tests/data/simple.root]
    key[000]: tree;1 "fake data" (TTree)
    [000][one]: 1
    [000][two]: 1.1
    [000][three]: uno
    ...

Each key gets a line - its path, cycle, title and class - and then what it
holds: a tree's or an RNTuple's entries a line per column, a histogram's
every bin with its edges, content and error, a graph's every point with its
bars, a function's formula and parameters, and anything else - a string, an
array, the members of a class - as it reads. What will not read says why
and the dump goes on.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from ..efficiency import Efficiency
from ..errors import ROOTError
from ..file import Directory, ROOTFile
from ..function import Function
from ..graph import Graph
from ..hist import Histogram
from ..profile import Profile
from ..tree import Jagged
from .target import CONTAINERS, key_of, opened, resolve, walk

__all__ = ["add_parser", "run", "dump", "text"]

#: Entries read from a tree at a time.
STEP = 1000


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "dump",
        help="print every object - bins, points, entries - as root-dump does",
        description="Print what every key holds: trees entry by entry, histograms bin by "
        "bin, graphs point by point, and everything else as it reads.",
    )
    parser.add_argument("files", nargs="+", metavar="FILE[:path]", help="a file, or a path in one")
    parser.add_argument("-k", "--key", help="the path in the file, for a name without .root")
    parser.add_argument(
        "-n", "--entries", type=int, default=None, help="at most this many entries of each tree"
    )


def run(args: argparse.Namespace) -> int:
    for text_ in args.files:
        with opened(text_, args.key) as (file, path):
            for line in dump(file, path, args.entries):
                print(line)
    return 0


def dump(file: ROOTFile, path: str = "", entries: int | None = None) -> Iterator[str]:
    """The lines ``dump`` prints for one file, or one path in it."""
    yield f">>> file[{file.name}]"
    target = resolve(file, path)
    if isinstance(target, Directory):
        keyed = list(walk(target, f"{path}/" if path else ""))
    else:
        directory, key = key_of(file, path)
        keyed = [(path, key, directory)]
    for index, (where, key, directory) in enumerate(keyed):
        yield f'key[{index:03d}]: {where};{key.cycle} "{key.title}" ({key.classname})'
        if key.classname not in CONTAINERS:
            yield from _body(directory, f"{key.name};{key.cycle}", entries)


def _body(directory: Directory, name: str, entries: int | None) -> Iterator[str]:
    """What one key holds, or why it would not read."""
    try:
        value = directory[name]
    except ROOTError as why:
        yield f"  unreadable: {why}"
        return
    yield from _describer(value)(value, entries)


def _describer(value: Any) -> Callable[[Any, int | None], Iterator[str]]:
    """How to write out an object of this kind."""
    if callable(getattr(value, "readable", None)) and hasattr(value, "iterate"):
        return _table
    for kind, describe in DESCRIBERS:
        if isinstance(value, kind):
            return describe
    return _plain


def _table(table: Any, entries: int | None) -> Iterator[str]:
    """A tree's or an RNTuple's entries: ``[entry][column]: value``, a line each."""
    names = table.readable()
    for name, why in table.unreadable.items():
        yield f"  unreadable column {name}: {why}"
    stop = len(table) if entries is None else min(entries, len(table))
    start = 0
    for batch in table.iterate(names, step=STEP, entry_stop=stop):
        count = len(batch[names[0]]) if names else 0
        for row in range(count):
            for name in names:
                yield f"[{start + row:03d}][{name}]: {text(batch[name][row])}"
        start += count


def _histogram(histogram: Histogram, entries: int | None) -> Iterator[str]:
    """Every bin of a histogram or profile: its edges, content and error.

    A profile's bins are means, which do not add up to anything, so it has
    no sum and nothing "outside the axes" to count.
    """
    counted = not isinstance(histogram, Profile)
    total = f"  sum: {text(histogram.sum())}" if counted else ""
    yield f"  entries: {text(histogram.entries)}{total}"
    edges = [histogram.edges(axis) for axis in range(len(histogram.shape))]
    for axis, (letter, along) in enumerate(zip("xyz", edges)):
        mean = f"  mean: {histogram.mean(axis):.6g}  std: {histogram.std(axis):.6g}"
        yield f"  {letter}: {len(along) - 1} bins in [{along[0]:g}, {along[-1]:g}){mean}"
    if counted:
        yield f"  outside the axes: {text(histogram.sum(flow=True) - histogram.sum())}"
    values, errors = histogram.values(), histogram.errors()
    for cell in np.ndindex(*histogram.shape):
        span = " x ".join(f"[{e[i]:g}, {e[i + 1]:g})" for e, i in zip(edges, cell))
        at = cell if len(cell) > 1 else cell[0]
        yield f"  bin {at} {span}: {_pm(values[cell], errors[cell])}"


def _graph(graph: Graph, entries: int | None) -> Iterator[str]:
    """Every point of a graph, with its bars either side."""
    xerr, layers = graph.xerr, graph.layers
    for index, (x, y) in enumerate(graph):
        bars = "" if xerr is None else f" x-{xerr[0][index]:g} x+{xerr[1][index]:g}"
        for low, high in layers:
            bars += f" y-{low[index]:g} y+{high[index]:g}"
        yield f"  point {index}: ({x:g}, {y:g}){bars}"


def _efficiency(efficiency: Efficiency, entries: int | None) -> Iterator[str]:
    """Every bin of an efficiency: passed over total, and the interval round it."""
    yield f"  method: {efficiency.method}  level: {efficiency.level:g}"
    values, (low, high) = efficiency.values(), efficiency.errors()
    passed, total = efficiency.passed.values(), efficiency.total.values()
    for cell in np.ndindex(*values.shape):
        yield (
            f"  bin {cell if len(cell) > 1 else cell[0]}: {passed[cell]:g}/{total[cell]:g}"
            f" = {values[cell]:.6g} -{low[cell]:.6g} +{high[cell]:.6g}"
        )


def _function(function: Function, entries: int | None) -> Iterator[str]:
    """A function's formula, range and parameters."""
    yield f"  formula: {function.formula}"
    yield f"  range: {text(function.range)}"
    errors = function.parameter_errors
    for index, (name, value) in enumerate(zip(function.parameter_names, function.parameters)):
        yield f"  [{index}] {name} = {_pm(value, errors[index])}"


def _plain(value: Any, entries: int | None) -> Iterator[str]:
    """Anything else - a string, a number, an array, the members of a class - as it reads.

    What holds several objects - a ``TClonesArray``, a ``TMultiGraph`` - has
    a line for each.
    """
    if isinstance(value, Sequence) and not isinstance(value, str):
        for index, item in enumerate(value):
            yield f"  [{index}]: {item!r}"
        return
    yield f"  {text(value)}"


#: Each kind of object, and how it is written out; the first that fits wins.
DESCRIBERS: list[tuple[type, Callable[[Any, int | None], Iterator[str]]]] = [
    (Histogram, _histogram),
    (Graph, _graph),
    (Efficiency, _efficiency),
    (Function, _function),
]


def _pm(value: Any, error: Any) -> str:
    return f"{float(value):.6g} ± {float(error):.6g}"


def text(value: Any) -> str:
    """One value as a line of text: arrays and lists space-separated in brackets, as go-hep does."""
    if (isinstance(value, np.ndarray) and value.ndim) or isinstance(value, (list, tuple, Jagged)):
        return "[" + " ".join(text(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {text(item)}" for key, item in value.items()) + "}"
    return str(value)
