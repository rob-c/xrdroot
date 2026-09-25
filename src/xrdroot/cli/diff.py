"""``xrdroot diff``: are two ROOT files the same, and where not - go-hep's ``root-diff``.

    $ xrdroot diff before.root after.root
    h1: contents differ in 3 of 102 bins, first at 17: 4 and 5
    only in after.root: h2
    $ echo $?
    1

The names in each directory, the class of each, and then what each holds:
a histogram's edges, contents, errors and entries, a graph's points and
bars, a tree's entries column by column, a function's parameters, and the
members of anything else. Numbers are the same within ``--atol`` and
``--rtol`` - exactly, unless told otherwise - with NaN the same as NaN.

The exit status is ``diff``'s: 0 for the same, 1 for different, 2 for
trouble. An object neither side can read is said on standard error and
not counted, since there is nothing to compare it by.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np

from ..efficiency import Efficiency
from ..errors import ROOTError
from ..file import Directory, open_root
from ..function import Function
from ..graph import Graph
from ..hist import Histogram
from ..tree import Jagged
from .target import CONTAINERS, location, resolve

__all__ = ["add_parser", "run", "Tolerance", "compare"]

#: Entries compared at a time.
STEP = 10_000


class Tolerance(NamedTuple):
    """How close two numbers are to be the same, and what the two sides are called."""

    atol: float = 0.0
    rtol: float = 0.0
    first: str = "the first"
    second: str = "the second"


Found = Iterator[str]


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "diff",
        help="compare two files: keys, histograms, graphs and trees (exit 1 if different)",
        description="Compare two ROOT files key by key and value by value. The exit status "
        "is 0 if they are the same, 1 if they differ and 2 for trouble.",
    )
    parser.add_argument("first", metavar="A[:path]", help="one file, or a path in it")
    parser.add_argument("second", metavar="B[:path]", help="the other")
    parser.add_argument("--atol", type=float, default=0.0, help="absolute tolerance (0)")
    parser.add_argument("--rtol", type=float, default=0.0, help="relative tolerance (0)")
    parser.add_argument(
        "-k", "--keys", help="comma-separated names to compare, rather than every one"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="say nothing; the exit status says it"
    )


def run(args: argparse.Namespace) -> int:
    one, one_path = location(args.first)
    two, two_path = location(args.second)
    keys = [name.strip() for name in args.keys.split(",")] if args.keys else None
    tolerance = Tolerance(args.atol, args.rtol, one, two)
    with open_root(one) as first, open_root(two) as second:
        found = list(
            compare(resolve(first, one_path), resolve(second, two_path), tolerance, keys, one_path)
        )
    if not args.quiet:
        for line in found:
            print(line)
    return 1 if found else 0


def compare(
    one: Any,
    two: Any,
    tolerance: Tolerance,
    keys: Sequence[str] | None = None,
    path: str = "",
) -> Found:
    """Every difference between two objects - files, directories, or anything in them."""
    if isinstance(one, Directory) and isinstance(two, Directory):
        return _directories(one, two, f"{path}/" if path else "", tolerance, keys)
    return _values(one, two, path or "the objects", tolerance)


def _directories(
    one: Directory, two: Directory, where: str, tolerance: Tolerance, keys: Sequence[str] | None
) -> Found:
    """The names each directory holds, then what each name common to both holds."""
    mine, theirs = one.keys(), two.keys()
    for name in keys if keys is not None else list(dict.fromkeys(mine + theirs)):
        path = f"{where}{name}"
        here, there = name in mine, name in theirs
        if here and there:
            yield from _keys(one, two, name, path, tolerance)
        elif here or there:
            yield f"only in {tolerance.first if here else tolerance.second}: {path}"
        else:
            yield f"in neither: {path}"


def _keys(one: Directory, two: Directory, name: str, path: str, tolerance: Tolerance) -> Found:
    """One name in both: its class, then what it holds."""
    mine, theirs = one.key(name).classname, two.key(name).classname
    if mine != theirs:
        yield f"{path}: a {mine} in {tolerance.first}, a {theirs} in {tolerance.second}"
        return
    if mine in CONTAINERS:
        yield from _directories(one[name], two[name], f"{path}/", tolerance, None)
        return
    try:
        values = one[name], two[name]
    except ROOTError as why:
        sys.stderr.write(f"xrdroot diff: {path} not compared: {why}\n")
        return
    yield from _values(*values, path, tolerance)


def _values(one: Any, two: Any, path: str, tolerance: Tolerance) -> Found:
    """Two objects of one kind, compared the way that kind is."""
    for test, differences in COMPARISONS:
        if test(one) and test(two):
            yield from differences(one, two, path, tolerance)
            return
    if type(one) is not type(two):
        yield f"{path}: a {type(one).__name__} and a {type(two).__name__}"
    elif one != two:
        yield f"{path}: {one!r} and {two!r}"


def _numbers(
    one: Any, two: Any, path: str, tolerance: Tolerance, what: str = "values", start: int = 0
) -> Found:
    """Two arrays, the same within the tolerance, NaN the same as NaN."""
    mine, theirs = np.asarray(one), np.asarray(two)
    if mine.shape != theirs.shape:
        yield f"{path}: {what} have shapes {mine.shape} and {theirs.shape}"
        return
    if _numeric(mine) and _numeric(theirs):
        close = np.isclose(mine, theirs, atol=tolerance.atol, rtol=tolerance.rtol, equal_nan=True)
    else:
        close = np.asarray(mine == theirs, dtype=bool)
    if not mine.ndim:
        if not close:
            yield f"{path}: {what} {mine} and {theirs}"
        return
    wrong = np.argwhere(~close)
    if len(wrong):
        at = tuple(int(i) for i in wrong[0])
        where = at[0] + start if len(at) == 1 else at
        yield (
            f"{path}: {what} differ in {len(wrong)} of {mine.size} places, "
            f"first at {where}: {mine[at]} and {theirs[at]}"
        )


def _numeric(array: np.ndarray[Any, Any]) -> bool:
    return array.dtype.kind in "biufc"


def _histograms(one: Histogram, two: Histogram, path: str, tolerance: Tolerance) -> Found:
    """A histogram's axes, then its contents, errors and entries."""
    if one.shape != two.shape:
        yield f"{path}: {one.shape} bins and {two.shape} bins"
        return
    for axis in range(len(one.shape)):
        yield from _numbers(
            one.edges(axis), two.edges(axis), path, tolerance, f"edges of axis {axis}"
        )
    yield from _numbers(one.values(flow=True), two.values(flow=True), path, tolerance, "contents")
    yield from _numbers(
        one.variances(flow=True), two.variances(flow=True), path, tolerance, "errors"
    )
    yield from _numbers(one.entries, two.entries, path, tolerance, "entries")


def _graphs(one: Graph, two: Graph, path: str, tolerance: Tolerance) -> Found:
    """A graph's points, then its bars."""
    if len(one) != len(two):
        yield f"{path}: {len(one)} points and {len(two)} points"
        return
    yield from _numbers(one.x, two.x, path, tolerance, "x")
    yield from _numbers(one.y, two.y, path, tolerance, "y")
    yield from _numbers(_bars(one.xerr), _bars(two.xerr), path, tolerance, "x errors")
    yield from _numbers(_bars(one.layers), _bars(two.layers), path, tolerance, "y errors")


def _bars(bars: Any) -> np.ndarray[Any, Any]:
    """Error bars, however many layers of them, as one array - none as an empty one."""
    return np.asarray(bars if bars is not None else (), dtype=float)


def _efficiencies(one: Efficiency, two: Efficiency, path: str, tolerance: Tolerance) -> Found:
    yield from _histograms(one.passed, two.passed, f"{path} (passed)", tolerance)
    yield from _histograms(one.total, two.total, f"{path} (total)", tolerance)


def _functions(one: Function, two: Function, path: str, tolerance: Tolerance) -> Found:
    if one.formula != two.formula:
        yield f"{path}: {one.formula!r} and {two.formula!r}"
    yield from _numbers(one.parameters, two.parameters, path, tolerance, "parameters")


def _tables(one: Any, two: Any, path: str, tolerance: Tolerance) -> Found:
    """A tree's or an RNTuple's entries and columns, then each column a batch at a time."""
    if len(one) != len(two):
        yield f"{path}: {len(one)} entries and {len(two)} entries"
        return
    mine, theirs = one.keys(), two.keys()
    yield from (f"only in {tolerance.first}: {path}.{name}" for name in mine if name not in theirs)
    yield from (f"only in {tolerance.second}: {path}.{name}" for name in theirs if name not in mine)
    common = [name for name in one.readable() if name in set(two.readable())]
    yield from _entries(one, two, common, path, tolerance)


def _entries(one: Any, two: Any, common: list[str], path: str, tolerance: Tolerance) -> Found:
    """The columns both tables read, a batch at a time; each column's first difference only."""
    told: set[str] = set()
    batches = zip(one.iterate(common, step=STEP), two.iterate(common, step=STEP))
    for index, (first, second) in enumerate(batches):
        for name in (name for name in common if name not in told):
            found = list(
                _column(first[name], second[name], f"{path}.{name}", tolerance, index * STEP)
            )
            if found:
                told.add(name)
            yield from found


def _column(one: Any, two: Any, path: str, tolerance: Tolerance, start: int) -> Found:
    """One batch of one column: numbers, rows of numbers, or objects."""
    if isinstance(one, Jagged) and isinstance(two, Jagged):
        yield from _numbers(one.lengths(), two.lengths(), path, tolerance, "row lengths", start)
        yield from _numbers(one.flat, two.flat, path, tolerance, "elements")
    elif isinstance(one, np.ndarray) and isinstance(two, np.ndarray) and one.dtype != object:
        yield from _numbers(one, two, path, tolerance, "entries", start)
    else:
        for index, (mine, theirs) in enumerate(zip(one, two)):
            if next(_values(mine, theirs, "", tolerance), None) is not None:
                yield f"{path}: entries differ, first at {start + index}: {mine!r} and {theirs!r}"
                return


def _dicts(one: dict[str, Any], two: dict[str, Any], path: str, tolerance: Tolerance) -> Found:
    """The members of a class the file describes, member by member."""
    for name in dict.fromkeys([*one, *two]):
        if name not in one or name not in two:
            yield f"{path}.{name}: only in {tolerance.first if name in one else tolerance.second}"
            continue
        yield from _values(one[name], two[name], f"{path}.{name}", tolerance)


def _sequences(one: Sequence[Any], two: Sequence[Any], path: str, tolerance: Tolerance) -> Found:
    if len(one) != len(two):
        yield f"{path}: {len(one)} items and {len(two)} items"
        return
    for index, (mine, theirs) in enumerate(zip(one, two)):
        yield from _values(mine, theirs, f"{path}[{index}]", tolerance)


def _arrays(one: Any, two: Any, path: str, tolerance: Tolerance) -> Found:
    return _numbers(one, two, path, tolerance)


def _is_table(value: Any) -> bool:
    return callable(getattr(value, "readable", None)) and hasattr(value, "iterate")


def _kind(kind: type | tuple[type, ...]) -> Callable[[Any], bool]:
    return lambda value: isinstance(value, kind)


def _is_number(value: Any) -> bool:
    return isinstance(value, (np.ndarray, np.generic, int, float)) and not isinstance(value, bool)


#: What two objects are compared as, the first test both pass winning.
COMPARISONS: list[tuple[Callable[[Any], bool], Callable[[Any, Any, str, Tolerance], Found]]] = [
    (_is_table, _tables),
    (_kind(Histogram), _histograms),
    (_kind(Graph), _graphs),
    (_kind(Efficiency), _efficiencies),
    (_kind(Function), _functions),
    (_kind(dict), _dicts),
    (_is_number, _arrays),
    (_kind((list, tuple)), _sequences),
]
