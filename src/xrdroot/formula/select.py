"""``arrays(["pt", "sqrt(px*px + py*py)"], cut="n > 2")``: expressions read from a tree.

What ``TTree.arrays`` and ``Chain.arrays`` do with a name that is not a
branch: compile it as an expression, read the branches it needs - and only
those, once each however many expressions share them - and give its value
under the text it was asked for by.

A ``cut`` is ROOT's selection. One that is a value per entry keeps the
entries where it is nonzero. One that loops over a collection is applied as
``TTree::Draw`` applies it: element by element, to every column that is
itself one value per element - an expression that loops, a variable-length
branch, a fixed-size array - pairing the cut's elements with the column's
up to the shorter of the two, and keeping every entry in which any element
passed, so that the columns still line up entry for entry.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from ..library import convert
from ..tree import Jagged, _bounds, take
from .formula import Formula, compile_formula
from .ops import truth

__all__ = ["select", "TreeNames", "dimensions"]

Array = Any

#: The bracketed sizes in a leaf's title, ``m[3][3]`` or ``x[N][3]``.
SIZES = re.compile(r"\[([^\]]*)\]")


def _underlying(found: Any) -> Any:
    """The branch of a chain's first file, which says what every file's is."""
    return found._branch(0) if hasattr(found, "chain") else found


def _fixed(branch: Any) -> list[int]:
    """The sizes of a branch's fixed dimensions, as its leaf title declares them."""
    sizes = [int(size) for size in SIZES.findall(branch.leaf.title) if size.isdigit()]
    if branch.is_jagged:
        return sizes
    if sizes and int(np.prod(sizes)) == branch.length:
        return sizes
    return [branch.length]


def dimensions(branch: Any) -> int:
    """How many dimensions a branch has: rows of lists, a variable length, fixed sizes."""
    nested = (branch.typename or "").count("list[")
    if nested:
        return nested
    if branch.is_jagged:
        return 1 + len(_fixed(branch))
    return len(_fixed(branch)) if branch.length > 1 else 0


def _shaped(branch: Any, value: Any) -> Any:
    """A column in the shape its title declares: ``m[3][3]`` read as 9 is made 3 by 3."""
    sizes = _fixed(branch)
    if isinstance(value, np.ndarray) and value.ndim == 2 and len(sizes) > 1:
        return value.reshape(len(value), *sizes)
    width = int(np.prod(sizes)) if branch.is_jagged and sizes else 1
    if width > 1 and isinstance(value, Jagged) and not (value.offsets % width).any():
        return Jagged(value.content.reshape(-1, *sizes), value.offsets // width)
    return value


class TreeNames(Mapping[str, int]):
    """Every name an expression over a tree may use, against its number of dimensions."""

    def __init__(self, source: Any) -> None:
        self.source = source
        names = list(source.keys())
        for alias, friend in getattr(source, "friends", {}).items():
            names += [f"{alias}.{name}" for name in friend.keys()]
            names += friend.keys()
        self.names = list(dict.fromkeys(names))

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __getitem__(self, name: str) -> int:
        return dimensions(_underlying(self.source[name]))


def _local(source: Any, numbers: Array) -> Array | None:
    """The number of each entry within its own file of a chain, which ``LocalEntry$`` is."""
    if not hasattr(source, "starts"):
        return None
    starts = np.asarray(source.starts(), dtype=np.int64)
    return numbers - starts[np.searchsorted(starts, numbers, side="right") - 1]


class _Read:
    """The entries one call reads, the branches read for them, and the numbers they have."""

    def __init__(self, source: Any, start: int, stop: int | None, entries: Any) -> None:
        self.source = source
        #: The entries named, or ``None`` for the range ``low`` to ``high``.
        self.rows: Array | None = None
        self.low, self.high = _bounds(len(source), start, stop)
        if entries is None:
            self.numbers = np.arange(self.low, max(self.low, self.high), dtype=np.int64)
        else:
            self.rows = source._selected(entries)
            self.numbers = np.asarray(self.rows, dtype=np.int64)
        self.local = _local(source, self.numbers)
        self.columns: dict[str, Any] = {}

    def column(self, name: str) -> Any:
        """One branch's values for these entries, read the first time it is asked for."""
        if name not in self.columns:
            branch = self.source[name]
            if self.rows is None:
                self.columns[name] = branch.array(self.low, self.high)
            else:
                self.columns[name] = branch.pick(self.rows)
        return self.columns[name]

    def evaluate(self, formula: Formula, masked: bool = False) -> Any:
        branches = {name: self.source[name] for name in formula.branches}
        columns = {
            name: _shaped(_underlying(branch), self.column(name))
            for name, branch in branches.items()
        }
        method = formula.evaluate_masked if masked else formula.evaluate
        return method(
            columns,
            entries=len(self.source),
            rows=len(self.numbers),
            entry_numbers=self.numbers,
            local_entries=self.local,
        )


def _as_rows(value: Any) -> Jagged:
    """A fixed-size array column as rows of its elements, for a cut to pick among."""
    if isinstance(value, Jagged):
        return value
    width = int(np.prod(value.shape[1:]))
    return Jagged(value.reshape(-1), np.arange(len(value) + 1, dtype=np.int64) * width)


def _loops(value: Any) -> bool:
    return isinstance(value, Jagged) or (isinstance(value, np.ndarray) and value.ndim > 1)


def _elementwise(value: Any, passing: Array, offsets: Array) -> Jagged:
    """The elements of a column the cut passed, paired up to the shorter row of the two."""
    rows = _as_rows(value)
    count = len(rows)
    paired = np.minimum(rows.lengths(), np.diff(offsets))
    row = np.repeat(np.arange(count, dtype=np.int64), paired)
    local = np.arange(len(row), dtype=np.int64) - np.repeat(np.cumsum(paired) - paired, paired)
    kept = passing[offsets[:-1][row] + local]
    lengths = np.bincount(row[kept], minlength=count)
    new = np.zeros(count + 1, np.int64)
    np.cumsum(lengths, out=new[1:])
    return Jagged(rows.content[(rows.offsets[:-1][row] + local)[kept]], new)


def _cut(columns: dict[str, Any], values: Any, valid: Any) -> dict[str, Any]:
    """The columns with the cut applied: entry by entry, or element by element."""
    if not isinstance(values, Jagged):
        kept = np.flatnonzero(np.asarray(valid) & truth(np.asarray(values)))
        return {name: take(value, kept) for name, value in columns.items()}
    passing = valid.content & truth(values.content)
    count = len(values)
    row = np.repeat(np.arange(count, dtype=np.int64), values.lengths())
    kept = np.flatnonzero(np.bincount(row[passing], minlength=count))
    out = {}
    for name, value in columns.items():
        if _loops(value):
            value = _elementwise(value, passing, values.offsets)
        out[name] = take(value, kept)
    return out


def _compiled(
    source: Any, wanted: list[str], cut: str | None, aliases: Mapping[str, str] | None
) -> tuple[dict[str, Formula], Formula | None]:
    """Every name that is not a branch compiled as an expression, and the cut."""
    texts = [text for text in wanted if text not in source]
    if not texts and cut is None:
        # A friend recorded in the file is opened to list its names, and
        # plain branches should not make that happen.
        return {}, None
    known = TreeNames(source)
    formulas = {text: compile_formula(text, known, aliases=aliases) for text in texts}
    selection = None if cut is None else compile_formula(cut, known, aliases=aliases)
    return formulas, selection


def select(
    source: Any,
    names: Sequence[str] | None,
    entry_start: int = 0,
    entry_stop: int | None = None,
    *,
    library: str = "np",
    entries: Any = None,
    cut: str | None = None,
    aliases: Mapping[str, str] | None = None,
) -> Any:
    """The columns and expressions ``names`` asks for, from a tree or a chain, cut by ``cut``.

    A name the source has as a branch is read as it is; any other is an
    expression, compiled against the source's branches and ``aliases``, and
    given under its own text. With ``cut``, only what it selects comes back.
    """
    wanted = source.readable() if names is None else list(names)
    formulas, selection = _compiled(source, wanted, cut, aliases)
    read = _Read(source, entry_start, entry_stop, entries)
    columns = {
        text: read.evaluate(formulas[text]) if text in formulas else read.column(text)
        for text in wanted
    }
    if selection is not None:
        columns = _cut(columns, *read.evaluate(selection, masked=True))
    return convert(columns, library)
