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
import weakref
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..jagged import Jagged, _bounds, take
from ..library import convert
from .formula import Formula, compile_formula
from .objects import SAMPLE, Whole, _whole, learned
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
    """Every name an expression over a tree may use, against its number of dimensions.

    A branch of the tree is a name, and so is a friend's; and so is a member of
    an object the tree holds whole - ``fNtrack``, or ``event.fNtrack`` - which
    is no branch at all and is read out of the object each entry holds.
    """

    def __init__(self, source: Any) -> None:
        self.source = source
        names = list(source.keys())
        for alias, friend in getattr(source, "friends", {}).items():
            names += [f"{alias}.{name}" for name in friend.keys()]
            names += friend.keys()
        self.names = list(dict.fromkeys(names))
        #: The members of the branches the tree holds whole, which are names too.
        self.whole = _members(source, self.names)
        self.names += [name for name in self.whole if name not in self.names]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __getitem__(self, name: str) -> int:
        if isinstance(name, str) and name in self.whole:
            return self.whole.dims[name]
        return dimensions(_underlying(self.source[name]))


#: The member tables learned from trees, by the tree and the names it offers as branches.
#: What a class holds does not change from one draw to the next, and learning it reads
#: the first entries of every whole-object branch - which the draw that learned it reads
#: again for its values - so a tree drawn many times, as ROOT's own tests draw one, says
#: what its objects hold once.
#:
#: The tree is pointed at rather than kept: a table is worth having while its tree is
#: open and worth nothing after, and a closed file must be given back, so a tree that is
#: gone takes its table with it. A table whose tree is gone is not trusted either, since
#: the next tree may be given the identity the last one had.
#:
#: A table asked for again is put back at the end of the queue, so the one that goes when
#: there are more trees than are kept is the one nothing has asked for longest.
_LEARNED: dict[tuple[int, tuple[str, ...]], tuple[Any, Whole]] = {}

#: How many trees' member tables are kept: a session's open files, not a run's.
LEARNED_KEPT = 8


def _members(source: Any, names: Sequence[str]) -> Whole:
    """What the branches the tree holds whole offer as names, learned from a few entries of them.

    A branch that cannot be read offers nothing: an expression naming one of its
    members is refused for the branch's own reason, and the other branches of
    the tree are read as they were.
    """
    key = (id(source), tuple(names))
    held = _LEARNED.get(key)
    if held is not None and held[0]() is not None:
        _LEARNED[key] = _LEARNED.pop(key)  # asked for again: the youngest table it has
        return held[1]
    whole = _learned(source, names)
    pointing = _pointing_at(source)
    if pointing is None:
        return whole  # a reader that cannot be pointed at weakly, learned every time
    _LEARNED[key] = (pointing, whole)
    while len(_LEARNED) > LEARNED_KEPT:
        del _LEARNED[next(iter(_LEARNED))]  # the tree kept longest, which nothing has asked for
    return whole


def _pointing_at(source: Any) -> Any:
    """A weak reference to a tree, or ``None`` for a reader of a kind that cannot have one."""
    try:
        return weakref.ref(source)
    except TypeError:
        return None


def _learned(source: Any, names: Sequence[str]) -> Whole:
    """The names the whole-object branches of a tree offer, from the first entries of each."""
    samples: dict[str, Any] = {}
    for name in names:
        branch = _underlying(source[name])
        if not _whole(branch):
            continue
        try:
            samples[name] = branch.array(0, SAMPLE)
        except UnsupportedFeatureError:
            continue
    return learned(samples, names) if samples else Whole()


def _local(source: Any, numbers: Array) -> Array | None:
    """The number of each entry within its own file of a chain, which ``LocalEntry$`` is."""
    if not hasattr(source, "starts"):
        return None
    starts = np.asarray(source.starts(), dtype=np.int64)
    return numbers - starts[np.searchsorted(starts, numbers, side="right") - 1]


class _Read:
    """The entries one call reads, the branches read for them, and the numbers they have."""

    def __init__(
        self,
        source: Any,
        start: int,
        stop: int | None,
        entries: Any,
        whole: Whole | None = None,
    ) -> None:
        self.source = source
        #: The members of the tree's whole objects, when the caller knows them.
        self.whole = whole
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
            if self.whole is not None and name in self.whole:
                self.columns[name] = self.whole.read(name, self.column(self.whole.branch(name)))
            else:
                branch = self.source[name]
                if self.rows is None:
                    self.columns[name] = branch.array(self.low, self.high)
                else:
                    self.columns[name] = branch.pick(self.rows)
        return self.columns[name]

    def shaped(self, name: str) -> Any:
        """One column as its branch's title declares it, or a member of a whole object as read."""
        if self.whole is not None and name in self.whole:
            return self.column(name)
        return _shaped(_underlying(self.source[name]), self.column(name))

    def evaluate(self, formula: Formula, masked: bool = False) -> Any:
        columns = {name: self.shaped(name) for name in formula.branches}
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
) -> tuple[dict[str, Formula], Formula | None, TreeNames | None]:
    """Every name that is not a branch compiled as an expression, the cut, and the names."""
    texts = [text for text in wanted if text not in source]
    if not texts and cut is None:
        # A friend recorded in the file is opened to list its names, and
        # plain branches should not make that happen.
        return {}, None, None
    known = TreeNames(source)
    formulas = {text: compile_formula(text, known, aliases=aliases) for text in texts}
    selection = None if cut is None else compile_formula(cut, known, aliases=aliases)
    return formulas, selection, known


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
    formulas, selection, known = _compiled(source, wanted, cut, aliases)
    whole = None if known is None else known.whole
    read = _Read(source, entry_start, entry_stop, entries, whole)
    columns = {
        text: read.evaluate(formulas[text]) if text in formulas else read.column(text)
        for text in wanted
    }
    if selection is not None:
        columns = _cut(columns, *read.evaluate(selection, masked=True))
    return convert(columns, library)
