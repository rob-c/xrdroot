"""A compiled expression: the branches it reads, and its value for a batch of them.

    >>> f = compile_formula("sqrt(px*px + py*py)", tree.keys())     # doctest: +SKIP
    >>> f.branches
    ('px', 'py')
    >>> f.evaluate(tree.arrays(f.branches, 0, 1000))

Compiling parses the text and resolves every name in it against the
branches, so a typo is refused before anything is read. Evaluating takes the
columns as ``TTree.arrays`` gives them - NumPy arrays, :class:`Jagged` rows,
lists of strings or of STL containers - and works on all their entries at
once: nothing loops over entries in Python except to take apart a column that
arrived as Python objects.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np

from ..tree import Jagged
from .binding import Specs, bind, loops
from .columns import Layout, layout
from .evaluate import Evaluator, Numbers, Value
from .loops import Space
from .names import Names
from .nodes import Node, Ref, children
from .parser import parse

__all__ = ["Formula", "compile_formula"]

Array = Any


def _branches(root: Node) -> tuple[str, ...]:
    """Every branch under ``root``, once each, in the order they are first written."""
    found: dict[str, None] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, Ref):
            found.setdefault(node.column)
        stack.extend(reversed(children(node)))
    return tuple(found)


def _dims(value: Any, name: str) -> int:
    """How many dimensions a column has, from a count or from the column itself."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return layout(name, value).dims


def _package(space: Space, value: Value, entries: int) -> tuple[Any, Any]:
    """The top value as ``evaluate_masked`` gives it: per entry, or rows of elements."""
    size = len(space) if space.looped else entries
    data = value.data
    if np.ndim(data) == 0:
        data = np.full(size, data, dtype=np.asarray(data).dtype)
    valid = np.ones(size, np.bool_) if value.valid is None else np.asarray(value.valid)
    if not space.looped:
        return data, valid
    offsets = space.offsets()
    return Jagged(data, offsets), Jagged(valid, offsets)


def _dropped(values: Jagged, valid: Jagged) -> Jagged:
    """Rows of elements with the missing ones taken out, as ROOT's Draw leaves them out."""
    keep = valid.content
    if keep.all():
        return values
    rows = np.repeat(np.arange(len(values), dtype=np.int64), values.lengths())
    offsets = np.zeros(len(values) + 1, np.int64)
    np.cumsum(np.bincount(rows[keep], minlength=len(values)), out=offsets[1:])
    return Jagged(values.content[keep], offsets)


def _blanked(values: Array, valid: Array) -> Array:
    """One value per entry, NaN where there is none; strings, which have no NaN, as they are."""
    if valid.all() or values.dtype.kind in "US":
        return values
    out = values.astype(np.float64)
    out[~valid] = np.nan
    return out


class Formula:
    """One compiled ``TTree::Draw`` expression.

    ``text`` is what was compiled, ``branches`` the columns evaluating it
    needs - resolved to the names they have in the tree, so ``evt.P3.Px``
    needs ``P3.Px`` - and :attr:`per_element` whether it gives a value per
    entry or one per element of the collections it loops over.
    """

    __slots__ = ("text", "branches", "_root", "_dims")

    def __init__(self, text: str, root: Node, dims: Mapping[str, int] | None) -> None:
        self.text = text
        self._root = root
        self.branches: tuple[str, ...] = _branches(root)
        self._dims = dims
        if dims is not None:
            bind(root, dims.__getitem__)  # refuses a bad index now rather than later

    def __repr__(self) -> str:
        return f"<Formula {self.text!r} of {', '.join(self.branches) or 'no branches'}>"

    @property
    def per_element(self) -> bool:
        """Does it loop over a collection, giving a value per element rather than per entry?

        Known when it was compiled with a mapping that says how many
        dimensions each branch has, as a tree's own compile does; a plain
        list of names cannot say whether ``pt`` is one number or many.
        """
        if self._dims is None:
            raise ValueError(
                f"whether {self.text!r} loops depends on which of its branches are "
                f"collections, and it was compiled with names alone; compile it with a "
                f"mapping of each name to its number of dimensions, or look at what "
                f"evaluate gives back"
            )
        return loops(self._root, bind(self._root, self._dims.__getitem__))

    def _layouts(self, columns: Mapping[str, Any]) -> dict[str, Layout]:
        layouts = {}
        for name in self.branches:
            if name not in columns:
                raise KeyError(
                    f"{self.text!r} reads {name!r}, which is not among the columns given; "
                    f"read formula.branches"
                )
            want = None if self._dims is None else self._dims[name]
            layouts[name] = layout(name, columns[name], want)
        return layouts

    def _entries(
        self, layouts: Mapping[str, Layout], columns: Mapping[str, Any], rows: int | None
    ) -> int:
        counts = {name: found.rows for name, found in layouts.items()}
        if rows is not None:
            counts["rows"] = rows
        if not counts and columns:
            name = next(iter(columns))  # reads none, but any column says how many rows
            counts[name] = len(columns[name])
        if not counts:
            raise ValueError(
                f"{self.text!r} reads no branches and was given no columns, so nothing "
                f"says how many entries there are; give rows="
            )
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"the columns given for {self.text!r} are of different lengths ("
                + ", ".join(f"{name} has {count}" for name, count in counts.items())
                + "), and an expression is evaluated entry for entry"
            )
        return next(iter(counts.values()))

    def evaluate_masked(
        self,
        columns: Mapping[str, Any],
        *,
        entry_start: int = 0,
        entries: int | None = None,
        rows: int | None = None,
        entry_numbers: Any = None,
        local_entries: Any = None,
    ) -> tuple[Any, Any]:
        """The values, and a mask of which of them are really there.

        Values are a NumPy array of one per entry, or - when the expression
        loops over a collection - a :class:`Jagged` of one row per entry and
        one element per iteration of the loop. The mask is the same shape,
        and false where an index ran past the end of its collection: those
        are the values ROOT's ``Draw`` leaves out, and what they hold here
        is meaningless.

        ``entry_start`` is the entry number of the first row, which is what
        ``Entry$`` counts from, and ``entries`` how many the whole tree has,
        which is ``Entries$`` (the number of rows, if not given).
        ``entry_numbers`` gives the number of each row instead, for rows
        that are not a range; ``local_entries`` the number of each within its
        own file of a chain, which is ``LocalEntry$`` (``Entry$`` if not
        given); and ``rows`` how many there are, which only an expression
        that reads no branches needs to be told.
        """
        layouts = self._layouts(columns)
        given = rows if entry_numbers is None else len(entry_numbers)
        count = self._entries(layouts, columns, given)
        numbers = _numbers(count, entry_start, entries, entry_numbers, local_entries)
        specs: Specs = bind(self._root, lambda name: layouts[name].dims)
        with np.errstate(all="ignore"):
            space, value = Evaluator(layouts, specs, count, numbers).top(self._root)
        return _package(space, value, count)

    def evaluate(
        self,
        columns: Mapping[str, Any],
        *,
        entry_start: int = 0,
        entries: int | None = None,
        rows: int | None = None,
        entry_numbers: Any = None,
        local_entries: Any = None,
    ) -> Any:
        """The values, with what is missing left out as ROOT's ``Draw`` leaves it out.

        One per entry as a NumPy array, NaN for an entry with no value; or,
        looping over a collection, a :class:`Jagged` whose rows hold only the
        elements that have one. :meth:`evaluate_masked` says which is which
        instead, and takes the same arguments.
        """
        values, valid = self.evaluate_masked(
            columns,
            entry_start=entry_start,
            entries=entries,
            rows=rows,
            entry_numbers=entry_numbers,
            local_entries=local_entries,
        )
        if isinstance(values, Jagged):
            return _dropped(values, valid)
        return _blanked(values, valid)


def _numbers(count: int, start: int, entries: int | None, numbers: Any, local: Any) -> Numbers:
    entry = (
        np.arange(start, start + count, dtype=np.int64)
        if numbers is None
        else np.asarray(numbers, dtype=np.int64)
    )
    here = entry if local is None else np.asarray(local, dtype=np.int64)
    return Numbers(entry, here, count if entries is None else entries)


def compile_formula(
    text: str, names: Collection[str], *, aliases: Mapping[str, str] | None = None
) -> Formula:
    """Compile a ``TTree::Draw`` expression against the branches it may read.

        >>> f = compile_formula("Sum$(jet_pt > 30)", tree.keys())   # doctest: +SKIP

    ``names`` are the branches, and resolve what the expression writes: a
    dotted ``evt.P3.Px`` finds the split member ``P3.Px``, a friend's
    ``alias.x`` its column, the longest branch a name spells winning. When
    ``names`` is a mapping - of each name to its number of dimensions, or to
    the column itself - the formula also knows before it is evaluated
    whether it loops, which :attr:`Formula.per_element` says.

    ``aliases`` are ROOT's ``SetAlias``: names that stand for expressions of
    their own, expanded wherever they are written, and refused when one
    stands for itself.
    """
    root = parse(text, Names(names), aliases)
    formula = Formula(text, root, None)
    if isinstance(names, Mapping):
        dims = {name: _dims(names[name], name) for name in formula.branches}
        formula = Formula(text, root, dims)
    return formula
