"""Trees merged and copied: baskets across as they are, or entries read and written.

A tree going into another file goes one of two ways. The fast way moves its
baskets across whole - see :mod:`.baskets` - and writes only a new ``TTree``
record saying where they went; nothing is decoded, and a tree of a billion
entries costs the time it takes to copy its bytes. That needs every branch
to be one this library writes a record for the same way ROOT did: a number,
a fixed run of them, a run counted by another branch, or a string, each a
branch of one leaf. Such a branch's record is made again from the one read -
the same leaf class, the same counter of the same type, the largest count
and longest string it had - so the baskets it points at mean what they
meant.

The slow way reads the entries, a batch at a time, and writes them through
:class:`~xrdroot.WritableTree` as any tree is written: a column of numbers
that ROOT kept in a leaf this writer does not make (a ``TLeafG``, a packed
float), or a vector ROOT split out of a class, goes this way and comes out
as the nearest thing this writer makes. A cut or a choice of columns
computed from expressions goes this way too, since the entries kept are no
longer the baskets' entries. What neither way can carry - a split object, a
branch of several leaves, a column this library cannot read - is refused by
name, with the tree and the file in the message, rather than dropped.

Several trees into one - ``hadd`` - is the same thing done once per input,
into one :class:`~xrdroot.WritableTree`: each input goes whichever way it
can, and a tree whose branches are not those of the first is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from ..compression import CODES
from ..errors import UnsupportedFeatureError
from ..tree import Group
from ..wtree import _Counter, _Text, _Variable, spec_of
from .baskets import Moved, move_baskets

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from ..tree import Branch, TTree
    from ..writer import WritableDirectory
    from ..wtree import WritableTree

__all__ = ["TreeMerge", "Plan", "plan", "normal_codes"]

#: How many entries a tree read the slow way is read at a time.
STEP = 100_000

#: The type code each leaf class this writer writes is, signed and unsigned.
LEAF_CODES = {
    "TLeafO": ("?", "?"),
    "TLeafB": ("b", "B"),
    "TLeafS": ("h", "H"),
    "TLeafI": ("i", "I"),
    "TLeafL": ("q", "Q"),
    "TLeafF": ("f", "f"),
    "TLeafD": ("d", "d"),
}

#: How many bits the integer counters are, for a maximum read as signed.
COUNTER_BITS = {"b": 8, "B": 8, "h": 16, "H": 16, "i": 32, "I": 32, "q": 64, "Q": 64}


def normal_codes(codes: int) -> int:
    """A file's compression setting as ``algorithm * 100 + level``, whatever its vintage.

    ROOT before 6 wrote a bare level, meaning its default algorithm, zlib;
    level zero is no compression, whatever the algorithm says.
    """
    if codes % 100 == 0:
        return 0
    if codes < 100:
        return CODES["zlib"] * 100 + codes
    return codes


class Plan(NamedTuple):
    """What a tree is declared as when written again, and whether its baskets can go across.

    ``columns`` and ``counters`` are what :meth:`~xrdroot.WritableDirectory.tree`
    takes; ``exact`` names every column whose branch can be made again as ROOT
    made it, and ``inexact`` the ones whose type has to be read off its values.
    """

    columns: dict[str, Any]
    counters: dict[str, str]
    exact: dict[str, Any]
    inexact: list[str]


def _counted_by(tree: TTree) -> dict[int, str]:
    """Each leaf a counter holds, by identity, against the name of its branch."""
    return {id(branch.leaf): name for name, branch in tree.branches.items()}


def _single(branch: Branch) -> bool:
    """Is this a branch of one leaf, at the start of the entry, with nothing under it?"""
    record = branch.record
    return (
        not isinstance(branch, Group)
        and len(record.leaves) == 1
        and not record.branches
        and not branch.leaf.offset
    )


def exact_spec(branch: Branch, counters: Mapping[int, str]) -> Any:
    """What a branch is declared as to be made again as ROOT made it, or ``None``.

    ``None`` is for a branch whose record this writer does not make the same
    way: several leaves or branches under it, a leaf not at the start of the
    entry, a leaf class this writer has no layout for, a run counted by a
    branch that is not a plain integer of its own, or a fixed-size column
    whose baskets nonetheless carry a table of where each entry begins.
    """
    if not _single(branch):
        return None
    leaf = branch.leaf
    if leaf.classname == "TLeafC":
        return str if leaf.count is None else None
    codes = LEAF_CODES.get(leaf.classname)
    if codes is None:
        return None
    return _numbers(branch, codes[int(leaf.unsigned)], counters)


def _numbers(branch: Branch, code: str, counters: Mapping[int, str]) -> Any:
    """A column of numbers: one per entry, a fixed run of them, or a counted run."""
    leaf = branch.leaf
    if leaf.count is not None:
        counter = counters.get(id(leaf.count))
        return (code, counter) if counter is not None and leaf.length == 1 else None
    if branch.record.entry_offset_len:
        return None
    return code if leaf.length == 1 else (code, leaf.length)


def _require_carried(tree: TTree, where: str) -> None:
    """Refuse a tree with branches that neither way of writing it can carry."""
    stuck = dict(tree.unreadable)
    for name in tree.groups():
        stuck[name] = "a split object, whose members this writer does not write as members"
    for name, branch in tree.branches.items():
        if name not in stuck and len(branch.record.leaves) > 1:
            stuck[name] = "one leaf of a branch of several, which this writer does not write"
    if stuck:
        listed = "; ".join(f"{name}: {why}" for name, why in stuck.items())
        raise UnsupportedFeatureError(
            f"the tree {where} has branches that cannot be written again here - {listed} - "
            f"so it is not merged or copied rather than copied without them; leave it out, "
            f"or pick the columns to copy"
        )


def _counter(spec: Any) -> str | None:
    """The counter a declaration names, if it is of a counted run."""
    return spec[1] if isinstance(spec, tuple) and isinstance(spec[1], str) else None


def _counters(exact: Mapping[str, Any], names: Sequence[str]) -> dict[str, str]:
    """The counters coming with their runs, each against its integer type."""
    counters: dict[str, str] = {}
    for spec in exact.values():
        counter = _counter(spec)
        if counter is None or counter not in names:
            continue
        code = exact.get(counter)
        if isinstance(code, str) and code in COUNTER_BITS:
            counters[counter] = code
    return counters


def plan(tree: TTree, names: Sequence[str]) -> Plan:
    """How the columns ``names`` of ``tree`` are declared when it is written again.

    A counter chosen along with the runs it counts is declared as their
    counter rather than as a column, so that it is filled from their lengths
    - and keeps its own integer type; one chosen without them is a column of
    numbers like any other, and a run chosen without its counter gets one of
    its own, its type read off its values.
    """
    counted_by = _counted_by(tree)
    exact = {name: exact_spec(tree.branches[name], counted_by) for name in names}
    counters = _counters(exact, names)
    columns: dict[str, Any] = {}
    inexact: list[str] = []
    for name in names:
        if name in counters:
            continue
        spec = exact[name]
        counter = _counter(spec)
        if counter is not None and counter not in counters:
            spec = None  # counted by a branch not coming too, or not an integer
        if spec is None:
            inexact.append(name)
        columns[name] = spec
    return Plan(columns, counters, {k: v for k, v in exact.items() if v is not None}, inexact)


def _mask(maximum: int, code: str) -> int:
    """A counter's largest value, read as signed, as the unsigned type it may be."""
    if maximum < 0 and code.isupper():
        return maximum + (1 << COUNTER_BITS[code])
    return maximum


class TreeMerge:
    """One tree being written from one or more trees read, each the fast way or the slow.

    The first tree decides the columns; every one after must have the same
    branches of the same types, or it is refused by name. ``names`` picks
    the columns to take - all of them unless said - and ``cut`` keeps only
    the entries that pass it, which sends every input the slow way.
    """

    def __init__(
        self,
        directory: WritableDirectory,
        name: str,
        first: TTree,
        where: str,
        *,
        names: Sequence[str] | None = None,
        cut: str | None = None,
        expressions: Mapping[str, str] | None = None,
        fast: bool = True,
    ) -> None:
        self.directory = directory
        self.name = name
        self.where = where
        self.cut = cut
        self.expressions = dict(expressions or {})
        chosen = list(first.keys()) if names is None else list(names)
        #: Whether the whole tree is taken, so every input must have the same branches.
        self.whole = names is None
        if self.whole:
            _require_carried(first, where)
        self.names = chosen
        self.plan = plan(first, chosen)
        self.fast = fast and cut is None and not self.expressions and not self.plan.inexact
        self.first_types = first.typenames()
        self.title = first.title
        self.tree: WritableTree | None = None
        self.moved = Moved(0, 0, 0)
        self.slow_entries = 0

    def _declared(self, first_batch: Mapping[str, Any]) -> WritableTree:
        """The tree being written, declared the first time there is something for it."""
        if self.tree is None:
            columns = dict(self.plan.columns)
            try:
                for name in [*self.plan.inexact, *self.expressions]:
                    columns[name] = spec_of(name, first_batch[name])
                self.tree = self.directory.tree(
                    self.name, columns, title=self.title, counters=self.plan.counters
                )
            except ValueError as why:
                raise UnsupportedFeatureError(
                    f"the tree {self.where} cannot be written again here: {why}"
                ) from None
        return self.tree

    def add(self, tree: TTree, where: str, *, verbatim: bool, in_place: bool = False) -> None:
        """Take in one more tree's entries, across as baskets if it can go that way."""
        self._require_alike(tree, where)
        if self.fast and self._fits(tree):
            self._move(tree, verbatim, in_place)
        else:
            self._copy(tree)

    def _require_alike(self, tree: TTree, where: str) -> None:
        """Refuse a tree whose branches are not the first one's, naming those that differ."""
        theirs = tree.typenames()
        names = set(self.first_types) | set(theirs) if self.whole else set(self.names)
        differ = sorted(name for name in names if theirs.get(name) != self.first_types.get(name))
        if not differ:
            return
        described = ", ".join(
            f"{name} ({theirs.get(name, 'missing')} here, "
            f"{self.first_types.get(name, 'missing')} in the first)"
            for name in differ
        )
        raise ValueError(
            f"the tree {where} does not have the branches of the first one merged into it: "
            f"{described}; trees are merged only when their branches are the same"
        )

    def _fits(self, tree: TTree) -> bool:
        """Can this tree's baskets go across: every branch as the first's, whole?"""
        counted_by = _counted_by(tree)
        for name, spec in self.plan.exact.items():
            branch = tree.branches[name]
            record = branch.record
            if exact_spec(branch, counted_by) != spec:
                return False
            if record.first_entry or record.entries != tree.num_entries:
                return False
        return True

    def _move(self, tree: TTree, verbatim: bool, in_place: bool) -> None:
        written = self._declared({})
        written._flush_all()
        moved = [0, 0, 0]
        for column in written._branches:
            branch = tree.branches[column.name]
            done = move_baskets(
                written, column, branch.record, tree._source, verbatim=verbatim, in_place=in_place
            )
            moved = [a + b for a, b in zip(moved, done)]
            self._carry_leaf(column, branch)
        written._entries += tree.num_entries
        self.moved = Moved(*(a + b for a, b in zip(self.moved, moved)))

    def _carry_leaf(self, column: Any, branch: Branch) -> None:
        """What a branch's record says about its values, said again for the column."""
        if isinstance(column, _Counter):
            column.maximum = max(column.maximum, _mask(branch.leaf.maximum, column.typecode))
        elif isinstance(column, _Text):
            column.widest = max(column.widest, branch.leaf.length - 1)
        if isinstance(column, _Variable) and branch.record.entry_offset_len:
            column.offset_len = branch.record.entry_offset_len

    def _copy(self, tree: TTree) -> None:
        """Read the entries a batch at a time and write them as any entries are written."""
        wanted = [name for name in self.names if name not in self.plan.counters]
        chosen = wanted + list(self.expressions.values())
        for batch in tree.iterate(chosen, step=STEP, cut=self.cut):
            given = {name: batch[name] for name in wanted}
            for label, expression in self.expressions.items():
                given[label] = batch[expression]
            written = self._declared(given)
            written.extend(given)
            self.slow_entries += len(next(iter(given.values()))) if given else 0
        if self.tree is None:  # nothing passed, or nothing was there: still a tree
            self._declared(_empty(tree, wanted, self.expressions))


def _empty(tree: TTree, names: Sequence[str], expressions: Mapping[str, str]) -> dict[str, Any]:
    """No entries of each column, for a tree with none to declare its types from."""
    found = tree.arrays(list(names) + list(expressions.values()), 0, 0)
    given = {name: found[name] for name in names}
    for label, expression in expressions.items():
        given[label] = found[expression]
    return given
