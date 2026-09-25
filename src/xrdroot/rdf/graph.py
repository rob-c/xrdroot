"""The computation graph: which entries reach a node, and how each column is made.

An ``RDataFrame`` is a graph built before anything is read. Its *selecting*
nodes decide which entries go on: the root lets every entry through, a
:class:`Filter` those its condition passes, a :class:`Range` those in its
span of what reached it. A ``Define`` does not select - it adds a column to
what the frame after it can see - so a frame is a selecting node and the
columns visible there, and ``Define`` makes a new frame over the same node.

A column is a :class:`Definition`: a column of the data, an entry or slot
number, or a :class:`Defined` one, made by an expression or a callable from
other columns. A defined column belongs to the node it was defined at, and
is computed for exactly the entries that reach that node - never for one a
filter above it rejected, as ROOT has it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .expression import Expression

__all__ = [
    "Definition",
    "SourceColumn",
    "EntryColumn",
    "SlotColumn",
    "Defined",
    "PerSample",
    "Selector",
    "Root",
    "Filter",
    "Range",
    "chain_of",
]


class Definition:
    """How one column is made."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def origin(self) -> str:
        """Where the column comes from, as ``Describe`` says it."""
        return "Dataset"


class SourceColumn(Definition):
    """A column the data has: a branch of a tree, a field of an RNTuple."""

    __slots__ = ()


class EntryColumn(Definition):
    """``rdfentry_``: the number of each entry in the data, counting from zero."""

    __slots__ = ()


class SlotColumn(Definition):
    """``rdfslot_``: the number of the worker processing each entry, zero when there is one."""

    __slots__ = ()


class Defined(Definition):
    """A column made from others, at the node it was defined at.

    ``compute`` is an :class:`~.expression.Expression`, or a callable given
    the input columns a batch at a time; ``inputs`` are the definitions of
    the columns it reads, by the names it reads them by.
    """

    __slots__ = ("compute", "inputs", "selector")

    def __init__(
        self,
        name: str,
        compute: Expression | Callable[..., Any],
        inputs: dict[str, Definition],
        selector: Selector,
    ) -> None:
        super().__init__(name)
        self.compute = compute
        self.inputs = inputs
        self.selector = selector

    @property
    def origin(self) -> str:
        return "Define"


class PerSample(Definition):
    """``DefinePerSample``: one value per file of the data, from what the file is."""

    __slots__ = ("compute",)

    def __init__(self, name: str, compute: Callable[[Any], Any]) -> None:
        super().__init__(name)
        self.compute = compute

    @property
    def origin(self) -> str:
        return "DefinePerSample"


class Selector:
    """A node that decides which entries go on to the nodes below it."""

    __slots__ = ("parent",)

    def __init__(self, parent: Selector | None) -> None:
        self.parent = parent


class Root(Selector):
    """The head of the graph: every entry of the data."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(None)


class Filter(Selector):
    """The entries whose condition holds, of those that reached it.

    ``condition`` is an expression or a callable, ``inputs`` the columns it
    reads, and ``name`` what ``Report`` calls it; a filter given no name is
    not in the report, as in ROOT.
    """

    __slots__ = ("condition", "inputs", "name")

    def __init__(
        self,
        parent: Selector,
        condition: Expression | Callable[..., Any],
        inputs: dict[str, Definition],
        name: str,
    ) -> None:
        super().__init__(parent)
        self.condition = condition
        self.inputs = inputs
        self.name = name


class Range(Selector):
    """``Range(begin, end, stride)``: a span of the entries that reached it, counted as they do."""

    __slots__ = ("begin", "end", "stride")

    def __init__(self, parent: Selector, begin: int, end: int | None, stride: int) -> None:
        super().__init__(parent)
        self.begin = begin
        self.end = end
        self.stride = stride


def chain_of(node: Selector) -> list[Selector]:
    """Every selecting node from the root down to ``node``, in that order."""
    found: list[Selector] = []
    at: Selector | None = node
    while at is not None:
        found.append(at)
        at = at.parent
    return found[::-1]


def named_filters(nodes: Sequence[Selector]) -> list[Filter]:
    """The filters among ``nodes`` that ``Report`` counts: those with names."""
    return [node for node in nodes if isinstance(node, Filter) and node.name]
