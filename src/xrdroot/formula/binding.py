"""What each branch in an expression is indexed by, once its dimensions are known.

A name says only which branch; how many dimensions that branch has is what
turns ``pt`` into a loop or a number. ROOT's rule is that every dimension not
given an index is looped over - so ``m`` of a ``[3][3]`` array is ``m[][]``,
and ``m[1]`` is ``m[1][]`` - and binding writes that out: each branch's
*specs*, one per dimension, either the index expression written for it or
``None`` for a dimension to loop over.

It also answers the questions everything above it asks: whether an expression
loops at all, which is whether its result is one value per entry or one per
element; and whether a value can differ from one element of an entry to the
next, which decides when an index has to be worked out element by element.
"""

from __future__ import annotations

from collections.abc import Callable

from ..errors import UnsupportedFeatureError
from .errors import FormulaError
from .nodes import Index, Node, Reduce, Ref, Size, Special, children

__all__ = ["Specs", "bind", "loops", "varies", "explicit"]

#: The index of each dimension of one branch, ``None`` for one that is looped over.
Specs = dict[int, tuple[Index, ...]]


def explicit(node: Node) -> tuple[Index, ...]:
    """The indices written after a branch, whether it is read or its size is."""
    ref = node.ref if isinstance(node, Size) else node
    assert isinstance(ref, Ref)
    return ref.indices


def column(node: Node) -> str:
    ref = node.ref if isinstance(node, Size) else node
    assert isinstance(ref, Ref)
    return ref.column


def _ref_specs(node: Ref, dims: int) -> tuple[Index, ...]:
    if len(node.indices) > dims:
        raise FormulaError(
            f"{node.column!r} has {dims} dimension{'' if dims == 1 else 's'} and is given "
            f"{len(node.indices)} indices"
        )
    return node.indices + (None,) * (dims - len(node.indices))


def _size_specs(node: Size, dims: int) -> tuple[Index, ...]:
    """Loop over all but the innermost dimension left, whose size is the value; with ``@``, none."""
    indices = node.ref.indices
    if len(indices) >= dims:
        raise FormulaError(
            f"{node.ref.column!r} indexed {len(indices)} times is one value, not a "
            f"collection, and has no size()"
        )
    left = 0 if node.outer else dims - len(indices) - 1
    return indices + (None,) * left


def _mismatch(node: Node, specs: tuple[Index, ...], table: Specs) -> None:
    """Refuse a branch that both loops and is indexed by something that loops."""
    written = [index for index in explicit(node) if index is not None]
    if None in specs and any(varies(index, table) for index in written):
        raise UnsupportedFeatureError(
            f"{column(node)!r} is indexed by a value that changes from element to element "
            f"and also looped over in another dimension; ROOT nests those loops inside one "
            f"another, which this does not do - give every other dimension an index"
        )


def bind(root: Node, dims_of: Callable[[str], int]) -> Specs:
    """The specs of every branch under ``root``, keyed by the ``id`` of its node."""
    table: Specs = {}
    _bind(root, dims_of, table)
    return table


def _bind(node: Node, dims_of: Callable[[str], int], table: Specs) -> None:
    below = children(node.ref) if isinstance(node, Size) else children(node)
    for child in below:
        _bind(child, dims_of, table)
    if isinstance(node, Size):
        table[id(node)] = _size_specs(node, dims_of(node.ref.column))
    elif isinstance(node, Ref):
        table[id(node)] = _ref_specs(node, dims_of(node.column))
    else:
        return
    _mismatch(node, table[id(node)], table)


def _differs(node: Node, table: Specs, iteration: bool) -> bool:
    if isinstance(node, Reduce):
        return False
    if isinstance(node, Special):
        return iteration and node.name == "Iteration$"
    if isinstance(node, (Ref, Size)):
        specs = table[id(node)]
        written = [index for index in explicit(node) if index is not None]
        return None in specs or any(_differs(index, table, iteration) for index in written)
    return any(_differs(child, table, iteration) for child in children(node))


def loops(node: Node, table: Specs) -> bool:
    """Does ``node`` loop over a collection, making one value per element rather than per entry?"""
    return _differs(node, table, iteration=False)


def varies(node: Node, table: Specs) -> bool:
    """Can ``node`` differ between elements of one entry: a loop, or ``Iteration$``?"""
    return _differs(node, table, iteration=True)
