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

from collections.abc import Callable, Sequence

from ..errors import UnsupportedFeatureError
from .errors import FormulaError
from .nodes import Index, Node, Reduce, Ref, Size, Special, children

__all__ = ["Specs", "bind", "loops", "varies", "explicit", "levels"]

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


def depth(node: Node, table: Specs) -> int:
    """How deep in the loop a node's value is known: how many levels of it the node crosses.

    A branch is read at the last level its own dimensions loop at, and an
    index reaching deeper than that is read there too, which is ROOT's rule
    that the index's loop is nested inside the branch's.
    """
    if isinstance(node, Reduce):
        return 0  # it loops on its own and gives one value per entry
    if isinstance(node, Special):
        return int(node.name == "Iteration$")
    if isinstance(node, (Ref, Size)):
        return _read_at(node, table)
    return max((depth(child, table) for child in children(node)), default=0)


def _read_at(node: Node, table: Specs) -> int:
    """The level one branch is read at: the dimensions it loops over, or a deeper index."""
    written = [index for index in explicit(node) if index is not None]
    inside = max((depth(index, table) for index in written), default=0)
    return max(sum(1 for spec in table[id(node)] if spec is None), inside)


def levels(specs: tuple[Index, ...], table: Specs) -> list[int]:
    """The level of the loop at which each dimension of one branch is taken.

    A dimension looped over makes the next level, and one given an index is
    taken at the level that index's own loop reaches - which is ROOT's rule
    that the index is worked out inside the loops the branch's other
    dimensions make, so ``m[][idx]`` indexes row by row.
    """
    out: list[int] = []
    level = 0
    for spec in specs:
        if spec is None:
            level += 1
        else:
            deeper = depth(spec, table)
            if deeper > level + 1:
                raise UnsupportedFeatureError(
                    f"an index reaches {deeper} levels into the loop where the dimension it "
                    f"indexes is {level + 1} deep; ROOT nests those loops inside one another "
                    f"and this does not, so give the branch an index at each dimension "
                    f"before that one"
                )
            level = max(level, deeper)
        out.append(level)
    return out


def bind(root: Node, dims_of: Callable[[str], int]) -> Specs:
    """The specs of every branch under ``root``, keyed by the ``id`` of its node."""
    table: Specs = {}
    found: list[Node] = []
    _bind(root, dims_of, table, found)
    _check(found, table)
    return table


def _bind(node: Node, dims_of: Callable[[str], int], table: Specs, found: list[Node]) -> None:
    below = children(node.ref) if isinstance(node, Size) else children(node)
    for child in below:
        _bind(child, dims_of, table, found)
    if isinstance(node, Size):
        table[id(node)] = _size_specs(node, dims_of(node.ref.column))
    elif isinstance(node, Ref):
        table[id(node)] = _ref_specs(node, dims_of(node.column))
    else:
        return
    found.append(node)


def _check(found: Sequence[Node], table: Specs) -> None:
    """Refuse a branch whose indexing reaches deeper into the loop than the loop goes.

    The levels a branch is taken at are made by the branches that loop on
    their own account, so a branch indexed at a level nothing else reaches has
    no loop to be read in: ROOT would make that loop out of the branch's own
    dimensions, which is a second kind of nesting this does not write.
    """
    reach = max(
        (
            sum(1 for spec in table[id(node)] if spec is None)
            for node in found
            if not _late(node, table)
        ),
        default=0,
    )
    for node in found:
        deepest = levels(table[id(node)], table)[-1] if explicit(node) else 0
        if deepest > reach:
            raise UnsupportedFeatureError(
                f"{column(node)!r} is read {deepest} levels into the loop and the expression "
                f"loops {reach} deep; ROOT makes that loop out of the branch's own dimensions, "
                f"which this does not nest - loop another branch alongside it, or give the "
                f"dimensions before it an index"
            )


def _late(node: Node, table: Specs) -> bool:
    return any(varies(index, table) for index in explicit(node) if index is not None)


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
