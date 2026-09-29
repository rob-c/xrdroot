"""Which of the old package's nodes are drawn: ``TNode::SetVisibility`` and ``TNode::Paint``.

A node is painted when it and its shape are visible, and its daughters
are painted after it unless it says its sons are hidden; the node drawn is
at the origin, each daughter placed in its mother by its shift and turn.
``SetVisibility`` sets a node - and, for most of its codes, the nodes
below it - in one of ROOT's seven ways:

====  ===========================================================
 1    the node is drawn
 0    the node is not drawn
 -1   neither the node nor anything below it is drawn
 -2   the node is drawn, nothing below it is
 -3   only the leaves below it are drawn
 -4   the node is not drawn, only its daughters are
 2    the node is not drawn, everything below it is
 3    the node and everything below it are drawn
====  ===========================================================
"""

from __future__ import annotations

from typing import Any

from ...geom import IDENTITY, Matrix, Solid

__all__ = ["set_visibility", "solids"]

#: ``TNode``'s bit saying its daughters are not drawn.
SONS_INVISIBLE = 1 << 17

#: Each code: the node's own visibility, whether its sons are hidden, and the code
#: passed to each of them (``None``: they are left as they are).
CODES: dict[int, tuple[int, bool, int | None]] = {
    -4: (0, False, -2), -3: (0, False, -3), -2: (1, True, -1), -1: (0, True, -1),
    0: (0, False, None), 1: (1, False, None), 2: (0, False, 3), 3: (1, False, 3),
}  # fmt: skip
#: The codes a node with no daughters answers by being drawn itself.
LEAF_SHOWN = (-4, -3)


def set_visibility(node: Any, code: int) -> None:
    """Set ``node``'s visibility - and its daughters', where the code says - as ROOT does."""
    node._bits &= ~SONS_INVISIBLE
    if code not in CODES:
        return
    own, hidden, passed = CODES[code]
    daughters = node.GetListOfNodes()
    node._visibility = 1 if code in LEAF_SHOWN and not daughters else own
    if hidden:
        node._bits |= SONS_INVISIBLE
    if passed is not None:
        for daughter in daughters:
            set_visibility(daughter, passed)


def _solid(node: Any, matrix: Matrix, bomb: float) -> Solid:
    """One node's shape where ``matrix`` puts it - its shift times the geometry's ``bomb``,
    as ``TNode::Local2Master`` spreads an exploded view - in the node's own look."""
    spread = Matrix(matrix.rotation, bomb * matrix.translation)
    return Solid(node.GetShape().mesh().transformed(spread), int(node.GetLineColor()),
                 int(node.GetLineWidth()), int(node.GetLineStyle()), int(node.GetFillColor()),
                 0, node.GetName())  # fmt: skip


def _paint(node: Any, matrix: Matrix, bomb: float, found: list[Solid]) -> None:
    if node.GetVisibility() and node.GetShape().GetVisibility():
        found.append(_solid(node, matrix, bomb))
    if node._bits & SONS_INVISIBLE:
        return
    for daughter in node.GetListOfNodes():
        _paint(daughter, matrix @ daughter.placement(), bomb, found)


def solids(node: Any) -> list[Solid]:
    """What drawing ``node`` paints: it at the origin, and the visible tree below it."""
    from .legacy import current_geometry

    found: list[Solid] = []
    _paint(node, IDENTITY, current_geometry().GetBomb(), found)
    return found
