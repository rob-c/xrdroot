"""``TMVA::BinarySearchTree``: the k-d tree ``PDERS`` keeps its training events in.

What PDERS asks of the tree - the weight of the events inside a box, and
which they are - does not depend on its shape, and is answered here by
testing every event at once. The shape matters for the weight file, which
is the tree itself, node by node: :func:`build` makes it as TMVA does, by
inserting the events one after another (each going left when its value in
the node's variable is not above the node's), or with ``NormTree`` by
inserting medians first so that it is balanced; :func:`tree_xml` writes it
and :func:`read_tree_xml` reads the events back, TMVA's files as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .xmlfile import Node, children, number

__all__ = ["TreeEvents", "build", "read_tree_xml", "tree_xml"]


@dataclass
class TreeEvents:
    """The events of a tree: values, class, weight and targets, in any order."""

    values: Any
    classes: Any
    weights: Any
    targets: Any


@dataclass
class _Node:
    index: int
    depth: int
    pos: str
    left: _Node | None = None
    right: _Node | None = None


def _insert(root: _Node | None, index: int, values: Any) -> _Node:
    """``Insert``: down from the root, left when not above the node's value in its variable."""
    if root is None:
        return _Node(index, 0, "s")
    nvar = values.shape[1]
    node = root
    while True:
        selector = node.depth % nvar
        side = "l" if values[index, selector] <= values[node.index, selector] else "r"
        below = node.left if side == "l" else node.right
        if below is None:
            made = _Node(index, node.depth + 1, side)
            if side == "l":
                node.left = made
            else:
                node.right = made
            return root
        node = below


def _balanced(order: list[int], values: Any, dim: int, root: list[_Node | None]) -> None:
    """``NormalizeTree``: the median inserted, then each side's, the variable turning each level."""
    if not order:
        return
    nvar = values.shape[1]
    dim %= nvar
    order = sorted(order, key=lambda i: (float(values[i, dim]), i))
    mid = len(order) // 2
    while mid > 0 and values[order[mid], dim] == values[order[mid - 1], dim]:
        mid -= 1
    root[0] = _insert(root[0], order[mid], values)
    _balanced(order[:mid], values, dim + 1, root)
    _balanced(order[mid + 1 :], values, dim + 1, root)


def build(values: Any, normalise: bool) -> _Node | None:
    """The tree TMVA's ``Fill`` (and with ``normalise`` its ``NormalizeTree``) makes."""
    holder: list[_Node | None] = [None]
    if normalise:
        _balanced(list(range(len(values))), values, 0, holder)
    else:
        for index in range(len(values)):
            holder[0] = _insert(holder[0], index, values)
    return holder[0]


def tree_xml(parent: Node, events: TreeEvents, normalise: bool) -> None:
    """``BinaryTree::AddXMLTo``: every node, its event's values and targets as its text."""
    tree = parent.add("BinaryTree", type="BinarySearchTree")
    root = build(events.values, normalise)
    nvar = events.values.shape[1]
    stack: list[tuple[Node, _Node]] = [(tree, root)] if root is not None else []
    while stack:
        into, node = stack.pop()
        i = node.index
        made = into.add(
            "Node",
            pos=node.pos,
            depth=node.depth,
            selector=node.depth % nvar,
            weight=number(events.weights[i]),
            type=int(events.classes[i]),
            NVars=nvar,
        )
        numbers = [*events.values[i], *(events.targets[i] if events.targets is not None else [])]
        made.text = "".join(" " + number(value) for value in numbers)
        stack.extend((made, below) for below in (node.right, node.left) if below is not None)


def read_tree_xml(node: Any) -> TreeEvents:
    """The events of a ``<BinaryTree>``, as ``BinarySearchTree::CreateFromXML`` reads them."""
    rows, classes, weights, targets = [], [], [], []
    stack = children(node, "Node")
    while stack:
        item = stack.pop()
        nvar = int(str(item.get("NVars")))
        numbers = np.asarray((item.text or "").split(), dtype=np.float32).astype(np.float64)
        rows.append(numbers[:nvar])
        targets.append(numbers[nvar:])
        classes.append(int(str(item.get("type"))))
        weights.append(float(np.float32(item.get("weight"))))
        stack.extend(children(item, "Node"))
    width = max((len(t) for t in targets), default=0)
    return TreeEvents(
        np.asarray(rows),
        np.asarray(classes),
        np.asarray(weights),
        np.asarray(targets) if width else None,
    )
