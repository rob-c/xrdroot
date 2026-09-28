"""TMVA's decision trees as arrays: made from scikit-learn's, written and read as TMVA's XML.

A BDT's forest is written as TMVA writes it - a ``<BinaryTree>`` of nested
``<Node>`` elements, each with the variable it cuts on (``IVar``), the cut,
which side of it is signal (``cType``), its purity, its response and, for a
leaf, whether it is signal or background (``nType``) - so the weight files
here read like TMVA's, and TMVA's read here. A tree is evaluated by walking
every event down it at once: an event goes right when ``(x >= cut)`` is
``cType``, as ``DecisionTreeNode::GoesRight`` has it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .xmlfile import Node, number

__all__ = ["Tree", "from_sklearn", "read_tree"]


@dataclass
class Tree:
    """One tree, a node per row: the arrays of ``DecisionTreeNode``'s members."""

    var: Any
    cut: Any
    ctype: Any
    left: Any
    right: Any
    response: Any
    purity: Any
    ntype: Any
    depth: Any
    rms: Any = None
    #: For a node cutting on a Fisher discriminant, its coefficients - the offset last.
    fisher: dict[int, Any] = field(default_factory=dict)

    def leaves(self, values: Any) -> Any:
        """The node each event ends in."""
        node = np.zeros(len(values), dtype=np.int64)
        rows = np.arange(len(values))
        while True:
            inner = self.left[node] >= 0
            if not inner.any():
                return node
            current = node[inner]
            x = self._projection(values[rows[inner]], current)
            right = (x >= self.cut[current]) == (self.ctype[current] == 1)
            node[inner] = np.where(right, self.right[current], self.left[current])

    def _projection(self, values: Any, nodes: Any) -> Any:
        """Each event's value of the variable its node cuts on - or the node's Fisher discriminant."""
        x = values[np.arange(len(nodes)), np.maximum(self.var[nodes], 0)].astype(np.float64)
        for index, coefficients in self.fisher.items():
            at = nodes == index
            if at.any():
                x[at] = values[at] @ coefficients[:-1] + coefficients[-1]
        return x

    def respond(self, values: Any, yes_no: bool, what: str = "auto") -> Any:
        """``CheckEvent``: each event's leaf type (``yes_no``), response or purity."""
        leaf = self.leaves(values)
        if yes_no:
            return self.ntype[leaf].astype(np.float64)
        if what == "purity":
            return self.purity[leaf].astype(np.float64)
        return self.response[leaf].astype(np.float64)

    def add_xml(self, parent: Node, boost_weight: float, index: int, extra: dict[str, Any]) -> None:
        """``DecisionTree::AddXMLTo``: the tree, its nodes nested as TMVA nests them."""
        tree = parent.add("BinaryTree", type="DecisionTree", **extra)
        tree.set("boostWeight", number(boost_weight)).set("itree", index)
        self._add_node(tree, 0, "s")

    def _add_node(self, parent: Node, node: int, pos: str) -> None:
        made = parent.add("Node", pos=pos, depth=int(self.depth[node]))
        coefficients = self.fisher.get(node)
        made.set("NCoef", 0 if coefficients is None else len(coefficients))
        for i, value in enumerate(coefficients if coefficients is not None else ()):
            made.set(f"fC{i}", number(value))
        made.set("IVar", int(self.var[node])).set("Cut", number(self.cut[node]))
        made.set("cType", int(self.ctype[node])).set("res", number(self.response[node]))
        made.set("rms", number(0.0 if self.rms is None else self.rms[node]))
        made.set("purity", number(self.purity[node])).set("nType", int(self.ntype[node]))
        if self.left[node] >= 0:
            self._add_node(made, int(self.left[node]), "l")
            self._add_node(made, int(self.right[node]), "r")


def _node_rows(element: Any, rows: list[dict[str, Any]], depth: int = 0) -> int:
    """The rows of ``element`` and every node below it, parents before children; its own row."""
    index = len(rows)
    row = {key: element.get(key) for key in element.keys()}
    row["left"] = row["right"] = -1
    rows.append(row)
    for child in element:
        if child.tag != "Node":
            continue
        side = "left" if child.get("pos") == "l" else "right"
        rows[index][side] = _node_rows(child, rows, depth + 1)
    return index


def read_tree(element: Any) -> Tree:
    """``DecisionTree::CreateFromXML``: a tree from its ``<BinaryTree>``."""
    rows: list[dict[str, Any]] = []
    root = next(child for child in element if child.tag == "Node")
    _node_rows(root, rows)

    def column(key: str, default: Any, kind: Any) -> Any:
        return np.array([kind(row.get(key, default) or default) for row in rows])

    fisher = {}
    for index, row in enumerate(rows):
        count = int(row.get("NCoef", 0) or 0)
        if count:
            fisher[index] = np.array([float(row[f"fC{i}"]) for i in range(count)])
    purity = column("purity", "nan", float)
    if "purity" not in rows[0] and "nS" in rows[0]:
        signal, background = column("nS", 0, float), column("nB", 0, float)
        purity = signal / (signal + background)
    return Tree(
        column("IVar", -1, int),
        np.float32(column("Cut", 0.0, float)).astype(np.float64),
        column("cType", 1, int),
        np.array([row["left"] for row in rows]),
        np.array([row["right"] for row in rows]),
        np.float32(column("res", -99.0, float)).astype(np.float64),
        np.float32(purity).astype(np.float64),
        column("nType", 0, int),
        column("depth", 0, int),
        np.float32(column("rms", 0.0, float)).astype(np.float64),
        fisher,
    )


def _above(threshold: Any) -> Any:
    """The least single-precision number above each threshold: a cut ``x >= cut`` iff ``x > t``."""
    single = np.asarray(threshold, dtype=np.float32)
    return np.where(single > threshold, single, np.nextafter(single, np.float32(np.inf)))


def from_sklearn(
    fitted: Any, signal_column: int | None, limit: float, response: Any = None
) -> Tree:
    """A scikit-learn tree as TMVA's: its cuts, purities, leaf types and leaf responses.

    ``signal_column`` is the class column holding the signal in the tree's
    values, for a classification tree; ``response`` the value each node
    answers with, for the boosted regression trees gradient boosting makes.
    """
    inner = fitted.tree_
    left, right = inner.children_left.copy(), inner.children_right.copy()
    leaf = left < 0
    values = inner.value[:, 0, :]
    if signal_column is None:
        purity = np.zeros(len(left))
    else:
        totals = values.sum(axis=1)
        purity = np.where(totals > 0, values[:, signal_column] / np.where(totals > 0, totals, 1), 0)
    ntype = np.where(leaf, np.where(purity > limit, 1, -1), 0)
    depth = np.zeros(len(left), dtype=np.int64)
    for node in range(len(left)):
        if not leaf[node]:
            depth[left[node]] = depth[right[node]] = depth[node] + 1
    cut = np.where(leaf, 0.0, _above(inner.threshold)).astype(np.float64)
    made_response = np.full(len(left), -99.0) if response is None else np.asarray(response, float)
    return Tree(
        np.where(leaf, -1, inner.feature),
        cut,
        np.ones(len(left), dtype=np.int64),
        np.where(leaf, -1, left),
        np.where(leaf, -1, right),
        np.float32(made_response).astype(np.float64),
        np.float32(purity).astype(np.float64),
        ntype,
        depth,
    )
