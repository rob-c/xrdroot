"""``TMVA::Experimental::SaveXGBoost`` and ``RBDT``: an XGBoost model saved, read and evaluated.

``SaveXGBoost(model, key, file, num_inputs)`` writes the model's trees -
every node's feature, threshold, children and leaf value, from XGBoost's own
JSON dump - into ``file`` as a tree of nodes called ``key``; ``RBDT(key,
file)`` reads it back and ``Compute`` evaluates it with NumPy, as XGBoost
does: the leaves' sum and the base margin, through the objective's link
(logistic, softmax, or none). TMVA writes an ``RBDT`` object, which only
ROOT reads; the tree here is what xrdroot reads.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

__all__ = ["RBDT", "SaveXGBoost"]

#: The objectives, by the code the saved tree keeps: 0 none, 1 logistic, 2 softmax.
LINKS = {"binary:logistic": 1, "reg:logistic": 1, "multi:softprob": 2, "multi:softmax": 2}


def _base_scores(config: dict[str, Any]) -> list[float]:
    """The model's base score - or, from XGBoost 3 on, a multiclass model's one for each class."""
    text = str(config["learner"]["learner_model_param"].get("base_score", "0.5"))
    return [float(part) for part in text.strip("[]").split(",")]


#: A saved tree's columns: its tree, a split's feature, threshold and children, a leaf's value.
COLUMNS = ("tree", "feature", "threshold", "yes", "no", "missing", "value")


def _flattened(tree: dict[str, Any], start: int) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """A tree's nodes, parents before children, and the row each node's id is at from ``start``."""
    stack: list[dict[str, Any]] = [tree]
    order: list[dict[str, Any]] = []
    ids: dict[int, int] = {}
    while stack:
        node = stack.pop()
        ids[int(node["nodeid"])] = start + len(order)
        order.append(node)
        stack.extend(reversed(node.get("children", [])))
    return order, ids


def _row(node: dict[str, Any], ids: dict[int, int]) -> tuple[Any, ...]:
    """A node's feature, threshold, children and value; a leaf's split and children ``-1``."""
    if "leaf" in node:
        return -1, 0.0, -1, -1, -1, float(node["leaf"])
    children = (ids[int(node[side])] for side in ("yes", "no", "missing"))
    return (int(str(node["split"]).lstrip("f")), float(node["split_condition"]), *children, 0.0)


def _nodes(dump: list[str]) -> dict[str, list[Any]]:
    """Every node of every tree, a row each, children as rows of the same table."""
    rows: dict[str, list[Any]] = {k: [] for k in COLUMNS}
    for index, text in enumerate(dump):
        order, ids = _flattened(json.loads(text), len(rows["tree"]))
        for node in order:
            for name, value in zip(COLUMNS, (index, *_row(node, ids))):
                rows[name].append(value)
    return rows


def SaveXGBoost(model: Any, key: Any, filename: Any, num_inputs: int = 0, **_: Any) -> None:
    """The XGBoost model's trees into ``filename`` (updated if it exists), called ``key``."""
    import xrdroot

    booster = model.get_booster()
    config = json.loads(booster.save_config())
    objective = str(config["learner"]["objective"]["name"])
    rows = _nodes(booster.get_dump(dump_format="json"))
    classes = int(getattr(model, "n_classes_", 2) or 2) if LINKS.get(objective) == 2 else 1
    count = len(rows["tree"])
    columns = {
        name: np.asarray(values, dtype=np.float64 if name in ("threshold", "value") else np.int32)
        for name, values in rows.items()
    }
    # Each tree's row keeps the base score of the class it adds to: tree ``k`` adds to ``k % n``.
    bases = np.asarray(_base_scores(config), dtype=np.float64)
    columns["base_score"] = bases[columns["tree"] % len(bases)]
    columns["link"] = np.full(count, LINKS.get(objective, 0), dtype=np.int32)
    columns["classes"] = np.full(count, classes, dtype=np.int32)
    columns["inputs"] = np.full(count, int(num_inputs), dtype=np.int32)
    opener = xrdroot.update if os.path.exists(str(filename)) else xrdroot.create
    with opener(str(filename)) as out:
        out.tree(str(key), {k: v.dtype for k, v in columns.items()}).extend(columns)


class RBDT:
    """``RBDT(key, filename)``: the saved trees, and ``Compute`` over rows of inputs."""

    def __init__(self, key: Any, filename: Any) -> None:
        import xrdroot

        with xrdroot.open_root(str(filename)) as source:
            tree = source[str(key)]
            found = tree.arrays(list(tree.keys()))
        self.nodes = {name: np.asarray(values) for name, values in found.items()}
        self.link = int(self.nodes["link"][0])
        self.classes = int(self.nodes["classes"][0])
        self.base = float(self.nodes["base_score"][0])
        self.roots = np.flatnonzero(np.r_[True, np.diff(self.nodes["tree"]) != 0])
        #: Each class's base margin, which its first tree's rows keep.
        self.bases = self.nodes["base_score"][self.roots[: self.classes]]

    def _margins(self, x: Any) -> Any:
        values = np.asarray(x, dtype=np.float32)
        nodes = self.nodes
        position = np.repeat(self.roots[:, None], len(values), axis=1)
        rows = np.broadcast_to(np.arange(len(values)), position.shape)
        while True:
            inner = nodes["feature"][position] >= 0
            if not inner.any():
                break
            here = position[inner]
            column = values[rows[inner], nodes["feature"][here]]
            below = column < nodes["threshold"][here].astype(np.float32)
            step = np.where(below, nodes["yes"][here], nodes["no"][here])
            position[inner] = np.where(np.isnan(column), nodes["missing"][here], step)
        leaves = nodes["value"][position]
        if self.link == 2:
            return np.stack([leaves[k :: self.classes].sum(axis=0) for k in range(self.classes)], 1)
        return leaves.sum(axis=0)[:, None]

    def Compute(self, x: Any) -> Any:
        """Each row's output: probabilities for a classifier, values for a regression."""
        from .tensor import RTensor, tensor_type
        from .tools import CxxVector

        single = not isinstance(x, (np.ndarray, RTensor)) and not np.ndim(x) > 1
        table = x.array if isinstance(x, RTensor) else np.asarray(x, dtype=np.float64)
        margins = self._margins(np.atleast_2d(table))
        if self.link == 1:
            base = np.log(self.base / (1.0 - self.base))
            found = 1.0 / (1.0 + np.exp(-(margins + base)))
        elif self.link == 2:
            margins = margins + self.bases
            shifted = np.exp(margins - margins.max(axis=1, keepdims=True))
            found = shifted / shifted.sum(axis=1, keepdims=True)
        else:
            found = margins + self.base
        found = found.astype(np.float32)
        if isinstance(x, RTensor):
            return tensor_type("float").wrap(found)
        return CxxVector(float(v) for v in found[0]) if single else found
