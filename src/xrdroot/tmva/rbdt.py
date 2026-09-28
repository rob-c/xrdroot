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


def _base_score(config: dict[str, Any]) -> float:
    text = str(config["learner"]["learner_model_param"].get("base_score", "0.5"))
    return float(text.strip("[]").split(",")[0])


def _nodes(dump: list[str]) -> dict[str, list[Any]]:
    """Every node of every tree, a row each, children as rows of the same table."""
    rows: dict[str, list[Any]] = {
        k: [] for k in ("tree", "feature", "threshold", "yes", "no", "missing", "value")
    }
    for index, text in enumerate(dump):
        stack = [json.loads(text)]
        start = len(rows["tree"])
        ids: dict[int, int] = {}
        order = []
        while stack:
            node = stack.pop()
            ids[int(node["nodeid"])] = start + len(order)
            order.append(node)
            stack.extend(reversed(node.get("children", [])))
        for node in order:
            leaf = "leaf" in node
            rows["tree"].append(index)
            rows["feature"].append(-1 if leaf else int(str(node["split"]).lstrip("f")))
            rows["threshold"].append(0.0 if leaf else float(node["split_condition"]))
            rows["yes"].append(-1 if leaf else ids[int(node["yes"])])
            rows["no"].append(-1 if leaf else ids[int(node["no"])])
            rows["missing"].append(-1 if leaf else ids[int(node["missing"])])
            rows["value"].append(float(node["leaf"]) if leaf else 0.0)
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
    columns["base_score"] = np.full(count, _base_score(config))
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
        """The model's output for each row: probabilities for a classifier, values for a regression."""
        from .tensor import RTensor
        from .tools import CxxVector

        single = not isinstance(x, (np.ndarray, RTensor)) and not np.ndim(x) > 1
        table = x.array if isinstance(x, RTensor) else np.asarray(x, dtype=np.float64)
        margins = self._margins(np.atleast_2d(table))
        if self.link == 1:
            base = np.log(self.base / (1.0 - self.base))
            found = 1.0 / (1.0 + np.exp(-(margins + base)))
        elif self.link == 2:
            shifted = np.exp(margins - margins.max(axis=1, keepdims=True))
            found = shifted / shifted.sum(axis=1, keepdims=True)
        else:
            found = margins + self.base
        found = found.astype(np.float32)
        if isinstance(x, RTensor):
            return RTensor["float"].wrap(found)
        return CxxVector(float(v) for v in found[0]) if single else found
