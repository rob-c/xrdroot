"""``ROOT.Experimental.ML.RDataLoader``: an ``RDataFrame``'s entries as batches for training.

The frame's columns - each vector column spread over ``max_vec_sizes``
columns ``name_0``, ``name_1``..., padded with zeros - are read once into a
table of ``float32``; ``train_test_split(test_size)`` keeps the last share
for validation, and each side's ``as_numpy()`` yields its batches, in a
fresh shuffled order each epoch when ``shuffle``, the short last batch
dropped when ``drop_remainder``. A ``target`` (and ``weights``) come as
batches of their own beside the inputs.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

__all__ = ["RDataLoader"]


def _names(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value]


def _spread(column: Any, width: int) -> Any:
    """A vector column as ``width`` columns, each row padded with zeros or cut short."""
    made = np.zeros((len(column), width), dtype=np.float32)
    for row, values in enumerate(column):
        found = np.asarray(values, dtype=np.float32)[:width]
        made[row, : len(found)] = found
    return made


class _Split:
    """One side of the split: its rows, and the batches ``as_numpy`` makes of them."""

    def __init__(self, loader: RDataLoader, rows: Any) -> None:
        self.loader, self.rows = loader, rows
        self.columns = loader.columns

    def as_numpy(self) -> Iterator[Any]:
        loader = self.loader
        order = self.rows
        if loader.shuffle:
            order = loader.random.permutation(order)
        size = loader.batch_size
        end = len(order) - len(order) % size if loader.drop_remainder else len(order)
        for start in range(0, end, size):
            picked = order[start : start + size]
            parts = [loader.inputs[picked]]
            if loader.targets is not None:
                parts.append(loader.targets[picked])
            if loader.weights is not None:
                parts.append(loader.weights[picked])
            yield parts[0] if len(parts) == 1 else tuple(parts)


class RDataLoader:
    """``RDataLoader(rdataframe, batch_size, target=None, weights=None, max_vec_sizes=None, ...)``."""

    def __init__(
        self,
        rdataframe: Any,
        batch_size: int,
        target: Any = None,
        weights: Any = None,
        max_vec_sizes: Any = None,
        shuffle: bool = True,
        drop_remainder: bool = True,
        set_seed: int = 0,
        **_: Any,
    ) -> None:
        self.batch_size, self.shuffle = int(batch_size), bool(shuffle)
        self.drop_remainder = bool(drop_remainder)
        self.random = np.random.default_rng(set_seed or None)
        sizes = dict(max_vec_sizes or {})
        targets, weighted = _names(target), _names(weights)
        names = [str(c) for c in rdataframe.GetColumnNames()]
        found = rdataframe.AsNumpy(names)
        inputs, self.columns = [], []
        for name in names:
            if name in targets or name in weighted:
                continue
            if name in sizes:
                inputs.append(_spread(found[name], int(sizes[name])))
                self.columns += [f"{name}_{i}" for i in range(int(sizes[name]))]
            else:
                inputs.append(np.asarray(found[name], dtype=np.float32)[:, None])
                self.columns.append(name)
        self.inputs = np.hstack(inputs) if inputs else np.zeros((0, 0), dtype=np.float32)
        self.targets = (
            np.column_stack([np.asarray(found[t], dtype=np.float32) for t in targets])
            if targets
            else None
        )
        self.weights = (
            np.column_stack([np.asarray(found[w], dtype=np.float32) for w in weighted])
            if weighted
            else None
        )

    def train_test_split(self, test_size: float = 0.0) -> tuple[_Split, _Split]:
        count = len(self.inputs)
        held = int(count * float(test_size))
        rows = np.arange(count)
        return _Split(self, rows[: count - held]), _Split(self, rows[count - held :])
