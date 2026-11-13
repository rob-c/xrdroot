"""``ROOT.Experimental.ML.RDataLoader``: an ``RDataFrame``'s entries as batches for training.

The frame's columns - each vector column spread over ``max_vec_sizes``
columns ``name_0``, ``name_1``..., padded with zeros - are read once into a
table of ``float32``; ``train_test_split(test_size)`` keeps the last share
for validation, and each side's ``as_numpy()`` yields its batches, in a
fresh shuffled order each epoch when ``shuffle``, the short last batch
dropped when ``drop_remainder``. A ``target`` (and ``weights``) come as
batches of their own beside the inputs.

A list of frames is read as one long frame, unless ``load_eager``: then it
is split frame by frame - each keeps its own share for validation, after
shuffling its entries when ``shuffle`` - and the sides are joined in the
list's order. Two frames can be balanced: with
``sampling_type="undersampling"`` the larger (majority) one is cut to
``len(minority) / sampling_ratio`` entries, refused when it has too few
unless ``replacement`` lets it repeat; with ``"oversampling"`` the smaller
one is repeated to ``sampling_ratio * len(majority)`` entries. Sampled frames
are split frame by frame, eagerly or not. The sampled
frame comes first, as ROOT's loader gives it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

__all__ = ["RDataLoader"]

#: The ways two frames' entries are balanced against each other.
SAMPLINGS = ("undersampling", "oversampling")


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


def _inputs(
    found: dict[str, Any], names: list[str], sizes: dict[str, Any]
) -> tuple[Any, list[str]]:
    """The input columns as one table, and its columns' names - a vector's spread over several."""
    parts, columns = [], []
    for name in names:
        if name in sizes:
            parts.append(_spread(found[name], int(sizes[name])))
            columns += [f"{name}_{i}" for i in range(int(sizes[name]))]
        else:
            parts.append(np.asarray(found[name], dtype=np.float32)[:, None])
            columns.append(name)
    return (np.hstack(parts) if parts else np.zeros((0, 0), dtype=np.float32)), columns


def _stacked(found: dict[str, Any], names: list[str]) -> Any:
    """The named columns side by side as ``float32``, or ``None`` for no names."""
    if not names:
        return None
    return np.column_stack([np.asarray(found[name], dtype=np.float32) for name in names])


def _joined(tables: list[Any]) -> Any:
    """The frames' tables one after another, or ``None`` when they have none."""
    return None if tables[0] is None else np.concatenate(tables)


def _count(value: float) -> int:
    """How many entries a ratio asks for - ``3 / 0.2`` is 15, though a float makes it 15.000…2."""
    return int(round(value, 9))


def _cycled(rows: Any, count: int) -> Any:
    """``count`` of ``rows``, from the first, starting over when they run out."""
    return np.take(rows, np.arange(count), mode="wrap")


class _Batches:
    """Entries batched as the loader says: ``as_numpy`` and how many batches it gives."""

    loader: RDataLoader

    @property
    def rows(self) -> Any:
        raise NotImplementedError

    @property
    def num_batches(self) -> int:
        count, size = len(self.rows), self.loader.batch_size
        return count // size if self.loader.drop_remainder else -(-count // size)

    @property
    def last_batch_no_of_rows(self) -> int:
        """The rows in the short last batch - none when it is dropped, or there is none."""
        return 0 if self.loader.drop_remainder else len(self.rows) % self.loader.batch_size

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


class _Split(_Batches):
    """One side of the split: its rows, and the batches ``as_numpy`` makes of them."""

    def __init__(self, loader: RDataLoader, rows: Any) -> None:
        self.loader, self._rows = loader, rows
        self.columns = loader.columns

    @property
    def rows(self) -> Any:
        return self._rows


class RDataLoader(_Batches):
    """``RDataLoader(rdataframe, batch_size, target, weights, max_vec_sizes, ...)``.

    ``rdataframe`` may be a list of frames, each split on its own; batching the
    loader itself, unsplit, gives every entry.
    """

    #: The rows of each frame - or of all of them, read as one - that a split divides.
    parts: list[Any]

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
        **options: Any,
    ) -> None:
        self.loader = self
        self.batch_size, self.shuffle = int(batch_size), bool(shuffle)
        self.drop_remainder = bool(drop_remainder)
        self.random = np.random.default_rng(set_seed or None)
        frames = list(rdataframe) if isinstance(rdataframe, (list, tuple)) else [rdataframe]
        self._sampling(len(frames), options)
        self._load([_read(frame, target, weights, dict(max_vec_sizes or {})) for frame in frames])
        if self.sampling is None and not options.get("load_eager", False):
            # Read lazily, ROOT's loader streams the frames as one, and splits them as one.
            self.parts = [np.concatenate(self.parts)]

    def _load(self, read: list[tuple[Any, Any, Any, list[str]]]) -> None:
        """The frames' tables one after another, and the rows each frame has among them."""
        self.columns = read[0][3]
        self.inputs = np.concatenate([inputs for inputs, _, _, _ in read])
        self.targets = _joined([targets for _, targets, _, _ in read])
        self.weights = _joined([weighted for _, _, weighted, _ in read])
        ends = np.cumsum([len(inputs) for inputs, _, _, _ in read])
        self.parts = [
            np.arange(end - len(part[0]), end) for end, part in zip(ends, read, strict=True)
        ]

    def _sampling(self, frames: int, options: dict[str, Any]) -> None:
        """How two frames are balanced: ``sampling_type``, ``sampling_ratio``, ``replacement``."""
        self.sampling = options.get("sampling_type")
        self.ratio = float(options.get("sampling_ratio", 1.0))
        self.replacement = bool(options.get("replacement", False))
        if self.sampling is None:
            return
        if self.sampling not in SAMPLINGS:
            raise ValueError(
                f"sampling_type {self.sampling!r} is neither 'undersampling' nor 'oversampling', "
                "the two ways the loader balances its frames"
            )
        if frames != 2:
            raise ValueError(
                f"sampling balances two frames, a majority and a minority, and was given {frames}"
            )

    @property
    def rows(self) -> Any:
        return self._sampled(list(self.parts))

    def train_test_split(self, test_size: float = 0.0) -> tuple[_Split, _Split]:
        trains, tests = [], []
        for part in self.parts:
            rows = self.random.permutation(part) if self.shuffle else part
            kept = len(rows) - int(len(rows) * float(test_size))
            trains.append(rows[:kept])
            tests.append(rows[kept:])
        return _Split(self, self._sampled(trains)), _Split(self, self._sampled(tests))

    def _sampled(self, parts: list[Any]) -> Any:
        """The frames' rows joined - balanced first, when two frames are sampled."""
        if self.sampling is None:
            return np.concatenate(parts)
        minor, major = sorted(range(2), key=lambda i: len(self.parts[i]))
        if self.sampling == "oversampling":
            count = _count(self.ratio * len(parts[major]))
            return np.concatenate([_cycled(parts[minor], count), parts[major]])
        count = _count(len(parts[minor]) / self.ratio)
        if count > len(parts[major]) and not self.replacement:
            raise ValueError(
                "The sampling_ratio is too low: not enough entries in the majority class to "
                f"sample from. {count} are needed and {len(parts[major])} are there; raise "
                "the ratio, or sample with replacement=True"
            )
        return np.concatenate([_cycled(parts[major], count), parts[minor]])


def _read(
    frame: Any, target: Any, weights: Any, sizes: dict[str, Any]
) -> tuple[Any, Any, Any, list[str]]:
    """One frame's inputs, targets and weights as tables, and its inputs' column names."""
    targets, weighted = _names(target), _names(weights)
    names = [str(c) for c in frame.GetColumnNames()]
    found = frame.AsNumpy(names)
    inputs = [name for name in names if name not in targets and name not in weighted]
    table, columns = _inputs(found, inputs, sizes)
    return table, _stacked(found, targets), _stacked(found, weighted), columns
