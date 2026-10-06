"""Writing an RNTuple from a model: ``RNTupleWriter``, its options, and filling from threads.

A writer takes a model, makes the RNTuple in a file - a new one,
``Recreate``, or an open ``TFile``, ``Append`` - and each ``Fill`` copies
the entry's values, the default entry's unless another is given, into a
batch that goes to :class:`xrdroot.rntuple.WritableRNTuple` a cluster's
worth at a time. When the writer is let go - ``reset()``, the end of the
function that made it, or the end of the program - what is left is
written, and a file the writer opened is closed, as ROOT's destructor does.

``RNTupleParallelWriter`` hands each thread a fill context of its own,
whose entries gather apart and join the RNTuple a cluster at a time,
under the writer's lock; a context that stages its clusters joins them
only when told to commit, which is how a program decides their order.
"""

from __future__ import annotations

import atexit
import threading
import weakref
from typing import Any

import numpy as np

from .fields import RFieldPtr
from .model import REntry, RNTupleModel

__all__ = ["RNTupleWriteOptions", "RNTupleWriter", "RNTupleParallelWriter", "RNTupleFillContext"]

#: How many entries a writer gathers before it hands them to the file.
BATCH = 8192


class RNTupleWriteOptions:
    """``RNTupleWriteOptions``: compression, and how large clusters and pages grow."""

    def __init__(self) -> None:
        self._compression = 505
        self._cluster = 128 << 20
        self._page = 1 << 20

    def SetCompression(self, setting: Any, level: Any = None) -> None:
        self._compression = int(setting) if level is None else 100 * int(setting) + int(level)

    def GetCompression(self) -> int:
        return self._compression

    def SetApproxZippedClusterSize(self, size: Any) -> None:
        self._cluster = int(size)

    def GetApproxZippedClusterSize(self) -> int:
        return self._cluster

    def SetMaxUnzippedPageSize(self, size: Any) -> None:
        self._page = int(size)

    def GetMaxUnzippedPageSize(self) -> int:
        return self._page

    def SetUseBufferedWrite(self, used: Any) -> None:
        """``SetUseBufferedWrite``: pages are gathered a cluster at a time either way."""

    def SetEnablePageChecksums(self, enabled: Any) -> None:
        """``SetEnablePageChecksums``: every page is written with its checksum, as ROOT's are."""

    def file_options(self) -> dict[str, Any]:
        """What :func:`xrdroot.create` makes the file with: nothing compressed for a setting of
        0, else zlib - which every installation has, where ROOT's default zstd needs an extra -
        at the setting's level; the values read back are the same either way."""
        level = self._compression % 100
        return {"compression": "zlib" if level else None, "level": level or None}


def _snapshot(held: Any) -> Any:
    """A holder's value as it is now, copied so that refilling the holder leaves it be."""
    if isinstance(held, RFieldPtr):
        return held.value
    return np.array(held)


class _Batch:
    """Entries gathered, a list of values per field, until they go to the file."""

    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.columns: dict[str, list[Any]] = {name: [] for name in names}
        self.size = 0

    def add(self, entry: REntry) -> int:
        """One entry's values; the bytes they take, as ``Fill`` reports them."""
        values = entry.values()
        taken = 0
        for name in self._names:
            value = _snapshot(values[name])
            self.columns[name].append(value)
            taken += value.nbytes if isinstance(value, np.ndarray) else 8
        self.size += 1
        return taken

    def take(self) -> tuple[dict[str, list[Any]], int]:
        """The entries gathered, and how many; the batch is empty again."""
        taken = self.columns, self.size
        self.columns, self.size = {name: [] for name in self._names}, 0
        return taken


class _Sink:
    """The RNTuple being written, the lock fills share, and the file opened for it, if one was."""

    def __init__(self, model: RNTupleModel, name: Any, place: Any, options: Any) -> None:
        options = options if isinstance(options, RNTupleWriteOptions) else RNTupleWriteOptions()
        directory, self.file = _directory(place, options)
        model.Freeze()
        specs = {field: made.kind().spec for field, made in model.fields().items()}
        sizes = {"cluster_size": max(options.GetApproxZippedClusterSize(), 1),
                 "page_size": max(options.GetMaxUnzippedPageSize(), 1)}  # fmt: skip
        self.ntuple = directory.rntuple(str(name), specs, **sizes)
        self.lock = threading.Lock()
        self.names = list(specs)
        self.closed = False

    def commit(self, taken: tuple[dict[str, list[Any]], int]) -> None:
        """Entries joining the RNTuple, whole, after those any other thread is adding."""
        columns, count = taken
        if count:
            with self.lock:
                self.ntuple.extend(columns)

    def close(self) -> None:
        self.closed = True
        if self.file is not None:
            self.file.close()
            self.file = None


def _directory(place: Any, options: RNTupleWriteOptions) -> tuple[Any, Any]:
    """Where the RNTuple goes, and the file to close after it if the writer made one."""
    if isinstance(place, str) or hasattr(place, "__fspath__"):
        from ... import create

        made = create(str(place), **options.file_options())
        return made, made
    writable = getattr(place, "_writable", None)
    found = writable() if callable(writable) else place
    if found is None or not hasattr(found, "rntuple"):
        raise ValueError(f"{place!r} is not open for writing, so no RNTuple can be added to it: "
                         "open it with RECREATE, CREATE or UPDATE")  # fmt: skip
    return found, None


#: The writers not yet let go, which the end of the program finishes as ROOT's would be.
_WRITING: weakref.WeakSet[Any] = weakref.WeakSet()


@atexit.register
def _finish_all() -> None:
    for writer in list(_WRITING):
        writer.reset()


class _Filling:
    """What a writer and a fill context share: a model, and a batch of entries to hand on."""

    def __init__(self, model: RNTupleModel, sink: _Sink) -> None:
        self._model, self._sink = model, sink
        self._batch = _Batch(sink.names)

    def GetModel(self) -> RNTupleModel:
        return self._model

    def CreateEntry(self) -> REntry:
        return self._model.CreateEntry()

    def Fill(self, entry: REntry | None = None) -> int:
        """``Fill``: the entry's values - the default entry's when none is given - added."""
        taken = self._batch.add(self._model.GetDefaultEntry() if entry is None else entry)
        if self._batch.size >= BATCH:
            self.FlushCluster()
        return taken

    def FlushCluster(self) -> None:
        """``FlushCluster``: the entries gathered so far handed to the file."""
        self._sink.commit(self._batch.take())

    CommitCluster = FlushCluster

    def FlushColumns(self) -> None:
        """``FlushColumns``: entries are kept whole until their cluster is handed on."""


class RNTupleWriter(_Filling):
    """``ROOT::RNTupleWriter``: entries filled into an RNTuple of a new file or an open one."""

    def __init__(self, model: RNTupleModel, sink: _Sink) -> None:
        super().__init__(model, sink)
        _WRITING.add(self)

    @staticmethod
    def Recreate(model: RNTupleModel, name: Any, path: Any, options: Any = None) -> RNTupleWriter:
        """``Recreate(model, name, path)``: a new file at ``path`` holding the RNTuple."""
        return RNTupleWriter(model, _Sink(model, name, str(path), options))

    @staticmethod
    def Append(model: RNTupleModel, name: Any, file: Any, options: Any = None) -> RNTupleWriter:
        """``Append(model, name, file)``: the RNTuple in a ``TFile`` open for writing."""
        return RNTupleWriter(model, _Sink(model, name, file, options))

    def GetNEntries(self) -> int:
        return len(self._sink.ntuple) + self._batch.size

    def get(self) -> RNTupleWriter:
        """``unique_ptr::get()``: the writer itself, as every pointer is here."""
        return self

    def reset(self) -> None:
        """``reset()``: the writer let go - the rest written, its file closed."""
        if not self._sink.closed:
            self.FlushCluster()
            self._sink.close()

    def __del__(self) -> None:
        if hasattr(self, "_sink"):
            self.reset()


class RNTupleFillContext(_Filling):
    """One thread's filling of a parallel writer's RNTuple, its clusters joined whole."""

    def __init__(self, model: RNTupleModel, sink: _Sink) -> None:
        super().__init__(model, sink)
        self._staging = False
        self._staged: list[tuple[dict[str, list[Any]], int]] = []

    def EnableStagedClusterCommitting(self, enabled: Any = True) -> None:
        """Clusters are kept back, once flushed, until :meth:`CommitStagedClusters`."""
        self._staging = bool(enabled)

    def IsStagedClusterCommittingEnabled(self) -> bool:
        return self._staging

    def FlushCluster(self) -> None:
        if not self._staging:
            super().FlushCluster()
        elif self._batch.size:
            self._staged.append(self._batch.take())

    def CommitStagedClusters(self) -> None:
        """The clusters kept back joining the RNTuple, in the order they were flushed."""
        for staged in self._staged:
            self._sink.commit(staged)
        self._staged.clear()


class RNTupleParallelWriter:
    """``ROOT::RNTupleParallelWriter``: an RNTuple filled from several threads at once.

    Each thread asks for a fill context, :meth:`CreateFillContext`, and fills
    through it; what the contexts gather joins the RNTuple a cluster at a
    time, under one lock, so entries of one thread stay together.
    """

    def __init__(self, model: RNTupleModel, sink: _Sink) -> None:
        self._model, self._sink = model, sink
        self._contexts: list[RNTupleFillContext] = []
        self._lock = threading.Lock()
        _WRITING.add(self)

    @staticmethod
    def Recreate(
        model: RNTupleModel, name: Any, path: Any, options: Any = None
    ) -> RNTupleParallelWriter:
        return RNTupleParallelWriter(model, _Sink(model, name, str(path), options))

    @staticmethod
    def Append(
        model: RNTupleModel, name: Any, file: Any, options: Any = None
    ) -> RNTupleParallelWriter:
        return RNTupleParallelWriter(model, _Sink(model, name, file, options))

    def CreateFillContext(self) -> RNTupleFillContext:
        """A fill context for one thread: its own entries, its own clusters."""
        made = RNTupleFillContext(self._model, self._sink)
        with self._lock:
            self._contexts.append(made)
        return made

    def GetModel(self) -> RNTupleModel:
        return self._model

    def get(self) -> RNTupleParallelWriter:
        return self

    def reset(self) -> None:
        """The writer let go: every context's entries written, staged ones too, the file closed."""
        if self._sink.closed:
            return
        for context in self._contexts:
            context.FlushCluster()
            context.CommitStagedClusters()
        self._sink.close()

    def __del__(self) -> None:
        if hasattr(self, "_sink"):
            self.reset()
