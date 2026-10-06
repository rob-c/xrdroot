"""Reading an RNTuple an entry at a time: ``RNTupleReader``, its views, and what it prints.

A reader opens the RNTuple of a file by name - with a model saying which
fields to read, or with the model of every field it can hold - and
``LoadEntry(i)`` puts entry ``i``'s values in the model's default entry, in
the holders ``MakeField`` handed out. A view, ``GetView<T>(name)``, reads
one field by entry number, or a vector's items by their own number
(``"vpx._0"``). The values come from :mod:`xrdroot.rntuple`, a field's
whole column at a time, the first time a field is asked for.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from .fields import RFieldPtr, field_type, typed
from .model import RField, RNTupleModel
from .printing import info_text, json_text

__all__ = ["RNTupleReader", "RNTupleView", "ENTupleInfo", "RNTupleDescriptor"]


class ENTupleInfo:
    """``ENTupleInfo``: what ``PrintInfo`` prints."""

    kSummary = 0
    kStorageDetails = 1
    kMetrics = 2


def _opened(args: tuple[Any, ...]) -> tuple[RNTupleModel | None, Any, Any]:
    """The model, the RNTuple's name and where it is, from ``Open``'s arguments."""
    model = args[0] if args and isinstance(args[0], RNTupleModel) else None
    rest = args[1:] if model is not None else args
    if rest and hasattr(rest[0], "num_entries"):  # file->Get<ROOT::RNTuple>(name)
        return model, rest[0], None
    if len(rest) < 2:
        raise TypeError("RNTupleReader::Open takes a model or not, then the RNTuple's name and "
                        "the file it is in - or an RNTuple a file gave")  # fmt: skip
    return model, rest[0], rest[1]


def _ntuple(name: Any, place: Any) -> tuple[Any, Any]:
    """The RNTuple, and the file opened to read it if this opened one."""
    if place is None:
        return name, None
    reading = getattr(place, "_reading", None)
    if reading is not None:  # an open TFile
        return reading[str(name)], None
    from ... import open_root

    opened = open_root(str(place))
    return opened[str(name)], opened


class RNTupleDescriptor:
    """``RNTupleDescriptor``: the RNTuple's name, description and size, as its header says."""

    def __init__(self, ntuple: Any) -> None:
        self._ntuple = ntuple

    def GetName(self) -> str:
        return str(self._ntuple.name)

    def GetDescription(self) -> str:
        return str(self._ntuple.description)

    def GetNEntries(self) -> int:
        return len(self._ntuple)

    def GetNClusters(self) -> int:
        return int(self._ntuple.num_clusters)

    def GetNFields(self) -> int:
        """Every field, the hidden top one ROOT counts included."""
        return len(self._ntuple._store.schema.fields) + 1


def _model_of(ntuple: Any) -> RNTupleModel:
    """A model of every field this layer can hold an entry's value of."""
    model = RNTupleModel()
    for name, cxx in ntuple.cxx_types().items():
        try:
            kind = field_type(cxx)
        except UnsupportedFeatureError:
            continue  # a record or a nested collection: read through views of its parts
        model.AddField(RField(name, kind))
    return model


def _load(held: Any, value: Any) -> None:
    """Put one entry's value in its holder: a number or string, or a vector's items."""
    if isinstance(held, RFieldPtr):
        held.value = value.item() if isinstance(value, np.generic) else value
    else:
        held.assign(value)


class RNTupleReader:
    """``ROOT::RNTupleReader``: an RNTuple's entries, loaded one at a time into a model's entry."""

    def __init__(self, ntuple: Any, model: RNTupleModel | None, file: Any = None) -> None:
        self._ntuple, self._file = ntuple, file
        self._model = model if model is not None else _model_of(ntuple)
        self._model.Freeze()
        self._columns: dict[str, Any] = {}

    @staticmethod
    def Open(*args: Any) -> RNTupleReader:
        """``Open([model,] name, path)``, or ``Open([model,] ntuple)`` of one a file gave."""
        model, name, place = _opened(args)
        ntuple, file = _ntuple(name, place)
        return RNTupleReader(ntuple, model, file)

    def __repr__(self) -> str:
        return f"<RNTupleReader of {self._ntuple.name!r}, {len(self._ntuple)} entries>"

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self._ntuple)))

    def GetEntryRange(self) -> range:
        return range(len(self._ntuple))

    def GetNEntries(self) -> int:
        return len(self._ntuple)

    def GetModel(self) -> RNTupleModel:
        return self._model

    def GetDescriptor(self) -> RNTupleDescriptor:
        return RNTupleDescriptor(self._ntuple)

    def get(self) -> RNTupleReader:
        return self

    def __del__(self) -> None:
        """The file this reader opened is closed with it, as ROOT's reader closes its own."""
        if getattr(self, "_file", None) is not None:
            self._file.close()

    def column(self, name: str) -> Any:
        """A field's values, every entry's, read the first time they are asked for."""
        if name not in self._columns:
            head, _, rest = name.rpartition(".")
            if rest == "_0" and head:  # a vector's items, numbered on their own
                self._columns[name] = self.column(head).flat
            else:
                self._columns[name] = self._ntuple[name].array()
        return self._columns[name]

    def LoadEntry(self, entry: Any, into: Any = None) -> None:
        """``LoadEntry(i)``: entry ``i``'s values put in the default entry, or in ``into``."""
        target = self._model.GetDefaultEntry() if into is None else into
        for name, held in target.values().items():
            _load(held, self.column(name)[int(entry)])

    @typed
    def GetView(self, kind: Any, name: Any) -> RNTupleView:
        """``GetView<T>(name)``: one field's values by number, ``view(i)``."""
        return RNTupleView(self, str(name))

    def GetCollectionView(self, name: Any) -> Any:
        """``GetCollectionView(name)``: a vector field's sizes, and views of its items."""
        from .views import RNTupleCollectionView

        return RNTupleCollectionView(self, str(name))

    def PrintInfo(self, what: Any = ENTupleInfo.kSummary) -> None:
        """``PrintInfo``: the summary box of the RNTuple's fields, as ROOT draws it."""
        if int(what) != ENTupleInfo.kSummary:
            raise UnsupportedFeatureError(
                "RNTupleReader::PrintInfo of the storage details or the metrics is not supported: "
                "they are the sizes ROOT's own writer gave the pages and the timings of its reads, "
                "which a reading here neither knows nor makes; the summary, kSummary, is printed.")
        print(info_text(self._ntuple), end="")

    def Show(self, entry: Any) -> None:
        """``Show(i)``: entry ``i`` in JSON, as ROOT writes it - every field of the RNTuple
        this layer can hold, whichever the reader's model reads."""
        names = _model_of(self._ntuple).GetFieldNames()
        values = {name: self.column(name)[int(entry)] for name in names}
        print(json_text(values), end="")


class RNTupleView:
    """``RNTupleView<T>``: a field's values by entry number, or a vector's items by theirs."""

    def __init__(self, reader: RNTupleReader, name: str) -> None:
        self._reader, self._name = reader, name

    def __class_getitem__(cls, kind: Any) -> Any:
        """``RNTupleView<T>(view)``: a view of the same values - the view itself."""
        return _same_view

    def __call__(self, index: Any) -> Any:
        found = self._reader.column(self._name)[int(index)]
        return found.item() if isinstance(found, np.generic) else found

    def GetFieldRange(self) -> range:
        """Every value's number: the entries, or a vector's items, all of them."""
        return range(len(self._reader.column(self._name)))


def _same_view(view: RNTupleView) -> RNTupleView:
    return view
