"""ROOT's RNTuple API: models, writers and readers, as macros and PyROOT scripts use them.

``ROOT::RNTupleModel``, ``RNTupleWriter``, ``RNTupleParallelWriter`` and
``RNTupleReader`` over :mod:`xrdroot.rntuple`, which reads and writes the
format: a model's fields and the entry holding their values, filled an
entry at a time and gathered a cluster at a time; read back by entry, by
view, and printed as ROOT prints them.
"""

from __future__ import annotations

from .fields import RFieldPtr
from .model import REntry, RField, RNTupleModel
from .processor import RNTupleOpenSpec, RNTupleProcessor
from .reading import ENTupleInfo, RNTupleDescriptor, RNTupleReader, RNTupleView
from .views import (
    RNTupleCollectionView,
    RNTupleLocalRange,
    RNTupleReadOptions,
    kInvalidDescriptorId,
)
from .writing import RNTupleFillContext, RNTupleParallelWriter, RNTupleWriteOptions, RNTupleWriter

__all__ = [
    "RNTupleModel",
    "REntry",
    "RField",
    "RFieldPtr",
    "RNTupleWriter",
    "RNTupleWriteOptions",
    "RNTupleParallelWriter",
    "RNTupleFillContext",
    "RNTupleReader",
    "RNTupleView",
    "RNTupleDescriptor",
    "ENTupleInfo",
    "RNTupleOpenSpec",
    "RNTupleProcessor",
    "RNTupleReadOptions",
    "RNTupleCollectionView",
    "RNTupleLocalRange",
    "kInvalidDescriptorId",
    "NTupleSize_t",
    "kInvalidNTupleIndex",
]

#: ``ROOT::NTupleSize_t``: an entry's number, which is a Python ``int``.
NTupleSize_t = int
#: ``ROOT::kInvalidNTupleIndex``: the number no entry has.
kInvalidNTupleIndex = 2**64 - 1
