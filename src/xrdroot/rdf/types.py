"""What ``GetColumnType`` and ``Describe`` say: a column's C++ type, and a frame in words.

A column's type is found the way C++ would find it - from what it is - by
computing it for no entries at all: a column of the data read over an empty
range, which reads nothing, and a defined column computed from those, which
gives its type without its values. A column of a tree is named with ROOT's
own typedefs, ``Float_t`` and ``Int_t``, as ROOT names a branch's; a defined
one with the C++ it would have been declared as, ``float`` and ``int``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..tree import Jagged
from .graph import Definition, EntryColumn, SlotColumn, SourceColumn

if TYPE_CHECKING:
    from .frame import RDataFrame

__all__ = ["cxx_type", "describe"]

#: The C++ name of each NumPy type, as a defined column would be declared.
CXX = {
    "bool": "bool",
    "int8": "char",
    "uint8": "unsigned char",
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "Long64_t",
    "uint64": "ULong64_t",
    "float32": "float",
    "float64": "double",
}

#: ROOT's typedef for each NumPy type, as a tree's branch is described.
TYPEDEFS = {
    "bool": "Bool_t",
    "int8": "Char_t",
    "uint8": "UChar_t",
    "int16": "Short_t",
    "uint16": "UShort_t",
    "int32": "Int_t",
    "uint32": "UInt_t",
    "int64": "Long64_t",
    "uint64": "ULong64_t",
    "float32": "Float_t",
    "float64": "Double_t",
}


def _element(dtype: Any, names: dict[str, str]) -> str:
    dtype = np.dtype(dtype)
    if dtype.kind in "US":
        return "std::string"
    return str(names.get(dtype.name, dtype.name))


#: The special columns' types, as ROOT declares them.
SPECIAL = {EntryColumn: "ULong64_t", SlotColumn: "unsigned int"}


def _shaped(empty: Any, names: dict[str, str]) -> str:
    """The type of what a column's values are: a number, a collection, a string."""
    if isinstance(empty, tuple):
        inner = _element(empty[0].content.dtype, names)
        return f"ROOT::VecOps::RVec<ROOT::VecOps::RVec<{inner}>>"
    if isinstance(empty, Jagged):
        return f"ROOT::VecOps::RVec<{_element(empty.content.dtype, names)}>"
    if isinstance(empty, np.ndarray):
        inner = _element(empty.dtype, names)
        return inner if empty.ndim == 1 else f"ROOT::VecOps::RVec<{inner}>"
    return "std::string" if isinstance(empty, list) else type(empty).__name__


def cxx_type(definition: Definition, empty: Any) -> str:
    """The C++ type of a column, from its values for no entries."""
    special = SPECIAL.get(type(definition))
    if special is not None:
        return special
    return _shaped(empty, TYPEDEFS if isinstance(definition, SourceColumn) else CXX)


def describe(frame: RDataFrame) -> str:
    """ROOT's ``Describe``: the data, a few numbers about the frame, and every column."""
    names = frame.GetColumnNames()
    defined = frame.GetDefinedColumnNames()
    lines = [
        f"Dataframe from {frame._graph.source.describe()}",
        "",
        f"{'Property':<24}Value",
        f"{'--------':<24}-----",
        f"{'Columns in total':<24}{len(names)}",
        f"{'Columns from defines':<24}{len(defined)}",
        f"{'Event loops run':<24}{frame.GetNRuns()}",
        f"{'Processing slots':<24}{frame.GetNSlots()}",
        "",
    ]
    types = [frame.GetColumnType(name) for name in names]
    width = max([len("Column"), *map(len, names)]) + 2
    wide = max([len("Type"), *map(len, types)]) + 2
    lines += [
        f"{'Column':<{width}}{'Type':<{wide}}Origin",
        f"{'------':<{width}}{'----':<{wide}}------",
    ]
    for name, kind in zip(names, types):
        lines.append(f"{name:<{width}}{kind:<{wide}}{frame._columns[name].origin}")
    return "\n".join(lines)
