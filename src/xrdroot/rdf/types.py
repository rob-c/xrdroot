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

__all__ = ["Description", "cxx_type", "describe"]

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


class Description(str):
    """``Describe``'s ``RDFDescription``: the data said in a line, then the frame's numbers and
    columns - the whole as the text, the line alone as :attr:`brief`."""

    brief: str

    def __new__(cls, brief: str, extended: str) -> Description:
        made = super().__new__(cls, f"{brief}\n\n{extended}")
        made.brief = brief
        return made

    def AsString(self, shortFormat: bool = False) -> str:
        return self.brief if shortFormat else str(self)

    def Print(self, shortFormat: bool = False) -> None:
        print(self.AsString(shortFormat))


#: ``Describe``'s table of the frame's numbers: each value ends this many characters in.
PROPERTY_WIDTH = 30
#: How far each column of the table of columns is past its longest entry.
COLUMN_GAP = 5


def describe(frame: RDataFrame) -> Description:
    """ROOT's ``Describe``: the data, a few numbers about the frame, and every column."""
    names = frame.GetColumnNames()
    numbers = [
        ("Columns in total", len(names)),
        ("Columns from defines", len(frame.GetDefinedColumnNames())),
        ("Event loops run", frame.GetNRuns()),
        ("Processing slots", frame.GetNSlots()),
    ]
    lines = [f"{'Property':<24}Value", f"{'--------':<24}-----"]
    lines += [f"{label}{value:>{PROPERTY_WIDTH - len(label)}}" for label, value in numbers]
    types = [frame.GetColumnType(name) for name in names]
    width = max([len("Column"), *map(len, names)]) + COLUMN_GAP
    wide = max([len("Type"), *map(len, types)]) + COLUMN_GAP
    lines += [
        "",
        f"{'Column':<{width}}{'Type':<{wide}}Origin",
        f"{'------':<{width}}{'----':<{wide}}------",
    ]
    for name, kind in zip(names, types, strict=False):
        lines.append(f"{name:<{width}}{kind:<{wide}}{frame._columns[name].origin}")
    return Description(frame._graph.source.root_description(), "\n".join(lines))
