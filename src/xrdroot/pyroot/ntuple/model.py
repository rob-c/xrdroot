"""``RNTupleModel``: an RNTuple's fields, and the entries that hold one row of their values.

A model is made empty, ``RNTupleModel::Create()``, given fields by
``MakeField<T>(name)`` - which hands back the holder its default entry
keeps the field's value in - and is then handed to a writer, or to a
reader as the fields to read. A bare model, ``CreateBare()``, has no
default entry: whoever fills with it makes entries of their own.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ...errors import UnsupportedFeatureError
from ..core.objects import typed
from .fields import FieldType, field_type

__all__ = ["RField", "REntry", "RNTupleModel"]


class RField:
    """``RField<T>``: a field's name and type, before a model holds it."""

    def __init__(self, name: Any, kind: Any = None, description: Any = "") -> None:
        self._name = str(name)
        self._type = kind if isinstance(kind, FieldType) else field_type(kind or self.kind_name)
        self._description = str(description)

    #: The C++ type an ``RField<T>`` made by ``RField[T]`` is of.
    kind_name = "double"

    def __class_getitem__(cls, kind: Any) -> type:
        return type(f"RField<{kind}>", (cls,), {"kind_name": str(kind)})

    def __repr__(self) -> str:
        return f"<RField {self._name!r} of {self._type.cxx}>"

    def GetFieldName(self) -> str:
        return self._name

    GetQualifiedFieldName = GetFieldName

    def GetTypeName(self) -> str:
        """The type as ROOT's files keep it: ``std::int32_t`` for ``int``."""
        return self._type.cxx

    def GetDescription(self) -> str:
        return self._description

    def SetDescription(self, description: Any) -> None:
        self._description = str(description)

    def Clone(self, name: Any) -> RField:
        """``Clone(newName)``: a field of the same type under another name."""
        return RField(name, self._type, self._description)

    def kind(self) -> FieldType:
        return self._type

    def _low_precision(self, *args: Any) -> None:
        raise UnsupportedFeatureError(
            "an RNTuple field stored at low precision - half-precision, truncated or quantized "
            "floats - is not supported: xrdroot's RNTuple writer stores a float in its own 32 "
            "bits, as ROOT does unless told otherwise.")

    SetHalfPrecision = SetTruncated = SetQuantized = _low_precision


class REntry:
    """``REntry``: one value per field - the holders a fill reads and a read writes into."""

    def __init__(self, fields: dict[str, RField]) -> None:
        self._values = {name: field.kind().holder() for name, field in fields.items()}
        self._fields = fields

    def __iter__(self) -> Iterator[RValue]:
        return (RValue(self._fields[name], held) for name, held in self._values.items())

    def _held(self, name: Any) -> Any:
        found = self._values.get(str(name))
        if found is None:
            raise KeyError(f"invalid field name: {name}; this entry has "
                           f"{', '.join(self._values) or 'no fields'}")  # fmt: skip
        return found

    @typed
    def GetPtr(self, kind: Any, name: Any) -> Any:
        """``GetPtr<T>(name)``: the holder of the field's value in this entry."""
        return self._held(name)

    @typed
    def BindValue(self, kind: Any, name: Any, value: Any) -> None:
        """``BindValue(name, ptr)``: the field's value is kept in ``ptr`` from now on."""
        self._held(name)
        self._values[str(name)] = value

    def values(self) -> dict[str, Any]:
        """The holders, by field name, as a writer reads them."""
        return self._values


class RValue:
    """One of an entry's values, as iterating the entry gives it: the field, and its holder."""

    def __init__(self, field: RField, held: Any) -> None:
        self._field, self._held = field, held

    def GetField(self) -> RField:
        return self._field

    @typed
    def GetPtr(self, kind: Any) -> Any:
        return self._held


class RNTupleModel:
    """``ROOT::RNTupleModel``: the fields, in order, and - unless it is bare - a default entry."""

    def __init__(self, bare: bool = False) -> None:
        self._fields: dict[str, RField] = {}
        self._entry: REntry | None = None if bare else REntry({})
        self._frozen = False

    @staticmethod
    def Create() -> RNTupleModel:
        return RNTupleModel()

    @staticmethod
    def CreateBare() -> RNTupleModel:
        """``CreateBare()``: a model with no default entry; fills bring their own."""
        return RNTupleModel(bare=True)

    def __repr__(self) -> str:
        return f"<RNTupleModel of {', '.join(self._fields) or 'no fields'}>"

    @typed
    def MakeField(self, kind: Any, name: Any, description: Any = "") -> Any:
        """``MakeField<T>(name)``: a new field, and the holder of its value in the default entry."""
        field = RField(name, kind, description)
        self.AddField(field)
        return None if self._entry is None else self._entry.GetPtr(field.GetFieldName())

    def AddField(self, field: RField) -> None:
        """``AddField``: a field made apart - an ``RField`` or a clone of another model's."""
        name = field.GetFieldName()
        if self._frozen:
            raise RuntimeError("invalid attempt to modify frozen model: a model a writer or a "
                               "reader holds has the fields it was given")  # fmt: skip
        if name in self._fields:
            raise ValueError(f"field name '{name}' already exists in NTuple model")
        self._fields[name] = field
        if self._entry is not None:
            self._entry._values[name] = field.kind().holder()
            self._entry._fields = self._fields

    def GetField(self, name: Any) -> RField:
        found = self._fields.get(str(name))
        if found is None:
            raise KeyError(f"invalid field: {name}")
        return found

    def GetFieldNames(self) -> list[str]:
        return list(self._fields)

    def fields(self) -> dict[str, RField]:
        return self._fields

    def GetDefaultEntry(self) -> REntry:
        if self._entry is None:
            raise RuntimeError("invalid attempt to use default entry of bare model: a bare "
                               "model's fills each bring an entry, CreateEntry()'s")  # fmt: skip
        return self._entry

    def CreateEntry(self) -> REntry:
        """``CreateEntry()``: an entry of fresh values, one per field."""
        return REntry(self._fields)

    CreateBareEntry = CreateEntry

    def Freeze(self) -> None:
        self._frozen = True

    def IsFrozen(self) -> bool:
        return self._frozen

    def Clone(self) -> RNTupleModel:
        made = RNTupleModel(bare=self._entry is None)
        for field in self._fields.values():
            made.AddField(field.Clone(field.GetFieldName()))
        return made
