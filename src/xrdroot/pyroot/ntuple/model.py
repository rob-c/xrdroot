"""``RNTupleModel``: an RNTuple's fields, and the entries that hold one row of their values.

A model is made empty, ``RNTupleModel::Create()``, given fields by
``MakeField<T>(name)`` - which hands back the holder its default entry
keeps the field's value in - and is then handed to a writer, or to a
reader as the fields to read. A bare model, ``CreateBare()``, has no
default entry: whoever fills with it makes entries of their own.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...rntuple.writer import LowPrecisionSpec
from ..core.objects import typed
from ..stl import std
from .fields import FieldType, RFieldPtr, field_type

__all__ = ["RField", "RFieldBase", "RResult", "REntry", "RNTupleModel"]


class RResult:
    """``ROOT::RResult<T>``: what a maker that can fail hands back - the value, or why none.

    ``RFieldBase::Create`` returns one because the type name it is given may be
    one no dictionary describes; a macro takes the field out with ``Unwrap()``.
    xrdroot refuses such a name where it is given, in a sentence, rather than
    handing back a result that has no field in it, so every one made here does.
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    def Unwrap(self) -> Any:
        """``Unwrap()``: the value, as ROOT hands it over and throws where it has none."""
        return self._value

    def IsValid(self) -> bool:
        return True

    def GetError(self) -> str:
        return ""

    def Inspect(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"<RResult {self._value!r}>"


class RFieldBase:
    """``ROOT::RFieldBase``: a field made from its C++ type's name, as ``Create`` makes one.

    ROOT builds whatever type name it is given, reading a dictionary to learn
    what a class is made of, so a field made by name is the field a templated
    ``MakeField<T>`` makes. xrdroot's entries hold the types whose layout is
    beyond doubt - a number, ``std::string``, ``std::vector`` of a number - and
    a class type, which becomes a record of the members its dictionary lists, so
    that ``AtlasLikeDataVector<CustomStruct>`` is the field ROOT writes for it.
    :func:`~.fields.field_type` refuses what is neither, in a sentence.
    """

    @staticmethod
    def Create(name: Any, type_name: Any, description: Any = "") -> RResult:
        """``Create(name, typeName)``: the field of that type, in an ``RResult``."""
        return RResult(RField(name, type_name, description))


class RField(RFieldBase):
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

    def __len__(self) -> int:
        """Refuse a walk over a field's hierarchy, which is what ROOT's ``begin()`` is for.

        ``for (auto &field : someField)`` visits the field and every subfield under
        it, so that a class's members can be told to be stored another way. Nothing
        here walks a field: a class that a model holds is written from the members
        its dictionary declares, and a member is changed by declaring it so.
        """
        raise UnsupportedFeatureError(
            f"walking the subfields of the RNTuple field {self._name!r} is not supported: "
            f"xrdroot writes a class's members from its dictionary, and a member is given "
            f"another precision by being declared with it - as "
            f"LowPrecisionSpec.truncated(16) - rather than found by walking the field"
        )

    def _low_precision(self, precision: LowPrecisionSpec) -> None:
        """Store this field in fewer bits than its type holds, as ROOT's columns do.

        Only a field holding one ``float`` or one ``double`` has a precision to give
        up - a vector's items are a field of their own, one lower, which ROOT reaches
        by walking the vector's field, and nothing here walks one.
        """
        if not _stores_one_float(self._type):
            raise UnsupportedFeatureError(
                f"the RNTuple field {self._name!r} is a {self._type.cxx}, and only a field "
                f"holding one float or one double has fewer bits to be stored in; a "
                f"vector's items are a field of their own, one lower"
            )
        self._type.precision = precision

    def SetHalfPrecision(self) -> None:
        """``SetHalfPrecision()``: an IEEE half, 16 bits where a float has 32."""
        self._low_precision(LowPrecisionSpec.half(self._type.dtype))

    def SetTruncated(self, bits: Any) -> None:
        """``SetTruncated(bits)``: the value's top ``bits``, every one below dropped."""
        self._low_precision(LowPrecisionSpec.truncated(int(bits), self._type.dtype))

    def SetQuantized(self, *args: Any) -> None:
        """``SetQuantized(low, high, bits)``: a step between ``low`` and ``high``.

        ROOT 6.40 spells this ``SetQuantized(min, max, nBits)``, replacing it with
        ``SetQuantized(nBits, {min, max})``; both are taken here.
        """
        low, high, bits = _quantized_arguments(args)
        self._low_precision(LowPrecisionSpec.quantized(bits, low, high, self._type.dtype))


def _stores_one_float(kind: FieldType) -> bool:
    """Whether a field holds one float or one double - the only field with bits to give up."""
    if kind.vector or kind.record or kind.collection:
        return False
    return kind.dtype is not None and kind.dtype.name in ("float32", "float64")


def _quantized_arguments(given: tuple[Any, ...]) -> tuple[float, float, int]:
    """The range and the bits of a ``SetQuantized`` call, in either spelling ROOT has."""
    if len(given) == 3:
        low, high, bits = given
    elif len(given) == 2:
        bits, (low, high) = given
    else:
        raise TypeError(
            "SetQuantized takes the range and the bits - SetQuantized(min, max, nBits), or "
            "SetQuantized(nBits, (min, max))"
        )
    return float(low), float(high), int(bits)


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

    def __getitem__(self, name: Any) -> Any:
        """``entry[name]``: the field's value - what PyROOT's ``GetPtr<T>(name)`` hands back."""
        held = self._held(name)
        return held.value if isinstance(held, RFieldPtr) else held

    def __setitem__(self, name: Any, value: Any) -> None:
        """``entry[name] = value``: the value into the holder, typed as the field is.

        The field's type decides what may be stored: a number for a number,
        a ``str`` for a ``std::string``, a sequence of numbers for a vector.
        Anything else is a ``TypeError``, as storing through a
        ``shared_ptr<T>`` of another type is in cppyy.
        """
        held, field = self._held(name), self._fields[str(name)]
        if field.kind().record:
            held.update(_stored(field, value))
        elif field.kind().collection:
            _fill(held, field, value)
        elif isinstance(held, RFieldPtr):
            held.value = _stored(field, value)
        else:
            held.assign(value)


def _stored(field: RField, value: Any) -> Any:
    """``value`` as the field's own type, or a ``TypeError`` naming both."""
    kind = field.kind()
    where = f"the RNTuple field {field.GetFieldName()!r} is a {field.GetTypeName()}"
    if kind.record:
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}, so it holds a dict of its members, not a "
                            f"{type(value).__name__}")  # fmt: skip
        return dict(value)
    if kind.bits:
        return _bitset_value(kind.bits, value)
    if kind.dtype is None:
        if not isinstance(value, str):
            raise TypeError(f"{where}, so it holds a str, not a {type(value).__name__}")
        return str(value)
    whole = isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
    if not (whole or isinstance(value, (float, np.floating))):
        raise TypeError(f"{where}, so it holds a number, not a {type(value).__name__}")
    return kind.dtype.type(value).item()


def _fill(held: Any, field: RField, value: Any) -> None:
    """A set's or a map's holder given the items an entry is to hold.

    C++ has no assignment of a list to a container either, so an entry is filled
    the way a macro fills one: what the holder held is cleared, then an item is
    inserted - for a map a key, and the value kept with it.
    """
    kind = field.kind()
    assert kind.item is not None
    where = f"the RNTuple field {field.GetFieldName()!r} is a {field.GetTypeName()}"
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{where}, so it holds the items of a list, not a {type(value).__name__}"
        )
    pairs, keyed = (value.items() if isinstance(value, Mapping) else value), kind.item.record
    held.clear()
    for item in pairs:
        if not keyed:
            held.insert(item)
            continue
        key, kept = item
        if hasattr(held, "__setitem__"):  # a map: the value kept under its key
            held[key] = kept
        else:  # a multimap: the pair inserted, so that a key may come more than once
            held.insert((key, kept))


def _bitset_value(size: int, value: Any) -> Any:
    """A ``std::bitset`` of ``size`` bits from a number, or from another bitset's bits."""
    handed = getattr(value, "to_ulong", None)
    return std.bitset[size](value if handed is None else handed())


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

    def _handed_over(self) -> RNTupleModel:
        """The model a writer or a reader holds, cloned from the one handed in.

        ROOT's Python layer clones the model it is given, so that the caller's
        own stays open to a ``MakeField`` afterwards - and it can do that
        without losing anything there, because a Python model has no default
        entry. Here one does, since it is what a translated macro fills
        through, so the clone keeps the caller's default entry: ``Fill()``
        with no entry writes what the caller put in it.
        """
        made = self.Clone()
        made._entry = self._entry
        return made
