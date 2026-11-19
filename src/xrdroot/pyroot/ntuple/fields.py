"""What an RNTuple field holds, and where an entry keeps its value while it is filled or read.

A field is declared with a C++ type - ``int``, ``std::string``,
``std::vector<float>`` - which ROOT normalises to the name its files keep,
``std::int32_t`` for ``int``. Its value lives in a holder the model hands
out: for a number or a string a :class:`RFieldPtr`, the ``shared_ptr<T>``
a macro writes through with ``*p = v``; for a vector the ``std::vector``
itself, which a macro fills with ``push_back``. A writer reads the holders
when it fills; a reader writes into them when it loads an entry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...rntuple.writer import CXX
from ..stl import _spelled, element_type, std, string

__all__ = ["FieldType", "RFieldPtr", "field_type"]


class FieldType:
    """A field's type: the C++ name ROOT's files keep, and the number type of its values.

    A class type is a *record*: the type name is the class's, as the file keeps it,
    and what it holds is its members, each a field of its own. ROOT learns them
    from the class's dictionary, which is where :func:`field_type` reads them.
    """

    def __init__(
        self,
        cxx: str,
        dtype: np.dtype[Any] | None,
        vector: bool,
        members: Any = None,
    ) -> None:
        self.cxx, self.dtype, self.vector = cxx, dtype, vector
        #: For a record, its members - ``(name, FieldType)`` - in the class's order;
        #: ``None`` for every other kind of field, which holds a value of its own.
        self.members: tuple[tuple[str, FieldType], ...] | None = (
            None if members is None else tuple(members)
        )

    @property
    def record(self) -> bool:
        """Whether this is a record: a class type, whose value is its members'."""
        return self.members is not None

    @property
    def spec(self) -> Any:
        """What :meth:`xrdroot.WritableFile.rntuple` declares a field of this type with."""
        if self.record:
            raise UnsupportedFeatureError(
                f"an RNTuple field of type {self.cxx} is a record, and writing one is not "
                f"supported yet: xrdroot's RNTuple writer declares leaf and collection "
                f"fields only, so it has no way to say that a field is a record of fields "
                f"with no column of its own. Give the record's members as fields of the "
                f"model instead, and read the class back with a view of them."
            )
        return self.cxx if self.dtype is not None else str

    def holder(self) -> Any:
        """A fresh value for an entry of this field: zero, empty, or an empty vector."""
        if self.record:
            assert self.members is not None
            return {name: member.holder() for name, member in self.members}
        if self.vector:
            return std.vector[CPP_ITEMS[str(self.dtype)]]()
        return RFieldPtr(self)

    def zero(self) -> Any:
        return "" if self.dtype is None else self.dtype.type(0).item()


#: The C++ name a vector of each number type is made with, by NumPy name.
CPP_ITEMS = {"float32": "float", "float64": "double", "bool": "bool", "int8": "signed char",
             "uint8": "unsigned char", "int16": "short", "uint16": "unsigned short",
             "int32": "int", "uint32": "unsigned int", "int64": "long",
             "uint64": "unsigned long"}  # fmt: skip


def _refusal(whole: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        f"an RNTuple field of type {whole} is not supported: xrdroot's RNTuple writer and "
        f"entries hold numbers, std::string and std::vector of a number, whose layout is "
        f"beyond doubt; records, nested collections and variants are not written, and a "
        f"class type would need the dictionary that describes its members to be read, "
        f"which xrdroot does not do."
    )


def _number(text: str, whole: str) -> np.dtype[Any] | None:
    """The NumPy type of a number spelled ``text``, ``None`` for a string; else a refusal."""
    try:
        kind = element_type(text)
    except TypeError:
        raise _refusal(whole) from None
    if kind is string:
        return None
    if isinstance(kind, np.dtype) and kind.name in CXX:
        return kind
    raise _refusal(whole)


#: How a vector's C++ name starts, however it is spelled.
VECTORS = ("vector<", "ROOT::VecOps::RVec<", "ROOT::RVec<", "RVec<")


def field_type(cxx: Any, seen: frozenset[str] = frozenset()) -> FieldType:
    """The field type a C++ type name declares: ``int`` is ROOT's ``std::int32_t``.

    ``seen`` is the classes already being laid out, so that a class holding one of
    itself by value - which no RNTuple record can be written from - is refused
    rather than descended into for ever.
    """
    text = _spelled(str(cxx))
    if text.startswith(VECTORS) and text.endswith(">"):
        item = _number(text[text.index("<") + 1 : -1], str(cxx))
        if item is None:  # a vector of strings: a nested collection
            raise _refusal(str(cxx))
        return FieldType(f"std::vector<{CXX[item.name]}>", item, True)
    try:
        found = _number(text, str(cxx))
    except UnsupportedFeatureError:
        if not _is_class(text):
            raise
        return _record(text, seen)
    return FieldType("std::string" if found is None else CXX[found.name], found, False)


def _is_class(text: str) -> bool:
    """Whether a type name names a class: every class in the dictionary is capitalised,
    where a name that is merely misspelt - ``flaot`` - is not."""
    head = text.split("<", 1)[0].rsplit("::", 1)[-1]
    return head[:1].isupper() and "*" not in head and "&" not in head


def _record(cxx: str, seen: frozenset[str]) -> FieldType:
    """A class type as a record: its name as the file keeps it, and its members.

    ROOT reads the class's dictionary for the members it lays the record out of. A
    class no dictionary describes is a record with no members, which is what ROOT
    writes for a class that holds nothing - the members of the base classes come out
    as subfields ROOT names ``:_0``, which hold nothing either.
    """
    if cxx in seen:
        raise UnsupportedFeatureError(
            f"an RNTuple field of type {cxx} is not supported: the class holds a {cxx} of "
            f"itself by value, which an RNTuple record cannot be laid out from, because "
            f"each member would have to be written as the record that holds it."
        )
    return FieldType(cxx, None, False, _members(cxx, seen))


def _members(cxx: str, seen: frozenset[str]) -> list[tuple[str, FieldType]]:
    """The members the class's dictionary lists, each a field of its own.

    The class's own data members are what the dictionary declares beneath it: ROOT
    lists the enum constants it declares among its members too, and a record holds
    none of those - nor its bases, which ROOT writes as records of their own. A
    library the kit cannot read describes nothing, and then there are no members.
    """
    from ..core.reflection import TClass

    try:
        found = TClass.GetClass(cxx)
        declared = [] if found is None else found._fields()
    except Exception:  # a dictionary xrdroot cannot read: no member of it is known
        return []
    return [
        (str(name), field_type(spelled, seen | {cxx}))
        for name, spelled, _comment, _dims in declared
    ]


class RFieldPtr:
    """``std::shared_ptr<T>`` to a number or a string an entry holds: ``*p`` is ``p.value``."""

    #: What tells the macro runtime's ``deref`` that ``*p`` is this holder's value.
    _cint_cell = True

    def __init__(self, kind: FieldType) -> None:
        self.kind = kind
        self.value: Any = kind.zero()

    def __repr__(self) -> str:
        return f"<shared_ptr<{self.kind.cxx}> holding {self.value!r}>"

    def __getitem__(self, index: int) -> Any:
        return self.value

    def __setitem__(self, index: int, value: Any) -> None:
        self.value = value
