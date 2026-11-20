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
from ...rntuple.writer import CXX, CollectionSpec, LowPrecisionSpec, RecordSpec
from ..stl import _spelled, _split_arguments, element_type, std, string

__all__ = ["FieldType", "RFieldPtr", "field_type"]


class FieldType:
    """A field's type: the C++ name ROOT's files keep, and the number type of its values.

    A class type is a *record*: the type name is the class's, as the file keeps it,
    and what it holds is its members, each a field of its own. ROOT learns them
    from the class's dictionary, which is where :func:`field_type` reads them. A
    set or a map is a *collection*: an item field under it - for a map the
    ``std::pair`` record of the key and the value - as a ``std::vector`` has.
    """

    def __init__(
        self,
        cxx: str,
        dtype: np.dtype[Any] | None,
        vector: bool,
        members: Any = None,
        bits: int = 0,
        item: FieldType | None = None,
        precision: LowPrecisionSpec | None = None,
    ) -> None:
        self.cxx, self.dtype, self.vector = cxx, dtype, vector
        #: For a float or a double told to be stored in fewer bits, how it is stored;
        #: ``None`` for every other field, which keeps its own type's precision.
        self.precision = precision
        #: For a ``std::bitset``, how many bits an entry holds; zero for every other field.
        self.bits = bits
        #: For a record, its members - ``(name, FieldType)`` - in the class's order;
        #: ``None`` for every other kind of field, which holds a value of its own.
        self.members: tuple[tuple[str, FieldType], ...] | None = (
            None if members is None else tuple(members)
        )
        #: For a set or a map, the type of the item under it; ``None`` for every other
        #: kind of field. A map's item is the ``std::pair`` of its key and its value.
        self.item = item

    @property
    def collection(self) -> bool:
        """Whether this is a set or a map: a field whose value is its items'."""
        return self.item is not None

    @property
    def record(self) -> bool:
        """Whether this is a record: a class type, whose value is its members'."""
        return self.members is not None

    @property
    def spec(self) -> Any:
        """What :meth:`xrdroot.WritableFile.rntuple` declares a field of this type with.

        A record is declared as a :class:`~xrdroot.rntuple.writer.RecordSpec`: the
        class's name, and each member's own declaration under it; a set or a map as
        a :class:`~xrdroot.rntuple.writer.CollectionSpec` of the item under it.
        """
        if self.precision is not None:
            return self.precision
        if self.record:
            assert self.members is not None
            return RecordSpec(self.cxx, [(name, member.spec) for name, member in self.members])
        if self.item is not None:
            return CollectionSpec(self.cxx, self.item.spec)
        if self.bits:
            return self.cxx
        return self.cxx if self.dtype is not None else str

    def holder(self) -> Any:
        """A fresh value for an entry of this field: zero, empty, or an empty vector.

        A record's is its members' values of an object of its class default-constructed,
        which is what ``MakeField<T>(name)`` hands out and a ``Fill()`` with nothing set
        writes: a class's default member initialisers are part of the value the file gets.
        """
        if self.record:
            assert self.members is not None
            made = _default_object(self.cxx)
            return {name: _member_holder(name, member, made) for name, member in self.members}
        if self.item is not None:
            return _container_class(self.cxx)()
        if self.vector:
            return std.vector[CPP_ITEMS[str(self.dtype)]]()
        return RFieldPtr(self)

    def value(self, data: Any) -> Any:
        """A value as the file gives it, held as C++ holds this type.

        A view hands out what C++ would: a ``std::bitset`` of its bits, a vector of its
        items, a set or a map of its, and a record as an object of its class - the class
        the interpreter declared, whose members are the ones the file keeps.
        """
        if self.record:
            return _record_value(self, data)
        if self.item is not None:
            return _collection_value(self, data)
        if self.vector:
            return _vector_value(self, data)
        if self.bits:
            return std.bitset[self.bits](_bits(data))
        return _scalar_value(self, data)

    def zero(self) -> Any:
        if self.bits:
            return std.bitset[self.bits]()
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


#: How a ``std::bitset<N>`` is spelled once the ``std::`` is off, its width inside.
BITSETS = "bitset<"

#: How a set or a map is spelled once the ``std::`` is off, and whether it is keyed.
COLLECTIONS = {
    "set": False,
    "multiset": False,
    "unordered_set": False,
    "unordered_multiset": False,
    "map": True,
    "multimap": True,
    "unordered_map": True,
    "unordered_multimap": True,
}

#: The cppstd template a set's or a map's holder is made with, by the name it is
#: spelled with. cppstd has no unordered multiset or multimap, and the ordered one
#: of the same kinds iterates its items the same way - a key with its value - so a
#: holder keeps them there; the field itself is written as the C++ type says.
HOLDERS = {
    "set": "set",
    "multiset": "multiset",
    "unordered_set": "unordered_set",
    "unordered_multiset": "multiset",
    "map": "map",
    "multimap": "multimap",
    "unordered_map": "unordered_map",
    "unordered_multimap": "multimap",
}


def _record_value(kind: FieldType, data: Any) -> Any:
    """A record as an object of its class, its members under their names - or, where no
    dictionary here declares the class, the members' values alone, as a ``dict``."""
    assert kind.members is not None
    members = {name: member.value(data[name]) for name, member in kind.members}
    made = _default_object(kind.cxx)
    if made is None:
        return members
    for name, held in members.items():
        setattr(made, name, held)
    return made


def _collection_value(kind: FieldType, data: Any) -> Any:
    """A set's or a map's items in the container an entry holds them in."""
    from .model import RField, _fill

    held = kind.holder()
    _fill(held, RField("", kind), data)
    return held


def _vector_value(kind: FieldType, data: Any) -> Any:
    """A vector's items, in a ``std::vector`` of the field's own element type."""
    vector = kind.holder()
    vector.assign(data)
    return vector


def _scalar_value(kind: FieldType, data: Any) -> Any:
    """A number as the type the file keeps it as, or a string's text."""
    found = data.item() if isinstance(data, np.generic) else data
    return str(found) if kind.dtype is None else kind.dtype.type(found).item()


def _bits(data: Any) -> int:
    """The number a column of a ``std::bitset``'s bits is, lowest bit first."""
    return sum(int(bool(bit)) << at for at, bit in enumerate(data))


def _default_object(cxx: str) -> Any:
    """An object of a class the interpreter declared, default-constructed - or ``None``.

    ROOT reads the class's dictionary for both the members a record is laid out of and
    the object one is filled from, so what the file holds of a class nobody declared
    here is its members' zeros alone.
    """
    from ..stl import _resolve_root_type

    try:
        found = _resolve_root_type(cxx)
        return None if not isinstance(found, type) else found()
    except Exception:  # a class with no constructor a dictionary here gives: no object
        return None


def _member_holder(name: str, member: FieldType, made: Any) -> Any:
    """A record member's holder: what the object's own member holds, or the member's zero.

    ``made`` is the default-constructed object of the class the member belongs to, or
    ``None`` where no dictionary here declared that class.
    """
    value = None if made is None else getattr(made, name, None)
    if member.record:
        return {sub: _member_holder(sub, sub_kind, value) for sub, sub_kind in member.members or ()}
    held = member.holder()
    if value is None:
        return held
    if isinstance(held, RFieldPtr):  # a number, a string or a bitset: held by a pointer
        held.value = value
        return held
    return value  # a vector, a set or a map: the object's own, which a fill reads


def _container_class(cxx: str) -> Any:
    """The cppstd class an entry of a set or a map field holds its items in."""
    head, _sep, rest = cxx[len("std::") :].partition("<")
    template = getattr(std, HOLDERS[head])
    arguments = _split_arguments(rest[:-1])
    return template[arguments[0]] if len(arguments) == 1 else template[tuple(arguments)]


def _bitset(text: str) -> int | None:
    """How many bits a ``std::bitset`` is of, or ``None`` if ``text`` is no bitset's name."""
    if not text.startswith(BITSETS) or not text.endswith(">"):
        return None
    written = text[len(BITSETS) : -1].strip()
    return int(written) if written.isdigit() else None


#: How a vector's C++ name starts, however it is spelled.
VECTORS = ("vector<", "ROOT::VecOps::RVec<", "ROOT::RVec<", "RVec<")


def field_type(cxx: Any, seen: frozenset[str] = frozenset()) -> FieldType:
    """The field type a C++ type name declares: ``int`` is ROOT's ``std::int32_t``.

    ``seen`` is the classes already being laid out, so that a class holding one of
    itself by value - which no RNTuple record can be written from - is refused
    rather than descended into for ever.
    """
    text = _spelled(str(cxx))
    bits = _bitset(text)
    if bits is not None:
        return FieldType(f"std::{BITSETS}{bits}>", None, False, bits=bits)
    found = _collection(text)
    if found is not None:
        return _collection_type(*found, str(cxx), seen)
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


def _collection(text: str) -> tuple[str, list[str]] | None:
    """A set's or a map's name with the ``std::`` off, and its template arguments."""
    head, sep, rest = text.partition("<")
    if not (sep and head in COLLECTIONS and rest.endswith(">")):
        return None
    return head, _split_arguments(rest[:-1])


def _collection_type(
    head: str, arguments: list[str], whole: str, seen: frozenset[str]
) -> FieldType:
    """A set or a map as a collection: the item under it, and for a map the ``std::pair``.

    ROOT writes a map as a collection over the record of its key and its value, which
    is what a reader of either library reads back as a key and a value.
    """
    if len(arguments) != (2 if COLLECTIONS[head] else 1):
        raise _refusal(whole)
    if not COLLECTIONS[head]:
        item = field_type(arguments[0], seen)
        return FieldType(f"std::{head}<{item.cxx}>", None, False, item=item)
    key, value = field_type(arguments[0], seen), field_type(arguments[1], seen)
    pair = FieldType(f"std::pair<{key.cxx},{value.cxx}>", None, False, (("_0", key), ("_1", value)))
    return FieldType(f"std::{head}<{key.cxx},{value.cxx}>", None, False, item=pair)


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
