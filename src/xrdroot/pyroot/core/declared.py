"""The classes C++ code declares, as ROOT's I/O sees them: described, written and read back.

A class a macro, a library or ACLiC declares is a Python class whose
translation says what its data members are (``_cxx_layout_``: its C++ name,
its bases, and each member's name, type as written, trailing comment and
array lengths) and, with a ``ClassDef``, its version (``Class_Version``). That
is all ROOT's dictionary knows of a class too, so an object of one is written
as ROOT would write it - :mod:`xrdroot.wdeclared` describes the class,
:mod:`xrdroot.wdobjects` lays out the bytes - and a record of it read back
becomes an object of it again: made by its default constructor, as ROOT
makes one, and given each member the file holds, converted to the type the
class declares it with now. That conversion is all the schema evolution a
class of plain members needs - an ``int`` written and a ``float`` read, a
member added since (left as the constructor made it) or dropped (ignored).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

import numpy as np

from ...cppstd.containers import std
from ...wclasses import TYPES
from ...wdeclared import (
    ARRAY_TEMPLATES,
    MAPS,
    STL_KINDS,
    Catalogue,
    Declaration,
    Field,
    Spelled,
    class_name,
    parse_type,
)
from ...wdobjects import Clones, DeclaredObject, Instance
from ...winfo import INFOS
from .objects import TNamed, TObject

__all__ = [
    "declared_class",
    "declaration_of",
    "find_class",
    "catalogue_of",
    "instance_of",
    "payload_of",
    "made_from",
]

#: The bits a ``TObject``-derived integer type is wrapped to, by the C++ type it is.
WIDTHS = {
    "char": (8, True), "short": (16, True), "int": (32, True), "long": (64, True),
    "long long": (64, True), "unsigned char": (8, False), "unsigned short": (16, False),
    "unsigned int": (32, False), "unsigned long": (64, False),
    "unsigned long long": (64, False),
}  # fmt: skip
#: The floating types: what a member of one is read back as.
FLOATS = ("float", "double", "Double32_t", "Float16_t", "long double")


def declared_class(cls: Any) -> bool:
    """Was ``cls`` declared in C++ - does its translation describe its members?"""
    return isinstance(cls, type) and "_cxx_layout_" in cls.__dict__


def find_class(name: str) -> type | None:
    """The class C++ run so far declared as ``name`` - qualified or not - if there is one.

    The translation names a class in a namespace, or inside another class, by
    its own name alone, so ``ns::Deep`` is found as ``Deep``.
    """
    from ...pycxx.runtime.namespace import DECLARED

    wanted = name.strip()
    last = wanted.split("::")[-1]
    fallback = None
    for cls in list(DECLARED.values()):
        if not declared_class(cls):
            continue
        own = str(cls.__dict__["_cxx_layout_"][0])
        if own == wanted:
            return cls  # type: ignore[no-any-return]
        if own.split("::")[-1] == last:
            fallback = cls
    return fallback


def declaration_of(cls: type, name: str | None = None) -> Declaration:
    """What ``cls``'s translation says of it, as :mod:`xrdroot.wdeclared` takes a class."""
    own, bases, fields = cls.__dict__["_cxx_layout_"]
    versioned = "Class_Version" in cls.__dict__
    version = int(cls.Class_Version()) if versioned else 1  # type: ignore[attr-defined]
    made = tuple(Field(str(n), str(s), comment_of(c), tuple(d)) for n, s, c, d in fields)
    return Declaration(name or str(own), version, foreign(cls), tuple(map(str, bases)), made,
                       issubclass(cls, TObject), _class_title(cls))  # fmt: skip


#: The doxygen markers a member's note opens with once its ``//`` or ``/*`` is off: a comment
#: written after the member it documents rather than above it. Doxygen's ``!`` and ``*`` are
#: doxygen's, not ROOT's - ROOT's transient mark is a bare ``!``.
DOXYGEN_TRAILERS = ("/<", "!<", "*<")


def comment_of(note: str) -> str:
    """A data member's trailing comment as ROOT reads it, which is what says whether the
    member is persistent.

    ROOT counts a member transient when its comment begins with ``!``, and doxygen's trailing
    comments begin with a marker instead: ``Int_t fPy; //!< text`` is a persistent member
    documented by ``text``, while ``Int_t fCache; ///<! text`` is a transient one. A note
    reaches here with its ``//`` or ``/*`` already off, so the marker is what it opens with -
    ``/<`` from ``///<``, ``*<`` from ``/**<``, and ``!<`` from ``//!<`` or ``/*!<``, the one
    form in which doxygen's ``!`` is not ROOT's. A ``!`` anywhere else, ``//! text``
    included, is ROOT's and stays.
    """
    text = str(note)
    if text[:2] in DOXYGEN_TRAILERS:
        text = text[2:]
    return text.lstrip()


def foreign(cls: type) -> bool:
    """Was ``cls`` declared without ``ClassDef`` - which gives it ``Class_Name`` as well as
    ``Class_Version`` - so that its objects carry its checksum where a version would go?"""
    return "Class_Name" not in cls.__dict__


def _class_title(cls: type) -> str:
    title = getattr(cls, "CLASS_TITLE", "")
    return "" if "CLASS_TITLE" not in cls.__dict__ else str(title)


def catalogue_of(cls: type) -> Catalogue:
    """Every declared class an object of ``cls`` can reach, described as a write needs."""
    found: dict[str, Declaration] = {}
    _gather(declaration_of(cls).name, cls, found)
    return Catalogue(found)


def catalogue_for(classname: str) -> Catalogue:
    """Every declared class a value of the type ``classname`` can reach: a class's, or what a
    container holds."""
    found: dict[str, Declaration] = {}
    _reach_type(parse_type(classname), found)
    return Catalogue(found)


def plain_value(value: Any, classname: str, catalogue: Catalogue) -> Any:
    """A value of the type ``classname`` as plain data, as a write takes it."""
    return _plain(value, parse_type(classname), catalogue)


def new_value(classname: str) -> Any:
    """A default-made value of the type ``classname``: a container of the kit, an object of a
    class C++ declared - or ``None`` for one nothing here makes."""
    found = parse_type(classname)
    if found.name in STL_KINDS:
        args = [arg.text() for arg in found.args]
        return getattr(std, found.name)[args[0] if len(args) == 1 else tuple(args)]()
    cls = find_class(found.name)
    if cls is None:
        from .reflection import _class_named

        cls = _class_named(found.name)
    return _new(cls) if cls is not None else None


def _gather(name: str, cls: type, found: dict[str, Declaration]) -> None:
    declared = declaration_of(cls, name)
    found[name] = declared
    for base in declared.bases:
        _reach(base, found)
    for field in declared.fields:
        _reach_type(parse_type(field.spelled), found)


def _reach_type(found: Spelled, into: dict[str, Declaration]) -> None:
    """Add every declared class a type reaches: the type itself - a template instance such as
    a macro's ``ArrayHolderT<int>`` is a class of its own - and whatever its arguments are."""
    for arg in found.args:
        _reach_type(arg, into)
    if found.name not in TYPES:
        # The name with its arguments and without the stars: a pointer to a class reads
        # the class, and a template instance is a class of its own.
        _reach(Spelled(found.name, found.args, 0).text(), into)


def _reach(name: str, into: dict[str, Declaration]) -> None:
    if name in into or name in INFOS:
        return
    cls = find_class(name)
    if cls is not None:
        _gather(name, cls, into)


# -- writing ------------------------------------------------------------------------------------


def payload_of(obj: Any, writer: Any = None) -> DeclaredObject | None:
    """What a file writes for ``obj``, if its class was declared in C++ - else ``None``.
    ``writer`` is the file being written into, which a ``TRef`` among the object's members
    names its process in - the current file's when not given."""
    cls = type(obj)
    if not declared_class(cls):
        return None
    catalogue = catalogue_of(cls)
    name = declaration_of(cls).name
    title = obj.GetTitle() if isinstance(obj, TObject) else ""
    token = WRITER.set(writer if writer is not None else _current_writer())
    try:
        return DeclaredObject(instance_of(obj, name, catalogue), catalogue, str(title))
    finally:
        WRITER.reset(token)


#: The file an object is being written into, for the references among its members.
WRITER: ContextVar[Any] = ContextVar("WRITER", default=None)


def _current_writer() -> Any:
    """The writable directory of the current file, if it is one being written."""
    from .directories import current_directory

    writable = getattr(current_directory(), "_writable", None)
    return writable() if callable(writable) else None


def instance_of(obj: Any, name: str, catalogue: Catalogue) -> Instance:
    """``obj`` as plain data, read by the declaration of its class ``name``."""
    declared = catalogue.declared(name)
    assert declared is not None  # every class reached is in the catalogue
    members: dict[str, Any] = {}
    for base in declared.bases:
        members[base] = _base_value(obj, base, catalogue)
    for field in declared.fields:
        if not field.comment.startswith("!"):
            value = getattr(obj, _attribute(field.name), None)
            members[field.name] = _plain_member(value, field, catalogue)
    return Instance(name, members)


def _attribute(name: str) -> str:
    from ...pycxx.symbols import member_name

    return member_name(name)


def _base_value(obj: Any, base: str, catalogue: Catalogue) -> Any:
    if catalogue.declared(base) is not None:
        return instance_of(obj, base, catalogue)
    return root_row(obj, base)


def root_row(obj: Any, classname: str) -> dict[str, Any]:
    """The part of ``obj`` one of ROOT's classes ``classname`` is, as the members its layout
    names - each base's under the base's name, a ``TObject``'s bits and a ``TNamed``'s name and
    title alongside, as either way of writing them reads them."""
    row: dict[str, Any] = {}
    if isinstance(obj, TObject):
        row["fBits"] = obj._bitword()
        row["fUniqueID"] = obj.GetUniqueID()
    if isinstance(obj, TNamed):
        row["fName"] = TNamed.GetName(obj)
        row["fTitle"] = TNamed.GetTitle(obj)
    for element in INFOS.get(classname, (0, 0, ()))[2]:
        member = element[1]
        if element[0] == "TStreamerBase":
            row[member] = root_row(obj, member)
        elif member not in row:
            row[member] = _root_member(obj, member)
    return row


def _root_member(obj: Any, member: str) -> Any:
    """A member of one of ROOT's bases, from its getter where it has one: ``fLineColor``."""
    getter = getattr(obj, "Get" + member[1:], None)
    return getter() if callable(getter) else getattr(obj, member, None)


def _plain_member(value: Any, field: Any, catalogue: Catalogue) -> Any:
    """A data member's value as plain data.

    An array of *objects* - ``T a[N]`` and ``std::array<T,N>`` alike - holds one plain
    instance per element, which is what the member-wise writer writes one record
    after another; an array of numbers stays the values it is.
    """
    found = parse_type(field.spelled)
    template = found.name in ARRAY_TEMPLATES and bool(found.args)
    content = found.args[0] if template else found  # a std::array's, or the element of T[N]
    is_array = bool(field.dims) or template
    if is_array and catalogue.declared(class_name(content)) is not None:  # an array of objects
        return [_plain(element, content, catalogue) for element in (value if value is not None
                                                                    else ())]  # fmt: skip
    return _plain(value, found, catalogue)


def _plain(value: Any, found: Spelled, catalogue: Catalogue) -> Any:
    """A member's value as plain data: objects as instances, containers as lists or dicts - an
    object held by value that is not there yet, a slot ``resize`` made, as a default-made one."""
    value = _present(value, found, catalogue)
    if value is None:
        return None
    if not found.stars and (found.name in STL_KINDS or found.name == "pair"):
        return _plain_container(value, found, catalogue)
    return _plain_object(value, found, catalogue)


def _present(value: Any, found: Spelled, catalogue: Catalogue) -> Any:
    """A member's value as there to write: an object held by value that is not made yet as
    a default-made one, and a pointer a macro set to 0 as the null it is."""
    if value is None and not found.stars and found.name in catalogue.declarations:
        return new_value(found.name)
    null = found.stars and isinstance(value, int) and value == 0
    return None if null else value


def _plain_object(value: Any, found: Spelled, catalogue: Catalogue) -> Any:
    """An object as plain data - one of a declared class as an instance, one of ROOT's as its
    members - and anything else as it is, a string class as its text.

    The class is looked up by the name a streamer spells, template arguments and all:
    a macro's ``ArrayHolderT<int>`` is declared under that name, where the spelled type
    of the member says only ``ArrayHolderT``.
    """
    name = class_name(found)
    if catalogue.declared(name) is not None and name not in ("string", "TString"):
        return _object_of(value, name, catalogue)
    if isinstance(value, TObject):
        return _root_object(value, found.name, catalogue)
    return value if isinstance(value, (int, float, np.ndarray, list)) else _scalar(value)


def _root_object(value: Any, classname: str, catalogue: Catalogue) -> Any:
    """One of ROOT's objects among a declared class's members: a ``TClonesArray`` as the
    objects it holds, one the writer takes - a histogram, a ``TRef``, a ``TBits`` - as it takes
    it, and any other by the members its layout names."""
    from .collections import TClonesArray
    from .records import recorded

    if isinstance(value, TClonesArray):
        return _clones(value, catalogue)
    made = recorded(value, WRITER.get())
    if made is not value:
        return made
    return root_row(value, classname) if classname in INFOS else value


def _clones(value: Any, catalogue: Catalogue) -> Clones:
    """A ``TClonesArray`` of objects of a declared class, each as an instance of it."""
    held = str(value._class)
    if held not in catalogue.declarations:
        cls = find_class(held)
        if cls is not None:
            catalogue.declarations.update(catalogue_of(cls).declarations)
    items = [None if item is None else instance_of(item, held, catalogue)
             for item in list(value)[: value.GetEntriesFast()]]  # fmt: skip
    return Clones(held, str(getattr(value, "_name", "")) or held + "s", items)


def _plain_container(value: Any, found: Spelled, catalogue: Catalogue) -> Any:
    """A container as a list of what it holds - a map's as pairs, not a dict, since a multimap
    holds a key as often as it was inserted - and a pair as the record of its two members."""
    if found.name == "pair":
        first, second = value
        members = {"first": _plain(first, found.args[0], catalogue),
                   "second": _plain(second, found.args[1], catalogue)}  # fmt: skip
        return Instance(class_name(found), members)
    if found.name in MAPS:
        key, held = found.args
        return [(_plain(k, key, catalogue), _plain(v, held, catalogue)) for k, v in _pairs(value)]
    return [_plain(item, found.args[0], catalogue) for item in value]


def _object_of(value: Any, declared: str, catalogue: Catalogue) -> Instance:
    """An object as plain data - by its own class, when it is one derived from the declared."""
    actual = type(value)
    name = declaration_of(actual).name if declared_class(actual) else declared
    if name != declared and catalogue.declared(name) is None:
        catalogue.declarations.update(catalogue_of(actual).declarations)
    return instance_of(value, name, catalogue)


def _pairs(value: Any) -> list[tuple[Any, Any]]:
    items = value.items() if hasattr(value, "items") else value
    return [(pair[0], pair[1]) for pair in items]


def _scalar(value: Any) -> Any:
    """A string class as the text it holds, anything else as it is."""
    if isinstance(value, (str, np.generic)) or not hasattr(value, "__str__"):
        return value
    return str(value) if type(value).__name__ in ("TString", "string") else value


# -- reading ------------------------------------------------------------------------------------


def made_from(classname: str, row: Any) -> Any:
    """An object of the declared class ``classname`` made from a record read - or ``None``
    when no class of that name was declared."""
    cls = find_class(classname)
    if cls is None or not isinstance(row, Mapping):
        return None
    made = _new(cls)
    _fill(made, cls, row)
    return made


def _new(cls: type) -> Any:
    """A default-made object, as ROOT makes one to read into."""
    try:
        return cls()
    except TypeError:
        return object.__new__(cls)


def _fill(obj: Any, cls: type, row: Mapping[str, Any]) -> None:
    """``obj`` given the members of ``row``, by the declaration of ``cls``."""
    _own, bases, fields = cls.__dict__["_cxx_layout_"]
    for base in bases:
        _fill_base(obj, cls, str(base), row.get(str(base)))
    for name, spelled, comment, _dims in fields:
        if name in row and not str(comment).startswith("!"):
            attribute = _attribute(str(name))
            current = getattr(obj, attribute, None)
            setattr(obj, attribute, converted(row[name], parse_type(str(spelled)), current))


def _fill_base(obj: Any, cls: type, base: str, row: Any) -> None:
    if not isinstance(row, Mapping):
        return
    declared = find_class(base)
    if declared is not None and isinstance(obj, declared):
        _fill(obj, declared, row)
        return
    fill_root(obj, row)


#: How ``TObject``'s and ``TNamed``'s members are given back to an object of the namespace's.
_ROOT_MEMBERS: dict[str, Any] = {
    "fBits": lambda obj, value: obj._store_bits(int(value)),
    "fUniqueID": lambda obj, value: obj.SetUniqueID(int(value)),
    "fName": lambda obj, value: TNamed.SetName(obj, str(value)),
    "fTitle": lambda obj, value: TNamed.SetTitle(obj, str(value)),
}


def fill_root(obj: Any, row: Mapping[str, Any]) -> None:
    """The part of ``obj`` one of ROOT's classes is, given the members read: a ``TObject``'s
    bits, a ``TNamed``'s name and title, each base's members, and every other member by its
    setter - ``fLineColor`` by ``SetLineColor``."""
    for key, value in row.items():
        special = _ROOT_MEMBERS.get(key)
        if special is not None:
            special(obj, value)
        elif isinstance(value, Mapping):
            fill_root(obj, value)
        else:
            setter = getattr(obj, "Set" + key[1:], None)
            if callable(setter):
                setter(value)


def converted(value: Any, found: Spelled, current: Any) -> Any:
    """A value read, as a member of the type ``found`` holds it now - into ``current``, the
    member as the constructor left it, where that is a container or an object."""
    if value is None:
        return None
    if found.stars and found.name not in TYPES:
        return _pointed(value, found)
    if found.name in STL_KINDS:
        return _refilled(value, found, current)
    return _held(value, found, current)


def _held(value: Any, found: Spelled, current: Any) -> Any:
    """A value read for a member that is no container: text - a string class's, or a
    ``char *``'s - a number, or an object."""
    if found.name in ("string", "TString") or isinstance(value, str):
        return _text(value, found, current)
    if found.name in TYPES or found.name in WIDTHS:
        return _number(value, found.name, current)
    return _object(value, found, current)


def _pointed(value: Any, found: Spelled) -> Any:
    made = made_from(found.name, value)
    return made if made is not None else value


def _object(value: Any, found: Spelled, current: Any) -> Any:
    if found.name == "pair":
        first, second = found.args
        one, two = _pair_of(value)
        made = std.pair[first.text(), second.text()]
        return made(converted(one, first, None), converted(two, second, None))
    cls = find_class(found.name)
    if not isinstance(value, Mapping):
        return value
    if cls is None:  # one of ROOT's classes, held by value: given its members where it is
        if current is None:
            return value
        fill_root(current, value)
        return current
    target = current if isinstance(current, cls) else _new(cls)
    _fill(target, cls, value)
    return target


def _text(value: Any, found: Spelled, current: Any) -> Any:
    text = str(value)
    if found.name == "TString" and current is not None:
        return type(current)(text)
    return text


def _number(value: Any, typename: str, current: Any) -> Any:
    """A number, or an array of them, as the C++ type ``typename`` holds it."""
    if isinstance(current, np.ndarray):
        flat = np.asarray(value).ravel()
        target = current.reshape(-1)
        count = min(len(flat), len(target))
        target[:count] = flat[:count]
        return current
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value)
    return _scalar_number(value, TYPES.get(typename, (0, typename, 0))[1])


def _scalar_number(value: Any, typename: str) -> Any:
    number = value.item() if isinstance(value, np.generic) else value
    if typename == "bool":
        return bool(number)
    if typename in FLOATS:
        return float(number)
    bits, signed = WIDTHS[_c_name(typename)]  # every integer a streamer names has a width
    whole = math.trunc(number) if isinstance(number, float) else int(number)
    whole &= (1 << bits) - 1
    if signed and whole >= 1 << (bits - 1):
        whole -= 1 << bits
    return whole


def _c_name(typename: str) -> str:
    """ROOT's streamer names for the integers as the C++ types they are."""
    return {"Long64_t": "long long", "ULong64_t": "unsigned long long"}.get(typename, typename)


def _refilled(value: Any, found: Spelled, current: Any) -> Any:
    """A container member given the items read, each converted to what it holds now."""
    if current is None or not hasattr(current, "clear"):
        return value
    current.clear()
    items = list(value.items()) if isinstance(value, Mapping) else list(value)
    if found.name in MAPS:
        key_type, value_type = found.args
        for key, item in map(_pair_of, items):
            _put(current, converted(key, key_type, None), converted(item, value_type, None))
        return current
    add = getattr(current, "push_back", None) or current.insert
    for item in items:
        add(converted(item, found.args[0], None))
    return current


def _pair_of(item: Any) -> tuple[Any, Any]:
    """A pair read - a tuple, or a record of ``first`` and ``second`` - as two values."""
    if isinstance(item, Mapping):
        return item.get("first"), item.get("second")
    first, second = item
    return first, second


def _put(target: Any, key: Any, value: Any) -> None:
    """A key and its value into a map - or a multimap, which only inserts."""
    if hasattr(target, "__setitem__"):
        target[key] = value
    else:
        target.insert((key, value))
