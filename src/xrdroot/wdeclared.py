"""What a class a program declares looks like to a ROOT file: its streamer information.

ROOT's interpreter, and the dictionary ACLiC builds, describe a class from its
declaration: each base, then each non-static data member in order, its type as
written (``Double32_t`` stays ``Double32_t``, ``Int_t`` becomes ``int``), its
trailing comment as its title - ``//!`` leaves it out, ``//[n]`` counts an
array, ``//[0,20,10]`` packs a float, ``//->`` promises a pointer is never
null - and the class version from its ``ClassDef``, else 1. A class without a
``ClassDef`` is *foreign*: its objects carry its checksum where a version
would go, so a reader can tell two layouts of it apart.

This module turns a :class:`Declaration` - plain data, which the layer that
knows the program's classes fills in - into those descriptions, element by
element as :data:`~.winfo.INFOS` holds ROOT's own, with the checksum
``TClass::GetCheckSum`` gives them. :mod:`.wdobjects` writes objects by them.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from .errors import UnsupportedFeatureError
from .wclasses import TYPES, checksum
from .winfo import INFOS, Element

__all__ = ["Field", "Declaration", "Spelled", "Catalogue", "parse_type", "normalised"]

#: ROOT's typedefs of numbers, as the C++ types they stand for.
ROOT_TYPEDEFS = {
    "Char_t": "char", "UChar_t": "unsigned char", "Short_t": "short",
    "UShort_t": "unsigned short", "Int_t": "int", "UInt_t": "unsigned int",
    "Seek_t": "int", "Long_t": "long", "ULong_t": "unsigned long", "Float_t": "float",
    "Double_t": "double", "Bool_t": "bool", "Byte_t": "unsigned char",
    "Version_t": "short", "Option_t": "char", "Ssiz_t": "int", "Real_t": "float",
    "Long64_t": "long long", "ULong64_t": "unsigned long long", "Axis_t": "double",
    "Stat_t": "double", "Font_t": "short", "Style_t": "short", "Marker_t": "short",
    "Width_t": "short", "Color_t": "short", "SCoord_t": "short", "Coord_t": "double",
    "Angle_t": "float", "Size_t": "float", "size_t": "unsigned long",
    "int8_t": "char", "uint8_t": "unsigned char", "int16_t": "short",
    "uint16_t": "unsigned short", "int32_t": "int", "uint32_t": "unsigned int",
    "int64_t": "long", "uint64_t": "unsigned long", "signed char": "char",
    "long int": "long", "unsigned long int": "unsigned long", "short int": "short",
    "unsigned short int": "unsigned short", "long long int": "long long",
    "unsigned": "unsigned int", "signed": "int",
}  # fmt: skip

#: ``ROOT::ESTLType``: each standard container's number, by its name.
STL_KINDS = {
    "vector": 1, "list": 2, "deque": 3, "map": 4, "multimap": 5, "set": 6,
    "multiset": 7, "forward_list": 9, "unordered_set": 10, "unordered_multiset": 11,
    "unordered_map": 12, "unordered_multimap": 13,
}  # fmt: skip
#: The names ``std::array`` may be spelled with: a fixed array whose N values are in place.
ARRAY_TEMPLATES = ("array", "std::array")
#: The names ``std::complex`` may be spelled with: a class of two numbers of one width, which
#: ROOT has a dictionary for and streams as the pair of values it is.
COMPLEX_TEMPLATES = ("complex", "std::complex")

#: The containers that hold pairs, a key and a value.
MAPS = ("map", "multimap", "unordered_map", "unordered_multimap")
#: ``kSTLstring``: what ``std::string`` is to a ``TStreamerSTL``.
STL_STRING = 365
#: The streamer types of members and contents that are objects: ``kObject``,
#: ``kAny``, a pointer to one never null (``kObjectp``, ``kAnyp``) and maybe null
#: (``kObjectP``, ``kAnyP``), and a container (``kSTL``).
OBJECT, ANY, OBJECT_P, OBJECT_PP, ANY_P, ANY_PP, STL = 61, 62, 63, 64, 68, 69, 300
#: ``kCharStar``: a C string, its length and then its characters.
CHAR_STAR = 7
#: ROOT's classes that stream themselves by hand and that a file holds no description of.
SELF_STREAMED = ("TRef", "TRefArray")
#: ``kTString``, ``kTObject``, ``kTNamed`` and ``kCounter``.
TSTRING, TOBJECT, TNAMED, COUNTER = 65, 66, 67, 6
#: ``kOffsetL`` and ``kOffsetP``: a fixed array of, and a counted pointer to, numbers.
OFFSET_L, OFFSET_P = 20, 40
#: How many bytes each kind of member takes in memory, as a 64-bit ROOT has it.
STRING_SIZE, CONTAINER_SIZE, POINTER_SIZE = 24, 24, 8
#: The five maximum indices of an element that is not an array.
NO_DIMS = (0, 0, 0, 0, 0)


class Field(NamedTuple):
    """One data member as declared: name, type as written, trailing comment, array lengths."""

    name: str
    spelled: str
    comment: str
    dims: tuple[Any, ...]


class Declaration(NamedTuple):
    """One class as a program declares it."""

    name: str
    version: int
    #: Declared without ``ClassDef``: its objects carry their checksum instead of a version.
    foreign: bool
    bases: tuple[str, ...]
    fields: tuple[Field, ...]
    #: Whether it derives, somewhere up, from ``TObject``.
    tobject: bool = False
    #: What its ``ClassDef`` comment says it is, a base element's title.
    title: str = ""


class Spelled(NamedTuple):
    """A type taken apart: its name, its template arguments and how many stars follow it."""

    name: str
    args: tuple[Spelled, ...]
    stars: int

    def text(self) -> str:
        """The type as ROOT spells it in a streamer: ``vector<vector<float> >``."""
        if not self.args:
            return self.name + "*" * self.stars
        inner = ",".join(arg.text() for arg in self.args)
        close = " >" if inner.endswith(">") else ">"
        return f"{self.name}<{inner}{close}" + "*" * self.stars


#: The pieces a type is spelled with: a name, an angle bracket, a comma or a star.
_TOKENS = re.compile(r"[<>,*&]|[^<>,*&]+")


@functools.lru_cache(maxsize=4096)
def parse_type(spelled: str, resolve: bool = True) -> Spelled:
    """``spelled`` taken apart, ``std::`` and ``const`` dropped and - unless ``resolve`` is
    false - typedefs of numbers resolved."""
    tokens = [token.strip() for token in _TOKENS.findall(spelled) if token.strip()]
    found, _rest = _parse(tokens, 0, resolve)
    return found


def _parse(tokens: list[str], at: int, resolve: bool = True) -> tuple[Spelled, int]:
    name = _plain_name(tokens[at], resolve)
    at += 1
    args: list[Spelled] = []
    if at < len(tokens) and tokens[at] == "<":
        at += 1
        while tokens[at] != ">":
            arg, at = _parse(tokens, at, resolve)
            args.append(arg)
            at += 1 if tokens[at] == "," else 0
        at += 1
    stars = 0
    while at < len(tokens) and tokens[at] in ("*", "&"):
        stars += tokens[at] == "*"
        at += 1
    return Spelled(name, tuple(args), stars), at


def _plain_name(text: str, resolve: bool = True) -> str:
    words = [word for word in text.split() if word not in ("const", "volatile", "struct", "class")]
    name = " ".join(words).removeprefix("::")
    name = name.removeprefix("std::")
    return ROOT_TYPEDEFS.get(name, name) if resolve else name


def normalised(spelled: str, resolve: bool = True) -> str:
    """A type as ROOT's streamer names it - or, unless ``resolve``, as its dictionary does,
    a typedef such as ``Int_t`` kept."""
    found = parse_type(spelled, resolve)
    return streamer_name(found)


def class_name(found: Spelled) -> str:
    """The class a type names, its template arguments as ROOT spells them: ``pair<int,float>``."""
    return streamer_name(Spelled(found.name, found.args, 0))


def streamer_name(found: Spelled) -> str:
    """A type taken apart, put back together as ROOT spells it in a streamer."""
    basic = TYPES.get(found.name) if not found.args else None
    name = basic[1] if basic is not None else found.name
    args = tuple(Spelled(streamer_name(arg), (), 0) for arg in found.args)
    return Spelled(name, args, found.stars).text()


class Catalogue:
    """The classes a write reaches, each described once: the program's and ROOT's.

    ``declarations`` are the program's classes by name. A class ROOT has is
    described as :data:`~.winfo.INFOS` describes it; the pair a map holds is
    described as ROOT emulates one, ``first`` and ``second`` with no version.
    """

    def __init__(
        self, declarations: Mapping[str, Declaration], enums: frozenset[str] = frozenset()
    ) -> None:
        self.declarations = dict(declarations)
        #: The program's enumerations, which are written as ``int`` under their own names.
        self.enums = enums
        #: Each class described so far: its checksum, version and elements.
        self.infos: dict[str, tuple[int, int, tuple[Element, ...]]] = {}
        #: The classes ROOT has that a write reaches, for the file's streamer record.
        self.used: dict[str, None] = {}
        #: The classes being described just now, which a member may point back at.
        self._building: set[str] = set()

    # -- what a class is ----------------------------------------------------------

    def declared(self, name: str) -> Declaration | None:
        """The program's class ``name``, or the pair a map holds, if it is one."""
        found = self.declarations.get(name)
        if found is None and name.startswith("pair<"):
            found = _pair(name)
            self.declarations[name] = found
        return found

    def known(self, name: str) -> bool:
        """Is ``name`` a class this write can describe - or one of ROOT's that streams itself
        and that the file has no description of, a ``TRef``?"""
        return self.declared(name) is not None or name in INFOS or name in SELF_STREAMED

    def tobject(self, name: str) -> bool:
        """Does the class ``name`` derive from ``TObject``?"""
        found = self.declared(name)
        if found is not None:
            return found.tobject
        return name in ("TObject", *SELF_STREAMED) or any(
            element[0] == "TStreamerBase" and self.tobject(element[1])
            for element in INFOS.get(name, (0, 0, ()))[2]
        )

    def info(self, name: str) -> tuple[int, int, tuple[Element, ...]]:
        """The class ``name``'s checksum, version and elements."""
        found = self.infos.get(name)
        if found is not None:
            return found
        declared = self.declared(name)
        if declared is None and name in SELF_STREAMED:
            return 0, 1, ()  # streamed by hand, and described by nothing in the file
        if declared is None:
            self.used[name] = None
            return INFOS[name]
        self._building.add(name)
        try:
            elements = self._elements(declared)
        finally:
            self._building.discard(name)
        made = (checksum(name, elements), declared.version, elements)
        self.infos[name] = made
        return made

    def describe(self, name: str) -> None:
        """See that ``name`` is described - unless it is being described already, as a class
        holding a pointer to its own kind is while its members are."""
        if name not in self._building:
            self.info(name)

    def version_of(self, name: str) -> tuple[int, int | None]:
        """What an object of ``name`` writes where its version goes: the version, and the
        checksum after it if the class is foreign."""
        found = self.declared(name)
        sum_, version, _elements = self.info(name)
        if found is not None and found.foreign and version <= 1:
            return 0, sum_
        return version, None

    def entries(self) -> dict[str, tuple[int, int, tuple[Element, ...]]]:
        """The program's classes this write described, for the file's streamer record."""
        return {name: self.infos[name] for name in self.infos}

    # -- elements -----------------------------------------------------------------

    def _elements(self, declared: Declaration) -> tuple[Element, ...]:
        counters = {_counter_of(field) for field in declared.fields} - {None}
        found = [self._base(base) for base in declared.bases]
        for field in declared.fields:
            if field.comment.startswith("!"):
                continue  # transient: ``//!``
            found.append(self._member(declared, field, field.name in counters))
        return tuple(found)

    def _base(self, name: str) -> Element:
        if not self.known(name):
            raise UnsupportedFeatureError(
                f"a class derives from {name}, which is neither a class the program declares "
                f"nor one this writer carries a layout for, so its part of the object cannot "
                f"be written"
            )
        sum_, version, _elements = self.info(name)
        borrowed = _base_element(name) if name not in self.declarations else None
        if borrowed is not None:
            return borrowed
        declared = self.declarations.get(name)
        title = declared.title if declared is not None else ""
        signed = sum_ - (1 << 32) if sum_ >= 1 << 31 else sum_
        own = _class_version(declared, version)
        return ("TStreamerBase", name, title, 0, 0, 0, 0, (0, signed, 0, 0, 0), "BASE", (own,))

    def _member(self, owner: Declaration, field: Field, counter: bool) -> Element:
        found = parse_type(field.spelled)
        _note_complexes(self, found)
        for rule in _MEMBER_RULES:
            made = rule(self, owner, field, found, counter)
            if made is not None:
                return made
        raise UnsupportedFeatureError(
            f"{owner.name}::{field.name} is a {field.spelled}, which this writer does not "
            f"know how to lay out in a ROOT file; mark it transient with //! to leave it out"
        )

    def _fixed_array(
        self, owner: Declaration, field: Field, found: Spelled, counter: bool
    ) -> Element | None:
        """``std::array<T,N>``: ROOT lays one out as the fixed array ``T a[N]`` it is - the
        element's type with the array offset added, its size N values' worth, and N as the
        length - so a member of one is read by another ROOT as the N values it holds.

        One nested inside another is laid out as ``T a[M][N]``: ROOT's dictionary resolves
        ``std::array<std::array<int,4>,3>`` to the very element it gives ``int a[3][4]``, one
        element of ``int``, a length of 12, two dimensions with the outermost first, so the
        nine values of the test's ``a2`` are written as the nine ints of its ``a3``.
        """
        shape = _array_shape(found)
        if shape is None:
            return None
        element, lengths = shape
        if len(lengths) > MAX_DIMS:
            raise UnsupportedFeatureError(
                f"{owner.name}::{field.name} is a {field.spelled}, nested {len(lengths)} arrays "
                f"deep, and a ROOT file's layout records at most {MAX_DIMS} array dimensions; "
                f"give the member a plain C array, or mark it transient with //! to leave it out"
            )
        basic = self.basic(element)
        if basic is not None:
            stype, typename, size = basic
            length = _product(lengths)
            return ("TStreamerBasicType", field.name, field.comment, stype + OFFSET_L,
                    size * length, length, len(lengths), (*lengths, *NO_DIMS)[:5],
                    typename, ())  # fmt: skip
        return self._object_array(owner, field, element, lengths)

    def _object_array(
        self, owner: Declaration, field: Field, element: Spelled, lengths: int | tuple[int, ...]
    ) -> Element | None:
        """``std::array<T,N>`` of objects: the element a member of one object is, with the
        array offset on its type and N values' worth of room - ROOT's ``TStreamerObject``
        at ``kOffsetL + kObject``, as the oracle writes it.

        ``lengths`` is one dimension for a ``std::array``, and all of them outermost first
        for one nested in another; a count from ``T a[N]`` is the one dimension of its own.
        """
        dims = (lengths,) if isinstance(lengths, int) else lengths
        name = class_name(element)
        if not self.known(name):
            return None
        self.describe(name)
        tobject = self.tobject(name)
        kind = "TStreamerObject" if tobject else "TStreamerObjectAny"
        length = _product(dims)
        return (kind, field.name, field.comment, (OBJECT if tobject else ANY) + OFFSET_L,
                POINTER_SIZE * length, length, len(dims), (*dims, *NO_DIMS)[:5],
                name, ())  # fmt: skip

    def _basic(
        self, owner: Declaration, field: Field, found: Spelled, counter: bool
    ) -> Element | None:
        basic = self.basic(found)
        if basic is None or found.stars:
            return None
        stype, typename, size = basic
        dims = _dimensions(owner, field)
        if dims:
            length = _product(dims)
            maxidx = (*dims, *NO_DIMS)[:5]
            return ("TStreamerBasicType", field.name, field.comment, stype + OFFSET_L,
                    size * length, length, len(dims), maxidx, typename, ())  # fmt: skip
        if counter and stype == 3:
            stype = COUNTER
        return ("TStreamerBasicType", field.name, field.comment, stype, size, 0, 0, NO_DIMS,
                typename, ())  # fmt: skip

    def basic(self, found: Spelled) -> tuple[int, str, int] | None:
        """A number's streamer type, name and size - an enumeration's being an ``int``'s."""
        if found.args:
            return None
        if found.name in self.enums:
            return 3, found.name, 4
        return TYPES.get(found.name)

    def _counted(
        self, owner: Declaration, field: Field, found: Spelled, counter: bool
    ) -> Element | None:
        basic = self.basic(found) if found.stars == 1 else None
        count = _counter_of(field)
        if basic is None or count is None:
            return None
        stype, typename, size = basic
        counted = (_class_version(owner, owner.version), count, owner.name)
        return ("TStreamerBasicPointer", field.name, field.comment, stype + OFFSET_P, size,
                0, 0, NO_DIMS, typename + "*", counted)  # fmt: skip

    def _char_star(
        self, owner: Declaration, field: Field, found: Spelled, counter: bool
    ) -> Element | None:
        """``char *name``, not counted: a C string, ``kCharStar``."""
        if found.name != "char" or found.stars != 1 or field.dims or _counter_of(field):
            return None
        return ("TStreamerBasicType", field.name, field.comment, CHAR_STAR, POINTER_SIZE, 0, 0,
                NO_DIMS, "char*", ())  # fmt: skip

    def _string(
        self, owner: Declaration, field: Field, found: Spelled, counter: bool
    ) -> Element | None:
        if found.stars or field.dims:
            return None
        if found.name == "TString":
            return ("TStreamerString", field.name, field.comment, TSTRING, STRING_SIZE, 0, 0,
                    NO_DIMS, "TString", ())  # fmt: skip
        if found.name in ("string", "basic_string"):
            return ("TStreamerSTLstring", field.name, field.comment, STL, STRING_SIZE, 0, 0,
                    NO_DIMS, "string", (STL_STRING, STL_STRING))  # fmt: skip
        return None

    def _container(
        self, owner: Declaration, field: Field, found: Spelled, counter: bool
    ) -> Element | None:
        """``vector<T> name``, or the array of containers ``vector<T> name[N]``: a ROOT file
        describes both as one ``TStreamerSTL``, and the array with the array offset on its
        type, room for N containers and N as its length - as the oracle writes the
        ``vector<double> f[4]`` this class holds.
        """
        kind = STL_KINDS.get(found.name)
        if kind is None or found.stars:
            return None
        content = OBJECT if found.name in MAPS else self.content_type(found.args[0])
        if found.name in MAPS:
            self.describe(pair_name(found))
        dims = _dimensions(owner, field)
        if dims:
            length = _product(dims)
            return ("TStreamerSTL", field.name, field.comment, STL + OFFSET_L,
                    CONTAINER_SIZE * length, length, len(dims), (*dims, *NO_DIMS)[:5],
                    streamer_name(found), (kind, content))  # fmt: skip
        return ("TStreamerSTL", field.name, field.comment, STL, CONTAINER_SIZE, 0, 0, NO_DIMS,
                streamer_name(found), (kind, content))  # fmt: skip

    def _complex(
        self, owner: Declaration, field: Field, found: Spelled, counter: bool
    ) -> Element | None:
        """``std::complex<T>``: ROOT's dictionary describes one as the pair of ``T`` it is -
        ``_real`` then ``_imag``, under the name ``complex<T>``, with no version (its objects
        carry that description's checksum instead) - and streams it in place, as
        ``TStreamerObjectAny``, so a member of one takes the room of its two parts rather than
        the pointer a member of a class held by value takes.

        The catalogue was told of the class before the rules ran, so a pointer to one is an
        ``TStreamerObjectAnyPointer`` whose class is known and a container of one is written
        member-wise, as ROOT writes a container of objects; this rule is what lays out the
        member that is the complex itself.
        """
        made = complex_declaration(found)
        if made is None:
            return None
        name, _declared, size = made
        if found.stars:  # a pointer to one: the object rules lay that out, now that it is known
            return None
        self.describe(name)
        return _complex_element(field, name, size, _dimensions(owner, field))

    def content_type(self, content: Spelled) -> int:
        """What a container's ``ctype`` says it holds: a number's type, or an object."""
        basic = self.basic(content) if not content.stars else None
        if basic is not None:
            return basic[0]
        if content.stars:
            self._require(content.name)
            return OBJECT_P
        if content.name not in STL_KINDS and content.name not in ("string", "TString"):
            name = class_name(content)
            self._require(name)
            self.describe(name)
        return OBJECT

    def _require(self, name: str) -> None:
        if not self.known(name):
            raise UnsupportedFeatureError(
                f"{name} is not a class the program declares nor one this writer carries a "
                f"layout for, so an object holding one cannot be written"
            )

    def _object(
        self, owner: Declaration, field: Field, found: Spelled, counter: bool
    ) -> Element | None:
        name = class_name(found)
        if found.name in STL_KINDS or found.stars > 1 or not self.known(name):
            return None
        if field.dims:  # T a[N] of objects: the array a std::array<T,N> is
            return self._object_array(owner, field, found, _product(_dimensions(owner, field)))
        self.describe(name)
        tobject = self.tobject(name)
        if not found.stars:
            kind = "TStreamerObject" if tobject else "TStreamerObjectAny"
            return (kind, field.name, field.comment, OBJECT if tobject else ANY, POINTER_SIZE,
                    0, 0, NO_DIMS, name, ())  # fmt: skip
        stype = _POINTERS[tobject, field.comment.startswith("->")]
        kind = "TStreamerObjectPointer" if tobject else "TStreamerObjectAnyPointer"
        return (kind, field.name, field.comment, stype, POINTER_SIZE, 0, 0, NO_DIMS,
                name + "*", ())  # fmt: skip


#: A pointer's streamer type, by whether it points at a ``TObject`` and whether ``//->``
#: promises it is never null.
_POINTERS = {(True, True): OBJECT_P, (True, False): OBJECT_PP,
             (False, True): ANY_P, (False, False): ANY_PP}  # fmt: skip


#: How a member is described, tried in order until one recognises it.
_MEMBER_RULES: tuple[
    Callable[[Catalogue, Declaration, Field, Spelled, bool], Element | None], ...
] = (
    Catalogue._fixed_array,
    Catalogue._basic,
    Catalogue._counted,
    Catalogue._char_star,
    Catalogue._string,
    Catalogue._container,
    Catalogue._complex,
    Catalogue._object,
)


def pair_name(found: Spelled) -> str:
    """The pair a map holds, as ROOT names it: ``pair<int,string>``."""
    key, value = found.args[0], found.args[1]
    return Spelled("pair", (Spelled(streamer_name(key), (), 0),
                            Spelled(streamer_name(value), (), 0)), 0).text()  # fmt: skip


def complex_declaration(found: Spelled) -> tuple[str, Declaration, int] | None:
    """What a type is of ``std::complex``: the name ROOT gives the class - ``complex<float>`` -
    the description its dictionary has of it - two numbers and no ``ClassDef``, so that its
    objects carry that description's checksum where a version would go - and how wide each of
    those numbers is, or ``None`` for a type that is no complex or one of a width this writer
    has no number for.

    A writer that is handed the class by name alone, as ``WriteObjectAny`` hands one, needs the
    description as much as a class holding one does, which is why it is made here.
    """
    if found.name not in COMPLEX_TEMPLATES or len(found.args) != 1:
        return None
    part = found.args[0]
    basic = TYPES.get(part.name) if not part.args and not part.stars else None
    if basic is None:
        return None
    typename = basic[1]
    name = Spelled("complex", (Spelled(typename, (), 0),), 0).text()
    fields = (Field("_real", typename, "", ()), Field("_imag", typename, "", ()))
    return name, Declaration(name, 1, True, (), fields), basic[2]


def _note_complexes(catalogue: Catalogue, found: Spelled) -> None:
    """See that every complex a member's type names is described, wherever it stands: a
    container of one is written member-wise and a pointer to one names it, and both ask the
    catalogue for its layout before the member's own rule is tried."""
    made = complex_declaration(found)
    if made is not None:
        catalogue.declarations.setdefault(made[0], made[1])
    for arg in found.args:
        _note_complexes(catalogue, arg)


def _complex_element(field: Field, name: str, size: int, dims: tuple[int, ...]) -> Element:
    """The element a member is: the two parts in place - ``kAny`` for one complex and
    ``kOffsetL + kAny`` for an array of them, room for every one of them - which is how ROOT
    streams a complex, rather than as a class held by value takes a pointer's room."""
    length = _product(dims) if dims else 1
    return ("TStreamerObjectAny", field.name, field.comment, ANY + OFFSET_L if dims else ANY,
            size * 2 * length, length if dims else 0, len(dims),
            (*dims, *NO_DIMS)[:5] if dims else NO_DIMS, name, ())  # fmt: skip


def _pair(name: str) -> Declaration:
    """The pair a map holds, as ROOT emulates it: ``first`` and ``second``, unversioned."""
    found = parse_type(name)
    first, second = (arg.text() for arg in found.args)
    fields = (Field("first", first, "", ()), Field("second", second, "", ()))
    return Declaration(name, 1, True, (), fields)


def _class_version(declared: Declaration | None, version: int) -> int:
    """The version ``TClass`` gives a class: -1 for a foreign one, whose layouts are told apart
    by checksum and whose descriptions are numbered 1 all the same."""
    return -1 if declared is not None and declared.foreign else version


def _counter_of(field: Field) -> str | None:
    """The member a ``//[n]`` comment names as how many values a pointer points to."""
    found = re.match(r"\[\s*(\w+)\s*\]", field.comment)
    if found is None or "*" not in field.spelled:
        return None
    return found.group(1)


def _whole(spelled: Spelled) -> int | None:
    """A template argument written as a whole number, which ``std::array``'s length is."""
    if spelled.args or spelled.stars:
        return None
    try:
        return int(spelled.name)
    except ValueError:
        return None


#: The most array dimensions ROOT's ``TStreamerElement`` records, ``fMaxIndex``'s five.
MAX_DIMS = 5


def _array_shape(found: Spelled) -> tuple[Spelled, tuple[int, ...]] | None:
    """A fixed array taken apart: the type it holds, and its lengths outermost first - from
    ``std::array<std::array<int,4>,3>`` the ``int`` and ``(3, 4)``.

    None for anything that is not a fixed array at all, and for one whose length nothing can
    count at compile time, which ``_fixed_array``'s caller then refuses by name.
    """
    lengths: list[int] = []
    while found.name in ARRAY_TEMPLATES and len(found.args) == 2 and not found.stars:
        element, count = found.args
        length = _whole(count)
        if length is None or length < 1:
            return None
        lengths.append(length)
        found = element
    if not lengths or found.stars or found.name in ARRAY_TEMPLATES:
        return None
    return found, tuple(lengths)


def _dimensions(owner: Declaration, field: Field) -> tuple[int, ...]:
    """A member's array lengths, which a file's layout says as numbers - a length worked out
    at run time is refused by name, since nothing in the record could hold it."""
    if any(dimension is None for dimension in field.dims):
        raise UnsupportedFeatureError(
            f"{owner.name}::{field.name} is a {field.spelled} whose array length is worked out "
            f"at run time, and a ROOT file's layout says a number of values: give the member a "
            f"length C++ can count at compile time, or mark it transient with //! to leave it "
            f"out of the file")
    return tuple(int(dimension) for dimension in field.dims)


def _product(dims: tuple[int, ...]) -> int:
    total = 1
    for dim in dims:
        total *= dim
    return total


def _base_element(name: str) -> Element | None:
    """The base element ROOT's own classes give the class ``name``, title and all."""
    for _sum, _version, elements in INFOS.values():
        for element in elements:
            if element[0] == "TStreamerBase" and element[1] == name:
                return element
    return None
