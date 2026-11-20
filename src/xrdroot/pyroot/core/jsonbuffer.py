"""``TBufferJSON``: an object as the JSON ROOT's own exporter writes it.

ROOT exports an object as the JSON its streamers describe it in - one member
per line, in streamer order, arrays of numbers as JSON arrays - and hands the
text to a program through ``TBufferJSON::ConvertToJSON``. A browser reads the
same text back through JSROOT, which is why it is compressed the way it is: a
hundred numbers that are mostly zero and run in tens become
``{"$arr":"Int32","len":100,"p":17,"v":7,"n":10}``, and a ``std::map`` asked
for with ``kMapAsObject`` becomes a JSON object.

This is that export, member for member and space for space, at each of ROOT's
four compact levels: the indented text ``kNoCompress`` writes, the unindented
and the single-line forms after it, and ``kNoSpaces``. A class's members come
from :data:`MEMBERS`, a table rather than a walk of the object, because ROOT
writes what the class's *streamer* writes - for a class with a hand-written
streamer, not every data member: ``TPolyMarker3D`` writes ``fN``, ``fP``,
``fOption`` and ``fName`` and never ``fLastPoint``. A class a macro declared
needs no table: the translator read its declaration, and
:func:`_declared` writes the members - its bases' first - in the order that
declaration gives them.

The numbers are :func:`_float_text`'s: whole values below ``1e15`` (a float)
or ``1e25`` (a double) written out digit by digit, and everything else from
``%.7g`` or ``%.15g`` with ROOT's own exponent - no ``+``, no zero padding -
and, below ``0.001``, as an exponent rather than the string of zeros it would
otherwise be. Each rule there was checked against ``TBufferText``'s
``ConvertDouble`` and ``ConvertFloat`` in the 6.40.04 oracle.

What it writes is what it can write *exactly*. Anything else is refused by
name, in a full sentence: a plausible-looking JSON that ROOT reads back as
something else is worse than an error.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...cppstd.containers import _Map, cpp_name, std, string
from ...errors import UnsupportedFeatureError
from ..vectors import TArrayC, TArrayD, TArrayF, TArrayI, TArrayL, TArrayS
from .strings import TString

__all__ = ["TBufferJSON"]

#: The shortest array ROOT compresses at all: below this the ``$arr`` object that says what
#: the numbers are would be larger than they are.
ARRAY_MIN = 6

#: How many times one value must repeat, in a row, for ROOT to write the value and a count.
RUN_MIN = 6

#: How many bytes of a base64 array's leading zeros ROOT drops before it is worth the ``o``
#: that says where they end, which costs more than the bytes it saves below this.
BYTES_MIN = 4

#: How many zeros in a row ROOT treats as a gap rather than as data: it writes the values on
#: either side as two stretches, the zeros between them nothing. Fewer are kept where they
#: are - inside a stretch, as the numbers it holds.
ZERO_MIN = 10

#: Below this a whole value is written out digit by digit, and above it the exponent form
#: is, since a ``Float_t`` (7 digits) or a ``Double_t`` (15) shows nothing more of it.
WHOLE = {7: 1e15, 15: 1e25}

#: The significant digits each ``$arr`` type keeps.
PRECISION = {name: (7 if np.dtype(kind).itemsize == 4 and np.dtype(kind).kind == "f" else 15)
             for name, kind in {"Int8": np.int8, "UInt8": np.uint8, "Int16": np.int16,
                                "UInt16": np.uint16, "Int32": np.int32, "UInt32": np.uint32,
                                "Int64": np.int64, "Float32": np.float32,
                                "Float64": np.float64, "Bool": np.bool_}.items()}  # fmt: skip

#: What JSON has no number for: infinity as a number too large to read back - JSROOT's own
#: trick, and its negative - and NaN as ``null``. With ``kStoreInfNaN`` ROOT names them
#: as strings instead, and a ``Float_t``'s with the ``f`` a C++ float literal carries.
SPECIAL = {
    "Float32": {"inf": "inff", "-inf": "-inff", "nan": "nanf"},
    "Float64": {"inf": "inf", "-inf": "-inf", "nan": "nan"},
}

#: The NumPy type each ``$arr`` type name stands for, and back.
DTYPES = {"Int8": np.dtype(np.int8), "UInt8": np.dtype(np.uint8),
          "Int16": np.dtype(np.int16), "UInt16": np.dtype(np.uint16),
          "Int32": np.dtype(np.int32), "UInt32": np.dtype(np.uint32),
          "Int64": np.dtype(np.int64), "Float32": np.dtype(np.float32),
          "Float64": np.dtype(np.float64), "Bool": np.dtype(np.bool_)}  # fmt: skip

#: The ``$arr`` type name of each array class ROOT writes as numbers. ``TArrayC`` is
#: missing because ROOT writes a ``char`` array as one JSON string.
ARRAY_NAMES = {TArrayS: "Int16", TArrayI: "Int32", TArrayL: "Int64",
               TArrayF: "Float32", TArrayD: "Float64"}  # fmt: skip

#: The class each ``$arr`` type name stands for, for reading one back.
ARRAY_TYPES = {name: kind for kind, name in ARRAY_NAMES.items()}

#: The same names for the NumPy arrays a drawn object keeps its numbers in.
NUMPY_NAMES = {kind: name for name, kind in DTYPES.items()}

#: The classes a macro names the way ROOT does, for reading JSON asked for by class.
ARRAY_CLASSES = {kind.__name__: kind for kind in (TArrayC, *ARRAY_NAMES)}


@dataclass(frozen=True)
class _Layout:
    """How one of ROOT's four compact levels lays the JSON out.

    ``member_comma`` is what stands between two members of an object - the newline the
    indented levels break the line with, the indent going after it - and ``item_comma``
    what stands between two numbers of an array, a space at every level but ``kNoSpaces``.
    """

    newline: bool
    indent: int
    colon: str
    member_comma: str
    item_comma: str


#: The four levels, the three flag levels above them ROOT counts in fives.
LAYOUTS = {
    0: _Layout(True, 2, " : ", ",\n", ", "),
    1: _Layout(True, 0, " : ", ",\n", ", "),
    2: _Layout(False, 0, " : ", ",", ", "),
    3: _Layout(False, 0, ":", ",", ","),
}


@dataclass(frozen=True)
class _Member:
    """One data member of a class this writes: its name, the C++ type it goes out as, and
    whether it is an array of them - with the lengths declared for it, which is what makes
    ``int fTest2[2][2]`` the nested arrays ROOT writes rather than one flat one."""

    name: str
    kind: Any
    array: bool = False
    dims: tuple[int | None, ...] = ()


#: Each class this writes, its members in the order ROOT's streamer writes them, with the
#: type each goes out as. ``TPolyMarker3D``'s ``fName`` comes after its own members because
#: ``TNamed`` is not its only base, and ``fLastPoint`` is absent because its hand-written
#: streamer never writes it.
MEMBERS: dict[str, tuple[_Member, ...]] = {
    "TObject": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32)),
    "TNamed": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32),
               _Member("fName", "string"), _Member("fTitle", "string")),
    "TObjString": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32),
                   _Member("fString", "string")),
    "TPolyMarker3D": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32),
                      _Member("fMarkerColor", np.int16), _Member("fMarkerStyle", np.int16),
                      _Member("fMarkerSize", np.float32), _Member("fN", np.int32),
                      _Member("fP", np.float32, True), _Member("fOption", "string"),
                      _Member("fName", "string")),
    "TPolyLine3D": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32),
                    _Member("fLineColor", np.int16), _Member("fLineStyle", np.int16),
                    _Member("fLineWidth", np.int16), _Member("fN", np.int32),
                    _Member("fP", np.float32, True), _Member("fOption", "string")),
}

#: Where a core class keeps a member ROOT names otherwise: ``TObject``'s ``fUniqueID`` is
#: the object's ``_unique_id``, and a ``TObjString``'s one string is its ``fName``.
ATTRIBUTES = {"fUniqueID": "_unique_id", "fBits": "_bits", "fName": "_name",
              "fTitle": "_title", "fString": "_name"}  # fmt: skip

#: The bits a class's *own* constructor sets and the object here does not carry, so that the
#: JSON would describe one ROOT never makes: ``TPolyMarker3D`` calls ``SetBit(kCanDelete)``,
#: and ``PolyMarker.ref`` shows the marker's ``fBits`` as 1 for that bit.
CLASS_BITS = {"TPolyMarker3D": 1}

#: The value a class's constructor gives a member the object here leaves unset.
#: ``TPolyMarker3D``'s ``fName`` is its own ``TString`` member, not the ``TNamed`` name it
#: does not inherit, and its constructor names the marker after its class.
DEFAULTS = {("TPolyMarker3D", "fName"): "TPolyMarker3D"}

#: The ``$arr`` type name each scalar a C++ member can be declared as goes out as: a
#: ``char`` is an integer, since ROOT writes one character of a member as its code point,
#: which is why ``TJsonEx4``'s ``fChar`` is 67 and not ``"C"``.
SCALARS = {"bool": np.bool_, "char": np.int8, "signed char": np.int8,
           "unsigned char": np.uint8, "short": np.int16, "unsigned short": np.uint16,
           "int": np.int32, "unsigned int": np.uint32, "long": np.int64,
           "unsigned long": np.uint64, "long long": np.int64,
           "float": np.float32, "double": np.float64}  # fmt: skip

#: The members that hold text rather than a number, by the C++ type they are declared as and
#: whether they are an array of it: a ``std::string``, a ``char[N]`` ROOT writes as one JSON
#: string up to its first NUL, and the pointers to one, which ROOT writes as ``null`` where
#: there is nothing there.
TEXTS = {("std::string", False): "string", ("TString", False): "string",
         ("std::string", True): "strings", ("std::string*", True): "strptrs",
         ("TString", True): "tstrings", ("TString*", True): "tstrings",
         ("char", True): "chars", ("char*", False): "cstring",
         ("std::string*", False): "strptr"}  # fmt: skip

#: The C++ containers ROOT writes as the values they hold: a sequence as one JSON array of
#: them, a map as a ``$pair`` per entry, and a ``bitset`` as a boolean per bit. A member
#: declared as one of these is no scalar, so the table of scalar types says nothing of it.
CONTAINERS = ("std::vector", "std::list", "std::deque", "std::set", "std::multiset",
              "std::map", "std::multimap", "std::bitset")  # fmt: skip

#: The ``_Member.kind`` a container is written as, which is also what the reader looks at:
#: a sequence or map of values, one that names itself an object of its keys, or its bits.
CONTAINER_KINDS = ("container", "map_object", "bitset")


def _remainder(value: int, mod: int) -> int:
    """``value % mod`` with the sign of ``value``, as C's ``%`` has it."""
    return int(math.fmod(value, mod))


def _digit(compact: int, place: int) -> int:
    """The digit of the compact level at ``place`` - the units, the tens, the hundreds, the
    thousands - as ROOT's ``SetCompact`` reads it: C's division truncates towards zero, so
    a negative level has the digits round it rather than borrowing from them."""
    whole = abs(compact) // place
    return _remainder(whole if compact >= 0 else -whole, 10)


def _trimmed(text: str) -> str:
    """``text`` without the zeros a float's last places keep, and without a bare ``.``."""
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def _special(value: float, name: str, stored: bool) -> str | None:
    """What a value JSON has no number for is written as, or ``None`` for one it has:
    ``2e308`` or ``"inf"``, ``null`` or ``"nan"``, as ``kStoreInfNaN`` asks."""
    if value != value:
        return '"' + SPECIAL[name]["nan"] + '"' if stored else "null"
    if value in (math.inf, -math.inf):
        sign = "-" if value < 0 else ""
        return '"' + SPECIAL[name][sign + "inf"] + '"' if stored else sign + "2e308"
    return None


def _exponent(value: float, digits: int) -> str:
    """``value`` as a mantissa and an exponent, with ROOT's exponent: no ``+``, no zeros."""
    mantissa, exponent = format(value, f".{digits - 1}e").split("e")
    return _trimmed(mantissa) + "e" + str(int(exponent))


def _float_text(value: float, name: str, stored: bool) -> str:
    """A ``Float32`` or a ``Float64`` as ROOT writes it, which is what JSROOT has always
    been handed: a whole value below :data:`WHOLE` digit by digit, and anything else from
    ``%.7g``/``%.15g`` - as an exponent when it is under ``0.001``, where the digits would
    be a string of zeros, and with the ``+`` and padding taken out of one ROOT does keep."""
    special = _special(value, name, stored)
    if special is not None:
        return special
    digits = PRECISION[name]
    if abs(value) < WHOLE[digits] and value == math.floor(value):
        return format(value, ".0f")
    if abs(value) < 1e-3:
        return _exponent(value, digits)
    text = format(value, f".{digits}g")
    return _exponent(value, digits) if "e" in text else _trimmed(text)


def _quoted(text: str) -> str:
    """A JSON string: what JavaScript's ``JSON.parse`` needs escaped, and the solidus with
    it, which ROOT escapes too."""
    return json.dumps(str(text)).replace("/", "\\/")


def _scalar(value: Any, name: str, stored: bool) -> str:
    """One value of ``$arr`` type ``name``, as ROOT writes that type."""
    kind = DTYPES[name]
    if kind.kind == "b":
        return "true" if value else "false"
    if kind.kind in "iu":
        return str(int(value))
    return _float_text(float(value), name, stored)


def _pointer_bits(bits: Any) -> int:
    """The bits of a ``TObject`` ROOT writes: its status word without the heap's two, which
    say where the object is rather than what it is."""
    return int(bits) & 0x00FFFFFF


def _classname(obj: Any) -> str:
    """The ROOT class an object is, by the name its JSON gives it: the object's own where it
    has one, and otherwise what the translation declared it as."""
    named = getattr(obj, "ClassName", None)
    given = str(named()) if callable(named) else str(getattr(obj, "classname", ""))
    return given or (_declared(obj) or ("",))[0]


def _declared(obj: Any) -> tuple[str, tuple[_Member, ...]] | None:
    """The members the translation declared for the class of ``obj``, with that class's name,
    or ``None`` for an object whose class the translation did not declare.

    A class a macro defined is written the way its own declaration says, so its members do
    not have to be listed here: ``_cxx_layout_`` is what the translator read off its base and
    member declarations. A member this cannot write refuses the whole object by name, since a
    member left out is a JSON that says the object holds less than it does.
    """
    return _layout(type(obj))


#: What :func:`_layout` found for a class, by the class: its name and members, or ``None``
#: for a class the translation did not declare.
_DECLARED: dict[type[Any], tuple[str, tuple[_Member, ...]] | None] = {}


def _layout(cls: type[Any]) -> tuple[str, tuple[_Member, ...]] | None:
    """``cls``'s members in the order its streamer writes them - its bases' first, then its
    own - looked at once and kept."""
    if cls not in _DECLARED:
        _DECLARED[cls] = _read_layout(cls)
    return _DECLARED[cls]


def _read_layout(cls: type[Any]) -> tuple[str, tuple[_Member, ...]] | None:
    """The members ``_cxx_layout_`` declares for ``cls``, or ``None`` for a class the
    translation did not declare."""
    layout = getattr(cls, "_cxx_layout_", None)
    if not (isinstance(layout, tuple) and len(layout) == 3):
        return None
    name, bases, writable = layout
    made = tuple(member for base in bases for member in _base_members(cls, str(base)))
    made += tuple(_declared_member(str(name), entry) for entry in writable)
    return str(name), made


def _base_members(cls: type[Any], name: str) -> tuple[_Member, ...]:
    """The members of ``cls``'s base class ``name``, whose streamer writes them before the
    object's own: a member of a base ROOT's streamer never writes - ``TObject``'s - is left
    out of the layout, and out of the JSON.

    A class derived from a container inherits no members of its own: ROOT writes the values
    it was made holding under ``fVector``, named after the kind of container it is."""
    for kind in cls.__mro__:
        layout = getattr(kind, "_cxx_layout_", None)
        if isinstance(layout, tuple) and layout[0] == name:
            found = _layout(kind)
            return found[1] if found else ()
    if name.startswith("std::vector"):
        return (_Member("fVector", "container", False, ()),)
    return ()


def _writable(obj: Any) -> tuple[_Member, ...]:
    """The members ``obj`` is written as: the ones its table gives it, or the ones its own
    declaration does."""
    found = MEMBERS.get(_classname(obj))
    if found is not None:
        return found
    return (_declared(obj) or ("", ()))[1]


#: What a member's declaration comment says about how many values it holds: ROOT's
#: ``// [fSize]``, which its streamer reads as "as many as ``fSize`` says" for a pointer.
#: The comment is kept as the declaration spells it, so one leading ``/`` of ``///`` is gone.
LENGTH = re.compile(r"\[\s*\w+\s*\]")


def _declared_member(classname: str, entry: tuple[str, str, str, tuple[int, ...]]) -> _Member:
    """One member of a declared class, as the type it goes out as: a scalar, a piece of text,
    or the refusal that names the member and what it is declared as.

    A pointer whose comment declares its length - ``bool *fBool; // [fSize]`` - is the
    array that length says, which is how ROOT's streamer reads it: a dynamic array written
    as the numbers it holds, not as an address.
    """
    name, kind, comment = entry[0], entry[1], entry[2]
    dims: tuple[int | None, ...] = entry[3]
    if not dims and kind.endswith("*") and LENGTH.search(comment or ""):
        kind, dims = kind.rstrip("*").strip(), (None,)
    of = f"an array of {kind}" if dims else f"a {kind}"
    style = _container_style(kind, comment)
    if style is not None:
        return _Member(name, style, bool(dims), dims)
    text = TEXTS.get((kind, bool(dims)))
    if text is not None:
        return _Member(name, text, bool(dims), dims)
    scalar = SCALARS.get(kind)
    if scalar is not None:
        return _Member(name, scalar, bool(dims), dims)
    raise UnsupportedFeatureError(
        f"TBufferJSON::ConvertToJSON does not yet write {classname}::{name}, which is "
        f"declared as {of}. It writes the members a class declares of a number type "
        f"({', '.join(SCALARS)}) and of text (std::string, TString, char[N], char*), and "
        f"the members of the classes in its table ({', '.join(sorted(MEMBERS))}); a "
        "container member needs a writer for the container it holds first"
    )


def _container_style(kind: str, comment: str) -> str | None:
    """How ROOT writes a member declared as ``kind``: a sequence or map of the values it
    holds, one JSON object of its keys where its declaration's comment says
    ``/ JSON_object``, or a boolean per bit; ``None`` for anything but a container."""
    bare = kind.rstrip("*").strip()
    if not bare.startswith(CONTAINERS):
        return None
    if bare.startswith("std::bitset"):
        return "bitset"
    return "map_object" if "JSON_object" in (comment or "") else "container"


def _held(obj: Any, member: _Member) -> Any:
    """The value of ``member`` in ``obj``: one of the members a drawn object keeps by name,
    where ``fBits`` carries the heap's bits too, or the attribute a core class keeps it in.

    A member the object leaves empty takes the value its class's constructor gives it, and
    the bits its class's constructor sets are added to the object's own, since what ROOT
    writes is the object its constructor made rather than the one here."""
    classname = _classname(obj)
    value = _recorded(obj, member)
    if value is None and member.name == "fVector" and _inherits_container(obj, member):
        value = obj
    default = DEFAULTS.get((classname, member.name))
    if default is not None and not (isinstance(value, str) and value):
        value = default
    if member.name == "fBits":
        return _pointer_bits(value or 0) | CLASS_BITS.get(classname, 0)
    return value


def _recorded(obj: Any, member: _Member) -> Any:
    """The value ``obj`` keeps for ``member``: by name where it holds its members in a dict,
    as a drawn object does, else the attribute its declaration names, and last the one a core
    class keeps it in."""
    found = getattr(obj, "members", None)
    declared = getattr(type(obj), "_cxx_layout_", None)
    if isinstance(found, dict):
        value: Any = found.get(member.name)
    elif isinstance(declared, tuple):
        value = getattr(obj, member.name, None)
    else:
        value = None
    if value is None:
        value = getattr(obj, ATTRIBUTES.get(member.name, "_" + member.name), None)
    return value


def _inherits_container(obj: Any, member: _Member) -> bool:
    """Is ``member`` the synthetic one ROOT writes for a container the class inherits?
    ``TJsonEx8 : std::vector<int>`` has no member of its own holding the base's values - the
    object is the vector - and ROOT writes them as ``fVector``, named after the base's kind."""
    if member.name != "fVector" or member.kind not in CONTAINER_KINDS:
        return False
    layout = getattr(type(obj), "_cxx_layout_", None)
    bases = layout[1] if isinstance(layout, tuple) and len(layout) == 3 else ()
    return any(str(base).startswith("std::vector") for base in bases)


def _content(values: list[Any]) -> tuple[int, int]:
    """Where the first and the last value there is are: the array without its zero ends."""
    first, last = 0, len(values) - 1
    while first <= last and not values[first]:
        first += 1
    while last >= first and not values[last]:
        last -= 1
    return first, last


def _run(values: list[Any], at: int, last: int) -> int:
    """How many times the value at ``at`` repeats, up to ``last``."""
    count = 1
    while at + count <= last and values[at + count] == values[at]:
        count += 1
    return count


def _breaker(values: list[Any], at: int, last: int) -> int:
    """How many values the run at ``at`` covers when ROOT writes it as one value and a count,
    and 0 otherwise: ``RUN_MIN`` of one value, and ``ZERO_MIN`` zeros - which are written as
    nothing at all rather than counted, an array's zeros being what it does not hold."""
    count = _run(values, at, last)
    return count if count >= (ZERO_MIN if not values[at] else RUN_MIN) else 0


def _stretch(values: list[Any], at: int, last: int) -> int:
    """Where the values from ``at`` stop being worth writing one by one: at the run ROOT
    would write as a count, or at the last value there is."""
    while at <= last and not _breaker(values, at, last):
        at += 1
    return at


def _pieces(values: list[Any], same: bool) -> list[tuple[int | None, list[Any], int | None]]:
    """The stretches ROOT keeps of an array: each one's position - ``None`` for one that
    follows the stretch before it - the value or values it holds, and how many times a
    single value repeats.

    Zeros at either end are dropped, so the first position is where the first value there
    is; the zeros that start a stretch are dropped the same way, which is why a zero inside
    a stretch is kept while a gap of ten moves the next stretch along.
    """
    _, last = _content(values)
    pieces: list[tuple[int | None, list[Any], int | None]] = []
    at = follows = 0
    while True:
        at = _skipped(values, at, last)
        if at > last:
            return pieces
        stop, count = _at(values, at, last, same)
        held = values[at:at + 1] if count else values[at:stop]
        pieces.append((at if at != follows else None, held, count or None))
        at = follows = stop


def _skipped(values: list[Any], at: int, last: int) -> int:
    """Where the next value there is: a stretch never starts with the zeros that mean an
    array holds nothing there."""
    while at <= last and not values[at]:
        at += 1
    return at


def _at(values: list[Any], at: int, last: int, same: bool) -> tuple[int, int]:
    """Where the stretch starting at ``at`` ends, and how many times one value repeats in it
    - its own length, where it is the run ROOT writes as one value and a count."""
    count = _breaker(values, at, last) if same else 0
    if count:
        return at + count, count
    return (_stretch(values, at, last) if same else last + 1), 0


def _chunk(suffix: str, position: int | None, held: list[str], count: int | None,
           comma: str) -> list[str]:  # fmt: skip
    """One stretch of a compressed array: where it starts, the value or values it holds, and
    how many times the one value repeats."""
    pairs = [] if position is None else [f'"p{suffix}":{position}']
    pairs.append(f'"v{suffix}":' + (held[0] if len(held) == 1 else "[" + comma.join(held) + "]"))
    if count is not None:
        pairs.append(f'"n{suffix}":{count}')
    return pairs


def _nested_dimensions(dims: tuple[int | None, ...]) -> bool:
    """Do ``dims`` describe an array of arrays: more than one length, each a number the
    declaration fixed, and so nested JSON arrays rather than one flat one."""
    return len(dims) > 1 and all(isinstance(dim, int) for dim in dims)


def _nest(values: list[Any], dims: tuple[int | None, ...],
          leaf: Callable[[list[Any]], Any]) -> Any:  # fmt: skip
    """``values``, a flat array, as the nested lists ``dims`` describe, each group of the
    innermost length written by ``leaf`` - a ``char[N]`` row as one JSON string."""
    if len(dims) == 1:
        return leaf(values)
    step = int(np.prod(dims[1:]))
    return [_nest(values[at : at + step], dims[1:], leaf) for at in range(0, len(values), step)]


def _text_tree(value: Any, comma: str) -> str:
    """The strings of a ``TString`` array as the JSON ROOT writes them, at whatever depth the
    object holds them: ``null`` for a pointer with nothing there, and one array per level."""
    if isinstance(value, (list, tuple)):
        return "[" + comma.join(_text_tree(each, comma) for each in value) + "]"
    return "null" if value is None else _quoted(str(value))


def _list_text(nested: Any, comma: str) -> str:
    """Nested JSON arrays of text, which ``leaf`` has already written as JSON."""
    if isinstance(nested, str):
        return nested
    return "[" + comma.join(_list_text(each, comma) for each in nested) + "]"


def _char_codes(value: Any) -> list[int]:
    """The code points of a ``char`` member's characters, however the class holds them: a
    Python ``str``, the buffer of an array, or the strings of a ``char[N][M]``'s rows."""
    if isinstance(value, str):
        return [ord(char) for char in value]
    if isinstance(value, list):
        return [code for each in value for code in _char_codes(each)]
    return [int(code) for code in np.asarray(value, dtype=np.int8).ravel()]


def _flat_numbers(value: Any) -> list[Any]:
    """The numbers a JSON array holds, nested as deeply as ROOT wrote it."""
    if isinstance(value, list):
        return [number for each in value for number in _flat_numbers(each)]
    return [value]


class _Writer:
    """One export: the JSON for an object, its members and their arrays, laid out as the
    compact level asks and compressed the way JSROOT reads."""

    def __init__(self, compact: int) -> None:
        #: The units digit: the four layouts, and the three flag levels counted above them,
        #: which leave it their remainder - ``kMapAsObject`` is 5, so it is 0 again, and 8
        #: and 9 are ``kNoSpaces``.
        level = _digit(compact, 1)
        self._layout = LAYOUTS[max(0, min(_remainder(level, 5), 3))]
        #: ``kZeroSuppression`` (1), ``kSameSuppression`` (2) or ``kBase64`` (3), as tens.
        self._compress = _digit(compact, 10)
        self._map_as_object = level >= 5
        self._skip_typename = _digit(compact, 100) == 1
        self._store_inf = _digit(compact, 1000) == 1

    # -- what to write ----------------------------------------------------------------

    def text(self, obj: Any) -> str:
        return self._value(obj, 0)

    def member(self, obj: Any, name: str) -> str:
        """The JSON of one member of ``obj``, which is what ``ToJSON(obj, level, member)``
        writes in place of the whole object."""
        members = _writable(obj)
        for member in members:
            if member.name == name:
                return self._member(member, obj, 0)
        raise UnsupportedFeatureError(
            f"{_classname(obj) or type(obj).__name__} has no member {name!r} for "
            "TBufferJSON to write on its own; it writes a whole object of a class whose "
            "members it knows, and one of them where the name is one of "
            + (
                ", ".join(member.name for member in members)
                or "the members of a class in its table"
            )
        )

    def _value(self, obj: Any, depth: int) -> str:
        """``obj`` as JSON: a number, a string, an array, a map, or an object."""
        for kind, write in WRITERS:
            if isinstance(obj, kind):
                return write(self, obj, depth)
        if _writable(obj):
            return self._object(obj, depth)
        raise UnsupportedFeatureError(
            "TBufferJSON::ConvertToJSON does not yet write the JSON of a "
            f"{_classname(obj) or type(obj).__name__}: it writes numbers and strings, the "
            f"TArrays ({', '.join(sorted(ARRAY_CLASSES))}), a std::map with the pairs it "
            "holds, and the classes whose members are in its table "
            f"({', '.join(sorted(MEMBERS))}). Writing another one means adding its "
            "members there, in the order its streamer writes them"
        )

    # -- the leaves -------------------------------------------------------------------

    def _null(self, obj: Any, depth: int) -> str:
        return "null"

    def _boolean(self, obj: Any, depth: int) -> str:
        return "true" if obj else "false"

    def _integer(self, obj: Any, depth: int) -> str:
        return str(int(obj))

    def _float(self, obj: Any, depth: int) -> str:
        return _float_text(float(obj), "Float64", self._store_inf)

    def _string(self, obj: Any, depth: int) -> str:
        return _quoted(obj)

    def _text_array(self, obj: TArrayC, depth: int) -> str:
        """``TArrayC``: ROOT writes the characters themselves, as one JSON string."""
        return _quoted("".join(chr(int(value) & 0xFF) for value in obj))

    # -- objects ----------------------------------------------------------------------

    def _object(self, obj: Any, depth: int) -> str:
        members = [self._pair(member, obj, depth) for member in _writable(obj)]
        return self._braced(self._typed(members, _classname(obj), self._layout.colon), depth)

    def _typed(self, written: list[str], classname: str, colon: str) -> list[str]:
        """``written`` with the ``_typename`` ROOT puts in front of it, unless the level
        says to skip the type information."""
        if self._skip_typename:
            return written
        return [_quoted("_typename") + colon + _quoted(classname), *written]

    def _pair(self, member: _Member, obj: Any, depth: int) -> str:
        return _quoted(member.name) + self._layout.colon + self._member(member, obj, depth)

    def _member(self, member: _Member, obj: Any, depth: int) -> str:
        value = _held(obj, member)
        if isinstance(member.kind, str):
            return self._text_member(member, value, depth)
        return self._number_member(member, value, depth)

    def _text_member(self, member: _Member, value: Any, depth: int) -> str:
        """A member that is not a number: text, an array of text, or a container."""
        if member.kind == "chars":
            return self._char_text(value, member)
        if member.kind in ("strings", "strptrs", "tstrings"):
            return self._texts(member, value, depth)
        if member.kind in CONTAINER_KINDS:
            return self._containers(member, value, depth)
        if member.kind in ("cstring", "strptr"):
            return "null" if value is None else _quoted(value)
        return _quoted("" if value is None else value)

    def _number_member(self, member: _Member, value: Any, depth: int) -> str:
        """A member of a number type: one number, or the array its declaration shapes."""
        name = NUMPY_NAMES[np.dtype(member.kind)]
        if not member.array:
            return _scalar(0 if value is None else value, name, self._store_inf)
        held = np.asarray([] if value is None else value, dtype=member.kind)
        if _nested_dimensions(member.dims):
            rows = _nest(held.ravel().tolist(), member.dims,
                         lambda group: self._items(group, name))  # fmt: skip
            return _list_text(rows, self._layout.item_comma)
        return self._array(held, name, depth)

    def _char_text(self, value: Any, member: _Member) -> str:
        """A ``char`` array of a declared class: one JSON string up to its first NUL, or - a
        ``char[N][M]``, which ROOT writes as a string per row - the arrays of them."""
        codes = [chr(code) for code in _char_codes(value)]
        if not _nested_dimensions(member.dims):
            return _quoted("".join(codes).split("\0")[0])
        rows = _nest(codes, member.dims,
                     lambda group: _quoted("".join(group).split("\0")[0]))  # fmt: skip
        return _list_text(rows, self._layout.item_comma)

    def _texts(self, member: _Member, value: Any, depth: int) -> str:
        """A ``std::string[N]``: one JSON array of its strings, a pointer with nothing there
        written as ``null``, or - a ``std::string[N][M]`` - an array of those.

        A ``TString`` array is the one ROOT's ``kStreamerLoop`` writes: it goes out at the
        depth the object holds it at rather than the depth its declaration gives it, since
        what a ``TString *fStr; // [fCnt]`` holds is as many strings as another member's
        counter says, and ``fStr2[3]`` is three of those."""
        if member.kind == "tstrings":
            return _text_tree(value, self._layout.item_comma)
        nested = _nest(list(value or []), member.dims,
                       lambda group: self._text_row(member, group))  # fmt: skip
        return _list_text(nested, self._layout.item_comma)

    def _text_row(self, member: _Member, group: list[Any]) -> str:
        """One row of a text array: its strings, or ``null`` for a ``std::string *`` with
        nothing there."""
        items = ["null" if each is None and member.kind == "strptrs"
                 else _quoted("" if each is None else str(each)) for each in group]
        return "[" + self._layout.item_comma.join(items) + "]"

    def _containers(self, member: _Member, value: Any, depth: int) -> str:
        """A member holding a container - or, declared ``vector<T>[N]``, an array of them,
        which ROOT writes as an array of the containers."""
        if not member.array:
            return self._container_text(member, value, depth)
        nested = _nest(list(value or []), member.dims,
                       lambda group: self._container_row(member, group, depth))  # fmt: skip
        return _list_text(nested, self._layout.item_comma)

    def _container_row(self, member: _Member, group: list[Any], depth: int) -> str:
        """One row of an array of containers: each container written as it goes out."""
        items = [self._container_text(member, each, depth) for each in group]
        return "[" + self._layout.item_comma.join(items) + "]"

    def _container_text(self, member: _Member, value: Any, depth: int) -> str:
        """One container as JSON: ``null`` for a pointer with nothing there, a boolean per
        bit of a ``bitset``, a map's keys as one object, or the values a sequence holds."""
        if value is None:
            return "null"
        if member.kind == "map_object":
            return self._map_object(value, list(value.items()), depth)
        if isinstance(value, _Map):
            return self._map(value, depth)
        if member.kind == "bitset":
            bits = ["true" if value[at] else "false" for at in range(value.size())]
            return "[" + self._layout.item_comma.join(bits) + "]"
        held = [self._value(each, depth + 1) for each in value]
        return "[" + self._layout.item_comma.join(held) + "]"

    def _braced(self, pairs: list[str], depth: int) -> str:
        """``{`` and ``}`` round members written as ``"name" : value``, on one line or with
        each member on a line of its own."""
        if not self._layout.newline:
            return "{" + self._layout.member_comma.join(pairs) + "}"
        pad = " " * (self._layout.indent * (depth + 1))
        closing = " " * (self._layout.indent * depth)
        inside = self._layout.member_comma.join(pad + pair for pair in pairs)
        return "{\n" + inside + "\n" + closing + "}"

    def _keys(self, pairs: list[str]) -> str:
        """``{`` and ``}`` round members written as keys of the object, which ROOT keeps on
        one line at every level: an ``$arr`` object's, a ``$pair``'s, a map's."""
        return "{" + self._layout.item_comma.join(pairs) + "}"

    # -- arrays -----------------------------------------------------------------------

    def _array(self, values: Any, name: str, depth: int) -> str:
        """An array of numbers of type ``name``: the numbers themselves, or the ``$arr``
        object that holds them compressed at a level that asks for it."""
        if self._compress == 3:
            return self._base64(values, name)
        if self._compress in (1, 2) and len(values) >= ARRAY_MIN:
            return self._suppressed(values, name)
        return "[" + self._layout.item_comma.join(self._items(values, name)) + "]"

    def _items(self, values: Any, name: str) -> list[str]:
        """Each value of an array written as its own type writes it."""
        return [_scalar(value, name, self._store_inf) for value in values]

    def _suppressed(self, values: Any, name: str) -> str:
        """The ``$arr`` object for an array with its zero ends and its long runs of one
        value taken out: where each stretch is, what it holds, and how many times."""
        listed = list(values.tolist() if isinstance(values, np.ndarray) else values)
        pairs = ['"$arr":' + _quoted(name), '"len":' + str(len(listed))]
        for index, (position, held, count) in enumerate(_pieces(listed, self._compress == 2)):
            suffix = "" if index == 0 else str(index)
            pairs.extend(_chunk(suffix, position, self._items(held, name), count,
                                self._layout.item_comma))  # fmt: skip
        return self._keys(pairs)

    def _base64(self, values: Any, name: str) -> str:
        """The ``$arr`` object ROOT writes for ``kBase64``: the numbers' own bytes, little
        endian as every ROOT file holds them, in base64, from the first value there is to
        the last. The zeros at the front are where the ``o`` says they end - in bytes, and
        only when there are enough of them to be worth the key."""
        listed = list(values.tolist() if isinstance(values, np.ndarray) else values)
        if not listed:
            return "[]"
        first, last = _content(listed)
        kind = DTYPES[name]
        #: The bytes of leading zeros ROOT leaves out of ``b``, and says where in ``o``.
        skipped = first * kind.itemsize
        dropped = skipped if first <= last and skipped > BYTES_MIN else 0
        held = listed[dropped // kind.itemsize : last + 1] if first <= last else []
        raw = np.asarray(held, dtype=kind.newbyteorder("<"))
        pairs = ['"$arr":' + _quoted(name), '"len":' + str(len(listed))]
        if dropped:
            pairs.append('"o":' + str(dropped))
        pairs.append('"b":"' + base64.b64encode(raw.tobytes()).decode("ascii") + '"')
        return self._keys(pairs)

    # -- collections ------------------------------------------------------------------

    def _map(self, obj: _Map, depth: int) -> str:
        """A ``std::map``: a ``$pair`` object per entry, or - with ``kMapAsObject`` - one
        JSON object whose keys are the map's."""
        entries = list(obj.items())
        if self._map_as_object:
            return self._map_object(obj, entries, depth)
        name = f"pair<{cpp_name(obj.kinds[0])},{cpp_name(obj.kinds[1])}>"
        pairs = [self._pair_entry(name, entry, depth) for entry in entries]
        return "[" + self._layout.item_comma.join(pairs) + "]"

    def _pair_entry(self, name: str, entry: tuple[Any, Any], depth: int) -> str:
        """One ``std::pair`` of a map, as ROOT writes it: the types it holds - or, at a
        level that skips them, their number - and its two values."""
        colon = self._layout.colon
        pairs = ['"$pair"' + colon + ("1" if self._skip_typename else _quoted(name)),
                 '"first"' + colon + self._value(entry[0], depth + 1),
                 '"second"' + colon + self._value(entry[1], depth + 1)]  # fmt: skip
        return self._keys(pairs)

    def _map_object(self, obj: _Map, entries: list[tuple[Any, Any]], depth: int) -> str:
        """A map written as a JSON object: its keys, each with the value it holds."""
        colon = ":" if self._layout.colon == ":" else ": "
        pairs = [_quoted(key) + colon + self._value(value, depth + 1) for key, value in entries]
        return self._keys(self._typed(pairs, str(obj.__cpp_name__), colon))


def _number_array(self: _Writer, obj: Any, depth: int) -> str:
    """A ``TArray`` of numbers, as the array of the type it holds."""
    return self._array(obj, ARRAY_NAMES[type(obj)], depth)


def _numpy_array(self: _Writer, obj: np.ndarray[Any, Any], depth: int) -> str:
    """The numbers a drawn object keeps, as the array of the type NumPy holds them in."""
    return self._array(obj, NUMPY_NAMES[obj.dtype], depth)


#: Which writer takes which Python type, in the order they are tried: ``bool`` before the
#: ``int`` it is a kind of, the strings ROOT and C++ both have, and the arrays before the
#: maps. A class is written by :data:`MEMBERS` once none of these takes it.
WRITERS: tuple[tuple[Any, Callable[[_Writer, Any, int], str]], ...] = (
    (type(None), _Writer._null),
    (bool, _Writer._boolean),
    (int, _Writer._integer),
    (float, _Writer._float),
    (str, _Writer._string),
    (string, _Writer._string),
    (np.integer, _Writer._integer),
    (np.floating, _Writer._float),
    (TArrayC, _Writer._text_array),
    ((TArrayD, TArrayF, TArrayI, TArrayL, TArrayS), _number_array),
    (np.ndarray, _numpy_array),
    (_Map, _Writer._map),
)


def _untext(text: Any) -> str:
    """The JSON a ``TString`` or a ``str`` holds."""
    return text.Data() if isinstance(text, TString) else str(text)


def _named(kind: Any) -> str:
    """The class a C++ type name names, without the namespaces and pointers round it."""
    return str(kind or "").replace("std::", "").strip("* ")


def _built(kind: type[Any], values: list[Any]) -> Any:
    """An array of class ``kind`` holding ``values``."""
    made = kind(len(values))
    for index, value in enumerate(values):
        made[index] = value
    return made


def _expanded(data: dict[str, Any], name: str) -> list[Any]:
    """The numbers a compressed ``$arr`` holds, back in their places: ``len`` of them, each
    stretch where its position says, and one repeated value as many times as its count. A
    stretch that runs past the length ROOT wrote gives it up, as ROOT reads only ``len``."""
    values: list[Any] = [0] * int(data.get("len", 0))
    if "b" in data:
        kind = DTYPES[name]
        held = np.frombuffer(base64.b64decode(str(data["b"])), dtype=kind.newbyteorder("<"))
        at = int(data.get("o", 0)) // kind.itemsize
        carried: list[Any] = held.tolist()
        return values[:at] + carried[: max(0, len(values) - at)] + values[at + len(held) :]
    at = index = 0
    while True:
        suffix = "" if index == 0 else str(index)
        at = int(data.get(f"p{suffix}", at))
        held = data.get(f"v{suffix}")
        if held is None:
            return values
        placed = held if isinstance(held, list) else [held] * int(data.get(f"n{suffix}", 1))
        placed = placed[: max(0, len(values) - at)]
        values[at : at + len(placed)] = placed
        at += len(placed)
        index += 1


def _entries(data: Any) -> list[tuple[Any, Any]]:
    """The key and value of each entry of a map written either way ROOT writes one: a list
    of ``$pair`` objects, or - with ``kMapAsObject`` - an object whose keys are the map's."""
    if isinstance(data, dict):
        return [(key, value) for key, value in data.items() if key != "_typename"]
    return [(entry["first"], entry["second"]) for entry in data]


def _rebuilt_map(data: Any, name: str) -> Any:
    """The ``std::map`` a JSON object or ``$pair`` list describes, of the types ``name``
    gives it: ``map<string,int>`` is a map of string to int."""
    made = std.map[tuple(name[name.index("<") + 1 : name.rindex(">")].split(","))]()
    for key, value in _entries(data):
        made[key] = value
    return made


def _decoded(text: str) -> Any:
    """The JSON a text holds, or the refusal that says it does not hold any."""
    try:
        return json.loads(text)
    except ValueError:  # json.JSONDecodeError, which is a ValueError
        raise UnsupportedFeatureError(
            "TBufferJSON::FromJSON reads the JSON text TBufferJSON::ConvertToJSON wrote, "
            f"and {text[:40]!r} is not JSON at all"
        ) from None


def _rebuilt(text: str, kind: str | None) -> Any:
    """What the JSON in ``text`` stands for, built as the class ``kind`` names it."""
    data = _decoded(text)
    name = _named(kind)
    if name.startswith("map<"):
        return _rebuilt_map(data, name)
    return _rebuilt_named(data, name)


def _rebuilt_named(data: Any, name: str) -> Any:
    """An array or an object from the JSON that names its class - an ``$arr`` object's own,
    or the one asked for - and the refusals for what does not name one."""
    if isinstance(data, dict) and "$arr" in data:
        return _rebuilt_array(data)
    if isinstance(data, dict):
        return _rebuilt_object(data)
    if name in ARRAY_CLASSES and isinstance(data, (list, str)):
        return _rebuilt_bare(data, ARRAY_CLASSES[name])
    raise UnsupportedFeatureError(
        "TBufferJSON::FromJSON cannot read this JSON as "
        f"{name or 'any class'}: a bare JSON array does not say what class it holds, so ask "
        "for one - FromJSON<TArrayI>(json) - or write it at a level that names its type"
    )


def _rebuilt_bare(data: Any, kind: type[Any]) -> Any:
    """An array of ``kind`` from the bare JSON ROOT writes for one where the level does not
    name its type: the numbers, or the characters of a ``char`` array's string."""
    return _built(kind, [ord(char) for char in data] if isinstance(data, str) else data)


def _class_for(name: str) -> type[Any]:
    """The class a JSON's ``_typename`` names: one the translation declared, or one of the
    ``TArray`` classes this writes, and otherwise the refusal that says what is read here.

    A class has to be named because a JSON object says what its members are and not what
    order the streamer writes them in; the class is what knows that.
    """
    from .declared import find_class

    found = find_class(name) if name else None
    if found is None:
        found = ARRAY_CLASSES.get(name)
    if found is None:
        raise UnsupportedFeatureError(
            "TBufferJSON::ConvertFromJSONAny does not know the class "
            f"{name or 'this JSON gives no name of'}: it reads back an object of a class a "
            f"macro declared, and the TArrays ({', '.join(sorted(ARRAY_CLASSES))}). Read "
            "anything else with a class this kit has, or ask FromJSON<T>(json) for the "
            "class it names"
        )
    return found


def _rebuilt_object(data: dict[str, Any]) -> Any:
    """The object a JSON object describes: the class its ``_typename`` names, made, and given
    back the members that JSON holds."""
    name = _named(str(data.get("_typename", "")))
    made = _class_for(name)()
    for member in _writable(made):
        if member.name in data:
            _restore(made, member, data[member.name])
    return made


def _restore(obj: Any, member: _Member, value: Any) -> None:
    """Put ``value``, read from JSON, in the attribute a declared class declares ``member``
    as, which is where the writer reads it back from.

    A container member keeps the container its constructor made: what such a JSON says is
    the values the container holds, and reading values back into a ``std::map`` keyed by an
    object - or into a ``bitset`` of a length its type fixes - is a writer and reader of its
    own. An empty one is what the constructor made, so it needs nothing read into it."""
    if member.kind == "bitset":
        held = getattr(obj, member.name, None)
        if not isinstance(value, list) or held is None:
            raise UnsupportedFeatureError(
                f"TBufferJSON::FromJSON does not know what {_classname(obj)}::{member.name} "
                "holds: a bitset is read back as one boolean per bit, in the order the "
                "bits are numbered, and this JSON holds something else. Ask for the "
                "members of a number type and of text, or read this JSON with the class it "
                "names"
            )
        for at, bit in enumerate(value):
            held[at] = bool(bit)
        return
    if member.kind in CONTAINER_KINDS:
        if not _blank(value):
            raise UnsupportedFeatureError(
                "TBufferJSON::FromJSON does not yet fill a member that holds values, and "
                f"{_classname(obj)}::{member.name} holds some: it reads back an object "
                "whose containers are empty, which is what a class's constructor makes "
                "them. Ask for the members of a number type and of text, or read this JSON "
                "with the class it names"
            )
        return
    setattr(obj, member.name, _converted(member, value))


def _blank(value: Any) -> bool:
    """Does this JSON hold no value at all: ``null``, an empty array, a map with nothing in
    it, or arrays of those?"""
    if value is None or value == []:
        return True
    if isinstance(value, dict):
        return set(value) <= {"_typename"}
    if isinstance(value, list):
        return all(_blank(each) for each in value)
    return False


def _converted(member: _Member, value: Any) -> Any:
    """What a member of a declared class holds, as the JSON wrote it: a number, a string of
    characters, or the array of them its declared lengths shape."""
    if isinstance(member.kind, str):
        return _converted_text(member, value)
    return _converted_number(member, value)


def _converted_text(member: _Member, value: Any) -> Any:
    """What a member of text, of an array of it, or of characters holds."""
    if member.kind == "chars":
        return "".join(chr(code) for code in _char_codes(value))
    if member.kind == "tstrings":
        return _converted_tree(value)
    if member.kind in ("strings", "strptrs"):
        return _converted_texts(member, value)
    if member.kind in ("cstring", "strptr"):
        return None if value is None else str(value)
    return "" if value is None else str(value)


def _converted_tree(value: Any) -> Any:
    """A ``TString`` array the JSON wrote, at the depth it wrote it: the strings it holds
    and ``None`` where a pointer had nothing there."""
    if isinstance(value, list):
        return [_converted_tree(each) for each in value]
    return None if value is None else str(value)


def _converted_texts(member: _Member, value: Any) -> list[str | None]:
    """An array of ``std::string`` - or of pointers to one - as the strings it holds, with
    ``None`` where a pointer had nothing there."""
    items = value if isinstance(value, list) else [value]
    return [None if each is None and member.kind == "strptrs" else str(each or "")
            for each in items]


def _converted_number(member: _Member, value: Any) -> Any:
    """What a member of a number type holds: one number, or the array its lengths shape."""
    if not member.array:
        return member.kind(0 if value is None else value).item()
    numbers = np.asarray(_flat_numbers(value), dtype=member.kind)
    if _nested_dimensions(member.dims) and numbers.size == int(np.prod(member.dims)):
        return numbers.reshape(member.dims)
    return numbers


def _rebuilt_array(data: dict[str, Any]) -> Any:
    """The ``TArray`` a ``$arr`` object describes."""
    name = str(data.get("$arr", ""))
    kind = ARRAY_TYPES.get(name)
    if kind is None:
        raise UnsupportedFeatureError(
            f"TBufferJSON::FromJSON has no class for an array written as {name!r}; it reads "
            "back " + ", ".join(ARRAY_TYPES)
        )
    return _built(kind, _expanded(data, name))


def _filled(target: Any, made: Any) -> Any:
    """``target`` holding what was read: the array it already is, or the cell a macro's
    pointer lives in, which is filled where the C++ writes through the pointer."""
    if any(isinstance(target, kind) for kind in ARRAY_CLASSES.values()):
        target.Set(len(made), list(made))
        return target
    target.value = made
    return made


class _FromJSON:
    """``TBufferJSON::FromJSON``: the object a JSON string was written from.

    ROOT's C++ has two overloads - one filling the pointer it is handed, one making the
    class its template argument names - and Python has neither a pointer to fill nor a
    template. The class comes from the cell a translated out-parameter arrives in (its
    ``ctype``), or from the ``[...]`` a macro's template argument is translated to; what is
    read is returned either way, so a macro's ``FromJSON(obj, json)`` reads back whole.
    """

    def __call__(self, target: Any, text: Any) -> Any:
        """``FromJSON(obj, json)``: the object read into ``target`` - the array it is, or the
        cell a pointer to one is - and returned."""
        if target is None:
            raise UnsupportedFeatureError(
                "TBufferJSON::FromJSON writes the object it read into the pointer it is "
                "given, and this one arrived as its value rather than as a cell holding it, "
                "so there is nowhere to write; read what FromJSON returns instead - "
                "FromJSON<T>(json), or the cell a translated pointer arrives in"
            )
        return _filled(target, _rebuilt(_untext(text), getattr(target, "ctype", None)))

    def __getitem__(self, kind: Any) -> Callable[[Any], Any]:
        """``FromJSON<T>(json)``: the reader for the class ``T`` names - ``TArrayI``,
        ``std::map<std::string,int>``."""
        return lambda text: _rebuilt(_untext(text), str(kind))


class TBufferJSON:
    """``TBufferJSON``: ROOT's object-to-JSON export, over the classes this writes exactly.

    The compact level is ROOT's: the four layouts, and above them, counted in tens, the
    compression the ``$arr`` objects carry - ``kZeroSuppression``, ``kSameSuppression``,
    ``kBase64`` - the ``kSkipTypeInfo`` in the hundreds and ``kStoreInfNaN`` in the
    thousands. A buffer is opened for writing, as ROOT's is when it is given no mode: the
    JSON is read back by :meth:`ConvertFromJSON` and :attr:`FromJSON`, not streamed in.
    """

    kRead = 0
    kWrite = 1
    kNoCompress = 0
    kNoIndent = 1
    kNoNewLine = 2
    kNoSpaces = 3
    kMapAsObject = 5
    kZeroSuppression = 10
    kSameSuppression = 20
    kBase64 = 30
    kSkipTypeInfo = 100
    kStoreInfNaN = 1000

    def __init__(self, mode: int = kWrite, *rest: Any) -> None:
        if int(mode) == self.kRead:
            raise UnsupportedFeatureError(
                "TBufferJSON is opened for reading only through TBufferJSON::FromJSON and "
                "ConvertFromJSON, which read the JSON a string holds; a buffer opened in "
                "kRead mode has no stream to read from here"
            )
        self._compact = 0

    def SetCompact(self, level: int) -> None:
        """``SetCompact(level)``: the layout and compression the next object is written at."""
        self._compact = int(level)

    def StoreObject(self, obj: Any, cl: Any = None) -> TString:
        """``StoreObject(obj)``: the object as JSON, at the compact level last set."""
        return TBufferJSON.ToJSON(obj, self._compact)

    @staticmethod
    def ToJSON(obj: Any, compact: int = 0, member: str | None = None) -> TString:
        """``ToJSON(obj[, compact[, member]])``: ``obj`` as JSON, or just the member named."""
        writer = _Writer(int(compact))
        return TString(writer.text(obj) if member is None else writer.member(obj, member))

    @staticmethod
    def ConvertToJSON(obj: Any, cl: Any = None, compact: int = 0,
                      member: str | None = None) -> TString:  # fmt: skip
        """``ConvertToJSON(obj[, cl[, compact[, member]]])``: what ROOT's static function is
        called, with the class it takes - which the object's own says here."""
        return TBufferJSON.ToJSON(obj, compact, member)

    @staticmethod
    def ConvertFromJSONAny(text: Any, cl: Any = None) -> Any:
        """``ConvertFromJSONAny(json, &cl)``: the object the JSON describes, and - in the
        cell ``cl`` - the class it was read as, since ROOT writes the object back through
        that class and ``roottest`` compares the two JSON strings."""
        data = _decoded(_untext(text))
        named = str(data.get("_typename", "")) if isinstance(data, dict) else ""
        made = _rebuilt_named(data, _named(named))
        if cl is not None:
            from .reflection import TClass

            cl.value = TClass.GetClass(named or type(made).__name__)
        return made

    @staticmethod
    def ConvertFromJSON(text: Any) -> Any:
        """``ConvertFromJSON(json)``: the object a JSON string was written from, which names
        its own class - a ``$arr`` array, or a ``$pair`` map."""
        return _rebuilt(_untext(text), None)

    #: ``FromJSON``: the reader, which the class to make may follow in ``[...]``.
    FromJSON = _FromJSON()
