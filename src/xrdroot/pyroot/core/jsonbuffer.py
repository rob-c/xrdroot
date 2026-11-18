"""``TBufferJSON``: an object as the JSON ROOT's own exporter writes it.

ROOT exports an object as the JSON its streamers describe it in - one member
per line, in streamer order, arrays of numbers as JSON arrays - and hands the
text to a program through ``TBufferJSON::ConvertToJSON``. JSROOT reads the
same text back in a browser, which is why it is compressed the way it is: a
hundred numbers that are mostly zero and run in tens become
``{"$arr":"Int32","len":100,"p":17,"v":7,"n":10}``, and a ``std::map`` with
``kMapAsObject`` becomes a JSON object.

This is that export, member for member and space for space, at each of ROOT's
four compact levels: the indented text ``kNoCompress`` gives, the unindented
and the single-line forms after it, and ``kNoSpaces``. A class's members come
from :data:`MEMBERS`, which is a table rather than a walk of the object
because ROOT writes what the class's *streamer* writes - which for a class
with a hand-written streamer is not every data member: ``TPolyMarker3D``
writes ``fN``, ``fP``, ``fOption`` and ``fName`` and never ``fLastPoint``.

What it writes is what it can write *exactly*. Anything else is refused by
name, in a full sentence: a plausible-looking JSON that ROOT reads back as
something else is worse than an error.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...cppstd.containers import _Map
from ...errors import UnsupportedFeatureError
from ..vectors import TArrayC, TArrayD, TArrayF, TArrayI, TArrayL, TArrayS
from .strings import TString

__all__ = ["TBufferJSON"]

#: The shortest array ROOT compresses: below this the ``$arr`` object would be larger.
ARRAY_MIN = 6

#: How many similar values in a row ROOT writes as one value and a count.
RUN_MIN = 6

#: The smallest exponent written positionally (``0.001``) and, for each number of
#: significant digits, the one it stops at: above it ROOT writes ``1e25``, not the
#: twenty-six digits of it. A ``Float_t`` has 7 digits and ``Double_t`` 15, which is
#: what puts a float's limit at ``1e16`` and a double's at ``1e25``.
EXPONENT_MIN = -3
EXPONENT_MAX = {7: 16, 15: 25}

#: The exponent ``%.Ne`` writes a value's mantissa and exponent with.
MANTISSA = {7: "%.6e", 15: "%.14e"}

#: ROOT's spelling of the values JSON has no numbers for: a number too large to read
#: (and its negative), and ``null`` - which is how JSROOT reads a NaN - or, with
#: ``kStoreInfNaN``, the strings it names them by.
SPECIAL = {
    "inf": ("2e308", '"inf"'),
    "nan": ("null", '"nan"'),
}

#: The ``$arr`` type name of each array class ROOT writes as numbers. ``Int8`` is
#: missing because ROOT writes a ``TArrayC`` - any ``char`` array - as a JSON string.
ARRAY_NAMES = {TArrayS: "Int16", TArrayI: "Int32", TArrayL: "Int64",
               TArrayF: "Float32", TArrayD: "Float64"}  # fmt: skip

#: The class each ``$arr`` type name stands for, for reading one back.
ARRAY_TYPES = {name: kind for kind, name in ARRAY_NAMES.items()}

#: The same names for the NumPy arrays a drawn object keeps its points in.
NUMPY_NAMES = {np.dtype(np.int8): "Int8", np.dtype(np.uint8): "UInt8",
               np.dtype(np.int16): "Int16", np.dtype(np.uint16): "UInt16",
               np.dtype(np.int32): "Int32", np.dtype(np.uint32): "UInt32",
               np.dtype(np.int64): "Int64", np.dtype(np.float32): "Float32",
               np.dtype(np.float64): "Float64", np.dtype(np.bool_): "Bool"}  # fmt: skip

#: The classes ROOT names by a ``TArray``'s name, for reading JSON asked for by type.
ARRAY_CLASSES = {kind.__name__: kind for kind in (TArrayC, *ARRAY_NAMES)}


@dataclass(frozen=True)
class _Layout:
    """How one of ROOT's four compact levels lays the JSON out.

    ``member_comma`` is what goes between two members of an object - the newline
    the indented levels break the line with, joined there with the indent - and
    ``item_comma`` what goes between two numbers of an array, which is a space at
    every level but ``kNoSpaces``.
    """

    newline: bool
    indent: int
    colon: str
    member_comma: str
    item_comma: str


#: Levels 0 to 3, and the three levels above them the flags are counted in.
LAYOUTS = {
    0: _Layout(True, 2, " : ", ",\n", ", "),
    1: _Layout(True, 0, " : ", ",\n", ", "),
    2: _Layout(False, 0, " : ", ",", ", "),
    3: _Layout(False, 0, ":", ",", ","),
}


def _trimmed(text: str) -> str:
    """``text`` without the zeros a float's last places keep, and without a bare ``.``."""
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def _number(value: float, digits: int, stored: bool) -> str:
    """A ``Float_t`` or ``Double_t`` as ROOT writes it: ``digits`` significant figures,
    positionally between ``0.001`` and the exponent its type stops at and as an exponent
    outside that, which is what JSROOT has always been handed."""
    special = _special(value, stored)
    if special is not None:
        return special
    mantissa, exponent = (f"%.{digits - 1}e" % value).split("e")
    if EXPONENT_MIN <= int(exponent) < EXPONENT_MAX[digits]:
        places = max(0, digits - 1 - int(exponent))
        return _trimmed(f"%.{places}f" % value)
    return _trimmed(mantissa) + "e" + str(int(exponent))


def _special(value: float, stored: bool) -> str | None:
    """What a value JSON cannot write is written as, or ``None`` for one it can."""
    if value != value:
        return SPECIAL["nan"][int(stored)]
    if value in (float("inf"), float("-inf")):
        sign = "-" if value < 0 else ""
        return sign + SPECIAL["inf"][int(stored)]
    return None


def _quoted(text: str) -> str:
    """A JSON string: ROOT escapes what JavaScript's ``JSON.parse`` needs, and the
    solidus with it, as its own writer does."""
    return json.dumps(str(text)).replace("/", "\\/")


def _scalar(value: Any, kind: np.dtype[Any], stored: bool) -> str:
    """One number of member ``kind``, written as ROOT writes that type."""
    if kind.kind in "iu":
        return str(int(value))
    return _number(float(value), 7 if kind.itemsize == 4 else 15, stored)


def _pointer_bits(bits: int) -> int:
    """The bits of a ``TObject`` ROOT writes: its status word without the heap's two,
    which say where the object is and not what it is."""
    return int(bits) & 0x00FFFFFF


@dataclass(frozen=True)
class _Member:
    """One data member of a class this writes: its name, the C++ type it goes out as, and
    whether the member is an array of them."""

    name: str
    kind: Any
    array: bool = False


#: Each class this writes, its members in the order ROOT's streamer writes them. A class
#: with a hand-written streamer writes what that streamer says, so ``TPolyMarker3D`` has
#: no ``fLastPoint`` and ``TPolyMarker3D``'s ``fName`` comes after its own members.
MEMBERS: dict[str, tuple[_Member, ...]] = {
    "TObject": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32)),
    "TNamed": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32),
               _Member("fName", "string"), _Member("fTitle", "string")),  # fmt: skip
    "TObjString": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32),
                   _Member("fString", "string")),  # fmt: skip
    "TPolyMarker3D": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32),
                      _Member("fMarkerColor", np.int16), _Member("fMarkerStyle", np.int16),
                      _Member("fMarkerSize", np.float32), _Member("fN", np.int32),
                      _Member("fP", np.float32, True), _Member("fOption", "string"),
                      _Member("fName", "string")),  # fmt: skip
    "TPolyLine3D": (_Member("fUniqueID", np.uint32), _Member("fBits", np.uint32),
                    _Member("fLineColor", np.int16), _Member("fLineStyle", np.int16),
                    _Member("fLineWidth", np.int16), _Member("fN", np.int32),
                    _Member("fP", np.float32, True), _Member("fOption", "string")),  # fmt: skip
}

#: Where a core class keeps a member ROOT names otherwise: ``TObject``'s ``fUniqueID`` is
#: the object's ``_unique_id``, and a ``TObjString``'s ``fString`` its one string.
ATTRIBUTES = {"fUniqueID": "_unique_id", "fBits": "_bits", "fName": "_name",
              "fTitle": "_title", "fString": "_name"}  # fmt: skip


def _classname(obj: Any) -> str:
    """The ROOT class an object is, by the name its JSON gives it."""
    named = getattr(obj, "ClassName", None)
    return str(named()) if callable(named) else str(getattr(obj, "classname", ""))


def _held(obj: Any, member: _Member) -> Any:
    """The value of ``member`` in ``obj``: the members a drawn object keeps by name - where
    ``fBits`` carries the heap's bits too - or the attribute a core class keeps its own in."""
    found = getattr(obj, "members", None)
    value = found[member.name] if isinstance(found, dict) and member.name in found else None
    if value is None:
        value = getattr(obj, ATTRIBUTES.get(member.name, "_" + member.name), None)
    return _pointer_bits(value) if member.name == "fBits" else value


def _pieces(values: list[Any], same: bool) -> list[tuple[int | None, Any, int | None]]:
    """The stretches ROOT keeps of an array: each one's position - ``None`` for one that
    follows the stretch before it - the value or the values it holds, and how many times a
    single value repeats.

    Zeros at either end are dropped, so the first position is the first value there is; a
    stretch of one value is one value and a count, and the zeros that start a stretch are
    dropped as the array's own ends are, which is why a zero inside a stretch is kept and
    one after a compressed run moves the next stretch along.
    """
    first, last = _content(values)
    if first > last:
        return []
    pieces: list[tuple[int | None, Any, int | None]] = []
    at, end, follows = first, first, None
    while at <= last:
        while at <= last and not values[at]:
            at += 1
        if at > last:
            break
        count = _run(values, at, last, same)
        if count:
            pieces.append((at if at != follows else None, values[at], count))
            at += count
        else:
            starts = at
            at = _distinct(values, at, last)
            pieces.append((starts if starts != follows else None, values[starts:at], None))
        follows = at
    return pieces


def _content(values: list[Any]) -> tuple[int, int]:
    """Where the first and last value there is are: the array without its zero ends."""
    first, last = 0, len(values) - 1
    while first <= last and not values[first]:
        first += 1
    while last >= first and not values[last]:
        last -= 1
    return first, last


def _run(values: list[Any], at: int, last: int, same: bool) -> int:
    """How many times the value at ``at`` repeats, where ROOT writes that as a count: a run
    of ``RUN_MIN`` or more with ``kSameSuppression``, and nothing for a zero, which is what
    is not there rather than a value to repeat."""
    if not same:
        return 0
    count = 1
    while at + count <= last and values[at + count] == values[at]:
        count += 1
    return count if count >= RUN_MIN else 0


def _distinct(values: list[Any], at: int, last: int) -> int:
    """Where the values from ``at`` stop being worth keeping one by one: the first run of a
    repeated value, or the last value there is."""
    while at <= last:
        if values[at]:
            count = _run(values, at, last, True)
            if count:
                return at
        at += 1
    return at


class _Writer:
    """One export: the JSON for an object, its members and their arrays, laid out as the
    compact level asks and compressed the way JSROOT reads."""

    def __init__(self, compact: int) -> None:
        #: ``level % 5``: ROOT counts the three flags above the four layouts in fives.
        self._layout = LAYOUTS[min(compact % 5, 3)]
        #: ``kZeroSuppression`` (1), ``kSameSuppression`` (2) or ``kBase64`` (3) as tens.
        self._compress = (compact // 10) % 10
        self._map_as_object = compact % 10 >= 5
        self._skip_typename = (compact // 100) % 10 == 1
        self._store_inf = (compact // 1000) % 10 == 1

    # -- what a value is --------------------------------------------------------------

    def text(self, obj: Any) -> str:
        return self._value(obj, 0)

    def member(self, obj: Any, name: str) -> str:
        """The JSON of one member of ``obj``, which is what ``ToJSON(obj, level, member)``
        writes instead of the whole object."""
        for member in self._members(obj):
            if member.name == name:
                return self._member(member, obj, 0)
        raise UnsupportedFeatureError(
            f"{_classname(obj)} has no member {name!r} to write as JSON; name one of "
            + ", ".join(member.name for member in self._members(obj))
        )

    def _members(self, obj: Any) -> tuple[_Member, ...]:
        """The members of ``obj`` as ROOT writes them, or the refusal that says which class
        this cannot write."""
        classname = _classname(obj)
        found = MEMBERS.get(classname)
        if found is None:
            raise UnsupportedFeatureError(
                f"TBufferJSON::ConvertToJSON writes {classname or type(obj).__name__} as "
                "JSON from a table of the members ROOT's streamer writes, and the classes "
                f"in that table are {', '.join(sorted(MEMBERS))}; writing this one means "
                "adding its members there, in its streamer's order"
            )
        return found

    def _value(self, obj: Any, depth: int) -> str:
        """``obj`` as JSON: a number, a string, an array, a map or an object."""
        for kind, write in _WRITERS:
            if isinstance(obj, kind):
                return write(self, obj, depth)
        raise UnsupportedFeatureError(
            f"TBufferJSON::ConvertToJSON has no JSON for a {type(obj).__name__}; it writes "
            "the numbers, strings, TArray and std containers, the objects whose members it "
            "knows and the lists of them"
        )

    # -- the leaves -------------------------------------------------------------------

    def _null(self, obj: Any, depth: int) -> str:
        return "null"

    def _boolean(self, obj: bool, depth: int) -> str:
        return "true" if obj else "false"

    def _integer(self, obj: int, depth: int) -> str:
        return str(int(obj))

    def _float(self, obj: float, depth: int) -> str:
        return _number(float(obj), 15, self._store_inf)

    def _string(self, obj: str, depth: int) -> str:
        return _quoted(obj)

    # -- the composites ---------------------------------------------------------------

    def _object(self, obj: Any, depth: int) -> str:
        members = self._members(obj)
        written = [self._pair(member, obj, depth) for member in members]
        typename = [] if self._skip_typename else [_quoted("_typename") + self._layout.colon
                                                   + _quoted(_classname(obj))]  # fmt: skip
        return self._braced(typename + written, depth)

    def _pair(self, member: _Member, obj: Any, depth: int) -> str:
        return _quoted(member.name) + self._layout.colon + self._member(member, obj, depth)

    def _member(self, member: _Member, obj: Any, depth: int) -> str:
        value = _held(obj, member)
        if isinstance(member.kind, str):
            return _quoted(value if value is not None else "")
        if member.array:
            array = np.asarray(value if value is not None else [], dtype=member.kind)
            return self._array(array, NUMPY_NAMES[member.kind], depth)
        return _scalar(0 if value is None else value, member.kind, self._store_inf)

    def _braced(self, pairs: list[str], depth: int) -> str:
        """``{`` and ``}`` round members written as ``"name" : value``."""
        if not self._layout.newline:
            return "{" + self._layout.member_comma.join(pairs) + "}"
        pad = " " * (self._layout.indent * (depth + 1))
        closing = " " * (self._layout.indent * depth)
        inside = self._layout.member_comma.join(pad + pair for pair in pairs)
        return "{\n" + inside + "\n" + closing + "}"

    def _entries(self, pairs: list[str]) -> str:
        """``{`` and ``}`` round members that are keys - an ``$arr`` object's, a map's when
        it is written as an object - which ROOT keeps on one line."""
        colon = ":" if self._layout.colon == ":" else ": "
        return "{" + self._layout.item_comma.join(pairs).replace(" : ", colon) + "}"

    # -- arrays -----------------------------------------------------------------------

    def _array(self, values: Any, name: str, depth: int) -> str:
        """An array of numbers: the numbers themselves, or the ``$arr`` object that holds
        them compressed at a level that asks for it."""
        items = self._items(values, name)
        if self._compress == 3:
            return self._base64(values, name)
        if self._compress in (1, 2) and len(items) >= ARRAY_MIN:
            return self._suppressed(values, name)
        return "[" + self._layout.item_comma.join(items) + "]"

    def _items(self, values: Any, name: str) -> list[str]:
        """Each number of an array written as its type writes it."""
        digits = 7 if name in ("Float32", "UInt8", "Int8") and name == "Float32" else 15
        kind = np.dtype(_DTYPES.get(name, np.float64))
        return [_scalar(value, kind if kind.kind != "f" else np.dtype(
            np.float32 if digits == 7 else np.float64), self._store_inf) for value in values]

    def _suppressed(self, values: Any, name: str) -> str:
        """The ``$arr`` object for an array with its zeros and its long runs taken out."""
        listed = [float(value) if name.startswith("Float") else int(value) for value in values]
        pairs = ['"$arr":' + _quoted(name), '"len":' + str(len(listed))]
        for index, (position, value, count) in enumerate(_pieces(listed, self._compress == 2)):
            suffix = "" if index == 0 else str(index)
            pairs.extend(_chunk(suffix, position, value, count, self._items))
        return self._entries(pairs)

    def _base64(self, values: Any, name: str) -> str:
        """The ``$arr`` object ROOT writes for ``kBase64``: the numbers' own bytes, little
        endian, as base64."""
        kind = np.dtype(_DTYPES.get(name, np.float64)).newbyteorder("<")
        raw = np.asarray(values, dtype=kind).tobytes()
        return self._entries(['"$arr":' + _quoted(name), '"len":' + str(len(values)),
                              '"b":' + _quoted(base64.b64encode(raw).decode("ascii"))])  # fmt: skip

    # -- collections ------------------------------------------------------------------

    def _map(self, obj: _Map, depth: int) -> str:
        """A ``std::map``: ``$pair`` objects, one per entry, or - with ``kMapAsObject`` - a
        JSON object whose keys are the map's."""
        entries = list(obj.items())
        if self._map_as_object:
            return self._map_object(obj, entries)
        return "[" + self._layout.item_comma.join(
            self._pair_objects(entries, depth)) + "]"  # fmt: skip

    def _pair_objects(self, entries: list[tuple[Any, Any]], depth: int) -> list[str]:
        return [self._entries([self._pairable(entry) for entry in entries])]

    def _pairable(self, entry: tuple[Any, Any]) -> str:
        name = f"pair<{cpp_name_of(entry[0])},{cpp_name_of(entry[1])}>"
        return ('"$pair":' + _quoted(name) + self._layout.item_comma
                + '"first":' + self._value(entry[0], 1) + self._layout.item_comma
                + '"second":' + self._value(entry[1], 1))  # fmt: skip

    def _map_object(self, obj: _Map, entries: list[tuple[Any, Any]]) -> str:
        typename = [] if self._skip_typename else [_quoted("_typename") + ": "
                                                   + _quoted(str(obj.__cpp_name__))]  # fmt: skip
        keys = [_quoted(str(key)) + ": " + self._value(value, 1) for key, value in entries]
        return "{" + self._layout.item_comma.join(typename + keys) + "}"


def _chunk(suffix: str, position: int | None, value: Any, count: int | None,
           items: Any) -> list[str]:  # fmt: skip
    """One stretch of a compressed array: where it is, what it holds, and how many."""
    pairs = [] if position is None else [f'"p{suffix}":{position}']
    spelled = items([value], "Int32")[0] if not isinstance(value, list) else None
    held = spelled if spelled is not None else "[" + ", ".join(
        items(value, "Int32")) + "]"  # fmt: skip
    pairs.append(f'"v{suffix}":{held}')
    if count is not None:
        pairs.append(f'"n{suffix}":{count}')
    return pairs


#: A ``std::pair`` element type's C++ name, which is what a ``$pair`` says it holds.
cpp_name_of = str
