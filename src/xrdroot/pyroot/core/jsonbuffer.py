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
``fOption`` and ``fName`` and never ``fLastPoint``.

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
    whether it is an array of them."""

    name: str
    kind: Any
    array: bool = False


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
              "fTitle": "_title", "fString": "_name"}


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
    """The ROOT class an object is, by the name its JSON gives it."""
    named = getattr(obj, "ClassName", None)
    return str(named()) if callable(named) else str(getattr(obj, "classname", ""))


def _held(obj: Any, member: _Member) -> Any:
    """The value of ``member`` in ``obj``: one of the members a drawn object keeps by name,
    where ``fBits`` carries the heap's bits too, or the attribute a core class keeps it in."""
    found = getattr(obj, "members", None)
    value = found.get(member.name) if isinstance(found, dict) else None
    if value is None:
        value = getattr(obj, ATTRIBUTES.get(member.name, "_" + member.name), None)
    return _pointer_bits(value) if member.name == "fBits" and value is not None else value


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
        members = MEMBERS.get(_classname(obj), ())
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
        if _classname(obj) in MEMBERS:
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
        members = [self._pair(member, obj, depth) for member in MEMBERS[_classname(obj)]]
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
            return _quoted("" if value is None else value)
        name = NUMPY_NAMES[np.dtype(member.kind)]
        if member.array:
            return self._array(np.asarray(value, dtype=member.kind), name, depth)
        return _scalar(0 if value is None else value, name, self._store_inf)

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
    """An array from the JSON that names its class - an ``$arr`` object's own, or the one
    asked for - and the refusals for what does not name one."""
    if isinstance(data, dict) and "$arr" in data:
        return _rebuilt_array(data)
    if isinstance(data, dict):
        raise UnsupportedFeatureError(
            "TBufferJSON::FromJSON does not yet read an object back: this one is a "
            f"{data.get('_typename', 'classless object')}, and what is read here is the "
            "JSON that names its own class - an $arr array, or a $pair map"
        )
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
    def ConvertFromJSON(text: Any) -> Any:
        """``ConvertFromJSON(json)``: the object a JSON string was written from, which names
        its own class - a ``$arr`` array, or a ``$pair`` map."""
        return _rebuilt(_untext(text), None)

    #: ``FromJSON``: the reader, which the class to make may follow in ``[...]``.
    FromJSON = _FromJSON()
