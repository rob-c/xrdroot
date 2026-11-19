"""Writing an RNTuple: fields declared once, then entries, then clusters.

What is written is laid out the way ROOT lays out its own: a page of values
per column per cluster - several, once a page reaches ROOT's megabyte - each
page compressed with the file's algorithm and followed by its XXH3, all in
the unlisted ``RBlob`` keys ROOT keeps them in; then, when the file closes,
the header describing the schema, a page list saying where every page went,
the footer pointing at the page list, and the ``ROOT::RNTuple`` anchor the
directory lists, which points at the header and the footer. The column
encodings are ROOT's defaults: split, zigzag and delta when the file is
compressed, which is what they are for, and the plain ones when it is not.

The fields that can be written are the ones whose layout is beyond doubt:
the fundamental numbers, ``bool``, ``std::string``, and a ``std::vector`` of a
number or of another such vector. A collection nests as ROOT nests it: the
outer collection's offset column says where each entry's items end, the one
under it where those items' own items end, and so on down to a column of the
leaves. A class is a :class:`RecordSpec` - the class's name, and a field of its
own for every member the class's dictionary declares, the record itself having
no column - which is how ROOT writes one. Anything else - variants,
collections of strings - is refused by name, on the principle the rest of the
writer keeps: a file ROOT misreads is worse than an error message.

Entries gather in memory a cluster at a time, and a cluster is written once
it holds ``cluster_size`` bytes of values, so an RNTuple far larger than
memory costs one cluster.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..compression import compress
from ..library import _module
from ..tree import Jagged
from ..winfo import INFOS
from .checksum import checksum
from .columns import ENCODINGS, Encoding, encode
from .envelope import FOOTER, HEADER, PAGE_LIST, Anchor, Builder, Link, envelope
from .fields import FUNDAMENTALS
from .schema import (
    COLLECTION,
    LEAF,
    RECORD,
    REPETITIVE,
    Column,
    Field,
    Schema,
    write_description,
)

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from ..writer import WritableDirectory

__all__ = [
    "WritableRNTuple",
    "CollectionSpec",
    "RecordSpec",
    "spec_of",
    "CLUSTER_BYTES",
    "PAGE_BYTES",
]

#: How many bytes of values gather before a cluster is written. A reader
#: reads a range a cluster at a time, so smaller clusters make a small range
#: cheaper to reach, and larger ones compress better and cost fewer pages.
CLUSTER_BYTES = 32 << 20
#: The most one page holds before compression: ROOT's own default.
PAGE_BYTES = 1 << 20
#: The largest a key may be; nothing written here comes near it.
MAX_KEY = 1 << 30
#: The format version written into the anchor: the first public release,
#: whose footer ends at the cluster groups, which is what is written.
VERSION = (1, 0, 0, 0)
#: What the header says wrote it.
WRITER = "xrdroot"

#: How ROOT describes the anchor's class, harvested from the streamer
#: information of ``test_int_float_rntuple_v1-0-0-0.root``, which ROOT 6.35
#: wrote - with the same ``TStreamerInfo`` record version :mod:`..writer`
#: writes - so that a file from here describes ``ROOT::RNTuple`` as ROOT does.
ANCHOR_INFO = (
    0x4BA21BF5,
    2,
    tuple(
        ("TStreamerBasicType", f"f{member}", "", stype, size, 0, 0, (0, 0, 0, 0, 0), cxx, ())
        for member, stype, size, cxx in [
            *((f"Version{part}", 12, 2, "unsigned short") for part in ("Epoch", "Major")),
            *((f"Version{part}", 12, 2, "unsigned short") for part in ("Minor", "Patch")),
            *((f"{part}Header", 17, 8, "ULong64_t") for part in ("Seek", "NBytes", "Len")),
            *((f"{part}Footer", 17, 8, "ULong64_t") for part in ("Seek", "NBytes", "Len")),
            ("MaxKeySize", 17, 8, "ULong64_t"),
        ]
    ),
)
INFOS.setdefault("ROOT::RNTuple", ANCHOR_INFO)

#: The column type each number is written as: split and plain.
SPLIT = {
    "int16": "SplitInt16",
    "uint16": "SplitUInt16",
    "int32": "SplitInt32",
    "uint32": "SplitUInt32",
    "int64": "SplitInt64",
    "uint64": "SplitUInt64",
    "float32": "SplitReal32",
    "float64": "SplitReal64",
}
PLAIN = {"bool": "Bit", "int8": "Int8", "uint8": "UInt8"}
#: The C++ name of each number, as ROOT normalises it.
CXX = {numpy: cxx for cxx, numpy in FUNDAMENTALS.items() if cxx not in ("char", "std::byte")}
#: The Python types a field can be declared with, and what each writes.
PYTHON = {bool: "bool", int: "int64", float: "float64"}
#: The spellings of a vector a field can be declared with.
VECTORS = ("std::vector<", "ROOT::VecOps::RVec<", "ROOT::RVec<", "vector<")
#: How a ``std::bitset<N>`` field is declared, its width between the angle brackets.
BITSETS = "std::bitset<"
_BY_NAME = {kind.name: kind for kind in ENCODINGS.values()}


def _refuse(name: str, spec: Any) -> ValueError:
    return ValueError(
        f"the field {name!r} is declared as {spec!r}; the fields an RNTuple is written "
        f"with here are numbers (a NumPy type, a C++ name such as 'std::int32_t', or "
        f"bool, int and float), str for std::string, and a vector of a number - or of "
        f"another such vector - spelled [type], 'std::vector<type>' or 'std::vector<"
        f"std::vector<type>>', as 'std::bitset<N>' for N bits, as a RecordSpec for a "
        f"class's record, or as a CollectionSpec for a set or a map. Variants and "
        f"collections of strings are not written, because their layout would be a guess"
    )


def _number(name: str, spec: Any) -> np.dtype[Any]:
    """The NumPy type a number field is declared as, however it was spelled."""
    if isinstance(spec, type) and spec in PYTHON:
        return np.dtype(PYTHON[spec])
    if isinstance(spec, str) and spec in FUNDAMENTALS:
        return np.dtype(FUNDAMENTALS[spec])
    try:
        dtype = np.dtype(spec)
    except TypeError:
        raise _refuse(name, spec) from None
    if dtype.newbyteorder("=").name not in CXX:
        raise _refuse(name, spec)
    return dtype.newbyteorder("=")


def _vector_item(spec: Any) -> Any:
    """What a vector is declared to hold, or ``None`` if ``spec`` is no vector."""
    if isinstance(spec, list) and len(spec) == 1:
        return spec[0]
    if isinstance(spec, str) and spec.startswith(VECTORS) and spec.endswith(">"):
        return spec[spec.index("<") + 1 : -1].strip()
    return None


def _vector_spec(name: str, spec: Any) -> tuple[np.dtype[Any], int] | None:
    """The type a vector holds at bottom and how deep its vectors nest, or ``None``.

    ``[[float]]`` and ``'std::vector<std::vector<float>>'`` are both a vector
    two deep over a fundamental number, which is the only thing a vector may
    hold here - a collection of strings or of records is refused whole, so
    that the sentence names the declaration that was given.
    """
    depth, item = 0, spec
    while (inner := _vector_item(item)) is not None:
        item, depth = inner, depth + 1
    if not depth:
        return None
    try:
        return _number(name, item), depth
    except ValueError:
        raise _refuse(name, spec) from None


def _bitset(name: str, spec: Any) -> int | None:
    """How many bits a ``std::bitset`` field is declared with, or ``None`` if ``spec`` is none.

    ROOT writes the bitset as a repetitive field - its width in the field, and
    its bits one element each in a single column of bits - so the width is all
    the declaration has to say beyond the type's own name.
    """
    if not isinstance(spec, str) or not spec.startswith(BITSETS) or not spec.endswith(">"):
        return None
    written = spec[len(BITSETS) : -1].strip()
    if not written.isdigit():
        raise _refuse(name, spec)
    return int(written)


def _vector(dtype: np.dtype[Any], depth: int) -> str:
    """The C++ name of a vector of ``dtype`` nested ``depth`` deep."""
    name = CXX[dtype.name]
    for _ in range(depth):
        name = f"std::vector<{name}>"
    return name


class _Out:
    """One column being written: its description, and the values gathered."""

    __slots__ = ("column", "kind", "pieces", "count", "first", "pages")

    def __init__(self, column: Column) -> None:
        self.column = column
        self.kind: Encoding = ENCODINGS[column.type]
        self.pieces: list[np.ndarray[Any, Any]] = []
        #: How many elements the cluster being gathered holds.
        self.count = 0
        #: The element the cluster being gathered starts at.
        self.first = 0
        #: For each cluster written, where its pages went.
        self.pages: list[tuple[int, list[tuple[int, int, int]]]] = []

    def add(self, values: np.ndarray[Any, Any]) -> None:
        self.pieces.append(values)
        self.count += len(values)

    def take(self) -> np.ndarray[Any, Any]:
        """Everything gathered for this cluster, and a fresh start for the next."""
        values = np.concatenate(self.pieces) if self.pieces else np.zeros(0, self.kind.dtype)
        self.pieces, self.count = [], 0
        return values


class _Writer:
    """What turns one field's values for many entries into its columns.

    :meth:`prepare` checks and converts the whole batch before anything is
    kept, so a batch that does not fit is refused whole; :meth:`put` then
    hands entries ``a`` to ``b`` of it to the columns, a cluster at a time.
    """

    __slots__ = ("name", "outs")

    def __init__(self, name: str) -> None:
        self.name = name
        self.outs: list[_Out] = []

    def prepare(self, values: Any) -> Any:
        raise NotImplementedError

    def length(self, prepared: Any) -> int:
        raise NotImplementedError

    def sizes(self, prepared: Any) -> np.ndarray[Any, Any]:
        """How many bytes each entry adds, which is what fills a cluster."""
        raise NotImplementedError

    def put(self, prepared: Any, a: int, b: int) -> None:
        raise NotImplementedError


class _Record(_Writer):
    """A class: no column of its own, and one field per member the class declares.

    ROOT writes a class as a record - ``RRecordField`` - whose members are fields
    of their own, so that a value of the class is its members' values and nothing
    else is stored. The record's own bytes are the bytes of the member fields,
    which is what a record's share of a cluster is.
    """

    __slots__ = ("members",)

    def __init__(self, name: str) -> None:
        super().__init__(name)
        #: Each member's name, and the writer of its own field.
        self.members: list[tuple[str, _Writer]] = []

    def prepare(self, values: Any) -> Any:
        rows = _members(self.name, values, [member for member, _ in self.members])
        return rows, [
            writer.prepare([row[member] for row in rows]) for member, writer in self.members
        ]

    def length(self, prepared: Any) -> int:
        return len(prepared[0])

    def sizes(self, prepared: Any) -> np.ndarray[Any, Any]:
        members = prepared[1]
        total = np.zeros(len(prepared[0]), np.int64)
        for (_member, writer), member_values in zip(self.members, members, strict=True):
            total += writer.sizes(member_values)
        return total

    def put(self, prepared: Any, a: int, b: int) -> None:
        members = prepared[1]
        for (_member, writer), member_values in zip(self.members, members, strict=True):
            writer.put(member_values, a, b)


class _Scalar(_Writer):
    """A number or a ``bool`` per entry, in one column."""

    __slots__ = ("dtype",)

    def __init__(self, name: str, dtype: np.dtype[Any]) -> None:
        super().__init__(name)
        self.dtype = dtype

    def prepare(self, values: Any) -> Any:
        return _numbers(self.name, values, self.dtype)

    def length(self, prepared: Any) -> int:
        return len(prepared)

    def sizes(self, prepared: Any) -> np.ndarray[Any, Any]:
        return np.full(len(prepared), self.dtype.itemsize, np.int64)

    def put(self, prepared: Any, a: int, b: int) -> None:
        self.outs[0].add(prepared[a:b])


class _Rows(_Writer):
    """A ``std::string`` or a vector - of numbers, or of vectors nested deeper.

    Every level of items has an offset column saying where that level's items
    end, and the leaves have a column of their own. The outermost offset
    column is fed each entry's length here, and every one of them is made
    into ends counted from the start of the cluster when the cluster is
    written; a ``std::vector<std::vector<float>>`` is therefore two offset
    columns over one of floats, which is how ROOT lays it out.
    """

    __slots__ = ("dtype", "depth")

    def __init__(self, name: str, dtype: np.dtype[Any], depth: int = 1) -> None:
        super().__init__(name)
        self.dtype = dtype
        #: How many vectors nest here; one is a plain vector of numbers.
        self.depth = depth

    def prepare(self, values: Any) -> Any:
        return _deep_rows(self.name, values, self.dtype, self.depth)

    def length(self, prepared: Any) -> int:
        return len(prepared[1][0]) - 1

    def sizes(self, prepared: Any) -> np.ndarray[Any, Any]:
        offsets, leaves = _per_entry(prepared[1])
        return offsets * 8 + leaves * self.dtype.itemsize

    def put(self, prepared: Any, a: int, b: int) -> None:
        content, levels = prepared
        self.outs[0].add(np.diff(levels[0][a : b + 1]))
        lo, hi = int(levels[0][a]), int(levels[0][b])
        for out, level in zip(self.outs[1 : self.depth], levels[1:], strict=True):
            out.add(np.diff(level[lo : hi + 1]))
            lo, hi = int(level[lo]), int(level[hi])
        self.outs[-1].add(content[lo:hi])


class _Text(_Rows):
    """A ``std::string`` per entry: its bytes, in a column of characters."""

    __slots__ = ()

    def __init__(self, name: str) -> None:
        super().__init__(name, np.dtype("int8"))

    def prepare(self, values: Any) -> Any:
        content, offsets = _strings(self.name, values)
        return content, [offsets]


class _Collection(_Writer):
    """A set, or a map: an index column saying where each entry's items end, then the items.

    ROOT writes a ``std::unordered_set<T>`` as a collection over ``T`` and a
    ``std::unordered_map<K, V>`` as one over the ``std::pair<K, V>`` record, which
    is the layout a vector has - the index column, then the item's own columns, one
    field under another. An entry's items are given as a row; a map's are its keys
    and values, as pairs or as a mapping, which become the pair's two members.
    """

    __slots__ = ("item", "keyed")

    def __init__(self, name: str, keyed: bool) -> None:
        super().__init__(name)
        #: The writer of the item field, which the declaration sets once it is made.
        self.item: _Writer | None = None
        #: Whether an item is a ``std::pair`` - a map's key and value.
        self.keyed = keyed

    def prepare(self, values: Any) -> Any:
        assert self.item is not None
        rows = [_row_items(self.name, row) for row in values]
        offsets = np.concatenate([[0], np.cumsum([len(row) for row in rows], dtype=np.int64)])
        items = [item for row in rows for item in row]
        if self.keyed:
            items = [_pair_item(self.name, item) for item in items]
        return offsets, self.item.prepare(items)

    def length(self, prepared: Any) -> int:
        return len(prepared[0]) - 1

    def sizes(self, prepared: Any) -> np.ndarray[Any, Any]:
        offsets, items = prepared
        assert self.item is not None
        kept = np.concatenate([[0], np.cumsum(self.item.sizes(items), dtype=np.int64)])
        return 8 * np.diff(offsets) + kept[offsets[1:]] - kept[offsets[:-1]]

    def put(self, prepared: Any, a: int, b: int) -> None:
        offsets, items = prepared
        assert self.item is not None
        self.outs[0].add(np.diff(offsets[a : b + 1]))
        self.item.put(items, int(offsets[a]), int(offsets[b]))


def _row_items(name: str, row: Any) -> list[Any]:
    """One entry's items: a row of them, or a mapping's keys and values paired up."""
    if isinstance(row, Mapping):
        return list(row.items())
    if isinstance(row, np.ndarray) and row.ndim != 1:
        raise ValueError(
            f"the field {name!r} is a collection, and an entry given for it is shaped "
            f"{row.shape}; give one row of items per entry"
        )
    if isinstance(row, (str, bytes)) or not isinstance(row, Iterable):
        raise ValueError(
            f"the field {name!r} is a collection, and an entry given for it is a "
            f"{type(row).__name__}; give one row of items per entry"
        )
    return list(row)


def _pair_item(name: str, item: Any) -> dict[str, Any]:
    """One item of a map as the two members of its ``std::pair`` record."""
    if isinstance(item, Mapping):
        return dict(item)
    given = list(item) if isinstance(item, Iterable) and not isinstance(item, (str, bytes)) else []
    if len(given) != 2:
        raise ValueError(
            f"the field {name!r} is a map, so each of its items is a key and a value - "
            f"a pair, or a mapping of its two members - not a {type(item).__name__}"
        )
    return {"_0": given[0], "_1": given[1]}


class _Bits(_Writer):
    """A ``std::bitset<N>``: its ``N`` bits, the lowest first, in one column of bits."""

    __slots__ = ("size",)

    def __init__(self, name: str, size: int) -> None:
        super().__init__(name)
        self.size = size

    def prepare(self, values: Any) -> Any:
        return _bit_rows(self.name, values, self.size)

    def length(self, prepared: Any) -> int:
        return len(prepared)

    def sizes(self, prepared: Any) -> np.ndarray[Any, Any]:
        return np.full(len(prepared), (self.size + 7) // 8, np.int64)

    def put(self, prepared: Any, a: int, b: int) -> None:
        self.outs[0].add(prepared[a:b].reshape(-1))


def _bit_rows(name: str, values: Any, size: int) -> np.ndarray[Any, Any]:
    """A batch of bitsets as one row of ``size`` bits per entry, the lowest bit first."""
    if isinstance(values, np.ndarray) and values.ndim == 2:
        if values.shape[1] != size:
            raise ValueError(
                f"the field {name!r} holds {size} bits per entry, and these are shaped "
                f"{values.shape}; give a number per entry, or that many bits of each"
            )
        return values != 0
    return np.array([_bit_row(name, value, size) for value in values], bool).reshape(
        len(values), size
    )


def _bit_row(name: str, value: Any, size: int) -> Any:
    """One entry's bits: a number, a row of bits, or a ``std::bitset``'s ``to_ulong()``."""
    handed = getattr(value, "to_ulong", None)
    if handed is not None:
        return _bit_row(name, int(handed()), size)
    if isinstance(value, (int, np.integer)):
        if value >> size:
            raise ValueError(
                f"the field {name!r} holds {size} bits per entry, and the number {value} "
                f"needs more than that, so it would not go in without losing its top bits"
            )
        return [(int(value) >> bit) & 1 for bit in range(size)]
    row = np.asarray(value)
    if row.shape != (size,):
        raise ValueError(
            f"the field {name!r} holds {size} bits per entry, and one of these is shaped "
            f"{row.shape}; give a number per entry, or that many bits of each"
        )
    return row != 0


def _numbers(name: str, values: Any, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    """A batch of numbers for one field, refusing any the field cannot hold.

    A float is not quietly truncated into an integer field, and an integer
    that does not fit the field is refused rather than wrapped round - but
    any integer that fits goes in, signed or not, so that a plain Python
    ``1`` fills a ``std::uint16_t``.
    """
    given = np.asarray(values)
    if given.ndim != 1:
        raise ValueError(
            f"the field {name!r} takes one {dtype} per entry, and these are shaped "
            f"{given.shape}; give a one-dimensional array with a value for every entry"
        )
    integers = dtype.kind in "iu" and given.dtype.kind in "iu"
    if given.size and not (integers or np.can_cast(given.dtype, dtype, "same_kind")):
        raise ValueError(
            f"the field {name!r} holds {dtype} values, and these are {given.dtype}, which "
            f"would not go into it without losing what they are"
        )
    if integers and given.size:
        _require_fits(name, given, dtype)
    return given.astype(dtype)


def _require_fits(name: str, given: np.ndarray[Any, Any], dtype: np.dtype[Any]) -> None:
    limits = np.iinfo(dtype)
    low, high = int(given.min()), int(given.max())
    if low < limits.min or high > limits.max:
        raise ValueError(
            f"the field {name!r} holds {dtype} values, from {limits.min} to "
            f"{limits.max}, and these run from {low} to {high}"
        )


def _vectors(name: str, values: Any, dtype: np.dtype[Any]) -> tuple[Any, Any]:
    """A batch of rows for one vector field: every item end to end, and where each ends."""
    if isinstance(values, Jagged):
        content, offsets = values.flat, values.offsets - values.offsets[0]
    elif type(values).__module__.partition(".")[0] == "awkward":
        content, offsets = _awkward_rows(values)
    else:
        content, offsets = _listed_rows(name, values, dtype)
    return _numbers(name, content, dtype), np.asarray(offsets, np.int64)


def _deep_rows(
    name: str, values: Any, dtype: np.dtype[Any], depth: int
) -> tuple[Any, list[Any]]:
    """The leaves end to end, and the offset column of every level of items above them.

    One level is a row per entry; deeper than that, each entry's items are
    themselves rows, and are descended one at a time, so what comes back is
    ``depth`` offset columns, each counted over the items of the level above
    it, then the leaves. A row's own offsets start at zero, so they are
    accumulated as the rows go by to give the ends the cluster will keep.
    """
    if depth == 1:
        content, offsets = _vectors(name, values, dtype)
        return content, [offsets]
    return _nested_rows(name, _sub_rows(name, values), dtype, depth)


def _nested_rows(
    name: str, rows: list[Any], dtype: np.dtype[Any], depth: int
) -> tuple[Any, list[Any]]:
    """One entry's rows made into the leaves and offsets of the batch they belong to."""
    outer = np.zeros(len(rows) + 1, np.int64)
    under: list[list[int]] = [[] for _ in range(depth - 1)]
    contents = []
    for at, row in enumerate(rows):
        content, levels = _deep_rows(name, row, dtype, depth - 1)
        contents.append(content)
        outer[at + 1] = len(levels[0]) - 1
        _collect(under, levels)
    return _joined(contents, under, outer, dtype)


def _collect(under: list[list[int]], levels: list[Any]) -> None:
    """Add one row's counts at every level to the batch's, which the offsets sum."""
    for keep, level in zip(under, levels, strict=True):
        keep.extend(np.diff(level).tolist())


def _joined(
    contents: list[Any], under: list[list[int]], outer: np.ndarray[Any, Any], dtype: np.dtype[Any]
) -> tuple[Any, list[Any]]:
    """The rows' leaves and counts joined end to end: the leaves, then every offset column."""
    filled = [item for item in contents if len(item)]
    content = np.concatenate(filled) if filled else np.zeros(0, dtype)
    offsets = [np.concatenate([[0], np.cumsum(keep, dtype=np.int64)]) for keep in under]
    return content, [np.cumsum(outer), *offsets]


def _sub_rows(name: str, values: Any) -> list[Any]:
    """Each element's own items, so that a vector of vectors can be descended row by row."""
    if type(values).__module__.partition(".")[0] == "awkward":
        return list(_module("ak").to_list(values))
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(
            f"the field {name!r} is a vector of vectors, and a batch given for it is a "
            f"{type(values).__name__}; give a list with one row of rows per entry"
        )
    return list(values)


def _per_entry(levels: list[Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """How many offset values, and how many leaves, every entry of the batch holds.

    The levels are offsets columns one inside the next, so what an entry holds
    of a deeper level is the difference between that level's offsets at the
    two ends of the range the level above gives it; the outermost column holds
    one value per entry whatever the rest does.
    """
    offsets = np.ones(len(levels[0]) - 1, np.int64)
    at = levels[0]
    for level in levels[1:]:
        offsets += np.diff(at)
        at = level[at]
    return offsets, np.diff(at)


def _listed_rows(name: str, values: Any, dtype: np.dtype[Any]) -> tuple[Any, Any]:
    """Rows given one by one - lists, or arrays - as their items and where each ends."""
    rows = [np.asarray(row) for row in values]
    if any(row.ndim != 1 for row in rows):
        raise ValueError(
            f"the field {name!r} is a vector of {dtype}, and an entry given for it is "
            f"not a row of numbers"
        )
    filled = [row for row in rows if len(row)]  # an empty row is float64 to NumPy
    content = np.concatenate(filled) if filled else np.zeros(0, dtype)
    return content, np.concatenate([[0], np.cumsum([len(row) for row in rows], dtype=np.int64)])


def _awkward_rows(values: Any) -> tuple[Any, Any]:
    ak = _module("ak")
    counts = ak.to_numpy(ak.num(values, axis=1))
    offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    return ak.to_numpy(ak.flatten(values, axis=1)), offsets


def _strings(name: str, values: Any) -> tuple[Any, Any]:
    """A batch of strings: every byte end to end, and where each string ends."""
    if isinstance(values, str):
        raise ValueError(
            f"the field {name!r} takes a str per entry, and was given one str for the "
            f"whole batch; give a list with a string for every entry"
        )
    raw = []
    for value in values:
        if not isinstance(value, (str, bytes)):
            raise ValueError(
                f"the field {name!r} holds strings, and an entry given for it is a "
                f"{type(value).__name__}"
            )
        raw.append(value.encode("utf-8", "surrogateescape") if isinstance(value, str) else value)
    lengths = [len(item) for item in raw]
    content = np.frombuffer(b"".join(raw), np.int8)
    return content, np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])


def spec_of(name: str, values: Any) -> Any:
    """The declaration a field of these values needs, read off the values.

    One number per entry is that number's type; rows of numbers - a
    :class:`~xrdroot.tree.Jagged`, an Awkward list, a list of arrays - are a
    vector of the type of their items, and rows of rows as many vectors as
    there are levels of them; strings are ``str``.
    """
    if isinstance(values, Jagged):
        return [values.content.dtype]
    if type(values).__module__.partition(".")[0] == "awkward":
        return _awkward_spec(name, values)
    given = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if given.dtype.kind in "biuf" and given.ndim == 1:
        return given.dtype
    if given.dtype.kind in "US":
        return str
    return _object_spec(name, list(values))


def _object_spec(name: str, values: list[Any]) -> Any:
    """The declaration a list of rows needs, worked out from what the rows hold.

    Each level of rows descended is a vector, so a list of rows of rows is a
    vector of vectors, however deep that goes. An empty row says nothing about
    what it holds and is no trouble, but a batch with no leaves in it at all
    cannot say what it holds and is refused.
    """
    if values and all(isinstance(value, (str, bytes)) for value in values):
        return str
    depth, leaves = _descend(values)
    if not depth or not leaves:
        raise _not_rows(name)
    found = np.asarray(leaves)
    if found.dtype.kind not in "biuf":
        raise _not_rows(name)
    return _nested_spec(found.dtype, depth)


def _descend(values: list[Any]) -> tuple[int, list[Any]]:
    """How many levels of rows ``values`` holds, and the leaves at the bottom of them."""
    depth, items = 0, values
    while items and all(_is_row(item) for item in items):
        items, depth = [leaf for item in items for leaf in item], depth + 1
    return depth, items


def _nested_spec(dtype: np.dtype[Any], depth: int) -> Any:
    """A vector of ``dtype`` declared ``depth`` vectors deep: ``[[float]]`` for two."""
    spec: Any = dtype
    for _ in range(depth):
        spec = [spec]
    return spec


def _is_row(value: Any) -> bool:
    """Whether a value is a row with items to count: a list, an array, an Awkward list."""
    return isinstance(value, (list, tuple, np.ndarray)) or (
        type(value).__module__.partition(".")[0] == "awkward"
    )


def _not_rows(name: str) -> ValueError:
    return ValueError(
        f"the field {name!r} is given values that are neither numbers, nor strings, nor "
        f"rows of numbers, which are what an RNTuple is written from here"
    )


def _awkward_spec(name: str, values: Any) -> Any:
    """What an Awkward list needs: one axis off is what its items are, however few."""
    ak = _module("ak")
    if values.ndim == 1:  # numbers, or strings, which NumPy holds as either
        return spec_of(name, ak.to_numpy(values))
    return [spec_of(name, ak.flatten(values, axis=1))]


class CollectionSpec:
    """A field that is a set or a map: the C++ name it is written with, and its item's.

    ROOT writes any of ``std::set``, ``std::unordered_set``, ``std::multiset`` and
    their unordered kin as a collection over their item, and ``std::map`` and its kin
    as one over a ``std::pair`` record - which is the layout a reader of either
    library finds. A collection is declared where a number would be:
    ``f.rntuple('events', {'spam': CollectionSpec('std::unordered_set<double>', float)})``.
    """

    __slots__ = ("cxx", "item", "keyed")

    def __init__(self, cxx: str, item: Any) -> None:
        self.cxx = str(cxx)
        #: The declaration of the item field under this one.
        self.item = item
        #: Whether the item is a ``std::pair``, as a map's is.
        self.keyed = str(item.cxx if isinstance(item, RecordSpec) else "").startswith(
            "std::pair<"
        )

    def __repr__(self) -> str:
        return f"<CollectionSpec {self.cxx}>"


class RecordSpec:
    """A field that is a class: its C++ type name, and its members as fields of their own.

    ROOT reads a class's dictionary and writes the class as a record of what the
    dictionary declares, so the name is the class's and the members are the whole
    of what is stored - a name such as ``'AtlasLikeDataVector<CustomStruct>'``
    with the members ``CustomStruct`` declares. A record is declared where a
    number would be: ``f.rntuple('events', {'evt': RecordSpec(...)})``.
    """

    __slots__ = ("cxx", "members")

    def __init__(self, cxx: str, members: Iterable[tuple[str, Any]]) -> None:
        self.cxx = str(cxx)
        #: Each member's name, and the declaration of its field.
        self.members = [(str(name), spec) for name, spec in members]

    def __repr__(self) -> str:
        return f"<RecordSpec {self.cxx} of {', '.join(name for name, _ in self.members)}>"


def _members(name: str, values: Any, members: list[str]) -> list[Mapping[str, Any]]:
    """A batch of a record's values: one mapping of member to value per entry."""
    rows = list(values)
    for row in rows:
        _check_record(name, row, members)
    return rows


def _check_record(name: str, row: Any, members: list[str]) -> None:
    """One entry of a record: every member, and nothing the record does not hold."""
    if not isinstance(row, Mapping):
        raise ValueError(
            f"the field {name!r} is a record, so each of its entries is "
            f"{_record_shape(members)}, not a {type(row).__name__}"
        )
    missing = [member for member in members if member not in row]
    unknown = [str(member) for member in row if member not in members]
    if missing or unknown:
        raise ValueError(
            f"this entry of the record {name!r} has {_trouble(missing, unknown)}; its "
            f"members are {', '.join(members) or 'none'}, and every entry needs all of them"
        )


def _record_shape(members: list[str]) -> str:
    """What one entry of a record is: a mapping of its members, or nothing at all."""
    if not members:
        return "an empty mapping, because the record has no members"
    return f"a mapping of member to value, as in {{{members[0]!r}: ...}}"


def _trouble(missing: list[str], unknown: list[str]) -> str:
    """What an entry has wrong with its names: those it lacks, and those it should not."""
    apart = [f"no {', '.join(missing)}"] if missing else []
    if unknown:
        apart.append(f"{', '.join(unknown)}, which it does not have")
    return " and ".join(apart)


class WritableRNTuple:
    """An RNTuple being written: named fields, then entries, a cluster at a time.

        >>> with xrdroot.create("out.root") as f:            # doctest: +SKIP
        ...     ntuple = f.rntuple("events", {"n": "std::int32_t", "pt": [np.float32]})
        ...     ntuple.extend({"n": counts, "pt": jagged_pts})
        ...     ntuple.fill(n=2, pt=[10.5, 3.25])

    A field is declared as a number - a NumPy type, a C++ name such as
    ``'std::uint16_t'`` or ``'double'``, or a Python ``bool``, ``int`` or
    ``float`` - or as ``str`` for a ``std::string``, or as a vector of a
    number, ``[np.float32]`` or ``'std::vector<float>'``, or as vectors of
    vectors, ``[[np.float32]]`` or ``'std::vector<std::vector<float>>'``, as
    deep as it goes, or as a :class:`RecordSpec` for a class, whose every entry
    is a mapping of member to value. Every entry needs a value for every field. The header,
    the page list, the footer and the anchor are written when the file closes.
    """

    def __init__(
        self,
        file: WritableDirectory,
        name: str,
        fields: Mapping[str, Any],
        cycle: int,
        cluster_size: int = CLUSTER_BYTES,
        page_size: int = PAGE_BYTES,
        description: str = "",
    ) -> None:
        _check_setup(name, fields, cluster_size, page_size)
        self._file = file
        self._cycle = cycle
        self.name = name
        self._cluster_size = cluster_size
        self._page_size = page_size
        self._split = file._codes != 0
        self._schema = Schema(name, description, WRITER)
        self._writers: dict[str, _Writer] = {}
        for field, spec in fields.items():
            _check_name(field, "field")
            self._writers[field] = self._declare(field, spec)
        self._schema.link()
        self._entries = 0
        self._clusters: list[tuple[int, int]] = []
        self._gathered = 0
        self._bytes = 0

    def __repr__(self) -> str:
        return (
            f"<WritableRNTuple {self.name!r} with {len(self._writers)} fields "
            f"and {self._entries} entries so far>"
        )

    def __len__(self) -> int:
        return self._entries

    @property
    def num_entries(self) -> int:
        return self._entries

    @property
    def closed(self) -> bool:
        """Whether the file this RNTuple is in has been closed - by the writer, or by whoever
        opened it - so that no entry can be added any more."""
        return bool(self._file.closed)

    @property
    def fields(self) -> dict[str, str]:
        """What each field is, in the C++ the file will say: ``{'pt': 'std::vector<float>'}``."""
        return {field.name: field.type for field in self._schema.tops()}

    def _declare(self, name: str, spec: Any, parent: int | None = None) -> _Writer:
        if isinstance(spec, RecordSpec):
            return self._record(name, spec, parent)
        writer: _Writer
        if spec is str or spec in ("std::string", "string"):
            writer = _Text(name)
            self._add(name, "std::string", LEAF, [self._index(), "Char"], writer, parent)
            return writer
        size = _bitset(name, spec)
        if size is not None:
            made = _Bits(name, size)
            self._add(name, str(spec), LEAF, ["Bit"], made, parent, size)
            return made
        if isinstance(spec, CollectionSpec):
            return self._collection(name, spec, parent)
        found = _vector_spec(name, spec)
        if found is None:
            dtype = _number(name, spec)
            writer = _Scalar(name, dtype)
            self._add(name, CXX[dtype.name], LEAF, [self._kind(dtype)], writer, parent)
            return writer
        dtype, depth = found
        writer = _Rows(name, dtype, depth)
        under = self._add(name, _vector(dtype, depth), COLLECTION, [self._index()], writer, parent)
        for level in range(depth - 1, 0, -1):
            under = self._add(
                "_0", _vector(dtype, level), COLLECTION, [self._index()], writer, under
            )
        self._add("_0", CXX[dtype.name], LEAF, [self._kind(dtype)], writer, under)
        return writer

    def _collection(self, name: str, spec: CollectionSpec, parent: int | None) -> _Writer:
        """A set or a map: the index column over the items, and the item field under it.

        The item is declared the way any other field is - a number, a string, a
        vector, or for a map the ``std::pair`` record of the key and the value - so
        that a collection of collections nests as ROOT nests it.
        """
        writer = _Collection(name, spec.keyed)
        under = self._add(name, spec.cxx, COLLECTION, [self._index()], writer, parent)
        writer.item = self._declare("_0", spec.item, under)
        writer.outs.extend(writer.item.outs)
        return writer

    def _record(self, name: str, spec: RecordSpec, parent: int | None) -> _Writer:
        """A record: the class's name, and a field under it for every member.

        The record keeps the columns of no field of its own, so every page it is
        written in belongs to a member - which is what ROOT writes for a class and
        what makes an RNTuple written here readable by a reader of either.
        """
        writer = _Record(name)
        under = self._add(name, spec.cxx, RECORD, [], writer, parent)
        for member, member_spec in spec.members:
            _check_name(member, "field")
            writer.members.append((member, self._declare(member, member_spec, under)))
        writer.outs = [out for _member, made in writer.members for out in made.outs]
        return writer

    def _index(self) -> str:
        return "SplitIndex64" if self._split else "Index64"

    def _kind(self, dtype: np.dtype[Any]) -> str:
        if dtype.name in PLAIN:
            return PLAIN[dtype.name]
        split = SPLIT[dtype.name]
        return split if self._split else split[len("Split") :]

    def _add(
        self,
        name: str,
        cxx: str,
        role: int,
        kinds: list[str],
        writer: _Writer,
        parent: int | None = None,
        array_size: int = 0,
    ) -> int:
        """A field, and its columns; a top-level field is its own parent.

        ``array_size`` is how many fixed elements a repetitive field - a
        ``std::bitset`` - holds to every entry, which is what ROOT keeps in the
        field itself rather than in a column of its own.
        """
        fields = self._schema.fields
        id = len(fields)
        field = Field(id, id if parent is None else parent, role, name, cxx)
        if array_size:
            field.flags |= REPETITIVE
            field.array_size = array_size
        fields.append(field)
        for kind in kinds:
            self._column(kind, id, writer)
        return id

    def _column(self, kind: str, field: int, writer: _Writer) -> None:
        code = _BY_NAME[kind]
        columns = self._schema.columns
        column = Column(len(columns), code.code, 8 * code.dtype.itemsize, field)
        column.bits = 1 if kind == "Bit" else column.bits
        columns.append(column)
        writer.outs.append(_Out(column))

    def fill(self, **values: Any) -> None:
        """Add one entry, with a value for every field.

        >>> ntuple.fill(n=2, pt=[10.5, 3.25], name="first")   # doctest: +SKIP
        """
        self.extend({name: [value] for name, value in values.items()})

    def extend(self, rows: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> None:
        """Add many entries: a batch of values for each field, or entry after entry.

            >>> ntuple.extend({"n": counts, "pt": pts})          # doctest: +SKIP

        A mapping of field name to values - an array of numbers, a list of
        strings, a :class:`~xrdroot.tree.Jagged` or an Awkward list or a list
        of rows for a vector - is the fast way; every field needs the same
        number of entries. Anything else is taken as entries, each a mapping
        of field to value. Either way a batch that does not fit is refused
        whole, and nothing of it is kept.
        """
        if not isinstance(rows, Mapping):
            rows = self._columns_of(list(rows))
        self._require_open()
        self._require_fields(rows)
        prepared = {name: writer.prepare(rows[name]) for name, writer in self._writers.items()}
        count = self._count(prepared)
        costs = sum(writer.sizes(prepared[name]) for name, writer in self._writers.items())
        ends = np.concatenate([[0], np.cumsum(costs, dtype=np.int64)])
        at = 0
        while at < count:
            room = self._cluster_size - self._bytes
            stop = int(np.searchsorted(ends, ends[at] + room, side="right")) - 1
            stop = min(max(stop, at + 1), count)
            for name, writer in self._writers.items():
                writer.put(prepared[name], at, stop)
            self._take(stop - at, int(ends[stop] - ends[at]))
            at = stop

    def _columns_of(self, rows: list[Mapping[str, Any]]) -> dict[str, list[Any]]:
        """Entries one after another, made into a batch of values per field."""
        for row in rows:
            self._require_fields(row)
        return {name: [row[name] for row in rows] for name in self._writers}

    def _count(self, prepared: dict[str, Any]) -> int:
        counts = {name: writer.length(prepared[name]) for name, writer in self._writers.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"the fields given to {self.name!r} hold different numbers of entries "
                f"({', '.join(f'{name}: {count}' for name, count in counts.items())}); an "
                f"RNTuple whose fields disagree about that is one nothing can read"
            )
        return next(iter(counts.values()))

    def _take(self, entries: int, size: int) -> None:
        self._entries += entries
        self._gathered += entries
        self._bytes += size
        if self._bytes >= self._cluster_size:
            self._flush()

    def _require_open(self) -> None:
        if self._file.closed:
            raise ValueError(
                f"the file this RNTuple is in is closed; {self.name!r} holds the "
                f"{self._entries} entries it was given"
            )

    def _require_fields(self, row: Mapping[str, Any]) -> None:
        missing = [name for name in self._writers if name not in row]
        unknown = [str(name) for name in row if name not in self._writers]
        if missing or unknown:
            raise ValueError(
                f"this entry of {self.name!r} has {_trouble(missing, unknown)}; its fields "
                f"are {', '.join(self._writers)}, and every entry needs all of them"
            )

    def _outs(self) -> list[_Out]:
        return [out for writer in self._writers.values() for out in writer.outs]

    def _flush(self) -> None:
        """Write the cluster gathered: every column's pages, then note where they went."""
        if not self._gathered:
            return
        for out in self._outs():
            values = out.take()
            if out.kind.index:
                values = np.cumsum(values, dtype=np.int64)
            out.pages.append((out.first, self._pages(out.kind, values)))
            out.first += len(values)
        self._clusters.append((self._entries - self._gathered, self._gathered))
        self._gathered, self._bytes = 0, 0

    def _pages(self, kind: Encoding, values: np.ndarray[Any, Any]) -> list[tuple[int, int, int]]:
        """One column of one cluster, in pages of at most ``page_size`` bytes each."""
        width = kind.dtype.itemsize * (1 if kind.name != "Bit" else 1 / 8)
        per = max(int(self._page_size // width), 1)
        pages = []
        for start in range(0, len(values), per):
            chunk = values[start : start + per]
            raw = self._squeezed(encode(kind, chunk))
            offset = self._blob(raw + checksum(raw).to_bytes(8, "little"))
            pages.append((len(chunk), len(raw), offset))
        return pages

    def _squeezed(self, raw: bytes) -> bytes:
        """``raw`` compressed with the file's algorithm, when that makes it smaller."""
        algorithm = self._file._file._algorithm
        if algorithm is None:
            return raw
        packed = compress(raw, algorithm, self._file._file._level)
        return packed if len(packed) < len(raw) else raw

    def _blob(self, payload: bytes) -> int:
        """An unlisted ``RBlob`` key holding ``payload``; where the payload starts."""
        seek, nbytes = self._file._put("RBlob", "", "", payload, 1, listed=False, packed=False)
        return seek + nbytes - len(payload)

    def _envelope(self, kind: int, payload: bytes) -> tuple[Link, bytes]:
        whole = envelope(kind, payload)
        stored = self._squeezed(whole)
        return Link(self._blob(stored), len(stored), len(whole)), whole

    def _finish(self) -> None:
        """Flush what is left, then write the header, page list, footer and anchor."""
        self._flush()
        header = Builder()
        header.pack("Q", 0)  # feature flags: nothing an older reader would misread
        for text in (self.name, self._schema.description, WRITER):
            header.string(text)
        write_description(header, self._schema)
        header_link, whole = self._envelope(HEADER, bytes(header.data))
        header_sum = int.from_bytes(whole[-8:], "little")
        footer = self._footer(header_sum)
        footer_link, _whole = self._envelope(FOOTER, footer)
        anchor = Anchor(VERSION, header_link, footer_link, MAX_KEY)
        self._file._put(
            "ROOT::RNTuple", self.name, "", anchor.payload(), self._cycle, listed=True, packed=False
        )

    def _footer(self, header_sum: int) -> bytes:
        out = Builder()
        out.pack("QQ", 0, header_sum)
        extension = out.record()
        write_description(out, None)  # the schema did not grow after the header
        out.close(extension)
        groups = out.list(1 if self._clusters else 0)
        if self._clusters:
            record = out.record()
            out.pack("QQI", 0, self._entries, len(self._clusters))
            out.link(self._page_list(header_sum))
            out.close(record)
        out.close(groups)
        return bytes(out.data)

    def _page_list(self, header_sum: int) -> Link:
        out = Builder()
        out.pack("Q", header_sum)
        summaries = out.list(len(self._clusters))
        for first, entries in self._clusters:
            record = out.record()
            out.pack("QQ", first, entries)
            out.close(record)
        out.close(summaries)
        outs = self._outs()
        top = out.list(len(self._clusters))
        for index in range(len(self._clusters)):
            columns = out.list(len(outs))
            for column in outs:
                self._column_pages(out, *column.pages[index])
            out.close(columns)
        out.close(top)
        link, _whole = self._envelope(PAGE_LIST, bytes(out.data))
        return link

    def _column_pages(self, out: Builder, first: int, pages: list[tuple[int, int, int]]) -> None:
        mark = out.list(len(pages))
        for count, size, offset in pages:
            out.pack("i", -count)  # negative: a checksum follows the page
            out.locator(size, offset)
        out.pack("qI", first, self._file._codes)
        out.close(mark)


def write_table(file: WritableDirectory, name: str, table: dict[str, Any] | None, obj: Any) -> None:
    """A table - a dict of arrays or any DataFrame - written as an RNTuple."""
    if table is None:
        raise ValueError(
            f"rntuple=True writes a table as an RNTuple, and a {type(obj).__name__} is not "
            f"one; give a dict of arrays, or a pandas, Polars or Arrow table"
        )
    fields = {column: spec_of(column, values) for column, values in table.items()}
    file.rntuple(name, fields).extend(table)


def _check_setup(name: str, fields: Any, cluster_size: Any, page_size: Any) -> None:
    _check_name(name, "RNTuple")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError(
            f"the fields of {name!r} are {fields!r}; an RNTuple is declared with a "
            f"mapping of field name to type, as in {{'pt': float}}, and needs at least one"
        )
    for what, size in (("cluster", cluster_size), ("page", page_size)):
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"the {what} size is {size!r}, which is not a size in bytes")


def _check_name(name: Any, what: str) -> None:
    """A name the specification allows: not empty, and no control, dot, space, slash."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"an RNTuple {what} name must be a string with something in it")
    bad = [char for char in name if char in "./\\ " or ord(char) < 32 or ord(char) == 127]
    if bad:
        raise ValueError(
            f"{name!r} is not a name an RNTuple {what} can have: the specification rules "
            f"out control characters, full stops, spaces, slashes and backslashes"
        )
