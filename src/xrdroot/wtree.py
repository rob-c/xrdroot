"""Writing a ``TTree``: columns declared once, then entries, then baskets.

A tree written here is laid out the way ROOT laid out the trees in
``tests/data`` - a ``TTree`` record holding one ``TBranch`` per column, each
with its own ``TLeaf`` saying what the values are, and the entries themselves
in ``TBasket`` records elsewhere in the file. That is what makes a tree worth
writing rather than an array per column: reading a range of entries back
costs one read per basket it touches, not one read of everything.

A column holds a number per entry, a fixed count of numbers per entry, a run
of numbers whose length changes from entry to entry, or a string. A run is
laid out the way ROOT lays out the leaf-list ``x[nx]/F``: a counter branch of
32-bit ints saying how long each row is, which the data leaf points at, and
baskets that carry a table of where each entry begins - rows of different
lengths cannot be found by arithmetic, so the basket has to say. A string is
a ``TLeafC``, a length and the bytes, with the same table behind it. Split
objects are refused by name rather than approximated, on the same principle
as the rest of the writer: a file ROOT misreads is worse than an error
message.

Entries are buffered a basket at a time, so a tree far larger than memory
costs one basket per column and nothing else. The tree's own record has to
say where every basket landed, so it is written when the file closes.
"""

from __future__ import annotations

import array
import struct
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .buffer import MAP_OFFSET
from .objects import LEAF_TYPES
from .tree import Jagged
from .writer import BASKET_BYTES, WBuffer, _checked, _keylen

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .writer import WritableFile

__all__ = ["WritableTree", "BASKET_BYTES"]

#: The record versions written here, which are the ones :mod:`.winfo`
#: describes: what the file says about its classes is what its records are.
TREE_VERSION = 19
BRANCH_VERSION = 12
LEAF_VERSION = 2
SUBLEAF_VERSION = 1
OBJARRAY_VERSION = 3
BASKET_VERSION = 3
ATTRIBUTE_VERSION = 2

#: What ROOT puts in the fields a tree written in one pass never uses: the
#: values ``TTree``'s own constructor sets, so nothing reading finds a tree
#: that looks unlike every other tree.
MAX_ENTRIES = 1_000_000_000_000
AUTO_SAVE = -300_000_000
AUTO_FLUSH = -30_000_000
ESTIMATE = 1_000_000
SCAN_FIELD = 25
OFFSET_LEN = 1000
WEIGHT = 1.0
#: How many basket slots a branch declares room for, as ROOT's does.
MIN_BASKETS = 10
#: The smallest table of entry offsets ROOT shrinks a branch's guess to.
MIN_OFFSET_LEN = 10

#: What ROOT calls a column of each :mod:`array` type code: the leaf class,
#: the letter a branch title spells the type with, how wide one value is, and
#: whether the values are unsigned.
LEAVES: dict[str, tuple[str, str, int, bool]] = {
    "?": ("TLeafO", "O", 1, False),
    "b": ("TLeafB", "B", 1, False),
    "B": ("TLeafB", "b", 1, True),
    "h": ("TLeafS", "S", 2, False),
    "H": ("TLeafS", "s", 2, True),
    "i": ("TLeafI", "I", 4, False),
    "I": ("TLeafI", "i", 4, True),
    "q": ("TLeafL", "L", 8, False),
    "Q": ("TLeafL", "l", 8, True),
    "f": ("TLeafF", "F", 4, False),
    "d": ("TLeafD", "D", 8, False),
}

#: The leaf a column of text is: ``TLeafC``, spelled ``C``, a byte a character.
TEXT_LEAF = ("TLeafC", "C", 1, False)

#: The type a counter is: a signed 32-bit int, which is what ``x[n]`` wants.
COUNTER_CODE = "i"

#: The type a column spelled with a plain Python type asks for. A Python int
#: is as big as it likes, so it gets the widest ROOT has rather than one that
#: would quietly stop fitting somewhere down the file.
PYTHON_TYPES: dict[type, str] = {bool: "?", int: "q", float: "d"}

#: Type codes whose width is the platform's, resolved to codes that are not.
_WIDE = array.array("l").itemsize == 8
PLATFORM = {"l": "q" if _WIDE else "i", "L": "Q" if _WIDE else "I"}

#: What packing gives back for a column: its bytes, and for a column whose
#: entries differ in size, how many of those bytes each entry took.
Packed = tuple[bytes, Optional["np.ndarray[Any, Any]"]]


def _require_name(name: Any, what: str) -> None:
    """A name a branch can carry, and that a reader will not take for syntax."""
    _checked(name, f"{what} name")
    if not name or any(bad in name for bad in "./;[] "):
        raise ValueError(
            f"{name!r} is not a name a {what} can have: a name is not empty, "
            f"and holds none of . / ; [ ] or a space, all of which mean "
            f"something else to anything reading the tree back"
        )


def _typecode(name: str, spec: Any) -> tuple[str, int | str]:
    """What a column was declared as: a type code, and values per entry.

    The count is a number for a column of fixed size, or the name of the
    counter saying how many there are for one that changes - ``None`` in the
    declaration asks for a counter of its own, ``n`` and the column's name.
    """
    length: int | str = 1
    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise ValueError(
                f"the column {name!r} is declared as {len(spec)} things; a pair says "
                f"the type and how many values every entry holds, as in ('f', 3), or "
                f"None for a number that changes from entry to entry, as in ('f', None)"
            )
        spec, length = spec
        length = _length(name, length)
    return _code(name, spec), length


def _length(name: str, length: Any) -> int | str:
    """How many values a column's entries hold, or the counter that says so."""
    if length is None:
        return f"n{name}"
    if isinstance(length, str):
        return length
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError(
            f"the column {name!r} says {length!r} values per entry, which is not a "
            f"count of them; a column holds a fixed number, one or more, or None or "
            f"a counter's name for a number that changes"
        )
    return length


def _code(name: str, spec: Any) -> str:
    """The :mod:`array` code for a column's type, however it was spelled.

    A Python ``bool``, ``int`` or ``float``; an :mod:`array` code such as
    ``'f'``; or anything NumPy calls a type - ``np.float32``, ``'float32'``,
    ``'<i8'``, a dtype - all come to the same letter, which is the one the
    leaf class is picked by.
    """
    if isinstance(spec, type) and spec in PYTHON_TYPES:
        return PYTHON_TYPES[spec]
    try:
        code = np.dtype(spec).char
    except TypeError:
        code = None
    code = PLATFORM.get(code, code) if code is not None else None
    if code in LEAVES:
        return str(code)
    if isinstance(spec, type):
        raise ValueError(
            f"the column {name!r} is declared as {spec.__name__}, and the Python "
            f"types a column can be spelled with are bool, int and float - and str, "
            f"on its own rather than in a pair, for a column of text"
        )
    if not isinstance(spec, (str, np.dtype)):
        raise ValueError(
            f"the column {name!r} is declared as {spec!r}, which is neither a type "
            f"code such as 'f' nor a Python type such as float"
        )
    raise ValueError(
        f"the column {name!r} is of type {str(spec)!r}, and the types a tree here "
        f"holds are {', '.join(LEAVES)} - numbers - and str"
    )


def spec_of(name: str, values: Any) -> Any:
    """The declaration a column of these values needs, read off the values.

    One dimension is a number per entry; more is a fixed-size array per
    entry, as many values as the trailing dimensions hold. Strings are a
    column of text, and rows of different lengths - a :class:`~.tree.Jagged`,
    an Awkward Array of lists, a list of arrays - a column that varies.
    """
    if _is_text(values):
        return str
    if isinstance(values, Jagged) or _is_rows(values):
        return (_rows_of(name, values)[0].dtype, None)
    array_ = np.asarray(values)
    if array_.ndim == 0:
        raise ValueError(
            f"the column {name!r} is a single {array_.dtype} rather than a value per "
            f"entry; give it as an array with one element for every entry"
        )
    if array_.ndim == 1:
        return array_.dtype
    return (array_.dtype, int(np.prod(array_.shape[1:])))


def _is_awkward(values: Any) -> bool:
    """An Awkward Array, told by where its class lives rather than by import."""
    return type(values).__module__.startswith("awkward")


def _is_text(values: Any) -> bool:
    """Is this a column of strings: NumPy's, Awkward's, or a list of them?"""
    if isinstance(values, np.ndarray) and values.dtype.kind == "U":
        return True
    if _is_awkward(values):
        return str(values.type.content) == "string"
    if isinstance(values, np.ndarray) and values.dtype == object:
        values = values.tolist()
    return (
        isinstance(values, (list, tuple))
        and bool(values)
        and all(isinstance(value, str) for value in values)
    )


def _is_rows(values: Any) -> bool:
    """Is this a run of rows of different lengths, rather than one rectangle?

    A list of arrays is, whatever lengths they happen to be; so is a list of
    lists that are not all as long as each other, or all empty, and an
    object array of sequences, which is what a frame's column of lists turns
    into. A list of lists all the same length stays what it always was, a
    fixed-size column.
    """
    if _is_awkward(values):
        return bool(values.ndim > 1)
    if isinstance(values, np.ndarray):
        plain = values.dtype == object and values.ndim == 1
        return plain and _sequences(values, (list, tuple, np.ndarray))
    return isinstance(values, (list, tuple)) and _ragged(values)


def _sequences(rows: Any, kinds: tuple[type, ...]) -> bool:
    """Is there at least one row, and is every row one of these kinds?"""
    return len(rows) > 0 and all(isinstance(row, kinds) for row in rows)


def _ragged(rows: list[Any] | tuple[Any, ...]) -> bool:
    """Arrays, or lists that are not all one length: rows rather than a rectangle."""
    if _sequences(rows, (np.ndarray,)) and all(row.ndim == 1 for row in rows):
        return True
    if not _sequences(rows, (list, tuple)):
        return False
    lengths = {len(row) for row in rows}
    return len(lengths) > 1 or lengths == {0}


def _rows_of(name: str, values: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Rows, however they came, as every value in one array and each row's length."""
    if isinstance(values, Jagged):
        return values.flat, values.lengths()
    if _is_awkward(values):
        return _awkward_rows(name, values)
    if isinstance(values, np.ndarray) and values.dtype != object and values.ndim == 2:
        return values.reshape(-1), np.full(len(values), values.shape[1], np.int64)
    return _listed_rows(name, values)


def _listed_rows(name: str, values: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Rows given one at a time, each anything NumPy can make a flat array of."""
    if not _iterable(values):
        raise ValueError(
            f"{name!r} takes a run of values for every entry, and was given one "
            f"{type(values).__name__} rather than a sequence of runs; give a Jagged, an "
            f"Awkward Array or a list of arrays"
        )
    rows = [_as_row(name, row) for row in values]
    counts = np.asarray([len(row) for row in rows], dtype=np.int64)
    kept = [row for row in rows if len(row)]  # an empty [] has a type of its own
    return (np.concatenate(kept) if kept else np.zeros(0)), counts


def _iterable(values: Any) -> bool:
    """Can this be walked an entry at a time? A lone string is one entry, not many."""
    if isinstance(values, (str, bytes)):
        return False
    try:
        iter(values)
    except TypeError:
        return False
    return True


def _awkward_rows(name: str, values: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """An Awkward Array of lists, taken apart into values and lengths in C."""
    from .library import _module

    ak = _module("ak")  # there, or the values could not have been an Awkward Array
    try:
        content = ak.to_numpy(ak.flatten(values, axis=1), allow_missing=False)
        counts = ak.to_numpy(ak.num(values, axis=1))
    except ValueError as why:
        raise ValueError(
            f"{name!r} takes a list of numbers for every entry, and an Awkward Array "
            f"of {values.type} is not that: {why}"
        ) from None
    return content, counts.astype(np.int64)


def _as_row(name: str, value: Any) -> np.ndarray[Any, Any]:
    """One entry of a varying column: a flat run of values, or a refusal."""
    try:
        given = np.asarray(value)
    except ValueError:
        given = None  # rows of rows of different lengths, which NumPy will not stack
    if given is None or given.ndim != 1:
        shape = "ragged" if given is None else f"shaped {given.shape}"
        raise ValueError(
            f"{name!r} takes a flat run of values for each entry, and this one is "
            f"{shape}; a row is one-dimensional"
        )
    return given


class _Column:
    """One branch being filled: how its values pack, and its bytes so far.

    This is a column of fixed size, where every entry takes the same bytes;
    the columns whose entries do not are the subclasses of :class:`_Variable`.
    """

    __slots__ = (
        "name",
        "typecode",
        "length",
        "classname",
        "letter",
        "itemsize",
        "unsigned",
        "form",
        "size",
        "basket_size",
        "buffer",
        "pending",
        "seeks",
        "sizes",
        "starts",
        "tot_bytes",
        "zip_bytes",
    )

    def __init__(self, name: str, typecode: str, length: int, basket_size: int) -> None:
        self.name = name
        self.typecode = typecode
        self.length = length
        self.classname, self.letter, self.itemsize, self.unsigned = self._leaf_class()
        self.form = f">{length}{typecode}" if length > 1 else f">{typecode}"
        #: How many bytes one entry of this column takes, which is what makes
        #: a basket sliceable without a table of where each entry begins.
        self.size = length * self.itemsize
        self.basket_size = basket_size
        self.buffer = bytearray()
        self.pending = 0
        self.seeks: list[int] = []
        self.sizes: list[int] = []
        #: Where each basket starts, in entries; the last is the total, which
        #: is why this starts out holding a zero rather than nothing.
        self.starts = [0]
        self.tot_bytes = 0
        self.zip_bytes = 0

    def _leaf_class(self) -> tuple[str, str, int, bool]:
        return LEAVES[self.typecode]

    @property
    def typename(self) -> str:
        """``'float32'``, ``'uint8'`` - what a reader will call this column."""
        name = LEAF_TYPES[self.classname][0]
        return f"u{name}" if self.unsigned else name

    @property
    def described(self) -> str:
        """What the tree says the column holds: ``'float32'``, ``'int32[4]'``."""
        return self.typename if self.length == 1 else f"{self.typename}[{self.length}]"

    @property
    def title(self) -> str:
        """The branch title, which is how ROOT spells a column's type."""
        return f"{self.leaf_title}/{self.letter}"

    @property
    def leaf_title(self) -> str:
        """The leaf title: the name, and the size of an entry if it is an array."""
        return self.name if self.length == 1 else f"{self.name}[{self.length}]"

    @property
    def leaf_len(self) -> int:
        """The leaf's ``fLen``: how many values one entry holds, at most."""
        return self.length

    @property
    def is_range(self) -> bool:
        """Whether the leaf keeps the largest value seen, which only a counter does."""
        return False

    @property
    def entry_offset_len(self) -> int:
        """The branch's ``fEntryOffsetLen``: none, since every entry is one size."""
        return 0

    @property
    def nevbuf_size(self) -> int:
        """A basket's ``fNevBufSize``: for a fixed column, one entry's bytes."""
        return self.size

    def limits(self) -> bytes:
        """The leaf's ``fMinimum`` and ``fMaximum``, both left at zero."""
        return bytes(2 * self.itemsize)

    def table(self, keylen: int) -> bytes:
        """What a basket carries after its entries: nothing, for a fixed column."""
        return b""

    def pack(self, value: Any) -> bytes:
        """One entry's bytes, or a ``ValueError`` saying what was wrong with it."""
        if self.typecode == "B" and isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)  # bytes are exactly what an unsigned byte column holds
            if len(raw) != self.length:
                raise ValueError(
                    f"{self.name!r} takes {self.length} bytes per entry, and this one has "
                    f"{len(raw)}"
                )
            return raw
        if self.length == 1:
            values: Any = (value,)
        else:
            try:
                values = list(value)
            except TypeError:
                raise ValueError(
                    f"{self.name!r} takes {self.length} values per entry, and {value!r} is "
                    f"not a sequence of them"
                ) from None
            if len(values) != self.length:
                raise ValueError(
                    f"{self.name!r} takes {self.length} values per entry, and this one has "
                    f"{len(values)}"
                )
        try:
            return struct.pack(self.form, *values)
        except (struct.error, TypeError) as exc:
            raise ValueError(
                f"{self.name!r} holds {self.typename} values, and this entry is not one "
                f"of those: {exc}"
            ) from None

    def pack_one(self, value: Any) -> Packed:
        """One entry, as :meth:`pack_many` gives many."""
        return self.pack(value), None

    def pack_many(self, values: Any) -> Packed:
        """A whole column of entries, packed at once."""
        return _pack_many(self, values), None

    def keep(self, raw: bytes, sizes: np.ndarray[Any, Any] | None) -> None:
        """Take in entries already packed, which nothing can refuse any more."""
        self.note(raw, sizes)
        self.buffer += raw
        self.pending += len(raw) // self.size if sizes is None else len(sizes)

    def note(self, raw: bytes, sizes: np.ndarray[Any, Any] | None) -> None:
        """Remember what the leaf will say about these entries; a plain one says nothing."""

    def emptied(self) -> None:
        """Start the next basket, the last one having gone out."""
        self.buffer = bytearray()
        self.pending = 0


class _Counter(_Column):
    """The branch saying how many values each entry of a varying column holds.

    Nobody fills it: every entry's count is the length of the rows it counts,
    which is exactly how ROOT's ``x[n]`` reads it back. Its leaf keeps the
    largest count it saw, as ROOT's does, which is what lets ROOT size the
    room it reads a row into.
    """

    __slots__ = ("maximum", "users")

    def __init__(self, name: str, basket_size: int) -> None:
        super().__init__(name, COUNTER_CODE, 1, basket_size)
        self.maximum = 0
        #: The columns this one counts, which must agree entry by entry.
        self.users: list[_Rows] = []

    @property
    def is_range(self) -> bool:
        return True

    def limits(self) -> bytes:
        return struct.pack(">ii", 0, self.maximum)

    def note(self, raw: bytes, sizes: np.ndarray[Any, Any] | None) -> None:
        self.maximum = max(self.maximum, int(np.frombuffer(raw, ">i4").max()))


def _room(size: int, entries: int) -> int:
    """How big ROOT's table of entry offsets had grown by the time a basket filled.

    A basket starts with room for the branch's guess and doubles it whenever
    an entry would not fit, so it always ends with more room than entries.
    """
    while size <= entries:
        size = max(MIN_OFFSET_LEN, 2 * size)
    return size


def _adapted(size: int, entries: int) -> int:
    """The branch's next guess at a basket's entry count, after one has gone out.

    ROOT shrinks the guess when a basket held far fewer entries than it had
    room for, and grows it when it held more, so the next basket is built
    about the right size; ``fEntryOffsetLen`` is that guess, written down.
    """
    if size > MIN_OFFSET_LEN and 4 * entries < size:
        return MIN_OFFSET_LEN if entries < 3 else 4 * entries
    if entries > size:
        return 2 * entries
    return size


class _Variable(_Column):
    """A column whose entries differ in size, so its baskets say where each begins."""

    __slots__ = ("offsets", "offset_len")

    def __init__(self, name: str, typecode: str, basket_size: int) -> None:
        super().__init__(name, typecode, 1, basket_size)
        #: Where each entry gathered so far begins, in the bytes gathered.
        self.offsets: list[int] = []
        self.offset_len = OFFSET_LEN

    @property
    def entry_offset_len(self) -> int:
        return self.offset_len

    @property
    def nevbuf_size(self) -> int:
        return _room(self.offset_len, self.pending)

    def table(self, keylen: int) -> bytes:
        """Where each entry begins, counted from the key, as ROOT writes it.

        The count is one more than the entries and the last slot is left at
        zero, because ROOT writes the whole of the array it kept the places
        in; a reader takes the end of the last entry from ``fLast`` instead.
        """
        places = np.append(np.asarray(self.offsets, dtype=np.int64) + keylen, 0)
        return struct.pack(">i", len(places)) + bytes(places.astype(">i4").tobytes())

    def pack_one(self, value: Any) -> Packed:
        raw = self.pack(value)
        return raw, np.asarray([len(raw)], dtype=np.int64)

    def keep(self, raw: bytes, sizes: np.ndarray[Any, Any] | None) -> None:
        assert sizes is not None
        self.offsets.extend((np.cumsum(sizes) - sizes + len(self.buffer)).tolist())
        super().keep(raw, sizes)

    def emptied(self) -> None:
        self.offset_len = _adapted(self.offset_len, self.pending)
        self.offsets = []
        super().emptied()


class _Rows(_Variable):
    """A run of numbers per entry, as long as its counter says: ROOT's ``x[n]``."""

    __slots__ = ("counter",)

    def __init__(self, name: str, typecode: str, basket_size: int, counter: _Counter) -> None:
        super().__init__(name, typecode, basket_size)
        self.counter = counter
        counter.users.append(self)

    @property
    def described(self) -> str:
        return f"{self.typename}[{self.counter.name}]"

    @property
    def leaf_title(self) -> str:
        return f"{self.name}[{self.counter.name}]"

    def pack(self, value: Any) -> bytes:
        return _cast(self, _as_row(self.name, value))

    def pack_many(self, values: Any) -> Packed:
        content, counts = _rows_of(self.name, values)
        if content.ndim != 1:
            raise ValueError(
                f"{self.name!r} takes a flat run of numbers for each entry, and these "
                f"rows hold values shaped {content.shape[1:]}"
            )
        return _cast(self, content), counts * self.itemsize


class _Text(_Variable):
    """A string per entry: ROOT's ``TLeafC``, a length and then the bytes."""

    __slots__ = ("widest",)

    def __init__(self, name: str, basket_size: int) -> None:
        super().__init__(name, "B", basket_size)
        #: The longest string kept so far, in bytes; none yet is -1.
        self.widest = -1

    def _leaf_class(self) -> tuple[str, str, int, bool]:
        return TEXT_LEAF

    @property
    def leaf_len(self) -> int:
        """One more than the longest string, as ROOT counts room for its NUL."""
        return max(1, self.widest + 1)

    def limits(self) -> bytes:
        return struct.pack(">ii", 0, self.widest + 1)

    def pack(self, value: Any) -> bytes:
        if not isinstance(value, str):
            raise ValueError(
                f"{self.name!r} holds text, and this entry is of type "
                f"{type(value).__name__}, not str"
            )
        if "\x00" in value:
            raise ValueError(
                f"{self.name!r} holds text, and this entry has a NUL in it, which is where "
                f"ROOT takes a string to end; whatever came after it would be lost"
            )
        buf = WBuffer()
        buf.string(value)
        return bytes(buf.data)

    def pack_many(self, values: Any) -> Packed:
        if not _iterable(values):
            raise ValueError(
                f"{self.name!r} takes a string for every entry, and was given one "
                f"{type(values).__name__} rather than a sequence of them"
            )
        pieces = [self.pack(value) for value in values]
        return b"".join(pieces), np.asarray([len(piece) for piece in pieces], dtype=np.int64)

    def note(self, raw: bytes, sizes: np.ndarray[Any, Any] | None) -> None:
        assert sizes is not None
        # A string under 255 bytes has one byte in front of it, a longer one five.
        lengths = np.where(sizes <= 255, sizes - 1, sizes - 5)
        self.widest = max(self.widest, int(lengths.max()))


def _pack_many(column: _Column, values: Any) -> bytes:
    """A whole column of entries at once, as the bytes a run of baskets holds.

    The same promise :meth:`_Column.pack` makes for one entry, made for all
    of them in one pass in C: the shape is checked, a float is not quietly
    truncated into an integer column, and an integer that does not fit the
    column is refused rather than wrapped round.
    """
    given = np.asarray(values)
    want = (len(given),) if column.length == 1 else (len(given), column.length)
    if given.ndim == 0 or int(np.prod(given.shape[1:])) != column.length:
        raise ValueError(
            f"{column.name!r} takes {column.length} values per entry, and these are "
            f"shaped {given.shape}; give an array of shape {('n', *want[1:])}"
        )
    return _cast(column, given.reshape(want))


def _cast(column: _Column, given: np.ndarray[Any, Any]) -> bytes:
    """Values as the column's big-endian bytes, refused if they would change."""
    target = np.dtype(column.typecode)
    if given.size and not np.can_cast(given.dtype, target, "same_kind"):
        raise ValueError(
            f"{column.name!r} holds {column.typename} values, and these are {given.dtype}, "
            f"which would not go into it without losing what they are"
        )
    _require_fits(column, given, target)
    return bytes(given.astype(target.newbyteorder(">")).tobytes())


def _require_fits(column: _Column, given: np.ndarray[Any, Any], target: np.dtype[Any]) -> None:
    """Refuse integers that the column's type is too narrow to hold."""
    if target.kind not in "iu" or given.dtype.kind not in "iu" or not given.size:
        return
    limits = np.iinfo(target)
    low, high = int(given.min()), int(given.max())
    if low < limits.min or high > limits.max:
        raise ValueError(
            f"{column.name!r} holds {column.typename} values, from {limits.min} to "
            f"{limits.max}, and these run from {low} to {high}"
        )


def _entries(column: _Column, packed: Packed) -> int:
    """How many entries a column's packed bytes hold."""
    raw, sizes = packed
    return len(raw) // column.size if sizes is None else len(sizes)


def _objarray(buf: WBuffer, count: int) -> int:
    """Open a ``TObjArray`` of ``count`` things, for the caller to write and end."""
    index = buf.start(OBJARRAY_VERSION)
    buf.tobject()
    buf.string("")
    buf.i32(count)
    buf.i32(0)  # the lower bound: every array here counts from zero
    return index


def _attributes(buf: WBuffer) -> None:
    """The line, fill and marker bases a tree carries, at ROOT's defaults."""
    index = buf.start(ATTRIBUTE_VERSION)
    buf.i16(602)  # fLineColor
    buf.i16(1)  # fLineStyle
    buf.i16(1)  # fLineWidth
    buf.end(index)
    index = buf.start(ATTRIBUTE_VERSION)
    buf.i16(0)  # fFillColor
    buf.i16(1001)  # fFillStyle
    buf.end(index)
    index = buf.start(ATTRIBUTE_VERSION)
    buf.i16(1)  # fMarkerColor
    buf.i16(1)  # fMarkerStyle
    buf.raw(struct.pack(">f", 1.0))  # fMarkerSize
    buf.end(index)


def _leaf(buf: WBuffer, column: _Column, count: int) -> int:
    """One ``TLeaf``; where it landed comes back, for the tree to point at.

    ``count`` is the reference to the counter's leaf, already written in the
    branch before this one, for a column whose length it gives - or zero, the
    null pointer, for a column whose size is its own.
    """
    at = buf.tag(column.classname)
    outer = buf.start(SUBLEAF_VERSION)
    inner = buf.start(LEAF_VERSION)
    buf.named(column.name, column.leaf_title)
    buf.i32(column.leaf_len)
    buf.i32(column.itemsize)
    buf.i32(0)  # fOffset: one leaf per branch, so an entry starts where it starts
    buf.u8(int(column.is_range))
    buf.u8(int(column.unsigned))
    buf.u32(count)  # fLeafCount
    buf.end(inner)
    buf.raw(column.limits())
    buf.end(outer)
    buf.end(at)
    return at


def _table(buf: WBuffer, values: list[int], width: int, code: str) -> None:
    """One of a branch's basket tables: a marker, then a value per slot."""
    buf.u8(1)  # the marker saying the table is here rather than absent
    buf.raw(struct.pack(f">{width}{code}", *values, *([0] * (width - len(values)))))


def _branch(buf: WBuffer, column: _Column, entries: int, compress: int, count: int) -> int:
    """One ``TBranch``, leaf and all; where its leaf landed comes back."""
    at = buf.tag("TBranch")
    index = buf.start(BRANCH_VERSION)
    buf.named(column.name, column.title)
    fill = buf.start(ATTRIBUTE_VERSION)
    buf.i16(0)  # fFillColor
    buf.i16(0)  # fFillStyle
    buf.end(fill)
    written = len(column.seeks)
    width = max(MIN_BASKETS, written + 1)
    buf.i32(compress)
    buf.i32(column.basket_size)
    buf.i32(column.entry_offset_len)
    buf.i32(written)
    buf.i64(entries)  # fEntryNumber
    buf.i32(0)  # fOffset
    buf.i32(width)  # fMaxBaskets
    buf.i32(0)  # fSplitLevel: nothing here is split, a column is a column
    buf.i64(entries)
    buf.i64(0)  # fFirstEntry
    buf.i64(column.tot_bytes)
    buf.i64(column.zip_bytes)
    empty = _objarray(buf, 0)  # fBranches: a column has nothing under it
    buf.end(empty)
    leaves = _objarray(buf, 1)
    place = _leaf(buf, column, count)
    buf.end(leaves)
    baskets = _objarray(buf, written + 1)  # fBaskets: all on file, none in here
    buf.raw(bytes(4 * (written + 1)))
    buf.end(baskets)
    _table(buf, column.sizes, width, "i")  # fBasketBytes
    _table(buf, column.starts, width, "q")  # fBasketEntry
    _table(buf, column.seeks, width, "q")  # fBasketSeek
    buf.string("")  # fFileName: the baskets are in this file, where they were put
    buf.end(index)
    buf.end(at)
    return place


def _mismatch(
    row: Mapping[str, Any], columns: Mapping[str, _Column], counters: Mapping[str, _Counter]
) -> list[str]:
    """What is wrong with the names an entry came with, if anything is."""
    missing = [name for name in columns if name not in row]
    return _row_trouble(missing, *_strangers(row, columns, counters))


def _strangers(
    row: Mapping[str, Any], columns: Mapping[str, _Column], counters: Mapping[str, _Counter]
) -> tuple[list[str], list[str]]:
    """The names an entry has that are not columns: unknown ones, then counters."""
    given = [str(name) for name in row if name not in columns]
    return [name for name in given if name not in counters], [
        name for name in given if name in counters
    ]


def _row_trouble(missing: list[str], unknown: list[str], counters: list[str]) -> list[str]:
    trouble = []
    if missing:
        trouble.append(f"nothing for {', '.join(missing)}")
    if unknown:
        trouble.append(f"{', '.join(unknown)}, which is not a column")
    if counters:
        trouble.append(f"{', '.join(counters)}, which is a counter filled from the rows it counts")
    return trouble


def _declared(name: str, spec: Any, basket_size: int, counters: dict[str, _Counter]) -> _Column:
    """The column a declaration asks for, sharing a counter already made if named."""
    if spec is str:
        return _Text(name, basket_size)
    typecode, length = _typecode(name, spec)
    if isinstance(length, int):
        return _Column(name, typecode, length, basket_size)
    _require_name(length, "counter")
    counter = counters.setdefault(length, _Counter(length, basket_size))
    return _Rows(name, typecode, basket_size, counter)


class WritableTree:
    """A tree being written: named columns, then one entry at a time.

        >>> with xrdroot.create("out.root") as f:            # doctest: +SKIP
        ...     tree = f.tree("events", {"energy": float, "hits": ("i", 4),
        ...                              "jets": ("f", None), "label": str})
        ...     tree.fill(energy=12.5, hits=[3, 1, 4, 1], jets=[40.5, 22.0], label="dijet")

    A column is declared as a type - a Python ``bool``, ``int`` or ``float``,
    or an :mod:`array` type code such as ``'f'`` for a narrower one - or as a
    pair of a type and how many values each entry holds. ``None`` for the
    count makes a column whose rows change length, counted by a branch of its
    own called ``n`` and the column's name; a name instead shares that counter
    between columns that always hold as many values as each other. ``str``
    is a column of text. Every entry needs a value for every column - but
    never for a counter, which is filled from the lengths of what it counts -
    because a tree whose columns disagree about how many entries they have is
    a tree nothing can read.

    Entries are written out a basket at a time as they gather, so the tree
    can be far larger than memory; nothing has to be kept but the file the
    tree is going into, and the tree's own record goes in when that closes.
    """

    def __init__(
        self,
        file: WritableFile,
        name: str,
        title: str,
        columns: Mapping[str, Any],
        basket_size: int,
        cycle: int,
    ) -> None:
        if not isinstance(columns, Mapping):
            raise TypeError(
                f"the columns are a {type(columns).__name__}; a tree is declared with a "
                f"mapping of column name to type, as in {{'energy': float}}"
            )
        if not columns:
            raise ValueError(
                "a tree with no columns holds nothing that could be read back; "
                "declare what each entry is made of"
            )
        if not isinstance(basket_size, int) or basket_size < 1:
            raise ValueError(f"the basket size is {basket_size!r}, which is not a size in bytes")
        self._file = file
        self._cycle = cycle
        #: What the tree is called, and what it says it is.
        self.name = name
        self.title = title
        self._entries = 0
        self._columns: dict[str, _Column] = {}
        self._counters: dict[str, _Counter] = {}
        for column, spec in columns.items():
            _require_name(column, "column")
            self._columns[column] = _declared(column, spec, basket_size, self._counters)
        self._branches = self._in_order()

    def _in_order(self) -> list[_Column]:
        """Every branch as it goes into the tree: each counter before what it counts.

        ROOT finds a leaf's counter by looking back through the leaves it has
        read, so the counter has to come first; it goes in just before the
        first column that uses it.
        """
        clash = [name for name in self._counters if name in self._columns]
        if clash:
            raise ValueError(
                f"{', '.join(clash)} is declared as a column and named as a counter; a "
                f"counter is filled from the lengths of the rows it counts, so leave it "
                f"out of the columns, or give the rows a counter of another name"
            )
        order: dict[str, _Column] = {}
        for column in self._columns.values():
            if isinstance(column, _Rows):
                order.setdefault(column.counter.name, column.counter)
            order[column.name] = column
        return list(order.values())

    def __repr__(self) -> str:
        return (
            f"<WritableTree {self.name!r} with {len(self._branches)} columns "
            f"and {self._entries} entries so far>"
        )

    def __len__(self) -> int:
        """How many entries have been filled, as ``len`` of a tree read back."""
        return self._entries

    @property
    def num_entries(self) -> int:
        """How many entries have been filled, under the reader's name for it."""
        return self._entries

    @property
    def columns(self) -> dict[str, str]:
        """What each column holds: ``{'energy': 'float64', 'hits': 'int32[4]'}``.

        The counters are here too, where they will be in the file: a column
        ``jets`` of ``('f', None)`` is ``'float32[njets]'``, after ``njets``.
        """
        return {column.name: column.described for column in self._branches}

    @property
    def classes(self) -> tuple[str, ...]:
        """The classes this tree will be made of, for the file to describe."""
        return ("TTree", "TBranch", *dict.fromkeys(c.classname for c in self._branches))

    def fill(self, **values: Any) -> None:
        """Add one entry, with a value for every column.

            >>> tree.fill(energy=12.5, hits=[3, 1, 4, 1])     # doctest: +SKIP

        An entry that does not fit its columns is refused whole: nothing is
        written for any column, so the tree is exactly as it was.
        """
        self._row(values)

    def extend(self, rows: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> None:
        """Add many entries: a column of values for each column, or row after row.

            >>> tree.extend({"energy": energies, "hits": hits})   # doctest: +SKIP
            >>> tree.extend([{"energy": 1.0}, {"energy": 2.0}])   # doctest: +SKIP

        A mapping of column name to an array - one element, or one row of a
        fixed-size column, per entry - is the fast way, packed in C a column
        at a time; every column needs the same number of entries, and they go
        into baskets exactly as filling them one by one would have put them.
        A varying column takes a :class:`~.tree.Jagged`, an Awkward Array or a
        list of arrays, and a column of text a list of strings. Anything else
        is taken as entries, each a mapping of column to value. Either way a
        batch that does not fit is refused whole.
        """
        if isinstance(rows, Mapping):
            self._columns_at_once(rows)
            return
        for row in rows:
            self._row(row)

    def _columns_at_once(self, arrays: Mapping[str, Any]) -> None:
        self._require_open()
        self._require_columns(arrays)
        packed = {name: col.pack_many(arrays[name]) for name, col in self._columns.items()}
        counts = {name: _entries(self._columns[name], pair) for name, pair in packed.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"the columns given to {self.name!r} hold different numbers of entries "
                f"({', '.join(f'{name}: {count}' for name, count in counts.items())}); a "
                f"tree whose columns disagree about that is a tree nothing can read"
            )
        self._take(packed, next(iter(counts.values())))

    def _row(self, row: Mapping[str, Any]) -> None:
        """One entry: packed in full before any of it is kept."""
        self._require_open()
        self._require_columns(row)
        self._take({name: col.pack_one(row[name]) for name, col in self._columns.items()}, 1)

    def _take(self, packed: dict[str, Packed], entries: int) -> None:
        """Count the rows, then keep what every column was given - or refuse it all."""
        for counter in self._counters.values():
            lengths = self._lengths(counter, packed)
            packed[counter.name] = (lengths.astype(">i4").tobytes(), None)
        for column in self._branches:
            self._feed(column, packed[column.name])
        self._entries += entries

    def _lengths(self, counter: _Counter, packed: dict[str, Packed]) -> np.ndarray[Any, Any]:
        """What a counter says for each entry: the one length every column it counts has."""
        first, *others = counter.users
        lengths = _counts(first, packed)
        for other in others:
            theirs = _counts(other, packed)
            if not np.array_equal(lengths, theirs):
                at = int(np.flatnonzero(lengths != theirs)[0])
                raise ValueError(
                    f"{first.name!r} and {other.name!r} share the counter {counter.name!r}, "
                    f"and entry {self._entries + at} has {lengths[at]} values in one and "
                    f"{theirs[at]} in the other; the columns one counter counts hold as "
                    f"many values as each other in every entry"
                )
        return lengths

    def _feed(self, column: _Column, packed: Packed) -> None:
        raw, sizes = packed
        if sizes is None:
            self._feed_fixed(column, raw)
        else:
            self._feed_rows(column, raw, sizes)

    def _feed_fixed(self, column: _Column, raw: bytes) -> None:
        """Pour many entries into one column, a basket's worth at a time.

        A basket goes out once it holds ``basket_size`` bytes, which for a
        column of fixed-size entries is a fixed number of them - the same
        number a row at a time would have reached - so the file is the same
        whichever way the entries were given.
        """
        per = -(-column.basket_size // column.size)
        at = 0
        while at < len(raw):
            take = (per - column.pending) * column.size
            chunk = raw[at : at + take]
            column.keep(chunk, None)
            at += len(chunk)
            if len(column.buffer) >= column.basket_size:
                self._flush(column)

    def _feed_rows(self, column: _Column, raw: bytes, sizes: np.ndarray[Any, Any]) -> None:
        """Pour entries of different sizes into one column, a basket at a time.

        The same rule as for fixed entries - a basket goes out with the entry
        that takes it to ``basket_size`` bytes or past - found for a whole
        batch at once by searching the running total of the entries' sizes.
        """
        ends = np.cumsum(sizes)
        done = 0
        while done < len(sizes):
            base = int(ends[done - 1]) if done else 0
            room = column.basket_size - len(column.buffer)
            stop = min(int(np.searchsorted(ends, base + room)) + 1, len(sizes))
            column.keep(raw[base : int(ends[stop - 1])], sizes[done:stop])
            done = stop
            if len(column.buffer) >= column.basket_size:
                self._flush(column)

    def _require_open(self) -> None:
        if self._file.closed:
            raise ValueError(
                f"the file this tree is in is closed; {self.name!r} holds the "
                f"{self._entries} entries it was given"
            )

    def _require_columns(self, row: Mapping[str, Any]) -> None:
        trouble = _mismatch(row, self._columns, self._counters)
        if not trouble:
            return
        raise ValueError(
            f"this entry of {self.name!r} has {' and '.join(trouble)}; its columns "
            f"are {', '.join(self._columns)}, and every entry needs all of them"
        )

    def _flush(self, column: _Column) -> None:
        """Write what one column has gathered as a basket of its own.

        A column whose entries differ in size carries a table of where each
        begins behind them, which ``fLast`` - where the entries end - is how a
        reader finds.
        """
        if not column.pending:
            return
        payload = bytes(column.buffer)
        keylen = _keylen("TBasket", column.name, self.name, extra=19)
        table = column.table(keylen)
        extra = struct.pack(
            ">hiiiiB",
            BASKET_VERSION,
            column.basket_size,
            column.nevbuf_size,
            column.pending,
            keylen + len(payload),  # fLast: where the entries end
            0,  # the entries are in the record behind this key, not in the key
        )
        seek, nbytes = self._file._put(
            "TBasket", column.name, self.name, payload + table, 0, listed=False, extra=extra
        )
        column.seeks.append(seek)
        column.sizes.append(nbytes)
        column.starts.append(column.starts[-1] + column.pending)
        column.tot_bytes += keylen + len(payload) + len(table)
        column.zip_bytes += nbytes
        column.emptied()

    def _finish(self) -> None:
        """Flush what is left, then write the record that ties it all together."""
        for column in self._branches:
            self._flush(column)
        keylen = _keylen("TTree", self.name, self.title)
        payload = self._payload(keylen)
        self._file._put("TTree", self.name, self.title, payload, self._cycle, listed=True)

    def _payload(self, origin: int) -> bytes:
        """The tree's own record: its fields, its branches, then its leaves.

        ``origin`` is how long the key in front of this will be, because the
        leaves are written once inside their branches and referred to after
        by where they landed - counted, as every place in a record is, from
        the start of the key rather than the start of the record. A varying
        column's leaf points at its counter's the same way.
        """
        buf = WBuffer()
        index = buf.start(TREE_VERSION)
        buf.named(self.name, self.title)
        _attributes(buf)
        columns = self._branches
        buf.i64(self._entries)
        buf.i64(sum(column.tot_bytes for column in columns))
        buf.i64(sum(column.zip_bytes for column in columns))
        buf.i64(0)  # fSavedBytes: nothing was saved part way, this went in one go
        buf.i64(0)  # fFlushedBytes
        buf.raw(struct.pack(">d", WEIGHT))
        buf.i32(0)  # fTimerInterval
        buf.i32(SCAN_FIELD)
        buf.i32(0)  # fUpdate
        buf.i32(OFFSET_LEN)  # fDefaultEntryOffsetLen
        buf.i32(0)  # fNClusterRange: no cluster boundaries were declared
        buf.i64(MAX_ENTRIES)
        buf.i64(MAX_ENTRIES)  # fMaxEntryLoop
        buf.i64(0)  # fMaxVirtualSize
        buf.i64(AUTO_SAVE)
        buf.i64(AUTO_FLUSH)
        buf.i64(ESTIMATE)
        buf.u8(0)  # fClusterRangeEnd, of which there are none
        buf.u8(0)  # fClusterSize, likewise
        branches = _objarray(buf, len(columns))
        places = self._write_branches(buf, origin)
        buf.end(branches)
        leaves = _objarray(buf, len(columns))
        for place in places.values():
            buf.u32(origin + place + MAP_OFFSET)  # the leaf itself is in its branch
        buf.end(leaves)
        buf.u32(0)  # fAliases: none
        buf.i32(0)  # fIndexValues: an empty TArrayD
        buf.i32(0)  # fIndex: an empty TArrayI
        buf.u32(0)  # fTreeIndex: none
        buf.u32(0)  # fFriends: none
        buf.u32(0)  # fUserInfo: none
        buf.u32(0)  # fBranchRef: none
        buf.end(index)
        return bytes(buf.data)

    def _write_branches(self, buf: WBuffer, origin: int) -> dict[str, int]:
        """Every branch in order; where each one's leaf landed comes back."""
        compress = self._file._codes
        places: dict[str, int] = {}
        for column in self._branches:
            count = 0
            if isinstance(column, _Rows):
                count = origin + places[column.counter.name] + MAP_OFFSET
            places[column.name] = _branch(buf, column, self._entries, compress, count)
        return places


def _counts(column: _Rows, packed: dict[str, Packed]) -> np.ndarray[Any, Any]:
    """How many values each entry of a varying column holds, from its packed sizes."""
    sizes = packed[column.name][1]
    assert sizes is not None
    return sizes // column.itemsize
