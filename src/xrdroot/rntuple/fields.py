"""How each kind of field turns the columns under it into values.

A field is read a range of its own *elements* at a time, counted from the
start of one cluster. For a top-level field an element is an entry; for the
item field of a collection it is one item, and the collection's offset
column says which items belong to which entry; for the item of a fixed-size
array there are that many elements to every one of the array's. Each class
here is one kind of field, and each reads what it needs from the columns and
the fields under it, so a ``vector<vector<float>>`` is a collection over a
collection over a number, and needs nothing written for it on its own.

What comes back is shaped the way the rest of this library shapes things. A
number per element is a NumPy array, and a fixed-size array of numbers one of
more dimensions. A collection of numbers is :class:`~xrdroot.tree.Jagged`,
the flat values and where each row starts. Anything else - strings, nested
collections, records, variants - is a list with one Python value per element:
``str``, ``list``, a ``dict`` of a record's members, a ``tuple`` for a
``std::pair`` or ``std::tuple``, a ``dict`` for a ``std::map``, the value or
``None`` for a ``std::optional``. That is the same split a ``TTree`` makes
between the columns NumPy holds and the ones it cannot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..tree import Jagged
from .columns import ENCODINGS
from .schema import COLLECTION, RECORD, REPETITIVE, STREAMER, VARIANT, Field, Schema

if TYPE_CHECKING:
    from .reader import Store
    from .schema import Cluster

__all__ = ["Node", "Refused", "build", "rows", "join"]

#: The NumPy type each fundamental C++ type reads into, whatever column it is
#: stored in: a ``float`` written as ``Real16`` still comes back ``float32``.
FUNDAMENTALS = {
    "bool": "bool",
    "char": "int8",
    "std::byte": "uint8",
    "std::int8_t": "int8",
    "std::uint8_t": "uint8",
    "std::int16_t": "int16",
    "std::uint16_t": "uint16",
    "std::int32_t": "int32",
    "std::uint32_t": "uint32",
    "std::int64_t": "int64",
    "std::uint64_t": "uint64",
    "float": "float32",
    "double": "float64",
}
#: Collections that are keyed, and so read as a ``dict`` per entry.
MAPS = ("std::map<", "std::unordered_map<")
#: Collections of zero or one, and so read as the value or ``None``.
OPTIONALS = ("std::optional<", "std::unique_ptr<")
#: Records whose members are positions rather than names.
TUPLES = ("std::pair<", "std::tuple<")
CARDINALITY = "ROOT::RNTupleCardinality<"


def rows(value: Any) -> list[Any]:
    """One Python value per element, whatever shape the elements came back in."""
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return list(value.tolist())
    return list(value)


def join(pieces: list[Any], empty: Any) -> Any:
    """What several clusters gave for one field, end to end."""
    if not pieces:
        return empty
    if isinstance(pieces[0], np.ndarray):
        return np.concatenate(pieces)
    if isinstance(pieces[0], Jagged):
        lengths = np.concatenate([piece.lengths() for piece in pieces])
        offsets = np.zeros(len(lengths) + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return Jagged(np.concatenate([piece.flat for piece in pieces]), offsets)
    return [item for piece in pieces for item in piece]


class Node:
    """One field, ready to read a range of its elements out of a cluster."""

    __slots__ = ("field", "reps", "multiplicity")

    def __init__(self, field: Field, multiplicity: int) -> None:
        self.field = field
        #: The columns attached, by representation; which is live is per cluster.
        self.reps = field.columns
        #: Elements per entry, when that is fixed - which is what places the
        #: zeros in front of a column that was only added partway through.
        self.multiplicity = multiplicity

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.field.name!r} of {self.typename}>"

    @property
    def typename(self) -> str:
        raise NotImplementedError

    @property
    def jagged(self) -> bool:
        """Whether this reads as :class:`~xrdroot.tree.Jagged`."""
        return False

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        raise NotImplementedError

    def column(self, store: Store, cluster: Cluster | None, slot: int, lo: int, hi: int) -> Any:
        """Elements ``lo`` to ``hi`` of this field's ``slot``-th column."""
        base = cluster.first_entry * self.multiplicity if cluster is not None else 0
        return store.elements(cluster, self.reps, slot, lo, hi, base)

    def bounds(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        """Where the items of elements ``lo`` to ``hi`` start, and each one's offsets.

        The offset column holds where each element's items *end*, counted
        from the start of the cluster, so the start of the first is the end
        of the one before it - or zero, at the front of a cluster.
        """
        if hi <= lo:
            return 0, np.zeros(1, np.int64)
        ends = self.column(store, cluster, 0, max(lo - 1, 0), hi).astype(np.int64)
        begin = int(ends[0]) if lo else 0
        offsets = np.concatenate([[begin], ends[1:] if lo else ends]) - begin
        return begin, offsets


class Refused(Node):
    """A field this reader will not decode, and the sentence saying why."""

    __slots__ = ("reason",)

    def __init__(self, field: Field, reason: str) -> None:
        super().__init__(field, 1)
        self.reason = reason

    @property
    def typename(self) -> str:
        return f"? ({self.field.type or 'untyped'})"

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        raise UnsupportedFeatureError(f"the field {self.field.name!r} holds {self.reason}")


class Leaf(Node):
    """A number per element, from one column."""

    __slots__ = ("dtype",)

    def __init__(self, field: Field, multiplicity: int, dtype: np.dtype[Any]) -> None:
        super().__init__(field, multiplicity)
        self.dtype = dtype

    @property
    def typename(self) -> str:
        return str(self.dtype)

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        return self.column(store, cluster, 0, lo, hi).astype(self.dtype, copy=False)


class String(Node):
    """A ``std::string``: where each one ends, and a column of their bytes."""

    __slots__ = ()

    @property
    def typename(self) -> str:
        return "str"

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        begin, offsets = self.bounds(store, cluster, lo, hi)
        raw = bytes(self.column(store, cluster, 1, begin, begin + int(offsets[-1])))
        return [
            raw[start:stop].decode("utf-8", "surrogateescape")
            for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist())
        ]


class Cardinality(Node):
    """How many items a collection holds, from that collection's offsets."""

    __slots__ = ("dtype",)

    def __init__(self, field: Field, multiplicity: int) -> None:
        super().__init__(field, multiplicity)
        self.dtype = np.dtype("uint32" if "uint32" in field.type else "uint64")

    @property
    def typename(self) -> str:
        return str(self.dtype)

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        _begin, offsets = self.bounds(store, cluster, lo, hi)
        return np.diff(offsets).astype(self.dtype)


class Collection(Node):
    """A ``std::vector``, ``RVec``, set or anything else of items in a row."""

    __slots__ = ("item",)

    def __init__(self, field: Field, multiplicity: int, item: Node) -> None:
        super().__init__(field, multiplicity)
        self.item = item

    @property
    def typename(self) -> str:
        return f"list[{self.item.typename}]"

    @property
    def jagged(self) -> bool:
        return isinstance(self.item, Leaf)

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        begin, offsets = self.bounds(store, cluster, lo, hi)
        items = self.item.read(store, cluster, begin, begin + int(offsets[-1]))
        if self.jagged:
            return Jagged(items, offsets)
        items = rows(items)
        return [self.shape(items[a:b]) for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist())]

    def shape(self, items: list[Any]) -> Any:
        return items


class Map(Collection):
    """A ``std::map`` or ``std::unordered_map``: a ``dict`` per element."""

    __slots__ = ()

    @property
    def typename(self) -> str:
        return "dict[" + self.item.typename[len("tuple[") :]

    def shape(self, items: list[Any]) -> Any:
        return dict(items)


class Optional(Collection):
    """A ``std::optional`` or ``std::unique_ptr``: the value, or ``None``."""

    __slots__ = ()

    @property
    def typename(self) -> str:
        return f"{self.item.typename} | None"

    @property
    def jagged(self) -> bool:
        return False

    def shape(self, items: list[Any]) -> Any:
        return items[0] if items else None


class Array(Node):
    """A ``std::array`` - or a C array - of the same number of items every time."""

    __slots__ = ("item", "size")

    def __init__(self, field: Field, multiplicity: int, item: Node) -> None:
        super().__init__(field, multiplicity)
        self.item = item
        self.size = field.array_size

    @property
    def typename(self) -> str:
        return f"{self.item.typename}[{self.size}]"

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        items = self.item.read(store, cluster, lo * self.size, hi * self.size)
        if isinstance(items, np.ndarray):
            return items.reshape((hi - lo, self.size, *items.shape[1:]))
        items = rows(items)
        return [items[at : at + self.size] for at in range(0, len(items), self.size)]


class Bitset(Node):
    """A ``std::bitset``: its bits in a column of its own, lowest first."""

    __slots__ = ("size",)

    def __init__(self, field: Field, multiplicity: int) -> None:
        super().__init__(field, multiplicity)
        self.size = field.array_size

    @property
    def typename(self) -> str:
        return f"bool[{self.size}]"

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        base = cluster.first_entry * self.multiplicity * self.size if cluster else 0
        bits = store.elements(cluster, self.reps, 0, lo * self.size, hi * self.size, base)
        return bits.astype(bool).reshape(hi - lo, self.size)


class Record(Node):
    """A class or struct: a ``dict`` of its members for every element.

    A base class is a member too, named after the class and holding that
    class's own members - the way a ``TTree``'s objects keep theirs - rather
    than the ``:_0`` RNTuple numbers it with.
    """

    __slots__ = ("members",)

    def __init__(self, field: Field, multiplicity: int, members: list[Node]) -> None:
        super().__init__(field, multiplicity)
        self.members = members

    @property
    def typename(self) -> str:
        pairs = zip(self.names, self.members)
        inside = ", ".join(f"{name}: {node.typename}" for name, node in pairs)
        return "{" + inside + "}"

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        if not self.members:
            return [{} for _ in range(hi - lo)]
        columns = [rows(node.read(store, cluster, lo, hi)) for node in self.members]
        return [dict(zip(self.names, values)) for values in zip(*columns)]

    @property
    def names(self) -> list[str]:
        """Each member's key: its name, or for a base class the class it is."""
        return [_member_name(node.field) for node in self.members]


def _member_name(field: Field) -> str:
    return field.type if field.name.startswith(":") and field.type else field.name


class Tuple(Record):
    """A ``std::pair`` or ``std::tuple``: a ``tuple`` for every element."""

    __slots__ = ()

    @property
    def typename(self) -> str:
        return "tuple[" + ", ".join(node.typename for node in self.members) + "]"

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        columns = [rows(node.read(store, cluster, lo, hi)) for node in self.members]
        return list(zip(*columns))


class Variant(Node):
    """A ``std::variant``: whichever alternative each element holds, or ``None``.

    Its column says, for every element, which alternative is live - counting
    from one, with zero for none - and where that alternative's own value is
    among the values it holds, since each alternative fills its columns only
    when it is the one chosen.
    """

    __slots__ = ("alternatives",)

    def __init__(self, field: Field, multiplicity: int, alternatives: list[Node]) -> None:
        super().__init__(field, multiplicity)
        self.alternatives = alternatives

    @property
    def typename(self) -> str:
        return " | ".join(node.typename for node in self.alternatives)

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        switch = self.column(store, cluster, 0, lo, hi)
        out: list[Any] = [None] * len(switch)
        for tag, node in enumerate(self.alternatives, 1):
            chosen = np.flatnonzero(switch["tag"] == tag)
            if len(chosen):
                self._place(out, store, cluster, node, chosen, switch["index"][chosen])
        return out

    @staticmethod
    def _place(
        out: list[Any], store: Store, cluster: Cluster | None, node: Node, at: Any, where: Any
    ) -> None:
        first = int(where.min())
        values = rows(node.read(store, cluster, first, int(where.max()) + 1))
        for position, index in zip(at.tolist(), where.tolist()):
            out[position] = values[index - first]


class Wrapper(Node):
    """A field around one other, as ``std::atomic`` and an enum are: read as it."""

    __slots__ = ("inner",)

    def __init__(self, field: Field, multiplicity: int, inner: Node) -> None:
        super().__init__(field, multiplicity)
        self.inner = inner

    @property
    def typename(self) -> str:
        return self.inner.typename

    @property
    def jagged(self) -> bool:
        return self.inner.jagged

    def read(self, store: Store, cluster: Cluster | None, lo: int, hi: int) -> Any:
        return self.inner.read(store, cluster, lo, hi)


class _Refusal(Exception):
    """A field under the one being built that cannot be read, and why."""


def build(schema: Schema, field: Field, multiplicity: int = 1) -> Node:
    """The reader for ``field`` and everything under it, or a refusal naming why."""
    try:
        return _node(schema, field, multiplicity)
    except _Refusal as why:
        return Refused(field, str(why))


def _node(schema: Schema, field: Field, multiplicity: int) -> Node:
    _check_columns(schema, field)
    if field.role == STREAMER:
        raise _Refusal(
            f"a {field.type} that the ROOT streamer wrote whole, which is its own "
            f"format inside the RNTuple and not one this reader unpacks"
        )
    if field.flags & REPETITIVE:
        return _repetitive(schema, field, multiplicity)
    if field.role == VARIANT:
        return Variant(field, multiplicity, _children(schema, field, 0))
    if field.role == COLLECTION:
        return _collection(schema, field, multiplicity)
    if field.role == RECORD:
        return _record(schema, field, multiplicity)
    return _plain(schema, field, multiplicity)


def _check_columns(schema: Schema, field: Field) -> None:
    for rep in field.columns:
        for column in rep:
            code = schema.columns[column].type
            if code not in ENCODINGS:
                raise _Refusal(
                    f"a column of type {code:#04x}, which is newer than the RNTuple "
                    f"specification this reader follows"
                )


def _children(schema: Schema, field: Field, multiplicity: int) -> list[Node]:
    return [_node(schema, schema.fields[child], multiplicity) for child in field.children]


def _only_child(schema: Schema, field: Field, multiplicity: int) -> Node:
    if len(field.children) != 1:
        raise _Refusal(
            f"a {field.type or 'field'} with {len(field.children)} fields under it, "
            f"where exactly one belongs"
        )
    return _node(schema, schema.fields[field.children[0]], multiplicity)


def _repetitive(schema: Schema, field: Field, multiplicity: int) -> Node:
    if field.children:
        item = _only_child(schema, field, multiplicity * field.array_size)
        return Array(field, multiplicity, item)
    return Bitset(field, multiplicity)


def _collection(schema: Schema, field: Field, multiplicity: int) -> Node:
    item = _only_child(schema, field, 0)
    if field.type.startswith(MAPS) and isinstance(item, Tuple):
        return Map(field, multiplicity, item)
    if field.type.startswith(OPTIONALS):
        return Optional(field, multiplicity, item)
    return Collection(field, multiplicity, item)


def _record(schema: Schema, field: Field, multiplicity: int) -> Node:
    members = _children(schema, field, multiplicity)
    if field.type.startswith(TUPLES):
        return Tuple(field, multiplicity, members)
    return Record(field, multiplicity, members)


def _plain(schema: Schema, field: Field, multiplicity: int) -> Node:
    """A field with no structural role: a number, a string, a count or a wrapper."""
    width = len(field.columns[0]) if field.columns else 0
    if field.type.startswith(CARDINALITY) and width == 1:
        return Cardinality(field, multiplicity)
    if width == 2:
        return String(field, multiplicity)
    if width == 1:
        return Leaf(field, multiplicity, _dtype(schema, field))
    if field.children:
        return Wrapper(field, multiplicity, _only_child(schema, field, multiplicity))
    raise _Refusal(f"a {field.type} with no columns and nothing under it to read")


def _dtype(schema: Schema, field: Field) -> np.dtype[Any]:
    named = FUNDAMENTALS.get(field.type)
    if named is not None:
        return np.dtype(named)
    stored = ENCODINGS[schema.columns[field.columns[0][0]].type]
    if stored.name == "Switch" or stored.index:
        raise _Refusal(f"a {field.type} stored as {stored.name}, which is not a number")
    return stored.dtype.newbyteorder("=")
