"""An RNTuple, read: fields you can ask for, a range of entries at a time.

ROOT 7's columnar format keeps its data the way this library hands it back -
a column of plain values per leaf of the schema, in pages, grouped into
clusters of consecutive entries - so reading one is mostly finding the pages.
The anchor says where the header and footer are; the footer says where the
page list of each group of clusters is; the page list says where each
column's pages are in each cluster. A range of entries costs the page lists
of the groups it touches and the pages of the columns asked for in the
clusters it touches, and nothing else, which over a network is the whole
point.

:class:`RNTuple` asks to be used the way a :class:`~xrdroot.tree.TTree` is:
``len``, ``keys``, ``typenames``, ``show``, a field by name with an
``array`` method, and ``arrays`` and ``iterate`` for several at once, in any
of the libraries :mod:`~xrdroot.library` hands tables to.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..compression import decompress
from ..errors import FormatError
from .checksum import checksum, verifying
from .columns import decode, encoding, stored_size
from .envelope import FOOTER, HEADER, PAGE_LIST, Anchor, open_envelope, read_blob, read_block
from .fields import Node, Refused, build, join
from .schema import (
    DEFERRED,
    RECORD,
    Cluster,
    Group,
    Pages,
    read_footer,
    read_header,
    read_page_list,
)

if TYPE_CHECKING:
    from ..file import Key, Source
    from .schema import Column, Field, Schema

__all__ = ["RNTuple", "RField"]

#: Entries per step when iterating, if nobody says otherwise - as for a tree.
DEFAULT_STEP = 10_000
#: How many decoded pages are kept, so that a walk in steps smaller than a
#: page does not decode the same page once per step.
PAGE_CACHE = 64


def _bounds(total: int, entry_start: int, entry_stop: int | None) -> tuple[int, int]:
    """A range of entries, with negative ends counted from the last, as in Python."""
    start = total + entry_start if entry_start < 0 else entry_start
    stop = total if entry_stop is None else entry_stop
    if stop < 0:
        stop += total
    return max(start, 0), min(stop, total)


class Store:
    """Where the pages are, and the pages already read.

    The page list of a cluster group is read the first time a range touches
    that group, and a page the first time a range needs one of its elements;
    the last few decoded pages are kept.
    """

    __slots__ = ("source", "schema", "anchor", "groups", "_lists", "_pages")

    def __init__(self, source: Source, schema: Schema, anchor: Anchor, groups: list[Group]):
        self.source = source
        self.schema = schema
        self.anchor = anchor
        self.groups = groups
        self._lists: dict[int, list[Cluster]] = {}
        self._pages: OrderedDict[tuple[int, int], np.ndarray[Any, Any]] = OrderedDict()

    def clusters(self, group: int) -> list[Cluster]:
        """The clusters of one group, reading its page list the first time."""
        if group not in self._lists:
            link = self.groups[group].page_list
            raw = read_block(self.source, link, self.anchor.max_key)
            self._lists[group] = read_page_list(open_envelope(raw, PAGE_LIST))
        return self._lists[group]

    def covering(self, start: int, stop: int) -> Iterator[Cluster]:
        """Every cluster holding an entry of ``[start, stop)``, in order."""
        for index, group in enumerate(self.groups):
            if group.first_entry >= stop or group.first_entry + group.entries <= start:
                continue
            for cluster in self.clusters(index):
                if cluster.first_entry < stop and cluster.first_entry + cluster.entries > start:
                    yield cluster

    def elements(
        self, cluster: Cluster | None, reps: list[list[int]], slot: int, lo: int, hi: int, base: int
    ) -> np.ndarray[Any, Any]:
        """Elements ``lo`` to ``hi`` of a field's ``slot``-th column in one cluster.

        A field written through more than one representation has one live
        in each cluster, the others suppressed there; whichever is live is
        read. ``base`` is the element this cluster starts at, which is where
        the zeros in front of a column added partway through end.
        """
        column = self.schema.columns[reps[0][slot]]
        if hi <= lo or cluster is None:
            return np.zeros(0, encoding(column.type).dtype)
        column, pages = self._live(cluster, reps, slot)
        if pages is None:
            return np.zeros(hi - lo, encoding(column.type).dtype)
        skew = max(pages.first - base, 0) if column.flags & DEFERRED else 0
        return self._span(column, pages, lo - skew, hi - skew)

    def _live(
        self, cluster: Cluster, reps: list[list[int]], slot: int
    ) -> tuple[Column, Pages | None]:
        for rep in reps:
            if rep[slot] < len(cluster.columns) and not cluster.columns[rep[slot]].suppressed:
                return self.schema.columns[rep[slot]], cluster.columns[rep[slot]]
        return self.schema.columns[reps[0][slot]], None  # added after this cluster

    def _span(self, column: Column, pages: Pages, lo: int, hi: int) -> np.ndarray[Any, Any]:
        """Elements ``lo`` to ``hi`` of the pages, zeros wherever a deferred column has none.

        The range is counted from the column's first stored element here, so
        a deferred column asks for some before zero - the entries written
        before it existed - and perhaps some past its end.
        """
        total = pages.total
        before = min(hi, 0) - lo if lo < 0 else 0
        after = hi - max(lo, total) if hi > total else 0
        if (before or after) and not column.flags & DEFERRED:
            raise FormatError(
                f"RNTuple column {column.id} holds {total} elements in a cluster, where "
                f"elements {lo} to {hi} were needed from it"
            )
        values = self._stored(column, pages, min(max(lo, 0), total), max(min(hi, total), 0))
        if not (before or after):
            return values
        kind = encoding(column.type).dtype
        return np.concatenate([np.zeros(before, kind), values, np.zeros(after, kind)])

    def _stored(self, column: Column, pages: Pages, lo: int, hi: int) -> np.ndarray[Any, Any]:
        """Elements ``lo`` to ``hi`` that are in the pages, decoding the pages they are in."""
        if hi <= lo:
            return np.zeros(0, encoding(column.type).dtype)
        first = int(np.searchsorted(pages.starts, lo, side="right")) - 1
        last = int(np.searchsorted(pages.starts, hi, side="left"))
        pieces = [self._page(column, pages, index) for index in range(first, last)]
        start = int(pages.starts[first])
        values = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return values[lo - start : hi - start].copy()

    def _page(self, column: Column, pages: Pages, index: int) -> np.ndarray[Any, Any]:
        """One page, decoded - from the cache when it was decoded lately."""
        key = (pages.offsets[index], column.id)
        cached = self._pages.get(key)
        if cached is not None:
            self._pages.move_to_end(key)
            return cached
        values = self._decode(column, pages, index)
        self._pages[key] = values
        if len(self._pages) > PAGE_CACHE:
            self._pages.popitem(last=False)
        return values

    def _decode(self, column: Column, pages: Pages, index: int) -> np.ndarray[Any, Any]:
        size, count = pages.sizes[index], pages.counts[index]
        extra = 8 if pages.checked[index] else 0
        raw = read_blob(self.source, pages.offsets[index], size + extra, self.anchor.max_key)
        body = raw[:size]
        if extra and verifying() and checksum(body) != int.from_bytes(raw[size:], "little"):
            raise FormatError(
                f"a page of RNTuple column {column.id} does not match the checksum "
                f"stored with it, so the file is damaged there"
            )
        length = stored_size(encoding(column.type), column.bits, count)
        return decode(column, body if size == length else decompress(body, length), count)


class RField:
    """One field of an RNTuple: a name, a type, and the entries under it.

    >>> ntuple["Muon_pt"].array(0, 1000)       # doctest: +SKIP
    <Jagged 1000 rows of 2372 float32 values>
    """

    __slots__ = ("name", "node", "_ntuple")

    def __init__(self, name: str, node: Node, ntuple: RNTuple) -> None:
        self.name = name
        #: How this field's columns turn into values.
        self.node = node
        self._ntuple = ntuple

    def __repr__(self) -> str:
        return f"<RField {self.name!r} of {self.cxx_type or 'an untyped record'}>"

    def __len__(self) -> int:
        return self._ntuple.num_entries

    @property
    def cxx_type(self) -> str:
        """The C++ type the file says this field is, as ROOT normalised it."""
        return self.node.field.type

    @property
    def typename(self) -> str:
        """``'float32'``, ``'list[float32]'``, ``'str'`` - what the values are in Python."""
        return self.node.typename

    @property
    def is_jagged(self) -> bool:
        """Whether this reads as :class:`~xrdroot.tree.Jagged`, rows of numbers."""
        return self.node.jagged

    def array(self, entry_start: int = 0, entry_stop: int | None = None) -> Any:
        """The values for a range of entries, reading only the clusters they are in.

        A number per entry comes back as a NumPy array, a fixed-size array of
        them as a two-dimensional one, a collection of numbers as
        :class:`~xrdroot.tree.Jagged`, and anything else as a list with one
        Python value per entry.
        """
        store = self._ntuple._store
        start, stop = _bounds(self._ntuple.num_entries, entry_start, entry_stop)
        pieces = []
        for cluster in store.covering(start, stop):
            low = max(start, cluster.first_entry) - cluster.first_entry
            high = min(stop, cluster.first_entry + cluster.entries) - cluster.first_entry
            pieces.append(self.node.read(store, cluster, low, high))
        return join(pieces, self.node.read(store, None, 0, 0))


class RNTuple:
    """An RNTuple, and the fields in it.

        >>> ntuple = f["Events"]                   # doctest: +SKIP
        <RNTuple 'Events' with 18 fields and 1000 entries>
        >>> ntuple.arrays(["nMuon", "Muon_pt"], library="ak")   # doctest: +SKIP

    ``len(ntuple)`` is the number of entries. The fields are the top-level
    ones; a member of a record is reached through it with a dot, as
    ``ntuple["event.muons"]``, and reads as a column of its own.
    """

    __slots__ = ("name", "num_entries", "fields", "unreadable", "_store")

    def __init__(self, name: str, store: Store) -> None:
        self.name = name
        self._store = store
        self.num_entries = sum(group.entries for group in store.groups)
        self.fields: dict[str, RField] = {}
        #: The fields this reader will not decode, and why.
        self.unreadable: dict[str, str] = {}
        for field in store.schema.tops():
            node = build(store.schema, field)
            self.fields[field.name] = RField(field.name, node, self)
            if isinstance(node, Refused):
                self.unreadable[field.name] = node.reason

    @classmethod
    def from_key(cls, key: Key, source: Source) -> RNTuple:
        """The RNTuple a ``ROOT::RNTuple`` key anchors: its header and footer read."""
        anchor = Anchor.parse(key.payload(source), key.name)
        header = read_block(source, anchor.header, anchor.max_key)
        schema = read_header(open_envelope(header, HEADER))
        footer = read_block(source, anchor.footer, anchor.max_key)
        groups = read_footer(open_envelope(footer, FOOTER), schema)
        return cls(key.name, Store(source, schema, anchor, groups))

    def __repr__(self) -> str:
        return (
            f"<RNTuple {self.name!r} with {len(self.fields)} fields and {self.num_entries} entries>"
        )

    def __len__(self) -> int:
        return self.num_entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._find(name) is not None

    def __getitem__(self, name: str) -> RField:
        found = self._find(name)
        if found is None:
            raise KeyError(
                f"{name!r} is not a field of {self.name!r}; there is " + ", ".join(self.fields)
            )
        return found

    def _find(self, name: str) -> RField | None:
        """A field by name, or a record's member by its dotted path."""
        if name in self.fields:
            return self.fields[name]
        head, dot, rest = name.partition(".")
        top = self.fields.get(head)
        if not dot or top is None:
            return None
        field = self._member(top.node.field, rest.split("."))
        if field is None:
            return None
        return RField(name, build(self._store.schema, field), self)

    def _member(self, field: Field, path: list[str]) -> Field | None:
        """The field a path of member names leads to, through records alone."""
        fields = self._store.schema.fields
        for part in path:
            if field.role != RECORD:
                return None
            found = [fields[child] for child in field.children if fields[child].name == part]
            if not found:
                return None
            field = found[0]
        return field

    @property
    def description(self) -> str:
        return self._store.schema.description

    @property
    def writer(self) -> str:
        """Which program wrote this, in its own words: ``'ROOT v6.34.00'``."""
        return self._store.schema.writer

    @property
    def version(self) -> tuple[int, ...]:
        """The version of the binary format it was written to, epoch first."""
        return self._store.anchor.version

    @property
    def num_clusters(self) -> int:
        return sum(group.clusters for group in self._store.groups)

    def keys(self) -> list[str]:
        """Every top-level field, readable here or not."""
        return list(self.fields)

    def readable(self) -> list[str]:
        """The fields this reader decodes, which is what ``arrays`` defaults to."""
        return [name for name in self.fields if name not in self.unreadable]

    def typenames(self) -> dict[str, str]:
        """What each field holds, in Python's words."""
        return {name: field.typename for name, field in self.fields.items()}

    def cxx_types(self) -> dict[str, str]:
        """What each field is, in the C++ the file was written from."""
        return {name: field.cxx_type for name, field in self.fields.items()}

    def show(self) -> str:
        """A one-line-per-field summary: name, Python type and C++ type."""
        return "\n".join(
            f"{name:<24} {field.typename:<24} {field.cxx_type}".rstrip()
            for name, field in self.fields.items()
        )

    def arrays(
        self,
        names: Sequence[str] | None = None,
        entry_start: int = 0,
        entry_stop: int | None = None,
        *,
        library: str = "np",
    ) -> Any:
        """Several fields at once, over the same range of entries.

            >>> ntuple.arrays(["nMuon", "Muon_pt"], library="pd")   # doctest: +SKIP

        With no names, every field this reader can decode; the ones it cannot
        are in :attr:`unreadable` with the reason. ``library`` is as for a
        tree: ``np``, the default, is a dict; ``pd``, ``ak``, ``pa`` and
        ``pl`` are a pandas DataFrame, an Awkward Array, an Arrow table and a
        Polars DataFrame.
        """
        from ..library import convert

        wanted = self.readable() if names is None else list(names)
        columns = {name: self[name].array(entry_start, entry_stop) for name in wanted}
        return convert(columns, library)

    def iterate(
        self,
        names: Sequence[str] | None = None,
        *,
        step: int = DEFAULT_STEP,
        entry_start: int = 0,
        entry_stop: int | None = None,
        library: str = "np",
    ) -> Iterator[Any]:
        """Walk the entries in batches, reading only what each batch needs."""
        if step <= 0:
            raise ValueError("step must be at least one entry")
        at, stop = _bounds(self.num_entries, entry_start, entry_stop)
        while at < stop:
            yield self.arrays(names, at, min(at + step, stop), library=library)
            at += step
