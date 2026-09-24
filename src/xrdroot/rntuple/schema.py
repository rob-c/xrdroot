"""What an RNTuple says about itself: fields, columns, clusters and pages.

The header describes the schema - a tree of *fields*, each a C++ type, and
the *columns* of plain values the leaves of that tree are stored in. The
footer adds whatever the schema grew while the file was being written, and
points at the *page lists*, one per group of clusters, which say where each
column's pages are in each cluster. Everything in this module is that
description, read out of its envelopes into plain objects and, for the
writer, back again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from .envelope import Builder, Cursor, Link

__all__ = ["Field", "Column", "Schema", "Group", "Cluster", "Pages", "read_header", "read_footer"]

#: The structural roles a field can have, by the number the file gives.
LEAF, COLLECTION, RECORD, VARIANT, STREAMER = range(5)
#: Field flags: a fixed-size array, a projection of another field, a checksum.
REPETITIVE, PROJECTED, TYPE_CHECKSUM = 0x01, 0x02, 0x04
#: Column flags: written from partway through, and carrying a range of values.
DEFERRED, RANGED = 0x01, 0x02
#: Flag bits in a cluster summary: sharded clusters, reserved for the future.
SHARDED = 0x01


class Field:
    """One node of the schema tree: a name, a C++ type and how it is stored."""

    __slots__ = (
        "id",
        "version",
        "type_version",
        "parent",
        "role",
        "flags",
        "name",
        "type",
        "alias",
        "description",
        "array_size",
        "source",
        "children",
        "columns",
    )

    def __init__(self, id: int, parent: int, role: int, name: str, type: str) -> None:
        self.id = id
        self.version = 0
        self.type_version = 0
        #: The enclosing field; a top-level field is its own parent.
        self.parent = parent
        self.role = role
        self.flags = 0
        self.name = name
        self.type = type
        self.alias = ""
        self.description = ""
        #: For a fixed-size array, how many elements each entry holds.
        self.array_size = 0
        #: For a projected field, the field it projects.
        self.source = -1
        self.children: list[int] = []
        #: The columns attached here, grouped by representation.
        self.columns: list[list[int]] = []

    def __repr__(self) -> str:
        return f"<Field {self.id} {self.name!r} of {self.type!r}>"

    @property
    def top(self) -> bool:
        return self.parent == self.id


class Column:
    """One column of plain values: its encoding, and the field it belongs to."""

    __slots__ = ("id", "type", "bits", "field", "flags", "rep", "first", "low", "high")

    def __init__(self, id: int, type: int, bits: int, field: int, rep: int = 0) -> None:
        self.id = id
        self.type = type
        self.bits = bits
        self.field = field
        self.flags = 0
        #: Which of the field's alternative representations this belongs to.
        self.rep = rep
        #: For a deferred column, the first element it holds; before it, zeros.
        self.first = 0
        #: For a column with a range, the least and greatest value it holds.
        self.low = 0.0
        self.high = 0.0

    def __repr__(self) -> str:
        return f"<Column {self.id} of type {self.type:#x} for field {self.field}>"


class Schema:
    """The fields and columns, in the order the file numbers them."""

    __slots__ = ("name", "description", "writer", "fields", "columns", "aliases")

    def __init__(self, name: str, description: str, writer: str) -> None:
        self.name = name
        self.description = description
        self.writer = writer
        self.fields: list[Field] = []
        self.columns: list[Column] = []
        #: ``(physical column, projected field)`` for every alias column.
        self.aliases: list[tuple[int, int]] = []

    def link(self) -> None:
        """Fill in the child and column lists the serialized order leaves implicit."""
        for field in self.fields:
            field.children, field.columns = [], []
        for field in self.fields:
            if not field.top:
                self.fields[field.parent].children.append(field.id)
        attached = [(column.field, column.id) for column in self.columns]
        for field_id, column_id in attached + [(f, c) for c, f in self.aliases]:
            self._attach(self.fields[field_id], self.columns[column_id])

    @staticmethod
    def _attach(field: Field, column: Column) -> None:
        """One column to its field, under the representation it belongs to."""
        while len(field.columns) <= column.rep:
            field.columns.append([])
        field.columns[column.rep].append(column.id)

    def tops(self) -> list[Field]:
        return [field for field in self.fields if field.top]


def read_header(cursor: Cursor) -> Schema:
    cursor.flags()
    schema = Schema(cursor.string(), cursor.string(), cursor.string())
    read_description(cursor, schema)
    schema.link()
    return schema


def read_description(cursor: Cursor, schema: Schema) -> None:
    """The four lists that describe fields and columns, in a header or an extension."""
    _each(cursor, lambda: schema.fields.append(_field(cursor, len(schema.fields))))
    _each(cursor, lambda: schema.columns.append(_column(cursor, len(schema.columns))))
    _each(cursor, lambda: schema.aliases.append((cursor.u32(), cursor.u32())))
    _each(cursor, lambda: None)  # extra type information, for streamer fields only


def _each(cursor: Cursor, read: Any) -> None:
    """Every record of a list frame, each stepped over by its own declared size."""
    end, count = cursor.frame(listed=True)
    for _ in range(count):
        record, _nothing = cursor.frame(listed=False)
        read()
        cursor.seek(record)
    cursor.seek(end)


def _field(cursor: Cursor, id: int) -> Field:
    version, type_version, parent = cursor.u32(), cursor.u32(), cursor.u32()
    role, flags = cursor.u16(), cursor.u16()
    field = Field(id, parent, role, cursor.string(), cursor.string())
    field.version, field.type_version, field.flags = version, type_version, flags
    field.alias, field.description = cursor.string(), cursor.string()
    if flags & REPETITIVE:
        field.array_size = cursor.u64()
    if flags & PROJECTED:
        field.source = cursor.u32()
    return field


def _column(cursor: Cursor, id: int) -> Column:
    kind, bits, field = cursor.u16(), cursor.u16(), cursor.u32()
    flags, rep = cursor.u16(), cursor.u16()
    column = Column(id, kind, bits, field, rep)
    column.flags = flags
    if flags & DEFERRED:
        column.first = cursor.i64()
    if flags & RANGED:
        column.low, column.high = cursor.f64(), cursor.f64()
    return column


class Group:
    """A run of clusters sharing one page list, and the entries they cover."""

    __slots__ = ("first_entry", "entries", "clusters", "page_list")

    def __init__(self, first_entry: int, entries: int, clusters: int, page_list: Link) -> None:
        self.first_entry = first_entry
        self.entries = entries
        self.clusters = clusters
        self.page_list = page_list


def read_footer(cursor: Cursor, schema: Schema) -> list[Group]:
    """The cluster groups, after folding the schema extension into ``schema``."""
    cursor.flags()
    cursor.u64()  # the header's checksum, which the header itself was checked against
    end, _nothing = cursor.frame(listed=False)
    read_description(cursor, schema)
    schema.link()
    cursor.seek(end)
    groups: list[Group] = []
    _each(cursor, lambda: groups.append(_group(cursor)))
    return groups  # linked attribute sets may follow; nothing here reads them


def _group(cursor: Cursor) -> Group:
    first, span, clusters = cursor.u64(), cursor.u64(), cursor.u32()
    return Group(first, span, clusters, cursor.link())


class Pages:
    """One column's pages in one cluster: how many elements each, and where.

    ``first`` is the column's element number at the start of the cluster;
    ``starts`` is where each page begins counted from there, with the total
    at the end, so finding the pages a range needs is a search.
    """

    __slots__ = ("first", "compression", "counts", "starts", "sizes", "offsets", "checked")

    def __init__(self, first: int, compression: int) -> None:
        self.first = first
        self.compression = compression
        self.counts: list[int] = []
        self.sizes: list[int] = []
        self.offsets: list[int] = []
        self.checked: list[bool] = []
        self.starts: np.ndarray[Any, Any] = np.zeros(1, np.int64)

    @property
    def suppressed(self) -> bool:
        """Whether another representation of the field is the live one here."""
        return self.first < 0

    @property
    def total(self) -> int:
        return int(self.starts[-1])


class Cluster:
    """A run of entries, and for each column where its pages are."""

    __slots__ = ("first_entry", "entries", "columns")

    def __init__(self, first_entry: int, entries: int) -> None:
        self.first_entry = first_entry
        self.entries = entries
        self.columns: list[Pages] = []


def read_page_list(cursor: Cursor) -> list[Cluster]:
    cursor.u64()  # the header's checksum, again
    clusters: list[Cluster] = []
    _each(cursor, lambda: clusters.append(_summary(cursor)))
    end, count = cursor.frame(listed=True)
    for cluster in clusters[:count]:
        outer, columns = cursor.frame(listed=True)
        cluster.columns = [_pages(cursor) for _ in range(columns)]
        cursor.seek(outer)
    cursor.seek(end)
    return clusters


def _summary(cursor: Cursor) -> Cluster:
    first, word = cursor.u64(), cursor.u64()
    if (word >> 56) & SHARDED:
        raise UnsupportedFeatureError(
            f"the RNTuple page list has a sharded cluster at entry {first}, which is "
            f"reserved for a version of the format after this reader"
        )
    return Cluster(first, word & ((1 << 56) - 1))


def _pages(cursor: Cursor) -> Pages:
    end, count = cursor.frame(listed=True)
    places = []
    for _ in range(count):
        places.append((cursor.i32(), *cursor.locator()))
    first = cursor.i64()
    pages = Pages(first, cursor.u32() if first >= 0 else 0)
    for elements, size, offset in places:
        pages.counts.append(abs(elements))
        pages.checked.append(elements < 0)
        pages.sizes.append(size)
        pages.offsets.append(offset)
    pages.starts = np.concatenate([[0], np.cumsum(pages.counts, dtype=np.int64)])
    cursor.seek(end)
    return pages


def write_description(out: Builder, schema: Schema | None) -> None:
    """The four lists of a schema description - all of ``schema``, or nothing."""
    fields = schema.fields if schema else []
    columns = schema.columns if schema else []
    mark = out.list(len(fields))
    for field in fields:
        _write_field(out, field)
    out.close(mark)
    mark = out.list(len(columns))
    for column in columns:
        record = out.record()
        out.pack("HHIHH", column.type, column.bits, column.field, 0, column.rep)
        out.close(record)
    out.close(mark)
    out.empty_list()  # no alias columns: nothing written here is projected
    out.empty_list()  # no extra type information: nothing is streamed


def _write_field(out: Builder, field: Field) -> None:
    record = out.record()
    out.pack("IIIHH", 0, 0, field.parent, field.role, field.flags)
    for text in (field.name, field.type, field.alias, field.description):
        out.string(text)
    if field.flags & REPETITIVE:
        out.pack("Q", field.array_size)
    out.close(record)
