"""What a tree looks like once it has been read: columns you can ask for.

The unit of I/O is the basket - a compressed block of consecutive entries for
one branch - and everything here is arranged around that. Asking for entries
100 to 200 reads the baskets that hold them and nothing else, which is what
makes it reasonable to iterate a hundred-gigabyte tree over the network from a
laptop: the bytes that cross the wire are the ones asked for.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .buffer import Buffer, gather
from .compression import decompress
from .errors import UnsupportedFeatureError
from .interp import Column, Flat, Members, Refused, Rows, Values, build

if TYPE_CHECKING:
    from .file import Source
    from .objects import BranchRecord, FriendRecord, LeafRecord

__all__ = ["Jagged", "Branch", "Group", "TTree"]

#: Entries per step when iterating, if nobody says otherwise.
DEFAULT_STEP = 10_000


class Jagged(Sequence[Any]):
    """Rows of different lengths: one flat array, and where each row starts.

        >>> jets.tolist()                          # doctest: +SKIP
        [[12.5, 3.5], [], [88.0, 1.25, 0.5]]

    This is the shape a physics file is usually in - a variable number of
    particles per collision - and it is kept flat because that is how it
    arrives and how a tensor wants it back. ``content`` is every value of
    every row in one NumPy array, and ``offsets`` the ``len + 1`` places the
    rows start and stop; row ``i`` is ``content[offsets[i]:offsets[i + 1]]``,
    which is exactly the layout Awkward Array and Arrow keep a list in.
    """

    __slots__ = ("content", "offsets")

    def __init__(self, content: Any, offsets: Any) -> None:
        self.content: np.ndarray[Any, Any] = np.asarray(content)
        self.offsets: np.ndarray[Any, Any] = np.asarray(offsets, dtype=np.int64)

    def __repr__(self) -> str:
        return f"<Jagged {len(self)} rows of {len(self.content)} {self.content.dtype} values>"

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            stop = max(start, stop)
            low, high = self.offsets[start], self.offsets[stop]
            return Jagged(self.content[low:high], self.offsets[start : stop + 1] - low)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("row out of range")
        return self.content[self.offsets[index] : self.offsets[index + 1]]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Jagged):
            return NotImplemented
        return bool(
            np.array_equal(self.offsets - self.offsets[0], other.offsets - other.offsets[0])
            and np.array_equal(self.flat, other.flat)
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def flat(self) -> np.ndarray[Any, Any]:
        """Every value of every row, in order, with nothing either side."""
        return self.content[self.offsets[0] : self.offsets[-1]]

    def lengths(self) -> np.ndarray[Any, Any]:
        """How long each row is."""
        return np.diff(self.offsets)

    def tolist(self) -> list[list[Any]]:
        return [row.tolist() for row in self]

    def padded(
        self, width: int | None = None, fill: float = 0.0
    ) -> tuple[np.ndarray[Any, Any], int]:
        """A rectangle: every row cut or filled to the same width.

        Gives back the flat values and the width, which is what a tensor of
        shape ``(rows, width)`` is made of - ``.reshape(-1, width)`` makes it
        that. ``width`` defaults to the longest row, so nothing is lost unless
        a number is asked for.
        """
        lengths = self.lengths()
        if width is None:
            width = int(lengths.max(initial=0))
        out = np.full((len(self), width), fill, dtype=self.content.dtype)
        kept = np.minimum(lengths, width)
        rows = np.repeat(np.arange(len(self)), kept)
        columns = np.arange(int(kept.sum())) - np.repeat(np.cumsum(kept) - kept, kept)
        out[rows, columns] = self.content[self.offsets[:-1][rows] + columns]
        return out.reshape(-1), width

    def to_awkward(self) -> Any:
        """The same rows as an Awkward Array, sharing the same memory."""
        from .library import awkward_list

        return awkward_list(self)

    def to_arrow(self) -> Any:
        """The same rows as an Arrow list array, which pandas and Polars take."""
        from .library import arrow_list

        return arrow_list(self)

    def take(self, rows: Any) -> Jagged:
        """The rows at the given places, in the order given, as rows of their own."""
        rows = np.asarray(rows, dtype=np.int64)
        starts = self.offsets[:-1][rows]
        lengths = self.offsets[1:][rows] - starts
        offsets = np.zeros(len(rows) + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        index = np.repeat(starts - offsets[:-1], lengths) + np.arange(offsets[-1])
        return Jagged(self.content[index], offsets)

    @staticmethod
    def join(pieces: Sequence[Jagged]) -> Jagged:
        """Rows from several places, one after another, with the offsets carried on."""
        lengths = np.concatenate([piece.lengths() for piece in pieces])
        offsets = np.zeros(len(lengths) + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return Jagged(np.concatenate([piece.flat for piece in pieces]), offsets)


def concatenate(pieces: Sequence[Any]) -> Any:
    """The values of several reads of one column, as one read of them all.

    What one read gives, several give joined: NumPy arrays end to end,
    :class:`Jagged` rows with each piece's offsets carried on from the last,
    and lists of values one list. This is how a range read across baskets,
    files of a chain, or a scattered set of entries comes back whole.
    """
    first = pieces[0]
    if isinstance(first, Jagged):
        return Jagged.join(pieces)
    if isinstance(first, np.ndarray):
        return np.concatenate(pieces)
    joined: list[Any] = []
    for piece in pieces:
        joined.extend(piece)
    return joined


def scattered(rows: Any, bounds: Any, read: Any, empty: Any) -> Any:
    """Entries picked from anywhere, read a block at a time and put back in order.

    ``bounds`` are where each block - a basket, a file of a chain - starts,
    and one past where the last ends. The entries are sorted into their
    blocks, ``read(block, entries)`` gives the values of each block's share,
    and the pieces are joined and put back in the order ``rows`` asked for
    them in; ``empty()`` is what asking for none at all gives.
    """
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    ordered = rows[order]
    homes = np.searchsorted(np.asarray(bounds, dtype=np.int64), ordered, side="right") - 1
    pieces = [read(int(home), ordered[homes == home]) for home in np.unique(homes)]
    if not pieces:
        return empty()
    joined = concatenate(pieces)
    if np.array_equal(order, np.arange(len(order))):
        return joined
    return take(joined, np.argsort(order, kind="stable"))


def take(values: Any, rows: Any) -> Any:
    """The entries at the given places of what one read gave, whatever its shape."""
    if isinstance(values, (np.ndarray, Jagged)):
        return values[rows] if isinstance(values, np.ndarray) else values.take(rows)
    return [values[row] for row in np.asarray(rows).tolist()]


def _bounds(total: int, entry_start: int, entry_stop: int | None) -> tuple[int, int]:
    """A range of entries, with negative ends counted from the last, as in Python."""
    start = total + entry_start if entry_start < 0 else entry_start
    stop = total if entry_stop is None else entry_stop
    if stop < 0:
        stop += total
    return max(start, 0), min(stop, total)


class Basket:
    """One compressed block of entries, decompressed and ready to slice.

    Nearly always a record of its own somewhere else in the file, which is
    what makes reading a range of entries cheap. A tree small enough never to
    have flushed one keeps its baskets inside the branch instead, and those
    arrive with their bytes already in hand.
    """

    __slots__ = ("keylen", "nevsize", "nevbuf", "last", "data", "offsets")

    def __init__(
        self,
        keylen: int,
        nevsize: int,
        nevbuf: int,
        last: int,
        data: bytes,
        offsets: list[int],
    ) -> None:
        self.keylen = keylen
        self.nevsize = nevsize
        self.nevbuf = nevbuf
        self.last = last
        self.data = data
        self.offsets = offsets

    @classmethod
    def keyed(cls, source: Source, seek: int, nbytes: int, has_offsets: bool) -> Basket:
        """The usual kind: a record of its own, read from where the branch says."""
        from .file import Key

        raw = source.read(seek, nbytes)
        head = Buffer(raw)
        key = Key(head)
        head.i16()  # the basket's own version, which adds nothing we use
        head.i32()  # the buffer size it was built with
        nevsize = head.i32()
        if nevsize < 0:  # a negative size means feature bits follow
            nevsize = -nevsize
            head.skip_record()
        nevbuf, last = head.i32(), head.i32()

        payload = raw[key.keylen :]
        data = decompress(payload, key.objlen) if key.compressed else payload
        offsets: list[int] = []
        if has_offsets:
            at = Buffer(data, key.keylen)
            at.pos = last
            offsets = at.i32s(at.i32())
        return cls(key.keylen, nevsize, nevbuf, last, data, offsets)

    @classmethod
    def inline(cls, buf: Buffer) -> Basket | None:
        """A basket written into the branch record, bytes and all.

        ``None`` says this one kept no bytes and the branch's seek point is
        where they are, which is how a tree that was flushed writes the
        placeholders it has already emptied.
        """
        from .file import Key

        key = Key(buf)
        version = buf.i16()
        buf.i32()  # the buffer size it was built with
        nevsize = buf.i32()
        if nevsize < 0:
            nevsize = -nevsize
            buf.skip_record()
        nevbuf, last = buf.i32(), buf.i32()
        flag, offsets = _inline_offsets(buf, buf.u8(), nevbuf)
        if flag != 1 and flag <= 10:
            return None
        size = last if version > 1 else buf.i32()
        return cls(key.keylen, nevsize, nevbuf, last, buf.take(size)[key.keylen :], offsets)

    def start_of(self, entry: int, offset: int) -> int:
        """Where a leaf's bytes for one entry begin, in the payload."""
        if self.offsets:
            return self.offsets[entry] - self.keylen + offset
        return entry * self.nevsize + offset

    def end_of(self, entry: int) -> int:
        """Where the entry after this one begins, for a row of unknown length."""
        boundary = self.offsets[entry + 1] if entry + 1 < self.nevbuf else self.last
        return boundary - self.keylen

    def starts(self, low: int, high: int, offset: int) -> np.ndarray[Any, Any]:
        """:meth:`start_of` for every entry from ``low`` up to ``high``, at once."""
        if self.offsets:
            return np.asarray(self.offsets[low:high], dtype=np.int64) - self.keylen + offset
        return np.arange(low, high, dtype=np.int64) * self.nevsize + offset

    def ends(self, low: int, high: int) -> np.ndarray[Any, Any]:
        """:meth:`end_of` for every entry from ``low`` up to ``high``, at once."""
        table = np.asarray([*self.offsets[: self.nevbuf], self.last], dtype=np.int64)
        return table[low + 1 : high + 1] - self.keylen


def _inline_offsets(buf: Buffer, flag: int, entries: int) -> tuple[int, list[int]]:
    if flag >= 80:
        return flag - 80, []  # offsets dropped; derive them from entry size
    if flag % 10 != 1 or not entries:
        return flag, []
    offsets = buf.i32s(buf.i32())
    if 20 < flag < 40:  # the top byte is a displacement, not a place
        offsets = [at & 0xFFFFFF for at in offsets]
    if flag > 40:
        buf.i32s(buf.i32())  # displacements into a tree of objects
    return flag, offsets


class Branch:
    """One column of a tree: a name, a type, and the entries under it.

        >>> tree["pt"].array(0, 1000)              # doctest: +SKIP
        array([22.5, 19. , ...], dtype=float32)

    A branch holding several leaves appears once per leaf, named
    ``branch.leaf``, which is how ROOT itself writes such a name.
    """

    __slots__ = ("name", "record", "leaf", "column", "_source", "_cached")

    def __init__(
        self,
        name: str,
        record: BranchRecord,
        leaf: LeafRecord,
        column: Column,
        source: Source,
    ) -> None:
        self.name = name
        self.record = record
        self.leaf = leaf
        #: How this column's bytes turn into values.
        self.column = column
        self._source = source
        self._cached: tuple[int, Basket] | None = None

    def __repr__(self) -> str:
        return f"<Branch {self.name!r} of {self.typename or self.leaf.classname}>"

    def __len__(self) -> int:
        return self.num_entries

    @property
    def title(self) -> str:
        return self.leaf.title

    @property
    def num_entries(self) -> int:
        return self.record.entries

    @property
    def typename(self) -> str | None:
        """``'float32'``, ``'list[str]'``, or ``None`` if it cannot be decoded."""
        return self.column.typename

    @property
    def length(self) -> int:
        """How many values each entry holds, for a column of fixed-size arrays.

        ``1`` for a plain number, ``10`` for ``x[10]``, which a fixed column
        gives back as an array of shape ``(entries, 10)``. A variable column
        says ``1`` here and means the lengths in :class:`Jagged` instead.
        """
        return self.column.length if isinstance(self.column, Flat) else 1

    @property
    def is_jagged(self) -> bool:
        """Does the number of values per entry change from entry to entry?"""
        return isinstance(self.column, Rows)

    @property
    def num_baskets(self) -> int:
        return max(len(self.record.basket_seek), len(self.record.baskets))

    def _refuse_if_unreadable(self) -> None:
        if isinstance(self.column, Refused):
            raise UnsupportedFeatureError(
                f"{self.name!r} holds {self.column.reason}; tree.unreadable lists every "
                f"column this file has that cannot be read, each with its reason"
            )
        if self.is_jagged and len(self.record.leaves) > 1:
            raise UnsupportedFeatureError(
                f"{self.name!r} is a variable-length leaf sharing a branch with "
                f"{len(self.record.leaves) - 1} others, where the file does not say alone how "
                f"long each row is; write it as its own branch and it reads"
            )

    def _bounds(self, entry_start: int, entry_stop: int | None) -> tuple[int, int]:
        return _bounds(self.num_entries, entry_start, entry_stop)

    def _spans(self, start: int, stop: int) -> Iterator[tuple[int, int, int]]:
        """Which baskets hold ``[start, stop)``, and the part of each to take."""
        bounds = self.record.basket_entry
        for index in range(self.num_baskets):
            low, high = bounds[index], bounds[index + 1]
            if high <= start or low >= stop:
                continue
            yield index, max(start, low) - low, min(stop, high) - low

    def basket(self, index: int) -> Basket:
        """Read one basket, remembering the last so a small step is not a reread."""
        if index < len(self.record.baskets):
            return self.record.baskets[index]  # already here, and never on its own
        if self._cached is not None and self._cached[0] == index:
            return self._cached[1]
        basket = Basket.keyed(
            self._source,
            self.record.basket_seek[index],
            self.record.basket_bytes[index],
            self.record.entry_offset_len > 0,
        )
        self._cached = (index, basket)
        return basket

    def array(
        self, entry_start: int = 0, entry_stop: int | None = None, *, entries: Any = None
    ) -> Any:
        """The values for a range of entries, or for the entries named.

        Fixed-size columns come back as a NumPy array - one value per entry, or
        ``length`` of them one after another - variable ones as
        :class:`Jagged`, and anything that is neither - strings, lists of
        lists, maps - as a list with one Python value per entry.

        ``entries`` picks entries rather than a range of them: an array of
        entry numbers, a mask of one bool per entry, or an
        :class:`~.entries.EntryList`. Only the baskets holding one of them
        are read, and the values come back in the order the entries were asked
        for in.
        """
        if entries is not None:
            from .entries import selected

            return self.pick(selected(entries, self.num_entries))
        self._refuse_if_unreadable()
        start, stop = self._bounds(entry_start, entry_stop)
        column = self.column
        if isinstance(column, Values):
            return self._objects(column, start, stop)
        if isinstance(column, Rows):
            return self._rows(column, start, stop)
        assert isinstance(column, Flat)
        return self._flat(column, start, stop)

    def pick(self, rows: np.ndarray[Any, Any]) -> Any:
        """The values of the entries numbered in ``rows``, reading only their baskets.

        The entries are sorted into the baskets that hold them, each basket
        is read once for the span from its first wanted entry to its last,
        and what is wanted of that span is taken; the pieces are put back in
        the order ``rows`` asked for them in.
        """
        bounds = self.record.basket_entry[: self.num_baskets + 1]

        def read(_home: int, inside: np.ndarray[Any, Any]) -> Any:
            low = int(inside[0])
            return take(self.array(low, int(inside[-1]) + 1), inside - low)

        return scattered(rows, bounds, read, lambda: self.array(0, 0))

    def _flat(self, column: Flat, start: int, stop: int) -> Any:
        size = column.length * column.itemsize
        offset = self.leaf.offset
        pieces = []
        for index, low, high in self._spans(start, stop):
            basket = self.basket(index)
            if not basket.offsets and offset == 0 and basket.nevsize == size:
                pieces.append(basket.data[low * size : high * size])  # the whole run at once
            else:
                starts = basket.starts(low, high, offset)
                pieces.append(gather(basket.data, starts, np.full(len(starts), size)))
        values = column.decode(b"".join(pieces))
        return values.reshape(-1, column.length) if column.length > 1 else values

    def _rows(self, column: Rows, start: int, stop: int) -> Jagged:
        pieces = []
        counts = [np.zeros(0, np.int64)]
        for index, low, high in self._spans(start, stop):
            basket = self.basket(index)
            at, end = column.spans(basket, low, high, self.leaf.offset)
            counts.append((end - at) // column.itemsize)
            pieces.append(gather(basket.data, at, end - at))
        values = column.decode(b"".join(pieces))
        lengths = np.concatenate(counts)
        offsets = np.zeros(len(lengths) + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return Jagged(values, offsets)

    def _objects(self, column: Values, start: int, stop: int) -> list[Any]:
        out = []
        for index, low, high in self._spans(start, stop):
            basket = self.basket(index)
            reader = Buffer(basket.data, basket.keylen)
            for entry in range(low, high):
                at = basket.start_of(entry, self.leaf.offset) + basket.keylen
                out.append(column.value(reader, at))
        return out


class Group(Branch):
    """A split C++ object, put back together from the branches under it.

        >>> tree["evt"].array(0, 1)                # doctest: +SKIP
        [{'I32': -1, 'Str': 'evt-000', 'P3': {'Px': -1, 'Py': -1.0, 'Pz': -1}}]

    ROOT splits an object into one branch per member and leaves the object
    itself holding nothing at all. Every member is readable on its own - and
    reading them on their own is what makes a large file cheap to walk - so
    this is a convenience rather than the only way in: it reads the same
    baskets and pays the same price, and gives back the object per entry
    instead of the columns across entries.

    A member this reader will not decode is left out of the dictionary and
    named in :attr:`unreadable`, which is the same sentence ``tree.unreadable``
    gives for it under its own name.
    """

    __slots__ = ("members", "_branches")

    def __init__(
        self,
        name: str,
        record: BranchRecord,
        leaf: LeafRecord,
        source: Source,
        branches: dict[str, Branch],
    ) -> None:
        super().__init__(name, record, leaf, Members(), source)
        #: The member's own name, against the branch name it is kept under.
        self.members: dict[str, str] = {}
        self._branches = branches

    def __repr__(self) -> str:
        return f"<Group {self.name!r} of {len(self.members)} members>"

    @property
    def unreadable(self) -> dict[str, str]:
        """The members left out of each dictionary, and why."""
        return {
            member: column.reason
            for member, label in self.members.items()
            if isinstance(column := self._branches[label].column, Refused)
        }

    def array(
        self, entry_start: int = 0, entry_stop: int | None = None, *, entries: Any = None
    ) -> list[dict[str, Any]]:
        """One dictionary per entry, a key for each member that reads."""
        if entries is not None:
            return list(super().array(entries=entries))
        start, stop = self._bounds(entry_start, entry_stop)
        return self._assemble(max(stop - start, 0), lambda branch: branch.array(start, stop))

    def pick(self, rows: np.ndarray[Any, Any]) -> list[dict[str, Any]]:
        """One dictionary for each entry numbered in ``rows``, read member by member."""
        return self._assemble(len(rows), lambda branch: branch.pick(rows))

    def _assemble(self, count: int, read: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [{} for _ in range(count)]
        for member, label in self.members.items():
            branch = self._branches[label]
            if isinstance(branch.column, Refused):
                continue
            values = read(branch)
            plain = isinstance(values, np.ndarray) and values.ndim == 1
            for index, row in enumerate(rows):
                row[member] = values[index].item() if plain else values[index]
        return rows


class TTree:
    """A tree, and the columns in it.

        >>> tree                                   # doctest: +SKIP
        <TTree 'events' with 4 branches and 1000 entries>
        >>> for batch in tree.iterate(["pt", "eta"], step=1000):   # doctest: +SKIP
        ...     train(batch["pt"], batch["eta"])

    ``len(tree)`` is the number of entries, because that is what a tree is a
    lot of; the number of columns is ``len(tree.branches)``.

    A tree can have friends - other trees of the same entries, read beside
    it, whether ROOT recorded them in the file or :meth:`add_friend` adds
    them - and a friend's column is asked for by name like one of the tree's
    own: ``tree["alias.branch"]``, or ``tree["branch"]`` when no other tree
    has one of that name.
    """

    __slots__ = (
        "name",
        "title",
        "num_entries",
        "branches",
        "unreadable",
        "_source",
        "_friends",
        "_recorded",
    )

    def __init__(
        self,
        name: str,
        title: str,
        num_entries: int,
        records: list[BranchRecord],
        source: Source,
        friends: Sequence[FriendRecord] = (),
    ) -> None:
        self.name = name
        self.title = title
        self.num_entries = num_entries
        self._source = source
        self.branches: dict[str, Branch] = {}
        self.unreadable: dict[str, str] = {}
        self._friends: dict[str, Any] = {}
        #: The friends the file recorded, found and opened the first time a
        #: friend is asked for rather than every time a tree is opened.
        self._recorded = list(friends)
        for top in records:
            self._add(top, source)

    def _add(self, record: BranchRecord, source: Source) -> list[str]:
        """Take in one branch and everything below it; say what it is called.

        A branch that holds no baskets but has branches under it is ROOT's
        way of writing a split object: nothing of it is in the file except
        its members, so it becomes a :class:`Group` over them rather than a
        column that cannot be read.
        """
        from .objects import LeafRecord

        many = len(record.leaves) > 1
        labels = self._add_leaves(record, source, many)
        split = record.branches and not record.basket_seek and not many
        if not split:
            self._add_children(record, source)
            return labels
        leaf = record.leaves[0] if record.leaves else LeafRecord("TLeafElement")
        group = Group(record.name, record, leaf, source, self.branches)
        self.branches[record.name] = group  # in place, where its own leaf was
        self.unreadable.pop(record.name, None)
        self._add_group_children(record, source, group)
        return [record.name]

    def _add_leaves(self, record: BranchRecord, source: Source, many: bool) -> list[str]:
        labels = []
        for leaf in record.leaves:
            label = f"{record.name}.{leaf.name}" if many else record.name
            column = build(record, leaf, source)
            self.branches[label] = Branch(label, record, leaf, column, source)
            if isinstance(column, Refused):
                self.unreadable[label] = column.reason
            labels.append(label)
        return labels

    def _add_children(self, record: BranchRecord, source: Source) -> None:
        for child in record.branches:
            self._add(child, source)

    def _add_group_children(self, record: BranchRecord, source: Source, group: Group) -> None:
        prefix = f"{record.name}."
        for child in record.branches:
            for label in self._add(child, source):
                group.members[label.removeprefix(prefix)] = label

    def __repr__(self) -> str:
        return (
            f"<TTree {self.name!r} with {len(self.branches)} branches "
            f"and {self.num_entries} entries>"
        )

    def __len__(self) -> int:
        return self.num_entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.branches)

    def __contains__(self, name: object) -> bool:
        if name in self.branches:
            return True
        try:
            return isinstance(name, str) and self._befriended(name) is not None
        except KeyError:
            return False

    def __getitem__(self, name: str) -> Any:
        found = self.branches.get(name)
        if found is None:
            found = self._befriended(name)
        if found is None:
            raise KeyError(
                f"{name!r} is not a branch of {self.name!r} or of any of its friends; "
                f"there is " + ", ".join(self.branches)
            )
        return found

    @property
    def friends(self) -> dict[str, Any]:
        """The trees read beside this one, by the name each is asked for under.

        Those ROOT recorded in the file are found the first time this is
        asked: a friend in another file is looked for beside this one when
        the file is local, and at the same place on the same server when it
        is not, and is opened then - and closed when this tree's file is.
        """
        while self._recorded:
            from .friends import open_friend

            record = self._recorded[0]
            self.add_friend(open_friend(record, self._source), record.alias)
            self._recorded.pop(0)  # only once found, so a friend not found stays missed
        return dict(self._friends)

    def add_friend(self, other: Any, alias: str | None = None) -> None:
        """Read ``other`` beside this tree, entry for entry, as ROOT's ``AddFriend`` does.

            >>> events.add_friend(f2["weights"], "w")          # doctest: +SKIP
            >>> events["w.nominal"].array(0, 10)

        ``other`` is a :class:`TTree` or a :class:`~.chain.Chain` of exactly
        as many entries, since entry ``i`` of a friend is read as entry ``i``
        of this tree and a count that differs means that is not what it is.
        Its columns are then here as ``alias.branch`` - the alias is the
        friend's own name unless given - and by their bare names too, when
        neither this tree nor another friend has one of the same name.
        """
        alias = str(alias or other.name)
        if len(other) != self.num_entries:
            raise ValueError(
                f"{alias!r} has {len(other)} entries and {self.name!r} has "
                f"{self.num_entries}: a friend is read entry for entry, so the two "
                f"have to be the same length"
            )
        if alias in self._friends:
            raise ValueError(f"{self.name!r} already has a friend called {alias!r}; give an alias")
        self._friends[alias] = other

    def _befriended(self, name: str) -> Any:
        """A friend's column: ``alias.branch``, or a bare name only one friend has."""
        friends = self.friends
        alias, dot, rest = name.partition(".")
        if dot and alias in friends and rest in friends[alias].keys():
            return friends[alias][rest]
        holders = [alias for alias, tree in friends.items() if name in tree.keys()]
        if len(holders) > 1:
            raise KeyError(
                f"{name!r} is a branch of {len(holders)} friends of {self.name!r} "
                f"({', '.join(holders)}); ask for it as alias.{name}"
            )
        return friends[holders[0]][name] if holders else None

    def keys(self) -> list[str]:
        """Every column, readable here or not."""
        return list(self.branches)

    def readable(self) -> list[str]:
        """The columns this reader can decode, which is what ``arrays`` defaults to.

        A split object is not one of them: its members are already here under
        their own names, and taking both would read every basket twice.
        """
        return [
            name
            for name, branch in self.branches.items()
            if name not in self.unreadable and not isinstance(branch, Group)
        ]

    def groups(self) -> list[str]:
        """The split objects: branches whose members are the branches under them."""
        return [name for name, branch in self.branches.items() if isinstance(branch, Group)]

    def typenames(self) -> dict[str, str]:
        """What each column holds, in Python's words."""
        return {
            name: branch.typename or f"? ({branch.leaf.classname})"
            for name, branch in self.branches.items()
        }

    def show(self) -> str:
        """A one-line-per-column summary, for looking before leaping.

        >>> print(tree.show())                 # doctest: +SKIP
        pt        float32   variable
        nmuon     int32
        """
        lines = []
        for name, branch in self.branches.items():
            kind = branch.typename or f"? ({branch.leaf.classname})"
            lines.append(f"{name:<24} {kind:<10}{' variable' if branch.is_jagged else ''}".rstrip())
        return "\n".join(lines)

    def arrays(
        self,
        names: Sequence[str] | None = None,
        entry_start: int = 0,
        entry_stop: int | None = None,
        *,
        library: str = "np",
        entries: Any = None,
        cut: str | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> Any:
        """Several columns at once, over the same range of entries.

            >>> tree.arrays(["pt", "eta"], library="pd")      # doctest: +SKIP
            >>> tree.arrays(["pt"], entries=f["selected"])    # doctest: +SKIP
            >>> tree.arrays(["Sum$(jet_pt > 30)"], cut="nJet >= 2")   # doctest: +SKIP

        With no names, every column this reader can decode; the ones it cannot
        are in :attr:`unreadable` with the reason, rather than quietly missing.
        ``library`` says what to hand them back as: ``np``, the default, is a
        dict of NumPy arrays; ``pd``, ``ak``, ``pa`` and ``pl`` are a pandas
        DataFrame, an Awkward Array, an Arrow table and a Polars DataFrame.

        ``entries`` reads the entries named instead of a range: an
        :class:`~.entries.EntryList` - the list for this tree, if it keeps one
        per tree - an array of entry numbers, or a mask of one bool per entry.
        Only the baskets holding them are read.

        A name that is not a branch is a ``TTree::Draw`` expression, given
        back under its own text, and ``cut`` is a selection in the same
        language; ``aliases`` are names standing for expressions, as ROOT's
        ``SetAlias`` makes them. Only the branches they need are read. See
        :mod:`xrdroot.formula.select` for how a cut over a collection applies.
        """
        from .formula.select import select

        return select(
            self,
            names,
            entry_start,
            entry_stop,
            library=library,
            entries=entries,
            cut=cut,
            aliases=aliases,
        )

    def _selected(self, entries: Any) -> np.ndarray[Any, Any]:
        from .entries import selected

        return selected(entries, self.num_entries, self.name, self._source.name)

    def iterate(
        self,
        names: Sequence[str] | None = None,
        *,
        step: int = DEFAULT_STEP,
        entry_start: int = 0,
        entry_stop: int | None = None,
        library: str = "np",
        entries: Any = None,
        cut: str | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> Iterator[Any]:
        """Walk the tree in batches, reading only what each batch needs.

            >>> for batch in tree.iterate(step=50_000):        # doctest: +SKIP
            ...     ...

        This is the one to reach for over a network: memory is one step, not
        one file, and a tree far larger than the machine goes through it.
        With ``entries``, the batches are ``step`` of the entries named at a
        time rather than ``step`` of the tree's. Expressions, ``cut`` and
        ``aliases`` are as :meth:`arrays` has them, a cut leaving each batch
        shorter than ``step`` by the entries it did not pass.
        """
        if step <= 0:
            raise ValueError("step must be at least one entry")
        chosen: dict[str, Any] = {"library": library, "cut": cut, "aliases": aliases}
        if entries is not None:
            rows = self._selected(entries)
            for at in range(0, len(rows), step):
                yield self.arrays(names, entries=rows[at : at + step], **chosen)
            return
        # The same counting from the end that one branch's ``array`` does, so a
        # negative start or stop means here what it means there.
        at, stop = _bounds(self.num_entries, entry_start, entry_stop)
        while at < stop:
            yield self.arrays(names, at, min(at + step, stop), **chosen)
            at += step
