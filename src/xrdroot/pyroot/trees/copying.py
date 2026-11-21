"""``CloneTree`` and ``CopyTree``: a tree's branches, and some of its entries, as a new tree.

A clone has a slot for every active leaf of the tree it came from, each
reading - as ROOT's clone shares its branches' buffers - the value that tree
last read into it, so ``CloneTree(0)`` followed by a loop of ``GetEntry`` and
``Fill`` copies the entries the loop chooses. Whatever the clone is to start
with - every entry, the first ``n``, or those a selection keeps - is copied
across whole columns at a time rather than an entry at a time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...interp import Each, Values
from ...streamers import read_infos
from ...tree import Jagged
from ...wbranch import DeclaredWhole, Described, clones_members
from ...winfo import INFOS
from .addresses import Address
from .objects import ClonedSlot
from .store import OBJECT, Slot, memory_tree

__all__ = ["Follow", "clone", "subset"]

#: Type codes whose width is the platform's, and the ones that are not.
PLATFORM = {"l": "q", "L": "Q"}


class Follow(Address):
    """A clone's address: whatever the tree it was cloned from holds for one leaf now."""

    room = None

    def __init__(self, tree: Any, column: str, sized: bool, text: bool) -> None:
        self.tree = tree
        self.column = column
        self.sized = sized
        self.text = text

    def get(self, count: int | None = None) -> Any:
        bound = self.tree._addresses.get(self.column)
        if bound is not None:
            return bound.get(count)
        value = self.tree._current(self.column)
        if value is None:
            raise ValueError(
                f"the clone reads {self.column!r} from the entry {self.tree.GetName()!r} last "
                f"read, and it has read none yet; call GetEntry on it before Fill"
            )
        if count is None:
            return value.item() if isinstance(value, np.generic) else value
        return np.asarray(value).reshape(-1)[:count]

    def put(self, value: Any) -> None:
        """Nothing: a clone's leaves are the other tree's, which ``GetEntry`` fills there."""


def _code(sample: Any) -> str:
    dtype = sample.content.dtype if isinstance(sample, Jagged) else np.asarray(sample).dtype
    return str(PLATFORM.get(dtype.char, dtype.char))


def _wanted(tree: Any) -> list[Any]:
    """The leaves a clone takes: every active one, and the counters those need."""
    leaves = tree._leaves()
    active = {leaf.column for leaf in leaves if tree._active(leaf.column)}
    active |= {leaf.counter for leaf in leaves if leaf.column in active and leaf.counter}
    return [leaf for leaf in leaves if leaf.column in active]


def _whole(leaf: Any, branch: Any) -> bool:
    """Is this a branch of one streamed object per entry, which a clone copies as its bytes?

    A split object is a branch per member and reads back member by member, so
    its leaves are copied as the columns they are; an unsplit one holds the
    whole object, and its class need not be one this process has.

    A column holding a list per entry - an :class:`~xrdroot.interp.Each` - is
    one member's values for every object of a split collection, not one streamed
    object, so ``Values`` alone does not make a branch a branch of whole
    objects: every base of every split class has such columns.
    """
    column = getattr(branch, "column", None)
    return (
        not leaf.text
        and leaf.classname != "TLeafObject"
        and isinstance(column, Values)
        and not isinstance(column, Each)
    )


def _infos(backing: Any) -> dict[str, tuple[int, int, tuple[Any, ...]]]:
    """What the file describes each class as: its checksum, version and elements."""
    source = getattr(backing, "_source", None)
    return dict(read_infos(source)) if source is not None else {}


def _object_spec(backing: Any, leaf: Any, branch: Any) -> Any:
    """The declaration a cloned branch of whole objects is written with.

    ROOT's fast clone moves such a branch's baskets without reading them, so
    the clone's file still needs the class described - and the description is
    the source file's own, which is all a reader of the clone has to go on.
    """
    classname = leaf.typename
    found = _infos(backing)
    if classname in INFOS or classname not in found:
        raise UnsupportedFeatureError(
            f"the branch {classname!r} holds whole objects of a class ROOT streams itself, "
            f"and a clone here copies only a class the file describes member by member, so "
            f"its bytes would not be written back as they were read; write the object's "
            f"members as columns of their own"
        )
    checksum, version, _elements = found[classname]
    described = tuple(Described(name, one[1], one[0], one[2]) for name, one in found.items())
    return DeclaredWhole(classname, checksum, version, described, split=0)


def _entries(branch: Any, rows: Any) -> list[bytes]:
    """These entries of a whole object's branch, the bytes its baskets hold them as."""
    found = []
    for row in rows:
        for index, low, _high in branch._spans(int(row), int(row) + 1):
            basket = branch.basket(index)
            at = basket.start_of(low, branch.leaf.offset)
            found.append(bytes(basket.data[at : basket.end_of(low)]))
    return found


def _rows(backing: Any, start: int, stop: int, selection: str | None) -> Any:
    """The entries asked for, numbered: a selection need not keep the same ones per column."""
    if not selection:
        return range(start, stop)
    return [int(row) for row in backing.arrays(["Entry$"], start, stop, cut=selection)["Entry$"]]


def _read(backing: Any, leaves: list[Any], start: int, stop: int, selection: str | None) -> Any:
    """Every column's entries, a branch of whole objects read as the bytes a fast clone copies."""
    whole = [leaf.column for leaf in leaves if _whole(leaf, backing[leaf.column])]
    plain = [leaf.column for leaf in leaves if leaf.column not in whole]
    read = backing.arrays(plain, start, stop, cut=selection or None) if plain else {}
    return _objects_read(backing, whole, read, start, stop, selection)


def _objects_read(backing: Any, whole: list[str], read: Any, start: int,
                  stop: int, selection: str | None) -> Any:  # fmt: skip
    """The columns read as values, and each branch of whole objects' entries among them."""
    if not whole:
        return read
    rows = _rows(backing, start, stop, selection)
    for column in whole:
        read[column] = _entries(backing[column], rows)
    return read


class Raw(Address):
    """A clone's address for a whole object: the bytes its source last read for one entry."""

    room = None

    def __init__(self, tree: Any, column: str) -> None:
        self.tree = tree
        self.column = column

    def get(self, count: int | None = None) -> Any:
        entry = getattr(self.tree, "_read_entry", -1)
        if entry is None or entry < 0:
            raise ValueError(
                f"the clone reads {self.column!r} from the entry {self.tree.GetName()!r} last "
                f"read, and it has read none yet; call GetEntry on it before Fill"
            )
        return _entries(self.tree._backing()[self.column], [entry])[0]

    def put(self, value: Any) -> None:
        """Nothing: a clone's leaves are the other tree's, which ``GetEntry`` fills there."""


class Streamed:
    """A clone's slot for a branch of whole objects, whose entries are already streamed.

    It answers what a :class:`~.store.Slot` answers, so a store holds it beside
    the slots of numbers, but the column it fills is the class's bytes rather
    than a value read back - and the declaration it hands the writer is the
    file's own description of the class.
    """

    kind = OBJECT
    counter = None
    size = 1
    code = ""

    #: The tree's own name for the column this branch's entries are read from.
    source: str

    def __init__(self, name: str, title: str, spec: Any, address: Address, source: str) -> None:
        self.name = self.branch = name
        self.title = title
        self.address = address
        self.source = source
        self._spec = spec
        self.pending: list[Any] = []
        self.chunks: list[Any] = []

    def __repr__(self) -> str:
        return f"<Streamed {self.name!r} of {self._spec.classname}>"

    def read(self, counts: Any = None) -> Any:
        return self.address.get()

    def taken(self, read: Any) -> Any:
        """The bytes of this branch's objects, out of the columns read in one go."""
        return read[self.source]

    def keep(self, value: Any) -> None:
        self.pending.append(value)

    def seal(self) -> None:
        if self.pending:
            self.chunks.append(list(self.pending))
            self.pending = []

    def columns(self) -> dict[str, Any]:
        self.seal()
        return {self.name: [entry for chunk in self.chunks for entry in chunk]}

    def spec(self) -> Any:
        return self._spec

    def takes(self, value: Any) -> int:
        """How many bytes of a basket one entry took: the bytes of its objects."""
        return len(value)


class Copied(Slot):
    """A leaf of numbers or text a clone copies, either a value at a time or a column at once.

    ``Fill`` reads a clone's leaves through the source tree's own addresses, so
    a clone filled entry by entry sees what ``GetEntry`` last read there; a
    clone given whole entries at once is handed each column already read.
    """

    #: The tree's own name for this leaf's column, which need not be the slot's.
    source: str

    def taken(self, read: Any) -> Any:
        return read[self.source]


class Counted(Slot):
    """A count of a split collection, which a clone copies without a leaf to copy it from.

    ROOT keeps how many objects an entry's collection holds in a branch of its
    own, and what this reader knows of that branch is the name the collection's
    member leaves carry as their ``counter`` - there is no leaf of it to make a
    slot of, yet the store must still be told the type the count is written as
    and how many values each entry has, which is what its member holds.
    """

    #: The member leaf whose values the count is: the collection's first.
    source: str

    def taken(self, read: Any) -> Any:
        rows = read[self.source]
        if isinstance(rows, Jagged):
            return np.diff(rows.offsets).astype(np.int32)
        return np.fromiter((len(np.atleast_1d(row)) for row in rows), np.int32, count=len(rows))


class CountOf(Address):
    """What a :class:`Counted` reads: the objects the entry last read holds of its member."""

    room = None

    def __init__(self, tree: Any, column: str) -> None:
        self.follower = Follow(tree, column, False, False)

    def get(self, count: int | None = None) -> Any:
        value = self.follower.get()
        if isinstance(value, (str, bytes)):  # one string is one object, however long
            return 1
        return len(np.atleast_1d(value))

    def put(self, value: Any) -> None:
        """Nothing: a clone's leaves are the other tree's, which ``GetEntry`` fills there."""


def _count_slots(slots: list[Any]) -> list[Any]:
    """A slot for each count the clone's own slots name and no leaf gives, ahead of them.

    A count comes first because a counted leaf's read asks the store how many
    values it takes, and ROOT writes the branch in that order too. Its type is
    ``Int_t``, the leaf ROOT makes of a split collection's count, and its values
    are the ones its collection's first member holds. A collection a single
    branch carries whole - :class:`~.objects.ClonedSlot` - names no count, so
    nothing is made up for it.
    """
    named = {slot.name for slot in slots}
    first: dict[str, str] = {}
    for slot in slots:
        if slot.counter and slot.counter not in named:
            first.setdefault(slot.counter, slot.source)
    counts = []
    for name, column in first.items():
        slot = Counted(name, name, "I", CountOf(_tree_of(slots, column), column), title=name)
        slot.source = column
        # The writer makes a count out of the rows it counts, so the store must not
        # offer it as a column of its own: it skips every name another slot counts.
        # A count names itself, which keeps it out of the columns without its own
        # read wanting a count in front of it.
        slot.counter = name
        counts.append(slot)
    return counts


def _tree_of(slots: list[Any], column: str) -> Any:
    """The tree a clone reads ``column`` from, whose entry last read a count is taken of."""
    for slot in slots:
        if slot.source == column:
            return slot.address.tree
    raise ValueError(f"no slot of this clone reads {column!r}, so its count has no entry")


def _slot(tree: Any, leaf: Any, backing: Any) -> Any:
    branch = backing[leaf.column]
    if _whole(leaf, branch):
        name = leaf.column if "." not in leaf.column else leaf.name
        spec = _object_spec(backing, leaf, branch)
        return Streamed(name, name, spec, Raw(tree, leaf.column), leaf.column)
    sample = branch.array(0, 0)
    code = "C" if leaf.text else _code(sample)
    follow = Follow(tree, leaf.column, leaf.vector, leaf.text)
    name = leaf.column if "." not in leaf.column else leaf.name
    slot = Copied(
        name,
        leaf.branch if "." not in leaf.branch else name,
        code,
        follow,
        size=leaf.size,
        counter=leaf.counter,
        title=leaf.title,
    )
    slot.source = leaf.column
    return slot


def _element(leaf: Any, backing: Any) -> tuple[str, int] | None:
    """The class of the split collection this leaf is the top of, and its split level.

    ROOT hangs a split ``TClonesArray`` from a branch of its own (``fType`` 3)
    whose baskets hold only how many objects each entry has, and puts every
    member of every object in a branch under it. A branch of a chain keeps no
    record of its own - its files do - so a chain's collections are not copied
    as one branch, which :func:`_members` goes on to say.
    """
    record = getattr(backing[leaf.column], "record", None)
    if record is None or not record.collection or not record.element:
        return None
    return record.element[1], record.split


def _members(top: str, element: str, leaves: dict[str, Any]) -> list[Any]:
    """The leaves of one split collection's members, in the order ROOT writes them."""
    names = [f"{top}.{member.name}" for member in clones_members(element)]
    missing = next((name for name in names if name not in leaves), None)
    if missing is not None:
        raise UnsupportedFeatureError(
            f"the branch {top!r} splits a TClonesArray of {element} and {missing!r} is not a "
            f"leaf this clone copies, so the collection could not be written back whole; "
            f"turn that branch back on, or do not clone the tree"
        )
    return [leaves[name] for name in names]


def _directory(backing: Any) -> Any:
    """The open file a clone's rows were read out of, or ``None`` for one read from memory.

    A ``TRef`` among those rows is the record its source file wrote, and a record names its
    process only by the index of a ``ProcessID`` record in that file; a tree keeps the bytes
    it read through rather than the file, so the session's own list of open files has both.
    """
    from ...session import gROOT

    source = getattr(backing, "_source", None)
    return next((file for file in gROOT.files if file._source is source), None)


def _slots(tree: Any, leaves: list[Any], backing: Any) -> tuple[list[Any], set[str]]:
    """A slot for each leaf the clone fills, and the split collections' top branches.

    Every member of every object of a split collection goes into the one branch
    ROOT writes the collection as, so those leaves get no slot of their own,
    and the collection's own top branch - whose value is its members put back
    together - is the only one of the two that is read.
    """
    elements = {leaf.column: _element(leaf, backing) for leaf in leaves}
    source = _directory(backing)
    by_column = {leaf.column: leaf for leaf in leaves}
    slots: list[Any] = []
    grouped: set[str] = set()
    inside: set[str] = set()
    for leaf in leaves:
        if leaf.column in inside:
            continue
        found = elements[leaf.column]
        if found is None:
            slots.append(_slot(tree, leaf, backing))
            continue
        element, split = found
        members = _members(leaf.column, element, by_column)
        follows = [Follow(tree, one.column, one.vector, one.text) for one in members]
        sources = [one.column for one in members]
        slots.append(ClonedSlot(leaf.column, element, split, sources, follows, source))
        grouped.add(leaf.column)
        inside |= {one.column for one in members}
    return _count_slots(slots) + slots, grouped


def clone(tree: Any, made: Any, start: int, stop: int, selection: str | None) -> Any:
    """Give ``made`` a slot for each leaf ``tree`` has, and the entries asked for."""
    backing = tree._backing()
    leaves = _wanted(tree)
    slots, grouped = _slots(tree, leaves, backing)
    for slot in slots:
        made._store.add(slot)
    if stop > start:
        wanted = [leaf for leaf in leaves if leaf.column not in grouped]
        read = _read(backing, wanted, start, stop, selection)
        count = len(next(iter(read.values()))) if read else 0
        made._store.extend({slot.name: slot.taken(read) for slot in slots}, count)
    made._changed()
    return made


def subset(backing: Any, rows: Any, name: str) -> Any:
    """The entries ``rows`` names of a tree, as a tree of their own, read from memory."""
    own = [label for label in backing.readable() if "." not in label]
    read = backing.arrays(own, entries=np.asarray(rows, dtype=np.int64))
    counted = {
        label: backing[label].leaf.count.name
        for label in own
        if backing[label].is_jagged and backing[label].leaf.count is not None
    }
    counted = {label: counter for label, counter in counted.items() if counter in read}

    def write(out: Any) -> None:
        from ...wtree import spec_of

        specs = {
            label: (_code(values), counted[label]) if label in counted else spec_of(label, values)
            for label, values in read.items()
            if label not in counted.values()
        }
        types = {counter: _code(read[counter]) for counter in counted.values()}
        tree = out.tree(name, specs, counters=types)
        tree.extend({label: read[label] for label in specs})

    return memory_tree(name, write)
