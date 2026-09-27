"""Reading the three classes a tree is made of: TTree, TBranch, TLeaf.

Every one of them is versioned, and the version decides which fields are
there - a file from 2008 and a file from last week are both valid and are not
the same bytes. The conditionals below are that history, and they are the
whole reason a reader can be short: after the fields it wants, it jumps to the
byte count the record started with and never has to know the rest.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .buffer import Buffer
from .errors import FormatError, UnsupportedFeatureError

if TYPE_CHECKING:
    from .file import Source
    from .streamers import Member
    from .tree import Basket

__all__ = ["TREE_CLASSES", "FriendRecord", "read_tree"]

#: Key classes this reader will open as a tree.
TREE_CLASSES = ("TTree", "TNtuple", "TNtupleD")

#: ROOT leaf class to the Python name, ``array`` type code and item size.
LEAF_TYPES = {
    "TLeafO": ("bool", "b", 1),
    "TLeafB": ("int8", "b", 1),
    "TLeafS": ("int16", "h", 2),
    "TLeafI": ("int32", "i", 4),
    "TLeafL": ("int64", "q", 8),
    "TLeafG": ("int64", "q", 8),
    "TLeafF": ("float32", "f", 4),
    "TLeafD": ("float64", "d", 8),
    "TLeafC": ("str", "", 0),
}

#: Leaves holding floats squeezed into fewer bytes, by the recipe in the title.
PACKED_LEAVES = ("TLeafF16", "TLeafD32")

#: Leaves that are real and are not plain numbers behind a name.
LEAF_REASONS = {
    "TLeafElement": "a split C++ object, which needs the file's streamer information",
    "TLeafObject": "a whole object per entry, which needs the file's streamer information",
}

UNSIGNED = {"b": "B", "h": "H", "i": "I", "q": "Q"}

#: The integer leaves, by how their ``fMinimum`` and ``fMaximum`` are read:
#: the largest value a counter ever held is what a copy of it has to say too.
LIMITS = {"TLeafB": "i8", "TLeafS": "i16", "TLeafI": "i32", "TLeafL": "i64"}


class LeafRecord:
    """One column's description, as the file states it."""

    __slots__ = (
        "classname",
        "name",
        "title",
        "length",
        "etype",
        "offset",
        "unsigned",
        "count",
        "ltype",
        "maximum",
    )

    def __init__(self, classname: str) -> None:
        self.classname = classname
        self.name = self.title = ""
        self.length = 1
        self.etype = self.offset = 0
        self.unsigned = False
        self.count: LeafRecord | None = None
        #: The streamer type a ``TLeafElement`` names; ``-1`` for a whole object.
        self.ltype = -1
        #: The largest value an integer leaf recorded - what a counter says
        #: its longest row was - and zero for any other leaf.
        self.maximum = 0

    def __repr__(self) -> str:
        return f"<LeafRecord {self.name!r} of class {self.classname}>"

    @property
    def typename(self) -> str | None:
        """``'float32'``, ``'str'``, or ``None`` if this reader cannot decode it."""
        found = LEAF_TYPES.get(self.classname)
        if found is None:
            return None
        name = found[0]
        return f"u{name}" if self.unsigned and name.startswith("int") else name

    @property
    def typecode(self) -> str:
        """The :mod:`array` type code the values come back in."""
        _name, code, _size = LEAF_TYPES[self.classname]
        return UNSIGNED[code] if self.unsigned and code in UNSIGNED else code

    @property
    def itemsize(self) -> int:
        return LEAF_TYPES[self.classname][2]

    @property
    def reason(self) -> str:
        """Why this column cannot be read, for a message that names it."""
        unknown = f"{self.classname}, which is not a kind of leaf this reader knows"
        return LEAF_REASONS.get(self.classname, unknown)


class BranchRecord:
    """Where a branch keeps its baskets, and which columns are in them."""

    __slots__ = (
        "name",
        "title",
        "classname",
        "entry_offset_len",
        "entries",
        "first_entry",
        "basket_bytes",
        "basket_entry",
        "basket_seek",
        "leaves",
        "branches",
        "baskets",
        "streamed",
        "whole",
        "basket_size",
        "tot_bytes",
        "zip_bytes",
        "collection",
        "file_name",
    )

    def __init__(self) -> None:
        self.name = self.title = ""
        #: The size ROOT aimed each basket at, and the bytes of the branch's
        #: baskets before and after compression - what ``TTree::Print`` says.
        self.basket_size = 0
        self.tot_bytes = self.zip_bytes = 0
        #: The C++ class a ``TBranchElement`` belongs to; empty for a plain branch.
        self.classname = ""
        self.entry_offset_len = 0
        self.entries = self.first_entry = 0
        self.basket_bytes: list[int] = []
        self.basket_entry: list[int] = []
        self.basket_seek: list[int] = []
        self.leaves: list[LeafRecord] = []
        self.branches: list[BranchRecord] = []
        #: Baskets written into this record rather than out to one of their
        #: own, which a tree too small to have flushed never does.
        self.baskets: list[Basket | None] = []
        #: Did the class stream itself, record and all, rather than the file's
        #: streamer information writing out its members bare?
        self.streamed = False
        #: Does this branch hold the whole class it names, rather than one
        #: member of one? ROOT says so by giving it no member to point at.
        self.whole = False
        #: Is this the branch a split collection's members hang from, whose
        #: baskets hold how many objects each entry has and nothing else?
        self.collection = False
        #: The file the baskets were written to, when ROOT was told to put
        #: them in one of their own; empty for the file the tree is in.
        self.file_name = ""

    def __repr__(self) -> str:
        return f"<BranchRecord {self.name!r} with {len(self.basket_seek)} baskets>"

    def walk(self) -> Any:
        """This branch and every branch under it, depth first."""
        yield self
        for child in self.branches:
            yield from child.walk()


def read_leaf(buf: Buffer, classname: str) -> LeafRecord:
    """A ``TLeaf`` of any class: the part every leaf shares, then a jump."""
    _version, end = buf.header()
    _base, inner = buf.header()
    leaf = LeafRecord(classname)
    leaf.name, leaf.title = buf.named()
    leaf.length = buf.i32() or 1
    leaf.etype = buf.i32()
    leaf.offset = buf.i32()
    buf.bool()  # whether a range was recorded, which only matters for writing
    leaf.unsigned = buf.bool()
    count = buf.any(CLASSES)
    if isinstance(count, LeafRecord):
        leaf.count = count
    buf.resume(inner)
    if classname == "TLeafElement":
        buf.i32()  # which member of the class this is, which its name says too
        leaf.ltype = buf.i32()
    elif classname in LIMITS:
        read = getattr(buf, LIMITS[classname])
        read()  # fMinimum, which nothing needs
        leaf.maximum = read()
    buf.resume(end)
    return leaf


def read_branch(buf: Buffer) -> BranchRecord:
    """A ``TBranch``: the layout ROOT 5 settled on, or the ROOT 4 one before it.

    The older one differs in its arithmetic rather than its shape: counters
    that later became 64-bit are doubles or 32-bit integers, and the seek
    points are 32-bit unless the file grew past two gigabytes, which the
    marker in front of them says.
    """
    version, end = buf.header()
    if version < 5:
        raise UnsupportedFeatureError(
            f"this file has a TBranch of version {version}, older than any this reader follows, "
            f"which kept no sizes for its baskets; copy it forward with hadd from any later "
            f"ROOT and it will open"
        )
    modern = version >= 10
    branch = BranchRecord()
    branch.name, branch.title = buf.named()
    if version == 5:
        return _ancient_branch(buf, branch, end)
    write_basket, max_baskets = _branch_header(buf, branch, version, modern)
    _branch_contents(buf, branch)
    _basket_tables(buf, branch, modern, max_baskets, write_basket)
    branch.file_name = buf.string()
    _inline_basket_bounds(branch)
    buf.resume(end)
    return branch


def _ancient_branch(buf: Buffer, branch: BranchRecord, end: int | None) -> BranchRecord:
    """A ``TBranch`` of version 5, as ROOT 2 wrote it and ``TBranch::Streamer`` reads it.

    The counters come in another order, the offset of the branch after its
    byte counts, and each table of the baskets carries its own length: the
    first entries by how many were written, the sizes as a counted array,
    and the seek points as a count ROOT ignores and then one per basket slot.
    """
    buf.i32(), buf.i32()  # compression and target basket size
    branch.entry_offset_len = buf.i32()
    max_baskets, written = buf.i32(), buf.i32()
    buf.i32()  # the entry the next basket would start at
    branch.entries = int(buf.f64())
    buf.f64(), buf.f64(), buf.i32()  # bytes both ways, and the offset in the parent
    _branch_contents(buf, branch)
    branch.basket_entry = buf.i32s(buf.i32())[: written + 1]
    branch.basket_bytes = buf.i32s(buf.i32())[:written]
    buf.i32()
    branch.basket_seek = buf.i32s(max_baskets)[:written]
    branch.file_name = buf.string()
    _inline_basket_bounds(branch)
    buf.resume(end)
    return branch


def _branch_header(
    buf: Buffer, branch: BranchRecord, version: int, modern: bool
) -> tuple[int, int]:
    if version > 7:
        buf.skip_record()  # TAttFill
    buf.i32()  # compression
    branch.basket_size = buf.i32()
    branch.entry_offset_len = buf.i32()
    write_basket = buf.i32()
    buf.i64() if modern else buf.i32()
    if version >= 13:
        buf.skip_record()  # TIOFeatures
    buf.i32()
    max_baskets = buf.i32()
    if version > 6:
        buf.i32()
    _branch_counts(buf, branch, version, modern)
    return write_basket, max_baskets


def _branch_counts(buf: Buffer, branch: BranchRecord, version: int, modern: bool) -> None:
    if modern:
        branch.entries = buf.i64()
        if version >= 11:
            branch.first_entry = buf.i64()
        branch.tot_bytes, branch.zip_bytes = buf.i64(), buf.i64()
    else:
        branch.entries = int(buf.f64())
        branch.tot_bytes, branch.zip_bytes = int(buf.f64()), int(buf.f64())


def _branch_contents(buf: Buffer, branch: BranchRecord) -> None:
    branch.branches = [item for item in buf.objarray(CLASSES) if isinstance(item, BranchRecord)]
    branch.leaves = [item for item in buf.objarray(CLASSES) if isinstance(item, LeafRecord)]
    branch.baskets = _held(buf.objarray(CLASSES))


def _basket_tables(
    buf: Buffer, branch: BranchRecord, modern: bool, maximum: int, written: int
) -> None:
    buf.u8()
    branch.basket_bytes = buf.i32s(maximum)[:written]
    buf.u8()
    if modern:
        branch.basket_entry = buf.i64s(maximum)[: written + 1]
        buf.u8()
        branch.basket_seek = buf.i64s(maximum)[:written]
        return
    branch.basket_entry = buf.i32s(maximum)[: written + 1]
    wide = buf.u8() == 2
    seeks = buf.i64s(maximum) if wide else buf.i32s(maximum)
    branch.basket_seek = seeks[:written]


def _inline_basket_bounds(branch: BranchRecord) -> None:
    """Where the held baskets' entries start: all of them, or the one after the flushed.

    A tree saved while its last basket was still being filled - ``Write``
    without a ``FlushBaskets`` - keeps that basket in the branch at the slot
    after the ones written out, and its entries follow theirs.
    """
    written = len(branch.basket_seek)
    after = branch.baskets[written:] if not written else branch.baskets[written : written + 1]
    bounds = branch.basket_entry[: written + 1] if written else [0]
    held: list[Basket | None] = []
    for basket in after:
        if basket is None:
            break
        held.append(basket)
        bounds.append(bounds[-1] + basket.nevbuf)
    branch.baskets = branch.baskets[:written] + held
    if held:
        branch.basket_entry = bounds


def read_branch_element(buf: Buffer) -> BranchRecord:
    """A ``TBranchElement``: a branch, plus the C++ class it was split out of.

    The class name is what tells a top-level branch apart from a member of
    something bigger - ``vector<float>`` says everything about how to read the
    column, where ``Event`` says to look at the branches under it instead.
    """
    version, end = buf.header()
    branch = read_branch(buf)
    branch.classname = buf.string()
    if version > 1:
        buf.string(), buf.string()  # the parent class, and the TClonesArray class
        buf.u32()  # the checksum of the class this was written from
    buf.u16() if version >= 10 else buf.u32()  # that class's version
    branch.whole = buf.i32() < 0  # which member this is, and -1 for none of them
    kind = buf.i32()  # ROOT calls this fType, and -1 is the whole object
    branch.streamed = kind < 0
    branch.collection = kind in SPLIT_COLLECTIONS
    buf.resume(end)
    return branch


#: The ``fType`` of the branch a split ``TClonesArray`` or STL collection
#: hangs from, whose own baskets hold only how many objects each entry has.
SPLIT_COLLECTIONS = (3, 4)


def read_branch_object(buf: Buffer) -> BranchRecord:
    """A ``TBranchObject``: a branch of whole objects, named a class at a time.

    This is how ROOT wrote an object before ``TBranchElement``, and the file
    still holds the layout for it: the class name is here, and every entry is
    that class streaming itself.
    """
    _version, end = buf.header()
    branch = read_branch(buf)
    branch.classname = buf.string()
    branch.streamed = True
    buf.resume(end)
    return branch


def read_derived_branch(buf: Buffer) -> BranchRecord:
    """A ``TBranchObject``, ``TBranchClones`` or ``TBranchSTL``.

    The C++ part on top is what this reader cannot follow; the branch part
    underneath is what makes the column appear in the tree at all, with a name
    and a reason instead of silently not being there.
    """
    _version, end = buf.header()
    branch = read_branch(buf)
    buf.resume(end)
    return branch


def read_basket(buf: Buffer) -> Basket | None:
    """A ``TBasket`` written into the branch instead of a record of its own."""
    from .tree import Basket

    return Basket.inline(buf)


def _held(items: list[Any]) -> list[Basket | None]:
    """The baskets a branch record carries, each in its own slot.

    A slot whose basket was flushed to a record of its own is ``None``, and
    the branch's seek table is what says where that one is; what is held is
    either every basket of a tree never flushed, or the last one of a tree
    saved with a basket still being filled.
    """
    from .tree import Basket

    held: list[Basket | None] = [item if isinstance(item, Basket) else None for item in items]
    while held and held[-1] is None:
        held.pop()
    return held


def _leaf_class(classname: str) -> Callable[[Buffer], LeafRecord]:
    def read(buf: Buffer) -> LeafRecord:
        return read_leaf(buf, classname)

    return read


#: What :meth:`Buffer.any` knows how to build; anything else is stepped over.
CLASSES: dict[str, Any] = {
    name: _leaf_class(name) for name in (*LEAF_TYPES, *LEAF_REASONS, *PACKED_LEAVES)
}
CLASSES["TBasket"] = read_basket
CLASSES["TBranch"] = read_branch
CLASSES["TBranchElement"] = read_branch_element
CLASSES["TBranchObject"] = read_branch_object
for _derived in ("TBranchClones", "TBranchSTL"):
    CLASSES[_derived] = read_derived_branch


def read_tree(buf: Buffer, source: Source, name: str, classname: str = "TTree") -> Any:
    """Read a ``TTree`` record and hand back something that can be iterated."""
    from .tree import TTree

    _tuple_header(buf, classname)
    version, end = buf.header()
    title = buf.named()[1]
    if version < 5:
        entries = _ancient_tree_fields(buf)
    elif DESCRIBED_TREES[0] <= version <= DESCRIBED_TREES[1]:
        entries = _described_tree_fields(buf, source, version)
    else:
        entries = _tree_fields(buf, version, version > 5)

    branches = [b for b in buf.objarray(CLASSES) if isinstance(b, BranchRecord)]
    friends = _tree_friends(buf, version)
    buf.resume(end)
    return TTree(name, title, entries, branches, source, friends)


#: How long ROOT 2's ``TAttLine``, ``TAttFill`` and ``TAttMarker`` are after
#: their version, which is all they had in front of them: three shorts, two,
#: and two and a float.
ANCIENT_ATTRIBUTES = (6, 4, 8)


def _ancient_tree_fields(buf: Buffer) -> int:
    """The fields of a tree ROOT 2 or 3 wrote, before streamer information existed.

    ``TTree::Streamer`` still reads these versions by hand, in this order:
    the three attribute records, the scan field and two limits as 32-bit
    integers, the entries and the bytes written as doubles, then the autosave
    size and the estimate - and then, as later, the branches.
    """
    for size in ANCIENT_ATTRIBUTES:
        _version, end = buf.header()
        buf.take(size) if end is None else buf.resume(end)
    buf.i32(), buf.i32(), buf.i32()  # the scan field, the loop and memory limits
    entries = int(buf.f64())
    buf.f64(), buf.f64()  # bytes before and after compression
    buf.i32(), buf.i32()  # the autosave size and the estimate
    return entries


#: The ``TTree`` versions, from ROOT 3.02 to 5.08, whose fixed fields changed
#: from release to release - ``fWeight`` arriving, the counters staying
#: doubles for years after ROOT 4 - and which are read the way ROOT itself
#: reads them, by the file's own description of the class.
DESCRIBED_TREES = (6, 15)

#: How a fundamental member of such a tree is read, by its streamer type.
TREE_MEMBER_READS = {3: "i32", 6: "i32", 8: "f64", 13: "u32", 16: "i64", 17: "i64"}


def _described_tree_fields(buf: Buffer, source: Source | None, version: int) -> int:
    """The fields in front of a middle-aged tree's branches, as its file lists them.

    ROOT reads a ``TTree`` of these versions member by member from the
    streamer information the file carries, so this does the same and stops
    at ``fBranches``: the bases after ``TNamed`` are records stepped over,
    and each number is read at the width the file declares it.
    """
    members = source.streamers().get("TTree", {}) if source is not None else {}
    if "fBranches" not in members:
        raise UnsupportedFeatureError(
            f"this tree is TTree version {version}, whose fields changed from one ROOT "
            f"release to the next, and its file does not describe the TTree class to say "
            f"which of them it has; hadd it forward and it will open"
        )
    entries = 0
    for member in itertools.takewhile(lambda m: m.name != "fBranches", members.values()):
        value = _tree_member(buf, member, version)
        if member.name == "fEntries":
            entries = int(value)
    return entries


def _tree_member(buf: Buffer, member: Member, version: int) -> float:
    """One member ahead of a described tree's branches: a base, or a number."""
    if member.typename == "BASE":
        if member.name != "TNamed":  # which the caller has already read
            buf.skip_record()
        return 0
    read = TREE_MEMBER_READS.get(member.stype)
    if read is None:
        raise UnsupportedFeatureError(
            f"this file describes TTree version {version} with a member {member.name} of "
            f"type {member.typename}, which is not a number this reader expected there"
        )
    value: float = getattr(buf, read)()
    return value


def _tuple_header(buf: Buffer, classname: str) -> None:
    if classname != "TTree":
        buf.header()


def _tree_fields(buf: Buffer, version: int, modern: bool) -> int:
    for _ in range(3):
        buf.skip_record()
    entries = buf.i64() if modern else int(buf.f64())
    _tree_counters(buf, version, modern)
    clusters = buf.i32() if version >= 19 else 0
    _tree_limits(buf, version, modern)
    _tree_clusters(buf, version, clusters)
    return entries


def _tree_counters(buf: Buffer, version: int, modern: bool) -> None:
    for _ in range(3):
        buf.i64() if modern else buf.f64()
    if version >= 18:
        buf.i64()
    if version >= 16:
        buf.f64()
    buf.i32(), buf.i32(), buf.i32()
    if version >= 17:
        buf.i32()


def _tree_limits(buf: Buffer, version: int, modern: bool) -> None:
    if modern:
        buf.i64()
    for _ in range(3):
        buf.i64() if modern else buf.i32()
    if version >= 18:
        buf.i64()
    buf.i64() if modern else buf.i32()


def _tree_clusters(buf: Buffer, version: int, clusters: int) -> None:
    if version >= 19:
        buf.u8()
        buf.i64s(clusters)
        buf.u8()
        buf.i64s(clusters)
    if version >= 20:
        buf.skip_record()


class FriendRecord:
    """A ``TFriendElement``: a tree another tree was told to read beside itself.

    ROOT writes one into the tree for every ``AddFriend`` it was given, and
    the tree is found again by what it says: what the friend is called here,
    which tree it is, and which file that tree is in - or nothing, for a
    friend in the same file as the tree it befriends.
    """

    __slots__ = ("alias", "tree_name", "file_name")

    def __init__(self, alias: str, tree_name: str, file_name: str) -> None:
        #: The name the friend's columns are asked for under, ``alias.branch``.
        self.alias = alias
        #: The friend's own name, in its own file.
        self.tree_name = tree_name
        #: The file it is in as the writer knew it, or empty for this file.
        self.file_name = file_name

    def __repr__(self) -> str:
        where = f" in {self.file_name!r}" if self.file_name else ""
        return f"<FriendRecord {self.alias!r}: {self.tree_name!r}{where}>"


def read_friend(buf: Buffer) -> FriendRecord:
    """A ``TFriendElement``: its name and title, then the tree's own name."""
    _version, end = buf.header()
    alias, file_name = buf.named()
    tree_name = buf.string() or alias
    buf.resume(end)
    return FriendRecord(alias, tree_name, file_name)


def _friend_list(buf: Buffer) -> list[Any]:
    return buf.tlist({"TFriendElement": read_friend})


#: What the pointer to a tree's friends can point at.
FRIENDS: dict[str, Any] = {"TList": _friend_list}


def _tree_friends(buf: Buffer, version: int) -> list[FriendRecord]:
    """The friends a tree keeps, from the members written after its branches.

    Between the branches and the friends are the leaves, the aliases, and
    the index a tree may have been sorted by, all stepped over. That layout
    holds from ``TTree`` version 16 on; an older tree has no friends read,
    and neither does one whose tail does not read the way the layout says,
    since the columns are there either way and a friend list is not worth
    refusing them for.
    """
    if version < 16:
        return []
    try:
        buf.skip_record()  # fLeaves, which the branches already hold
        buf.any({})  # fAliases
        buf.take(8 * buf.i32())  # fIndexValues
        buf.take(4 * buf.i32())  # fIndex
        buf.any({})  # fTreeIndex
        friends = buf.any(FRIENDS)
    except FormatError:
        return []
    if not isinstance(friends, list):
        return []
    return [friend for friend in friends if isinstance(friend, FriendRecord)]
