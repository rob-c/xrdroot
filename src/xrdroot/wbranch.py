"""Branches of C++ objects: ``TBranchElement`` and ``TBranchObject``, laid out as ROOT lays them.

A tree's object branch is one of a handful of shapes, each written here as
ROOT 6 writes it (``tests/data/object-branches-6.40.root`` holds one of every
kind, and each record is held to it field by field):

- a ``std::vector`` of numbers, or a whole object streamed by its class's
  streamer - a ``TBranchElement`` of ``fID`` -1 whose every entry is the
  object's bytes, its byte count and version in front, with a table of
  where each entry begins;
- an object of a class that streams itself its own way, unsplit - a
  ``TBranchObject``, every entry the class name and then the object;
- an object split member by member - a ``TBranchElement`` of ``fID`` -2
  holding no baskets of its own, a branch under it for each member (``fID``
  the member's place in the class, ``fStreamerType`` its type), and under a
  member that is itself an object a branch of ``fType`` 2 for it, with one
  branch per member of that;
- a ``std::vector`` of objects split as a collection - a ``TBranchElement``
  of ``fType`` 4 whose baskets hold how many objects each entry has, and a
  branch of ``fType`` 41 for each member, its rows that long.

The numbers in a member's baskets are big-endian, packed by
:mod:`.wpacking` when the member is a ``Double32_t`` or ``Float16_t``.
Everything here is a branch-writer: :meth:`write` puts its record into the
tree's, :meth:`leaves` names what the tree's list of leaves points at, and
:meth:`columns` hands the tree the columns whose baskets it fills.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, NamedTuple

from .buffer import MAP_OFFSET
from .errors import UnsupportedFeatureError
from .wclasses import Layout, Member, _mixed, harvested
from .wcolumns import CountColumn, MemberColumn, MemberRows, StreamedColumn, VectorColumn
from .winfo import INFOS
from .wmeasure import Measuring, version
from .wobjects import IGNORED
from .wpacking import BASIC
from .writer import BITS, WBuffer

__all__ = ["Spec", "Vector", "Whole", "Split", "Collection", "LeafRefs", "column_keys"]

#: The record versions written here: the 6.08 ``TBranchElement`` and ``TLeafElement``,
#: and the ``TBranchObject`` and ``TLeafObject`` the donors of :mod:`.winfo` have.
ELEMENT_VERSION = 10
LEAF_ELEMENT_VERSION = 1
OBJECT_VERSION = 1
LEAF_OBJECT_VERSION = 4
#: The versions of the records a leaf and a branch are built on, as :mod:`.wtree` writes them.
LEAF_VERSION = 2
BRANCH_VERSION = 12
OBJARRAY_VERSION = 3
ATTRIBUTE_VERSION = 2
#: How many basket slots a branch declares room for, as ROOT's does.
MIN_BASKETS = 10
#: What ROOT's ``TBranch`` constructor gives a branch that will hold no baskets: room
#: for a thousand entries' places, which a branch with baskets shrinks to fit.
NODE_OFFSET_LEN = 1000
#: The fill style every ``TBranch`` ROOT 6 makes carries.
FILL_STYLE = 1001
#: The bits ROOT 6.40's own branches of objects carry beyond every object's, kind by kind,
#: as the donor's do: a member or a vector of numbers, the top of a split object or of a
#: collection, a whole object, and a ``TBranchObject``.
ELEMENT_BITS, SPLIT_BITS, WHOLE_BITS, OBJECT_BITS = 0x500000, 0x520000, 0x101000, 0x408000


class ElementInfo(NamedTuple):
    """What a ``TBranchElement`` says beyond its ``TBranch``: whose member it is, and how."""

    classname: str
    parent: str
    clones: str
    checksum: int
    version: int
    fid: int
    btype: int
    stype: int
    maximum: int = 0
    #: The branch counting this one's rows - a split collection's, for its members - which
    #: is streamed before it, so this is a reference back to it.
    count: Any = None

    def write(self, buf: WBuffer, refs: LeafRefs) -> None:
        buf.string(self.classname)
        buf.string(self.parent)
        buf.string(self.clones)
        buf.u32(self.checksum)
        buf.i16(self.version)
        for value in (self.fid, self.btype, self.stype, self.maximum):
            buf.i32(value)
        refs.pointer(buf, self.count)  # fBranchCount
        buf.u32(0)  # fBranchCount2: no row here is counted twice over


class LeafRefs:
    """Where each leaf and branch landed in the tree's record, so a pointer to one can say so.

    ROOT streams an object the first time anything points at it - a counted
    leaf's ``fLeafCount`` before the counter's own branch has listed it -
    and every pointer after that is a reference back to that place. It also
    carries what every branch of the tree says alike: how many entries it
    holds, and how the file compresses.
    """

    def __init__(self, origin: int, entries: int = 0, compress: int = 0) -> None:
        self.origin = origin
        self.entries, self.compress = entries, compress
        self.places: dict[int, int] = {}

    def pointer(self, buf: WBuffer, thing: Any) -> None:
        """A pointer to ``thing``: none, a reference back, or the thing itself, streamed here."""
        if thing is None:
            buf.u32(0)
            return
        place = self.places.get(id(thing))
        if place is None:
            thing.write(buf, self)
            return
        buf.u32(self.origin + place + MAP_OFFSET)

    def ref(self, leaf: Any) -> int:
        """What the tree's list of leaves holds for ``leaf``, once it has been written."""
        return self.origin + self.places[id(leaf)] + MAP_OFFSET


class ElementLeaf:
    """A ``TLeafElement``: what one branch of an object holds, by its ``fID`` and ``fType``."""

    def __init__(
        self, name: str, title: str, lentype: int, fid: int, ftype: int, count: Any = None
    ) -> None:
        self.name, self.title, self.lentype = name, title, lentype
        self.fid, self.ftype, self.count = fid, ftype, count

    def write(self, buf: WBuffer, refs: LeafRefs) -> None:
        at = buf.tag("TLeafElement")
        refs.places[id(self)] = at
        outer = buf.start(LEAF_ELEMENT_VERSION)
        inner = buf.start(LEAF_VERSION)
        buf.named(self.name, self.title)
        buf.i32(1)  # fLen: one value, or one row of the collection counting it
        buf.i32(self.lentype)
        buf.i32(0)  # fOffset
        buf.u8(0)  # fIsRange
        buf.u8(0)  # fIsUnsigned
        refs.pointer(buf, self.count)
        buf.end(inner)
        buf.i32(self.fid)
        buf.i32(self.ftype)
        buf.end(outer)
        buf.end(at)


class ObjectLeaf:
    """A ``TLeafObject``: the leaf of a ``TBranchObject``, titled with the object's class."""

    def __init__(self, name: str, classname: str) -> None:
        self.name, self.classname = name, classname

    def write(self, buf: WBuffer, refs: LeafRefs) -> None:
        at = buf.tag("TLeafObject")
        refs.places[id(self)] = at
        outer = buf.start(LEAF_OBJECT_VERSION)
        inner = buf.start(LEAF_VERSION)
        buf.named(self.name, self.classname)
        for value in (1, 4, 0):  # fLen, fLenType - a pointer's four bytes - and fOffset
            buf.i32(value)
        buf.u8(0)  # fIsRange
        buf.u8(0)  # fIsUnsigned
        buf.u32(0)  # fLeafCount
        buf.end(inner)
        buf.u8(1)  # fVirtual: each entry names its class, as ROOT's always do here
        buf.end(outer)
        buf.end(at)


class NoBaskets:
    """What a branch holding nothing of its own reports in place of baskets."""

    tot_bytes = zip_bytes = 0
    entry_offset_len = NODE_OFFSET_LEN

    def __init__(self, basket_size: int) -> None:
        self.basket_size = basket_size
        self.seeks: list[int] = []
        self.sizes: list[int] = []
        self.starts = [0]



def streamed_size(branch: Branch, entries: int, held: bool) -> int:
    """How long ROOT streams this branch on its own: what ``Print`` counts beyond its baskets."""
    buf = Measuring(held)
    # The branch itself is streamed by its TBranch part alone, with no place of its own in
    # the buffer: a branch under it pointing at it, as a collection's members point at what
    # counts them, streams it again, whole, where the pointer is - and so ROOT counts it twice.
    tbranch(buf, branch, LeafRefs(0, entries))
    return len(buf.data)


def _objarray(buf: WBuffer, count: int) -> int:
    index = buf.start(OBJARRAY_VERSION)
    buf.tobject()
    buf.string("")
    buf.i32(count)
    buf.i32(0)  # the lower bound
    return index


def _table(buf: WBuffer, values: list[int], width: int, code: str) -> None:
    buf.u8(1)
    buf.raw(struct.pack(f">{width}{code}", *values, *([0] * (width - len(values)))))


def tbranch(buf: WBuffer, branch: Branch, refs: LeafRefs) -> None:
    """The ``TBranch`` part of a branch, its branches and leaves streamed inside it."""
    index = buf.start(version(buf, "TBranch", BRANCH_VERSION))
    buf.named(branch.name, branch.title, BITS | branch.bits)
    fill = buf.start(ATTRIBUTE_VERSION)
    buf.i16(0)  # fFillColor
    buf.i16(FILL_STYLE)
    buf.end(fill)
    _counts(buf, branch, refs)
    for kids in (branch.children, branch.own):  # fBranches, then fLeaves
        listed = _objarray(buf, len(kids))
        for kid in kids:
            refs.pointer(buf, kid)  # one already streamed is a reference back to it
        buf.end(listed)
    _baskets(buf, branch.baskets)
    buf.string("")  # fFileName
    buf.end(index)


def _counts(buf: WBuffer, branch: Branch, refs: LeafRefs) -> None:
    """A branch's sizes and counts, from its compression to how many bytes it zipped to."""
    baskets = branch.baskets
    node = isinstance(baskets, NoBaskets)
    written = len(baskets.seeks)
    for value in (refs.compress, baskets.basket_size, baskets.entry_offset_len, written):
        buf.i32(value)
    buf.i64(0 if node else refs.entries)  # fEntryNumber: a node fills nothing of its own
    if isinstance(buf, Measuring):
        buf.features()
    buf.i32(0)  # fOffset
    buf.i32(max(MIN_BASKETS, written + 1))  # fMaxBaskets
    buf.i32(branch.split)
    for count in (refs.entries, 0, baskets.tot_bytes, baskets.zip_bytes):
        buf.i64(count)  # fEntries, fFirstEntry, fTotBytes, fZipBytes


def _baskets(buf: WBuffer, baskets: Any) -> None:
    """The branch's baskets - all on file, none in here - and its tables of them."""
    written = len(baskets.seeks)
    width = max(MIN_BASKETS, written + 1)
    slots = 0 if isinstance(baskets, NoBaskets) else written + 1
    if isinstance(buf, Measuring) and not buf.held:
        slots = 0  # a tree read from a file holds none of its baskets' places at all
    held = _objarray(buf, slots)
    buf.raw(bytes(4 * slots))
    buf.end(held)
    _table(buf, baskets.sizes, width, "i")  # fBasketBytes
    _table(buf, baskets.starts, width, "q")  # fBasketEntry
    _table(buf, baskets.seeks, width, "q")  # fBasketSeek


class Branch:
    """One branch of an object, and the branches under it: a ``TBranchElement``.

    ``baskets`` is the column whose baskets it fills, or :class:`NoBaskets`
    for one that only holds the branches under it; ``own`` is its leaves.
    """

    kind = "TBranchElement"

    def __init__(
        self,
        name: str,
        title: str,
        split: int,
        baskets: Any,
        own: list[Any],
        info: ElementInfo,
        children: list[Branch] | None = None,
    ) -> None:
        self.name, self.title, self.split = name, title, split
        self.baskets, self.own, self.info = baskets, own, info
        self.children = children or []

    @property
    def bits(self) -> int:
        """The bits ROOT's own branch of this kind carries beyond every object's."""
        info = self.info
        if info.fid == TOP or info.btype == COLLECTION:
            return SPLIT_BITS
        if info.fid == WHOLE and not info.classname.startswith("vector<"):
            return WHOLE_BITS
        return ELEMENT_BITS

    def write(self, buf: WBuffer, refs: LeafRefs) -> None:
        at = buf.tag(self.kind)
        refs.places[id(self)] = at  # where the branches under it find what counts them
        index = buf.start(ELEMENT_VERSION)
        tbranch(buf, self, refs)
        self.info.write(buf, refs)
        buf.end(index)
        buf.end(at)

    def leaves(self) -> list[Any]:
        """Every leaf here and under here, in the order the tree's list has them."""
        return [*self.own, *(leaf for child in self.children for leaf in child.leaves())]

    def columns(self) -> list[Any]:
        """Every column here and under here whose baskets go on file."""
        mine = [] if isinstance(self.baskets, NoBaskets) else [self.baskets]
        return [*mine, *(column for child in self.children for column in child.columns())]


class ObjectBranch(Branch):
    """A ``TBranchObject``: whole objects of a class that streams itself its own way."""

    kind = "TBranchObject"

    @property
    def bits(self) -> int:
        return OBJECT_BITS

    def write(self, buf: WBuffer, refs: LeafRefs) -> None:
        at = buf.tag(self.kind)
        index = buf.start(OBJECT_VERSION)
        tbranch(buf, self, refs)
        buf.string(self.info.classname)
        buf.end(index)
        buf.end(at)


class CollectionBranch(Branch):
    """The top of a split collection, whose ``fMaximum`` is the most objects an entry held."""

    def write(self, buf: WBuffer, refs: LeafRefs) -> None:
        self.info = self.info._replace(maximum=self.baskets.maximum)
        super().write(buf, refs)


# -- what a tree is told a branch of objects is ----------------------------------------

#: ``TBranchElement``'s ``fID`` for the top of a split object, and for no member at all.
TOP, WHOLE = -2, -1
#: ``fType``: a member or a whole object, an object member split, a split collection
#: and one member of every object in it, and a whole object of a self-streaming class.
MEMBER, OBJECT_MEMBER, COLLECTION, COLLECTION_MEMBER, CUSTOM = 0, 2, 4, 41, -1
#: ``fStreamerType`` for a whole object, and for an object member split.
NOT_A_MEMBER, OBJECT_ANY = -1, 62
#: The version the dictionary gives every ``std::vector``.
VECTOR_CLASS_VERSION = 6
#: The C++ name of the numbers each :mod:`array` type code is.
CXX = {
    "b": "char",
    "B": "unsigned char",
    "h": "short",
    "H": "unsigned short",
    "i": "int",
    "I": "unsigned int",
    "q": "Long64_t",
    "Q": "ULong64_t",
    "f": "float",
    "d": "double",
    "?": "bool",
}


def column_keys(name: str, branch: Branch) -> list[str]:
    """What a tree calls each column a branch fills: the branch's own by its name, those
    under it by both names with a NUL between - which no branch's name holds - so two
    objects of one class cannot clash."""
    return [c.name if c is branch.baskets else f"{name}\0{c.name}" for c in branch.columns()]


def _collection_info(classname: str, clones: str, btype: int) -> ElementInfo:
    """A ``std::vector``'s element: its checksum is its name's, as ROOT 6.08's was."""
    checksum = _mixed(0, classname)
    return ElementInfo(
        classname, "", clones, checksum, VECTOR_CLASS_VERSION, WHOLE, btype, NOT_A_MEMBER
    )


@dataclass(frozen=True, kw_only=True)
class Spec:
    """What every kind of object branch is told: its split level, and its baskets' size
    when it is not the tree's - ``TTree::Branch``'s ``splitlevel`` and ``bufsize``."""

    split: int = 99
    basket_size: int | None = None

    def build(self, name: str, basket_size: int) -> Branch:  # pragma: no cover - each has one
        """The branch, and the columns under it, a tree of baskets this size writes."""
        raise NotImplementedError

    def classes(self) -> tuple[str, ...]:
        """The classes the file has to describe for this branch to be read."""
        return ("TBranchElement", "TLeafElement")

    def declared(self) -> tuple[Any, ...]:
        """The macro's classes among them, described from their declarations."""
        return ()

    def _size(self, given: int) -> int:
        return self.basket_size or given


@dataclass(frozen=True)
class Vector(Spec):
    """A ``std::vector`` of numbers per entry, of the :mod:`array` type ``code``."""

    code: str

    def build(self, name: str, basket_size: int) -> Branch:
        column = VectorColumn(name, self.code, self._size(basket_size))
        info = _collection_info(f"vector<{CXX[self.code]}>", "", MEMBER)
        leaf = ElementLeaf(name, name, 0, WHOLE, NOT_A_MEMBER)
        return Branch(name, name, self.split, column, [leaf], info)


@dataclass(frozen=True)
class Whole(Spec):
    """Whole objects of a class ROOT has, each entry already streamed to its bytes.

    ``custom`` is for a class whose streamer is its own (``TLorentzVector``),
    which ROOT marks with ``fType`` -1; ``object`` writes the ``TBranchObject``
    ROOT makes for an unsplit ``TObject`` of such a class (``TH2F``, a
    ``TClonesArray``), every entry its class's name and then the object.
    """

    classname: str
    custom: bool = False
    object: bool = False
    #: The classes it holds that its own description does not name: a clones array's.
    holds: tuple[str, ...] = ()

    def build(self, name: str, basket_size: int) -> Branch:
        column = StreamedColumn(name, self._size(basket_size))
        checksum, version, _elements = INFOS[self.classname]
        btype = CUSTOM if self.custom else MEMBER
        info = ElementInfo(self.classname, "", "", checksum, version, WHOLE, btype, NOT_A_MEMBER)
        if self.object:
            return ObjectBranch(name, name, self.split, column, [ObjectLeaf(name, self.classname)],
                                info)  # fmt: skip
        leaf = ElementLeaf(name, name, 0, WHOLE, NOT_A_MEMBER)
        return Branch(name, name, self.split, column, [leaf], info)

    def classes(self) -> tuple[str, ...]:
        kinds = ("TBranchObject", "TLeafObject") if self.object else super().classes()
        return (*kinds, *(name for name in self._every() if name not in IGNORED))

    def declared(self) -> tuple[Any, ...]:
        """The classes written without their ``TObject``, described as written."""
        return tuple(_ignoring(name) for name in self._every() if name in IGNORED)

    def _every(self) -> list[str]:
        found = _held(self.classname)
        for held in self.holds:
            found += [name for name in _held(held) if name not in found]
        return found


class Described(NamedTuple):
    """A class described otherwise than the donors describe it, as a file has to say it."""

    name: str
    version: int
    checksum: int
    described: tuple[Any, ...]

    def elements(self) -> tuple[Any, ...]:
        return self.described


def _held(classname: str) -> list[str]:
    """A class, and every class its members are objects of, as the donors describe them."""
    found = [classname]
    for kind, _name, _t, _s, _z, _a, _d, _m, typename, _x in INFOS[classname][2]:
        if kind in ("TStreamerObject", "TStreamerObjectAny") and typename in INFOS:
            found += [name for name in _held(typename) if name not in found]
    return found


def _ignoring(classname: str) -> Described:
    """``IgnoreTObjectStreamer``: the class's ``TObject`` base given no type, as ROOT says so."""
    checksum, version, elements = INFOS[classname]
    changed = tuple(
        (*e[:3], NOT_A_MEMBER, *e[4:]) if e[0] == "TStreamerBase" and e[1] == "TObject" else e
        for e in elements
    )
    return Described(classname, version, checksum, changed)


def _member_info(owner: Layout, parent: str, fid: int, btype: int, stype: int) -> ElementInfo:
    return ElementInfo(owner.name, parent, "", owner.checksum, owner.version, fid, btype, stype)


def _basic(name: str, member: Member, info: ElementInfo, split: int, size: int) -> Branch:
    """One number of a split object: a branch, its column and its leaf."""
    column = MemberColumn(name, member.stype, member.title, size)
    leaf = ElementLeaf(name, name, BASIC[member.stype][1], info.fid, member.stype)
    return Branch(name, name, split, column, [leaf], info)


def _rows(
    name: str, title: str, member: Member, info: ElementInfo, count: ElementLeaf, size: int
) -> Branch:
    """One number of every object of a split collection: a row of them per entry,
    titled with the member's name and the count's, ``fX[tracks_]``."""
    title = f"{title}[{count.name}]"
    column = MemberRows(name, member.stype, member.title, size)
    leaf = ElementLeaf(name, title, BASIC[member.stype][1], info.fid, member.stype, count)
    return Branch(name, title, 0, column, [leaf], info)


def nested(member: Member) -> Layout:
    """The class of an object member, which ROOT has to have described it."""
    found = harvested(member.typename)
    if found is None:
        raise UnsupportedFeatureError(
            f"{member.name} is a {member.typename}, a class this writer has no description "
            f"of member by member, so it cannot be split; write the branch unsplit"
        )
    return found


@dataclass(frozen=True)
class Split(Spec):
    """An object split member by member, as ``TTree::Branch`` splits one at level 1 or more.

    ``own`` says the class is a macro's, described in the file as it was
    declared rather than as ROOT's donors describe it.
    """

    layout: Layout
    own: bool = False

    def build(self, name: str, basket_size: int) -> Branch:
        size, owner = self._size(basket_size), self.layout
        prefix = name if name.endswith(".") else ""
        children = [self._member(prefix, m, at, size) for at, m in enumerate(owner.members)]
        info = _member_info(owner, "", TOP, MEMBER, NOT_A_MEMBER)
        top = ElementLeaf(name, name, 0, TOP, NOT_A_MEMBER)
        return Branch(name, name, self.split, NoBaskets(size), [top], info, children)

    def _member(self, prefix: str, member: Member, fid: int, size: int) -> Branch:
        owner = self.layout
        name = prefix + member.name
        if member.stype != OBJECT_ANY:
            info = _member_info(owner, owner.name, fid, MEMBER, member.stype)
            return _basic(name, member, info, self.split - 1, size)
        inner = nested(member)
        children = [
            _basic(f"{name}.{m.name}", m, _member_info(inner, inner.name, at, MEMBER, m.stype),
                   0, size)
            for at, m in enumerate(inner.members)
        ]  # fmt: skip
        node = _member_info(owner, owner.name, fid, OBJECT_MEMBER, OBJECT_ANY)
        return Branch(name, name, self.split - 1, NoBaskets(size), [], node, children)

    def classes(self) -> tuple[str, ...]:
        mine = () if self.own else (self.layout.name,)
        inner = [m.typename for m in self.layout.members if m.stype == OBJECT_ANY]
        return (*super().classes(), *mine, *inner)

    def declared(self) -> tuple[Any, ...]:
        return (self.layout,) if self.own else ()


def vector_of(name: str) -> str:
    """``std::vector`` of a class, spelled as ROOT spells it: ``> >`` and all."""
    return f"vector<{name}{' >' if name.endswith('>') else '>'}"


@dataclass(frozen=True)
class Collection(Spec):
    """A ``std::vector`` of objects of class ``element``, split as ROOT splits a collection."""

    element: Layout

    def build(self, name: str, basket_size: int) -> Branch:
        size, owner = self._size(basket_size), self.element
        count = ElementLeaf(f"{name}_", f"{name}_", 0, WHOLE, NOT_A_MEMBER)
        info = _collection_info(vector_of(owner.name), owner.name, COLLECTION)
        counts = CountColumn(name, size)
        top = CollectionBranch(name, f"{name}_", self.split, counts, [count], info)
        for fid, member in enumerate(owner.members):
            top.children += self._rows(top, member, fid, count, size)
        return top

    def _rows(
        self, top: Branch, member: Member, fid: int, count: ElementLeaf, size: int
    ) -> list[Branch]:
        owner, name = self.element, f"{top.name}.{member.name}"
        if member.stype != OBJECT_ANY:
            info = _member_info(owner, owner.name, fid, COLLECTION_MEMBER, member.stype)
            return [_rows(name, member.name, member, info._replace(count=top), count, size)]
        inner = nested(member)
        return [
            _rows(f"{name}.{m.name}", m.name, m,
                  _member_info(inner, owner.name, at, COLLECTION_MEMBER, m.stype)._replace(
                      count=top), count, size)
            for at, m in enumerate(inner.members)
        ]  # fmt: skip

    def classes(self) -> tuple[str, ...]:
        inner = [m.typename for m in self.element.members if m.stype == OBJECT_ANY]
        return (*super().classes(), self.element.name, *inner)
