"""Branches of objects, filled from the object a branch was given as ``TTree::Bronch`` makes them.

``T->Branch("v3", &v, 32000, 1)`` hands a tree an object, and what ROOT
makes of it depends on the object's class and the split level:

- a class of a macro's, or a GenVector four-vector, split above level 0 -
  a branch per member (:class:`~xrdroot.wbranch.Split`);
- a ``std::vector`` of four-vectors, split - a collection, a branch per
  member of every object in it (:class:`~xrdroot.wbranch.Collection`);
- a ``TLorentzVector`` - which cannot be split, as ROOT warns - whole;
- a histogram, unsplit: whole, a ``TBranchElement`` for a class that
  streams itself by its description (``TH1F``) and a ``TBranchObject`` for
  one with a streamer of its own (``TH2F``, ``TProfile``);
- a ``TClonesArray``, unsplit - a ``TBranchObject``, its objects written
  member by member when ``BypassStreamer`` asked for that.

Each kind is a :class:`Kind`: the branch it makes, what one entry of it is,
and the columns many entries make. An :class:`ObjectSlot` is the
:class:`~.store.Store` slot that reads the object at every ``Fill``.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...tree import Jagged
from ...wbranch import (
    COLLECTION,
    NOT_A_MEMBER,
    WHOLE,
    Branch,
    Collection,
    ElementLeaf,
    Spec,
    Split,
    SplitClones,
    Whole,
    _collection_info,
    clones_members,
    column_keys,
    vector_of,
)
from ...wclasses import declared, harvested
from ...wcolumns import StreamedColumn
from ...wobjects import named, stream, stream_clones
from ...writer import WBuffer
from .store import OBJECT

__all__ = ["ObjectSlot", "object_slot"]

#: GenVector's Cartesian four-vector, the one class of it this writer splits.
LORENTZ = "ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >"
#: Histogram classes ROOT streams by their description, and so writes as a TBranchElement.
DESCRIBED = ("TH1C", "TH1S", "TH1I", "TH1F", "TH1D")
#: A record's byte count and version: what an object streamed by its class has in front.
RECORD_HEAD = 6


def _held(address: Any) -> Any:
    """The object a branch was given: through a pointer's cell, or the object itself."""
    inner = getattr(address, "value", address)
    return address if isinstance(inner, (int, float)) and inner == 0 else inner


class Kind:
    """One kind of object: the branch it makes, one entry of it, many entries' columns."""

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:  # pragma: no cover
        raise NotImplementedError

    def entry(self, obj: Any) -> Any:  # pragma: no cover - every kind has one
        raise NotImplementedError

    def columns(self, keys: list[str], entries: list[Any]) -> dict[str, Any]:
        """A column per key from entries that are each a tuple of numbers, one per key."""
        table = np.asarray(entries, dtype=np.float64).reshape(len(entries), len(keys))
        return {key: table[:, at] for at, key in enumerate(keys)}


class Macro(Kind):
    """An object of a class a macro declared, split member by member."""

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        return Split(declared(type(obj)), own=True, split=split, basket_size=bufsize)

    def entry(self, obj: Any) -> Any:
        return tuple(getattr(obj, _attribute(name)) for name, *_ in obj._cxx_layout_[2])


def _attribute(name: str) -> str:
    """A C++ member's name as the translation made it a Python attribute."""
    from ...pycxx.symbols import member_name

    return member_name(name)


class Lorentz(Kind):
    """A GenVector ``XYZTVector``, split into its coordinates."""

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        layout = harvested(LORENTZ)
        assert layout is not None  # harvested from the 6.40 donor, see winfo
        return Split(layout, split=max(split, 1), basket_size=bufsize)

    def entry(self, obj: Any) -> Any:
        return tuple(obj._cartesian())


class Lorentzes(Kind):
    """A ``std::vector`` of ``XYZTVector``, split as a collection of their coordinates."""

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        layout = harvested(LORENTZ)
        assert layout is not None
        return Collection(layout, split=max(split, 1), basket_size=bufsize)

    def entry(self, obj: Any) -> Any:
        rows = np.asarray([vector._cartesian() for vector in obj], dtype=np.float64)
        return rows.reshape(len(rows), 4)

    def columns(self, keys: list[str], entries: list[Any]) -> dict[str, Any]:
        counts = np.asarray([len(rows) for rows in entries], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        flat = np.concatenate(entries) if entries else np.zeros((0, 4))
        columns: dict[str, Any] = {keys[0]: counts}
        for at, key in enumerate(keys[1:]):
            columns[key] = Jagged(np.ascontiguousarray(flat[:, at]), offsets)
        return columns


class _Whole(Kind):
    """Objects streamed whole, one entry the bytes of one."""

    def columns(self, keys: list[str], entries: list[Any]) -> dict[str, Any]:
        return {keys[0]: list(entries)}


class MacroWhole(_Whole):
    """An object of a class a macro declared that this writer does not split - one given split
    level 0, or with bases, strings, containers or objects among its members - streamed whole
    each entry by its class's description, as ROOT streams an object it does not split."""

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        from ...wbranch import DeclaredWhole, Described
        from ...wdobjects import write_members
        from ..core.declared import payload_of

        made = payload_of(obj)
        assert made is not None  # a macro's class is what made this kind
        catalogue = made.catalogue
        # One entry written aside, and every class reached described: what the file must say.
        # The object a branch is given is not always whole yet - a pointer it promises is
        # never null set only before the first Fill - so what it cannot write yet is left.
        with contextlib.suppress(Exception):
            write_members(WBuffer(), made.instance, catalogue)
        for name in list(catalogue.declarations):
            catalogue.info(name)
        _describe_pointed_at(catalogue)
        checksum, version, _elements = catalogue.info(made.classname)
        described = tuple(Described(name, info[1], info[0], info[2])
                          for name, info in catalogue.entries().items())  # fmt: skip
        own, _sum = catalogue.version_of(made.classname)
        return DeclaredWhole(made.classname, checksum, version if own else -1, described,
                             tuple(catalogue.used), split=0, basket_size=bufsize)  # fmt: skip

    def entry(self, obj: Any) -> bytes:
        from ...wdobjects import write_members
        from ..core.declared import payload_of

        made = payload_of(obj)
        assert made is not None
        buf = WBuffer()
        write_members(buf, made.instance, made.catalogue)
        return bytes(buf.data)


def _describe_pointed_at(catalogue: Any) -> None:
    """Describe every class the program declares that derives from one described already.

    A pointer member's class is only known when an entry holds one - ``BaseABC *fABC`` may
    point at a ``Derived`` - and ROOT's streamer record names every class it wrote, so a file
    that left the derived class out cannot be laid out by a reader.
    """
    from ...pycxx.runtime.namespace import DECLARED
    from ..core.declared import declaration_of, declared_class

    described = set(catalogue.declarations)
    growing = True
    while growing:
        growing = False
        for name, cls in list(DECLARED.items()):
            if name in described or not declared_class(cls):
                continue
            bases = {base.split("::")[-1] for base in declaration_of(cls, name).bases}
            if bases & described:
                described.add(name)
                catalogue.declarations[name] = declaration_of(cls, name)
                catalogue.info(name)
                growing = True


def _splittable(obj: Any, split: int) -> bool:
    """Can this writer split an object of a macro's class: a class of numbers alone, standing
    on no base, at a split level above 0?"""
    if split < 1:
        return False
    try:
        declared(type(obj))
    except UnsupportedFeatureError:
        return False
    return True


class Lorentz4(_Whole):
    """A ``TLorentzVector``, which streams itself and so is never split."""

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        if split > 0:
            print(
                "Warning in <TTree::Bronch>: TLorentzVector cannot be split, resetting "
                "splitlevel to 0",
                file=sys.stderr,
            )
        return Whole("TLorentzVector", custom=True, split=0, basket_size=bufsize)

    def entry(self, obj: Any) -> bytes:
        p = obj.Vect()
        coordinates = {"fX": p.X(), "fY": p.Y(), "fZ": p.Z()}
        members = {"fP": coordinates.__getitem__, "fE": obj.E()}
        buf = WBuffer()
        stream(buf, "TLorentzVector", members.__getitem__)
        return bytes(buf.data)


class Histogram(_Whole):
    """A histogram, unsplit: as its key's payload would be, its class named for a TBranchObject."""

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        classname = obj.ClassName()
        own = classname not in DESCRIBED
        return Whole(classname, object=own, split=0, basket_size=bufsize)

    def entry(self, obj: Any) -> bytes:
        from ...writer import _payload
        from ..core.wrapping import unwrap

        classname, payload, _used = _payload(unwrap(obj))
        if classname in DESCRIBED:
            # A TBranchElement streams its object's members by the class's description,
            # without the byte count and version the class's own streamer puts round them.
            return payload[RECORD_HEAD:]
        return named(classname, payload)


class Clones(_Whole):
    """A ``TClonesArray``, unsplit: its objects whole, or member by member when bypassing."""

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        return Whole(
            "TClonesArray", object=True, holds=(obj._class,), split=0, basket_size=bufsize
        )

    def entry(self, obj: Any) -> bytes:
        getters = [_members_of(item) for item in obj]
        return named("TClonesArray", stream_clones(obj._class, getters, obj._bypass))


class SplitArray(Kind):
    """A ``TClonesArray`` split: how many objects each entry holds, and a row per member."""

    def __init__(self, element: str) -> None:
        self.members = clones_members(element)

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        return SplitClones(obj._class, split=split, basket_size=bufsize)

    def entry(self, obj: Any) -> Any:
        return [tuple(_member(item, m.name) for m in self.members) for item in obj]

    def columns(self, keys: list[str], entries: list[Any]) -> dict[str, Any]:
        counts = np.asarray([len(rows) for rows in entries], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        columns: dict[str, Any] = {keys[0]: counts}
        for at, (key, member) in enumerate(zip(keys[1:], self.members, strict=True)):
            rows = [[values[at] for values in entry] for entry in entries]
            columns[key] = rows if member.stype in KEPT_ROWS else _jagged(rows, offsets)
        return columns


def _jagged(rows: list[list[Any]], offsets: Any) -> Jagged:
    """Rows of numbers, one per entry, as one column."""
    return Jagged(np.asarray([value for row in rows for value in row], dtype=np.float64), offsets)


#: ``fStreamerType`` of a ``TString`` member, whose rows are lists of strings.
TSTRING = 65
#: ``fStreamerType`` (``kObject``) of a member that is a whole object of its own for every
#: object of the collection - ``TRef`` in ``roottest``'s ``tref_test_pid`` - whose rows are
#: that object, streamed by its class rather than packed as a number.
OBJECT_TYPE = 61
#: The member types whose rows are not numbers: they go to the writer as the objects or the
#: strings they are, one list per entry, for a column that streams each of them.
KEPT_ROWS = (TSTRING, OBJECT_TYPE)
#: ``TObject::kIsOnHeap | kNotDeleted``, which a split ``TClonesArray`` leaves out of ``fBits``.
HEAP_BITS = 0x03000000


def _member(obj: Any, name: str) -> Any:
    """One member of an object a ``TClonesArray`` holds, by ROOT's name for it."""
    if name == "fUniqueID":
        return int(obj.GetUniqueID())
    if name == "fBits":
        return int(obj._bitword()) & ~HEAP_BITS
    getter = getattr(obj, GETTERS.get(name, ""), None)
    return getter() if callable(getter) else _members_of(obj)(name)


#: The members ROOT's own classes hand back through a method rather than keep as attributes.
GETTERS = {"fName": "GetName", "fTitle": "GetTitle", "fString": "GetString"}


def _members_of(obj: Any) -> Any:
    """How a drawing object's members are read: its own table of them, by ROOT's names."""
    members = getattr(obj, "members", None)
    return members.__getitem__ if members is not None else (lambda name: getattr(obj, name))


def _kind(obj: Any, split: int = 0) -> Kind | None:
    """The kind of object this is, if it is one a branch of objects can hold."""
    from ..core.collections import TClonesArray
    from ..core.genvector import LorentzVector
    from ..core.vectors import TLorentzVector

    if hasattr(type(obj), "_cxx_layout_"):
        return Macro()
    if isinstance(obj, LorentzVector) and obj.SYSTEM == "PxPyPzE4D":
        return Lorentz()
    if isinstance(obj, TLorentzVector):
        return Lorentz4()
    if isinstance(obj, TClonesArray):
        return Clones() if split < 1 else SplitArray(obj._class)
    return _held_kind(obj)


def _held_kind(obj: Any) -> Kind | None:
    """The kind of a container object, which says its class by what it holds rather than by
    what it is: a histogram of the file, four-vectors, or a collection of collections."""
    if _histogram(obj):
        return Histogram()
    if _of_lorentz(obj):
        return Lorentzes()
    nested = _nested(obj)
    return None if nested is None else Nested(nested)


def _histogram(obj: Any) -> bool:
    name = obj.ClassName() if hasattr(obj, "ClassName") else ""
    return getattr(obj, "_xrd", None) is not None and name.startswith(("TH1", "TH2", "TH3", "TPr"))


def _of_lorentz(obj: Any) -> bool:
    """Is this a ``std::vector`` of GenVector's Cartesian four-vectors?"""
    kind = str(getattr(obj, "value_type", ""))
    return hasattr(obj, "push_back") and ("XYZTVector" in kind or "PxPyPzE4D" in kind)


def _nested(obj: Any) -> str | None:
    """The ROOT spelling of the class of a ``std::vector`` of collections, if this is one.

    One branch here counts one level of elements, so a collection whose elements count
    their own values cannot be laid out as rows: ROOT writes such a ``vector`` whole, as
    the container its streamer describes, and that is what :class:`Nested` does. Only a
    collection of collections is one - ROOT splits a ``vector`` of a class of its own,
    such as ``std::pair``, into a branch per member, which this writer refuses for want
    of a count no writer here makes.
    """
    kind = str(getattr(obj, "value_type", ""))
    return vector_of(kind) if hasattr(obj, "push_back") and kind.startswith("vector<") else None


class Nested(_Whole):
    """A ``std::vector`` of containers, written whole each entry as its class streams it.

    ROOT gives such a branch the collection's own class name - ``vector<vector<double> >``
    - and one leaf, its basket holding every entry's count and inner counts, which no
    split branch of this writer expresses: a branch of a collection splits it into a
    count and a row per member of each element, and an element here has no members.
    """

    def __init__(self, classname: str) -> None:
        self.classname = classname

    def spec(self, obj: Any, split: int, bufsize: int | None) -> Spec:
        return WholeCollection(self.classname, split=0, basket_size=bufsize)

    def entry(self, obj: Any) -> bytes:
        from ...wdeclared import Catalogue
        from ...wdobjects import write_value

        buf = WBuffer()
        write_value(buf, self.classname, [list(inner) for inner in obj], Catalogue({}))
        return bytes(buf.data)


@dataclass(frozen=True)
class WholeCollection(Spec):
    """A ``std::vector`` of containers, which ROOT lays out whole: one entry the collection
    as its own streamer writes it, since a nested count is no shape a split branch holds.

    The spec sits beside the kind that makes it rather than with the other specs of
    :mod:`~xrdroot.wbranch`: only a branch given an object holds a collection of
    collections, and only this module's :class:`Nested` builds one.
    """

    classname: str

    def build(self, name: str, basket_size: int) -> Branch:
        column = StreamedColumn(name, self._size(basket_size))
        info = _collection_info(self.classname, "", COLLECTION)
        leaf = ElementLeaf(name, name, 0, WHOLE, NOT_A_MEMBER)
        return Branch(name, name, self.split, column, [leaf], info)


class ObjectSlot:
    """The slot of a branch of objects: the object read at every ``Fill``, one entry each.

    It answers what a :class:`~.store.Slot` answers, so a store holds it
    beside the slots of numbers; what it hands the writer is the branch's
    :class:`~xrdroot.wbranch.Spec`, and a column for each column under it.
    """

    kind = OBJECT
    counter = None
    size = 1
    code = ""

    def __init__(self, name: str, address: Any, kind: Kind, branch: Spec) -> None:
        self.name = self.branch = self.title = name
        self.address = address
        self._kind, self._spec = kind, branch
        self.keys = column_keys(name, branch.build(name, 32000))
        self.pending: list[Any] = []
        self.chunks: list[dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"<ObjectSlot {self.name!r} of {type(self._kind).__name__}>"

    def read(self, counts: Any) -> Any:
        """This entry: the object the branch was given, as it is now."""
        return self._kind.entry(_held(self.address))

    def keep(self, value: Any) -> None:
        self.pending.append(value)

    def seal(self) -> None:
        if self.pending:
            self.chunks.append(self._kind.columns(self.keys, self.pending))
            self.pending = []

    def columns(self) -> dict[str, Any]:
        """Every entry filled, a column per key: asked only of a branch with entries."""
        self.seal()
        if len(self.chunks) > 1:
            self.chunks = [{key: _joined([c[key] for c in self.chunks]) for key in self.keys}]
        return self.chunks[0]

    def spec(self) -> Spec:
        return self._spec

    def nbytes(self, value: Any) -> int:
        """What ``Fill`` says this entry took: its bytes, or eight a number."""
        return len(value) if isinstance(value, bytes) else 8 * int(np.size(value))


def _joined(pieces: list[Any]) -> Any:
    from ...tree import concatenate

    if isinstance(pieces[0], list):
        return [item for piece in pieces for item in piece]
    return concatenate(pieces)


class ClonedSlot:
    """A split collection a clone copies: ROOT's one branch again, its members' rows within.

    ROOT writes a split ``TClonesArray`` as a branch counting the objects each
    entry holds and a branch per member of every object, so its members are no
    columns of a tree of their own - no column's name holds a dot. A clone
    therefore hands the writer the spec its source was written with, and the
    rows of every member for it to lay out again exactly as ROOT did.
    """

    kind = OBJECT
    counter = None
    size = 1
    code = ""

    def __init__(
        self, name: str, element: str, split: int, sources: list[str], follows: list[Any]
    ) -> None:
        self.name = self.branch = self.title = name
        self._kind = SplitArray(element)
        self._spec = SplitClones(element, split=split)
        #: What the tree this copies calls each member's column, in ROOT's order.
        self.sources = sources
        self._follows = follows
        #: The names the writer keeps the count and each member's rows under.
        self.keys = column_keys(name, self._spec.build(name, 32000))
        self.pending: list[Any] = []
        self.chunks: list[Any] = []

    def __repr__(self) -> str:
        return f"<ClonedSlot {self.name!r} of {len(self.sources)} members>"

    def read(self, counts: Any = None) -> Any:
        """One entry as ``Fill`` reads it: each object of the entry, and its members' values."""
        values = [follow.get() for follow in self._follows]
        return [tuple(one) for one in zip(*values, strict=True)]

    def taken(self, read: Any) -> list[Any]:
        """Every entry copied at once: each of its objects' members, from the columns read."""
        entries = zip(*[read[column] for column in self.sources], strict=True)
        return [[tuple(values) for values in zip(*entry, strict=True)] for entry in entries]

    def keep(self, value: Any) -> None:
        self.pending.append(value)

    def seal(self) -> None:
        if self.pending:
            self.chunks.append(self.pending)
            self.pending = []

    def columns(self) -> dict[str, Any]:
        """How many objects each entry holds, and each member's rows, as ROOT writes them."""
        self.seal()
        rows = [row for chunk in self.chunks for row in chunk]
        return self._kind.columns(self.keys, rows)

    def spec(self) -> Spec:
        return self._spec

    def nbytes(self, value: Any) -> int:
        """What ``Fill`` says this entry took: the size of what each of its objects read."""
        return 8 * sum(int(np.size(one)) for row in value for one in row)


def object_slot(name: str, address: Any, split: int, bufsize: int | None) -> ObjectSlot | None:
    """The slot for a branch given an object, or ``None`` when ``address`` holds no object."""
    obj = _held(address)
    kind = _kind(obj, split)
    if kind is None:
        return None
    if isinstance(kind, Macro) and not _splittable(obj, split):
        kind = MacroWhole()
    return ObjectSlot(name, address, kind, kind.spec(obj, split, bufsize))
