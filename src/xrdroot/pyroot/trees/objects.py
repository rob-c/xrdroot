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

import sys
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...tree import Jagged
from ...wbranch import Collection, Spec, Split, Whole, column_keys
from ...wclasses import declared, harvested
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
        if split < 1:
            raise UnsupportedFeatureError(
                f"a {obj._cxx_layout_[0]} branch at split level 0 is one object streamed "
                f"whole per entry, which this writer does for ROOT's own classes; give it "
                f"split level 1 or more"
            )
        return Split(declared(type(obj)), own=True, split=split, basket_size=bufsize)

    def entry(self, obj: Any) -> Any:
        return tuple(getattr(obj, _attribute(name)) for name, *_ in obj._cxx_layout_[2])


def _attribute(name: str) -> str:
    """A C++ member's name as the translation made it a Python attribute."""
    from ...cint.symbols import member_name

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
        if split > 0:
            raise UnsupportedFeatureError(
                "a TClonesArray split into a branch per member is not written here; give "
                "the branch split level 0 and it is written whole, as ROOT also writes it"
            )
        return Whole(
            "TClonesArray", object=True, holds=(obj._class,), split=0, basket_size=bufsize
        )

    def entry(self, obj: Any) -> bytes:
        getters = [_members_of(item) for item in obj]
        return named("TClonesArray", stream_clones(obj._class, getters, obj._bypass))


def _members_of(obj: Any) -> Any:
    """How a drawing object's members are read: its own table of them, by ROOT's names."""
    members = getattr(obj, "members", None)
    return members.__getitem__ if members is not None else (lambda name: getattr(obj, name))


def _kind(obj: Any) -> Kind | None:
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
        return Clones()
    if _histogram(obj):
        return Histogram()
    return Lorentzes() if _of_lorentz(obj) else None


def _histogram(obj: Any) -> bool:
    name = obj.ClassName() if hasattr(obj, "ClassName") else ""
    return getattr(obj, "_xrd", None) is not None and name.startswith(("TH1", "TH2", "TH3", "TPr"))


def _of_lorentz(obj: Any) -> bool:
    """Is this a ``std::vector`` of GenVector's Cartesian four-vectors?"""
    kind = str(getattr(obj, "value_type", ""))
    return hasattr(obj, "push_back") and ("XYZTVector" in kind or "PxPyPzE4D" in kind)


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


def object_slot(name: str, address: Any, split: int, bufsize: int | None) -> ObjectSlot | None:
    """The slot for a branch given an object, or ``None`` when ``address`` holds no object."""
    obj = _held(address)
    kind = _kind(obj)
    if kind is None:
        return None
    return ObjectSlot(name, address, kind, kind.spec(obj, split, bufsize))
