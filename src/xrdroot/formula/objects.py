"""Names that are no branch: the members of an object a tree holds whole.

ROOT writes a tree's object branch one of two ways. Split, every member is a
branch of its own, so ``fPx`` is a column and ``evt.fPx`` the same column
named through its object. Whole - ``split=0`` - one branch streams the whole
object and no member is a branch at all, so ROOT's ``TTreeFormula`` looks the
name up inside the object each entry holds: ``fNtrack``, ``event.fNtrack``,
``fEvtHdr.fEvtNum``, and ``fTracks.fPx``, which loops over the objects of a
collection the way a branch of a collection is looped over.

Reading a whole branch gives its objects as their members: one dictionary an
entry, with each base class's members nested under the base's own name where
ROOT writes them at the derived object's level. :func:`_found` is that rule -
a name is a member of the object itself or of one of its bases, never of a
member object, so ``fH.fEntries`` and ``fEvtHdr.fEvtNum`` are read and
``fEvtNum`` alone is not.

A method is a name too, because ROOT's formulas resolve one the way they
resolve a member: ``GetNtrack()`` asks the class for the member it keeps
behind the getter, and ``fH.GetXaxis().GetXmax()`` walks on through what each
call gives. Nothing here has a class to ask, so a getter and the member it
names are two spellings of one member (:func:`spelled`), and a call's
arguments index that member - ``GetClosestDistance(2)`` is
``fClosestDistance[2]``, which is how ROOT's formulas reach into an array.

A few of ROOT's methods keep nothing to name: a histogram works its mean out
of the sums of weights and of ``x * w`` it accumulated as it was filled, so
``GetMean()`` is arithmetic over two members and not a member at all. Those
are :data:`DERIVED`, read from the members a file's object does hold, and
only where it holds all of them - a name the object gives no reason to
trust is not offered.

A ``TBits`` is drawn rather than read: ``TSelectorDraw::ProcessFillObject``
fills the number of each bit such an object holds *set*, one fill a bit, so
``event.fTriggerBits`` is a `TBits` member an expression names as it names a
member which is a number. What says an object is one is the three members
ROOT's class keeps - ``fNbits``, ``fNbytes`` and ``fAllBits`` - so a member
holding them is offered under two names: itself, whose value is the numbers
of the bits set, and ``TestBitNumber``, whose call's argument indexes a value
per bit position - ``TestBitNumber(28)``, which is what ROOT's own macro
selects on.

What a whole branch offers as a name is learned here from the values of a few
entries of it, because what the class holds is the file's description of it,
which is above this layer in the component graph. That is a limit on names,
not on values: what an expression names is read from every entry of the batch
it is evaluated over. A member that none of the sampled entries holds is a
member an expression does not know - a collection that is empty for the first
entries of a file is learned from the first entry that fills it.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError

__all__ = ["Whole", "SAMPLE", "spelled"]

#: How ROOT's classes name the member a getter reads: ``GetNtrack()`` is ``fNtrack``
#: and ``IsValid()`` is ``fIsValid``. Named here because a name learnt from a file
#: has no class behind it to answer a method call with anything else.
GETTER = re.compile(r"^(Get|Is)([A-Z]\w*)$")

Array = Any

#: How many entries of a whole-object branch are read to learn what its class holds.
#: One basket is read whatever its count, and a collection that is empty in the
#: first entries still offers its members if a later one is looked at.
SAMPLE = 8

#: How many objects of one collection the members are learned from. A collection
#: holds one class, so its objects offer the same names one another does, and a
#: ``TClonesArray`` of a few hundred of them says nothing the first few did not:
#: the entries are sampled for which members are held at all, not for what any
#: one object holds, and walking every object of a large collection to learn
#: names - which the draw then reads again for its values - is time spent twice.
LEARNED_OBJECTS = 8

#: The members ``TBits`` keeps, which is how a file's object says it is one: a
#: ``TBits`` the tree holds is drawn a set bit at a time, and no other object is.
BITS_MEMBERS = ("fNbits", "fNbytes", "fAllBits")

#: ROOT's method asking after one bit of a ``TBits``, whose argument is the bit.
TEST_BIT = "TestBitNumber"

#: What a ``TBits`` name reads: the numbers of the bits it holds set - what ROOT's
#: draw fills - or, for ``TestBitNumber``, a value per bit position the call indexes.
SET = "set"
ASKED = "asked"


class _Absent:
    """A member an object of this entry does not hold, which is not the same as an empty one."""


#: What :func:`_found` says when a name is not in an object.
MISSING = _Absent()


def _whole(branch: Any) -> bool:
    """Does this branch hold an object whole - as its members, rather than one branch each?"""
    return bool(branch.typename == "dict") and not branch.record.branches


def _is_base(name: str) -> bool:
    """Is a nested dictionary under ``name`` a base class's members, not a member object?

    ROOT names a data member ``fSomething`` and a base class by its own name,
    which is what the reader nests beside the object's own members; a base's
    members are written at the derived object's level, a member's inside it.
    """
    return not name.startswith("f")


def _found(obj: Mapping[str, Any], name: str) -> Any:
    """What ``name`` is in one object: its own member, or one a base class holds."""
    if name in obj:
        return obj[name]
    for key, value in obj.items():
        if _is_base(key) and isinstance(value, dict):
            got = _found(value, name)
            if got is not MISSING:
                return got
    return MISSING


def _names(obj: Mapping[str, Any]) -> list[str]:
    """Every name one object has: its own members, and each base class's at its own level."""
    names = list(obj)
    for key, value in obj.items():
        if _is_base(key) and isinstance(value, dict):
            names.extend(_names(value))
    return list(dict.fromkeys(names))


def _objects(values: Sequence[Any]) -> tuple[list[Any], int] | None:
    """Every object the values are, and how many collections were crossed to reach them.

    ``None`` for values that are numbers or strings, which is what a member an
    expression can read holds; an empty collection is a crossing of no objects,
    which offers no names but does not make the member a value.
    """
    if values and all(isinstance(value, dict) for value in values):
        return list(values), 0
    if values and all(isinstance(value, (list, tuple)) for value in values):
        return _one_level_down(values)
    return None


def _one_level_down(values: Sequence[Any]) -> tuple[list[Any], int] | None:
    """The elements of one level of collections, when they are objects and not numbers."""
    rows = [item for value in values for item in value]
    if rows and all(isinstance(item, dict) for item in rows):
        return rows[:LEARNED_OBJECTS], 1
    return None


def _kind(values: Sequence[Any]) -> str:
    """What kind of number or string a member's values are, as NumPy names the kind."""
    for value in values:
        if isinstance(value, np.ndarray):
            return str(value.dtype.kind)
        if isinstance(value, str):
            return "U"
        if isinstance(value, (list, tuple)):
            return _kind(list(value))
        return str(np.asarray(value).dtype.kind)
    return "O"


def _dims(values: Sequence[Any]) -> int:
    """How many dimensions a member's values have beyond the entry: a fixed array's, or a row's."""
    for value in values:
        if isinstance(value, np.ndarray):
            return int(value.ndim)
        if isinstance(value, (list, tuple)):
            return 1
    return 0


def _bits(objects: Sequence[Any]) -> bool:
    """Are these objects a ``TBits``, which ROOT's draw fills a set bit at a time?"""
    return bool(objects) and all(
        isinstance(obj, dict) and all(_found(obj, name) is not MISSING for name in BITS_MEMBERS)
        for obj in objects
    )


def _learn(
    branch: str,
    objects: Sequence[Any],
    path: tuple[str, ...],
    learned: _Learned,
    dims: int,
    crossed: bool = False,
) -> None:
    """Offer every member the objects of one level hold, and go on into the ones that hold objects.

    ``crossed`` says a collection was met on the way here, whose elements' last
    member is also a name of its own: ROOT reads ``fPx`` of a tree that holds
    ``fTracks`` whole as the tracks' momentum, looped over as ``fTracks.fPx`` is.
    """
    found: dict[str, list[Any]] = {}
    for obj in objects[:LEARNED_OBJECTS]:
        if not isinstance(obj, dict):
            continue
        for name in _names(obj):
            found.setdefault(name, []).append(_characters(_found(obj, name)))
    for name, values in found.items():
        _offer(branch, (*path, name), values, learned, dims, crossed)
    _derive(branch, path, found, learned, crossed)


def _derive(
    branch: str,
    path: tuple[str, ...],
    found: Mapping[str, Sequence[Any]],
    learned: _Learned,
    crossed: bool,
) -> None:
    """Offer the methods of ROOT's classes that are arithmetic over members this object holds.

    Every member one needs must be here and be a value: a method is not offered
    on the strength of one of its members, since an object that is not quite a
    histogram would then answer for a mean it never computed.
    """
    values = {name for name, held in found.items() if _objects(held) is None}
    for method, (needs, op) in DERIVED.items():
        if not all(need in values for need in needs):
            continue
        written = ".".join((*path, method))
        members = tuple(".".join((*path, need)) for need in needs)
        names = [written, f"{branch}.{written}"]
        if crossed:
            names.append(method)
        for name in names:
            learned.derived.setdefault(name, (members, op))
            learned.dims.setdefault(name, 0)
            learned.kinds.setdefault(name, "d")


def _offer_bits(
    branch: str,
    path: tuple[str, ...],
    learned: _Learned,
    dims: int,
    crossed: bool,
) -> None:
    """The two names a ``TBits`` member offers: the bits it holds set, and the method asking one.

    Both are a row per object - or, through a collection, a row per element -
    and both are whole numbers, which is what ``TTreeFormula::IsInteger`` says
    of a ``TBits``: ROOT bins it a whole number wide.
    """
    asked = (*path[:-1], f"{path[-1]}.{TEST_BIT}")
    for name, reading in ((path, SET), (asked, ASKED)):
        written = ".".join(name)
        names = [written, f"{branch}.{written}"]
        if crossed:
            names.append(name[-1])
        for one in names:
            learned.paths.setdefault(one, (branch, path))
            learned.bits.setdefault(one, reading)
            learned.dims.setdefault(one, dims + 1)
            learned.kinds.setdefault(one, "i")


def _offer(
    branch: str,
    path: tuple[str, ...],
    values: Sequence[Any],
    learned: _Learned,
    dims: int,
    crossed: bool,
) -> None:
    """One member, by what its values are: an object to go into, or a value an expression reads."""
    deeper = _objects(values)
    if deeper is not None:
        elements, rows = deeper
        if _bits(elements):
            _offer_bits(branch, path, learned, dims, crossed)
        _learn(branch, elements, path, learned, dims + rows, crossed or bool(rows))
        return
    written = ".".join(path)
    names = [written, f"{branch}.{written}"]
    if crossed:
        names.append(path[-1])
    for name in names:
        learned.paths.setdefault(name, (branch, path))
        learned.dims.setdefault(name, dims + _dims(values))
        learned.kinds.setdefault(name, _kind(values))


#: The methods that are arithmetic over members rather than a member themselves: the
#: members each needs, and how they are combined. ``GetMean`` of a histogram is
#: ``fTsumwx / fTsumw`` - ROOT's ``TH1::GetMean`` for an axis of equal bins.
DERIVED: dict[str, tuple[tuple[str, ...], str]] = {
    "GetMean": (("fTsumwx", "fTsumw"), "mean"),
}


def _mean(sum_wx: Array, sum_w: Array) -> Array:
    """``fTsumwx / fTsumw``, and zero for a histogram nothing was filled into."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(np.asarray(sum_wx, np.float64), np.asarray(sum_w, np.float64))
    return np.where(np.asarray(sum_w) == 0, 0.0, out)


#: How each :data:`DERIVED` method combines the members it reads.
DERIVED_OPS: dict[str, Callable[..., Array]] = {"mean": _mean}


class _Learned:
    """What one whole-object branch offers, as it is learned member by member."""

    __slots__ = ("paths", "dims", "kinds", "derived", "bits")

    def __init__(self) -> None:
        self.paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        self.dims: dict[str, int] = {}
        self.kinds: dict[str, str] = {}
        #: The methods computed from members, against the members they read.
        self.derived: dict[str, tuple[tuple[str, ...], str]] = {}
        #: The members that are a ``TBits``, against what of its bits each name reads.
        self.bits: dict[str, str] = {}


class Whole:
    """The members a tree's whole-object branches offer, and how one of them is read.

    ``paths`` is what was learned, a name as an expression may write it - bare,
    ``fNtrack``, or with its object in front, ``event.fNtrack`` - against the
    branch it is read from and the path into each entry's object.
    """

    __slots__ = ("paths", "dims", "kinds", "derived", "bits")

    def __init__(
        self,
        paths: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
        dims: Mapping[str, int] | None = None,
        kinds: Mapping[str, str] | None = None,
        derived: Mapping[str, tuple[tuple[str, ...], str]] | None = None,
        bits: Mapping[str, str] | None = None,
    ) -> None:
        self.paths = dict(paths or {})
        #: How many dimensions each name has, as binding needs it before anything is read.
        self.dims = dict(dims or {})
        self.kinds = dict(kinds or {})
        #: The methods computed from members - ``fH.GetMean`` - against what they read.
        self.derived = dict(derived or {})
        #: The members that are a ``TBits``, against what of its bits each name reads.
        self.bits = dict(bits or {})

    def __repr__(self) -> str:
        objects = len({branch for branch, _path in self.paths.values()})
        return f"<Whole {len(self.paths) + len(self.derived)} members of {objects} objects>"

    def __contains__(self, name: object) -> bool:
        return name in self.paths or name in self.derived

    def __iter__(self) -> Iterator[str]:
        return itertools.chain(self.paths, self.derived)

    def whole_number(self, name: str) -> bool:
        """Is the member a whole number, which ROOT bins a whole number wide?

        Of the values the member was learned from, which is the type the reader
        gave it: an ``Int_t`` member is a NumPy integer, a ``Float_t`` one not.
        """
        return self.kinds.get(name, "O") in ("i", "u", "b")

    def branch(self, name: str) -> str:
        """The whole-object branch a member is read from."""
        if name in self.paths:
            return self.paths[name][0]
        return self.paths[self.derived[name][0][0]][0]

    def member(self, parts: Sequence[str]) -> str | None:
        """The member ``parts`` spells, as written or as a getter reads it, or ``None``.

        Every part may be either spelling, which is why each combination is
        tried rather than each part mapped on its own: a formula mixes them,
        as ROOT's own code does, writing ``fH.GetXaxis().GetXmax()`` for the
        member ``fH.fXaxis.fXmax``. A getter no member is spelt after - see
        :meth:`answered` - has the names of the rest of the expression say
        which member ROOT's method gives back.
        """
        for name in self.written(parts):
            if name in self:
                return name
        for name in self.answered(parts):
            if name in self:
                return name
        return None

    def written(self, parts: Sequence[str]) -> Iterator[str]:
        """The names ``parts`` spell, each part as written and as the getter of it reads."""
        for candidate in itertools.product(*(spelled(part) for part in parts)):
            yield ".".join(candidate)

    def answered(self, parts: Sequence[str]) -> Iterator[str]:
        """The names a getter that spells no member here would give if it gave a member.

        ROOT's formula calls the method its class declares and reads the rest of
        the expression off what comes back, so ``GetHistogram().GetXaxis()`` is
        the axis of the member that method returns. A file has members and no
        method bodies, so which member a getter of a name like that answers with
        is written nowhere - but which member reads the rest of the name is:
        tried as each object the expression has reached, only the one holding
        what follows answers, and where none does nothing is offered. A getter
        with nothing after it names an object and not a number, so it has no
        such reading.
        """
        for at, part in enumerate(parts[:-1]):
            if not self.unnamed(part):
                continue
            heads = list(self.written(parts[:at]))
            for left in self.objects():
                steps = left.split(".")
                if len(steps) != at + 1 or ".".join(steps[:-1]) not in heads:
                    continue  # a name of this object's own, not one of a member of it
                yield from self.written([*parts[:at], steps[-1], *parts[at + 1 :]])

    def unnamed(self, part: str) -> bool:
        """Is ``part`` a getter naming no member - neither as written nor as ``f`` and its name?"""
        if GETTER.fullmatch(part) is None:
            return False
        return not any(step in self.segments() for step in spelled(part))

    def segments(self) -> set[str]:
        """Every name one step of a member's path is written with here."""
        return {step for name in self for step in name.split(".")}

    def objects(self) -> list[str]:
        """Every name here that holds members, in the order they were learned."""
        found: list[str] = []
        for name in self:
            parts = name.split(".")
            for stop in range(1, len(parts)):
                prefix = ".".join(parts[:stop])
                if prefix not in found:
                    found.append(prefix)
        return found

    def owner(self, parts: Sequence[str]) -> str | None:
        """The deepest object here that ``parts`` leads into, whether or not it has the last name.

        What tells a member this reader cannot read from a name that is nothing
        at all: the object a formula walked into is here, and the method it asks
        of it is one a file's objects do not answer.
        """
        for stop in range(len(parts), 0, -1):
            prefix = ".".join(parts[:stop])
            if any(name.startswith(f"{prefix}.") for name in self):
                return prefix
        return None

    def read(self, name: str, entries: Sequence[Any]) -> Any:
        """One member's column: its value per entry, or a row per entry through a collection."""
        if name in self.derived:
            members, op = self.derived[name]
            return DERIVED_OPS[op](*(self.read(member, entries) for member in members))
        _branch, path = self.paths[name]
        values = [_take(entry, path) for entry in entries]
        if name in self.bits:
            return _bits_column(name, values, self.bits[name])
        return _column(name, values)

    def tested(self, parts: Sequence[str]) -> str | None:
        """The name ``parts`` give if it asks into a ``TBits`` - ``fTriggerBits.TestBitNumber``.

        The call's argument indexes the bit, so what it is read from is a value
        per bit position rather than a value per object: the name ROOT's macro
        selects on, which the caller answers with an indexed ``Ref``.
        """
        for name in self.written(parts):
            if self.bits.get(name) == ASKED:
                return name
        return None


def _take(value: Any, path: tuple[str, ...]) -> Any:
    """What ``path`` is in one entry's object: one value, or a row of them through a collection."""
    if not path:
        return _characters(value)
    if isinstance(value, (list, tuple)):
        return [_take(item, path) for item in value]
    if not isinstance(value, dict):
        return MISSING
    found = _found(value, path[0])
    return MISSING if found is MISSING else _take(found, path[1:])


def _characters(value: Any) -> Any:
    """A ``char[N]`` member as the C string it is, as ROOT's ``TLeafC`` reads one.

    ROOT streams a fixed-size array of ``char`` as the characters it holds, so a
    formula sees a string and not twenty numbers: ``fType == "type1"``, which is
    how ROOT's own tree tests select on a ``char fType[20]`` member. The chars
    stop at the first NUL, as ``strlen`` does, and a byte that is not UTF-8 is
    carried through as its own character rather than refused.
    """
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.int8:
        text = value.tobytes()
        end = text.find(b"\x00")
        return text[: len(text) if end < 0 else end].decode("utf-8", "surrogateescape")
    return value


def _absent(value: Any) -> bool:
    if value is MISSING:
        return True
    return isinstance(value, (list, tuple)) and any(item is MISSING for item in value)


def _row(value: Any) -> bool:
    """Is one entry's value a row of values - an array member, or a collection crossed?"""
    return isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim > 0)


def _consistent(name: str, values: Sequence[Any]) -> None:
    """Refuse entries whose objects do not agree on what the member is."""
    rows = {_row(value) for value in values}
    if len(rows) <= 1 and not any(_absent(value) for value in values):
        return
    raise UnsupportedFeatureError(
        f"{name!r} is a member of the objects this tree holds whole, and the entries differ "
        f"in holding it - one does not, or holds one value where another holds a row; an "
        f"expression is read entry for entry, so select the entries that agree"
    )


def _column(name: str, values: Sequence[Any]) -> Any:
    """A member's values as a column: one an entry, or a row an entry through a collection."""
    _consistent(name, values)
    if all(isinstance(value, str) for value in values):
        return list(values)
    if not any(_row(value) for value in values):
        return np.asarray(values)
    return [list(value) if isinstance(value, (list, tuple)) else value for value in values]


def _bits_column(name: str, values: Sequence[Any], reading: str) -> Any:
    """A ``TBits`` member's column: per object, what its bits are.

    ROOT draws a ``TBits`` rather than reading it as one value -
    ``TSelectorDraw::ProcessFillObject`` fills the number of each bit that is
    set, one fill a bit, and warns and fills nothing for any other object - so
    the column of a name that is a ``TBits`` is a row of bit numbers per object,
    empty for an object whose bits are all clear. ``TestBitNumber`` asks after
    one bit instead, so what it reads is a value per bit position, which the
    call's argument indexes.
    """
    _consistent(name, values)
    return [_bit_row(name, value, reading) for value in values]


def _bit_row(name: str, value: Any, reading: str) -> Any:
    """One object's bits - or, through a collection, one row of them per element."""
    if isinstance(value, (list, tuple)):
        return [_bit_row(name, item, reading) for item in value]
    if not isinstance(value, dict):
        raise UnsupportedFeatureError(
            f"{name!r} is a ``TBits`` of the objects this tree holds whole, and one entry "
            f"does not hold it; an expression is read entry for entry, so select the "
            f"entries that all hold it"
        )
    bits = _bits_of(value)
    return np.flatnonzero(bits) if reading == SET else bits


def _bits_of(obj: Mapping[str, Any]) -> Any:
    """One ``TBits`` as a value per bit position: ``1`` where the bit is set, else ``0``.

    The bytes are ``fAllBits`` and the count ``fNbits``, which is ``TBits``'s own
    layout - its set bits are the ones ``FirstSetBit`` walks, so drawing them is
    the numbers of the bits ROOT's own loop would take one after another.
    """
    held = np.asarray(_found(obj, "fAllBits"), dtype=np.uint8).reshape(-1)
    count = int(np.asarray(_found(obj, "fNbits")).reshape(-1)[0])
    return np.unpackbits(held, bitorder="little")[:count].astype(np.int64)


def spelled(part: str) -> tuple[str, ...]:
    """A name as written, then as the member the getter of that name reads.

    ``GetNtrack`` gives ``('GetNtrack', 'fNtrack')``, ``IsValid`` gives
    ``('IsValid', 'fIsValid')``, and a member written as it is has no other
    spelling - which is the one place ROOT's two conventions differ, ``f`` and
    ``fIs`` being how its classes prefix what a getter and an ``Is`` answer.
    """
    found = GETTER.fullmatch(part)
    if found is None:
        return (part,)
    prefix, rest = found.groups()
    return (part, f"f{rest}" if prefix == "Get" else f"fIs{rest}")


def learned(samples: Mapping[str, Sequence[Any]], taken: Sequence[str] = ()) -> Whole:
    """The members each whole-object branch offers, from a few values of it, one an entry.

    ``taken`` are names that are already something else - a branch of the tree -
    which win over a member of the same name, as ROOT reads the branch first.
    """
    both: dict[str, tuple[str, tuple[str, ...]]] = {}
    dims: dict[str, int] = {}
    kinds: dict[str, str] = {}
    derived: dict[str, tuple[tuple[str, ...], str]] = {}
    bits: dict[str, str] = {}
    for branch, values in samples.items():
        own = _Learned()
        _learn(branch, list(values), (), own, 0)
        _members(own, both, dims, kinds, bits, taken)
        _methods(own, both, derived, dims, kinds, taken)
    return Whole(both, dims, kinds, derived, bits)


def _members(
    own: _Learned,
    both: dict[str, tuple[str, tuple[str, ...]]],
    dims: dict[str, int],
    kinds: dict[str, str],
    bits: dict[str, str],
    taken: Sequence[str],
) -> None:
    """Keep the members one whole branch offers, a name an earlier branch had staying with it."""
    for name, found in own.paths.items():
        if name in taken:  # a branch of that name is read first, as ROOT reads it
            continue
        both.setdefault(name, found)  # the first whole branch of the tree wins
        dims.setdefault(name, own.dims[name])
        kinds.setdefault(name, own.kinds[name])
        if name in own.bits:
            bits.setdefault(name, own.bits[name])


def _methods(
    own: _Learned,
    both: Mapping[str, tuple[str, tuple[str, ...]]],
    derived: dict[str, tuple[tuple[str, ...], str]],
    dims: dict[str, int],
    kinds: dict[str, str],
    taken: Sequence[str],
) -> None:
    """Keep the methods one branch's members spell, where it holds every member they read."""
    for name, (members, op) in own.derived.items():
        named = all(member in both or member in own.paths for member in members)
        if name in taken or name in both or not named:
            continue
        derived.setdefault(name, (members, op))
        dims.setdefault(name, own.dims[name])
        kinds.setdefault(name, own.kinds[name])
