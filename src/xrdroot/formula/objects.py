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
from collections.abc import Iterator, Mapping, Sequence
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
        return rows, 1
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
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        for name in _names(obj):
            found.setdefault(name, []).append(_found(obj, name))
    for name, values in found.items():
        _offer(branch, (*path, name), values, learned, dims, crossed)


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


class _Learned:
    """What one whole-object branch offers, as it is learned member by member."""

    __slots__ = ("paths", "dims", "kinds")

    def __init__(self) -> None:
        self.paths: dict[str, tuple[str, tuple[str, ...]]] = {}
        self.dims: dict[str, int] = {}
        self.kinds: dict[str, str] = {}


class Whole:
    """The members a tree's whole-object branches offer, and how one of them is read.

    ``paths`` is what was learned, a name as an expression may write it - bare,
    ``fNtrack``, or with its object in front, ``event.fNtrack`` - against the
    branch it is read from and the path into each entry's object.
    """

    __slots__ = ("paths", "dims", "kinds")

    def __init__(
        self,
        paths: Mapping[str, tuple[str, tuple[str, ...]]] | None = None,
        dims: Mapping[str, int] | None = None,
        kinds: Mapping[str, str] | None = None,
    ) -> None:
        self.paths = dict(paths or {})
        #: How many dimensions each name has, as binding needs it before anything is read.
        self.dims = dict(dims or {})
        self.kinds = dict(kinds or {})

    def __repr__(self) -> str:
        objects = len({branch for branch, _path in self.paths.values()})
        return f"<Whole {len(self.paths)} members of {objects} objects>"

    def __contains__(self, name: object) -> bool:
        return name in self.paths

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths)

    def whole_number(self, name: str) -> bool:
        """Is the member a whole number, which ROOT bins a whole number wide?

        Of the values the member was learned from, which is the type the reader
        gave it: an ``Int_t`` member is a NumPy integer, a ``Float_t`` one not.
        """
        return self.kinds.get(name, "O") in ("i", "u", "b")

    def branch(self, name: str) -> str:
        """The whole-object branch a member is read from."""
        return self.paths[name][0]

    def member(self, parts: Sequence[str]) -> str | None:
        """The member ``parts`` spells, as written or as a getter reads it, or ``None``.

        Every part may be either spelling, which is why each combination is
        tried rather than each part mapped on its own: a formula mixes them,
        as ROOT's own code does, writing ``fH.GetXaxis().GetXmax()`` for the
        member ``fH.fXaxis.fXmax``.
        """
        for candidate in itertools.product(*(spelled(part) for part in parts)):
            name = ".".join(candidate)
            if name in self.paths:
                return name
        return None

    def owner(self, parts: Sequence[str]) -> str | None:
        """The deepest object here that ``parts`` leads into, whether or not it has the last name.

        What tells a member this reader cannot read from a name that is nothing
        at all: the object a formula walked into is here, and the method it asks
        of it is one a file's objects do not answer.
        """
        for stop in range(len(parts), 0, -1):
            prefix = ".".join(parts[:stop])
            if any(name.startswith(f"{prefix}.") for name in self.paths):
                return prefix
        return None

    def read(self, name: str, entries: Sequence[Any]) -> Any:
        """One member's column: its value per entry, or a row per entry through a collection."""
        _branch, path = self.paths[name]
        return _column(name, [_take(entry, path) for entry in entries])


def _take(value: Any, path: tuple[str, ...]) -> Any:
    """What ``path`` is in one entry's object: one value, or a row of them through a collection."""
    if not path:
        return value
    if isinstance(value, (list, tuple)):
        return [_take(item, path) for item in value]
    if not isinstance(value, dict):
        return MISSING
    found = _found(value, path[0])
    return MISSING if found is MISSING else _take(found, path[1:])


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
    for branch, values in samples.items():
        own = _Learned()
        _learn(branch, list(values), (), own, 0)
        for name, found in own.paths.items():
            if name in taken:  # a branch of that name is read first, as ROOT reads it
                continue
            both.setdefault(name, found)  # the first whole branch of the tree wins
            dims.setdefault(name, own.dims[name])
            kinds.setdefault(name, own.kinds[name])
    return Whole(both, dims, kinds)
