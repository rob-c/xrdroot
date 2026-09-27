"""``RooArgSet`` and ``RooArgList``: the collections of variables and functions RooFit passes.

A set holds each name once and a list keeps its order and may repeat, and
both hand back what they hold by name or by position. PyROOT lets any
Python iterable stand where one is wanted - ``{x}``, ``[a0, a1]`` - so
:func:`as_list` takes those too.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

from .printing import (
    RooPrintable,
    kClassName,
    kCollectionHeader,
    kName,
    kSingleLine,
    kValue,
)

__all__ = ["RooAbsCollection", "RooArgList", "RooArgSet", "as_list", "names_of"]


def _flat(items: Iterable[Any]) -> Iterator[Any]:
    for item in items:
        if isinstance(item, (RooAbsCollection, list, tuple, set, frozenset)):
            yield from _flat(item)
        else:
            yield item


#: A collection of the same kind as the one a method is called on.
_C = TypeVar("_C", bound="RooAbsCollection")


class RooAbsCollection(RooPrintable):
    """What a set and a list share: their members, found by name or position."""

    #: Whether a second member of the same name is refused.
    unique = True

    def __init__(self, *args: Any) -> None:
        self._list: list[Any] = []
        self._name = ""
        items = list(args)
        if items and isinstance(items[-1], str):
            self._name = items.pop()
        for item in _flat(items):
            self.add(item)

    # -- membership ---------------------------------------------------------------

    def add(self, item: Any, silent: bool = False) -> bool:
        """Add ``item``, or each of a collection's; a set refuses a name it has."""
        if isinstance(item, (RooAbsCollection, list, tuple, set, frozenset)):
            return all([self.add(one, silent) for one in _flat(item)])  # noqa: C419 - add all
        if self.unique and self.find(item.GetName()) is not None:
            return False
        self._list.append(item)
        return True

    addOwned = add
    addClone = add

    def remove(self, item: Any, silent: bool = False, matchByNameOnly: bool = False) -> bool:
        """Take out ``item``, or each member of a collection, matched by name."""
        if isinstance(item, (RooAbsCollection, list, tuple, set)):
            return all([self.remove(one) for one in _flat(item)])  # noqa: C419 - remove all
        name = item.GetName()
        before = len(self._list)
        self._list = [one for one in self._list if one.GetName() != name]
        return len(self._list) != before

    def removeAll(self) -> None:
        self._list = []

    def replace(self, old: Any, new: Any) -> bool:
        for index, one in enumerate(self._list):
            if one.GetName() == old.GetName():
                self._list[index] = new
                return True
        return False

    def find(self, name: Any) -> Any:
        """The member called ``name`` - or with the name of ``name`` - or ``None``."""
        wanted = name if isinstance(name, str) else name.GetName()
        return next((one for one in self._list if one.GetName() == wanted), None)

    def contains(self, item: Any) -> bool:
        return self.find(item) is not None

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def containsInstance(self, item: Any) -> bool:
        return any(one is item for one in self._list)

    def index(self, item: Any) -> int:
        name = item if isinstance(item, str) else item.GetName()
        return next((i for i, one in enumerate(self._list) if one.GetName() == name), -1)

    def at(self, index: int) -> Any:
        return self._list[index] if 0 <= index < len(self._list) else None

    def first(self) -> Any:
        return self._list[0] if self._list else None

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            found = self.find(key)
            if found is None:
                raise KeyError(f"{type(self).__name__} has no member called {key!r}")
            return found
        return self._list[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(list(self._list))

    def __len__(self) -> int:
        return len(self._list)

    def __bool__(self) -> bool:
        return True

    def size(self) -> int:
        return len(self._list)

    getSize = size

    def empty(self) -> bool:
        return not self._list

    def names(self) -> list[str]:
        return [one.GetName() for one in self._list]

    def GetName(self) -> str:
        return self._name

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def GetTitle(self) -> str:
        return self._name

    def ClassName(self) -> str:
        return type(self).__name__

    def overlaps(self, other: Iterable[Any]) -> bool:
        return any(self.find(one) is not None for one in _flat(other))

    def hasSameLayout(self, other: RooAbsCollection) -> bool:
        return self.names() == other.names()

    def equals(self, other: RooAbsCollection) -> bool:
        return sorted(self.names()) == sorted(other.names())

    # -- selections and copies ----------------------------------------------------

    def _like(self, members: Iterable[Any]) -> RooAbsCollection:
        made = type(self)()
        for one in members:
            made.add(one)
        return made

    def selectByName(self, patterns: str, verbose: bool = False) -> RooAbsCollection:
        """The members whose names match any of the comma-separated wildcard ``patterns``."""
        import fnmatch

        wanted = [one for one in str(patterns).split(",") if one]
        return self._like(
            one for one in self._list if any(fnmatch.fnmatchcase(one.GetName(), w) for w in wanted)
        )

    def selectByAttrib(self, name: str, value: bool) -> RooAbsCollection:
        return self._like(one for one in self._list if bool(one.getAttribute(name)) == bool(value))

    def selectCommon(self, other: Iterable[Any]) -> RooAbsCollection:
        names = {one.GetName() for one in _flat(other)}
        return self._like(one for one in self._list if one.GetName() in names)

    def Clone(self, name: Any = None) -> RooAbsCollection:
        """A collection of the same members - not copies of them - called ``name``."""
        made = self._like(self._list)
        made._name = self._name if name is None else str(name)
        return made

    def snapshot(self, deepCopy: bool = True) -> RooAbsCollection:
        """Copies of the members, their values as they are now."""
        return self._like(one.clone(one.GetName()) for one in self._list)

    def assign(self, other: Iterable[Any]) -> RooAbsCollection:
        """Take the values of ``other``'s members of the same names."""
        for one in _flat(other):
            mine = self.find(one.GetName())
            if mine is not None:
                mine.copy_value_from(one)
        return self

    assignValueOnly = assign
    assignFast = assign

    def setRealValue(self, name: str, value: float, verbose: bool = False) -> bool:
        found = self.find(name)
        if found is None:
            return True
        found.setVal(float(value))
        return False

    def getRealValue(self, name: str, defaultValue: float = 0.0, verbose: bool = False) -> float:
        found = self.find(name)
        return float(defaultValue) if found is None else float(found.getVal())

    def setCatIndex(self, name: str, index: int, verbose: bool = False) -> bool:
        found = self.find(name)
        if found is None:
            return True
        found.setIndex(int(index))
        return False

    def getCatIndex(self, name: str, defaultValue: int = 0, verbose: bool = False) -> int:
        found = self.find(name)
        return int(defaultValue) if found is None else int(found.getIndex())

    def setCatLabel(self, name: str, label: str, verbose: bool = False) -> bool:
        found = self.find(name)
        if found is None:
            return True
        found.setLabel(str(label))
        return False

    def getCatLabel(self, name: str, defaultValue: str = "", verbose: bool = False) -> str:
        found = self.find(name)
        return str(defaultValue) if found is None else str(found.getLabel())

    def setAttribAll(self, name: str, value: bool = True) -> None:
        for one in self._list:
            one.setAttribute(name, value)

    def sort(self, reverse: bool = False) -> None:
        self._list.sort(key=lambda one: one.GetName(), reverse=reverse)

    def sorted_copy(self: _C) -> _C:
        """The same members sorted by name, as ``getParameters`` hands them back."""
        made = self._like(sorted(self._list, key=lambda one: one.GetName()))
        made._name = self._name
        return made  # type: ignore[return-value]

    def isConstant(self) -> bool:
        return all(one.isConstant() for one in self._list)

    # -- printing -----------------------------------------------------------------

    def printName(self) -> str:
        return self._name

    def printTitle(self) -> str:
        return self._name

    def printClassName(self) -> str:
        return type(self).__name__

    def printValue(self) -> str:
        return "(" + ",".join(one.GetName() for one in self._list) + ")"

    def defaultPrintContents(self, option: Any) -> int:
        text = str(option or "")
        if text == "I":
            return kValue
        if "v" in text:
            from .printing import kAddress, kArgs, kExtras, kTitle

            return kAddress | kName | kArgs | kClassName | kValue | kTitle | kExtras
        return kName | kClassName | kValue

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        """Each member on a line of its own, numbered, the names in one width."""
        text = ""
        if self._name and contents & kCollectionHeader:
            text += f"{indent}{type(self).__name__}::{self._name}:\n"
        saved = self.name_length[0]
        if saved == 0:
            self.name_length[0] = max([1, *(len(one.GetName()) for one in self._list)]) + 1
        try:
            for number, one in enumerate(self._list, start=1):
                text += f"{indent}{number:3d}) " + one.printStream(contents, kSingleLine, "")
        finally:
            self.name_length[0] = saved
        return text

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.printValue()}>"

    def __str__(self) -> str:
        """What PyROOT prints of a collection: cppyy's view of it, pointers to its members."""
        return "{ " + ", ".join(f"@{hex(id(one))}" for one in self._list) + " }"


class RooArgSet(RooAbsCollection):
    """``RooArgSet``: variables and functions, each name at most once."""

    unique = True


class RooArgList(RooAbsCollection):
    """``RooArgList``: variables and functions in order, where a name may come twice."""

    unique = False


def as_list(items: Any) -> list[Any]:
    """The RooFit objects in ``items`` - one, a collection, or any Python iterable of them."""
    if items is None:
        return []
    if isinstance(items, (RooAbsCollection, list, tuple, set, frozenset)):
        return list(_flat(items))
    return [items]


def names_of(items: Any) -> list[str]:
    return [one.GetName() for one in as_list(items)]
