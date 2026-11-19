"""``TTreeReader``, ``TTreeReaderValue`` and ``TTreeReaderArray``: a tree walked entry by entry.

    >>> reader = TTreeReader("ntuple", f)                          # doctest: +SKIP
    >>> px = TTreeReaderValue['float'](reader, "px")               # doctest: +SKIP
    >>> while reader.Next():                                       # doctest: +SKIP
    ...     h.Fill(px.__deref__())          # *px in C++; px.Get()[0] as PyROOT has it

A numeric value's ``Get()`` is a bound pointer into its branch buffer, so
``px.Get()[0]`` reads or writes that value. A branch of a class C++ declared
reads as an object of that class made from the entry - ROOT's reader hands a
macro the object the branch holds - so ``r->ToString()``, which the translation
writes as ``r.ToString()``, is the class's own method; ``__deref__()`` - C++'s
``*px`` - is the number or that object. An array reads as a sequence:
``len``, ``[i]``, iteration, ``GetSize()``. Each is read from the tree's own
block of entries, so a loop over a reader costs what a loop of ``GetEntry`` does.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from ..core.declared import made_from
from ._base import _TObjectLike
from .core import python_value
from .layout import element_class
from .readerbuffer import ReaderBuffer, buffer_for

__all__ = ["TTreeReader", "TTreeReaderValue", "TTreeReaderArray"]

#: ``TTreeReader::EEntryStatus``: what ``SetEntry`` says.
ENTRY_VALID, ENTRY_NO_TREE, ENTRY_BEYOND_END = 0, 2, 7


def _unqualified(leaf: Any) -> str:
    """A leaf's name without the branch it is under: ``fPx`` for ``C1.fPx``."""
    return str(leaf.column).rsplit(".", 1)[-1]


def _ambiguous(name: str) -> None:
    """ROOT's ``AddReader`` warning: one name, several branches, and the first one read."""
    print(f"Warning in <AddReader>: Ignored branch {name} because a branch with the same name "
          f"already exists. TTreeReader requires an unique name for the branches. You may "
          f"need to put a dot at the end of the name of top-level branches.", file=sys.stderr)


def _tree_of(tree: Any, where: Any) -> Any:
    """The tree a reader walks: a tree, or a tree's name and the file or directory it is in."""
    from . import wrap

    if not isinstance(tree, str):
        return tree if hasattr(tree, "_batch") else wrap(tree)
    if where is None:
        raise ValueError(
            f"a TTreeReader of {tree!r} needs the file it is in, as in TTreeReader({tree!r}, f)"
        )
    if isinstance(where, str):
        from ...file import open_root

        where = open_root(where)
    found = where.Get(tree) if hasattr(where, "Get") else where[tree]
    return found if hasattr(found, "_batch") else wrap(found)


class TTreeReader(_TObjectLike):
    """``TTreeReader``: see the module's docstring."""

    _classname = "TTreeReader"

    def __init__(self, tree: Any = None, where: Any = None, entrylist: Any = None) -> None:
        super().__init__()
        self._tree = None if tree is None else _tree_of(tree, where)
        self._entry = -1
        self._begin, self._end = 0, -1
        self._list = entrylist

    def __iter__(self) -> Iterator[int]:
        while self.Next():
            yield self._entry

    def _stop(self) -> int:
        total = self.GetEntries()
        return total if self._end < 0 else min(self._end, total)

    def Next(self) -> bool:
        """Move to the next entry; whether there was one."""
        return self.SetEntry(max(self._entry + 1, self._begin)) == ENTRY_VALID

    def SetEntry(self, entry: int) -> int:
        """Make ``entry`` the current one; ``0``, ``kEntryValid``, if it is there."""
        self._entry = int(entry)
        if self._tree is None:
            return ENTRY_NO_TREE
        if not 0 <= self._entry < self._stop():
            return ENTRY_BEYOND_END
        number = self._entry if self._list is None else self._list.GetEntry(self._entry)
        self._tree._read_entry = int(number)
        return ENTRY_VALID

    def SetEntriesRange(self, begin: int, end: int = -1) -> int:
        """Walk only entries ``begin`` up to ``end``, ``-1`` meaning to the last."""
        self._begin, self._end = int(begin), int(end)
        self._entry = int(begin) - 1
        return ENTRY_VALID

    def Restart(self) -> None:
        self._entry = self._begin - 1

    def GetCurrentEntry(self) -> int:
        return self._entry

    def GetEntries(self, force: bool = False) -> int:
        if self._tree is None:
            return 0
        return int(self._tree.GetEntries() if self._list is None else self._list.GetN())

    def GetTree(self) -> Any:
        return self._tree

    def SetTree(self, tree: Any, where: Any = None) -> None:
        self._tree = _tree_of(tree, where)
        self._entry = self._begin - 1

    def IsChain(self) -> bool:
        return self._tree is not None and self._tree.ClassName() == "TChain"


class _Read:
    """What a value and an array share: the reader, the branch, and its value now."""

    #: The C++ type asked for, when the class was made for one.
    _type: Any = None

    def __class_getitem__(cls, kind: Any) -> type:
        return type(f"{cls.__name__}<{kind}>", (cls,), {"_type": kind})

    def __init__(self, reader: TTreeReader, name: str) -> None:
        self._reader = reader
        self._name = str(name)
        self._leaf: Any = None
        #: The branch the name is, if it is one: what says which class its objects are.
        self._branch: Any = None
        self._resolved = False
        self._attach()

    def _attach(self) -> None:
        """Find the branch this name reads, now that the reader has a tree.

        ROOT's skeleton builds its readers in the class's constructor, before ``Init``
        gives the reader the tree - the member initializers name only the reader and a
        branch - so a reader of a tree it does not have yet resolves at the first read.
        The name may also be one several branches share, which ROOT's reader resolves to
        the first of them and warns about, as its ``AddReader`` does.
        """
        tree = self._reader._tree
        if self._resolved or tree is None:
            return
        leaf = self._found(tree)
        self._branch = None if leaf is None else tree._branch_info(self._name)
        self._leaf = leaf
        self._resolved = True

    def _found(self, tree: Any) -> Any:
        """The leaf this name reads, warning when several branches hold one of that name."""
        leaf = tree._leaf_info(self._name)
        matches = [one for one in tree._leaves()
                   if self._name in (one.column, one.name)]
        if leaf is None:
            # A name of the leaf alone, which a split branch of objects names twice: ROOT
            # reads the first branch that holds one of that name.
            matches = [one for one in tree._leaves() if _unqualified(one) == self._name]
            leaf = matches[0] if matches else None
        if leaf is None:
            raise KeyError(
                f"the tree {tree.GetName()!r} has no branch called {self._name!r} to read"
            )
        if len(matches) > 1:
            _ambiguous(self._name)
        return leaf

    def _value(self) -> Any:
        self._attach()
        tree = self._reader._tree
        if tree is None or tree._read_entry < 0:
            raise ValueError(
                f"{self._name!r} has no value until the reader is at an entry; call Next first"
            )
        held = self._buffer()
        return held.read(tree._read_entry) if held is not None else tree._current(self._leaf.column)

    def _object_value(self) -> Any:
        """What the branch holds at the reader's entry, as PyROOT hands it over: an object of
        the class the branch holds when C++ declared that class - ROOT's reader hands a macro
        the ``ArrayHolder`` itself, so that ``r->ToString()`` runs the class's own method -
        and otherwise, or when the class was never declared here, what it gave before."""
        value = python_value(self._value(), self._leaf)
        if self._branch is None or not isinstance(value, Mapping):
            return value
        made = made_from(self._branch.classname, value)
        return value if made is None else made

    def _buffer(self) -> ReaderBuffer | None:
        tree = self._reader._tree
        return buffer_for(tree, self._leaf) if tree is not None and self._leaf is not None else None

    def IsValid(self) -> bool:
        self._attach()
        return self._leaf is not None

    def GetBranchName(self) -> str:
        return self._name


class TTreeReaderValue(_Read):
    """``TTreeReaderValue<T>``: one branch's value at the reader's entry."""

    def Get(self) -> Any:
        """A numeric branch's writable pointer, or the existing object/collection value."""
        # The read runs before the buffer check: it is what moves the current entry into
        # the borrowed storage the pointer then exposes.
        value = self._object_value()
        held = self._buffer()
        if held is not None:
            return held.reference()
        return value

    def __deref__(self) -> Any:
        """C++'s ``*value``: the number itself, or the object."""
        return self._object_value()

    def __getattr__(self, name: str) -> Any:
        """``r->ToString()``: C++'s ``operator->`` of a value reader is the object the branch
        holds, and the translation writes a member or method of it on the reader, so a name
        the reader does not have is the object's. A name of the reader's own - ``Get``,
        ``value``, ``IsValid`` - is found before this runs; a private one is never the
        object's and is refused here, without reading an entry."""
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
        return getattr(self.__deref__(), name)

    _cint_cell = True

    @property
    def value(self) -> Any:
        return self.__deref__()

    @value.setter
    def value(self, value: Any) -> None:
        self._value()
        held = self._buffer()
        if held is None:
            raise TypeError("a mutable reader value needs a fixed numeric leaf")
        held.write(0, value)

    def __float__(self) -> float:
        return float(self.__deref__())

    def __int__(self) -> int:
        return int(self.__deref__())


class TTreeReaderArray(_Read):
    """``TTreeReaderArray<T>``: one branch's collection at the reader's entry, as a sequence."""

    def _items(self) -> Any:
        self._attach()
        value = self._value()
        if isinstance(value, Mapping):
            return self._members(value)
        return np.asarray(value).reshape(-1)

    def _members(self, value: Mapping[str, Any]) -> list[Any]:
        """``arr[i]`` of a branch of objects: element ``i``, made from its members' values.

        A collection of a class this session declared reads as its members' columns; ROOT's
        reader hands a macro the objects, so ``arr[i].fPosX`` is the member's value at that
        index, and a class that was never declared here keeps the mapping it came as.
        """
        columns = [item for item in value.values() if hasattr(item, "__len__")]
        length = max((len(item) for item in columns), default=0)
        elements = []
        for index in range(length):
            members = {name: (item[index] if hasattr(item, "__len__") else item)
                       for name, item in value.items()}
            made = made_from(element_class(self._branch, self._leaf), members)
            elements.append(members if made is None else made)
        return elements

    def __len__(self) -> int:
        return len(self._items())

    def __getitem__(self, index: Any) -> Any:
        found = self._items()[index]
        return found.item() if isinstance(found, np.generic) else found

    def __setitem__(self, index: Any, value: Any) -> None:
        self._cint_ref(index).value = value

    def _cint_ref(self, index: Any) -> Any:
        self._value()
        held = self._buffer()
        if held is None:
            raise TypeError("a mutable reader array needs a fixed numeric leaf")
        return held.reference(index)

    def __iter__(self) -> Iterator[Any]:
        items = self._items()
        return iter(items if isinstance(items, list) else items.tolist())

    def GetSize(self) -> int:
        return len(self)

    size = GetSize

    def IsEmpty(self) -> bool:
        return len(self) == 0

    def At(self, index: int) -> Any:
        return self[index]
