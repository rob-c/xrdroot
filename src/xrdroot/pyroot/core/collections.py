"""``TList``, ``TObjArray``, ``TIter`` and the strings kept in them.

ROOT's containers are lists of objects with ROOT's names for what a Python
list already does, and they are Python lists too: iterable, indexable and
sized, so ``for h in f.GetListOfKeys()`` and ``len(lst)`` work as PyROOT's do.
``Print`` and ``ls`` print ROOT's header and each thing held, one level in.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .objects import Indent, TNamed, TObject

__all__ = [
    "TCollection",
    "TSeqCollection",
    "TList",
    "THashList",
    "TObjArray",
    "TOrdCollection",
    "TIter",
    "TObjString",
    "TObjLink",
]

#: Not ROOT's name, so not in the namespace; what ``GetListOfFunctions`` hands back.
_PRIVATE = ["FunctionList"]


def _named(item: Any, name: str) -> bool:
    """Is ``item`` called ``name``, as ``FindObject`` compares them?"""
    return hasattr(item, "GetName") and item.GetName() == name


class TCollection(TObject):
    """``TCollection``: a named collection of objects, held in order."""

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self._items: list[Any] = []
        self._name = ""
        self._owner = False

    # -- Python's view ------------------------------------------------------------

    def __iter__(self) -> Iterator[Any]:
        return iter(list(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: Any) -> Any:
        return self._items[index]

    def __contains__(self, item: object) -> bool:
        return self.Contains(item)

    def __bool__(self) -> bool:
        return True

    # -- ROOT's --------------------------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def SetName(self, name: Any) -> None:
        self._name = str(name)

    def Add(self, obj: Any) -> None:
        """``Add``: at the end."""
        self._items.append(obj)

    def AddLast(self, obj: Any) -> None:
        self._items.append(obj)

    def AddFirst(self, obj: Any) -> None:
        self._items.insert(0, obj)

    def AddAll(self, other: Any) -> None:
        self._items.extend(other)

    def GetSize(self) -> int:
        return len(self._items)

    def GetEntries(self) -> int:
        return len(self._items)

    def IsEmpty(self) -> bool:
        return not self._items

    def Contains(self, item: Any) -> bool:
        """By name for a string, by being the very object otherwise."""
        return self.FindObject(item) is not None

    def FindObject(self, name: Any) -> Any:
        """The first object called ``name``, or the object itself if held; ``None`` if neither."""
        if isinstance(name, str):
            return next((item for item in self._items if _named(item, name)), None)
        return next((item for item in self._items if item is name), None)

    def IndexOf(self, obj: Any) -> int:
        return next((at for at, item in enumerate(self._items) if item is obj), -1)

    def Remove(self, obj: Any) -> Any:
        """``Remove``: take ``obj`` out and hand it back, or ``None`` if it is not held."""
        at = self.IndexOf(obj)
        return None if at < 0 else self._items.pop(at)

    def Clear(self, option: str = "") -> None:
        self._items.clear()

    def Delete(self, option: str = "") -> None:
        self._items.clear()

    def SetOwner(self, enable: bool = True) -> None:
        self._owner = bool(enable)

    def IsOwner(self) -> bool:
        return self._owner

    def MakeIterator(self, dir: bool = True) -> TIter:
        return TIter(self, dir)

    def Print(self, option: str = "", *rest: Any) -> None:
        """``Print``: ROOT's ``Collection name=...`` header, then each entry one level in."""
        print(
            f"{Indent.text()}Collection name='{self.GetName()}', class='{self.ClassName()}', "
            f"size={self.GetSize()}"
        )
        Indent.deeper()
        try:
            for item in self._items:
                print(Indent.text(), end="")
                item.Print(option)
        finally:
            Indent.deeper(-1)

    def ls(self, option: str = "") -> None:
        """``ls``: the collection's ``OBJ:`` line, then each entry's ``ls`` one level in."""
        print(f"{Indent.text()}OBJ: {self.ClassName()}\t{self.GetName()}\t{self.GetTitle()} : 0")
        Indent.deeper()
        try:
            for item in self._items:
                item.ls(option)
        finally:
            Indent.deeper(-1)


class TSeqCollection(TCollection):
    """``TSeqCollection``: a collection in an order that can be indexed."""

    def At(self, index: int) -> Any:
        """``At(i)``: the ``i``th object, or ``None`` past the end."""
        return self._items[index] if 0 <= index < len(self._items) else None

    def UncheckedAt(self, index: int) -> Any:
        return self._items[index]

    def First(self) -> Any:
        return self.At(0)

    def Last(self) -> Any:
        return self.At(len(self._items) - 1)

    def AddAt(self, obj: Any, index: int) -> None:
        self._items.insert(int(index), obj)

    def AddBefore(self, before: Any, obj: Any) -> None:
        self._items.insert(max(self.IndexOf(before), 0), obj)

    def AddAfter(self, after: Any, obj: Any) -> None:
        at = self.IndexOf(after)
        self._items.insert(len(self._items) if at < 0 else at + 1, obj)

    def After(self, obj: Any) -> Any:
        at = self.IndexOf(obj)
        return None if at < 0 else self.At(at + 1)

    def Before(self, obj: Any) -> Any:
        at = self.IndexOf(obj)
        return None if at <= 0 else self.At(at - 1)

    def RemoveAt(self, index: int) -> Any:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def RemoveFirst(self) -> Any:
        return self.RemoveAt(0)

    def RemoveLast(self) -> Any:
        return self.RemoveAt(len(self._items) - 1)

    def GetLast(self) -> int:
        """The index of the last object, ``-1`` for an empty collection."""
        return len(self._items) - 1

    def Sort(self, order: bool = True) -> None:
        """``Sort``: by name - ``Compare`` - up, or down with ``order`` false."""
        self._items.sort(key=lambda item: item.GetName(), reverse=not order)

    def IsSorted(self) -> bool:
        names = [item.GetName() for item in self._items]
        return names == sorted(names)


class TList(TSeqCollection):
    """``TList``: a doubly linked list, which is a Python list here."""

    def FirstLink(self) -> TObjLink | None:
        return TObjLink(self, 0) if self._items else None

    def LastLink(self) -> TObjLink | None:
        return TObjLink(self, len(self._items) - 1) if self._items else None


class FunctionList(TList):
    """``GetListOfFunctions``: an object's own list, read and changed where it is kept.

    A fitted function or one a script adds is kept in the xrdroot object's
    ``functions``, so it is written with it; anything else - a stats box the
    drawing puts there, a line a macro hangs on a histogram - in a list of
    its own beside them, which is drawn but not written.
    """

    def __init__(self, functions: list[Any], extras: list[Any]) -> None:
        super().__init__()
        self._functions, self._extras = functions, extras

    @property
    def _items(self) -> list[Any]:
        from .wrapping import wrap

        return [wrap(item) for item in self._functions] + list(self._extras)

    @_items.setter
    def _items(self, value: list[Any]) -> None:
        """What ``TCollection.__init__`` sets, which the two lists stand in for."""

    def _home(self, obj: Any) -> list[Any]:
        from ...function import Function

        return self._functions if isinstance(getattr(obj, "_xrd", None), Function) else self._extras

    def Add(self, obj: Any) -> None:
        home = self._home(obj)
        home.append(obj._xrd if home is self._functions else obj)

    AddLast = Add

    def AddFirst(self, obj: Any) -> None:
        home = self._home(obj)
        home.insert(0, obj._xrd if home is self._functions else obj)

    def Remove(self, obj: Any) -> Any:
        """``Remove``: take ``obj`` off its list and hand it back - ``None`` if on none."""
        for held in (self._extras, self._functions):
            for at, item in enumerate(held):
                if item is obj or item is getattr(obj, "_xrd", None):
                    del held[at]
                    return obj
        return None

    def Clear(self, option: str = "") -> None:
        del self._functions[:], self._extras[:]

    Delete = Clear


class THashList(TList):
    """``THashList``: a list also found by name quickly, which a list is here."""


class TOrdCollection(TSeqCollection):
    """``TOrdCollection``: an ordered collection."""


class TObjArray(TSeqCollection):
    """``TObjArray``: an array of objects, with room made as it is asked for."""

    def __init__(self, size: int = 16, lower: int = 0) -> None:
        super().__init__()
        self._lower = int(lower)

    def AddAt(self, obj: Any, index: int) -> None:
        """``AddAt``: into slot ``index``, growing the array with empty slots to reach it."""
        while len(self._items) <= index:
            self._items.append(None)
        self._items[index] = obj

    def AddAtAndExpand(self, obj: Any, index: int) -> None:
        self.AddAt(obj, index)

    def Expand(self, size: int) -> None:
        del self._items[size:]
        self._items.extend([None] * (size - len(self._items)))

    def GetEntries(self) -> int:
        """The slots holding something, which ``GetEntriesFast`` does not count."""
        return sum(item is not None for item in self._items)

    def GetEntriesFast(self) -> int:
        return len(self._items)

    def GetLast(self) -> int:
        """The index of the last slot holding something."""
        filled = [at for at, item in enumerate(self._items) if item is not None]
        return filled[-1] if filled else -1

    def Compress(self) -> None:
        self._items = [item for item in self._items if item is not None]

    def LowerBound(self) -> int:
        return self._lower

    def RemoveAt(self, index: int) -> Any:
        """``RemoveAt``: empty slot ``index`` and hand back what was there."""
        if not 0 <= index < len(self._items):
            return None
        found, self._items[index] = self._items[index], None
        return found

    def Remove(self, obj: Any) -> Any:
        at = self.IndexOf(obj)
        return None if at < 0 else self.RemoveAt(at)


class TObjLink:
    """``TObjLink``: one place in a :class:`TList`, and the ones either side of it."""

    def __init__(self, held: TSeqCollection, at: int) -> None:
        self._held, self._at = held, at

    def GetObject(self) -> Any:
        return self._held.At(self._at)

    def Next(self) -> TObjLink | None:
        return TObjLink(self._held, self._at + 1) if self._at + 1 < len(self._held) else None

    def Prev(self) -> TObjLink | None:
        return TObjLink(self._held, self._at - 1) if self._at > 0 else None

    def GetOption(self) -> str:
        return ""


class TIter:
    """``TIter``: ``next()`` through a collection, ``None`` at the end, as ROOT's is."""

    def __init__(self, collection: Any, dir: bool = True) -> None:
        self._collection = collection
        self._forward = bool(dir)
        self.Reset()

    def Reset(self) -> None:
        items = list(self._collection) if self._collection is not None else []
        self._items = items if self._forward else items[::-1]
        self._at = 0

    def Next(self) -> Any:
        """The next object, or ``None`` once there are no more."""
        if self._at >= len(self._items):
            return None
        self._at += 1
        return self._items[self._at - 1]

    def __call__(self) -> Any:
        return self.Next()

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._at >= len(self._items):
            raise StopIteration
        return self.Next()

    def GetCollection(self) -> Any:
        return self._collection


class TObjString(TNamed):
    """``TObjString``: a string that can be kept in a collection."""

    def __init__(self, text: Any = "") -> None:
        super().__init__(str(text), "")

    def GetString(self) -> str:
        return self.GetName()

    def String(self) -> str:
        return self.GetName()

    def SetString(self, text: Any) -> None:
        self.SetName(text)

    def __str__(self) -> str:
        return self.GetName()

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(self.GetName())
