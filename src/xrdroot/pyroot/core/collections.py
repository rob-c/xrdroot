"""``TList``, ``TObjArray``, ``TIter`` and the strings kept in them.

ROOT's containers are lists of objects with ROOT's names for what a Python
list already does, and they are Python lists too: iterable, indexable and
sized, so ``for h in f.GetListOfKeys()`` and ``len(lst)`` work as PyROOT's do.
``Print`` and ``ls`` print ROOT's header and each thing held, one level in.

PyROOT gives every collection a Python list's methods - ``append``, ``remove``,
``extend``, ``count``, ``+`` and ``*`` - and an ordered one the rest:
``insert``, ``pop``, ``reverse``, ``sort``, ``index``, and ``[i]``, ``[i:j:k]``
to read, assign and delete, each raising what a list raises; a slice read is
a collection of the same class.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ...cppstd.containers import TEMPLATES as _TEMPLATES
from . import objects
from .objects import Indent, TNamed, TObject
from .strings import TString

#: The ``std::`` templates cppyy gives by a bare C++ name too - a macro's ``list<Track>`` is
#: ``ROOT.list['Track']`` - and this kit's table of them is that list of names.
BARE_TEMPLATES = tuple(_TEMPLATES)

__all__ = [
    *BARE_TEMPLATES,
    "TCollection",
    "TSeqCollection",
    "TList",
    "THashList",
    "TObjArray",
    "TClonesArray",
    "TOrdCollection",
    "TIter",
    "TIterator",
    "TCollectionIter",
    "TListIter",
    "TObjArrayIter",
    "TMapIter",
    "THashTableIter",
    "TObjString",
    "TObjLink",
    "THashTable",
    "TPair",
    "TMap",
]

#: Not ROOT's name, so not in the namespace; what ``GetListOfFunctions`` hands back.
_PRIVATE = ["FunctionList"]


#: ``TObject::kSingleKey``: a collection written under one key, rather than a key an object.
SINGLE_KEY = 1 << 0


def _name_of(item: Any) -> str | None:
    """The name a collection indexes ``item`` under, or ``None`` for one that says none yet.

    A histogram is appended to its directory while it is still being made, before it has a
    name to give; it is indexed as soon as it has one.
    """
    found = getattr(item, "GetName", None)
    if found is None:
        return None
    try:
        return str(found())
    except Exception:  # an object being torn down cannot say its name
        return None


def _named(item: Any, name: str) -> bool:
    """Is ``item`` called ``name``, as ``FindObject`` compares them?"""
    return hasattr(item, "GetName") and item.GetName() == name


def _named_by(obj: Any, option: str) -> None:
    """``Add(obj, option)``'s option, which names the object ROOT is about to add."""
    if option and hasattr(obj, "SetName"):
        obj.SetName(option)


def __getattr__(name: str) -> Any:
    """``ROOT.list`` and its kin: the ``std::`` template of that name, of the element type
    asked for.

    They are answered here and not assigned above because ``list``, ``map``, ``set`` and
    ``pair`` are Python's own names, and every ``list(...)`` and ``set(...)`` in this module
    means the builtin's; a module's ``__getattr__`` is asked only for what its globals do not
    hold, so the two keep to themselves. Anything else is an ``AttributeError``, which is how
    the namespace reads "not here".
    """
    made = _TEMPLATES.get(name)
    if made is not None:
        return made
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class TCollection(TObject):
    """``TCollection``: a named collection of objects, held in order."""

    CLASS_TITLE = "Collection abstract base class"

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self._items: list[Any] = []
        self._name = ""
        self._owner = False
        #: The objects by name, and the marks the index was built at: how many times this
        #: collection has changed, and how many names have been set in the process. A
        #: directory asks its list for an object by name once per object appended - its
        #: name-clash check - and scanning a thousand objects with a Python call each was
        #: most of what booking a histogram cost.
        self._by_name: dict[str, list[Any]] = {}
        self._stale = True
        self._stamp = 0
        self._marks = -1
        #: Objects added before they could say their name - a histogram is appended to its
        #: directory while it is still being made - which are indexed as soon as they can.
        self._unnamed: list[Any] = []

    def _changed(self) -> None:
        """Note that this collection has changed in a way its index by name cannot follow one
        object at a time - an object moved, or the order changed: the index is built again."""
        self._stale = True

    def _included(self, obj: Any) -> None:
        """One object added at the end: into the index, unless it is stale already."""
        name = _name_of(obj)
        if name is None:
            self._unnamed.append(obj)  # still being made: indexed as soon as it can be
        elif not self._stale:
            self._by_name.setdefault(name, []).append(obj)
        self._stamp += 1

    def _excluded(self, obj: Any) -> None:
        """One object taken out: out of the index, unless it is stale already."""
        if not self._stale:
            name = _name_of(obj)
            if name is None:
                self._stale = True  # one that cannot say its name may still be in the index
            bucket = self._by_name.get(name) if name is not None else None
            if bucket and obj in bucket:
                bucket.remove(obj)
                if not bucket:
                    self._by_name.pop(str(name), None)
        self._stamp += 1

    def _settle(self) -> None:
        """Index the objects that could not say their name when they were added."""
        waiting, self._unnamed = self._unnamed, []
        for obj in waiting:
            name = _name_of(obj)
            if name is None:
                self._unnamed.append(obj)
            elif not self._stale:
                self._by_name.setdefault(name, []).append(obj)

    def _indexed(self) -> dict[str, list[Any]]:
        """The objects by name, in the order they are held, built when the index is stale.

        A directory asks its list for an object by name once per object appended - ROOT's
        name-clash check - so a scan over a thousand objects, with a Python call each, was
        most of what booking a histogram cost.
        """
        self._settle()
        marks = objects.RENAMED
        if self._stale or self._marks != marks:
            found: dict[str, list[Any]] = {}
            for item in self._items:
                name = _name_of(item)
                if name is not None:
                    found.setdefault(name, []).append(item)
            self._by_name = found
            self._stale = False
            self._marks = marks
        return self._by_name

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

    def append(self, obj: Any) -> None:
        self.Add(obj)

    def extend(self, other: Any) -> None:
        for obj in list(other):
            self.Add(obj)

    def remove(self, obj: Any) -> None:
        """``remove``: the first time ``obj`` is held taken out - a ``ValueError`` if never."""
        if self.Remove(obj) is None:
            raise ValueError("list.remove(x): x not in list")

    def count(self, obj: Any) -> int:
        return sum(1 for item in self._items if item == obj)

    def __add__(self, other: Any) -> Any:
        made = type(self)()
        made.extend(self)
        made.extend(other)
        return made

    def __mul__(self, times: int) -> Any:
        made = type(self)()
        for _ in range(int(times)):
            made.extend(self)
        return made

    __rmul__ = __mul__

    def __imul__(self, times: int) -> Any:
        held = list(self._items)
        for _ in range(int(times) - 1):
            self.extend(held)
        return self

    # -- ROOT's --------------------------------------------------------------------

    def GetName(self) -> str:
        """The collection's name - its class's, as ROOT's, when it was given none."""
        return self._name or self.ClassName()

    def SetName(self, name: Any) -> None:
        self._name = str(name)

    def Write(self, name: Any = None, option: int = 0, bufsize: int = 0) -> int:
        """``Write``: the collection under one key with ``TObject::kSingleKey``, else each
        object it holds under a key of its own - named ``name``, if given, as ROOT does."""
        if int(option) & SINGLE_KEY:
            return super().Write(name, option, bufsize)
        return sum(
            int(obj.Write(name, int(option) & ~SINGLE_KEY, bufsize))
            for obj in self._items
            if obj is not None
        )

    def Add(self, obj: Any, opt: str = "") -> None:
        """``Add(obj[, opt])``: at the end, ROOT's option naming the object when it names one."""
        _named_by(obj, opt)
        self._items.append(obj)
        self._included(obj)

    def AddLast(self, obj: Any) -> None:
        self._items.append(obj)
        self._included(obj)

    def AddFirst(self, obj: Any) -> None:
        self._items.insert(0, obj)
        self._changed()

    def AddAll(self, other: Any) -> None:
        self._items.extend(other)
        self._changed()

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
            found = self._indexed().get(name)
            return found[0] if found else None
        return next((item for item in self._items if item is name), None)

    def IndexOf(self, obj: Any) -> int:
        return next((at for at, item in enumerate(self._items) if item is obj), -1)

    def Remove(self, obj: Any) -> Any:
        """``Remove``: take ``obj`` out and hand it back, or ``None`` if it is not held."""
        at = self.IndexOf(obj)
        if at < 0:
            return None
        self._excluded(obj)
        return self._items.pop(at)

    def Clear(self, option: str = "") -> None:
        self._items.clear()
        self._by_name.clear()
        self._stamp += 1

    def Delete(self, option: str = "") -> None:
        from ...pycxx.runtime.objects import delete

        for item in self._items:
            if self._owner and item is not None:
                delete(item)
        self._items.clear()
        self._by_name.clear()
        self._stamp += 1

    def SetOwner(self, enable: bool = True) -> None:
        self._owner = bool(enable)

    def IsOwner(self) -> bool:
        return self._owner

    def MakeIterator(self, dir: bool = True) -> TIterator:
        """``MakeIterator(dir)``: a walk over the objects this holds, forward or back."""
        return TCollectionIter(self, dir)

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
                if item is not None:  # an array's empty slots list nothing
                    item.ls(option)
        finally:
            Indent.deeper(-1)


def _position(index: Any, size: int) -> int:
    """An index of a list ``size`` long - negative from the end - or what a list raises."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"list indices must be integers or slices, not {type(index).__name__}")
    at = index + size if index < 0 else index
    if not 0 <= at < size:
        raise IndexError("list index out of range")
    return at


class TSeqCollection(TCollection):
    """``TSeqCollection``: a collection in an order that can be indexed."""

    CLASS_TITLE = "Sequenceable collection ABC"

    def __getitem__(self, index: Any) -> Any:
        """``c[i]`` the object there; ``c[i:j:k]`` a collection of the same class of them."""
        if isinstance(index, slice):
            made = type(self)()
            made.extend(self._items[index])
            return made
        return self._items[_position(index, len(self._items))]

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice):
            self._items[index] = list(value)
        else:
            self._items[_position(index, len(self._items))] = value
        self._changed()

    def __delitem__(self, index: Any) -> None:
        if isinstance(index, slice):
            del self._items[index]
        else:
            del self._items[_position(index, len(self._items))]
        self._changed()

    def insert(self, index: int, obj: Any) -> None:
        self._items.insert(int(index), obj)
        self._changed()

    def pop(self, index: Any = -1) -> Any:
        found = self._items.pop(_position(index, len(self._items)))
        self._excluded(found)
        return found

    def reverse(self) -> None:
        self._items.reverse()
        self._changed()

    def sort(self, key: Any = None, reverse: bool = False) -> None:
        """``sort()``: ROOT's ``Sort``; with a ``key`` or ``reverse``, Python's sort of them."""
        if key is None and not reverse:
            self.Sort()
            return
        self._items.sort(key=key, reverse=reverse)
        self._changed()

    def index(self, obj: Any) -> int:
        at = self.IndexOf(obj)
        if at < 0:
            raise ValueError(f"{obj!r} is not in list")
        return at

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
        self._changed()

    def AddBefore(self, before: Any, obj: Any) -> None:
        self._items.insert(max(self.IndexOf(before), 0), obj)
        self._changed()

    def AddAfter(self, after: Any, obj: Any) -> None:
        at = self.IndexOf(after)
        self._items.insert(len(self._items) if at < 0 else at + 1, obj)
        self._changed()

    def After(self, obj: Any) -> Any:
        at = self.IndexOf(obj)
        return None if at < 0 else self.At(at + 1)

    def Before(self, obj: Any) -> Any:
        at = self.IndexOf(obj)
        return None if at <= 0 else self.At(at - 1)

    def RemoveAt(self, index: int) -> Any:
        if not 0 <= index < len(self._items):
            return None
        found = self._items.pop(index)
        self._changed()
        return found

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
        self._changed()

    def IsSorted(self) -> bool:
        names = [item.GetName() for item in self._items]
        return names == sorted(names)

    def Merge(self, li: Any) -> int:
        """``TSeqCollection::Merge``: each object here merged, by its own ``Merge``, with the
        object at its place in each collection of ``li``, as ROOT pairs them - one that has no
        ``Merge`` joined, at the end, by copies of them; how many objects were merged in."""
        if not self._mergeable(li):
            return 0
        merged, unmerged = 0, []
        for at, obj in enumerate(list(self._items)):
            others = [one.At(at) for one in li if one.At(at) is not None]
            if hasattr(obj, "Merge"):
                gathered = TList()
                gathered._items = others
                obj.Merge(gathered)
                merged += len(others)
            else:
                unmerged += others
        for one in unmerged:
            self.Add(one.Clone())
        return merged

    def _mergeable(self, li: Any) -> bool:
        """Whether there is anything to merge, and with collections - or ROOT's refusal."""
        from .messages import message

        where = f"{self.ClassName()}::Merge"
        if not self._items or li is None:
            message("Warning", where, "list is empty - nothing to merge")
        elif not len(li):
            message("Warning", where, "input list is empty - nothing to merge with")
        elif not all(isinstance(one, TSeqCollection) for one in li):
            message("Error", where, "some objects in the input list are not collections - "
                    "merging aborted")  # fmt: skip
        else:
            return True
        return False


class TList(TSeqCollection):
    """``TList``: a doubly linked list, which is a Python list here."""

    CLASS_TITLE = "Doubly linked list"

    def MakeIterator(self, dir: bool = True) -> TIterator:
        """``MakeIterator``: ROOT's ``TList`` makes a ``TListIter``, where a plain collection
        makes the one that walks its objects in order."""
        return TListIter(self, dir)

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

    def Add(self, obj: Any, opt: str = "") -> None:
        _named_by(obj, opt)
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

    CLASS_TITLE = "Doubly linked list with hashtable for lookup"

    def GetListForObject(self, obj: Any) -> TList:
        """``GetListForObject``: what shares the name of ``obj`` - or the name itself - in the
        order the list holds them, as its hash table's slot for that name does."""
        name = obj if isinstance(obj, str) else obj.GetName()
        made = TList()
        for item in self._items:
            if _named(item, str(name)):
                made.Add(item)
        return made


class TOrdCollection(TSeqCollection):
    """``TOrdCollection``: an ordered collection."""

    CLASS_TITLE = "An ordered collection"


class TObjArray(TSeqCollection):
    """``TObjArray``: an array of objects, with room made as it is asked for."""

    CLASS_TITLE = "An array of objects"

    def __init__(self, size: int = 16, lower: int = 0) -> None:
        super().__init__()
        self._lower = int(lower)

    def MakeIterator(self, dir: bool = True) -> TIterator:
        """``MakeIterator``: an array's walk leaves its empty slots out, as ROOT's does."""
        return TObjArrayIter(self, dir)

    def AddAt(self, obj: Any, index: int) -> None:
        """``AddAt``: into slot ``index``, growing the array with empty slots to reach it."""
        while len(self._items) <= index:
            self._items.append(None)
        self._items[index] = obj
        self._changed()

    def AddAtAndExpand(self, obj: Any, index: int) -> None:
        self.AddAt(obj, index)

    def Expand(self, size: int) -> None:
        del self._items[size:]
        self._items.extend([None] * (size - len(self._items)))
        self._changed()

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
        self._changed()

    def LowerBound(self) -> int:
        return self._lower

    def RemoveAt(self, index: int) -> Any:
        """``RemoveAt``: empty slot ``index`` and hand back what was there."""
        if not 0 <= index < len(self._items):
            return None
        found, self._items[index] = self._items[index], None
        self._changed()
        return found

    def Remove(self, obj: Any) -> Any:
        at = self.IndexOf(obj)
        return None if at < 0 else self.RemoveAt(at)


class TClonesArray(TObjArray):
    """``TClonesArray``: an array of objects of one class, as a tree's branch of them holds.

    ``new (clones[i]) TLine(...)`` puts an object in slot ``i``; ``Clear``
    empties every slot for the next entry; ``BypassStreamer`` asks for the
    objects to be written member by member, which is how a tree writes them.
    """

    CLASS_TITLE = "An array of clone objects"

    def __init__(self, classname: Any = "TObject", size: int = 1000, call_dtor: bool = False):
        super().__init__(size)
        self._class = getattr(classname, "GetName", lambda: str(classname))()
        self._bypass = False

    def GetClass(self) -> Any:
        from .objects import TNamed

        return TNamed(self._class, "")

    def SetClass(self, classname: Any, *rest: Any) -> None:
        """``SetClass(classname)``: the class every slot holds from now on. ROOT 5 read a
        second word as how many slots to make for it, which an array that grows to the
        slot asked of it has no use for."""
        self._class = getattr(classname, "GetName", lambda: str(classname))()

    def BypassStreamer(self, bypass: bool = True) -> None:
        self._bypass = bool(bypass)

    def CanBypassStreamer(self) -> bool:
        return self._bypass

    def Clear(self, option: str = "") -> None:
        self._items = []
        self._by_name.clear()
        self._stamp += 1

    def ConstructedAt(self, index: int) -> Any:
        """The object in slot ``index``, made with its class's default constructor if empty."""
        found = self.At(index) if 0 <= index < len(self._items) else None
        if found is None:
            import importlib

            found = getattr(importlib.import_module("xrdroot.pyroot"), self._class)()
            self.AddAt(found, index)
        return found


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


class TIterator(TObject):
    """``TIterator``: ROOT's iterator protocol - ``Next()`` hands out the objects one at a
    time and ``None`` past the last, ``operator*`` reads the one just handed out, and
    ``Reset`` walks them again.

    ROOT's iterators keep the collection and a cursor into it; one here keeps the objects
    themselves, taken when the iterator was made, which walks the same and cannot trip over
    the collection changing under it. A collection says which iterator it makes: a
    ``TList`` a :class:`TListIter`, an array a :class:`TObjArrayIter`, a map the pairs of a
    :class:`TMapIter`.
    """

    CLASS_TITLE = "Iterator abstract base class"

    def __init__(self, collection: Any = None, items: Any = (), forward: bool = True) -> None:
        super().__init__()
        self._collection = collection
        self._forward = bool(forward)
        self._items = list(items)[:: 1 if self._forward else -1]
        self._at = -1

    def Next(self) -> Any:
        """The next object, or ``None`` once there are no more - where the walk stays, so
        that ``operator*`` asks for one past the end and gets nothing."""
        self._at += 1
        return self._items[self._at] if self._at < len(self._items) else None

    def __call__(self) -> Any:
        """``operator*``: the object the walk is at, which is the one ``Next`` gave last."""
        return self._items[self._at] if 0 <= self._at < len(self._items) else None

    def Reset(self) -> None:
        """``Reset``: back to before the first object, so that the walk hands them all out
        again."""
        self._at = -1

    def Skip(self, count: int = 1) -> None:
        """``Skip(n)``: ``Next`` n times without keeping what it gives."""
        for _ in range(max(int(count), 0)):
            self.Next()

    def GetCollection(self) -> Any:
        return self._collection

    def GetOption(self) -> str:
        return ""

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        found = self.Next()
        if found is None:
            raise StopIteration
        return found

    def __eq__(self, other: object) -> bool:
        """``operator==``: two walks are the same when they are at the same place."""
        if not isinstance(other, TIterator):
            return NotImplemented
        return self._items == other._items and self._at == other._at

    def __ne__(self, other: object) -> bool:
        same = self.__eq__(other)
        return same if same is NotImplemented else not same

    __hash__ = TObject.__hash__

    def _destruct(self) -> None:
        """``~TIterator``: a walk holds nothing of its own to release."""


class TCollectionIter(TIterator):
    """``TCollectionIter``: a walk over any collection, the one it makes when it has no
    iterator of its own."""

    CLASS_TITLE = "Collection iterator"

    def __init__(self, collection: Any = None, dir: bool = True) -> None:
        held = [] if collection is None else list(collection)
        super().__init__(collection, held, dir)


class TListIter(TCollectionIter):
    """``TListIter``: a walk over a linked list, forward or back."""

    CLASS_TITLE = "Linked list iterator"


class TObjArrayIter(TCollectionIter):
    """``TObjArrayIter``: a walk over an array's objects, which leaves its empty slots out
    as ROOT's does."""

    CLASS_TITLE = "Object array iterator"

    def __init__(self, collection: Any = None, dir: bool = True) -> None:
        held = [] if collection is None else [item for item in collection if item is not None]
        super(TCollectionIter, self).__init__(collection, held, dir)


class THashTableIter(TCollectionIter):
    """``THashTableIter``: a walk over a hash table's objects."""

    CLASS_TITLE = "Iterator of hash table"


class TMapIter(TIterator):
    """``TMapIter``: a walk over a map, which hands out its ``TPair``s."""

    CLASS_TITLE = "Iterator of map"

    def __init__(self, collection: Any = None, dir: bool = True) -> None:
        held = [] if collection is None else [pair for pair in collection]
        super().__init__(collection, held, dir)

    def GetValue(self) -> Any:
        """``GetValue``: the value of the pair the walk is at, which is the last it gave."""
        pair = self.__call__()
        return None if pair is None else pair.Value()


class TIter:
    """``TIter``: ``Next()`` through a collection, ``None`` at the end, as ROOT's is.

    ROOT's ``TIter`` is a wrapper that owns the ``TIterator`` its collection made, deletes
    it when the wrapper is destroyed, and replaces it - deleting the one it held - when
    another iterator is assigned to it. So does this one: ``_destruct`` deletes what it
    holds, which is what ``delete p`` on one of ROOT's iterators does, and the wrapper is
    what a macro's ``TIter iter(&lst)`` holds.

    A Python list, dict or ``None`` is wrapped in a walk over the objects it has, which is
    how a walk of things that are not a collection is spelled here.
    """

    def __init__(self, collection: Any, dir: bool = True) -> None:
        self._iterator = _held_iterator(collection, dir)

    def Reset(self) -> None:
        if self._iterator is not None:
            self._iterator.Reset()

    def Next(self) -> Any:
        """The next object, or ``None`` once there are no more."""
        return None if self._iterator is None else self._iterator.Next()

    def __call__(self) -> Any:
        return self.Next()

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        found = self.Next()
        if found is None:
            raise StopIteration
        return found

    def GetCollection(self) -> Any:
        return None if self._iterator is None else self._iterator.GetCollection()

    def _destruct(self) -> None:
        """``~TIter``: the iterator this wrapper holds is deleted with it."""
        from ...pycxx.runtime.objects import delete

        held, self._iterator = self._iterator, None
        if held is not None:
            delete(held)

    def __del__(self) -> None:
        if not getattr(self, "_cxx_released", False) and not getattr(self, "_torn", False):
            self._torn = True
            self._destruct()


def _held_iterator(collection: Any, dir: bool = True) -> TIterator | None:
    """The iterator a ``TIter`` holds for what it was given: a collection's own, an
    iterator's as it is, and anything else wrapped in a walk over its objects."""
    if isinstance(collection, TIterator):
        return collection
    if isinstance(collection, TCollection):
        return collection.MakeIterator(dir)
    if collection is None:
        return None
    return TIterator(None, list(collection), dir)


class THashTable(TCollection):
    """``THashTable``: a collection found by hash - held in order here, which finds the same."""

    CLASS_TITLE = "A hash table"

    def MakeIterator(self, dir: bool = True) -> TIterator:
        """``MakeIterator``: ROOT's hash table makes a ``THashTableIter``."""
        return THashTableIter(self, dir)


def _text(other: Any) -> str | None:
    """What a ``TObjString`` is compared with as a string - a string, a ``TString`` or
    another ``TObjString`` - or ``None`` for anything else, which it is never equal to."""
    return str(other) if isinstance(other, (str, TObjString, TString)) else None


class TObjString(TNamed):
    """``TObjString``: a string that can be kept in a collection, compared and measured as
    its string is, and shown as Python shows a string."""

    CLASS_TITLE = "Collectable string class"

    def __init__(self, text: Any = "") -> None:
        super().__init__(str(text), "")

    def GetString(self) -> str:
        return self.GetName()

    def String(self) -> TString:
        """``String``: the string, as ROOT's ``const TString &`` - a ``TString`` rather than a
        Python string, so that ``String() + "*"`` is a ``TString`` too and answers ``Data``."""
        return TString(self.GetName())

    def SetString(self, text: Any) -> None:
        self.SetName(text)

    def __str__(self) -> str:
        return self.GetName()

    def __repr__(self) -> str:
        return repr(self.GetName())

    def __len__(self) -> int:
        return len(self.GetName())

    def __eq__(self, other: object) -> bool:
        return str(self) == _text(other)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)

    def __le__(self, other: Any) -> bool:
        return str(self) <= str(other)

    def __gt__(self, other: Any) -> bool:
        return str(self) > str(other)

    def __ge__(self, other: Any) -> bool:
        return str(self) >= str(other)

    def __hash__(self) -> int:
        return hash(self.GetName())


class TPair(TObject):
    """``TPair``: one entry of a ``TMap``, the key object and the value it holds.

    ROOT calls a pair after its key, which is how a ``TMap`` finds the pair of a key at
    all: the table of pairs hashes and compares what each pair is called.
    """

    CLASS_TITLE = "Pair of objects"

    def __init__(self, key: Any = None, value: Any = None) -> None:
        super().__init__()
        self._key = key
        self._value = value

    def GetName(self) -> str:
        """The key's name: a pair with no key has none, as ROOT's has."""
        return self._key.GetName() if self._key is not None else ""

    def GetTitle(self) -> str:
        return self.GetName()

    def Key(self) -> Any:
        return self._key

    def Value(self) -> Any:
        return self._value

    def SetKey(self, key: Any) -> None:
        self._key = key

    def SetValue(self, value: Any) -> None:
        self._value = value

    def IsSortable(self) -> bool:
        return True

    def Compare(self, obj: Any) -> int:
        """By key name, as ROOT compares the pairs of a map."""
        return int(self.GetName() > obj.GetName()) - int(self.GetName() < obj.GetName())


class TMap(THashTable):
    """``TMap``: a collection of ``TPair``s, looked up by key.

    ROOT keeps the pairs of a map in a ``THashTable``, so it hands them back in bucket
    order; here they are held in the order they were added, as the hash tables above are,
    which answers the same lookups and walks the same pairs.
    """

    CLASS_TITLE = "Map with keys and values"

    def __init__(self, name: Any = None, *rest: Any) -> None:
        """``TMap()``, or ``TMap(name)``: ROOT names a map after the object it is given,
        and a table held in order has no use for the capacity and load factor a hash of
        buckets takes."""
        super().__init__()
        if isinstance(name, TObject):
            self.SetName(name.GetName())
        elif name is not None:
            self.SetName(name)

    def Add(self, obj: Any, opt: Any = None) -> None:
        """``Add(key, value)``: a new pair at the end - or, when a pair with that key is
        already there, that pair's value replaced, as ROOT's own ``Add`` does. The second
        word is the value, since the option a collection's ``Add`` takes says nothing about
        a map."""
        found = self.FindObject(obj)
        if found is None:
            pair = TPair(obj, opt)
            self._items.append(pair)
            self._included(pair)
        else:
            found.SetValue(opt)

    def MakeIterator(self, dir: bool = True) -> TIterator:
        """``MakeIterator``: a map's walk hands out its ``TPair``s, as ROOT's does."""
        return TMapIter(self, dir)

    def Put(self, key: Any, value: Any) -> None:
        """``Put``: ROOT's old name for ``Add``, which it keeps as the same call."""
        self.Add(key, value)

    def FindObject(self, key: Any) -> Any:
        """The pair of ``key`` - by name for a string, else the key object itself - or
        ``None`` when the map holds no such key."""
        called = _text(key)
        if called is not None:
            return next((pair for pair in self._items if pair.GetName() == called), None)
        return next((pair for pair in self._items if pair.Key() is key), None)

    def GetValue(self, key: Any) -> Any:
        """``GetValue(key)``: the value that key holds, or ``None`` for a key not held."""
        found = self.FindObject(key)
        return None if found is None else found.Value()

    def GetKey(self, value: Any) -> Any:
        """``GetKey(value)``: the key of the entry that holds ``value`` - by name for a
        string, else the value object itself - or ``None``."""
        found = self._holding(value)
        return None if found is None else found.Key()

    def _holding(self, value: Any) -> TPair | None:
        """The pair whose value is ``value``, compared as ``GetValue``'s keys are: by the
        name a string gives it, else by being that object."""
        called = _text(value)
        if called is not None:
            return next((pair for pair in self._items if _named(pair.Value(), called)), None)
        return next((pair for pair in self._items if pair.Value() is value), None)

    def Remove(self, key: Any) -> Any:
        """``Remove(key)``: the pair taken out of the map, handing back the key it held as
        ROOT does, or ``None`` when no entry has that key."""
        found = self.FindObject(key)
        if found is None:
            return None
        self._items.remove(found)
        self._excluded(found)
        return found.Key()

    def Delete(self, option: str = "") -> None:
        """``Delete``: every key and every value deleted with the map, as ROOT's deletes
        them."""
        from ...pycxx.runtime.objects import delete

        for pair in self._items:
            for obj in (pair.Key(), pair.Value()):
                if obj is not None:
                    delete(obj)
        self._items.clear()
        self._by_name.clear()
        self._stamp += 1

    def __getitem__(self, key: Any) -> Any:
        return self.GetValue(key)

    def __call__(self, key: Any) -> Any:
        return self.GetValue(key)
