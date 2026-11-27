"""``TVirtualCollectionProxy``: how ROOT's I/O sees a container, whatever it is.

``TClass::GetCollectionProxy`` gives one for a standard container - which
container it is, what it holds, and its items one by one - and a program
derives its own from it to make a class of its read and written as a
collection. Here a proxy over a container the kit made answers from the
container itself: ``PushProxy`` points it at one, ``Size`` and ``At`` read it.
"""

from __future__ import annotations

from typing import Any

from ...cppstd.cells import ItemRef
from .objects import TObject

__all__ = ["TVirtualCollectionProxy", "EDataType"]

#: ``ROOT::ESTLType``, by the container's name.
STL_TYPES = {"vector": 1, "list": 2, "deque": 3, "map": 4, "multimap": 5, "set": 6,
             "multiset": 7, "bitset": 8, "forward_list": 9, "unordered_set": 10,
             "unordered_multiset": 11, "unordered_map": 12, "unordered_multimap": 13}  # fmt: skip


class EDataType:
    """``EDataType``: the codes ROOT gives the fundamental types."""

    kChar_t, kShort_t, kInt_t, kLong_t, kFloat_t, kCounter = 1, 2, 3, 4, 5, 6
    kCharStar, kDouble_t, kDouble32_t, kchar = 7, 8, 9, 10
    kUChar_t, kUShort_t, kUInt_t, kULong_t, kBits = 11, 12, 13, 14, 15
    kLong64_t, kULong64_t, kBool_t, kFloat16_t = 16, 17, 18, 19
    kVoid_t, kDataTypeAliasUnsigned_t, kDataTypeAliasSignedChar_t = 20, 21, 22
    kOther_t, kNoType_t, kNumDataTypes = -1, 0, 23


#: Each fundamental type's ``EDataType``, by name.
DATA_TYPES = {"char": 1, "short": 2, "int": 3, "long": 4, "float": 5, "double": 8,
              "Double32_t": 9, "unsigned char": 11, "unsigned short": 12, "unsigned int": 13,
              "unsigned long": 14, "long long": 16, "Long64_t": 16, "unsigned long long": 17,
              "ULong64_t": 17, "bool": 18, "Float16_t": 19}  # fmt: skip


class TVirtualCollectionProxy(TObject):
    """``TVirtualCollectionProxy``: a container's kind, contents and items, as I/O reads them."""

    #: How many bytes an iterator's arena holds, which a derived proxy checks its own against.
    fgIteratorArenaSize = 16
    #: ``TPushPop``: ROOT's guard that points a proxy at one collection for a block, defined below
    #: the class because it names the proxy as its argument.
    TPushPop: Any
    kIsInitialized, kIsAssociative, kIsEmulated, kNeedDelete, kCustomAlloc = 1, 2, 4, 8, 16

    def __init__(self, cl: Any = None) -> None:
        super().__init__()
        from .reflection import TClass

        self._class = cl if isinstance(cl, TClass) or cl is None else TClass(str(cl))
        self._held: list[Any] = []

    def _name(self) -> str:
        return self._class.GetName() if self._class is not None else ""

    def _parts(self) -> tuple[str, list[str]]:
        from ...wdeclared import parse_type

        if not self._name():
            return "", []
        found = parse_type(self._name())
        return found.name, [arg.text() for arg in found.args]

    # -- what the container is ---------------------------------------------------------------

    def GetCollectionClass(self) -> Any:
        return self._class

    def GetCollectionType(self) -> int:
        return STL_TYPES.get(self._parts()[0], 0)

    def GetProperties(self) -> int:
        associative = self.GetCollectionType() in (4, 5, 6, 7, 10, 11, 12, 13)
        return self.kIsInitialized | (self.kIsAssociative if associative else 0)

    def GetValueClass(self) -> Any:
        """The class of what the container holds - ``nullptr`` for numbers."""
        from .reflection import TClass

        name, args = self._parts()
        if name in ("map", "multimap", "unordered_map", "unordered_multimap"):
            return TClass.GetClass(f"pair<{args[0]},{args[1]}>")
        held = args[0].rstrip("*") if args else ""
        return None if not held or held in DATA_TYPES else TClass.GetClass(held)

    def GetType(self) -> int:
        """The ``EDataType`` of what the container holds, ``kOther_t`` for objects."""
        _name, args = self._parts()
        return DATA_TYPES.get(args[0], EDataType.kOther_t) if args else EDataType.kOther_t

    def HasPointers(self) -> bool:
        _name, args = self._parts()
        return bool(args) and args[0].endswith("*")

    def Sizeof(self) -> int:
        return 24

    def GetIncrement(self) -> int:
        return 8

    def Generate(self) -> TVirtualCollectionProxy:
        return TVirtualCollectionProxy(self._class)

    # -- the container it points at --------------------------------------------------------------

    def PushProxy(self, objectstart: Any) -> None:
        """``PushProxy``: point at the container ``objectstart``."""
        self._held.append(objectstart)

    def PopProxy(self) -> None:
        if self._held:
            self._held.pop()

    def _items(self) -> list[Any]:
        held = _unwrapped(self._held[-1]) if self._held else ()
        return list(held.items() if hasattr(held, "items") else held)

    def Size(self) -> int:
        return len(self._items())

    def At(self, index: int) -> Any:
        """``At``: element ``index`` - a number's or a pointer's own slot, which a program writes
        through (``*((int *)proxy->At(i)) = v``), and the object itself for a class."""
        held = _unwrapped(self._held[-1]) if self._held else None
        if held is None or not 0 <= index < len(held):
            return None
        if self._parts()[0] in ASSOCIATIVE:
            # a map or a set is keyed: its n-th element is the pair it holds, not ``held[n]``
            items = self._items()
            return items[index] if index < len(items) else None
        value = self._parts()[1][0] if self._parts()[1] else ""
        if self.GetType() == EDataType.kOther_t and not value.endswith("*"):
            return held[index]  # a collection of objects: the object itself
        return ItemRef(held, index)  # a number's, or a pointer's, own slot

    def Allocate(self, n: Any, force: bool = False) -> None:
        """``Allocate``: room for ``n`` elements of what the container holds, which ``At`` answers
        afterwards.

        A basic type gets a :class:`~xrdroot.cppstd.cells.Cell` per element, because ROOT fills such
        a collection by writing through the address ``At`` gives - ``*((int *)proxy->At(i)) = v`` in
        roottest's generator - and a cell is what Python has to write a number through. A class gets
        the objects the proxy's own class makes.
        """
        del force  # ROOT's flag asks for the arena's memory to be grown; a Python list grows itself
        made = [self._element() for _ in range(int(n))]
        held = _unwrapped(self._held[-1]) if self._held else None
        _refill(held, made)

    def _element(self) -> Any:
        """One fresh element: a basic type's zero, a pointer's null, or the class's own object."""
        value = self._parts()[1][0] if self._parts()[1] else "int"
        if value.endswith("*"):
            return None
        if self.GetType() == EDataType.kOther_t:
            # the *value* class, not the container's: ``New`` makes a container, and an element is
            # what the container holds
            from .reflection import TClass

            return TClass(value).New()
        return 0.0 if value in ("double", "float") else 0

    def Clear(self, opt: str = "") -> None:
        if self._held and hasattr(self._held[-1], "clear"):
            self._held[-1].clear()

    def New(self) -> Any:
        return self._class.New() if self._class is not None else None


def _unwrapped(held: Any) -> Any:
    """The container a proxy was pointed at, looked through the cell ``&collection`` makes.

    ``generate(vI)`` hands the proxy ``&vI``, a cell, and the collection is what the cell holds -
    ROOT's own ``GetObject()``-style unwrapping at the boundary of a runtime call.
    """
    return getattr(held, "value", held)


#: The containers a proxy indexes by key rather than by position.
ASSOCIATIVE = frozenset({
    "map", "multimap", "set", "multiset", "unordered_map", "unordered_multimap", "unordered_set",
    "unordered_multiset",
})


def _refill(held: Any, made: list[Any]) -> None:
    """Give the container ``held`` the elements ``made``: a list becomes them, a C++ container is
    cleared and pushed into - the protocol ROOT's own ``Allocate`` asks a collection for."""
    if isinstance(held, list):
        held[:] = made
    elif hasattr(held, "clear") and hasattr(held, "push_back"):
        held.clear()
        for one in made:
            held.push_back(one)


class _TPushPop:
    """``TVirtualCollectionProxy::TPushPop``: the proxy works on ``object`` while this is in scope.

    ROOT's guard makes a proxy point at one collection for the length of a block - roottest's
    generator wraps a fill in it - and restores what the proxy held as its scope ends.
    """

    __slots__ = ("_proxy", "_pushed")

    def __init__(self, proxy: TVirtualCollectionProxy, object: Any) -> None:
        self._proxy = proxy
        self._pushed = object is not None
        if self._pushed:
            proxy.PushProxy(object)

    def _destruct(self) -> None:
        if self._pushed:
            self._proxy.PopProxy()
            self._pushed = False


TVirtualCollectionProxy.TPushPop = _TPushPop
