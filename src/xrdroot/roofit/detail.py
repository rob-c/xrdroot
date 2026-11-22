"""``RooFit::Detail``'s JSON document: a tree built a node at a time, then written out.

ROOT 6.40's HS3 interface (``roofit/hs3/inc/RooFit/Detail/JSONInterface.h``) hides a JSON
library behind two names, because its writers build a document node by node and walk it
later: a :class:`JSONTree` owns a document and a :class:`JSONNode` is one value in it - an
object, an array, or a scalar. This is the same two classes over Python's ``dict`` and
``list``: a node holds the container it stands in, so ``rootnode["map"] << aMap`` stores the
map where ``["map"]`` made room for it, and :meth:`JSONNode.writeJSON` is the text ROOT's
``writeJSON(std::ostream &)`` writes.

A node keeps its *children as nodes*, not as plain values, which is what makes a subscript
stay writable: ``rootnode["a"]["b"] << 1`` and ``rootnode["a"]`` name the same node.
"""

from __future__ import annotations

import json
from typing import Any, cast

__all__ = ["Detail", "JSONNode", "JSONTree"]


class JSONNode:
    """One value of a document: an object, an array, or a scalar.

    A node made with no value is an empty object, which is what ``rootnode["key"]`` makes for
    a key that is not there yet; :meth:`set_seq` makes an array, and :meth:`append_child` adds
    to it. ``<<`` stores a value of ROOT's that a writer hands it - a string, a number, a
    ``std::map`` or ``std::unordered_map`` of them, a ``TVectorD``, or a sequence.
    """

    __slots__ = ("_children", "_key", "_value")

    def __init__(self, key: Any = "", value: Any = None) -> None:
        self._key = str(key)
        #: The children of an object or an array; ``None`` for a scalar. An object's are by
        #: key, an array's in order, so the kind is decided by what a writer stored.
        self._children: Any = {}
        self._value = value
        if value is not None:
            self._from_value(value)

    # -- what a node is -------------------------------------------------------------

    def key(self) -> str:
        """The key this node is held under - ``""`` for the root."""
        return self._key

    def is_map(self) -> bool:
        return isinstance(self._children, dict)

    def is_seq(self) -> bool:
        return isinstance(self._children, list)

    def is_container(self) -> bool:
        return self.is_map() or self.is_seq()

    def is_null(self) -> bool:
        return self._value is None and not self.is_container()

    # -- what a node becomes --------------------------------------------------------

    def set_map(self) -> JSONNode:
        """Make this node an object, forgetting whatever it held, as ROOT's ``set_map`` does."""
        self._children, self._value = {}, None
        return self

    def set_seq(self) -> JSONNode:
        """Make this node an array, forgetting whatever it held."""
        self._children, self._value = [], None
        return self

    def set_null(self) -> JSONNode:
        """Make this node null: no container, no value."""
        self._children, self._value = None, None
        return self

    def clear(self) -> None:
        """Empty this node, keeping the kind of container it is."""
        if self.is_container():
            assert self._children is not None
            self._children.clear()
        self._value = None

    # -- the children of a node -----------------------------------------------------

    def __getitem__(self, key: Any) -> JSONNode:
        """``operator[]``: the object's child under ``key``, made empty when it is not there."""
        if not self.is_map():
            raise TypeError(
                f"a JSON node keyed by {key!r} is not an object: an object is what set_map() "
                "makes, and this node is an array or a scalar"
            )
        assert self._children is not None
        found = self._children.get(str(key))
        if found is None:
            found = self._children[str(key)] = JSONNode(key)
        return cast(JSONNode, found)

    def append_child(self) -> JSONNode:
        """``append_child``: a new empty child at the end of this array."""
        if not self.is_seq():
            raise TypeError(
                "a JSON node appended to is not an array: an array is what set_seq() makes, "
                "and this node is an object or a scalar"
            )
        assert self._children is not None
        made = JSONNode(len(self._children))
        self._children.append(made)
        return made

    def child(self, pos: int) -> JSONNode:
        """``child(pos)``: this array's element at ``pos``."""
        if not self.is_seq():
            raise TypeError(
                f"a JSON node's child {pos} is an array's, and this node is not an array"
            )
        assert self._children is not None
        return cast(JSONNode, self._children[int(pos)])

    def num_children(self) -> int:
        """How many children this node has: none, for a scalar."""
        return 0 if self._children is None else len(self._children)

    def has_child(self, key: Any) -> bool:
        """Is an object's child under ``key`` there - without making it, as ``[]`` would?"""
        return self.is_map() and self._children is not None and str(key) in self._children

    def __iter__(self) -> Any:
        """``children()``: the children in order, an object's and an array's alike."""
        if self._children is None:
            return iter(())
        return iter(self._children.values() if self.is_map() else self._children)

    def __len__(self) -> int:
        return self.num_children()

    # -- the scalar of a node -------------------------------------------------------

    def val(self) -> str:
        """``val()``: the scalar as text - ``true``, ``1``, ``1.5`` - as ROOT's ``val()`` has it."""
        if self.is_container():
            raise TypeError(
                f"the value of {self.key()!r} is asked of a scalar, and this node is a container"
            )
        if isinstance(self._value, bool):
            return "true" if self._value else "false"
        return "" if self._value is None else str(self._value)

    def val_int(self) -> int:
        return int(float(self.val()))

    def val_double(self) -> float:
        return float(self.val())

    def val_bool(self) -> bool:
        return self.val() in ("true", "1")

    # -- storing a value ------------------------------------------------------------

    def __lshift__(self, value: Any) -> JSONNode:
        """``operator<<``: ``value`` stored in this node, as ROOT's overloads store it."""
        self._from_value(value)
        return self

    def _from_value(self, value: Any) -> None:
        for kind in (str, bool, int, float):
            if isinstance(value, kind):
                self._children, self._value = None, value
                return
        if value is None:
            self.set_null()
            return
        items = _pairs(value)
        if items is not None:
            self.set_map()
            for key, item in items:
                self[key] << item
            return
        self.set_seq()
        for item in _items(value):
            self.append_child() << item

    # -- the document ---------------------------------------------------------------

    def _plain(self) -> Any:
        """What :mod:`json` writes for this node: plain Python values, children and all."""
        if self.is_map():
            assert self._children is not None
            return {key: child._plain() for key, child in self._children.items()}
        if self.is_seq():
            assert self._children is not None
            return [child._plain() for child in self._children]
        return self._value

    def writeJSON(self, stream: Any = None) -> str:
        """``writeJSON``: this node as JSON text - into ``stream`` when ROOT is given one."""
        text = json.dumps(self._plain(), indent=3, sort_keys=False)
        if stream is not None:
            if hasattr(stream, "write"):
                stream.write(text)
            else:
                raise TypeError("writeJSON takes a stream to write into, or nothing")
        return text

    def __repr__(self) -> str:
        return f"JSONNode({self._plain()!r})"


def _pairs(value: Any) -> list[tuple[Any, Any]] | None:
    """``value`` as the key/value pairs it is a map of, or ``None`` for anything else."""
    if isinstance(value, dict):
        return list(value.items())
    if hasattr(value, "keys") and type(value).__name__ != "TVectorD":
        return [(key, value[key]) for key in value.keys()]
    return None


def _items(value: Any) -> list[Any]:
    """``value`` as the items it is a sequence of: a ``TVectorD``'s elements, or the iterable."""
    if hasattr(value, "GetNrows") and hasattr(value, "__getitem__"):
        return [value[at] for at in range(int(value.GetNrows()))]
    try:
        return list(value)
    except TypeError as why:
        raise TypeError(
            f"a JSON node holds a string, a number, a map or a sequence of them, and "
            f"{type(value).__name__} is none of those: give it one of ROOT's containers"
        ) from why


class JSONTree:
    """An owned JSON document: ``JSONTree::create()``, then ``rootnode()``."""

    __slots__ = ("_root",)

    def __init__(self, text: str | None = None) -> None:
        self._root = JSONNode()
        if text is not None:
            self._root = _node_of(json.loads(text))

    @classmethod
    def create(cls, text: str | None = None) -> JSONTree:
        """``JSONTree::create()``, or ``create(text)``: a document from the JSON it reads."""
        return cls(text)

    def rootnode(self) -> JSONNode:
        return self._root

    def __repr__(self) -> str:
        return f"JSONTree({self._root._plain()!r})"


def _node_of(value: Any) -> JSONNode:
    """A document read out of JSON as the tree of nodes it stands for."""
    made = JSONNode()
    made << value
    return made


class Detail:
    """``RooFit::Detail``: the classes RooFit's own writers build a document with."""

    JSONNode = JSONNode
    JSONTree = JSONTree
