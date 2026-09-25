"""The tree an expression is parsed into, one small class per kind of thing in it.

Nothing here evaluates anything. A node says what was written - a branch and
the indices after it, an operator and its operands, a call - with every name
already resolved against the tree's branches, so what evaluates it never has
to look a name up again, and what asks which branches an expression reads
walks this rather than the text.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "Node",
    "Number",
    "Text",
    "Ref",
    "Size",
    "Special",
    "Unary",
    "Binary",
    "Ternary",
    "Cast",
    "Call",
    "Reduce",
    "Alt",
    "children",
]


class Node:
    """Anything an expression is made of."""

    __slots__ = ()


class Number(Node):
    """A number as written: ``3``, ``0x1f``, ``2.5f``, ``1e3``."""

    __slots__ = ("value",)

    def __init__(self, value: int | float) -> None:
        self.value = value


class Text(Node):
    """A string in quotes, which a column of strings is compared against."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value


#: One index after a branch: an expression, or ``None`` for ``[]``, every one.
Index = Optional[Node]


class Ref(Node):
    """A branch, and the indices written after it: ``pt``, ``m[2][]``, ``x[n-1]``."""

    __slots__ = ("column", "indices")

    def __init__(self, column: str, indices: tuple[Index, ...] = ()) -> None:
        self.column = column
        self.indices = indices


class Size(Node):
    """``x.size()`` or ``@x.size()``: how many elements a collection holds.

    With ``@`` - ``outer`` - it is the size of the collection itself; without,
    of each innermost collection, which for a collection of numbers is the
    same thing and for a vector of vectors is the size of each inner one.
    """

    __slots__ = ("ref", "outer")

    def __init__(self, ref: Ref, outer: bool = False) -> None:
        self.ref = ref
        self.outer = outer


class Special(Node):
    """One of ROOT's names for where the loop is: ``Entry$``, ``Iteration$`` and the rest."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class Unary(Node):
    """``-x``, ``!x``, ``~x``, ``+x``."""

    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand: Node) -> None:
        self.op = op
        self.operand = operand


class Binary(Node):
    """Two operands and the operator between them."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Node, right: Node) -> None:
        self.op = op
        self.left = left
        self.right = right


class Ternary(Node):
    """``condition ? then : otherwise``."""

    __slots__ = ("condition", "then", "otherwise")

    def __init__(self, condition: Node, then: Node, otherwise: Node) -> None:
        self.condition = condition
        self.then = then
        self.otherwise = otherwise


class Cast(Node):
    """``(int)x``, ``double(x)``: a value made the type named, as C++ converts it."""

    __slots__ = ("type", "operand")

    def __init__(self, type: str, operand: Node) -> None:
        #: The C++ spelling, which :data:`~.evaluate.CASTS` knows the NumPy type of.
        self.type = type
        self.operand = operand


class Call(Node):
    """A function of numbers: ``TMath::Abs(x)``, ``sqrt(x)``, ``strstr(s, "a")``."""

    __slots__ = ("name", "args")

    def __init__(self, name: str, args: tuple[Node, ...]) -> None:
        #: The name as written, which :data:`~.functions.FUNCTIONS` is keyed by.
        self.name = name
        self.args = args


class Reduce(Node):
    """``Sum$``, ``Length$``, ``Min$``, ``Max$``, ``MinIf$`` and ``MaxIf$``.

    Its arguments loop over their collections on their own, whatever the
    expression around them loops over, and give back one value per entry.
    """

    __slots__ = ("kind", "args")

    def __init__(self, kind: str, args: tuple[Node, ...]) -> None:
        self.kind = kind
        self.args = args


class Alt(Node):
    """``Alt$(primary, alternate)``: the first where it has a value, else the second."""

    __slots__ = ("primary", "alternate")

    def __init__(self, primary: Node, alternate: Node) -> None:
        self.primary = primary
        self.alternate = alternate


def _none(_node: Node) -> tuple[Node, ...]:
    return ()


def _ref(node: Node) -> tuple[Node, ...]:
    assert isinstance(node, Ref)
    return tuple(index for index in node.indices if index is not None)


def _size(node: Node) -> tuple[Node, ...]:
    assert isinstance(node, Size)
    return (node.ref,)


def _unary(node: Node) -> tuple[Node, ...]:
    assert isinstance(node, (Unary, Cast))
    return (node.operand,)


def _binary(node: Node) -> tuple[Node, ...]:
    assert isinstance(node, Binary)
    return (node.left, node.right)


def _ternary(node: Node) -> tuple[Node, ...]:
    assert isinstance(node, Ternary)
    return (node.condition, node.then, node.otherwise)


def _args(node: Node) -> tuple[Node, ...]:
    assert isinstance(node, (Call, Reduce))
    return node.args


def _alt(node: Node) -> tuple[Node, ...]:
    assert isinstance(node, Alt)
    return (node.primary, node.alternate)


#: How to find the nodes directly under each kind of node.
CHILDREN = {
    Number: _none,
    Text: _none,
    Special: _none,
    Ref: _ref,
    Size: _size,
    Unary: _unary,
    Cast: _unary,
    Binary: _binary,
    Ternary: _ternary,
    Call: _args,
    Reduce: _args,
    Alt: _alt,
}


def children(node: Node) -> tuple[Node, ...]:
    """The nodes directly under ``node``, in the order they were written."""
    return CHILDREN[type(node)](node)
