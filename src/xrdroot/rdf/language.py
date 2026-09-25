"""The C++ expressions ``RDataFrame`` takes as strings, parsed into a tree of nodes.

``df.Define("pt2", "pt * pt")`` and ``df.Filter("nMuon == 2")`` hand ROOT a
C++ expression, which it compiles with the columns as variables and
``ROOT::VecOps`` in scope. This parses the same expressions: C++'s operators
with C++'s precedence (``^`` is exclusive or, not a power), literals with
their suffixes and so their types, casts in all three spellings, calls of
``<cmath>``, ``TMath`` and ``ROOT::VecOps`` functions, indexing - by a
number, or by a mask as ``v[v > 30]`` - and the methods of an ``RVec`` that
an expression uses: ``size()``, ``empty()``, ``front()``, ``back()`` and
``at(i)``.

Names are resolved as the expression is parsed, against the columns the
frame has where it is defined, so a column nobody has is refused when
``Define`` is called rather than when the loop runs. What is C++ but not an
expression - a lambda, statements, ``auto`` - is refused by name, with the
Python callable that does the same job as the alternative.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, NoReturn

import numpy as np

from ..errors import UnsupportedFeatureError
from ..formula.errors import FormulaError
from ..formula.functions import CASTS
from ..formula.lexer import PATTERN, Token
from ..formula.names import Names

__all__ = [
    "Node",
    "Literal",
    "Column",
    "Unary",
    "Binary",
    "Ternary",
    "Cast",
    "Call",
    "Index",
    "Method",
    "parse",
    "columns_of",
    "TYPES",
]


class Node:
    """Anything an expression is made of."""

    __slots__ = ()


class Literal(Node):
    """A number or a string as written, with the C++ type its spelling gives it."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class Column(Node):
    """A column of the frame, by the name it has there."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class Unary(Node):
    __slots__ = ("op", "operand")

    def __init__(self, op: str, operand: Node) -> None:
        self.op = op
        self.operand = operand


class Binary(Node):
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Node, right: Node) -> None:
        self.op = op
        self.left = left
        self.right = right


class Ternary(Node):
    __slots__ = ("condition", "then", "otherwise")

    def __init__(self, condition: Node, then: Node, otherwise: Node) -> None:
        self.condition = condition
        self.then = then
        self.otherwise = otherwise


class Cast(Node):
    """``(int)x``, ``float(x)``, ``static_cast<double>(x)``: a value made another type."""

    __slots__ = ("dtype", "operand")

    def __init__(self, dtype: Any, operand: Node) -> None:
        self.dtype = dtype
        self.operand = operand


class Call(Node):
    """A function called by name: ``sqrt(x)``, ``Sum(v)``, ``ROOT::VecOps::DeltaR(...)``."""

    __slots__ = ("name", "args")

    def __init__(self, name: str, args: tuple[Node, ...]) -> None:
        self.name = name
        self.args = args


class Index(Node):
    """``v[i]``, or ``v[mask]``: an element of a collection, or the elements a mask keeps."""

    __slots__ = ("target", "index")

    def __init__(self, target: Node, index: Node) -> None:
        self.target = target
        self.index = index


class Method(Node):
    """``v.size()``, ``v.at(2)``: a method of the collection before the dot."""

    __slots__ = ("target", "name", "args")

    def __init__(self, target: Node, name: str, args: tuple[Node, ...]) -> None:
        self.target = target
        self.name = name
        self.args = args


def _children(node: Node) -> tuple[Node, ...]:
    if isinstance(node, (Unary, Cast)):
        return (node.operand,)
    if isinstance(node, Binary):
        return (node.left, node.right)
    if isinstance(node, Ternary):
        return (node.condition, node.then, node.otherwise)
    if isinstance(node, Index):
        return (node.target, node.index)
    if isinstance(node, Method):
        return (node.target, *node.args)
    if isinstance(node, Call):
        return node.args
    return ()


def columns_of(root: Node) -> tuple[str, ...]:
    """Every column an expression reads, once each, in the order they are first written."""
    found: dict[str, None] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, Column):
            found.setdefault(node.name)
        stack.extend(reversed(_children(node)))
    return tuple(found)


# -- the words -------------------------------------------------------------------


def tokenize(text: str) -> list[Token]:
    """The words of ``text``, ending in an ``end`` token so the parser never runs off."""
    tokens = []
    at = 0
    while at < len(text):
        found = PATTERN.match(text, at)
        if found is None:
            raise FormulaError(
                f"{text!r} has {text[at]!r} at character {at}, which is not part of the C++ "
                f"expressions a string given to RDataFrame may be here; for a lambda or "
                f"statements, give a Python callable instead"
            )
        if found.lastgroup != "space":
            tokens.append(Token(str(found.lastgroup), found.group(), at))
        at = found.end()
    tokens.append(Token("end", "", len(text)))
    return tokens


#: The C++ types a value can be cast to, and what NumPy calls each.
TYPES: dict[str, Any] = {
    **CASTS,
    "size_t": np.uint64,
    "std::size_t": np.uint64,
    **{
        f"std::{sign}int{bits}_t": getattr(np, f"{sign}int{bits}")
        for sign in ("", "u")
        for bits in (8, 16, 32, 64)
    },
}

#: Binary operators and how tightly each binds: C++'s precedence, loosest first.
PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "^": 4,
    "&": 5,
    "==": 6,
    "!=": 6,
    "<": 7,
    "<=": 7,
    ">": 7,
    ">=": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}

#: Names that are numbers in C++ and in ROOT's headers.
CONSTANTS: dict[str, Any] = {
    "true": np.bool_(True),
    "false": np.bool_(False),
    "kTRUE": np.bool_(True),
    "kFALSE": np.bool_(False),
    "M_PI": np.float64(np.pi),
    "M_E": np.float64(np.e),
}

#: The largest ``int`` C++ has, past which an unsuffixed literal is a ``long``.
INT_MAX = 2**31 - 1


def _integer(body: str, suffix: str, base: int) -> Any:
    value = int(body, base)
    unsigned, long = "u" in suffix, "l" in suffix
    if unsigned:
        return np.uint32(value) if not long and value < 2**32 else np.uint64(value)
    return np.int32(value) if not long and value <= INT_MAX else np.int64(value)


def number(text: str) -> Any:
    """A numeric literal as C++ reads it: its value, in the type its spelling gives it."""
    lower = text.lower()
    if lower.startswith("0x"):
        body = lower.rstrip("ul")
        return _integer(body, lower[len(body) :], 16)
    body = lower.rstrip("ful")
    suffix = lower[len(body) :]
    if any(mark in body for mark in ".e") or "f" in suffix:
        return np.float32(body) if "f" in suffix else np.float64(body)
    octal = len(body) > 1 and body.startswith("0")
    return _integer(body, suffix, 8 if octal else 10)


def _string(text: str) -> Any:
    body = text[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
    return np.str_(body)


# -- the grammar ---------------------------------------------------------------


class Parser:
    """A recursive descent over one expression's words."""

    def __init__(self, text: str, names: Collection[str], functions: Collection[str]) -> None:
        self.text = text
        self.names = Names(names)
        self.functions = functions
        self.tokens = tokenize(text)
        self.at = 0

    @property
    def peek(self) -> Token:
        return self.tokens[self.at]

    def take(self) -> Token:
        token = self.tokens[self.at]
        self.at += 1
        return token

    def expect(self, op: str) -> Token:
        if not self.peek.is_op(op):
            self.fail(f"a {op!r}")
        return self.take()

    def fail(self, wanted: str) -> NoReturn:
        token = self.peek
        found = "the end" if token.kind == "end" else repr(token.text)
        raise FormulaError(
            f"{self.text!r} could not be parsed: {wanted} was expected at character "
            f"{token.at}, where there is {found}"
        )

    def parse(self) -> Node:
        if self.peek.kind == "end":
            raise FormulaError("an empty expression has nothing to evaluate")
        node = self.ternary()
        if self.peek.kind != "end":
            self.fail("an operator or the end")
        return node

    def ternary(self) -> Node:
        condition = self.binary(1)
        if not self.peek.is_op("?"):
            return condition
        self.take()
        then = self.ternary()
        self.expect(":")
        return Ternary(condition, then, self.ternary())

    def binary(self, least: int) -> Node:
        left = self.unary()
        while self.peek.kind == "op" and PRECEDENCE.get(self.peek.text, 0) >= least:
            op = self.take().text
            left = Binary(op, left, self.binary(PRECEDENCE[op] + 1))
        return left

    def unary(self) -> Node:
        if self.peek.is_op("!", "-", "+", "~"):
            op = self.take().text
            return Unary(op, self.unary())
        cast = self._cast_ahead()
        if cast is not None:
            return Cast(cast, self.unary())
        return self.postfix(self.primary())

    def _cast_ahead(self) -> Any:
        """``(int)``, ``(unsigned int)``: a C cast, consumed if it is one."""
        if not self.peek.is_op("("):
            return None
        stop = self.at + 1
        while self.tokens[stop].kind == "name":
            stop += 1
        spelled = " ".join(token.text for token in self.tokens[self.at + 1 : stop])
        if spelled not in TYPES or not self.tokens[stop].is_op(")"):
            return None
        if self.names.find(spelled) is not None:
            return None  # a column called ``x`` in brackets is the column
        self.at = stop + 1
        return TYPES[spelled]

    def postfix(self, node: Node) -> Node:
        """What follows an operand: ``[index]`` and ``.method(...)``, any number of them."""
        while self.peek.is_op("[") or self.peek.kind == "member":
            if self.take().kind == "member":
                node = Method(node, self.tokens[self.at - 1].text[1:], self._arguments())
                continue
            index = self.ternary()
            self.expect("]")
            node = Index(node, index)
        return node

    def primary(self) -> Node:
        token = self.take()
        if token.kind == "number":
            return Literal(number(token.text))
        if token.kind == "string":
            return Literal(_string(token.text))
        if token.kind == "name":
            return self._name(token.text)
        if token.is_op("("):
            node = self.ternary()
            self.expect(")")
            return node
        self.at -= 1
        if token.is_op("["):
            raise UnsupportedFeatureError(
                f"{self.text!r} starts a lambda, which a string here cannot hold; give "
                f"Define or Filter a Python callable instead, which is handed the columns "
                f"a batch at a time"
            )
        self.fail("a number, a name or '('")

    def _arguments(self) -> tuple[Node, ...]:
        self.expect("(")
        args: list[Node] = []
        if not self.peek.is_op(")"):
            args.append(self.ternary())
            while self.peek.is_op(","):
                self.take()
                args.append(self.ternary())
        self.expect(")")
        return tuple(args)

    def _name(self, text: str) -> Node:
        if self.names.find(text) is not None:
            return Column(str(self.names.find(text)))
        if self.peek.is_op("("):
            return self._called(text)
        if text == "static_cast" and self.peek.is_op("<"):
            return self._static_cast()
        if text in CONSTANTS:
            return Literal(CONSTANTS[text])
        raise FormulaError(
            f"{text!r} in {self.text!r} is not a column of this frame, nor a name C++ "
            f"knows here; {self.names.nearest(text)}"
        )

    def _called(self, text: str) -> Node:
        if text in self.functions:
            return Call(text, self._arguments())
        if text in TYPES:
            (operand,) = self._exactly(text, self._arguments(), 1)
            return Cast(TYPES[text], operand)
        owner, _, method = text.rpartition(".")
        if owner and self.names.find(owner) is not None:
            return Method(Column(str(self.names.find(owner))), method, self._arguments())
        known = Names(list(self.functions))
        raise FormulaError(
            f"{text!r} in {self.text!r} is not a function this knows; {known.nearest(text)}"
        )

    def _exactly(self, what: str, args: tuple[Node, ...], count: int) -> tuple[Node, ...]:
        if len(args) != count:
            raise FormulaError(
                f"{what} takes {count} argument{'' if count == 1 else 's'}, and "
                f"{self.text!r} gives it {len(args)}"
            )
        return args

    def _static_cast(self) -> Node:
        self.expect("<")
        stop = self.at
        while self.tokens[stop].kind == "name":
            stop += 1
        spelled = " ".join(token.text for token in self.tokens[self.at : stop])
        if spelled not in TYPES:
            raise FormulaError(f"{spelled!r} in {self.text!r} is not a type this casts to")
        self.at = stop
        self.expect(">")
        (operand,) = self._exactly("static_cast", self._arguments(), 1)
        return Cast(TYPES[spelled], operand)


def parse(text: str, names: Collection[str], functions: Collection[str]) -> Node:
    """The tree of nodes ``text`` means, with its columns resolved against ``names``."""
    if not isinstance(text, str):
        raise TypeError(f"an expression is a string, and was given {type(text).__name__}")
    return Parser(text, names, functions).parse()
