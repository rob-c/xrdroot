"""The text of a ``TFormula``, cut into words and parsed into :mod:`.nodes`.

The language is ROOT 6's: C++ arithmetic over the variables ``x``, ``y`` and
``z`` - also written ``x[0]``, ``x[1]`` and ``x[2]`` - and the parameters,
which are written in square brackets, by number as ``[0]`` or by name as
``[mean]``. ``^`` and ``**`` are powers, binding tighter than a unary minus
and to the right, so ``-x^2`` is ``-(x^2)`` and ``2^3^2`` is ``2^9``; the
constants are ``TFormula``'s, ``pi``, ``e``, ``ln10``, ``infinity`` and the
rest; the functions are those in :data:`~.library.CALLS`.

The predefined shapes - ``gaus``, ``pol3``, ``expo`` and the others - are
not part of this grammar: :mod:`.shapes` has already written them out as
the arithmetic they stand for, as ROOT does, before the text gets here.
Parameter names are resolved here, against the names the formula was
built with, so every :class:`~.nodes.Param` carries its index.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import NoReturn

from ..errors import UnsupportedFeatureError
from ..formula.errors import FormulaError
from .library import CALLS, CONSTANTS, PHYSICAL
from .nodes import Apply, Binary, Cond, Const, Node, Param, Unary, Var

__all__ = ["Token", "tokenize", "parse", "VARIABLES"]

#: The variables a formula may use, and the coordinate each one is.
VARIABLES = {"x": 0, "y": 1, "z": 2}

#: Every word the language has, tried in this order at each place in the text.
PATTERN = re.compile(
    r"""
    (?P<space>\s+)
    |(?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
    |(?P<param>\[[^\[\]]*\])
    |(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)
    |(?P<op>&&|\|\||==|!=|<=|>=|\*\*|[-+*/%^<>!?:(),])
    """,
    re.VERBOSE,
)

#: Binary operators and how tightly each binds, loosest first.
PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "%": 6,
}


class Token:
    """One word of a formula, and where in the text it starts."""

    __slots__ = ("kind", "text", "at")

    def __init__(self, kind: str, text: str, at: int) -> None:
        #: ``number``, ``param``, ``name``, ``op`` or ``end``.
        self.kind = kind
        self.text = text
        self.at = at

    def is_op(self, *texts: str) -> bool:
        return self.kind == "op" and self.text in texts


def tokenize(text: str) -> list[Token]:
    """The words of ``text``, ending in an ``end`` token so the parser never runs off."""
    tokens: list[Token] = []
    at = 0
    while at < len(text):
        found = PATTERN.match(text, at)
        if found is None:
            raise FormulaError(
                f"{text!r} has {text[at]!r} at character {at}, which is not part of any "
                f"formula TFormula reads"
            )
        kind = str(found.lastgroup)
        if kind != "space":
            tokens.append(Token(kind, found.group(), at))
        at = found.end()
    tokens.append(Token("end", "", len(text)))
    return tokens


class Parser:
    """A recursive descent over one formula's words."""

    def __init__(self, text: str, names: Mapping[str, int]) -> None:
        self.text = text
        self.names = names
        self.tokens = tokenize(text)
        self.at = 0

    @property
    def peek(self) -> Token:
        return self.tokens[self.at]

    def take(self) -> Token:
        token = self.tokens[self.at]
        self.at += 1
        return token

    def expect(self, op: str) -> None:
        if not self.peek.is_op(op):
            self.fail(f"a {op!r}")
        self.take()

    def fail(self, wanted: str) -> NoReturn:
        token = self.peek
        found = "the end" if token.kind == "end" else repr(token.text)
        raise FormulaError(
            f"{self.text!r} could not be parsed: {wanted} was expected at character "
            f"{token.at}, where there is {found}"
        )

    def parse(self) -> Node:
        if self.peek.kind == "end":
            raise FormulaError("an empty formula has nothing to evaluate")
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
        return Cond(condition, then, self.ternary())

    def binary(self, least: int) -> Node:
        left = self.unary()
        while self.peek.kind == "op" and PRECEDENCE.get(self.peek.text, 0) >= least:
            op = self.take().text
            left = Binary(op, left, self.binary(PRECEDENCE[op] + 1))
        return left

    def unary(self) -> Node:
        if self.peek.is_op("-", "+", "!"):
            op = self.take().text
            return Unary(op, self.unary())
        return self.power()

    def power(self) -> Node:
        base = self.primary()
        if not self.peek.is_op("^", "**"):
            return base
        self.take()
        return Binary("^", base, self.unary())

    def primary(self) -> Node:
        token = self.take()
        if token.kind == "number":
            return Const(float(token.text))
        if token.kind == "param":
            return self._param(token)
        if token.kind == "name":
            return self._name(token)
        if token.is_op("("):
            node = self.ternary()
            self.expect(")")
            return node
        self.at -= 1
        self.fail("a number, a parameter, a name or '('")

    def _param(self, token: Token) -> Node:
        label = token.text[1:-1].strip()
        if label.isdigit():
            return Param(int(label))
        index = self.names.get(label)
        if index is None:
            raise FormulaError(
                f"{token.text} in {self.text!r} is not a parameter of this formula; "
                f"it has {', '.join(f'[{name}]' for name in self.names) or 'none by name'}"
            )
        return Param(index)

    def _name(self, token: Token) -> Node:
        text = token.text
        if self.peek.is_op("("):
            return self._call(token)
        if text in VARIABLES:
            return self._variable(token)
        if text in CONSTANTS:
            return Const(CONSTANTS[text])
        if text in PHYSICAL:
            raise UnsupportedFeatureError(
                f"{text!r} in {self.text!r} is one of TFormula's physical constants, whose "
                f"value ROOT has changed between releases; write the number instead"
            )
        if text == "t":
            raise UnsupportedFeatureError(
                f"{self.text!r} uses t, a fourth variable, and there is no TF4 to hold it"
            )
        raise FormulaError(f"{text!r} in {self.text!r} is not a variable, a constant or a function")

    def _variable(self, token: Token) -> Node:
        if self.peek.kind == "param":
            raise FormulaError(
                f"{token.text}{self.peek.text} in {self.text!r} is not a variable: x[0], x[1] "
                f"and x[2] are x, y and z"
            )
        return Var(VARIABLES[token.text])

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

    def _call(self, token: Token) -> Node:
        name = token.text
        args = self._arguments()
        if name in CONSTANTS and not args:
            return Const(CONSTANTS[name])  # ``pi()``, as older formulas write it
        call = CALLS.get(name)
        if call is None:
            raise FormulaError(
                f"{name!r} in {self.text!r} is not a function TFormula knows here; it knows "
                f"TMath, <cmath> and the ROOT::Math densities its predefined shapes use"
            )
        if not call.least <= len(args) <= call.most:
            wanted = str(call.least) if call.least == call.most else f"{call.least} to {call.most}"
            raise FormulaError(
                f"{name} takes {wanted} arguments, and {self.text!r} gives it {len(args)}"
            )
        return Apply(name, call, args)


def parse(text: str, names: Mapping[str, int]) -> Node:
    """The tree ``text`` means, with parameter names resolved against ``names``."""
    return Parser(text, names).parse()
