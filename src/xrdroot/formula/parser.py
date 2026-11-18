"""From the text of an expression to the tree of :mod:`.nodes` it means.

The grammar is C++'s expression grammar, precedence and all, with ROOT's
additions to it: ``[]`` to loop over every element of a dimension, ``@`` in
front of a collection whose own ``size()`` is wanted, names ending in ``$``,
and ``^`` - which ROOT's formulas have always read as a power, as physicists
write ``px^2 + py^2``, and not as C's exclusive or.

Names are resolved here, against the branches the expression may read and
the aliases it was given, so that a name nothing has is refused when the
expression is compiled and not when it is first evaluated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn

from ..errors import UnsupportedFeatureError
from .errors import FormulaError
from .functions import CASTS, FUNCTIONS, STRING_FUNCTIONS
from .lexer import Token, tokenize
from .names import Names
from .nodes import (
    Alt,
    Binary,
    Call,
    Cast,
    Node,
    Number,
    Reduce,
    Ref,
    Size,
    Special,
    Ternary,
    Text,
    Unary,
)

__all__ = ["parse", "SPECIALS", "REDUCERS"]

#: Binary operators, and how tightly each binds: C's precedence, loosest first.
BINARY = {
    "||": 1,
    "&&": 2,
    "|": 3,
    "&": 4,
    "==": 5,
    "!=": 5,
    "<": 6,
    "<=": 6,
    ">": 6,
    ">=": 6,
    "<<": 7,
    ">>": 7,
    "+": 8,
    "-": 8,
    "*": 9,
    "/": 9,
    "%": 9,
}

#: ROOT's special names that stand on their own, without an argument list.
SPECIALS = ("Entry$", "Entries$", "LocalEntry$", "Iteration$", "Length$")

#: ROOT's special functions that make one value per entry of a collection, and their arity.
REDUCERS = {"Sum$": 1, "Length$": 1, "Min$": 1, "Max$": 1, "MinIf$": 2, "MaxIf$": 2}

#: Plain names that mean a number when no branch is called them.
CONSTANTS = {"pi": 3.141592653589793, "true": 1, "false": 0, "kTRUE": 1, "kFALSE": 0}

#: The methods a collection may be asked for, which is the one ROOT users ask for.
METHODS = ("size",)


def _number(text: str) -> Number:
    """A numeric literal as C++ reads it, suffixes and all."""
    if text[:2] in ("0x", "0X"):
        return Number(int(text.rstrip("uUlL"), 16))
    body = text.rstrip("fFuUlL")
    if any(mark in body for mark in ".eE") or text[-1] in "fF":
        return Number(float(body))
    return Number(int(body))


def _string(text: str) -> Text:
    body = text[1:-1]
    return Text(body.encode("latin-1", "backslashreplace").decode("unicode_escape"))


class Parser:
    """A recursive descent over one expression's tokens."""

    def __init__(
        self,
        text: str,
        names: Names,
        aliases: Mapping[str, str],
        expanding: tuple[str, ...] = (),
    ) -> None:
        self.text = text
        self.names = names
        self.aliases = aliases
        #: The aliases being expanded on the way here, to refuse one that is its own.
        self.expanding = expanding
        self.tokens = tokenize(text)
        self.at = 0

    # -- the tokens ---------------------------------------------------------

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

    # -- the grammar, loosest binding first -----------------------------------

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
        while self.peek.kind == "op" and BINARY.get(self.peek.text, 0) >= least:
            op = self.take().text
            left = Binary(op, left, self.binary(BINARY[op] + 1))
        return left

    def unary(self) -> Node:
        if self.peek.is_op("!", "-", "+", "~"):
            op = self.take().text
            return Unary(op, self.unary())
        cast = self._cast_ahead()
        if cast is not None:
            return Cast(cast, self.unary())
        return self.power()

    def _cast_ahead(self) -> str | None:
        """``(int)``, ``(unsigned int)``: a C cast, consumed if it is one."""
        if not self.peek.is_op("("):
            return None
        stop = self.at + 1
        while self.tokens[stop].kind == "name":
            stop += 1
        spelled = " ".join(token.text for token in self.tokens[self.at + 1 : stop])
        if spelled not in CASTS or not self.tokens[stop].is_op(")"):
            return None
        if self.names.find(spelled) is not None:
            return None  # a branch called ``int`` in brackets is the branch
        self.at = stop + 1
        return spelled

    def power(self) -> Node:
        base = self.primary()
        if not self.peek.is_op("^"):
            return base
        self.take()
        return Binary("^", base, self.unary())

    def primary(self) -> Node:
        token = self.take()
        handler = PRIMARY.get(token.kind)
        if handler is None:
            self.at -= 1
            self.fail("a number, a name or '('")
        return handler(self, token)

    def _op(self, token: Token) -> Node:
        if token.text == "(":
            node = self.ternary()
            self.expect(")")
            return node
        if token.text == "@":
            return self._outer()
        self.at -= 1
        self.fail("a number, a name or '('")

    def _outer(self) -> Node:
        """``@x.size()``: the size of the collection itself, not of what it holds."""
        node = self.primary()
        if not isinstance(node, Size):
            raise UnsupportedFeatureError(
                f"{self.text!r} puts '@' in front of something other than a collection "
                f"whose size() is asked for, which is the one use of '@' this reads"
            )
        return Size(node.ref, outer=True)

    # -- names ---------------------------------------------------------------

    def _name(self, token: Token) -> Node:
        if self.peek.is_op("("):
            return self._call(token)
        text = token.text
        if text.endswith("$"):
            return self._special(token)
        if text in self.aliases:
            return self._indexed(self._alias(text))
        return self._column(text)

    def _column(self, text: str) -> Node:
        found = self.names.resolve(text)
        if found is None:
            if text in CONSTANTS:
                return Number(CONSTANTS[text])
            raise FormulaError(
                f"{text!r} in {self.text!r} is not a branch, an alias or a name ROOT's "
                f"formulas know; {self.names.nearest(text, self.aliases)}"
            )
        column, rest = found
        if rest:
            self._member(text, column, rest)
        return self._indexed(Ref(column))

    def _member(self, text: str, column: str, rest: str) -> NoReturn:
        """Refuse a name that is no value an expression reads.

        ``column`` is a branch, or an object the tree holds whole, and ``rest`` is
        what of the name is left after it. Three ways a name is no value: it leads
        into an object - ``event.fH`` - which ROOT draws as nothing at all and
        this names as out of reach; it leads into an object this tree holds whole
        and this reader has learnt no such member of; or the object is split into
        names and this one is simply not among them, which the branches near it
        say, as ROOT's own refusal does.
        """
        deeper = sorted(name for name in self.names.names if name.startswith(f"{text}."))
        if deeper:
            raise UnsupportedFeatureError(
                f"{text!r} in {self.text!r} names {rest!r} of {column!r}, which is an object "
                f"and not a number or a string that an expression reads; name one of its "
                f"members instead, as {deeper[0]!r}"
            )
        if not any(name.startswith(f"{column}.") for name in self.names.names):
            raise UnsupportedFeatureError(
                f"{text!r} in {self.text!r} asks for {rest!r} inside {column!r}, which this "
                f"tree holds whole: what such an object holds is learnt from the objects "
                f"the tree's first entries hold, and none of them has {rest!r} - a member "
                f"behind a collection those entries leave empty is not known here"
            )
        raise FormulaError(
            f"{text!r} in {self.text!r} is not a member of {column!r} this tree has a branch "
            f"for; {self.names.nearest(text)}"
        )

    def _indexed(self, node: Node) -> Node:
        """The ``[i]`` and ``[]`` after a name, and a ``.size()`` after those."""
        if not self.peek.is_op("[") and self.peek.kind != "member":
            return node
        if not isinstance(node, Ref):
            raise UnsupportedFeatureError(
                f"{self.text!r} indexes an alias that stands for an expression rather than "
                f"a branch; index the branches inside the alias instead"
            )
        indices = list(node.indices)
        while self.peek.is_op("["):
            self.take()
            indices.append(None if self.peek.is_op("]") else self.ternary())
            self.expect("]")
        ref = Ref(node.column, tuple(indices))
        if self.peek.kind != "member":
            return ref
        size = self._sized(ref, self.take().text[1:])
        self.expect("(")
        self.expect(")")
        return size

    def _sized(self, ref: Ref, method: str) -> Node:
        if method not in METHODS:
            raise UnsupportedFeatureError(
                f"{self.text!r} calls {method}() on {ref.column!r}; the only method an "
                f"expression here may call is size(), on a collection"
            )
        return Size(ref)

    def _alias(self, text: str) -> Node:
        if text in self.expanding:
            chain = " -> ".join((*self.expanding, text))
            raise FormulaError(f"the alias {text!r} stands for itself, through {chain}")
        inner = Parser(self.aliases[text], self.names, self.aliases, (*self.expanding, text))
        return inner.parse()

    def _special(self, token: Token) -> Node:
        if token.text not in SPECIALS:
            raise FormulaError(
                f"{token.text!r} in {self.text!r} is not one of ROOT's special names; "
                f"there is {', '.join(SPECIALS)}, and the functions "
                f"{', '.join([*REDUCERS, 'Alt$'])}"
            )
        return Special(token.text)

    # -- calls ---------------------------------------------------------------

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
        for kind, build in CALLS:
            if kind(self, name):
                return build(self, name, self._arguments())
        args = self._arguments()
        node = self._object_call(name, args)
        if node is not None:
            return node
        if _is_member(self, name):
            return self._member_call(name, args)
        raise FormulaError(
            f"{name!r} in {self.text!r} is not a function ROOT's formulas know; "
            f"{Names(list(FUNCTIONS)).nearest(name)}"
        )

    def _object_call(self, name: str, args: tuple[Node, ...]) -> Node | None:
        """``GetNtrack()``, ``fH.GetXaxis().GetXmax()``: a member of an object, named by a getter.

        ROOT's formula asks the object's class for the method and calls what it
        finds; an object read from a file is a dictionary of its members, so a
        getter names a member instead - ``GetNtrack()`` is ``fNtrack`` - and
        the calls chained after it walk on through what each one gives. ``None``
        for a call that is nothing of the tree's, which the branch's own methods
        and ROOT's functions then have their say about.
        """
        parts = name.split(".")
        while not args and (step := self._chain()) is not None:
            parts.append(step)
            args = self._arguments() if self.peek.is_op("(") else ()
        found = self.names.member(parts)
        if found is not None:
            return self._indexed(Ref(found, args))
        owner = self.names.holds(parts)
        if owner is not None:
            raise UnsupportedFeatureError(
                f"{name}() in {self.text!r} is a method of {owner!r}, which this tree holds "
                f"whole: read from a file the object is a dictionary of its members, and the "
                f"only method a formula reads off one is the getter naming a member - "
                f"GetXmax() for fXmax; name the member the method works from instead"
            )
        return None

    def _chain(self) -> str | None:
        """The ``.Member`` written after a call, as the part of a name to go on with.

        ``.size()`` is not one: it is the length of the collection the name
        itself gave, which the name answers.
        """
        if self.peek.kind != "member":
            return None
        if self.peek.text[1:] in METHODS and self.tokens[self.at + 1].is_op("("):
            return None
        return self.take().text[1:]

    def _arity(self, name: str, args: tuple[Node, ...], least: int, most: int) -> None:
        if least <= len(args) <= most:
            return
        wanted = str(least) if least == most else f"{least} to {most}"
        raise FormulaError(
            f"{name} takes {wanted} argument{'' if wanted == '1' else 's'}, and "
            f"{self.text!r} gives it {len(args)}"
        )

    def _reducer(self, name: str, args: tuple[Node, ...]) -> Node:
        if name == "Alt$":
            self._arity(name, args, 2, 2)
            return Alt(args[0], args[1])
        self._arity(name, args, REDUCERS[name], REDUCERS[name])
        return Reduce(name, args)

    def _function(self, name: str, args: tuple[Node, ...]) -> Node:
        function = FUNCTIONS.get(name) or STRING_FUNCTIONS[name]
        self._arity(name, args, function.least, function.most)
        return Call(name, args)

    def _functional_cast(self, name: str, args: tuple[Node, ...]) -> Node:
        self._arity(name, args, 1, 1)
        return Cast(name, args[0])

    def _member_call(self, name: str, args: tuple[Node, ...]) -> Node:
        self._arity(f"{name}()", args, 0, 0)
        owner, _, method = name.rpartition(".")
        column = self._column(owner)
        assert isinstance(column, Ref)
        return self._sized(column, method)


def _is_reducer(parser: Parser, name: str) -> bool:
    return name in REDUCERS or name == "Alt$"


def _is_function(parser: Parser, name: str) -> bool:
    return name in FUNCTIONS or name in STRING_FUNCTIONS


def _is_cast(parser: Parser, name: str) -> bool:
    return name in CASTS


def _is_member(parser: Parser, name: str) -> bool:
    """``pt.size()``: a collection's own length, which is the only method a branch answers."""
    return "." in name and parser.names.resolve(name.rpartition(".")[0]) is not None


#: What a name followed by ``(`` can be, tried in order, and what each makes of it.
CALLS = (
    (_is_reducer, Parser._reducer),
    (_is_function, Parser._function),
    (_is_cast, Parser._functional_cast),
)


def _literal_number(parser: Parser, token: Token) -> Node:
    return _number(token.text)


def _literal_string(parser: Parser, token: Token) -> Node:
    return _string(token.text)


#: What each kind of token starts, where an operand is expected.
PRIMARY = {
    "number": _literal_number,
    "string": _literal_string,
    "name": Parser._name,
    "op": Parser._op,
}


def parse(text: str, names: Names, aliases: Mapping[str, str] | None = None) -> Node:
    """The tree of nodes ``text`` means, with its names resolved against ``names``."""
    return Parser(text, names, aliases or {}).parse()
