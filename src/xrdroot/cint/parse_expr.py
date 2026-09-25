"""Reading C++ expressions, with C++'s precedence and its ambiguities guessed as a compiler would.

Binary operators are read by precedence climbing over :data:`LEVELS`; the
rest - prefix operators, casts in their four spellings, ``sizeof``, ``new``
and ``delete``, lambdas, braced lists, and the postfix calls, subscripts
and member accesses - each have a small reader. The guesses a C++ parser
must make are made from what is known about names: ``(T)x`` is a cast when
``T`` is a type, ``f<T>(x)`` a template call when ``T`` is.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .ctype import BUILTIN_WORDS, CType, builtin_name, canonical
from .cursor import KEYWORDS, KNOWN_TEMPLATES, STD_NAMES, NoParse
from .literals import number
from .nodes import (
    Assign,
    Binary,
    Block,
    Call,
    Capture,
    Cast,
    Comma,
    Delete,
    Expr,
    Index,
    InitList,
    Lambda,
    Literal,
    Member,
    Name,
    New,
    Param,
    SizeOf,
    Ternary,
    This,
    Throw,
    Unary,
)
from .parse_types import TypeParser
from .tokens import Token, unescape

__all__ = ["ExprParser", "LEVELS", "ASSIGNMENTS"]

#: The binary operators, loosest first.
LEVELS = [
    ("||", "or"),
    ("&&", "and"),
    ("|", "bitor"),
    ("^", "xor"),
    ("&", "bitand"),
    ("==", "!=", "not_eq"),
    ("<", ">", "<=", ">=", "<=>"),
    ("<<", ">>"),
    ("+", "-"),
    ("*", "/", "%"),
    (".*", "->*"),
]

#: The alternative spellings C++ has for some operators.
ALTERNATIVES = {"or": "||", "and": "&&", "bitor": "|", "xor": "^", "bitand": "&", "not_eq": "!="}

#: The assignment operators.
ASSIGNMENTS = frozenset("= += -= *= /= %= &= |= ^= <<= >>=".split())

#: The prefix operators.
PREFIX = frozenset("+ - ! ~ * & ++ -- not compl".split())

#: The four named casts.
NAMED_CASTS = {
    "static_cast": "static",
    "dynamic_cast": "dynamic",
    "reinterpret_cast": "reinterpret",
    "const_cast": "const",
}


class ExprParser(TypeParser):
    """The part of the parser that reads expressions."""

    def block(self) -> Block:
        raise NotImplementedError

    def parameters(self) -> tuple[list[Param], bool]:
        raise NotImplementedError

    # -- the precedence ladder ------------------------------------------------

    def expression(self) -> Expr:
        """A full expression, commas and all."""
        first = self.assignment()
        if not self.at_(","):
            return first
        items = [first]
        while self.accept(","):
            items.append(self.assignment())
        return Comma(first.where, items)

    def assignment(self) -> Expr:
        """An expression where a comma would end it: an argument, an initialiser."""
        if self.at_("throw"):
            where = self.take().where
            operand = None if self.at_(";", ")", ",") else self.assignment()
            return Throw(where, operand)
        left = self.ternary()
        token = self.peek()
        if token.kind == "op" and token.text in ASSIGNMENTS and not self._closes_angle(token):
            self.take()
            value = self.initializer_value()
            return Assign(token.where, token.text, left, value)
        return left

    def initializer_value(self) -> Expr:
        return self.braced() if self.at_("{") else self.assignment()

    def ternary(self) -> Expr:
        cond = self.binary(0)
        if not self.at_("?"):
            return cond
        where = self.take().where
        self.angle, saved = 0, self.angle
        try:
            yes = self.expression()
        finally:
            self.angle = saved
        self.expect(":")
        return Ternary(where, cond, yes, self.assignment())

    def constant(self) -> Any:
        return self.ternary()

    def _closes_angle(self, token: Token) -> bool:
        return self.angle > 0 and token.text in (">", ">>", ">=", ">>=")

    def binary(self, level: int) -> Expr:
        if level == len(LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        operators = LEVELS[level]
        while True:
            token = self.peek()
            if token.kind not in ("op", "id") or token.text not in operators:
                return left
            if self._closes_angle(token):
                return left
            self.take()
            right = self.binary(level + 1)
            left = Binary(token.where, ALTERNATIVES.get(token.text, token.text), left, right)

    # -- prefix ---------------------------------------------------------------

    def unary(self) -> Expr:
        token = self.peek()
        special = self._PREFIXED.get(token.text) if token.kind in ("id", "op") else None
        if special is not None:
            return special(self)
        if token.text in PREFIX and token.kind in ("op", "id"):
            self.take()
            op = {"not": "!", "compl": "~"}.get(token.text, token.text)
            return Unary(token.where, op, self.unary())
        if token.is_("("):
            cast = self.trial(self._c_cast)
            if cast is not None:
                return cast
        return self.postfix(self.primary())

    def _c_cast(self) -> Expr:
        where = self.take().where
        strict = self.trial(lambda: self.type_id(strict=True))
        ctype = strict or self.type_id()
        self.expect(")")
        if not _casts(strict is not None, ctype, self.peek()):
            raise NoParse
        return Cast(where, ctype, self.unary(), "c")

    def _sizeof(self) -> Expr:
        where = self.take().where
        if self.at_("..."):
            self.take()
        if self.at_("("):
            ctype = self.trial(self._bracketed_type)
            if ctype is not None:
                return SizeOf(where, ctype, None)
        return SizeOf(where, None, self.unary())

    def _bracketed_type(self) -> CType:
        self.expect("(")
        ctype = self.type_id(strict=True)
        self.expect(")")
        return ctype

    def _new(self) -> Expr:
        where = self.take().where
        if self.at_("("):
            raise self.refuse("placement new, which builds an object in memory given to it")
        ctype = self.pointers(self.specifiers().ctype)
        count = None
        if self.accept("["):
            count = self.expression()
            self.expect("]")
            while self.accept("["):
                self.expression()
                self.expect("]")
        if self.at_("("):
            return New(where, ctype, self.arguments(), count)
        if self.at_("{"):
            return New(where, ctype, self.braced().items, count, braces=True)
        return New(where, ctype, None, count)

    def _delete(self) -> Expr:
        where = self.take().where
        array = False
        if self.accept("["):
            self.expect("]")
            array = True
        return Delete(where, self.unary(), array)

    def _rooted(self) -> Expr:
        if self.peek(1).is_("new"):
            self.take()
            return self._new()
        if self.peek(1).is_("delete"):
            self.take()
            return self._delete()
        return self.postfix(self.primary())

    def _unsupported(self) -> Expr:
        word = self.peek().text
        raise self.refuse(f"{word} is not something this translator turns into Python")

    _PREFIXED: ClassVar[dict[str, Callable[[ExprParser], Expr]]] = {
        "sizeof": _sizeof,
        "alignof": _unsupported,
        "new": _new,
        "delete": _delete,
        "::": _rooted,
        "typeid": _unsupported,
        "co_await": _unsupported,
        "noexcept": _unsupported,
    }

    # -- postfix --------------------------------------------------------------

    def postfix(self, expr: Expr) -> Expr:
        while True:
            token = self.peek()
            handler = self._POSTFIX.get(token.text) if token.kind == "op" else None
            if handler is None or (token.is_("{") and not self._constructs(expr)):
                return expr
            expr = handler(self, expr)

    def _constructs(self, expr: Expr) -> bool:
        """Is ``expr {`` a type built from a braced list - ``T{a, b}`` - rather than a block?"""
        return isinstance(expr, Name) and self._type_name(expr)

    def _call(self, expr: Expr) -> Expr:
        where = self.peek().where
        return Call(where, expr, self.arguments())

    def arguments(self) -> list[Expr]:
        """``(a, b, c)``: the arguments of a call or constructor."""
        self.expect("(")
        self.angle, saved = 0, self.angle
        args: list[Expr] = []
        try:
            while not self.at_(")"):
                args.append(self.initializer_value())
                self.accept("...")
                if not self.accept(","):
                    break
            self.expect(")")
        finally:
            self.angle = saved
        return args

    def _index(self, expr: Expr) -> Expr:
        where = self.take().where
        self.angle, saved = 0, self.angle
        try:
            index = self.expression()
        finally:
            self.angle = saved
        self.expect("]")
        return Index(where, expr, index)

    def _member(self, expr: Expr) -> Expr:
        token = self.take()
        self.accept("template")
        if self.accept("~"):
            name = "~" + self.identifier()
        elif self.at_("operator"):
            name = self.operator_name()
        else:
            name = self.identifier()
        targs = self._member_targs() if self.at_("<") else None
        return Member(token.where, expr, name, token.text == "->", targs)

    def _member_targs(self) -> list[Any] | None:
        return self.trial(self._call_targs)

    def _increment(self, expr: Expr) -> Expr:
        token = self.take()
        return Unary(token.where, token.text, expr, postfix=True)

    def _braces(self, expr: Expr) -> Expr:
        assert isinstance(expr, Name)
        braced = self.braced()
        braced.ctype = CType(canonical(expr.text), expr.targs or [])
        return braced

    _POSTFIX: ClassVar[dict[str, Callable[[ExprParser, Expr], Expr]]] = {
        "(": _call,
        "[": _index,
        ".": _member,
        "->": _member,
        "++": _increment,
        "--": _increment,
        "{": _braces,
    }

    def _type_name(self, name: Name) -> bool:
        return bool(name.targs) or self.is_type(name.parts)

    # -- primaries ------------------------------------------------------------

    def primary(self) -> Expr:
        token = self.peek()
        reader = self._PRIMARY_KIND.get(token.kind)
        if reader is not None:
            return reader(self)
        special = self._PRIMARY_WORD.get(token.text)
        if special is not None:
            return special(self)
        if token.kind == "id" and token.text in BUILTIN_WORDS:
            return self._functional_cast()
        if token.kind == "id" and token.text not in KEYWORDS:
            return self.name()
        if token.is_("::"):
            return self.name()
        self.fail("an expression was expected")
        raise AssertionError  # pragma: no cover

    def _number(self) -> Expr:
        token = self.take()
        value, ctype = number(token.text, token.where)
        kind = "int" if isinstance(value, int) else "float"
        return Literal(token.where, kind, value, ctype)

    def _string(self) -> Expr:
        token = self.peek()
        parts = []
        while self.peek().kind == "str":
            parts.append(unescape(self.take().text))
        ctype = "char*"
        suffix = self.peek()
        if suffix.kind == "id" and not suffix.space and suffix.text in ("s", "sv"):
            self.take()
            ctype = "std::string"
        return Literal(token.where, "str", "".join(parts), ctype)

    def _character(self) -> Expr:
        token = self.take()
        text = unescape(token.text)
        value = 0
        for char in text:
            value = value * 256 + ord(char)
        return Literal(token.where, "char", value, "char")

    _PRIMARY_KIND: ClassVar[dict[str, Callable[[ExprParser], Expr]]] = {
        "num": _number,
        "str": _string,
        "chr": _character,
    }

    def _true(self) -> Expr:
        token = self.take()
        return Literal(token.where, "bool", token.text == "true", "bool")

    def _null(self) -> Expr:
        return Literal(self.take().where, "null", None, "nullptr_t")

    def _this(self) -> Expr:
        return This(self.take().where)

    def _parenthesised(self) -> Expr:
        self.take()
        self.angle, saved = 0, self.angle
        try:
            inner = self.expression()
        finally:
            self.angle = saved
        self.expect(")")
        return inner

    def _named_cast(self) -> Expr:
        token = self.take()
        self.expect("<")
        self.angle += 1
        try:
            ctype = self.type_id()
            self.split_shift()
            self.expect(">")
        finally:
            self.angle -= 1
        self.expect("(")
        operand = self.expression()
        self.expect(")")
        return Cast(token.where, ctype, operand, NAMED_CASTS[token.text])

    def _functional_cast(self) -> Expr:
        where = self.peek().where
        words = []
        while self.peek().kind == "id" and self.peek().text in BUILTIN_WORDS:
            words.append(self.take().text)
        ctype = CType(builtin_name(words))
        if self.at_("{"):
            items = self.braced()
            operand: Expr = items.items[0] if items.items else Literal(where, "int", 0, "int")
        else:
            args = self.arguments()
            operand = args[0] if args else Literal(where, "int", 0, "int")
        return Cast(where, ctype, operand, "functional")

    def _lambda(self) -> Expr:
        where = self.take().where
        captures = []
        while not self.at_("]"):
            captures.append(self._capture())
            if not self.accept(","):
                break
        self.expect("]")
        params: list[Param] = []
        if self.at_("("):
            params, _ = self.parameters()
        while self.accept("mutable", "constexpr", "noexcept"):
            pass
        returns = None
        if self.accept("->"):
            returns = self.type_id()
        return Lambda(where, captures, params, self.block(), returns)

    def _capture(self) -> Capture:
        where = self.where
        if self.accept("="):
            return Capture(where, "=", False)
        byref = self.accept("&")
        if self.at_("this"):
            self.take()
            return Capture(where, "this", False)
        if self.accept("*"):
            self.expect("this")
            return Capture(where, "this", False)
        if byref and self.at_(",", "]"):
            return Capture(where, "&", True)
        name = self.identifier()
        if self.accept("="):
            raise self.refuse(f"the lambda capture {name} = ..., which initialises a new name")
        return Capture(where, name, byref)

    def braced(self) -> InitList:
        """``{a, b, {c, d}}``: a braced list, as an initialiser or an argument."""
        where = self.expect("{").where
        self.angle, saved = 0, self.angle
        items: list[Expr] = []
        try:
            while not self.at_("}"):
                if self.at_("."):
                    raise self.refuse("a designated initialiser, .name = value")
                items.append(self.initializer_value())
                if not self.accept(","):
                    break
            self.expect("}")
        finally:
            self.angle = saved
        return InitList(where, items)

    _PRIMARY_WORD: ClassVar[dict[str, Callable[[ExprParser], Expr]]] = {
        "true": _true,
        "false": _true,
        "nullptr": _null,
        "NULL": _null,
        "this": _this,
        "(": _parenthesised,
        "[": _lambda,
        "{": braced,
        "static_cast": _named_cast,
        "dynamic_cast": _named_cast,
        "reinterpret_cast": _named_cast,
        "const_cast": _named_cast,
    }

    # -- names ----------------------------------------------------------------

    def operator_name(self) -> str:
        """``operator+``, ``operator()``, ``operator[]``, ``operator double``: as written."""
        self.expect("operator")
        if self.at_("(") and self.peek(1).is_(")"):
            self.take()
            self.take()
            return "operator()"
        if self.at_("[") and self.peek(1).is_("]"):
            self.take()
            self.take()
            return "operator[]"
        if self.at_("new", "delete"):
            raise self.refuse("a class's own operator new or delete")
        token = self.peek()
        if token.kind == "op":
            self.take()
            return "operator" + token.text
        target = self.pointers(self.specifiers().ctype)
        return f"operator {target.name}"

    def name(self) -> Expr:
        """A name, qualified or not, with the template arguments it is followed by."""
        where = self.where
        rooted = self.accept("::")
        parts: list[str] = []
        targs: list[Any] | None = None
        while True:
            self.accept("template")
            special = self._special_part()
            if special is not None:
                parts.append(special)
                break
            parts.append(self.identifier())
            if self.at_("<"):
                targs = self._name_targs(parts) or targs
            if not ((self.at_("::") and self.peek(1).kind == "id") or self._scoped_operator()):
                break
            self.take()
        return Name(where, self._standard(parts), targs, rooted)

    def _special_part(self) -> str | None:
        """``operator+`` or ``~Name`` where a name's next part stands, else ``None``."""
        if self.at_("operator"):
            return self.operator_name()
        return None

    def _standard(self, parts: list[str]) -> list[str]:
        """``vector`` as ``std::vector`` after ``using namespace std``, unless the macro's own."""
        if len(parts) != 1 or parts[0] not in STD_NAMES or self.is_variable(parts[0]):
            return parts
        if parts[0] in self.types or parts[0] in self.functions:
            return parts
        return ["std", parts[0]]

    #: The functions the macro defines, which a standard name may be hidden by.
    functions: set[str]

    def _scoped_operator(self) -> bool:
        return self.at_("::") and self.peek(1).is_("operator")

    def _name_targs(self, parts: list[str]) -> list[Any] | None:
        last = parts[-1]
        if last in KNOWN_TEMPLATES or last in self.templates:
            return self.trial(self.template_args)
        if self.is_variable(last) and len(parts) == 1:
            return None
        return self.trial(self._call_targs)

    def _call_targs(self) -> list[Any]:
        """Template arguments that must all be types, and be followed by ``(``, ``::`` or ``{``."""
        self.expect("<")
        self.angle += 1
        args: list[Any] = []
        try:
            while not self.at_(">", ">>"):
                number = self.peek().kind == "num"
                args.append(self._number() if number else self.type_id(strict=True))
                if not self.accept(","):
                    break
            self.split_shift()
            self.expect(">")
        finally:
            self.angle -= 1
        if not self.at_("(", "::", "{", ",", ")"):
            raise NoParse
        return args


def _casts(known: bool, ctype: CType, token: Token) -> bool:
    """Is ``(T)`` followed by ``token`` a cast - rather than a bracketed name in an expression?"""
    if not known and not _operand_after_unknown(ctype, token):
        return False
    if token.kind == "op" and token.text not in PREFIX and not token.is_("(", "{", "::"):
        return False
    return not (token.is_("*", "&", "+", "-") and not _pointerish(ctype))


def _operand_after_unknown(ctype: CType, token: Token) -> bool:
    """After ``(name)`` of no known type, only a pointer type or a plain operand makes a cast."""
    if token.kind == "id":
        return token.text not in KEYWORDS - {"this", "new"}
    return _pointerish(ctype) or token.kind == "num"


def _pointerish(ctype: CType) -> bool:
    return ctype.pointer > 0 or ctype.reference or ctype.name in BUILTIN_WORDS or ctype.arithmetic
