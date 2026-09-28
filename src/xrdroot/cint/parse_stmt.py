"""Reading C++ statements: blocks, declarations, the loops and branches, and the jumps.

A statement that starts with a keyword is read by that keyword's reader.
Anything else is a declaration if it looks like one - a type, then a name,
then ``=``, ``(``, ``{``, ``[``, ``,`` or ``;`` - and an expression if it does
not; once it looks like a declaration it is read as one, so an error inside
it is reported as what it is rather than as a failed expression. ``goto`` and
its labels are refused by name: there is no Python for a jump.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from .ctype import CType
from .cursor import KEYWORDS, NoParse
from .errors import Where
from .nodes import (
    Block,
    Break,
    Case,
    ClassDecl,
    Continue,
    DeclStmt,
    DoWhile,
    Empty,
    EnumDecl,
    Expr,
    ExprStmt,
    For,
    Handler,
    If,
    Param,
    RangeFor,
    Return,
    Stmt,
    Switch,
    Try,
    Typedef,
    VarDecl,
    While,
)
from .parse_expr import ExprParser
from .parse_types import Specifiers

__all__ = ["StmtParser"]

#: What may follow a declarator's name in a declaration, and never in an expression statement.
DECLARATOR_ENDS = frozenset({"=", ";", "(", "[", ",", "{", ":"})


class StmtParser(ExprParser):
    """The part of the parser that reads statements."""

    def class_declaration(self, name: str | None = None, typedef: bool = False) -> ClassDecl:
        raise NotImplementedError

    def enum_declaration(self) -> EnumDecl:
        raise NotImplementedError

    # -- blocks -----------------------------------------------------------------

    def block(self) -> Block:
        where = self.expect("{").where
        self.push()
        body: list[Stmt] = []
        try:
            while not self.at_("}"):
                body.append(self.statement())
            self.take()
        finally:
            self.pop()
        return Block(where, body)

    def statement(self) -> Stmt:
        self.attributes()
        token = self.peek()
        handler = self._STATEMENTS.get(token.text) if token.kind in ("id", "op") else None
        if handler is not None:
            return handler(self)
        if token.kind == "id" and token.text not in KEYWORDS and self.peek(1).is_(":"):
            raise self.refuse(f"the label {token.text}:, which only a goto jumps to")
        if self.looks_declaration():
            return self.declaration_statement()
        expr = self.expression()
        self.expect(";")
        return ExprStmt(expr.where, expr)

    def attributes(self) -> None:
        """Pass over ``[[likely]]`` and its kind, which change nothing a translation does."""
        while self.at_("[") and self.peek(1).is_("["):
            self.take()
            self.skip_brackets()
            self.expect("]")

    # -- declarations -----------------------------------------------------------

    def looks_declaration(self) -> bool:
        """Does a declaration start here - a type, a declarator's name, then what ends one?"""
        return self.lookahead(self._declaration_head)

    def _declaration_head(self) -> bool:
        if self.peek().text in self.functions and self.peek(1).is_("("):
            raise NoParse
        spec = self.specifiers()
        if spec.ctype.is_auto and self._binding_ahead():
            return True
        self.declarator(spec.ctype)
        if not self.at_(*DECLARATOR_ENDS):
            raise NoParse
        return True

    def declaration_statement(self) -> DeclStmt:
        where = self.where
        spec = self.specifiers()
        if self._binding_ahead():
            decl = self.binding(spec)
            self.expect(";")
            return DeclStmt(where, [decl])
        decls = self.declarators(spec)
        self.expect(";")
        return DeclStmt(where, decls)

    def declarators(self, spec: Specifiers) -> list[VarDecl]:
        """``a = 1, *b, c[3]``: every variable a declaration's specifiers are shared by."""
        decls: list[VarDecl] = []
        while True:
            where = self.where
            name, ctype = self.declarator(spec.ctype)
            decl = VarDecl(where, name, ctype, static=spec.static)
            self.initializer(decl)
            self.declare(name)
            decls.append(decl)
            if not self.accept(","):
                return decls

    def initializer(self, decl: VarDecl) -> None:
        """What a declared variable starts as: ``= value``, ``(arguments)`` or ``{items}``."""
        if self.accept("="):
            decl.style = "="
            decl.init = self.initializer_value()
        elif self.at_("("):
            decl.args = self.arguments()
            decl.style = "()" if decl.args else None
            if self.at_("{"):
                raise self.refuse("a function defined inside another function")
        elif self.at_("{"):
            decl.args = self.braced().items
            decl.style = "{}"

    def _binding_ahead(self) -> bool:
        """``[a, b]`` or ``&[a, b]`` after ``auto``: a structured binding's names come next."""
        while self.at_("&", "&&") and self.peek(1).is_("["):
            self.take()
        return self.at_("[")

    def binding(self, spec: Specifiers) -> VarDecl:
        """``auto [a, b] = pair``: a structured binding, its names, and what it unpacks."""
        where = self.expect("[").where
        names = self.listed(self.identifier, "]")
        self.expect("]")
        for name in names:
            self.declare(name)
        decl = VarDecl(where, "_".join(names), spec.ctype, binding=names)
        if not self.at_(":"):
            self.initializer(decl)
        return decl

    # -- the statements ---------------------------------------------------------

    def _empty(self) -> Stmt:
        return Empty(self.take().where)

    def _block(self) -> Stmt:
        return self.block()

    def condition(self) -> Expr | VarDecl:
        """What ``if``, ``while`` and ``switch`` test: an expression, or a declaration."""
        if self.looks_declaration():
            spec = self.specifiers()
            name, ctype = self.declarator(spec.ctype)
            decl = VarDecl(self.where, name, ctype)
            self.initializer(decl)
            self.declare(name)
            return decl
        return self.expression()

    def _if(self) -> Stmt:
        where = self.take().where
        self.accept("constexpr")
        self.expect("(")
        self.push()
        try:
            init = self._if_init()
            cond = self.condition()
            self.expect(")")
            then = self.statement()
            orelse = self.statement() if self.accept("else") else None
        finally:
            self.pop()
        return If(where, cond, then, orelse, init)

    def _if_init(self) -> Stmt | None:
        """``if (auto x = f(); x > 0)``: the statement before the condition, if there is one."""
        start = self.at
        depth = 0
        while not (depth == 0 and self.at_(")", ";")):
            depth += self.at_("(", "[", "{") - self.at_(")", "]", "}")
            self.take()
        found = self.at_(";")
        self.at = start
        if not found:
            return None
        if self.looks_declaration():
            return self.declaration_statement()
        expr = self.expression()
        self.expect(";")
        return ExprStmt(expr.where, expr)

    def _while(self) -> Stmt:
        where = self.take().where
        self.expect("(")
        self.push()
        try:
            cond = self.condition()
            self.expect(")")
            body = self.statement()
        finally:
            self.pop()
        return While(where, cond, body)

    def _do(self) -> Stmt:
        where = self.take().where
        body = self.statement()
        self.expect("while")
        self.expect("(")
        cond = self.expression()
        self.expect(")")
        self.expect(";")
        return DoWhile(where, body, cond)

    def _for(self) -> Stmt:
        where = self.take().where
        self.expect("(")
        self.push()
        try:
            ranged = self.trial(self._range_head)
            if ranged is not None:
                iterable = self.initializer_value()
                self.expect(")")
                return RangeFor(where, ranged, iterable, self.statement())
            return self._classic_for(where)
        finally:
            self.pop()

    def _range_head(self) -> VarDecl:
        spec = self.specifiers()
        if self._binding_ahead():
            decl = self.binding(spec)
        else:
            where = self.where
            name, ctype = self.declarator(spec.ctype)
            self.declare(name)
            decl = VarDecl(where, name, ctype)
        if not self.accept(":"):
            raise NoParse
        return decl

    def _classic_for(self, where: Where) -> Stmt:
        init: Stmt | None = None
        if not self.accept(";"):
            if self.looks_declaration():
                init = self.declaration_statement()
            else:
                expr = self.expression()
                self.expect(";")
                init = ExprStmt(expr.where, expr)
        cond = None if self.at_(";") else self.expression()
        self.expect(";")
        step = None if self.at_(")") else self.expression()
        self.expect(")")
        body = self.statement()
        return For(where, init, cond, step, body)

    def _switch(self) -> Stmt:
        where = self.take().where
        self.expect("(")
        cond = self.expression()
        self.expect(")")
        if not self.at_("{"):
            raise self.refuse("a switch whose body is not a block")
        return Switch(where, cond, self.block())

    def _case(self) -> Stmt:
        where = self.take().where
        value = self.ternary()
        if self.at_("..."):
            raise self.refuse("a case range, case a ... b:, which is a GNU extension")
        self.expect(":")
        return Case(where, value)

    def _default(self) -> Stmt:
        where = self.take().where
        self.expect(":")
        return Case(where, None)

    def _break(self) -> Stmt:
        where = self.take().where
        self.expect(";")
        return Break(where)

    def _continue(self) -> Stmt:
        where = self.take().where
        self.expect(";")
        return Continue(where)

    def _return(self) -> Stmt:
        where = self.take().where
        value = None if self.at_(";") else self.initializer_value()
        self.expect(";")
        return Return(where, value)

    def _goto(self) -> Stmt:
        raise self.refuse("goto, which jumps to a label, and Python has no jump")

    def _asm(self) -> Stmt:
        raise self.refuse("inline assembly")

    def _try(self) -> Stmt:
        where = self.take().where
        body = self.block()
        handlers = []
        while self.at_("catch"):
            handlers.append(self._handler())
        return Try(where, body, handlers)

    def _handler(self) -> Handler:
        where = self.take().where
        self.expect("(")
        self.push()
        try:
            decl = None
            if self.accept("..."):
                pass
            else:
                spec = self.specifiers()
                ctype = self.pointers(spec.ctype)
                name = self.identifier() if self.peek().kind == "id" else ""
                decl = VarDecl(where, name, ctype)
                if name:
                    self.declare(name)
            self.expect(")")
            return Handler(where, decl, self.block())
        finally:
            self.pop()

    def _using(self) -> Stmt:
        where = self.take().where
        if self.accept("namespace"):
            self.skip_to(";")
            return Empty(where)
        if self.peek().kind == "id" and self.peek(1).is_("="):
            name = self.take().text
            self.take()
            ctype = self.type_id()
            self.expect(";")
            return self.alias(Typedef(where, name, ctype))
        self.skip_to(";")
        return Empty(where)

    def _typedef(self) -> Stmt:
        where = self.take().where
        if self.at_("struct", "class", "union") and self._defines_type():
            return self._typedef_class()
        if self.at_("enum") and self._defines_type():
            raise self.refuse("a typedef of an enum defined in place, typedef enum {...} T")
        spec = self.specifiers()
        name, ctype = self.declarator(spec.ctype)
        while self.accept(","):
            self.declarator(spec.ctype)
        self.expect(";")
        return self.alias(Typedef(where, name, ctype))

    def _typedef_class(self) -> Stmt:
        """``typedef struct [Tag] {...} T, *PT;``: the class, named ``Tag``, or ``T`` if unnamed."""
        named = not self.peek(1).is_("{")
        return self.class_declaration(None if named else self._alias_after_body(), typedef=True)

    def _alias_after_body(self) -> str:
        """The first name after the ``{...}`` of an unnamed class, which the typedef gives it."""
        start = self.at
        self.take()
        self.skip_brackets()
        name = self.identifier()
        self.at = start
        return name

    def _defines_type(self) -> bool:
        """Does ``struct X`` here go on to define ``X`` - ``{`` or a base list after its name?"""
        ahead = 1
        while self.peek(ahead).kind == "id" or self.peek(ahead).is_("::", "<"):
            ahead = self._past_angles(ahead) if self.peek(ahead).is_("<") else ahead + 1
        return self.peek(ahead).is_("{", ":")

    def _past_angles(self, ahead: int) -> int:
        """Where the ``<...>`` starting ``ahead`` tokens on ends, nested ones included."""
        depth = 0
        while True:
            token = self.peek(ahead)
            depth += token.is_("<") - token.is_(">")
            ahead += 1
            if depth <= 0 or token.kind == "eof":
                return ahead

    def alias(self, typedef: Typedef) -> Typedef:
        """Remember ``typedef``, so that its name is read as the type it stands for."""
        self.aliases[typedef.name] = typedef.ctype
        self.types.add(typedef.name)
        return typedef

    def skip_to(self, text: str) -> None:
        while not self.at_(text):
            if self.at_("(", "[", "{"):
                self.skip_brackets()
            else:
                self.take()
        self.take()

    def _local_namespace(self) -> Stmt:
        """``namespace GUI = ROOT::GUITutorials;`` in a function: a second name for one."""
        where = self.take().where
        name = self.identifier()
        self.expect("=")
        return self.namespace_alias(where, name)

    def namespace_alias(self, where: Where, name: str) -> Stmt:
        """What follows ``namespace name =``: the namespace ``name`` now stands for."""
        self.accept("::")
        parts = [self.identifier()]
        while self.accept("::"):
            parts.append(self.identifier())
        self.expect(";")
        self.namespace_aliases[name] = self.unaliased(parts)
        return Empty(where)

    def _static_assert(self) -> Stmt:
        where = self.take().where
        self.skip_brackets()
        self.expect(";")
        return Empty(where)

    def _local_type(self) -> Stmt:
        if not self._defines_type():
            return self.declaration_statement()
        if self.at_("enum"):
            return self.enum_declaration()
        return self.class_declaration()

    _STATEMENTS: ClassVar[dict[str, Callable[[StmtParser], Stmt]]] = {
        ";": _empty,
        "{": _block,
        "if": _if,
        "while": _while,
        "do": _do,
        "for": _for,
        "switch": _switch,
        "case": _case,
        "default": _default,
        "break": _break,
        "continue": _continue,
        "return": _return,
        "goto": _goto,
        "asm": _asm,
        "__asm__": _asm,
        "try": _try,
        "using": _using,
        "typedef": _typedef,
        "static_assert": _static_assert,
        "namespace": _local_namespace,
        "struct": _local_type,
        "class": _local_type,
        "union": _local_type,
        "enum": _local_type,
    }

    def parameters(self) -> tuple[list[Param], bool]:
        """``(int a, double b = 1)``: a function's parameters, and whether it ends in ``...``."""
        self.expect("(")
        params: list[Param] = []
        variadic = False
        if self.at_("void") and self.peek(1).is_(")"):
            self.take()
        while not self.at_(")"):
            if self.accept("..."):
                variadic = True
                break
            params.append(self.parameter())
            if not self.accept(","):
                break
        self.expect(")")
        return params, variadic

    def parameter(self) -> Param:
        where = self.where
        self.attributes()
        spec = self.specifiers()
        ctype = self.pointers(spec.ctype)
        name = None
        if self.at_("(") and self.peek(1).is_("*", "&"):
            name, ctype = self.declarator(ctype)
        elif self.peek().kind == "id" and self.peek().text not in KEYWORDS:
            name = self.identifier()
            ctype = self.dimensions(ctype)
        else:
            ctype = self.dimensions(ctype)
        if self.accept("..."):
            raise self.refuse("a parameter pack, which a variadic template expands")
        default = self.initializer_value() if self.accept("=") else None
        if name:
            self.declare(name)
        return Param(where, name, _decayed(ctype), default)


def _decayed(ctype: CType) -> CType:
    """An array parameter is a pointer, as C++ says: ``double x[]`` is ``double *x``.

    A reference to an array, ``double (&x)[N]``, stays the array it is.
    """
    if not ctype.dims or ctype.reference:
        return ctype
    return CType(ctype.name, ctype.args, ctype.pointer + 1, ctype.reference, ctype.const)
