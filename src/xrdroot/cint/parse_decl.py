"""Reading a macro's declarations: functions, classes, enums, namespaces, templates.

This is the top of the parser. A macro is a sequence of declarations - the
function named after the file among them - or a single brace-only block, an
unnamed macro. Before reading anything the parser runs once over the tokens
noting what names are declared as types and templates, since whether
``Foo * p;`` declares ``p`` depends on it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from .ctype import CType
from .cursor import KEYWORDS, NoParse
from .errors import Where
from .nodes import (
    Base,
    Block,
    ClassDecl,
    DeclStmt,
    Empty,
    EnumDecl,
    Expr,
    Function,
    Namespace,
    Stmt,
    Unit,
    VarDecl,
)
from .parse_stmt import StmtParser
from .parse_types import Specifiers
from .tokens import Token

__all__ = ["Parser", "parse"]

#: What may follow a function's parameter list, and never a constructor call's arguments.
AFTER_PARAMETERS = frozenset(
    "{ ; const override final noexcept = : -> throw try volatile & && mutable".split()
)

#: Words before a constructor's name that say nothing about what it builds.
CONSTRUCTOR_WORDS = frozenset({"explicit", "inline", "virtual", "constexpr"})

#: The access labels in a class body.
ACCESS = frozenset({"public", "private", "protected", "signals", "slots"})


class Parser(StmtParser):
    """A macro's tokens read into a :class:`Unit`."""

    def __init__(self, tokens: list[Token]) -> None:
        super().__init__(tokens)
        self.aliases = {}
        self.functions = set()
        #: The classes being read, innermost last: their names are their constructors'.
        self.classes: list[str] = []
        self._prescan()

    def _prescan(self) -> None:
        tokens = self.tokens
        depth = 0
        for index, token in enumerate(tokens[:-1]):
            depth += token.is_("{") - token.is_("}")
            name = _declared_type(tokens, index)
            if name is not None:
                self.types.add(name)
            elif token.kind == "id" and tokens[index + 1].is_("(") and depth == 0:
                self.functions.add(token.text)
        self._prescan_templates(tokens)

    def _prescan_templates(self, tokens: list[Token]) -> None:
        for index, token in enumerate(tokens[:-2]):
            if token.is_("template") and tokens[index + 1].is_("<"):
                name = _templated(tokens, index + 2)
                if name is not None:
                    self.templates.add(name)

    # -- the whole macro ------------------------------------------------------

    def unit(self) -> Unit:
        where = self.where
        decls: list[Stmt] = []
        unnamed: Block | None = None
        while self.peek().kind != "eof":
            if self.at_("{"):
                unnamed = self._unnamed(unnamed)
                continue
            decls.append(self.top())
        return Unit(where, decls, unnamed)

    def _unnamed(self, before: Block | None) -> Block:
        block = self.block()
        if before is None:
            return block
        before.body.extend(block.body)
        return before

    def top(self) -> Stmt:
        """One declaration at namespace scope."""
        self.attributes()
        token = self.peek()
        handler = self._TOP.get(token.text) if token.kind in ("id", "op") else None
        if handler is not None:
            return handler(self)
        return self.function_or_variable()

    def _namespace(self) -> Stmt:
        where = self.take().where
        name = None
        while self.peek().kind == "id":
            name = self.take().text
            self.accept("::")
        if self.accept("="):
            self.skip_to(";")
            return Empty(where)
        self.expect("{")
        body: list[Stmt] = []
        while not self.accept("}"):
            body.append(self.top())
        return Namespace(where, name, body)

    def _extern(self) -> Stmt:
        where = self.where
        if self.peek(1).kind != "str":
            return self.function_or_variable()
        self.take()
        self.take()
        if not self.accept("{"):
            return self.top()
        body: list[Stmt] = []
        while not self.accept("}"):
            body.append(self.top())
        return Namespace(where, None, body)

    def _type_at_top(self) -> Stmt:
        if not self._defines_type():
            return self.function_or_variable()
        if self.at_("enum"):
            return self.enum_declaration()
        return self.class_declaration()

    def _template(self) -> Stmt:
        where = self.take().where
        if not self.at_("<"):
            self.skip_to(";")
            return Empty(where)
        names = self.template_parameters()
        if self.at_("class", "struct", "union") and self._defines_type():
            decl = self.class_declaration()
            decl.template = names
            return decl
        if self.at_("class", "struct") and self.peek(2).is_(";"):
            self.skip_to(";")
            return Empty(where)
        stmt = self.function_or_variable()
        if isinstance(stmt, Function):
            stmt.template = names
        return stmt

    def template_parameters(self) -> list[str]:
        """``<typename T, int N = 3>``: the names a template declares, each now a type or value."""
        self.expect("<")
        self.angle += 1
        names: list[str] = []
        try:
            while not self.at_(">"):
                names.append(self._template_parameter())
                if not self.accept(","):
                    break
            self.split_shift()
            self.expect(">")
        finally:
            self.angle -= 1
        return names

    def _template_parameter(self) -> str:
        if self.at_("template"):
            raise self.refuse("a template template parameter")
        typename = self.accept("typename", "class")
        if self.accept("..."):
            raise self.refuse("a variadic template, whose parameter pack Python has no form for")
        if typename:
            name = self.identifier() if self.peek().kind == "id" else "_"
            self.types.add(name)
            if self.accept("="):
                self.type_id()
            return name
        spec = self.specifiers()
        name, _ = self.declarator(spec.ctype)
        if self.accept("="):
            self.ternary()
        return name

    _TOP: ClassVar[dict[str, Callable[[Parser], Stmt]]] = {
        ";": StmtParser._empty,
        "namespace": _namespace,
        "inline": lambda self: self._inline_namespace(),
        "using": StmtParser._using,
        "typedef": StmtParser._typedef,
        "static_assert": StmtParser._static_assert,
        "template": _template,
        "extern": _extern,
        "class": _type_at_top,
        "struct": _type_at_top,
        "union": _type_at_top,
        "enum": _type_at_top,
    }

    def _inline_namespace(self) -> Stmt:
        if self.peek(1).is_("namespace"):
            self.take()
            return self._namespace()
        return self.function_or_variable()

    # -- functions and variables ----------------------------------------------

    def function_or_variable(self) -> Stmt:
        """A function (declared or defined) or variables, at namespace or class scope."""
        where = self.where
        head = self._constructor_head()
        if head is not None:
            scope, name, kind = head
            void = CType("void")
            return self.function_rest(where, Specifiers(void, set()), void, scope, name, kind)
        spec = self.specifiers()
        if self.accept(";"):
            return Empty(where)
        ctype = self.pointers(spec.ctype)
        if self.at_("(") and self.peek(1).is_("*", "&"):
            return self._variables(where, spec, *self.declarator(ctype))
        scope, name = self.declared_name()
        if self.at_("(") and self._is_function():
            kind = "operator" if name.startswith("operator") else "function"
            return self.function_rest(where, spec, ctype, scope, name, kind)
        qualified = "::".join([*scope, name])
        return self._variables(where, spec, qualified, self.dimensions(ctype))

    def _variables(self, where: Where, spec: Specifiers, name: str, ctype: CType) -> Stmt:
        first = VarDecl(where, name, ctype, static=spec.static)
        self.initializer(first)
        if self.at_(":"):
            raise self.refuse(f"the bit-field {name}, a member narrower than its type")
        self.declare(name)
        decls = [first]
        if self.accept(","):
            decls.extend(self.declarators(spec))
        self.expect(";")
        return DeclStmt(where, decls)

    def _constructor_head(self) -> tuple[list[str], str, str] | None:
        start = self.at
        while self.accept(*CONSTRUCTOR_WORDS):
            pass
        found = self._in_class_head() if self.classes else self._out_of_class_head()
        if found is None:
            self.at = start
        return found

    def _in_class_head(self) -> tuple[list[str], str, str] | None:
        own = self.classes[-1]
        if self.at_(own) and self.peek(1).is_("("):
            return [], self.take().text, "constructor"
        if self.at_("~") and self.peek(1).is_(own) and self.peek(2).is_("("):
            self.take()
            return [], "~" + self.take().text, "destructor"
        return None

    def _out_of_class_head(self) -> tuple[list[str], str, str] | None:
        parts, ahead = self._scope_ahead()
        if not parts:
            return None
        last = self.peek(ahead)
        destructor = last.is_("~")
        if destructor:
            ahead += 1
            last = self.peek(ahead)
        if last.text != parts[-1] or not self.peek(ahead + 1).is_("("):
            return None
        self.at += ahead + 1
        kind = "destructor" if destructor else "constructor"
        return parts, ("~" if destructor else "") + last.text, kind

    def _scope_ahead(self) -> tuple[list[str], int]:
        """``A::B::`` at the cursor: its parts, and how many tokens they take."""
        ahead = 0
        parts: list[str] = []
        while self.peek(ahead).kind == "id" and self.peek(ahead + 1).is_("::"):
            parts.append(self.peek(ahead).text)
            ahead += 2
        return parts, ahead

    def declared_name(self) -> tuple[list[str], str]:
        """The name a declaration declares - ``f``, ``Foo::bar``, ``operator+`` - and its scope."""
        self.accept("::")
        parts: list[str] = []
        while True:
            if self.at_("operator"):
                return parts, self.operator_name()
            parts.append(self.identifier())
            if self.at_("<") and parts[-1] in self.templates:
                self.skip_brackets()
            if not (self.at_("::") and (self.peek(1).kind == "id" or self.peek(1).is_("~"))):
                return parts[:-1], parts[-1]
            self.take()
            if self.accept("~"):
                return parts, "~" + self.identifier()

    def _is_function(self) -> bool:
        return self.lookahead(self._parameters_then)

    def _parameters_then(self) -> bool:
        self.push()
        try:
            params, _ = self.parameters()
        finally:
            self.pop()
        if not self.at_(*AFTER_PARAMETERS):
            raise NoParse
        if self.at_(";") and not all(p.name or self.is_type([p.ctype.name]) for p in params):
            raise NoParse
        return True

    def function_rest(
        self,
        where: Where,
        spec: Specifiers,
        returns: CType,
        scope: list[str],
        name: str,
        kind: str,
    ) -> Stmt:
        """A function from its parameter list on: qualifiers, initialisers and its body."""
        self.functions.add(name)
        self.push()
        try:
            params, variadic = self.parameters()
            func = Function(where, name, returns, params, None, scope, kind)
            func.static, func.variadic = spec.static, variadic
            func.virtual = "virtual" in spec.words
            self._qualifiers(func)
            if self.accept("="):
                func.special = self.take().text
                self.expect(";")
                return func
            if self.accept(":"):
                func.inits = self._initialisers()
            if self.at_("try"):
                raise self.refuse("a function-try-block, a try around a whole function body")
            if not self.accept(";"):
                func.body = self.block()
        finally:
            self.pop()
        return func

    def _qualifiers(self, func: Function) -> None:
        while True:
            if self.accept("const"):
                func.const = True
            elif self.accept("noexcept", "throw"):
                if self.at_("("):
                    self.skip_brackets()
            elif self.accept("->"):
                func.returns = self.type_id()
            elif not self.accept("override", "final", "volatile", "&", "&&", "mutable"):
                return

    def _initialisers(self) -> list[tuple[str, list[Expr]]]:
        inits: list[tuple[str, list[Expr]]] = []
        while True:
            parts, _ = self.qualified_type()
            args = self.braced().items if self.at_("{") else self.arguments()
            inits.append(("::".join(parts), args))
            if not self.accept(","):
                return inits

    # -- classes and enums ------------------------------------------------------

    def class_declaration(self) -> ClassDecl:
        """``class Foo : public TObject { ... };`` - or its forward declaration."""
        where = self.where
        kind = self.take().text
        self.attributes()
        if self.peek().kind != "id" or self.peek().text in KEYWORDS:
            raise self.refuse(f"an unnamed {kind}, which has no name to make a Python class of")
        name = self.take().text
        self.types.add(name)
        if self.at_("<"):
            self.skip_brackets()
        self.accept("final")
        if self.accept(";"):
            return ClassDecl(where, name, kind, [], [], forward=True)
        bases = self._bases() if self.accept(":") else []
        decl = ClassDecl(where, name, kind, bases, self._members(name))
        if not self.at_(";"):
            decl.declarators = self.declarators(Specifiers(CType(name), set()))
        self.expect(";")
        return decl

    def _bases(self) -> list[Base]:
        bases: list[Base] = []
        while True:
            where = self.where
            access = "private"
            while self.at_("public", "private", "protected", "virtual"):
                word = self.take().text
                access = access if word == "virtual" else word
            bases.append(Base(where, self.named_type(False), access))
            if not self.accept(","):
                return bases

    def _members(self, name: str) -> list[Stmt]:
        self.expect("{")
        self.classes.append(name)
        self.push()
        members: list[Stmt] = []
        try:
            while not self.accept("}"):
                member = self.member()
                if member is not None:
                    members.append(member)
        finally:
            self.pop()
            self.classes.pop()
        return members

    def member(self) -> Stmt | None:
        """One member of a class body, or ``None`` for an access label or a stray ``;``."""
        self.attributes()
        if self.at_(*ACCESS):
            self.take()
            self.accept("slots")
            self.expect(":")
            return None
        if self.accept(";"):
            return None
        if self.at_("friend"):
            self._skip_friend()
            return None
        if self.at_("using"):
            return self._using()
        handler = self._MEMBER.get(self.peek().text)
        if handler is not None:
            return handler(self)
        return self.function_or_variable()

    def _skip_friend(self) -> None:
        while not self.accept(";"):
            if self.at_("{"):
                self.skip_brackets()
                return
            if self.at_("("):
                self.skip_brackets()
            else:
                self.take()

    _MEMBER: ClassVar[dict[str, Callable[[Parser], Stmt]]] = {
        "typedef": StmtParser._typedef,
        "template": _template,
        "class": _type_at_top,
        "struct": _type_at_top,
        "union": _type_at_top,
        "enum": _type_at_top,
        "static_assert": StmtParser._static_assert,
    }

    def _enumerators(self) -> list[tuple[str, Expr | None]]:
        """``{ kA, kB = 5 }``: each enumerator, and its value if one is written."""
        self.expect("{")
        items: list[tuple[str, Expr | None]] = []
        while not self.at_("}"):
            item = self.identifier()
            items.append((item, self.ternary() if self.accept("=") else None))
            if not self.accept(","):
                break
        self.expect("}")
        return items

    def enum_declaration(self) -> EnumDecl:
        """``enum Color { kA, kB = 5 };`` or ``enum class E : int { ... }``."""
        where = self.take().where
        scoped = self.accept("class", "struct")
        name = self.take().text if self.peek().kind == "id" else None
        if self.accept(":"):
            self.type_id()
        if name is not None:
            self.types.add(name)
            self.aliases[name] = CType("int")
        if self.accept(";"):
            return EnumDecl(where, name, [], scoped)
        decl = EnumDecl(where, name, self._enumerators(), scoped)
        if not self.at_(";"):
            decl.declarators = self.declarators(Specifiers(CType(name or "int"), set()))
        self.expect(";")
        return decl


def _declared_type(tokens: list[Token], index: int) -> str | None:
    """The type ``class X``, ``enum class X``, ``typename X`` or ``using X =`` declares here."""
    token, after = tokens[index], tokens[index + 1]
    if token.is_("class", "struct", "union", "enum", "typename") and after.kind == "id":
        return (tokens[index + 2] if after.is_("class", "struct") else after).text
    if token.is_("using") and after.kind == "id" and tokens[index + 2].is_("="):
        return after.text
    return None


def _templated(tokens: list[Token], ahead: int) -> str | None:
    """The name a ``template <...>`` starting at ``ahead`` declares: the one before ``(`` or ``{``."""
    depth = 1
    last = len(tokens) - 1
    while depth and ahead < last:
        depth += tokens[ahead].is_("<") - tokens[ahead].is_(">")
        ahead += 1
    while ahead < last and not tokens[ahead + 1].is_("(", "{", ":", ";"):
        ahead += 1
    return tokens[ahead].text if tokens[ahead].kind == "id" else None


def parse(tokens: list[Token]) -> Unit:
    """A preprocessed macro's tokens as a syntax tree."""
    return Parser(tokens).unit()
