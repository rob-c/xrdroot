"""Reading the types of declarations: specifiers, declarators and template arguments.

``static const unsigned int n``, ``TH1F *h[3]``, ``std::vector<std::pair<int,
double>> &v``, ``double (*f)(double)``: a type is specifiers (the base type and
its qualifiers, in any order C++ allows), then a declarator (``*``, ``&``, the
name, ``[N]``), and this module reads both into a :class:`CType`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .ctype import BUILTIN_WORDS, CType, builtin_name, canonical
from .cursor import KEYWORDS, STD_NAMES, Cursor, NoParse

__all__ = ["TypeParser", "Specifiers"]

#: Words that qualify a declaration without being its type.
QUALIFIERS = frozenset(
    """const volatile static extern inline constexpr consteval constinit mutable virtual
    explicit friend register thread_local typename __extension__ struct class union
    R__EXTERN __inline __restrict restrict""".split()
)

#: The words among those that say something the translation keeps.
KEPT = frozenset({"const", "constexpr", "static", "virtual", "friend", "explicit", "extern"})


@dataclass
class Specifiers:
    """What came before a declarator: the base type, and the qualifiers written with it."""

    ctype: CType
    words: set[str]

    @property
    def static(self) -> bool:
        return "static" in self.words


class TypeParser(Cursor):
    """The part of the parser that reads types."""

    #: The declared aliases, ``typedef`` and ``using``: name to the type it stands for.
    aliases: dict[str, CType]

    # -- specifiers -----------------------------------------------------------

    def specifiers(self, strict: bool = False) -> Specifiers:
        """The specifiers at the cursor; ``strict`` wants a name known to be a type."""
        words: set[str] = set()
        builtin: list[str] = []
        base: CType | None = None
        while True:
            token = self.peek()
            if token.kind != "id" and not token.is_("::"):
                break
            text = token.text
            if text in QUALIFIERS:
                self._qualifier(words, text)
            elif text in BUILTIN_WORDS and base is None:
                builtin.append(self.take().text)
            elif text == "decltype" and base is None and not builtin:
                base = self._decltype()
            elif base is None and not builtin and text not in KEYWORDS:
                base = self.named_type(strict)
            else:
                break
        return Specifiers(self._base(base, builtin, words), words & KEPT)

    def _qualifier(self, words: set[str], text: str) -> None:
        self.take()
        words.add(text)
        if text == "enum":
            self.accept("class", "struct")

    def _base(self, base: CType | None, builtin: list[str], words: set[str]) -> CType:
        if base is None and not builtin:
            raise NoParse
        if base is None:
            base = CType(builtin_name(builtin))
        if "const" in words or "constexpr" in words:
            base = replace(base, const=True)
        return base

    def _decltype(self) -> CType:
        self.take()
        self.expect("(")
        depth = 1
        while depth:
            depth += self.at_("(") - self.at_(")")
            self.take()
        return CType("auto")

    def named_type(self, strict: bool) -> CType:
        """A type by name, ``TH1F`` or ``std::map<int, TString>``, typedefs looked through."""
        parts, args = self.qualified_type()
        if strict and not self.is_type(parts):
            raise NoParse
        if len(parts) == 1 and self.is_variable(parts[0]):
            raise NoParse
        text = "::".join(parts)
        alias = self.aliases.get(text) or self.aliases.get(parts[-1])
        if alias is not None and not args:
            return replace(alias)
        if len(parts) == 1 and parts[0] in STD_NAMES and parts[0] not in self.types:
            text = "std::" + text
        return CType(canonical(text), args)

    def qualified_type(self) -> tuple[list[str], list[Any]]:
        """``a::b<...>::c<...>``: the parts of a type's name, and the last template arguments."""
        parts: list[str] = []
        args: list[Any] = []
        self.accept("::")
        while True:
            token = self.peek()
            if token.kind != "id" or token.text in KEYWORDS - {"template"}:
                raise NoParse
            self.accept("template")
            parts.append(self.take().text)
            args = self.template_args() if self.at_("<") else []
            if not (self.at_("::") and self.peek(1).kind == "id"):
                return parts, args
            self.take()

    # -- template arguments ---------------------------------------------------

    def template_args(self) -> list[Any]:
        """``<int, 3, std::string>``: each a :class:`CType` or an expression node."""
        self.expect("<")
        self.angle += 1
        args: list[Any] = []
        try:
            self.split_shift()
            while not self.at_(">"):
                args.append(self._template_arg())
                if not self.accept(","):
                    break
            self.split_shift()
            self.expect(">")
        finally:
            self.angle -= 1
        return args

    def _template_arg(self) -> Any:
        found = self.trial(self._typed_arg)
        if found is not None:
            return found
        return self.constant()

    def _typed_arg(self) -> CType:
        ctype = self.type_id()
        self.split_shift()
        if not self.at_(",", ">", "..."):
            raise NoParse
        self.accept("...")
        return ctype

    def constant(self) -> Any:
        """An expression where a template argument or an array size stands."""
        raise NotImplementedError

    # -- declarators ------------------------------------------------------------

    def type_id(self, strict: bool = False) -> CType:
        """A type with no name declared: ``const TH1*``, ``double(double)``, ``int[3]``."""
        base = self.specifiers(strict).ctype
        ctype = self.pointers(base)
        if self.at_("(") and self.peek(1).is_("*", "&"):
            self._function_pointer(abstract=True)
            return CType("function", callable=True)
        if self.at_("("):
            self.skip_brackets()
            return CType("function", callable=True)
        return self.dimensions(ctype)

    def pointers(self, base: CType) -> CType:
        """The ``*``, ``&`` and ``&&`` after a base type, and the ``const`` between them."""
        ctype = base
        while True:
            if self.accept("*"):
                ctype = replace(ctype, pointer=ctype.pointer + 1)
                while self.accept("const", "volatile", "__restrict", "restrict"):
                    pass
            elif self.accept("&", "&&"):
                ctype = replace(ctype, reference=True)
            else:
                return ctype

    def dimensions(self, ctype: CType) -> CType:
        """The ``[N][M]`` after a declarator's name."""
        dims: list[Any] = []
        while self.accept("["):
            dim = None if self.at_("]") else self.constant()
            literal = getattr(dim, "kind", None) == "int"
            dims.append(dim.value if literal else dim)  # type: ignore[union-attr]
            self.expect("]")
        return replace(ctype, dims=dims) if dims else ctype

    def declarator(self, base: CType) -> tuple[str, CType]:
        """A declarator that must name something: its name, and the full type it gives it."""
        ctype = self.pointers(base)
        if self.at_("(") and self.peek(1).is_("*", "&"):
            return self._function_pointer(abstract=False), CType("function", callable=True)
        name = self.identifier()
        return name, self.dimensions(ctype)

    def _function_pointer(self, abstract: bool) -> str:
        """``(*name)(params)`` or ``(Class::*name)(params)``: a pointer to something to call."""
        self.expect("(")
        name = ""
        while not self.at_(")"):
            token = self.take()
            if token.kind == "id":
                name = token.text
        self.expect(")")
        if not abstract and not name:
            raise NoParse
        self.dimensions(CType("function"))
        if self.at_("("):
            self.skip_brackets()
        return name

    def skip_brackets(self) -> None:
        """Pass over a bracketed group, whatever is inside it."""
        opening = self.take().text
        closing = {"(": ")", "[": "]", "{": "}", "<": ">"}[opening]
        depth = 1
        while depth:
            token = self.take()
            depth += token.is_(opening) - token.is_(closing)
