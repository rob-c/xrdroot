"""``typeid`` and ``std::type_info``: a type's identity, and the name the compiler mangles it to.

``typeid(x).name()`` is the Itanium C++ ABI's mangled name that GCC's and
Clang's libraries give: ``i`` for ``int``, ``P1A`` for ``A*``, ``5Point``
for ``Point``, ``N2ns5PointIdEE`` for a template in a namespace,
with the ABI's substitutions (``S_``, ``S0_``...) for a name repeated in
it. The translator spells the type it knows statically; the runtime
mangles it, so the same rules serve a type known only from a value - the
dynamic type of a polymorphic object.

Each type has one :class:`type_info`, so ``&typeid(A)`` is the same object
every time, as a map keyed by ``const type_info*`` needs. The library's
default template arguments (a ``vector``'s allocator) are not spelled out,
so the names of the standard containers are shorter than a library's.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["type_info", "typeid_of", "typeid_dynamic", "bad_typeid"]

#: Each built-in type's one-letter (or ``D``-prefixed) code in the Itanium ABI.
BUILTINS = {
    "void": "v",
    "bool": "b",
    "char": "c",
    "signed char": "a",
    "unsigned char": "h",
    "short": "s",
    "unsigned short": "t",
    "int": "i",
    "unsigned int": "j",
    "long": "l",
    "unsigned long": "m",
    "long long": "x",
    "unsigned long long": "y",
    "float": "f",
    "double": "d",
    "long double": "e",
    "wchar_t": "w",
    "char16_t": "Ds",
    "char32_t": "Di",
    "char8_t": "Du",
    "std::nullptr_t": "Dn",
    "nullptr_t": "Dn",
}

#: The names of ``std::string``, which every library mangles through its own inline namespace;
#: given as libstdc++ (C++11 ABI) gives it.
STRING_NAMES = frozenset({"std::string", "string"})
#: libstdc++'s mangled ``std::string``.
STRING_MANGLED = "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"

#: The words a built-in type is spelled with.
BUILTIN_WORDS = frozenset(
    "void bool char short int long float double signed unsigned wchar_t char16_t char32_t "
    "char8_t".split()
)

#: A spelling's tokens: names, ``::``, brackets, commas, stars, ampersands and numbers.
TOKEN = re.compile(r"\s*(::|[A-Za-z_]\w*|\d+|[<>,*&\[\]])")

#: A parsed type: ``("builtin", code)``, ``("class", components)``, ``("pointer", type)``,
#: ``("const", type)`` or ``("array", n, type)``; a component is ``(name, args or None)``.
Type = tuple[Any, ...]


class bad_typeid(Exception):
    """``std::bad_typeid``: ``typeid(*p)`` of a null pointer to a polymorphic class."""


class type_info:
    """``std::type_info``: one type's identity, compared, ordered and hashed by its name."""

    __slots__ = ("_mangled", "_spelled")

    def __init__(self, mangled: str, spelled: str) -> None:
        self._mangled = mangled
        self._spelled = spelled

    def name(self) -> str:
        """The mangled name, as GCC's and Clang's ``type_info::name`` give it."""
        return self._mangled

    def hash_code(self) -> int:
        """A number the same for the same type: FNV-1a of the mangled name, 64 bits."""
        value = 0xCBF29CE484222325
        for byte in self._mangled.encode():
            value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return value

    def before(self, other: type_info) -> bool:
        """The library's order of types: their mangled names compared, as libstdc++ does."""
        return self._mangled < other._mangled

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type_info) and other._mangled == self._mangled

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self._mangled)

    def __str__(self) -> str:
        """The type as a reflection dictionary names it - ``vector<int>``, no ``std::``."""
        return self._spelled.replace("std::", "")

    def __repr__(self) -> str:
        return f"<type_info {self._mangled}>"


#: Every type_info made, by mangled name: one object per type.
_INTERNED: dict[str, type_info] = {}


def typeid_of(spelled: str) -> type_info:
    """``typeid(T)``, ``T`` spelled as C++ writes it: ``int``, ``const A*``, ``ns::B<int>``."""
    mangled = mangle(spelled)
    found = _INTERNED.get(mangled)
    if found is None:
        found = _INTERNED[mangled] = type_info(mangled, _plain(spelled))
    return found


def typeid_dynamic(value: Any, static: str | None = None) -> type_info:
    """``typeid(e)`` of an object whose class may be one derived from its static type.

    The program's classes say what they are; a value that says nothing of
    its class is taken at the static type, when there is one, else at the
    type a Python value of its kind stands for.
    """
    if value is None:
        raise bad_typeid("typeid of a null pointer to a polymorphic class: std::bad_typeid")
    named = _class_name(value)
    if named is None:
        named = static if static is not None else _value_type(value)
    return typeid_of(named)


def _class_name(value: Any) -> str | None:
    """The C++ class a value is an object of, when it says: the program's own classes do."""
    layout = getattr(type(value), "_cxx_layout_", None)
    return str(layout[0]) if layout is not None else None


def _value_type(value: Any) -> str:
    """The C++ type a Python value with no class of its own stands for."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "std::string"
    return type(value).__name__


def _plain(spelled: str) -> str:
    """``spelled`` with its top-level ``const`` and reference taken off, which typeid ignores."""
    text = spelled.strip().rstrip("&").strip()
    if text.startswith("const ") and "*" not in text:
        text = text[len("const ") :]
    return text


# -- mangling ---------------------------------------------------------------


def mangle(spelled: str) -> str:
    """The Itanium ABI's mangled name of the type ``spelled``, top-level cv and ``&`` dropped."""
    tokens = TOKEN.findall(spelled)
    parsed, _ = _type(tokens, 0)
    while parsed[0] == "const":
        parsed = parsed[1]
    return _Mangler().type(parsed)


def _type(tokens: list[str], at: int) -> tuple[Type, int]:
    """A whole type from ``tokens[at]``: its base, then its stars, consts and dimensions."""
    const = False
    while at < len(tokens) and tokens[at] in ("const", "volatile"):
        const = const or tokens[at] == "const"
        at += 1
    base, at = _base(tokens, at)
    parsed: Type = ("const", base) if const else base
    return _declarator(parsed, tokens, at)


def _declarator(parsed: Type, tokens: list[str], at: int) -> tuple[Type, int]:
    """The ``*``, ``const``, ``&`` and ``[N]`` after a base type."""
    dims: list[str] = []
    while at < len(tokens) and tokens[at] in ("*", "&", "const", "volatile", "["):
        token = tokens[at]
        if token == "*":
            parsed = ("pointer", parsed)
        elif token == "const":
            parsed = ("const", parsed)
        elif token == "[":
            dims.append(tokens[at + 1])
            at += 2
        at += 1
    for size in reversed(dims):
        parsed = ("array", size, parsed)
    return parsed, at


def _base(tokens: list[str], at: int) -> tuple[Type, int]:
    """A built-in type's words, or a class's qualified name with its template arguments."""
    words: list[str] = []
    while at < len(tokens) and tokens[at] in BUILTIN_WORDS:
        words.append(tokens[at])
        at += 1
    if words:
        return ("builtin", BUILTINS[_builtin(words)]), at
    return _qualified(tokens, at)


def _qualified(tokens: list[str], at: int) -> tuple[Type, int]:
    """``a::b<int>::c``: a class's name, part by part, each with its template arguments."""
    components: list[tuple[str, tuple[Type, ...] | None]] = []
    while True:
        name = tokens[at]
        at += 1
        args: tuple[Type, ...] | None = None
        if at < len(tokens) and tokens[at] == "<":
            args, at = _arguments(tokens, at + 1)
        components.append((name, args))
        if at >= len(tokens) or tokens[at] != "::":
            return _named(components), at
        at += 1


def _named(components: list[tuple[str, tuple[Type, ...] | None]]) -> Type:
    """A class type - or ``std::nullptr_t``, the one library name that is a built-in's."""
    spelled = "::".join(name for name, _ in components)
    if spelled in BUILTINS:
        return ("builtin", BUILTINS[spelled])
    if spelled in STRING_NAMES:
        return ("builtin", STRING_MANGLED)
    return ("class", tuple(components))


def _arguments(tokens: list[str], at: int) -> tuple[tuple[Type, ...], int]:
    """``<int, A*>``: each template argument, a type or a number, to the closing ``>``."""
    args: list[Type] = []
    while tokens[at] != ">":
        if tokens[at].isdigit():
            args.append(("value", tokens[at]))
            at += 1
        else:
            parsed, at = _type(tokens, at)
            args.append(parsed)
        if tokens[at] == ",":
            at += 1
    return tuple(args), at + 1


def _builtin(words: list[str]) -> str:
    """The canonical spelling of a built-in type's words: ``long int`` is ``long``."""
    longs = words.count("long")
    named = [word for word in words if word not in ("signed", "unsigned", "int", "long", "short")]
    base = named[0] if named else _integer(words, longs)
    if base == "double" and longs:
        return "long double"
    return _signed(base, words)


def _integer(words: list[str], longs: int) -> str:
    """The integer type ``short``, ``int``, ``long`` or ``long long`` the words spell."""
    if "short" in words:
        return "short"
    return ("int", "long", "long long")[min(longs, 2)]


def _signed(base: str, words: list[str]) -> str:
    """``base`` with the sign the words give it: ``unsigned int``, ``signed char``."""
    if "unsigned" in words and base in ("char", "short", "int", "long", "long long"):
        return "unsigned " + base
    if base == "char" and "signed" in words:
        return "signed char"
    return base


class _Mangler:
    """One name's mangling, with the ABI's table of what it has already written."""

    def __init__(self) -> None:
        self.seen: dict[Any, int] = {}

    def _reference(self, key: Any) -> str:
        index = self.seen[key]
        return "S_" if index == 0 else f"S{_base36(index - 1)}_"

    def _remember(self, key: Any) -> None:
        self.seen.setdefault(key, len(self.seen))

    def type(self, parsed: Type) -> str:
        kind = parsed[0]
        if kind == "builtin":
            return str(parsed[1])
        if kind == "value":
            return f"Li{parsed[1]}E"
        key = _key(parsed)
        if key in self.seen:
            return self._reference(key)
        text = self._compound(parsed)
        self._remember(key)
        return text

    def _compound(self, parsed: Type) -> str:
        kind = parsed[0]
        if kind == "pointer":
            return "P" + self.type(parsed[1])
        if kind == "const":
            return "K" + self.type(parsed[1])
        if kind == "array":
            return f"A{parsed[1]}_" + self.type(parsed[2])
        return self._name(parsed[1])

    def _name(self, components: tuple[tuple[str, Any], ...]) -> str:
        """A class's name: ``1A``, ``N2ns1AE``, ``St6vectorIiE``, ``NSt6chrono8durationE``."""
        std = _in_std(components)
        rest = components[1:] if std else components
        key: tuple[Any, ...] = ("std",) if std else ()
        parts: list[str] = []
        last = len(rest) - 1
        for index, (name, args) in enumerate(rest):
            key = (*key, name)
            parts = self._component(parts, key, f"{len(name)}{name}", index < last or args)
            if args is not None:
                key = (*key, args)
                written = "I" + "".join(self.type(arg) for arg in args) + "E"
                parts = self._component(parts, key, written, index < last)
        body = "".join(parts)
        if len(rest) > 1:
            return ("NSt" if std else "N") + body + "E"
        return ("St" if std else "") + body

    def _component(self, parts: list[str], key: Any, written: str, prefix: Any) -> list[str]:
        """One more part of a name: a reference to it, if it is a prefix already written."""
        if prefix and key in self.seen:
            return [self._reference(key)]
        if prefix:
            self._remember(key)
        return [*parts, written]


def _in_std(components: tuple[tuple[str, Any], ...]) -> bool:
    """Is the name in ``std``, which the ABI abbreviates to ``St``?"""
    return components[0][0] == "std" and len(components) > 1


def _key(parsed: Type) -> Any:
    """What the substitution table knows a type by: a class by the path its name's parts
    are remembered by, so ``ns::A`` the type and ``ns::A`` the prefix of ``ns::A::B`` are one."""
    if parsed[0] != "class":
        return parsed
    components = parsed[1]
    key: tuple[Any, ...] = ("std",) if _in_std(components) else ()
    for name, args in components[1:] if key else components:
        key = (*key, name) if args is None else (*key, name, args)
    return key


def _base36(number: int) -> str:
    """``number`` in the ABI's base 36 of digits and capital letters."""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = ""
    while True:
        number, remainder = divmod(number, 36)
        text = digits[remainder] + text
        if not number:
            return text
