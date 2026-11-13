"""``<type_traits>`` asked of types that are known: ``std::is_same_v<T, int>`` as a constant.

In an instance of a template every type is known, and so is every trait of
one; the translation writes ``True`` or ``False`` where C++ has the trait,
and ``if constexpr`` keeps only the branch the constant chooses - the other
is not even instantiated in C++, and need not make sense for these types.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .ctype import FLOATING, INTEGRAL, CType
from .nodes import Name

__all__ = ["trait", "is_trait"]


def _unsigned(ctype: CType) -> bool:
    return ctype.integral and (ctype.name.startswith("unsigned") or ctype.name == "bool")


def _plain(ctype: CType) -> bool:
    return not (ctype.pointer or ctype.dims or ctype.reference)


def _number(ctype: CType) -> bool:
    """A number held by value - not an enum, which is no arithmetic type, though held as one."""
    return _plain(ctype) and ctype.enum is None and ctype.arithmetic


def _class(ctype: CType, program: Any) -> bool | None:
    """A class of the program's - not a number or pointer - or unknown: a library's."""
    if not _plain(ctype) or ctype.arithmetic or ctype.enum is not None or ctype.is_void:
        return False
    return True if ctype.name in program.classes or ctype.is_string else None


def _enum(ctype: CType, program: Any) -> bool | None:
    """An enum of the program's, or unknown: a library's name may be one."""
    if ctype.enum is not None:
        return _plain(ctype)
    return False if _class(ctype, program) is not None else None


#: Each one-type trait, by name, as a test of the type: ``None`` when it cannot tell.
UNARY: dict[str, Callable[[CType, Any], bool | None]] = {
    "is_floating_point": lambda c, p: _number(c) and c.name in FLOATING,
    "is_integral": lambda c, p: _number(c) and c.name in INTEGRAL,
    "is_arithmetic": lambda c, p: _number(c),
    "is_signed": lambda c, p: _number(c) and (c.floating or not _unsigned(c)),
    "is_unsigned": lambda c, p: _number(c) and _unsigned(c),
    "is_pointer": lambda c, p: c.is_pointer and not c.reference,
    "is_reference": lambda c, p: c.reference,
    "is_lvalue_reference": lambda c, p: c.reference,
    "is_array": lambda c, p: bool(c.dims) and not c.reference,
    "is_const": lambda c, p: c.const and not c.pointer and not c.reference,
    "is_void": lambda c, p: c.is_void,
    "is_enum": _enum,
    "is_class": _class,
    "is_fundamental": lambda c, p: _number(c) or (_plain(c) and c.name in ("void", "nullptr_t")),
    "is_scalar": lambda c, p: not c.reference and not c.dims and (c.arithmetic or c.pointer > 0),
}


def _spelling(ctype: CType) -> tuple[Any, ...]:
    """What makes two types one type: their name, arguments, and every qualifier."""
    args = tuple(_spelling(arg) if isinstance(arg, CType) else repr(arg) for arg in ctype.args)
    shape = (ctype.pointer, ctype.reference, ctype.const, tuple(map(str, ctype.dims)))
    return ctype.enum or ctype.name, args, shape


def _derives(derived: CType, base: CType, program: Any) -> bool:
    """``std::is_base_of<B, D>``: ``D`` is ``B``, or a class of the program's derived from it."""
    if derived.name == base.name:
        return derived.name in program.classes
    info = program.classes.get(derived.name)
    bases = [parent.ctype for parent in info.decl.bases] if info is not None else []
    return any(_derives(parent, base, program) for parent in bases)


#: Each two-type trait, by name.
BINARY: dict[str, Callable[[CType, CType, Any], bool]] = {
    "is_same": lambda one, other, p: _spelling(one) == _spelling(other),
    "is_base_of": lambda base, derived, p: _derives(derived, base, p),
}


def is_trait(name: Name) -> bool:
    """Is ``name`` one of the traits here: ``std::is_same_v``, ``std::is_integral<T>::value``?"""
    found = _named(name)
    return found is not None and (found in UNARY or found in BINARY)


def _named(name: Name) -> str | None:
    """The trait ``name`` asks for, or ``None``: ``is_same_v`` and ``is_same<...>::value``."""
    parts = name.parts[1:] if name.parts[0] == "std" else name.parts
    if len(parts) == 1 and parts[0].endswith("_v"):
        return parts[0][:-2]
    if len(parts) == 2 and parts[1] == "value":
        return parts[0]
    return None


def trait(name: Name, program: Any) -> bool | None:
    """The value of the trait ``name`` - or ``None`` when its types are not all known."""
    kinds = _types(name)
    found = _named(name) or ""
    if kinds is not None and len(kinds) == 1 and found in UNARY:
        return UNARY[found](kinds[0], program)
    test = BINARY.get(found) if kinds is not None and len(kinds) == 2 else None
    return test(kinds[0], kinds[1], program) if test is not None and kinds else None


def _types(name: Name) -> list[CType] | None:
    """The types a trait is asked of - ``None`` if a value is among them."""
    args = (name.scope_targs if name.parts[-1] == "value" else name.targs) or []
    kinds = [arg for arg in args if isinstance(arg, CType)]
    return kinds if len(kinds) == len(args) else None
