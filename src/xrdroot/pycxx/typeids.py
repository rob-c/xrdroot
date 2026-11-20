"""What ``typeid`` needs of a type: its full C++ spelling, and whether it is polymorphic.

``typeid(e)`` names the dynamic type of ``e`` only when ``e`` is an object
of a polymorphic class - one with a virtual function - and does not even
evaluate ``e`` otherwise: ``typeid(*p)`` of a null ``A*`` is ``A`` when
``A`` has no virtual function. So the translator decides which, from the
program's classes; a library's classes, which it does not see, are taken
to be polymorphic. The type is spelled with the namespaces the macro declared it
in, which the mangled name the runtime makes of it includes.
"""

from __future__ import annotations

from typing import Any

from .ctype import CType
from .nodes import ClassDecl, EnumDecl, Namespace, Stmt
from .program import ClassInfo

__all__ = ["qualified_names", "spelled", "polymorphic"]


def qualified_names(decls: list[Stmt], prefix: str = "") -> dict[str, str]:
    """Each of the macro's classes and enumerations, by its name, as its namespaces and outer
    classes qualify it - which is the name a ``typeid`` of one spells, and the name the
    interpreter keeps it under."""
    found: dict[str, str] = {}
    for decl in decls:
        if isinstance(decl, Namespace):
            inner = f"{prefix}{decl.name}::" if decl.name else prefix
            found.update(qualified_names(decl.body, inner))
        elif isinstance(decl, EnumDecl) and decl.name:
            found[decl.name] = prefix + decl.name
        elif isinstance(decl, ClassDecl):
            found[decl.name] = prefix + decl.name
            found.update(qualified_names(decl.members, f"{prefix}{decl.name}::"))
    return found


def spelled(ctype: CType, qualified: dict[str, str]) -> str | None:
    """``ctype`` as C++ spells it in full - ``const ns::A*``, ``int[3]`` - or ``None`` if one
    of its array sizes is not a number the translator knows."""
    if not all(isinstance(dim, int) for dim in ctype.dims):
        return None
    name = ctype.enum or ctype.name
    text = qualified.get(name, name)
    if ctype.args:
        text += "<" + ", ".join(_argument(arg, qualified) for arg in ctype.args) + ">"
    const = "const " if ctype.const else ""
    return const + text + "*" * ctype.pointer + "".join(f"[{dim}]" for dim in ctype.dims)


def _argument(arg: Any, qualified: dict[str, str]) -> str:
    """A template argument: a type spelled, or a value as written."""
    if isinstance(arg, CType):
        return spelled(arg, qualified) or arg.name
    return str(getattr(arg, "value", arg))


def polymorphic(name: str, classes: dict[str, ClassInfo]) -> bool:
    """Has the class ``name`` a virtual function - its own, or a base's? A library's class, which
    the translator does not see, is taken to have one."""
    info = classes.get(name.split("::")[-1])
    if info is None:
        return True
    if any(func.virtual for funcs in info.methods.values() for func in funcs):
        return True
    return any(polymorphic(base.ctype.name, classes) for base in info.decl.bases)
