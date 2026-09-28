"""What cling prints of a value a macro's function gives back: ``(int) 0``.

``root -b -q macro.C`` runs ``macro()`` as a line typed at the prompt, and
cling prints what a line typed there gives, when it gives something: the
type as C++ spells it, with ROOT's typedefs looked through, then the value -
``(int) 0``, ``(double) 0.10000000``, ``(float) 2.50000f``, ``(bool) true``,
``(const char *) "hi"``, ``(TH1F *) nullptr``, ``(TCanvas *) 0x7f...``. A
``void`` function gives nothing, and nothing is printed.
"""

from __future__ import annotations

from typing import Any

from .ctype import CType

__all__ = ["spelled", "shown"]

#: The class types given back by value that cling prints as a quoted string.
QUOTED = {"std::string", "string", "TString"}


def spelled(ctype: CType) -> str | None:
    """The type as cling names it, or None when it gives nothing or cannot be printed here."""
    if ctype.is_void or ctype.reference or ctype.dims or ctype.is_auto:
        return None
    if ctype.pointer:
        const = "const " if ctype.const else ""
        return f"{const}{ctype.name} {'*' * ctype.pointer}"
    if ctype.arithmetic or ctype.name == "bool":
        return ctype.name
    return "std::string" if ctype.name in ("std::string", "string") else None


def _number(kind: str, value: Any) -> str:
    """A number as cling prints it: eight significant figures for a double, six and ``f``."""
    if kind == "bool":
        return "true" if value else "false"
    if kind in ("float", "double", "long double"):
        figures = 6 if kind == "float" else 8
        return f"{float(value):#.{figures}g}" + ("f" if kind == "float" else "")
    if kind in ("char", "signed char", "unsigned char"):
        character = value if isinstance(value, str) else chr(int(value))
        return f"'{character}'"
    return str(int(value))


def shown(kind: str, value: Any) -> str | None:
    """The line cling prints for ``value`` of the type it names ``kind``.

    None when a value was declared but none was given back - a function
    that falls off its end, where C++ leaves the value undefined.
    """
    if value is None and not kind.endswith("*"):
        return None
    if kind.endswith("*"):
        if value is None:
            text = "nullptr"
        elif kind.replace(" ", "") in ("constchar*", "char*"):
            text = f'"{value}"'
        else:
            text = f"0x{id(value):x}"
    elif kind == "std::string":
        text = f'"{value}"'
    else:
        text = _number(kind, value)
    return f"({kind}) {text}"
