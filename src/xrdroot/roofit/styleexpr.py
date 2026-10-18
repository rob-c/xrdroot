"""RooFit's style strings evaluate C++ constants and expose interpreter output.

Matplotlib shorthands bypass the interpreter; expressions ending in a semicolon
suppress its value display, as they do in ROOT's interactive prompt.
"""

from __future__ import annotations

from . import cout
from .names import ENUM_TYPES, named_constant

__all__ = ["style_expression"]


def style_expression(text: str) -> int:
    expression = text.strip()
    quiet = expression.endswith(";")
    expression = expression[:-1].strip() if quiet else expression
    value = named_constant(expression)
    if not quiet:
        kind = ENUM_TYPES.get(expression)
        display = (f"({kind}) ({expression}) : (unsigned int) {value}"
                   if kind else f"(int) {value}")
        cout.line(display)
    return value
