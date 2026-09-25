"""What an expression can be wrong with, as opposed to what it asks for that is not done.

A :class:`FormulaError` is a mistake in the expression - a syntax error, a name
no branch has, a function called with the wrong number of arguments - and is a
:class:`ValueError`, since the text is the value that is wrong. C++ that is
right but that this does not evaluate is refused with
:class:`~xrdroot.errors.UnsupportedFeatureError` instead, as everything else in
this package refuses what it does not do.
"""

from __future__ import annotations

from ..errors import ROOTError

__all__ = ["FormulaError"]


class FormulaError(ROOTError, ValueError):
    """An expression that is not one: bad syntax, an unknown name, a wrong call."""
