"""``TCut``: a selection, as ``TTree::Draw`` and TMVA's ``DataLoader`` take one.

A cut is its expression: ``TCut("x > 0")`` stands for ``x > 0`` wherever a
selection is asked for, and cuts combine as ROOT's operators combine them -
``a + b`` and ``a && b`` are ``(a)&&(b)``, ``a || b`` is ``(a)||(b)``,
``a * b`` a weight times a selection, ``!a`` is ``!(a)`` - with an empty cut
leaving the other as it is, as ``TCut::operator+`` does.
"""

from __future__ import annotations

from typing import Any

from .core.objects import TNamed

__all__ = ["TCut"]


def _joined(left: str, right: str, operator: str) -> str:
    """Two expressions joined by ``operator``, an empty one giving way to the other."""
    if not left:
        return right
    if not right:
        return left
    return f"({left}){operator}({right})"


class TCut(TNamed):
    """ROOT's ``TCut``: the title is the expression.

    >>> str(TCut("x > 0") + TCut("y < 1"))
    '(x > 0)&&(y < 1)'
    """

    def __init__(self, *arguments: Any) -> None:
        name, title = ("CUT", arguments[0]) if len(arguments) == 1 else (arguments or ("", ""))
        super().__init__(str(name), str(title))

    def ClassName(self) -> str:
        return "TCut"

    def _with(self, other: Any, operator: str) -> TCut:
        return TCut(_joined(self.GetTitle(), str(other), operator))

    def __add__(self, other: Any) -> TCut:
        return self._with(other, "&&")

    def __radd__(self, other: Any) -> TCut:
        return TCut(str(other)) + self

    __and__ = __add__

    def __or__(self, other: Any) -> TCut:
        return self._with(other, "||")

    def __mul__(self, other: Any) -> TCut:
        return self._with(other, "*")

    def __invert__(self) -> TCut:
        return TCut(f"!({self.GetTitle()})")

    def __iadd__(self, other: Any) -> TCut:
        self.SetTitle(_joined(self.GetTitle(), str(other), "&&"))
        return self

    def __imul__(self, other: Any) -> TCut:
        self.SetTitle(_joined(self.GetTitle(), str(other), "*"))
        return self

    def __eq__(self, other: object) -> bool:
        return self.GetTitle() == str(other)

    def __hash__(self) -> int:
        return hash(self.GetTitle())

    def __bool__(self) -> bool:
        return True

    def __str__(self) -> str:
        return self.GetTitle()

    def IsNull(self) -> bool:
        return not self.GetTitle()

    def Data(self) -> str:
        return self.GetTitle()
