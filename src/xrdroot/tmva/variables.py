"""What TMVA knows of a variable, a target, a spectator and a class.

A variable is declared as an expression over the tree - ``"var3"``, or
``"myvar1 := var1+var2"`` to give the expression a label - with a title and
a unit for the plots. TMVA names everything after the variable's *internal
name*, the label with every character a histogram name cannot hold spelled
out (``+`` as ``_P_``, ``*`` as ``_T_``...), as
``Tools::ReplaceRegularExpressions`` spells them, and the same is done here
so that the histograms of the output file have TMVA's names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ClassInfo", "VariableInfo", "internal_name"]

#: The characters a name cannot hold, each replaced by ``_``.
PLAIN = "$&|!%^&()'<>?= "
#: The rest of ``ReplaceRegularExpressions``'s replacements, in its order.
SPELLED = (
    ("::", "_"),
    ("$", "_S_"),
    ("&", "_A_"),
    ("%", "_MOD_"),
    ("|", "_O_"),
    ("*", "_T_"),
    ("/", "_D_"),
    ("+", "_P_"),
    ("-", "_M_"),
    (" ", "_"),
    ("[", "_"),
    ("]", "_"),
    ("=", "_E_"),
    (">", "_GT_"),
    ("<", "_LT_"),
    ("(", "_"),
    (")", "_"),
)


def internal_name(label: str, replacement: str = "_") -> str:
    """``Tools::ReplaceRegularExpressions(label, "_")``."""
    for character in PLAIN:
        label = label.replace(character, replacement)
    for old, new in SPELLED:
        label = label.replace(old, new)
    return label


class VariableInfo:
    """``TMVA::VariableInfo``: a declared expression, its label, title, unit and range.

    >>> v = VariableInfo("myvar1 := var1+var2")
    >>> v.GetLabel(), v.GetExpression(), v.GetTitle(), v.GetInternalName()
    ('myvar1', 'var1+var2', 'myvar1', 'myvar1')
    """

    def __init__(
        self,
        expression: Any,
        title: Any = "",
        unit: Any = "",
        vartype: Any = "F",
        minimum: float = 0.0,
        maximum: float = 0.0,
        index: int | None = None,
    ) -> None:
        text = str(expression).replace(" ", "")
        label, bound, rest = text.partition(":=")
        self.expression = rest if bound else text
        self.label = label if bound else text
        self.title = str(title) or self.label
        self.unit = str(unit)
        self.vartype = chr(vartype) if isinstance(vartype, int) else str(vartype or "F")
        self.internal = internal_name(self.label)
        #: The element of an array expression this variable is, for ``AddVariablesArray``.
        self.index = index
        if index is not None:
            self.internal += f"[{index}]"
        self.minimum, self.maximum = float(minimum), float(maximum)

    def GetExpression(self) -> str:
        return self.expression

    def GetLabel(self) -> str:
        return self.label

    def GetTitle(self) -> str:
        return self.title

    def GetUnit(self) -> str:
        return self.unit

    def GetInternalName(self) -> str:
        return self.internal

    def GetVarType(self) -> str:
        return self.vartype

    def GetMin(self) -> float:
        return self.minimum

    def GetMax(self) -> float:
        return self.maximum

    def __repr__(self) -> str:
        return f"VariableInfo({self.label!r} := {self.expression!r})"


@dataclass
class ClassInfo:
    """``TMVA::ClassInfo``: a class's name, number, weight expression and cut."""

    name: str
    number: int
    weight: str = ""
    cut: str = ""

    def GetName(self) -> str:
        return self.name

    def GetNumber(self) -> int:
        return self.number

    def GetWeight(self) -> str:
        return self.weight

    def GetCut(self) -> str:
        return self.cut
