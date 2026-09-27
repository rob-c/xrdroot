"""``RooFormula``: the C++ expressions ``RooFormulaVar``, ``RooGenericPdf`` and cuts are written in.

RooFit's formulas are ``TFormula``'s C++ - ``TMath::Exp(-x/tau)``, ``x>0 &&
y<5`` - over its variables, named or numbered: ``@0`` is the first of the
list the formula was given, ``x[0]`` too, and ``c::Plus`` a category's label
as its index. That language is the one :mod:`xrdroot.formula` evaluates
over tree columns, so a formula here is translated into it - each variable
under a name of its own, each label into its number - and evaluated over a
context's values as it would be over columns.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .collections import as_list
from .compiled import Compiled

__all__ = ["RooFormula", "translate"]

#: ``@3``, or ``x[3]``: a variable by its place in the list the formula was given.
NUMBERED = re.compile(r"@(\d+)|\bx\[(\d+)\]")
#: ``c::Plus``: a label of a category.
LABEL = re.compile(r"\b([A-Za-z_]\w*)::([A-Za-z_]\w*)\b")
#: An identifier that is not part of a longer name or a ``::`` path.
WORD = r"(?<![\w:.]){}(?![\w(])"


def _alias(index: int) -> str:
    return f"_rf{index}_"


def translate(expression: str, args: list[Any]) -> tuple[str, list[int]]:
    """``expression`` with each variable under an alias of its own, and which ones it uses."""
    text = str(expression)
    used: set[int] = set()

    def numbered(match: re.Match[str]) -> str:
        index = int(match.group(1) or match.group(2))
        used.add(index)
        return _alias(index)

    text = NUMBERED.sub(numbered, text)
    text = LABEL.sub(lambda m: _label(m, args), text)
    order = sorted(range(len(args)), key=lambda i: -len(args[i].GetName()))
    for index in order:
        pattern = re.compile(WORD.format(re.escape(args[index].GetName())))
        if pattern.search(text):
            used.add(index)
            text = pattern.sub(_alias(index), text)
    return text, sorted(used)


def _label(match: re.Match[str], args: list[Any]) -> str:
    name, label = match.group(1), match.group(2)
    category = next((a for a in args if a.GetName() == name and hasattr(a, "lookupIndex")), None)
    if category is None:
        return match.group(0)
    return str(category.lookupIndex(label))


class RooFormula:
    """A formula over some variables, evaluated for a context's values of them."""

    def __init__(self, expression: str, dependents: Any) -> None:
        self.expression = str(expression)
        self.args = as_list(dependents)
        self.text, self.used = translate(self.expression, self.args)
        aliases = [_alias(i) for i in range(len(self.args))]
        self._compiled = Compiled(self.text, aliases)

    def actual(self) -> list[Any]:
        """The variables the formula really uses, in the order given."""
        return [self.args[i] for i in self.used]

    def evaluate(self, ctx: dict[str, Any]) -> Any:
        values = {_alias(i): self.args[i].compute(ctx) for i in self.used}
        found = self._compiled(values)
        return found if np.ndim(found) else float(found)

    def GetTitle(self) -> str:
        return self.expression
