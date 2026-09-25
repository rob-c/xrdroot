"""ROOT's ``TTree::Draw`` expressions, evaluated over whole columns with NumPy.

    >>> from xrdroot.formula import compile_formula
    >>> f = compile_formula("Sum$(jet_pt > 30)", tree.keys())    # doctest: +SKIP
    >>> f.evaluate(tree.arrays(f.branches))

The language is ``TTreeFormula``'s: C++ arithmetic over branch names, with
``TMath`` and ``<cmath>``, ROOT's special names - ``Entry$``, ``Sum$``,
``Alt$`` and the rest - and its implicit loop, in which a branch that is a
collection makes the expression one value per element rather than per entry.
The loop follows ROOT's documented rule: dimensions without an index are
matched left to right across branches and run to the shortest of them.

:func:`compile_formula` parses and resolves the names, and a
:class:`Formula` evaluates, for a batch of columns at a time, without a
Python loop over entries.
"""

from __future__ import annotations

from .errors import FormulaError
from .formula import Formula, compile_formula

__all__ = ["Formula", "FormulaError", "compile_formula"]
