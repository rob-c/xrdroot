"""Several expressions in one loop, the way ``TTree::Draw`` runs its axes and its cut.

``Draw("y:x", "sel")`` compiles three formulas and hands them to one
``TTreeFormulaManager``, which makes them share a single implicit loop: the
dimensions each loops over are matched across all of them, and the loop runs
to the shortest, exactly as if they were one expression. So ``jet_pt`` drawn
with the cut ``jet_eta < 2.5`` pairs each jet's momentum with that same jet's
pseudorapidity, and a number per entry - ``met`` drawn with a cut on the jets
- is repeated for every jet the cut looks at.

That is what one :class:`~.formula.Formula` already does with the branches
inside it, so this runs the same loop over the branches of several: one
:class:`~.evaluate.Evaluator`, one scope over every root, and each root's
value read off it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .binding import Specs, bind
from .columns import Layout
from .evaluate import Evaluator, lift
from .formula import Formula, _numbers, _package

__all__ = ["evaluate_together"]


def evaluate_together(
    formulas: Sequence[Formula],
    columns: Mapping[str, Any],
    *,
    entries: int | None = None,
    rows: int | None = None,
    entry_numbers: Any = None,
    local_entries: Any = None,
) -> list[tuple[Any, Any]]:
    """Each formula's values and mask, as :meth:`Formula.evaluate_masked` gives them, in one loop.

    Every formula is one value per entry, or - when any of them loops - every
    one that depends on the loop is a :class:`~xrdroot.tree.Jagged` of the
    same rows, and the ones that do not are repeated along them; the rows are
    as long as the loop they share. The arguments are ``evaluate_masked``'s.
    """
    layouts: dict[str, Layout] = {}
    for formula in formulas:
        layouts.update(formula._layouts(columns))
    given = rows if entry_numbers is None else len(entry_numbers)
    count = formulas[0]._entries(layouts, columns, given)
    numbers = _numbers(count, 0, entries, entry_numbers, local_entries)
    roots = list({id(formula._root): formula._root for formula in formulas}.values())
    specs: Specs = {}
    for root in roots:
        specs.update(bind(root, lambda name: layouts[name].dims))
    evaluator = Evaluator(layouts, specs, count, numbers)
    with np.errstate(all="ignore"):
        scope = evaluator.scope(tuple(roots))
        values = [
            lift(evaluator.value(formula._root, scope), scope.space) for formula in formulas
        ]
    return [_package(scope.space, value, count) for value in values]
