"""``T->Draw("hpx.GetRMS():hprof.GetMean()")``: methods of the objects a branch holds.

A tree of whole objects - a histogram an entry, say - can be drawn by what
a method of each says: ROOT's ``TTreeFormula`` makes the object of each
entry and calls the method on it. ``Draw()`` itself draws the object, entry
by entry; any other method gives a number, and the numbers are drawn as
any expression's are - as a tree of them, filled here, would draw them,
under the expression ROOT titles its histogram with.
"""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np

from ...drawspec import split_names
from .bind import is_object
from .rebuild import rebuilt

__all__ = ["method_draw"]

#: ``branch.Method(arguments)``: a call on the object a branch holds.
CALL = re.compile(r"^\s*([A-Za-z_]\w*)\.(\w+)\((.*)\)\s*$")


def _arguments(text: str) -> tuple[Any, ...]:
    """A call's arguments - numbers and strings - as Python reads them."""
    text = text.strip()
    if not text:
        return ()
    found = ast.literal_eval(f"({text},)")
    return tuple(found)


def method_draw(
    tree: Any, varexp: str, selection: str, option: str, count: int | None, first: int
) -> int | None:
    """Draw methods of each entry's objects, how many entries were drawn; ``None`` when the
    expression calls no method of a branch of objects, and is an expression like any other."""
    calls = None if selection else _calls(tree, varexp)
    if calls is None:
        return None
    stop = tree.GetEntries() if count is None else min(tree.GetEntries(), first + count)
    entries = range(int(first), stop)
    if len(calls) == 1 and calls[0][1] == "Draw":
        branch, _name, args = calls[0]
        for entry in entries:
            _object(tree, branch, entry).Draw(*args)
        return len(entries)
    columns = [_called(tree, call, entries) for call in calls]
    tree._drawn = columns  # what GetV1 and its kin hand back
    return _drawn(varexp, columns, option)


def _calls(tree: Any, varexp: str) -> list[tuple[Any, str, tuple[Any, ...]]] | None:
    """Each part of the expression as a branch of objects, a method and its arguments - or
    ``None`` unless every part is such a call."""
    found = []
    for part in split_names(varexp.split(">>")[0]):
        call = CALL.match(part)
        branch = tree._branch_info(call.group(1)) if call is not None else None
        if call is None or branch is None or not is_object(branch):
            return None
        found.append((branch, call.group(2), _arguments(call.group(3))))
    return found


def _called(tree: Any, call: tuple[Any, str, tuple[Any, ...]], entries: range) -> Any:
    """What one method of each entry's object gives, entry by entry."""
    branch, name, args = call
    return np.asarray([float(getattr(_object(tree, branch, e), name)(*args)) for e in entries])


def _object(tree: Any, branch: Any, entry: int) -> Any:
    """The object a branch holds in one entry, made again from what the entry holds."""
    return rebuilt(branch.classname, tree._batch.value(branch.leaves[0].column, entry), None)


def _drawn(varexp: str, columns: list[Any], option: str) -> int:
    """The numbers drawn as a tree of them draws them, titled with the expression."""
    from ._base import hooks
    from .tree import TTree

    names = [f"v{at}" for at in range(len(columns))]
    held = np.zeros(len(columns))
    temporary = TTree("methods", "", dir=False)
    temporary.Branch(names[0], held, ":".join(f"{name}/D" for name in names))
    for row in np.column_stack(columns) if columns else ():
        held[:] = row
        temporary.Fill()
    drawn = temporary.Draw(":".join(names), "", option)
    hooks.wrap(hooks.registry()["htemp"]).SetTitle(varexp)  # what every such draw makes
    return int(drawn)
