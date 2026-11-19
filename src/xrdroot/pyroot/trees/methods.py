"""``T->Draw("hpx.GetRMS():hprof.GetMean()")``: methods of the objects a branch holds.

A tree of whole objects - a histogram an entry, say - can be drawn by what
a method of each says: ROOT's ``TTreeFormula`` makes the object of each
entry and calls the method on it. ``Draw()`` itself draws the object, entry
by entry; any other method gives a number, and the numbers are drawn as
any expression's are - as a tree of them, filled here, would draw them,
under the expression ROOT titles its histogram with.

A method that gives back an object - ``GetTriggerBits()`` of an event - is
not drawn from here at all: ROOT has no such split, and its formula draws
that object as it draws a member that is one, a ``TBits`` a set bit at a
time. So this declines the whole expression, and the formula draws it.
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
    entries = _entries(tree, count, first)
    if len(calls) == 1 and calls[0][1] == "Draw":
        return _drawn_each(tree, calls[0], entries)
    columns = _numbers(tree, calls, entries)
    return None if columns is None else _drawn(tree, varexp, columns, option)


def _entries(tree: Any, count: int | None, first: int) -> range:
    """The entries a draw of ``count`` of them from ``first`` walks - all of them for neither."""
    stop = tree.GetEntries() if count is None else min(tree.GetEntries(), first + count)
    return range(int(first), stop)


def _drawn_each(tree: Any, call: tuple[Any, str, tuple[Any, ...]], entries: range) -> int:
    """``Draw()`` of each entry's object, which the object itself draws as ROOT's does."""
    branch, _name, args = call
    for entry in entries:
        _object(tree, branch, entry).Draw(*args)
    return len(entries)


def _numbers(
    tree: Any, calls: list[tuple[Any, str, tuple[Any, ...]]], entries: range
) -> list[np.ndarray] | None:
    """What each call gives, entry by entry - or ``None`` when one gives back an object."""
    columns = [_called(tree, call, entries) for call in calls]
    return None if any(column is None for column in columns) else columns


def _calls(tree: Any, varexp: str) -> list[tuple[Any, str, tuple[Any, ...]]] | None:
    """Each part of the expression as a branch of objects, a method and its arguments - or
    ``None`` unless every part is such a call."""
    found = []
    for part in split_names(varexp.split(">>")[0]):
        call = CALL.match(part)
        branch = tree._branch_info(call.group(1)) if call is not None else None
        if call is None or branch is None or not is_object(branch):
            return None
        try:
            arguments = _arguments(call.group(3))
        except (SyntaxError, ValueError, TypeError):
            # A call of a call, `event.GetHistogram().GetXaxis()`, has no literal arguments,
            # and is an expression like any other: drawn by the formula, not by the objects.
            return None
        found.append((branch, call.group(2), arguments))
    return found


def _called(
    tree: Any, call: tuple[Any, str, tuple[Any, ...]], entries: range
) -> np.ndarray | None:
    """What one method of each entry's object gives, entry by entry - or ``None`` when it
    gives an object rather than a number, which only the formula knows how to draw."""
    branch, name, args = call
    numbers = []
    for entry in entries:
        given = getattr(_object(tree, branch, entry), name)(*args)
        try:
            numbers.append(float(given))
        except TypeError:  # a `TBits`, or any other object: no number to draw it by
            return None
    return np.asarray(numbers)


def _object(tree: Any, branch: Any, entry: int) -> Any:
    """The object a branch holds in one entry, made again from what the entry holds."""
    return rebuilt(branch.classname, tree._batch.value(branch.leaves[0].column, entry), None)


def _drawn(tree: Any, varexp: str, columns: list[Any], option: str) -> int:
    """The numbers drawn as a tree of them draws them, titled with the expression."""
    from .tree import TTree

    names = [f"v{at}" for at in range(len(columns))]
    held = np.zeros(len(columns))
    temporary = TTree("methods", "", dir=False)
    tree.CopyTAttLine(temporary)
    tree.CopyTAttFill(temporary)
    tree.CopyTAttMarker(temporary)
    temporary.SetEstimate(tree.GetEstimate())
    temporary.Branch(names[0], held, ":".join(f"{name}/D" for name in names))
    for row in np.column_stack(columns) if columns else ():
        held[:] = row
        temporary.Fill()
    drawn = temporary.Draw(":".join(names), "", option)
    histogram = temporary.GetHistogram()
    histogram.SetTitle(varexp)
    tree._draw_histogram = histogram
    tree._drawn = [temporary.GetVal(at) for at in range(len(columns))]
    _axes(histogram, varexp, len(columns))
    return int(drawn)


def _axes(histogram: Any, varexp: str, dimensions: int) -> None:
    """The temporary column names must not appear on the original expression's axes."""
    if histogram.GetDimension() != dimensions:
        return
    parts = split_names(varexp.split(">>")[0])
    for letter, part in zip("XYZ", reversed(parts), strict=False):
        getattr(histogram, f"Get{letter}axis")().SetTitle(part.strip())
