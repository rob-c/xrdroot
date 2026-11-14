"""What an earlier input means once a later one has changed how it is translated.

The unit is translated whole each time, so a later input can change the
Python of an earlier one that has run already: ``int f(double)`` typed after
``int f(int)`` makes both overloads, named apart, and ``int A::g() {...}``
typed after ``class A`` puts ``g`` in the class; ``int *p = &i;`` makes ``i``,
declared before, a cell. Cling's unit has them all, so here:

- a function or class whose Python changed is defined again - a class into
  the class object that is there already, so the objects made of it before
  have its new methods too;
- a variable that is now a cell is put in one, holding the value it had.

Anything else that ran is not run again: ``TH1F h(...)`` makes one histogram.
"""

from __future__ import annotations

import ast
from typing import Any

__all__ = ["revised", "merged"]

#: What a class's namespace has that is no member of the class to carry over.
NOT_MEMBERS = frozenset({"__dict__", "__weakref__", "__module__", "__qualname__", "__doc__"})


def revised(statement: ast.stmt, before: str | None) -> ast.stmt | None:
    """What to run again of ``statement``, which ran as ``before`` (its dump) - or ``None``."""
    if before is None or ast.dump(statement) == before:
        return None
    if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
        return statement
    return _celled(statement)


def _celled(statement: ast.stmt) -> ast.stmt | None:
    """``i = Cell(3, 'int')`` for a variable that ran as ``i = 3``: its value put in a cell,
    unless it is in one already."""
    if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1):
        return None
    target, value = statement.targets[0], statement.value
    if not (isinstance(target, ast.Name) and isinstance(value, ast.Call)):
        return None
    if not (isinstance(value.func, ast.Name) and value.func.id == "Cell"):
        return None
    ctype = ast.unparse(value.args[1]) if len(value.args) > 1 else "None"
    name = target.id
    text = f"{name} = {name} if isinstance({name}, Cell) else Cell({name}, {ctype})"
    return ast.copy_location(ast.parse(text).body[0], statement)


def merged(namespace: dict[str, Any], classes: dict[str, type]) -> None:
    """The classes defined again, each put into the class object it replaced: ``classes`` are
    those objects, by name."""
    for name, old in classes.items():
        new = namespace.get(name)
        if not isinstance(new, type) or new is old:
            continue
        for member, value in vars(new).items():
            if member not in NOT_MEMBERS:
                setattr(old, member, value)
        namespace[name] = old
