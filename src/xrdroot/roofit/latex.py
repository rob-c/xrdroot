"""``printLatex``: a collection of variables as a LaTeX table, as ``RooAbsCollection`` prints one.

``params.printLatex()`` writes ``\\begin{tabular}{lc}``, a row a variable -
its name and its value with the error, ``RooRealVar::format(1, "NEYVU")``,
``$\\verb+a0+ $ & $  0.6\\pm 0.2$`` - and ``\\end{tabular}``. ``Columns(n)``
lays the variables out in ``n`` blocks side by side, down each block first;
``Sibling(other)`` puts the same variables of another collection beside
each, without their names; ``Format(...)`` and ``OutputFile(name)`` say how
and where. Only ``RooRealVar``\\ s are printed, as ROOT prints only them.
"""

from __future__ import annotations

from typing import Any

from .cmdargs import RooCmdArg, commands
from .messages import WARNING, log

__all__ = ["latex_table", "print_latex"]

#: ``printLatex``'s default format and significant digits.
FORMAT, DIGITS = "NEYVU", 1


def _real_vars(collection: Any) -> list[Any]:
    """The collection's ``RooRealVar``\\ s, a warning said for anything else."""
    found = []
    for arg in collection:
        if arg.ClassName() == "RooRealVar":
            found.append(arg)
            continue
        said = f"skipping non-RooRealVar object named {arg.GetName()}"
        log(
            None,
            WARNING,
            "InputArguments",
            f"RooAbsCollection::printLatex: can only print RooRealVar in LateX, {said}",
        )
    return found


def _formatter(options: Any) -> tuple[Any, Any]:
    """How a variable of the collection, and one of a sibling - its name left out - is
    written: by ``Format(what, digits)``, ``Format(what, AutoPrecision(n), ...)`` or ROOT's
    ``"NEYVU"`` to one significant digit."""
    from .formatting import format_command, format_var

    given = options.every("Format")
    command = given[-1] if given else RooCmdArg("Format", FORMAT, DIGITS)
    what = str(command.value(0, FORMAT))
    plain = what.replace("N", "").replace("n", "")
    rest = command.args[1:]
    if rest and all(isinstance(one, RooCmdArg) for one in rest):
        sibling = RooCmdArg("Format", plain, *rest)
        return (lambda var: format_command(var, command)), (
            lambda var: format_command(var, sibling)
        )
    digits = int(rest[0]) if rest else DIGITS
    return (lambda var: format_var(var, digits, what)), (lambda var: format_var(var, digits, plain))


def latex_table(collection: Any, columns: int, siblings: list[Any], options: Any) -> str:
    """The table's text, line by line."""
    lists = [_real_vars(collection)] + [_real_vars(one) for one in siblings]
    if any(len(one) != len(lists[0]) for one in lists[1:]):
        log(None, WARNING, "InputArguments", "RooAbsCollection::printLatex: ERROR: sibling "
            "list(s) must have same length as self")  # fmt: skip
        return ""
    own, sibling = _formatter(options)
    rows = int(len(lists[0]) / columns + 0.99)
    block = "l" + "c" * len(lists)
    lines = ["\\begin{tabular}{" + "|".join([block] * columns) + "}"]
    for i in range(rows):
        cells = [_cell(lists, k, i + j * rows, own if k == 0 else sibling)
                 for j in range(columns) for k in range(len(lists))]  # fmt: skip
        lines.append(" & ".join(cells) + "\\\\")
    return "\n".join([*lines, "\\end{tabular}"]) + "\n"


def _cell(lists: list[list[Any]], k: int, at: int, written: Any) -> str:
    """The ``k``-th list's variable at ``at``, written - nothing past the last of it."""
    return written(lists[k][at]) if at < len(lists[k]) else ""


def print_latex(collection: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """``printLatex(Columns(n), Sibling(set), Format(...), OutputFile(name))``."""
    from .cout import write

    options = commands(args, kwargs)
    columns = int(options.get("Columns", 0, 1))
    siblings = [one.value(0) for one in options.every("Sibling")]
    text = latex_table(collection, max(columns, 1), siblings, options)
    name = str(options.get("OutputFile", 0, "") or "")
    if name:
        with open(name, "w", encoding="utf-8") as out:
            out.write(text)
        return
    write(text)
