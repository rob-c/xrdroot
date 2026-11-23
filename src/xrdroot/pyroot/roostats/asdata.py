"""``RooStats::GetAsTTree``: a dataset's numbers as a tree, a ``"%s/D"`` branch a column.

ROOT keeps a chain's points in a ``RooDataSet`` and the tutorials draw them with
``TTree::Draw``, so RooStats hands a tree over: one branch for each ``RooRealVar`` of the
dataset - in the order ``std::map`` keeps its names, which is alphabetical - plus
``_errlo``/``_errhi`` where the variable carries an asymmetric error and ``_err`` where it
carries a symmetric one (``roofit/roostats/src/RooStatsUtils.cxx:311``, whose ``BranchStore``
fills each entry from the dataset's own values). A branch of a variable the entry cannot give
keeps ``-999``, as ROOT's ``fInval`` does.
"""

from __future__ import annotations

from array import array
from typing import Any

from ..trees import TTree

__all__ = ["GetAsTTree"]

#: ``BranchStore``'s ``fInval``: what a branch no entry fills keeps.
INVALID = -999.0


def _columns(data: Any) -> list[tuple[str, str]]:
    """Each branch's name and what it reads: a variable's value, or one of its errors."""
    found: list[tuple[str, str]] = []
    for one in data.get():
        name = one.GetName()
        found.append((name, "value"))
        if one.hasAsymError():
            found += [(f"{name}_errlo", "errlo"), (f"{name}_errhi", "errhi")]
        elif one.hasError():
            found.append((f"{name}_err", "err"))
    return sorted(found)


def _read(one: Any, which: str) -> float:
    if which == "value":
        return float(one.getVal())
    if which == "err":
        return float(one.getError())
    return float(one.getAsymErrorLo() if which == "errlo" else one.getAsymErrorHi())


def GetAsTTree(name: Any, desc: Any, data: Any) -> Any:
    """``RooStats::GetAsTTree(name, desc, data)``: the dataset, filled into a tree of its own."""
    tree = TTree(str(name), str(desc))
    columns = _columns(data)
    buffers = {column: array("d", [INVALID]) for column, _ in columns}
    for column, _ in columns:
        tree.Branch(column, buffers[column], f"{column}/D")
    for entry in range(data.numEntries()):
        row = data.get(entry)
        for column, which in columns:
            buffers[column][0] = _read(row.find(column.rsplit("_err", 1)[0]), which)
        tree.Fill()
    return tree
