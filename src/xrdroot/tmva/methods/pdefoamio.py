"""A PDE-Foam's ``_foams.root`` file: each foam a tree of its cells, named as TMVA names the foam.

TMVA streams its ``PDEFoam`` objects into the file; xrdroot writes each
foam's cells as the entries of a tree of the same name instead. Both are
read back: a ``PDEFoam`` by its cells, each ``PDEFoamCell`` pointing at its
parent and daughters by ``TRef`` - the identifier of the cell it means.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ...errors import ROOTError
from ..foamcells import Cells
from ..log import Logger

__all__ = ["pdefoam_columns", "read_foams", "write_foams"]

#: The bits of a ``TObject``'s identifier that number it within its process, as ``TRef`` keeps.
UID_MASK = 0xFFFFFF


def write_foams(path: str, foams: list[Cells], names: list[str]) -> None:
    """Every foam into ``path``, recreated, a tree per foam."""
    import xrdroot

    from ...wtree import spec_of

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with xrdroot.create(path) as out:
        for foam, name in zip(foams, names):
            columns = foam.columns()
            specs = {column: spec_of(column, values) for column, values in columns.items()}
            out.tree(name, specs, title=name).extend(columns)


def read_foams(path: str, names: list[str], dim: int, log: Logger) -> list[Cells]:
    """The foams ``names`` of ``path``, as :func:`write_foams` wrote them."""
    import xrdroot

    found = []
    with xrdroot.open_root(path) as source:
        for name in names:
            try:
                columns = _columns(source[name], dim)
            except (KeyError, ROOTError, TypeError, AttributeError, ValueError):
                raise log.refuse(
                    f"The foam {name} of {path} is neither a PDEFoam of cells holding at most "
                    "two numbers each nor a tree of cells xrdroot wrote; retrain the method "
                    "with xrdroot"
                ) from None
            found.append(Cells.from_columns(dim, {k: np.asarray(v) for k, v in columns.items()}))
    return found


def _columns(stored: Any, dim: int) -> dict[str, Any]:
    """The cell columns of a foam: TMVA's streamed ``PDEFoam``, or xrdroot's tree."""
    if isinstance(stored, dict) and "TMVA::PDEFoam" in stored:
        return pdefoam_columns(stored["TMVA::PDEFoam"])
    return dict(stored.arrays(list(Cells(dim, 0).columns())))


def _serials(cells: list[dict[str, Any]]) -> dict[int, int]:
    """Each cell's ``TRef`` identifier against its place in the foam, ``0`` (null) as -1."""
    serials = {int(cell["TObject"]["fUniqueID"]) & UID_MASK: int(cell["fSerial"]) for cell in cells}
    return {0: -1, **serials}


def _element(cell: dict[str, Any]) -> list[float]:
    """A cell's ``TVectorD`` of what it holds, two numbers at most, zeros for none."""
    held = cell["fElement"]
    values = [] if held is None else [float(v) for v in held["fElements"]]
    if len(values) > 2:
        raise ValueError(f"a cell holding {len(values)} numbers")
    return values + [0.0] * (2 - len(values))


#: The cell columns read straight off each ``PDEFoamCell``, against its member and type.
PLAIN = {
    "status": ("fStatus", int),
    "best": ("fBest", int),
    "xdiv": ("fXdiv", float),
    "intg": ("fIntegral", float),
    "driv": ("fDrive", float),
}
#: The cell columns that are a ``TRef`` to another cell, against its member.
LINKS = {"parent": "fParent", "dau0": "fDaught0", "dau1": "fDaught1"}


def pdefoam_columns(foam: dict[str, Any]) -> dict[str, Any]:
    """TMVA's streamed ``PDEFoam`` as the columns :meth:`Cells.from_columns` takes."""
    cells = sorted(foam["fCells"], key=lambda cell: int(cell["fSerial"]))
    serial = _serials(cells)
    columns: dict[str, Any] = {
        column: [kind(cell[member]) for cell in cells] for column, (member, kind) in PLAIN.items()
    }
    for column, member in LINKS.items():
        uids = [int(cell[member]["TObject"]["fUniqueID"]) & UID_MASK for cell in cells]
        columns[column] = [serial[uid] for uid in uids]
    elements = [_element(cell) for cell in cells]
    columns["element0"] = [pair[0] for pair in elements]
    columns["element1"] = [pair[1] for pair in elements]
    return columns
