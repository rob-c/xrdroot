"""A PDE-Foam's ``_foams.root`` file: each foam a tree of its cells, named as TMVA names the foam.

TMVA streams its ``PDEFoam`` objects into the file; xrdroot writes each
foam's cells as the entries of a tree of the same name instead, which it
reads back. A file of TMVA's own holds no such trees, and is refused in
so many words.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ...errors import ROOTError
from ..foamcells import Cells
from ..log import Logger

__all__ = ["read_foams", "write_foams"]


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
                tree = source[name]
                columns = tree.arrays(list(Cells(dim, 0).columns()))
            except (KeyError, ROOTError, TypeError, AttributeError):
                raise log.refuse(
                    f"The foam {name} of {path} is not one xrdroot wrote: TMVA's own foam files "
                    "hold PDEFoam objects, which xrdroot cannot read; retrain the method with "
                    "xrdroot"
                ) from None
            found.append(Cells.from_columns(dim, {k: np.asarray(v) for k, v in columns.items()}))
    return found
