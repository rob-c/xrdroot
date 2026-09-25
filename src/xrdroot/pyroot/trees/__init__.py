"""ROOT's trees, as PyROOT scripts and translated macros use them.

``TTree``, ``TNtuple``, ``TNtupleD`` and ``TChain``, their ``TBranch`` and
``TLeaf``, ``TEntryList``, and ``TTreeReader`` with its values and arrays -
by ROOT's names, with ROOT's arguments and what ROOT gives back - over
xrdroot's own reader, writer, ``Draw`` and ``Scan``. A tree is filled
through addresses, as in C++: see :mod:`.addresses` for what stands for a
pointer here, and :mod:`.tree` for how ``Branch`` and ``Fill`` use them.

:func:`wrap` is how ``TFile::Get`` - ``core``'s - hands back a tree it read:
the :class:`xrdroot.TTree` or :class:`xrdroot.Chain` as one of these. The
places ``core`` plugs itself in are in :mod:`._base`.
"""

from __future__ import annotations

from typing import Any

from ._base import hooks as hooks
from .branches import TBranch, TLeaf
from .chain import TChain
from .entrylist import TEntryList
from .ntuple import TNtuple, TNtupleD
from .reader import TTreeReader, TTreeReaderArray, TTreeReaderValue
from .tree import TTree

__all__ = [
    "TTree",
    "TNtuple",
    "TNtupleD",
    "TChain",
    "TBranch",
    "TLeaf",
    "TEntryList",
    "TTreeReader",
    "TTreeReaderValue",
    "TTreeReaderArray",
]

#: The class each of ROOT's tree classes is read back as.
CLASSES: dict[str, Any] = {"TTree": TTree, "TNtuple": TNtuple, "TNtupleD": TNtupleD}


def wrap(source: Any, classname: str = "TTree") -> Any:
    """The tree class here over an :class:`xrdroot.TTree` or :class:`xrdroot.Chain` read already."""
    if hasattr(source, "spans"):
        made = TChain(source.name)
        made._source = source
        made._files = [(name, source.name) for name in source.files]
        return made
    return CLASSES.get(classname, TTree)._over(source, classname)
