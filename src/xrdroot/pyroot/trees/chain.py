"""``TChain``: one tree's name across many files, read as one tree.

``Add`` takes a file, a wildcard of them, a file with the tree's name after
it (``run1.root/events``), or another chain; everything a ``TTree`` reads
with, a chain reads with, entry numbers counting across the files in the
order they were added. The reading is :class:`xrdroot.Chain`'s.
"""

from __future__ import annotations

from typing import Any

from ._base import ListOf, _TObjectLike
from .tree import TTree

__all__ = ["TChain"]


class _Element(_TObjectLike):
    """``TChainElement``: one file of a chain; the tree's name, and the file's as its title."""

    _classname = "TChainElement"


def _split(name: str, tree: str) -> tuple[str, str]:
    """A file and the tree in it, from ``file.root/tree`` or a file alone."""
    head, dot_root, rest = str(name).partition(".root/")
    if dot_root and rest:
        return head + ".root", rest
    return str(name), tree


class TChain(TTree):
    """``TChain``: see the module's docstring."""

    _classname = "TChain"

    def __init__(self, name: str = "", title: str = "", *rest: Any) -> None:
        super().__init__(name, title, dir=False)
        self._directory = None
        self._store = None
        self._files: list[tuple[str, str]] = []

    def _backing(self) -> Any:
        if self._source is None:
            if not self._files:
                raise ValueError(f"the chain {self._name!r} has no files; Add some first")
            from ...chain import Chain

            names = {tree for _, tree in self._files}
            if len(names) > 1:
                raise ValueError(
                    f"the chain {self._name!r} holds trees called {', '.join(sorted(names))}, "
                    f"and a chain reads one tree's name in every file"
                )
            self._source = Chain(names.pop(), [path for path, _ in self._files])
            self._befriend(self._source)
        return self._source

    def GetEntries(self, selection: str | None = None) -> int:
        if not self._files:
            return 0
        if selection:
            return self._selected(selection)
        return len(self._backing())

    def _layout(self) -> list[Any]:
        self._backing()
        return super()._layout()

    def Add(self, name: Any, nentries: int = -1) -> int:
        """Add a file, the files a wildcard matches, or another chain's; how many were added."""
        if isinstance(name, TChain):
            self._files.extend(name._files)
            self._changed()
            return len(name._files)
        from ...chain import _expanded

        path, tree = _split(str(name), self._name)
        found = [str(each) for each in _expanded(path)]
        self._files.extend((each, tree) for each in found)
        self._source = None
        self._changed()
        return len(found)

    def AddFile(self, name: str, nentries: int = -1, tname: str = "") -> int:
        return self.Add(f"{name}/{tname}" if tname else name)

    def GetNtrees(self) -> int:
        return len(self._files)

    def GetListOfFiles(self) -> ListOf:
        return ListOf(_Element(tree, path) for path, tree in self._files)

    def GetTreeNumber(self) -> int:
        """Which file the entry last read is in, counting from zero."""
        if self._read_entry < 0:
            return -1
        return self._home(self._read_entry)[0]

    def _home(self, entry: int) -> tuple[int, int]:
        starts = self._backing().starts()
        number = max(at for at, start in enumerate(starts[:-1]) if start <= entry)
        return number, entry - starts[number]

    def LoadTree(self, entry: int) -> int:
        """Make ``entry`` the one being read; its number in its own file's tree comes back."""
        if not 0 <= entry < self.GetEntries():
            return -2
        self._read_entry = int(entry)
        return self._home(int(entry))[1]

    def GetTree(self) -> Any:
        """The tree of the file the entry last read is in, as a ``TTree`` of its own."""
        from . import wrap

        number = max(self.GetTreeNumber(), 0)
        return wrap(self._backing().tree(number))

    def GetTreeOffset(self) -> list[int]:
        return list(self._backing().starts())
