"""``TChain``: one tree's name across many files, read as one tree.

``Add`` takes a file, a wildcard of them, a file with the tree's name after
it (``run1.root/events``, or ``run1.root?#events``), or another chain; everything a ``TTree`` reads
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
    """A file and the tree in it, from ``file.root?#tree``, ``file.root/tree`` or a file alone."""
    path, query, named = str(name).partition("?#")
    if query and named:
        return path, named
    head, dot_root, rest = str(name).partition(".root/")
    if dot_root and rest:
        return head + ".root", rest
    return str(name), tree


def _synced(entries: Any, files: list[tuple[str, str]]) -> None:
    """Each sub-list of ``entries`` numbered by the file of the chain it is of, refusing as
    ``TChain::SetEntryList(..., "sync")`` refuses lists that are not the chain's files."""
    from ...cppstd.objects import CppRuntimeError as RuntimeError

    where = "In 'TChain::SetEntryList': "
    lists = entries._lists
    if lists is None:
        raise RuntimeError(f"{where}the input TEntryList doesn't have sub entry lists. Please "
                           "make sure too add them through TEntryList::AddSubList")
    if len(lists) != len(files):
        raise RuntimeError(f"{where}the number of sub entry lists in the input TEntryList "
                           f"({len(lists)}) is not equal to the number of files in the chain "
                           f"({len(files)})")
    for index, (sub, (path, tree)) in enumerate(zip(lists, files, strict=True)):
        if (sub.GetTreeName(), sub.GetFileName()) != (tree, path):
            raise RuntimeError(f"{where}the sub entry list at index {index} doesn't correspond to "
                               f"treename '{tree}' and filename '{path}': it has treename "
                               f"'{sub.GetTreeName()}' and filename '{sub.GetFileName()}'")
        sub.SetTreeNumber(index)


class TChain(TTree):
    """``TChain``: see the module's docstring."""

    _classname = "TChain"

    #: The ``TFile`` of the entry read last, while ``gROOT`` still lists it: ROOT's chain
    #: opens the file it reads, and holds no file at all once that one is deleted.
    _file: Any = None
    #: Which of :attr:`_files` that file is, or ``-1`` for none.
    _file_number = -1

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

    def SetEntryList(self, entries: Any, option: str = "") -> None:
        """``SetEntryList(list[, "sync"])``: go through those entries; with ``"sync"`` the
        list's sub-lists are the chain's files in order, each numbered as its file is."""
        if entries is not None and "sync" in str(option).lower():
            _synced(entries, self._files)
        super().SetEntryList(entries, option)

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
        self._take_up(int(entry))
        return self._home(int(entry))[1]

    def GetEntry(self, entry: int = 0, getall: int = 0) -> int:
        """``TChain::GetEntry``: the entry read; the file it was read from is held after."""
        read = super().GetEntry(entry, getall)
        self._take_up(int(entry))
        return read

    def GetCurrentFile(self) -> Any:
        """``TChain::GetCurrentFile``: the file the entry read last is in, or ``None``.

        ROOT's chain points at the file it opened, and at nothing once that file is
        deleted - ``gROOT->GetListOfFiles()->Delete()``, or the file closed by hand. So
        the file is handed back only while ``gROOT`` still lists it, and a chain asked
        afterwards says it holds none, as ``TChain::RecursiveRemove`` leaves it.
        """
        held = self._file
        if held is None or self._still_listed(held):
            return held
        self._let_go()
        return None

    def _still_listed(self, held: Any) -> bool:
        """Whether ``gROOT`` still lists ``held``: a file is open exactly while it is there."""
        from ..core.troot import gROOT

        return held in gROOT.GetListOfFiles()

    def _let_go(self) -> None:
        """Hold no file: the one there is closed, as ``~TChain`` closes the file it opened.

        A file deleted from ``gROOT``'s list is closed here too, because ROOT's deleting it
        is what the chain is told about; ``TFile::Close`` on one already closed does nothing.
        """
        held, self._file, self._file_number = self._file, None, -1
        if held is not None:
            held.Close()

    def _take_up(self, entry: int) -> None:
        """Open the file of ``entry``, letting go of the one read before.

        ROOT's chain holds one file at a time: ``TChain::LoadTree`` opens the file of the
        entry it reads and deletes the one before, and opens that file again when it has
        been deleted since - which is what a file no longer listed comes to here.
        """
        if not 0 <= entry < self.GetEntries():
            return
        number = self._home(entry)[0]
        if number == self._file_number and self.GetCurrentFile() is not None:
            return
        self._let_go()
        self._file = self._open_file(self._files[number][0])
        self._file_number = number

    def _open_file(self, path: str) -> Any:
        """The ``TFile`` of one file of the chain, opened and the script left where it was.

        ``TChain::LoadTree`` opens its file inside a ``TDirectory::TContext``, so that
        reading a chain does not move where the objects a script makes are kept.
        """
        from ..core.directories import current_directory, set_current
        from ..core.files import TFile

        previous = current_directory()
        try:
            return TFile(path)
        finally:
            set_current(previous)

    def _destruct(self) -> None:
        """``~TChain``: the file the chain opened is closed, as ROOT's destructor closes it."""
        self._let_go()

    def GetTree(self) -> Any:
        """The tree of the file the entry last read is in, as a ``TTree`` of its own."""
        from . import wrap

        number = max(self.GetTreeNumber(), 0)
        return wrap(self._backing().tree(number))

    def GetTreeOffset(self) -> list[int]:
        return list(self._backing().starts())
