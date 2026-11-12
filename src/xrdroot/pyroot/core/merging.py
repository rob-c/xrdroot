"""``TFileMerger``: the merging ``hadd`` does, from ROOT's API.

A merger collects input files and one output file, and ``Merge`` writes every
object of every input into the output. ``PartialMerge`` merges what has been
added so far - into an output that is already there, which is what ``hadd -a``
does and what the merging tutorial asks for with ``kIncremental`` - and
``Reset`` forgets the inputs so the same merger can be given the next batch.

The work itself is :func:`xrdroot.merge`, so a merged file is the file ``hadd``
would have written: the same cycles, the same basket layout, the same
compression.
"""

from __future__ import annotations

from typing import Any

from ...merging.files import merge as _merge

__all__ = ["TFileMerger", "TFileMergerOutput"]


class TFileMergerOutput:
    """The merger's output, as ROOT's ``GetOutputFile`` hands it over: a file by name."""

    def __init__(self, name: str) -> None:
        self._name = name

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"<TFileMergerFile {self._name!r}>"


class TFileMerger:
    """``TFileMerger``: inputs merged into one output file, as ``hadd`` merges them.

    The flags are ROOT's own and combine with ``|``: ``kAll`` merges everything,
    ``kIncremental`` merges into an output already there, ``kNoTrees`` leaves
    the trees out, and ``kFastMethod`` asks for the fast tree merge, which is the
    only one this library has - so setting it changes nothing.
    """

    kNoTrees = 1
    kFastMethod = 2
    kCachedRead = 4
    kCachedWrite = 8
    kIncremental = 16
    kAll = -1
    kMessages = 32
    kVerbose = 64

    def __init__(self, printout: bool = True) -> None:
        self._printout = bool(printout)
        self._inputs: list[str] = []
        self._output: str | None = None
        self._fast = True
        self._options = TFileMerger.kAll

    # --- what to merge -------------------------------------------------------

    def OutputFile(self, name: Any, xrootd: bool = False) -> bool:
        """``OutputFile``: where the merged file goes."""
        self._output = str(name)
        return True

    def GetOutputFile(self) -> TFileMergerOutput | None:
        """``GetOutputFile``: the output's own object, or nothing before one is named."""
        return None if self._output is None else TFileMergerOutput(self._output)

    def AddFile(self, source: Any, adopt: bool = True) -> bool:
        """``AddFile``: one more input, named or already open."""
        opened = getattr(source, "GetName", None)
        self._inputs.append(str(opened() if callable(opened) else source))
        return True

    def AddFileImpl(self, source: Any) -> bool:
        """``AddFileImpl``: what a Python-side caller adds, as ``AddFile`` takes it."""
        return self.AddFile(source)

    def SetFastMethod(self, fast: bool = True) -> None:
        """``SetFastMethod``: this library merges trees the fast way only, so the flag is kept."""
        self._fast = bool(fast)

    def SetCompressionAlgorithm(self, algorithm: Any = 0) -> None:
        """``SetCompressionAlgorithm``: kept; the merged file keeps its inputs' compression."""

    def SetMsgLevel(self, level: Any = 0) -> None:
        """``SetMsgLevel``: how much ROOT reports while merging - nothing, here."""

    def SetMergeOptions(self, options: int) -> None:
        """``SetMergeOptions``: the flags a later ``PartialMerge`` merges with."""
        self._options = int(options)

    def GetMergeOptions(self) -> int:
        return self._options

    # --- merging -------------------------------------------------------------

    def PartialMerge(self, mode: int = kAll) -> bool:
        """``PartialMerge``: merge what has been added so far, leaving the merger usable.

        A ``mode`` of ``kAll`` or with ``kIncremental`` set merges into an output
        that is already there - a second cycle of each name, as ``hadd -a``
        writes it - and ``kNoTrees`` leaves the trees out.
        """
        if self._output is None or not self._inputs:
            return False
        flags = int(mode)
        trees = not (flags > 0 and flags & TFileMerger.kNoTrees)
        _merge(self._output, list(self._inputs), append=True, trees=trees)
        return True

    def Merge(self) -> bool:
        """``Merge``: everything added, into the output file."""
        return self.PartialMerge(self._options)

    def Reset(self) -> None:
        """``Reset``: forget the inputs, keeping the output, as the tutorial's loop does."""
        self._inputs.clear()

    def __enter__(self) -> TFileMerger:
        return self

    def __exit__(self, *why: Any) -> None:
        """Leaving the ``with`` merges whatever the block left unmerged."""
        if self._inputs:
            self.Merge()

    def __repr__(self) -> str:
        return f"<TFileMerger {len(self._inputs)} input(s) -> {self._output!r}>"
