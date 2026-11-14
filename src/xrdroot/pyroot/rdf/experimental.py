"""``ROOT.RDF.Experimental``: a dataset described sample by sample, and distributed RDF.

``RDatasetSpec`` says what a frame reads: samples, each trees in files with
a name and metadata of its own, friends read beside them, and the range of
entries to go through. ``RDataFrame(spec)`` reads it as a chain of the
samples' trees. ``Distributed`` is ``ROOT.RDF.Distributed`` under the name
ROOT 6.36 gave up, warning as ROOT does when it is used.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .treeutils import RFriendInfo, chain_of

__all__ = ["RMetaData", "RSample", "RDatasetSpec", "Experimental"]


class RMetaData:
    """``RMetaData``: a sample's metadata, a value for each key."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def Add(self, key: str, value: Any) -> None:
        self._values[str(key)] = value

    def _get(self, key: str, default: Any, kind: type) -> Any:
        if str(key) in self._values:
            return kind(self._values[str(key)])
        if default is None:
            raise RuntimeError(f"No key with name {key} in the metadata object.")
        return kind(default)

    def GetI(self, key: str, default: Any = None) -> int:
        return int(self._get(key, default, int))

    def GetD(self, key: str, default: Any = None) -> float:
        return float(self._get(key, default, float))

    def GetS(self, key: str, default: Any = None) -> str:
        return str(self._get(key, default, str))

    def Dump(self, key: str) -> str:
        return repr(self._values[str(key)])


def _names(given: Any) -> list[str]:
    """One name, or several - a list, a tuple or a vector of them - as a list."""
    if isinstance(given, str):
        return [given]
    return [str(each) for each in given]


class RSample:
    """``RSample``: trees of a name, or of a name each, in files, with metadata."""

    def __init__(
        self, sampleName: str, treeNames: Any, fileNameGlobs: Any, metaData: Any = None
    ) -> None:
        self._name = str(sampleName)
        self._files = _names(fileNameGlobs)
        trees = _names(treeNames)
        self._trees = trees * len(self._files) if len(trees) == 1 else trees
        if len(self._trees) != len(self._files):
            raise ValueError(
                f"the sample {self._name!r} has {len(self._trees)} tree names for "
                f"{len(self._files)} files; give one name, or one for each file"
            )
        self._meta = RMetaData() if metaData is None else metaData

    def GetSampleName(self) -> str:
        return self._name

    def GetTreeNames(self) -> list[str]:
        return list(self._trees)

    def GetFileNameGlobs(self) -> list[str]:
        return list(self._files)

    def GetMetaData(self) -> RMetaData:
        return self._meta


class RDatasetSpec:
    """``RDatasetSpec``: the samples a frame reads, their friends and the range of entries."""

    def __init__(self) -> None:
        self._samples: list[RSample] = []
        self._friends = RFriendInfo()
        self._range: tuple[int, int | None] = (0, None)

    def AddSample(self, sample: Any) -> RDatasetSpec:
        self._samples.append(sample if isinstance(sample, RSample) else RSample(*sample))
        return self

    def WithGlobalRange(self, entries: Any) -> RDatasetSpec:
        begin, end = entries
        self._range = (int(begin), None if end is None else int(end))
        return self

    def WithGlobalFriends(
        self, treeNames: Any, fileNameGlobs: Any, alias: str = ""
    ) -> RDatasetSpec:
        trees, files = _names(treeNames), _names(fileNameGlobs)
        trees = trees * len(files) if len(trees) == 1 else trees
        self._friends.add(trees[0], str(alias or trees[0]), files, trees)
        return self

    def GetSampleNames(self) -> list[str]:
        return [sample.GetSampleName() for sample in self._samples]

    def GetTreeNames(self) -> list[str]:
        return [tree for sample in self._samples for tree in sample.GetTreeNames()]

    def GetFileNameGlobs(self) -> list[str]:
        return [file for sample in self._samples for file in sample.GetFileNameGlobs()]

    def GetSamples(self) -> list[RSample]:
        return list(self._samples)

    def GetEntryRangeBegin(self) -> int:
        return self._range[0]

    def GetEntryRangeEnd(self) -> int | None:
        return self._range[1]

    def GetFriendInfo(self) -> RFriendInfo:
        return self._friends

    def frame(self, **options: Any) -> Any:
        """xrdroot's frame of the samples' trees, chained, through the range of entries."""
        from ... import rdf as _rdf
        from ...rdf.distributed import Window, source_of
        from .frame import RDataFrame

        chain = chain_of(self.GetTreeNames(), self.GetFileNameGlobs(), self._friends)
        whole = RDataFrame(chain, **options)._inner
        begin, end = self._range
        if (begin, end) == (0, None):
            return whole
        source = source_of(whole)
        stop = len(source) if end is None else min(end, len(source))
        return _rdf.RDataFrame(Window(source, begin, stop), **options)


class _Experimental:
    """``ROOT.RDF.Experimental``."""

    RDatasetSpec = RDatasetSpec
    RSample = RSample
    RMetaData = RMetaData

    def __getattr__(self, name: str) -> Any:
        if name == "Distributed":
            from .. import _distrdf

            return _distrdf.distributed_module(experimental=True)
        raise AttributeError(f"ROOT has RDF.Experimental.{name}; xrdroot.pyroot does not yet")

    def __repr__(self) -> str:
        return "<namespace ROOT::RDF::Experimental>"

    def __dir__(self) -> Iterable[str]:
        return [*super().__dir__(), "Distributed"]


#: ``ROOT.RDF.Experimental``.
Experimental = _Experimental()
