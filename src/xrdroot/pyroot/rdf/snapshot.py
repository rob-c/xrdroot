"""``ROOT.RDF.RSnapshotOptions``: how ``Snapshot`` writes, as ROOT's options say it.

ROOT passes ``Snapshot`` a struct of options as its fourth argument; this is
that struct, and :func:`keywords_of` the keywords xrdroot's own ``Snapshot``
takes for it: the mode, laziness, entries per cluster, the compression
algorithm and whether the output is a ``TTree`` or an RNTuple.
"""

from __future__ import annotations

from typing import Any

__all__ = ["RSnapshotOptions", "ESnapshotOutputFormat", "keywords_of"]

#: ``ROOT::RCompressionSetting::EAlgorithm``'s algorithms, by number, as xrdroot names them.
ALGORITHMS = {1: "zlib", 2: "lzma", 4: "lz4", 5: "zstd"}


class ESnapshotOutputFormat:
    """``ROOT::RDF::ESnapshotOutputFormat``: what ``Snapshot`` writes."""

    kDefault = 0
    kTTree = 1
    kRNTuple = 2


class RSnapshotOptions:
    """``ROOT::RDF::RSnapshotOptions``, with ROOT's members and their defaults."""

    def __init__(
        self,
        mode: str = "RECREATE",
        comprAlgo: int = 0,
        comprLevel: int = 1,
        autoFlush: int = 0,
        splitLevel: int = 99,
        lazy: bool = False,
        overwriteIfExists: bool = False,
        vector2RVec: bool = True,
        basketSize: int = -1,
        outputFormat: int = ESnapshotOutputFormat.kDefault,
    ) -> None:
        self.fMode = mode
        self.fCompressionAlgorithm = comprAlgo
        self.fCompressionLevel = comprLevel
        self.fAutoFlush = autoFlush
        self.fSplitLevel = splitLevel
        self.fLazy = lazy
        self.fOverwriteIfExists = overwriteIfExists
        self.fVector2RVec = vector2RVec
        self.fBasketSize = basketSize
        self.fOutputFormat = outputFormat
        self.fIncludeVariations = False

    def __repr__(self) -> str:
        return f"<ROOT.RDF.RSnapshotOptions mode={self.fMode!r} lazy={self.fLazy}>"


def keywords_of(options: Any) -> dict[str, Any]:
    """What ``options`` asks of xrdroot's ``Snapshot``, as its keywords; nothing for none."""
    if options is None:
        return {}
    found: dict[str, Any] = {
        "mode": str(options.fMode).upper(),
        "lazy": bool(options.fLazy),
        "auto_flush": int(options.fAutoFlush),
        "rntuple": int(options.fOutputFormat) == ESnapshotOutputFormat.kRNTuple,
    }
    if int(options.fCompressionLevel) == 0:
        found["compression"] = None
    elif int(options.fCompressionAlgorithm) in ALGORITHMS:
        found["compression"] = ALGORITHMS[int(options.fCompressionAlgorithm)]
    return found
