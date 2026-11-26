"""``ROOT::RCompressionSetting`` and ``ROOT::CompressionSettings``: how ROOT names a compression.

A setting is one number, ``algorithm * 100 + level``: ``101`` is zlib at
level 1, ``505`` zstd at level 5. ``RCompressionSetting`` names the
algorithms, the levels and the settings ROOT recommends, and
``CompressionSettings`` spells a setting from an algorithm and a level, so a
script that says ``RCompressionSetting.EDefaults.kUseAnalysis`` writes what
it would have written with ROOT.
"""

from __future__ import annotations

from typing import Any

__all__ = ["RCompressionSetting", "CompressionSettings", "R__zipMultipleAlgorithm"]

#: What ``AlgorithmToString`` calls each algorithm; any other is undefined.
NAMES = {1: "zlib", 2: "LZMA", 3: "Old compression algorithm", 4: "lz4", 5: "zstd"}


class RCompressionSetting:
    """``ROOT::RCompressionSetting``: the names of ROOT's algorithms, levels and settings."""

    class EDefaults:
        """The settings ROOT recommends, whole: ``kUseAnalysis`` is LZ4 at level 4."""

        kUseGlobal = 0
        kUseCompiledDefault = 101
        kUseAnalysis = 404
        kUseGeneralPurpose = 505
        kUseSmallest = 207

    class ELevel:
        """The levels: ``kDefaultZSTD`` is the one ``505`` uses."""

        kInherit = -1
        kUncompressed = 0
        kUseMin = 1
        kDefaultZLIB = 1
        kDefaultLZ4 = 4
        kDefaultZSTD = 5
        kDefaultOld = 6
        kDefaultLZMA = 7

    class EAlgorithm:
        """The algorithms, by the number a setting carries in its hundreds."""

        kInherit = -1
        kUseGlobal = 0
        kZLIB = 1
        kLZMA = 2
        kOldCompressionAlgo = 3
        kLZ4 = 4
        kZSTD = 5
        kUndefined = 6

    @staticmethod
    def AlgorithmToString(algorithm: Any) -> str:
        return NAMES.get(int(algorithm), "Undefined compression algorithm")

    @staticmethod
    def AlgorithmFromCompressionSettings(settings: Any) -> int:
        return min(RCompressionSetting.EAlgorithm.kUndefined, int(settings) // 100)


def CompressionSettings(algorithm: Any, level: Any) -> int:
    """``ROOT::CompressionSettings``: a setting from its algorithm and level.

    The level is kept within 0 to 99 and an algorithm past the last ROOT
    knows is the global default's 0, as ROOT has it; one below zero is
    taken as it is, as ROOT takes it.
    """
    kept = max(0, min(int(level), 99))
    number = int(algorithm)
    if number >= RCompressionSetting.EAlgorithm.kUndefined:
        number = 0
    return number * 100 + kept


#: ``RZip.h``'s header: a two-letter signature, the method, then the compressed and the
#: uncompressed size, each three bytes little-endian.
HDRSIZE = 9
#: The zlib method byte a block's header carries for deflate.
Z_DEFLATED = 8
#: The two letters each algorithm writes, and the method byte after them.
SIGNATURES = {"zlib": b"ZL", "old": b"CS", "lzma": b"XZ", "lz4": b"L4", "zstd": b"ZS"}
#: The small set ``R__zipMultipleAlgorithm`` names, by the number ``EAlgorithm`` gives it.
ALGORITHMS = {
    RCompressionSetting.EAlgorithm.kZLIB: "zlib",
    RCompressionSetting.EAlgorithm.kLZMA: "lzma",
    RCompressionSetting.EAlgorithm.kOldCompressionAlgo: "old",
    RCompressionSetting.EAlgorithm.kLZ4: "lz4",
    RCompressionSetting.EAlgorithm.kZSTD: "zstd",
}


def R__zipMultipleAlgorithm(
    cxlevel: int, srcsize: Any, source: Any, tgtsize: Any, target: Any, irep: Any,
    algorithm: Any,
) -> None:
    """``R__zipMultipleAlgorithm(level, &srcsize, src, &tgtsize, tgt, &irep, algo)``.

    ROOT's own zipper: what it wrote is a nine-byte header and the compressed bytes, handed
    to ``tgt`` where it lies, with at most ``*tgtsize`` bytes to use. ``*irep`` is how many
    bytes were written - zero when the compression did not fit, which is how a caller knows
    to store the object raw. The sizes are ``int*`` in C, so they arrive as cells.
    """
    from ... import compression

    irep.value = 0
    size = int(srcsize.value)
    room = int(tgtsize.value)
    # ROOT's own guards: too little to be worth a header, no room for one, no level.
    if size < 1 + HDRSIZE + 1 or room <= HDRSIZE or int(cxlevel) <= 0:
        return
    named = _algorithm_name(algorithm)
    packed = _packed_block(compression, named, int(cxlevel), _bytes_of(source, size))
    if len(packed) > room:
        return
    _store(target, packed)
    irep.value = len(packed)


def _algorithm_name(algorithm: Any) -> str:
    """The algorithm a call names, ``kUseGlobal`` being whichever ROOT's global one is (zlib)."""
    number = int(algorithm)
    return ALGORITHMS.get(number, "zlib")


def _bytes_of(source: Any, size: int) -> bytes:
    """The ``size`` bytes a ``char *`` points at, as Python's own."""
    import numpy as np

    whole: bytes = np.asarray(source[:size], dtype=np.uint8).tobytes()
    return whole


def _packed_block(compression: Any, named: str, level: int, data: bytes) -> bytes:
    """One ROOT block: the signature, the method, the two sizes, then the compressed bytes."""
    if named == "old":
        body: bytes = compression.old_block(data, level)
        return SIGNATURES["old"] + bytes([Z_DEFLATED]) + _sizes(body, data) + body
    packed: bytes = compression.compress(data, named, level)
    return packed


def _sizes(packed: bytes, data: bytes) -> bytes:
    """The two three-byte little-endian sizes a block's header carries."""
    return len(packed).to_bytes(3, "little") + len(data).to_bytes(3, "little")


def _store(target: Any, packed: bytes) -> None:
    """``packed`` written where ``target`` points, in one go, as C writes its bytes."""
    import numpy as np

    target[: len(packed)] = np.frombuffer(packed, dtype=np.uint8).view(np.int8)
