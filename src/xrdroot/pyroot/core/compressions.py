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

__all__ = ["RCompressionSetting", "CompressionSettings"]

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
