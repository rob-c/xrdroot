"""The spectra the ``TSpectrum`` tests run, and ROOT 6.40's answers for them.

Each spectrum is built of sums and quotients alone, so it is the same to
the bit on every machine and in the C++ macro that printed ROOT's answers
(``tests/data/spectrum-6.40.txt``).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

__all__ = ["reference", "y64", "y128", "z2"]

DATA = Path(__file__).parent / "data" / "spectrum-6.40.txt"

#: How many words of each kind of line are its tag.
TAG_WORDS = {
    "bg": 6,
    "mk": 2,
    "gd": 2,
    "rl": 2,
    "uf": 2,
    "sd": 4,
    "sp": 4,
    "b2": 3,
    "g2": 2,
    "s2": 3,
    "x2": 3,
    "y2": 3,
}


def y64(n: int = 64) -> np.ndarray:
    """Two peaks on a sloping, stepping background: what the background tests clip."""
    return np.array(
        [
            5
            + (i * 7 % 11) * 0.25
            + 400 / (1 + (i - 20) * (i - 20) / 4.0)
            + 250 / (1 + (i - 41) * (i - 41) / 9.0)
            + (30 if i >= 50 else 0)
            for i in range(n)
        ]
    )


def y128() -> np.ndarray:
    """Two close peaks and a wide one: what the peak searches look in."""
    return np.array(
        [
            3
            + 500 / (1 + (i - 30) * (i - 30) / 4.0)
            + 300 / (1 + (i - 37) * (i - 37) / 4.0)
            + 200 / (1 + (i - 90) * (i - 90) / 9.0)
            + (i % 5)
            for i in range(128)
        ]
    )


def z2(nx: int = 24, ny: int = 18) -> np.ndarray:
    """Two peaks on a patterned floor, ``nx`` by ``ny``: the two-dimensional spectrum."""
    return np.array(
        [
            [
                2
                + ((3 * i + 5 * j) % 7)
                + 300 / (1 + ((i - 8) ** 2 + (j - 6) ** 2) / 4.0)
                + 200 / (1 + ((i - 17) ** 2 + (j - 12) ** 2) / 6.0)
                for j in range(ny)
            ]
            for i in range(nx)
        ]
    )


@cache
def reference() -> dict[str, list[float]]:
    """ROOT's numbers by tag."""
    found = {}
    for line in DATA.read_text().splitlines():
        if line.startswith("#"):
            continue
        words = line.split()
        width = TAG_WORDS.get(words[0], 1)
        found[" ".join(words[:width])] = [float(word) for word in words[width:]]
    return found


def tags(kind: str) -> list[str]:
    """The tags of every reference line of ``kind``."""
    return [tag for tag in reference() if tag.split()[0] == kind]
