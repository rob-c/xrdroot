"""The header-data units of a FITS file, and the pixels of an image.

A FITS file is its primary unit and any extensions after it, each a header
and then its data, padded to whole blocks. The size of the data comes from
the header alone - ``|BITPIX|/8 x GCOUNT x (PCOUNT + NAXIS1 x NAXIS2 ...)``
- so the units are found without reading any data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cards import BLOCK, Record, header, number, text

__all__ = ["HDU", "IMAGE_TYPES", "image", "read_units"]

#: The big-endian type of an image's pixels, by ``BITPIX``.
IMAGE_TYPES = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}

#: What ``TFITSHDU`` calls each kind of unit, by its ``XTENSION``.
KINDS = {"TABLE": "ASCII TABLE", "BINTABLE": "BINARY TABLE"}


@dataclass
class HDU:
    """One header-data unit: its records, and the bytes of its data."""

    index: int
    records: list[Record]
    data: bytes = field(repr=False)

    def value(self, keyword: str) -> str | None:
        """A keyword's value as written, or ``None`` for one the header lacks."""
        return next((r.value for r in self.records if r.keyword == keyword), None)

    def integer(self, keyword: str, default: int = 0) -> int:
        found = self.value(keyword)
        return default if found is None else int(number(found))

    def real(self, keyword: str, default: float) -> float:
        found = self.value(keyword)
        return default if found is None else number(found)

    def string(self, keyword: str) -> str:
        return text(self.value(keyword) or "")

    @property
    def kind(self) -> str:
        """``IMAGE``, ``ASCII TABLE`` or ``BINARY TABLE``."""
        return KINDS.get(self.string("XTENSION"), "IMAGE")

    @property
    def sizes(self) -> list[int]:
        """The length of every axis, ``NAXIS1`` first."""
        return [self.integer(f"NAXIS{k}") for k in range(1, self.integer("NAXIS") + 1)]


def _data_size(records: list[Record]) -> int:
    unit = HDU(0, records, b"")
    if not unit.sizes:
        return 0
    count = int(np.prod(unit.sizes))
    bytes_per = abs(unit.integer("BITPIX", 8)) // 8
    return bytes_per * unit.integer("GCOUNT", 1) * (unit.integer("PCOUNT") + count)


def read_units(data: bytes) -> list[HDU]:
    """Every unit of a FITS file's bytes, the primary one first."""
    units: list[HDU] = []
    at = 0
    while at < len(data) and data[at:at + 8].strip(b" \0"):
        records, at = header(data, at)
        size = _data_size(records)
        units.append(HDU(len(units), records, data[at:at + size]))
        at += -(-size // BLOCK) * BLOCK
    return units


def image(unit: HDU) -> np.ndarray[Any, Any]:
    """An image's pixels as doubles, ``BSCALE`` and ``BZERO`` applied, ``NAXIS1`` fastest."""
    count = int(np.prod(unit.sizes)) if unit.sizes else 0
    raw = np.frombuffer(unit.data, IMAGE_TYPES[unit.integer("BITPIX", 8)], count)
    return raw.astype(np.float64) * unit.real("BSCALE", 1.0) + unit.real("BZERO", 0.0)
