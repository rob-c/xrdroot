"""``TSpectrum2Transform``'s ``Transform``, ``FilterZonal`` and ``Enhance``, as ROOT does them.

Unlike its 1-D sibling, ROOT's 2-D transform changes nothing it holds when
it runs, so the same settings give the same answer every time. Its filter
and enhancement differ from the 1-D ones too: the spectrum's sum is kept by
scaling the spectrum that comes back, not the coefficients, and when that
comes back summing to zero ROOT writes nothing at all - which here is
``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .transform import workspace_length
from .transform2general import general2
from .transform2passes import four_cos2, haar_walsh2
from .transformtypes import COS, FORWARD, FOURIER, FOURIER_MIXED, FOURIER_WALSH, INVERSE, WALSH

__all__ = ["Settings2", "enhance2", "filter_zonal2", "transform2"]


@dataclass
class Settings2:
    """What a ``TSpectrum2Transform`` holds: ROOT's ``fSizeX``, ``fSizeY`` and the rest."""

    sizex: int = 0
    sizey: int = 0
    kind: int = COS
    degree: int = 0
    direction: int = FORWARD
    xmin: int = 0
    xmax: int = 0
    ymin: int = 0
    ymax: int = 0
    filter_coeff: float = 0.0
    enhance_coeff: float = 0.5


def has_imaginary(kind: int) -> bool:
    """Whether a kind's coefficients have an imaginary part: Fourier's and its mixed ones'."""
    return kind == FOURIER or kind in FOURIER_MIXED


def _run(settings: Settings2, matrix: Any, direction: int) -> None:
    """The 2-D transform of ``matrix`` in place, through ROOT's one working vector."""
    kind = settings.kind
    wv = np.zeros(workspace_length(kind, max(settings.sizex, settings.sizey)))
    if kind <= WALSH:
        haar_walsh2(matrix, wv, direction, kind)
    elif kind < FOURIER_WALSH:
        four_cos2(matrix, wv, direction, kind)
    else:
        general2(matrix, wv, direction, kind, settings.degree)


def _matrix(settings: Settings2, source: Any, imaginary: bool) -> Any:
    """The working matrix: ``source``'s real part, and its imaginary part if one is read."""
    nx, ny = settings.sizex, settings.sizey
    matrix = np.zeros((nx, 2 * ny))
    matrix[:, : 2 * ny if imaginary else ny] = source[:nx, : 2 * ny if imaginary else ny]
    return matrix


def transform2(settings: Settings2, source: Any) -> Any:
    """``Transform``: the coefficients - the imaginary ones beside them - or the spectrum."""
    ny = settings.sizey
    imaginary = has_imaginary(settings.kind)
    matrix = _matrix(settings, source, imaginary and settings.direction != FORWARD)
    _run(settings, matrix, settings.direction)
    wide = imaginary and settings.direction == FORWARD
    return matrix[:, : 2 * ny if wide else ny].copy()


def _sum(values: Any) -> float:
    """A sum added term by term, rows outermost, as ROOT's loops add it."""
    total = 0.0
    for value in np.asarray(values).ravel().tolist():
        total += value
    return total


def _modified(settings: Settings2, source: Any, change: Any) -> Any:
    """Forward, the region's coefficients changed, back, and scaled to the old sum."""
    nx, ny = settings.sizex, settings.sizey
    matrix = _matrix(settings, source, False)
    old_area = _sum(matrix[:, :ny])
    _run(settings, matrix, FORWARD)
    region = (slice(settings.xmin, settings.xmax + 1), slice(settings.ymin, settings.ymax + 1))
    matrix[region] = change(matrix[region])
    if has_imaginary(settings.kind):
        imag = (region[0], slice(ny + settings.ymin, ny + settings.ymax + 1))
        matrix[imag] = change(matrix[imag])
    _run(settings, matrix, INVERSE)
    new_area = _sum(matrix[:nx, :ny])
    if new_area == 0:
        return None
    return matrix[:, :ny] * (old_area / new_area)


def filter_zonal2(settings: Settings2, source: Any) -> Any:
    """``FilterZonal``: the region's coefficients set to the filter coefficient."""
    return _modified(settings, source, lambda part: np.full(part.shape, settings.filter_coeff))


def enhance2(settings: Settings2, source: Any) -> Any:
    """``Enhance``: the region's coefficients multiplied by the enhancement coefficient."""
    return _modified(settings, source, lambda part: part * settings.enhance_coeff)
