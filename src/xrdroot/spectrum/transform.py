"""``TSpectrumTransform``'s ``Transform``, ``FilterZonal`` and ``Enhance``, as ROOT does them.

A transform's settings are kept in a :class:`Settings`, which the three
change as ROOT's methods change their object: a cosine or sine transform
doubles ``size`` - ``Transform`` and the inverse half of ``FilterZonal`` and
``Enhance`` never halve it back - and a cosine or sine mixed transform adds
one to ``degree`` each time it runs. So a second call on the same object does
what ROOT's second call does, however surprising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..random import libm
from .transformfold import (
    correct_cos,
    correct_sin,
    deal,
    gather,
    load_mirrored,
    undeal,
    unfold_cos,
    unfold_sin,
)
from .transformkernels import bit_reverse, fourier, haar, walsh
from .transformmixed import bit_reverse_haar, general_exe, general_inv
from .transformtypes import (
    COS,
    COS_MIXED,
    COS_WALSH,
    FORWARD,
    FOURIER,
    FOURIER_MIXED,
    HAAR,
    HARTLEY,
    INVERSE,
    PI,
    PLAIN_MIXED,
    SIN,
    SIN_HAAR,
    SQRT2,
    WALSH,
    power2,
)

__all__ = [
    "Settings", "cos_forward", "cos_inverse", "enhance", "filter_zonal", "forward", "inverse",
    "sin_forward", "sin_inverse", "transform",
]  # fmt: skip


@dataclass
class Settings:
    """What a ``TSpectrumTransform`` holds: ROOT's ``fSize``, ``fTransformType`` and the rest."""

    size: int = 0
    kind: int = COS
    degree: int = 0
    direction: int = FORWARD
    xmin: int = 0
    xmax: int = 0
    filter_coeff: float = 0.0
    enhance_coeff: float = 0.5


def workspace_length(kind: int, size: int) -> int:
    """How many ``Double_t`` ROOT allocates for its working space: 2, 4 or 8 spectra's worth."""
    if kind in (HAAR, WALSH):
        return 2 * size
    return 8 * size if kind >= COS_WALSH else 4 * size


def source_length(settings: Settings) -> int:
    """How many numbers a call reads from its source: two spectra's for a Fourier inverse."""
    fourier_like = settings.kind == FOURIER or settings.kind in FOURIER_MIXED
    if settings.direction == INVERSE and fourier_like:
        return 2 * settings.size
    return settings.size


def _begin(settings: Settings) -> Any:
    """What every call does first: a cosine mixed transform's degree raised, the space made."""
    if COS_WALSH <= settings.kind <= SIN_HAAR:
        settings.degree += 1
    return np.zeros(workspace_length(settings.kind, settings.size))


def cos_forward(ws: Any, source: Any, n: int) -> None:
    """The cosine transform: the spectrum and its mirror through the FFT, the shift divided out."""
    ws[:n], ws[n : 2 * n] = source[:n], source[:n][::-1]
    fourier(ws, 2 * n, 0, FORWARD, 0)
    ws[:n] = ws[:n] / libm.cos(PI * np.arange(n, dtype=np.float64) / float(2 * n))
    ws[2 * n : 3 * n] = 0.0
    ws[0] = ws[0] / SQRT2


def sin_forward(ws: Any, source: Any, n: int) -> None:
    """The sine transform: the spectrum against its negated mirror, one place lower when done."""
    ws[:n], ws[n : 2 * n] = source[:n], -source[:n][::-1]
    fourier(ws, 2 * n, 0, FORWARD, 0)
    ws[: n - 1] = ws[1:n] / libm.sin(PI * np.arange(1, n, dtype=np.float64) / float(2 * n))
    ws[2 * n : 3 * n] = 0.0
    ws[n - 1] = ws[n] / SQRT2


def _mixed_forward(ws: Any, source: Any, n: int, kind: int, degree: int) -> None:
    """A mixed transform of degree ``degree``, its coefficients dealt out as ROOT deals them."""
    if kind in PLAIN_MIXED:
        ws[:n] = source[:n]
        k = power2(degree)
        for i in range(n // k):
            bit_reverse_haar(ws, n, k, i * k)
        general_exe(ws, 0, n, degree, kind)
    else:
        load_mirrored(ws, source[:n], degree, 1.0 if kind in COS_MIXED else -1.0)
        m = power2(degree)
        for i in range(2 * n // m):
            bit_reverse_haar(ws, 2 * n, m, i * m)
        general_exe(ws, 0, 2 * n, degree, kind)
        (correct_cos if kind in COS_MIXED else correct_sin)(ws, n, degree, 2 * n)
    deal(ws, n, degree, kind, 2 * n)


def forward(ws: Any, source: Any, n: int, kind: int, degree: int) -> None:
    """The forward transform of ``source`` into the working space, whatever its kind."""
    if kind == HAAR:
        ws[:n] = source[:n]
        haar(ws, n, FORWARD)
    elif kind == WALSH:
        ws[:n] = source[:n]
        walsh(ws, n)
        bit_reverse(ws, n)
    elif kind == COS:
        cos_forward(ws, source, n)
    elif kind == SIN:
        sin_forward(ws, source, n)
    elif kind in (FOURIER, HARTLEY):
        ws[:n] = source[:n]
        fourier(ws, n, int(kind == HARTLEY), FORWARD, 0)
    else:
        _mixed_forward(ws, source, n, kind, degree)


def _mirror_back(ws: Any, n: int) -> None:
    """The inverse cosine's and sine's second halves: the first half's mirror, conjugated."""
    big = 2 * n
    ws[n + 1 : big] = ws[1:n][::-1]
    ws[big + n + 1 : 2 * big] = -ws[big + 1 : big + n][::-1]


def cos_inverse(ws: Any, n: int) -> Any:
    """The inverse cosine transform, through an inverse FFT of twice the length."""
    big = 2 * n
    ws[0] = ws[0] * SQRT2
    angle = PI * np.arange(n, dtype=np.float64) / float(big)
    ws[big : big + n] = ws[:n] * libm.sin(angle)
    ws[:n] = ws[:n] * libm.cos(angle)
    _mirror_back(ws, n)
    ws[n] = 0.0
    ws[n + big] = 0.0
    fourier(ws, big, 0, INVERSE, 1)
    return ws[:n].copy()


def sin_inverse(ws: Any, n: int) -> Any:
    """The inverse sine transform, each coefficient first moved one place up."""
    big = 2 * n
    ws[n] = ws[n - 1] * SQRT2
    angle = PI * np.arange(1, n, dtype=np.float64) / float(big)
    lower = ws[: n - 1].copy()
    ws[big + 1 : big + n] = -lower * libm.cos(angle)
    ws[1:n] = lower * libm.sin(angle)
    _mirror_back(ws, n)
    ws[0] = 0.0
    ws[big] = 0.0
    ws[n + big] = 0.0
    fourier(ws, big, 0, INVERSE, 0)
    return ws[:n].copy()


def _mixed_inverse(ws: Any, n: int, kind: int, degree: int) -> Any:
    """A mixed transform undone: coefficients gathered back, butterflies reversed."""
    k = undeal(ws, n, degree, kind, 2 * n)
    if kind in PLAIN_MIXED:
        general_inv(ws, n, degree, kind)
        for i in range(n // k):
            bit_reverse_haar(ws, n, k, i * k)
        return ws[:n].copy()
    (unfold_cos if kind in COS_MIXED else unfold_sin)(ws, n, degree)
    general_inv(ws, 2 * n, degree, kind)
    m = power2(degree)
    for i in range(2 * n // m):
        bit_reverse_haar(ws, 2 * n, m, i * m)
    return gather(ws, n, degree)


def inverse(ws: Any, n: int, kind: int, degree: int) -> Any:
    """The inverse transform of the working space: the spectrum it stands for."""
    if kind == HAAR:
        haar(ws, n, INVERSE)
    elif kind == WALSH:
        bit_reverse(ws, n)
        walsh(ws, n)
    elif kind == COS:
        return cos_inverse(ws, n)
    elif kind == SIN:
        return sin_inverse(ws, n)
    elif kind in (FOURIER, HARTLEY):
        fourier(ws, n, int(kind == HARTLEY), INVERSE, 0)
    else:
        return _mixed_inverse(ws, n, kind, degree)
    return ws[:n].copy()


def _coefficients(ws: Any, n: int, kind: int) -> Any:
    """What a forward ``Transform`` hands back: the coefficients, the imaginary ones after."""
    if kind == FOURIER:
        return ws[: 2 * n].copy()
    if kind in FOURIER_MIXED:
        return np.concatenate([ws[:n], ws[2 * n : 3 * n]])
    return ws[:n].copy()


def _load_coefficients(ws: Any, source: Any, n: int, kind: int) -> None:
    """An inverse ``Transform``'s coefficients put where the forward one left them."""
    if kind == FOURIER:
        ws[: 2 * n] = source[: 2 * n]
        return
    ws[:n] = source[:n]
    if kind in FOURIER_MIXED:
        ws[2 * n : 3 * n] = source[n : 2 * n]


def _grown(settings: Settings) -> None:
    """A cosine's or sine's doubled ``fSize``, which ROOT leaves doubled."""
    if settings.kind in (COS, SIN):
        settings.size *= 2


def transform(settings: Settings, source: Any) -> Any:
    """``Transform``: the forward coefficients of ``source``, or the spectrum they stand for."""
    ws = _begin(settings)
    n, kind, degree = settings.size, settings.kind, settings.degree
    if settings.direction == FORWARD:
        forward(ws, source, n, kind, degree)
        result = _coefficients(ws, n, kind)
    else:
        _load_coefficients(ws, source, n, kind)
        result = inverse(ws, n, kind, degree)
    _grown(settings)
    return result


def _reweighted(part: Any, inside: Any, change: Any) -> None:
    """Coefficients in the region changed, then all scaled back to the same sum as before."""
    old_area = _sum(part)
    part[inside] = change(part[inside])
    new_area = _sum(part)
    if new_area != 0:
        part *= old_area / new_area


def _sum(values: Any) -> float:
    """A sum added term by term from the first, as ROOT's loop adds it."""
    total = 0.0
    for value in values.tolist():
        total += value
    return total


def _modified(settings: Settings, source: Any, change: Any) -> Any:
    """Forward, the region's coefficients changed in every part the kind has, and back."""
    ws = _begin(settings)
    n, kind, degree = settings.size, settings.kind, settings.degree
    forward(ws, source, n, kind, degree)
    inside = np.arange(n)
    inside = (inside >= settings.xmin) & (inside <= settings.xmax)
    _reweighted(ws[:n], inside, change)
    if kind == FOURIER:
        _reweighted(ws[n : 2 * n], inside, change)
    elif kind in FOURIER_MIXED:
        _reweighted(ws[2 * n : 3 * n], inside, change)
    result = inverse(ws, n, kind, degree)
    _grown(settings)
    return result


def filter_zonal(settings: Settings, source: Any) -> Any:
    """``FilterZonal``: the coefficients in the region set to the filter coefficient."""
    return _modified(settings, source, lambda part: np.full(len(part), settings.filter_coeff))


def enhance(settings: Settings, source: Any) -> Any:
    """``Enhance``: the coefficients in the region multiplied by the enhancement coefficient."""
    return _modified(settings, source, lambda part: part * settings.enhance_coeff)
