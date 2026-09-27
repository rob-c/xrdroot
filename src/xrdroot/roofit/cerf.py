"""The complex error function as RooFit computes it: ``RooHeterogeneousMath``'s Faddeeva function.

A decay convolved with a Gaussian resolution is, in closed form, the
Faddeeva function ``w(z) = exp(-z^2) erfc(-iz)`` of a complex argument made
of the decay's lifetime and frequency and the resolution's width. RooFit
computes it with Manuel Schiller's approximation - a Fourier sum, a
continued fraction far from the origin and Taylor series near the poles of
the sum - and so does this module, operation for operation and on NumPy
arrays, so that a convolution here has ROOT's value to the last bit.

:func:`eval_cerf` is ``evalCerf``, ``exp(-u^2) w(swt c + i(u+c))``, which is
what ``RooGaussModel`` asks for; along the imaginary axis it is a real
``exp * erfc``, and where the argument is far below the real axis it is
``evalCerfApprox``, which cancels the divergence by hand.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import mathfuncs as mf

__all__ = ["eval_cerf", "eval_cerf_approx", "faddeeva_fast"]

#: ``npi11``: ``n pi`` for the eleven terms of the fast Fourier sum.
NPI = np.array(
    [
        0.00000000000000000e00,
        3.14159265358979324e00,
        6.28318530717958648e00,
        9.42477796076937972e00,
        1.25663706143591730e01,
        1.57079632679489662e01,
        1.88495559215387594e01,
        2.19911485751285527e01,
        2.51327412287183459e01,
        2.82743338823081391e01,
        3.14159265358979324e01,
    ]
)
#: ``a11``: the Fourier coefficients' prefactors.
A = np.array(
    [
        4.43113462726379007e-01,
        3.79788034073635143e-01,
        2.39122407410867584e-01,
        1.10599187402169792e-01,
        3.75782250080904725e-02,
        9.37936104296856288e-03,
        1.71974046186334976e-03,
        2.31635559000523461e-04,
        2.29192401420125452e-05,
        1.66589592139340077e-06,
        8.89504561311882155e-08,
    ]
)
#: ``taylorarr11``: per pole, three complex Taylor coefficients, the highest order first.
TAYLOR = np.array(
    [
        -1.00000000000000000e00,
        0.00000000000000000e00,
        0.00000000000000000e-01,
        1.12837916709551257e00,
        1.00000000000000000e00,
        0.00000000000000000e00,
        -5.92741768247463996e-01,
        -7.19914991991294310e-01,
        -6.73156763521649944e-01,
        8.14025039279059577e-01,
        8.57089811121701143e-01,
        4.00248106586639754e-01,
        1.26114512111568737e-01,
        -7.46519337025968199e-01,
        -8.47666863706379907e-01,
        1.89347715957263646e-01,
        5.39641485816297176e-01,
        5.97805988669631615e-01,
        4.43238482668529408e-01,
        -3.03563167310638372e-01,
        -5.88095866853990048e-01,
        -2.32638360700858412e-01,
        2.49595637924601714e-01,
        5.77633779156009340e-01,
        3.33690792296469441e-01,
        3.97048587678703930e-02,
        -2.66422678503135697e-01,
        -3.18469797424381480e-01,
        8.48049724711137773e-02,
        4.60546329221462864e-01,
        1.42043544696751869e-01,
        1.24094227867032671e-01,
        -8.31224229982140323e-02,
        -2.40766729258442100e-01,
        2.11669512031059302e-02,
        3.48650139549945097e-01,
        3.92113167048952835e-02,
        9.03306084789976219e-02,
        -1.82889636251263500e-02,
        -1.53816215444915245e-01,
        3.88103861995563741e-03,
        2.72090310854550347e-01,
        7.37741897722738503e-03,
        5.04625223970221539e-02,
        -2.87394336989990770e-03,
        -9.96122819257496929e-02,
        5.22745478269428248e-04,
        2.23361039070072101e-01,
        9.69251586187208358e-04,
        2.83055679874589732e-02,
        -3.24986363596307374e-04,
        -6.97056268370209313e-02,
        5.17231862038123061e-05,
        1.90681117197597520e-01,
        9.01625563468897100e-05,
        1.74961124275657019e-02,
        -2.65745127697337342e-05,
        -5.22070356354932341e-02,
        3.75952450449939411e-06,
        1.67018782142871146e-01,
        5.99057675687392260e-06,
        1.17993805017130890e-02,
        -1.57660578509526722e-06,
        -4.09165023743669707e-02,
        2.00739683204152177e-07,
        1.48879348585662670e-01,
    ]
).reshape(11, 3, 2)
#: The fast variant's numbers of terms - sum, Taylor series, continued fraction - and its ``tm``.
N, NTAYLOR, NCF, TM = 11, 3, 3, 8.0
#: ``9/1000000``: how close to a pole - squared - a point must be for its Taylor series to be used.
MAXNORM = 9.0 / 1000000.0
#: ``1/sqrt(pi)`` and ``2 sqrt(pi)`` as RooFit spells them.
ISQRTPI, TWOSQRTPI = 5.64189583547756287e-01, 3.54490770181103205e00

Array = Any


def _cexp(re: Array, im: Array) -> tuple[Array, Array]:
    """``cexp``: ``exp(re + i im)`` as its real and imaginary parts."""
    e = np.exp(re)
    return e * np.cos(im), e * np.sin(im)


def _taylor(zre: Array, zim: Array) -> tuple[Array, Array, Array]:
    """Where a point is close to a pole of the sum: that pole's Taylor series, and where that is."""
    zim2 = zim * zim
    dnsing = TM * zre / NPI[1]
    near = (zim2 < MAXNORM) & (dnsing * dnsing < (N - 0.5) * (N - 0.5))
    nsing = np.where(near, np.abs(dnsing) + 0.5, 0.0).astype(np.int64)
    zmnpire = np.abs(zre) - NPI[nsing]
    near &= zmnpire * zmnpire + zim2 < MAXNORM
    coeffs = TAYLOR[nsing]
    sumre, sumim = coeffs[..., 0, 0], coeffs[..., 0, 1]
    for i in range(1, NTAYLOR):
        re = sumre * zmnpire - sumim * zim
        im = sumim * zmnpire + sumre * zim
        sumre, sumim = re + coeffs[..., i, 0], im + coeffs[..., i, 1]
    return near, sumre, np.where(zre < 0.0, -sumim, sumim)


def _fraction_steps(z2re: Array, z2im: Array) -> tuple[Array, Array, Array]:
    """The continued fraction's ``NCF`` steps, from the innermost out."""
    cfre, cfim, cfnorm = np.ones_like(z2re), np.zeros_like(z2re), np.ones_like(z2re)
    for k in range(NCF, 0, -1):
        cfre = +(k / 2.0) * cfre / cfnorm
        cfim = -(k / 2.0) * cfim / cfnorm
        if k & 1:
            cfre, cfim = cfre - z2re, cfim - z2im
        else:
            cfre = cfre + 1.0
        cfnorm = cfre * cfre + cfim * cfim
    return cfre, cfim, cfnorm


def _continued_fraction(zre: Array, zim: Array, negimz: Array) -> tuple[Array, Array]:
    """``w(z)`` far from the origin, for ``Im z >= 0`` - flipped back if ``z`` was below."""
    z2re, z2im = (zre + zim) * (zre - zim), 2.0 * zre * zim
    cfre, cfim, cfnorm = _fraction_steps(z2re, z2im)
    sumre = (zim * cfre - zre * cfim) * ISQRTPI / cfnorm
    sumim = -(zre * cfre + zim * cfim) * ISQRTPI / cfnorm
    ez2re, ez2im = _cexp(-z2re, -z2im)
    return (
        np.where(negimz, 2.0 * ez2re - sumre, sumre),
        np.where(negimz, 2.0 * ez2im - sumim, sumim),
    )


def _numerators(tmzre: Array, tmzim: Array) -> tuple[Array, ...]:
    """``1 -/+ exp(i tm z)``, and each times ``tm z``: the numerators of the Fourier sum's terms."""
    eitmzre, eitmzim = _cexp(-tmzim, tmzre)
    n = (1.0 - eitmzre, -eitmzim, 1.0 + eitmzre, +eitmzim)
    return (
        *n,
        tmzre * n[0] - tmzim * n[1],
        tmzre * n[1] + tmzim * n[0],
        tmzre * n[2] - tmzim * n[3],
        tmzre * n[3] + tmzim * n[2],
    )


def _fourier_term(
    i: int,
    numer: tuple[Array, ...],
    tmzre: Array,
    imtmz2: Array,
    reimtmzm2: Array,
    reimtmzm22: Array,
) -> tuple[Array, Array]:
    """The ``i``-th term of the Fourier sum, real and imaginary parts, as the sum takes it away."""
    j = 4 + ((i << 1) & 2)
    wk = imtmz2 + (NPI[i] + tmzre) * (NPI[i] - tmzre)
    f = 2.0 * TM * A[i] / (wk * wk + reimtmzm22)
    return f * (numer[j] * wk + numer[j + 1] * reimtmzm2), f * (
        numer[j + 1] * wk - numer[j] * reimtmzm2
    )


def _fourier_terms(zre: Array, zim: Array, znorm: Array) -> tuple[Array, Array]:
    """The Fourier sum's real and imaginary parts, before they are turned into ``w(z)``."""
    tmzre, tmzim = TM * zre, TM * zim
    numer = _numerators(tmzre, tmzim)
    reimtmzm2 = -2.0 * tmzre * tmzim
    shared = (tmzre, tmzim * tmzim, reimtmzm2, reimtmzm2 * reimtmzm2)
    sumre = (-A[0] / znorm) * (numer[0] * zre + numer[1] * zim)
    sumim = (-A[0] / znorm) * (numer[1] * zre - numer[0] * zim)
    for i in range(N):
        re, im = _fourier_term(i, numer, *shared)
        sumre, sumim = sumre - re, sumim - im
    return sumre, sumim


def _fourier(zre: Array, zim: Array, znorm: Array, negimz: Array) -> tuple[Array, Array]:
    """``w(z)`` from the Fourier sum, for ``Im z >= 0`` - flipped back if ``z`` was below."""
    sumre, sumim = _fourier_terms(zre, zim, znorm)
    ez2re, ez2im = _cexp(-(zre + zim) * (zre - zim), -2.0 * zre * zim)
    return (
        np.where(negimz, 2.0 * ez2re + sumim / TWOSQRTPI, -sumim / TWOSQRTPI),
        np.where(negimz, 2.0 * ez2im - sumre / TWOSQRTPI, sumre / TWOSQRTPI),
    )


def faddeeva_fast(z: Array) -> Array:
    """``RooHeterogeneousMath::faddeeva_fast``: ``w(z)``, to about 1e-13, for complex ``z``."""
    z = np.asarray(z, dtype=np.complex128)
    zre, zim = np.real(z).astype(np.float64), np.imag(z).astype(np.float64)
    with np.errstate(all="ignore"):
        near, tre, tim = _taylor(zre, zim)
        negimz = zim < 0.0
        fre, fim = np.where(negimz, -zre, zre), np.abs(zim)
        znorm = fre * fre + zim * zim
        cre, cim = _continued_fraction(fre, fim, negimz)
        sre, sim = _fourier(fre, fim, znorm, negimz)
    far = znorm > TM * TM
    re = np.where(near, tre, np.where(far, cre, sre))
    im = np.where(near, tim, np.where(far, cim, sim))
    found = _complex(re, im)
    return found if found.ndim else complex(found)


def _complex(re: Array, im: Array) -> Array:
    """The complex numbers of parts ``re`` and ``im``, put together without any arithmetic."""
    re, im = np.broadcast_arrays(np.asarray(re, dtype=np.float64), np.asarray(im, dtype=np.float64))
    found = np.empty(re.shape, dtype=np.complex128)
    found.real, found.imag = re, im
    return found


def _cdiv(a: Array, b: Array, c: Array, d: Array) -> tuple[Array, Array]:
    """``(a + ib) / (c + id)`` as libc++ divides: the divisor scaled by a power of two first."""
    biggest = np.maximum(np.abs(c), np.abs(d))
    exponent = np.frexp(biggest)[1] - 1
    scale = np.where(np.isfinite(biggest) & (biggest != 0), exponent, 0).astype(np.int64)
    c, d = np.ldexp(c, -scale), np.ldexp(d, -scale)
    denom = c * c + d * d
    return np.ldexp((a * c + b * d) / denom, -scale), np.ldexp((b * c - a * d) / denom, -scale)


def eval_cerf_approx(swt: Array, u: Array, c: Array) -> Array:
    """``evalCerfApprox``: ``erf(z) ~ exp(-z^2)/(sqrt(pi) z)``, which cancels ``exp(y^2)`` by hand.

    Every complex operation is spelt out as libc++ performs it - ``exp`` as
    ``exp(re) (cos, sin)``, division with its power-of-two scaling - so the
    result is ROOT's to the bit.
    """
    swt, u, c = (np.asarray(one, dtype=np.float64) for one in (swt, u, c))
    rootpi = math.sqrt(math.atan2(0.0, -1.0))
    zre, zim = swt * c, u + c
    zsqre, zsqim = (zre + zim) * (zre - zim), 2.0 * zre * zim
    evre, evim = _cexp(-zsqre - u * u, -zsqim)
    e2re, e2im = _cexp(zsqre, zsqim)
    mre, mim = _cdiv(-e2re, -e2im, (u + c) * rootpi, (-zre) * rootpi)
    tre = mre + 1.0
    found = _complex(2.0 * (evre * tre - evim * mim), 2.0 * (evre * mim + evim * tre))
    return found if found.ndim else complex(found)


def eval_cerf(swt: Array, u: Array, c: Array) -> Array:
    """``evalCerf``: ``exp(-u^2) w(swt c + i(u + c))``, with its numerically safe forms."""
    swt, u, c = np.broadcast_arrays(*(np.asarray(one, dtype=np.float64) for one in (swt, u, c)))
    z = u + c
    found = np.zeros(z.shape, dtype=np.complex128)
    safe = z > -4.0
    axis, general = safe & (swt == 0.0), safe & (swt != 0.0)
    with np.errstate(all="ignore"):
        if axis.any():
            found[axis] = np.exp(c[axis] * (c[axis] + 2.0 * u[axis])) * mf.erfc(z[axis])
        if general.any():
            w = np.atleast_1d(faddeeva_fast(_complex(swt[general] * c[general], z[general])))
            scale = np.exp(-u[general] * u[general])
            found[general] = _complex(scale * w.real, scale * w.imag)
        if not safe.all():
            found[~safe] = eval_cerf_approx(swt[~safe], u[~safe], c[~safe])
    return found if found.ndim else complex(found)
