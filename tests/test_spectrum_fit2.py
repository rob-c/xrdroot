"""``TSpectrum2Fit``: 2-D peaks and their ridges fitted as ROOT 6.40 fits them.

Two peaks of sigma 1.8 on a sloping, rippled floor, with a ridge in x and
one in y, fitted through ROOT's own calls - ``SetPeakParameters`` with its
fourteen arrays, getters filling arrays and cells - and every number held
to what ROOT printed (:mod:`spectrumfit2refs`).
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from refmachine import roots
from spectrumfit2refs import CASES2

#: Channels in x and in y.
N = 24


def g(v: float, centre: float, sigma: float) -> float:
    p = (v - centre) / sigma
    return math.exp(-((p * p) / 2))


def spectrum() -> list[np.ndarray]:
    """The spectrum ROOT fitted, as rows of a ``Double_t **``, added up as the macro added it."""
    rows = []
    for x in range(N):
        row = np.zeros(N)
        for y in range(N):
            v = 2.0 + 0.05 * x
            v = v + 0.03 * y
            v = v + 0.5 * ((7 * x + 3 * y) % 5)
            v = v + (100.0 * g(x, 7.3, 1.8)) * g(y, 8.6, 1.8)
            v = v + (60.0 * g(x, 16.2, 1.8)) * g(y, 15.1, 1.8)
            v = v + 5.0 * g(x, 11.5, 1.8)
            row[y] = v + 3.0 * g(y, 5.2, 1.8)
        rows.append(row)
    return rows


def fitter(mode: int, ro: bool, background: bool, tails: bool):
    """A fitter set up as the case says.

    Mode 1 frees the first ridges, 2 fixes the positions and sigmas, 3 frees
    every ridge, 4 fixes everything, 5 frees the tails' slopes alone, and 6
    fixes the correlation at 1.
    """
    fix_peaks, fix_ridges = mode in (2, 4), mode not in (1, 3)
    positions = ([7.0, 16.0], [9.0, 15.0], [11.0, 12.0], [5.0, 20.0])
    amplitudes = ([90.0, 50.0], [4.0, 0.0], [2.5, 0.0])
    fixed = [fix_peaks, fix_peaks]
    ridges = [fix_ridges, mode != 3]
    amps = [mode == 4, mode == 4]
    f = ROOT.TSpectrum2Fit(2)
    peaks = (positions[0], fixed, positions[1], fixed, positions[2], ridges, positions[3], ridges,
             amplitudes[0], amps, amplitudes[1], ridges, amplitudes[2], ridges)  # fmt: skip
    return f, peaks, (fix_peaks, not ro)


def printed(f, source) -> list[list[float]]:
    """What the macro printed: each peak's row, the shared row, chi and three channels."""
    arrays = [np.zeros(2) for _ in range(16)]
    f.GetPositions(*arrays[0:4])
    f.GetPositionErrors(*arrays[4:8])
    f.GetAmplitudes(*arrays[8:11])
    f.GetAmplitudeErrors(*arrays[11:14])
    f.GetVolumes(arrays[14])
    f.GetVolumeErrors(arrays[15])
    order = [(0, 4), (1, 5), (2, 6), (3, 7), (8, 11), (9, 12), (10, 13), (14, 15)]
    rows = [[float(arrays[n][k]) for pair in order for n in pair] for k in (0, 1)]
    cells = [ctypes.c_double() for _ in range(28)]
    f.GetSigmaX(*cells[0:2])
    f.GetSigmaY(*cells[2:4])
    f.GetRo(*cells[4:6])
    f.GetBackgroundParameters(*cells[6:12])
    f.GetTailParameters(*cells[12:28])
    rows.append([c.value for c in cells])
    return [*rows, [f.GetChi()], [source[7][9], source[16][15], source[0][23]]]


def fitted(stiefel, statistic, optim, power, taylor, mode, ro, background, tails):
    f, peaks, (fix_sigmas, fix_ro) = fitter(mode, ro, background, tails)
    f.SetFitParameters(0, N - 1, 0, N - 1, 10, 0.5, statistic, optim, power, taylor)
    f.SetPeakParameters(2.0, fix_sigmas, 2.0, fix_sigmas, 1.0 if mode == 6 else 0.1, fix_ro, *peaks)
    if background:
        f.SetBackgroundParameters(1.0, False, 0.01, False, 0.01, False)
    if tails:
        f.SetTailParameters(0.1, False, 0.1, False, 0.1, False, 1.2, False, 1.3, False, 0.05, False,
                            0.02, False, 0.03, False)  # fmt: skip
    if mode == 5:
        f.SetTailParameters(0.0, True, 0.0, True, 0.0, True, 1.2, False, 1.3, False, 0.0, True,
                            0.0, True, 0.0, True)  # fmt: skip
    source = spectrum()
    (f.FitStiefel if stiefel else f.FitAwmi)(source)
    return printed(f, source)


@pytest.mark.parametrize("label", list(CASES2))
def test_each_way_of_fitting_peaks_and_ridges_finds_what_root_finds(label):
    arguments, tolerance, expected = CASES2[label]
    got = fitted(*arguments)
    # Where ROOT read memory it never wrote, what it printed is no number to hold to.
    expected = [[h if e is None else e for h, e in zip(row, want, strict=False)]
                for row, want in zip(got, expected, strict=False)]  # fmt: skip
    # A correlation of 1 leaves ROOT's errors NaN, and NaN is no number to compare.
    got, expected = [marked(row) for row in got], [marked(row) for row in expected]
    assert got == [roots(row, rel=tolerance) for row in expected]


def marked(row: list[float]) -> list[float]:
    """``row`` with each NaN made -1, which no error or position here is."""
    return [-1.0 if math.isnan(v) else v for v in row]
