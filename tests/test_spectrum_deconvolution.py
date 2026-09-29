"""``TSpectrum``'s Markov smoothing, Gold and Richardson-Lucy deconvolution and unfolding.

The references are ROOT 6.40.04's (``tests/data/spectrum-6.40.txt``). What
is sums, products and quotients alone - Gold's and Richardson-Lucy's
iterations and unfolding without a boost - is ROOT's to the bit on every
machine. The Markov chain's weights are ``exp`` and a boost is ``pow``, the C
library's: on ROOT's machine those are ROOT's to the bit too, and elsewhere
a weight may round the other way in the last place. A smoothed channel is a
running product of 64 such ratios, and a boosted answer twenty iterations
of quotients of them, which carry a last-place difference forward without
growing it much: ``1e-10`` relative holds either.
"""

from __future__ import annotations

import numpy as np
import pytest

from refmachine import roots
from spectrumcases import reference, y64
from xrdroot.pyroot import TSpectrum
from xrdroot.spectrum.gold import Response, response_of
from xrdroot.spectrum.unfolding import unfold

#: The response the deconvolutions sharpen by: a Lorentzian of five channels peaked at 2.
RESPONSE = np.array([100 / (1 + (i - 2) * (i - 2) / 1.0) if i < 6 else 0.0 for i in range(64)])


def _close(tag: str) -> object:
    return roots(reference()[tag], rel=1e-10)


@pytest.mark.parametrize("window", [1, 4, 7])
def test_markov_smoothing_is_roots_for_every_window(window):
    spectrum = y64()
    assert TSpectrum().SmoothMarkov(spectrum, 64, window) is None
    assert spectrum.tolist() == _close(f"mk {window}")


def test_markov_smoothing_refuses_a_window_that_is_not_positive():
    spectrum = y64()
    assert TSpectrum().SmoothMarkov(spectrum, 64, 0) == "Averaging Window must be positive"
    assert spectrum.tolist() == y64().tolist()


def test_markov_smoothing_leaves_a_spectrum_of_nothing_as_it_is():
    spectrum = np.zeros(8)
    assert TSpectrum().SmoothMarkov(spectrum, 8, 3) is None
    assert spectrum.tolist() == [0.0] * 8


def test_gold_deconvolution_is_roots_to_the_bit():
    spectrum = y64()
    assert TSpectrum().Deconvolution(spectrum, RESPONSE, 64, 20, 1, 1.5) is None
    assert spectrum.tolist() == reference()["gd 1"]


def test_boosted_gold_deconvolution_is_roots():
    spectrum = y64()
    TSpectrum().Deconvolution(spectrum, RESPONSE, 64, 20, 2, 1.5)
    assert spectrum.tolist() == _close("gd 2")


@pytest.mark.parametrize("repetitions", [1, 2])
def test_richardson_lucy_is_roots_wherever_root_solves(repetitions):
    """ROOT's last ``length - 1`` channels, never solved for, hold whatever its working memory
    held - here the Gold deconvolution's leftovers - and are zero here, shifted to the ends."""
    spectrum = y64()
    assert TSpectrum().DeconvolutionRL(spectrum, RESPONSE, 64, 20, repetitions, 1.5) is None
    solved = reference()[f"rl {repetitions}"][2:61]
    assert spectrum[2:61].tolist() == (solved if repetitions == 1 else roots(solved, rel=1e-10))
    assert spectrum[[0, 1, 61, 62, 63]].tolist() == [0.0] * 5


@pytest.mark.parametrize("method", ["Deconvolution", "DeconvolutionRL"])
def test_a_deconvolution_refuses_what_root_refuses(method):
    s = TSpectrum()
    spectrum = y64()
    assert getattr(s, method)(spectrum, RESPONSE, 0, 20, 1, 1.0) == "Wrong Parameters"
    assert getattr(s, method)(spectrum, RESPONSE, 64, 20, 0, 1.0) == "Wrong Parameters"
    assert getattr(s, method)(spectrum, np.zeros(64), 64, 20, 1, 1.0) == "ZERO RESPONSE VECTOR"
    assert spectrum.tolist() == y64().tolist()


def test_a_response_is_read_to_its_last_non_zero_channel_and_its_first_highest():
    found = response_of(np.array([0.0, 2.0, 5.0, 5.0, 1.0, 0.0]))
    assert isinstance(found, Response)
    assert (found.length, found.area, found.position) == (5, 13.0, 2)
    assert Response(np.array([-1.0, 0.0])).position == 0


def _matrix() -> list[list[float]]:
    return [[1 / (1 + (i - 1.5 * j) * (i - 1.5 * j)) + ((i + j) % 3) * 0.01 for i in range(12)]
            for j in range(8)]  # fmt: skip


def test_unfolding_through_a_response_matrix_is_roots():
    spectrum = y64(50)[5::3][:12].copy()
    assert TSpectrum().Unfolding(spectrum, _matrix(), 12, 8, 15, 1, 1.5) is None
    assert spectrum.tolist() == reference()["uf 1"]
    spectrum = y64(50)[5::3][:12].copy()
    TSpectrum().Unfolding(spectrum, _matrix(), 12, 8, 15, 2, 1.5)
    assert spectrum.tolist() == _close("uf 2")


def test_unfolding_refuses_what_root_refuses():
    s, spectrum = TSpectrum(), np.ones(12)
    assert s.Unfolding(spectrum, _matrix(), 0, 8, 15, 1, 1.0) == "Wrong Parameters"
    assert s.Unfolding(spectrum, _matrix(), 4, 8, 15, 1, 1.0) == "Sizex must be greater than sizey)"
    assert (
        s.Unfolding(spectrum, _matrix(), 12, 8, 0, 1, 1.0)
        == "Number of iterations must be positive"
    )
    zero = _matrix()
    zero[3] = [0.0] * 12
    assert s.Unfolding(spectrum, zero, 12, 8, 15, 1, 1.0) == "ZERO COLUMN IN RESPONSE MATRIX"
    assert spectrum.tolist() == [1.0] * 12
    assert unfold(np.ones(2), np.zeros((0, 2)), 1, 1, 1.0) == "Wrong Parameters"
