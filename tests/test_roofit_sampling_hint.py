"""Where a curve of a binned function is sampled: ROOT's ``plotSamplingHint``.

A ``RooHistFunc`` is constant inside a bin and jumps at its edges, and
``RooCurve``'s recursive straight-line test cannot find a jump it was never
seeded next to - so the function tells the curve where to sample, just either
side of every bin edge and of the plot's own limits
(``RooCurve::plotSamplingHintForBinBoundaries``, ``delta = (xhi - xlo) * 1e-9``),
and every class above it merges its children's hints
(``RooRealSumPdf::plotSamplingHint``, ``RooProdPdf``, ``RooProduct``). ROOT's own
numbers, measured on the 6.40 oracle for ``testHistFactoryPlotting``'s model:
35 points, the first at 2896.4482758620688, the second at 2896.5517241379312.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.functions import RooProduct
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.histpdf import SAMPLE_EPSILON, RooHistFunc, RooHistPdf
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.realsum import RooRealSumPdf
from xrdroot.roofit.plot.curve import sample
from xrdroot.roofit.plot.curves import _sampling_hint
from xrdroot.roofit.real import merge_hints
from xrdroot.roofit.variables import RooRealVar

#: The plot's limits in the tests below.
LOW, HIGH = 0.5, 2.5


def histogram() -> Any:
    """``x`` on ``[0, 3]`` cut at 1 and 2, and its contents: three bins of width one."""
    from xrdroot.pyroot.core import TH1D

    h = TH1D("h", "", 3, np.array([0.0, 1.0, 2.0, 3.0]))
    h.SetDirectory(0)
    for bin_number, content in ((1, 2.0), (2, 5.0), (3, 1.0)):
        h.SetBinContent(bin_number, content)
    return h


def binned(name: str = "hf") -> RooHistFunc:
    x = RooRealVar("x", "x", 0, 3)
    return RooHistFunc(name, "", [x], RooDataHist("dh", "", [x], histogram()))


def a_density() -> RooHistPdf:
    x = RooRealVar("x", "x", 0, 3)
    return RooHistPdf("p", "", [x], RooDataHist("dh2", "", [x], histogram()))


def delta(low: float = LOW, high: float = HIGH) -> float:
    return (high - low) * SAMPLE_EPSILON


def test_a_step_asks_to_be_sampled_either_side_of_every_bin_edge_in_the_range() -> None:
    hint = binned().plot_sampling_hint("x", LOW, HIGH)
    step = delta()
    assert hint == pytest.approx([LOW + step, 1 - step, 1 + step, 2 - step, 2 + step, HIGH - step])
    assert hint == sorted(hint)


def test_an_edge_at_the_plots_limits_is_left_to_the_limits_themselves() -> None:
    """Edges at, beyond, or within ``delta`` of the limits add nothing: they are not interior."""
    hint = binned().plot_sampling_hint("x", 0.0, 3.0)
    step = delta(0.0, 3.0)
    assert hint == pytest.approx([step, 1 - step, 1 + step, 2 - step, 2 + step, 3 - step])


def test_an_edge_within_delta_of_the_limit_is_not_interior_either() -> None:
    """``xhi - x > delta`` is asked with the plot's own ``delta``, so a limit a hair past an
    edge leaves that edge to the limit - the curve starts and ends there anyway."""
    high = 1.0 + 1e-12
    hint = binned().plot_sampling_hint("x", 0.0, high)
    assert len(hint) == 2
    assert hint == pytest.approx([high * SAMPLE_EPSILON, high * (1 - SAMPLE_EPSILON)])


def test_an_interpolated_bin_asks_for_no_hint_and_is_sampled_uniformly() -> None:
    """``RooHistPdf::plotSamplingHint``: a smooth interpolation has no jump to catch."""
    func = binned()
    func.setInterpolationOrder(1)
    assert func.plot_sampling_hint("x", LOW, HIGH) is None


def test_a_variable_the_function_does_not_have_asks_for_no_hint() -> None:
    assert binned().plot_sampling_hint("y", LOW, HIGH) is None


def test_a_density_asks_where_its_function_does() -> None:
    assert a_density().plot_sampling_hint("x", LOW, HIGH) == binned().plot_sampling_hint(
        "x", LOW, HIGH
    )


def test_a_hint_the_children_share_is_asked_for_once() -> None:
    """``RooRealSumPdf::plotSamplingHint`` sorts the merge and removes consecutive duplicates."""
    assert merge_hints([[2.0, 1.0], [1.0, 3.0]]) == [1.0, 2.0, 3.0]
    assert merge_hints([None, []]) is None


def test_a_sum_a_product_and_a_product_of_densities_ask_their_children_s_hints() -> None:
    func = binned()
    x = func.observables[0]
    own = func.plot_sampling_hint("x", LOW, HIGH)
    value = RooRealVar("v", "v", 1.0)
    assert RooProduct("r", "", [func, value]).plot_sampling_hint("x", LOW, HIGH) == own
    assert RooRealSumPdf("s", "", [func], []).plot_sampling_hint("x", LOW, HIGH) == own
    gauss = RooGaussian("g", "", x, RooRealVar("m", "m", 1.5), RooRealVar("s", "s", 0.5))
    assert RooProdPdf("pd", "", [gauss, func]).plot_sampling_hint("x", LOW, HIGH) == own


def test_a_function_that_asks_nothing_is_sampled_on_the_frames_own_bins() -> None:
    assert _sampling_hint(RooRealVar("v", "v", 1.0), "x", LOW, HIGH) is None


def test_a_step_curve_is_drawn_at_the_hint_and_not_at_the_regular_grid() -> None:
    """The hint is the whole starting grid, as ``RooCurve::addPoints`` takes it: the curve
    starts at the frame's low limit **plus** ``delta``, never on a bin edge itself."""
    func = binned()
    name = func.observables[0].GetName()
    hint = func.plot_sampling_hint("x", 0.0, 3.0)
    xs, _ = sample(lambda one: func.compute({name: one}), 0.0, 3.0, 40, hint=hint)
    assert xs.min() == pytest.approx(0.0 - (3.0 / (len(hint) - 1)) * 1.001)
    inside = xs[(xs > 0.0) & (xs < 3.0)]
    on_an_edge = np.isclose(inside[:, None], np.array([1.0, 2.0]), rtol=0, atol=1e-12)
    assert not on_an_edge.any()
