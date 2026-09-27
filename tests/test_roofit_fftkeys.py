"""RooFFTConvPdf, RooKeysPdf and interpolated histogram densities, held to ROOT 6.40.04's numbers.

Every reference number below was printed by ROOT itself through PyROOT, the
same script run against ``xrdroot.pyroot``: densities evaluated, normalised
and integrated, events generated from a fixed seed, a fit, and the messages
RooFit prints about its caches. FFT results agree with FFTW's to rounding,
so they are compared to a relative ``1e-8``; everything else closer.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit import copies
from xrdroot.roofit.data.interpolate import Axis, clone_rows, neville, weights_interpolated
from xrdroot.roofit.pdfs.fftcache import buffered, scan_layout

#: RooKeysPdf of the eight weighted events below, per ``(mirror, rho)``: the value at 0, 0.37,
#: 2.55, 5, 9.99 and 10, at -1 and 12 through a function of a wider variable, normalised at 4,
#: and the integrals over the whole range, over [2.2, 6.1] and over [3.301, 3.305].
KEYS = {
    (0, 1.0): [
        0.0718313322717,
        0.0954199001529,
        0.100390986184,
        0.0729359134851,
        0.0966720785096,
        0.096494537952,
        0.00869450725402,
        0.059887248259,
        0.102134091204,
        0.807832595859,
        0.311931573504,
        0.000347944956316,
    ],
    (3, 1.0): [
        0.051653368794,
        0.051669845564,
        0.0343445917801,
        0.0217683485213,
        0.0665314328475,
        0.0665335892044,
        0.0516829801321,
        0.0661634073236,
        0.0763514703105,
        0.33336609654,
        0.0994576245611,
        0.000113023926134,
    ],
    (1, 0.5): [
        0.056285390795,
        0.065872044086,
        0.0538595714149,
        0.0359926065362,
        0.0837498121242,
        0.0833301279678,
        0.0555865643684,
        0.0,
        0.0810793124867,
        0.447497306,
        0.1556355059,
        0.000201795314001,
    ],
    (2, 2.0): [
        0.0282562735672,
        0.0301041690853,
        0.03581098354,
        0.0345807273584,
        0.067489594168,
        0.067490047826,
        0.0230161344678,
        0.0673993162287,
        0.0840069197232,
        0.417624609575,
        0.137807729777,
        0.000142762333056,
    ],
    (5, 1.0): [
        0.0363241063603,
        0.0452340649688,
        0.050646684682,
        0.03446047493,
        0.109201839969,
        0.109206702379,
        0.0116884779011,
        0.108529807395,
        0.0874247026274,
        0.460705367071,
        0.153584225473,
        0.000176198285427,
    ],
    (7, 1.0): [
        0.0765080327971,
        0.0772459114577,
        0.05019659468,
        0.0340714882336,
        0.0413594394671,
        0.0413034499674,
        0.0764460588539,
        0.0297842580865,
        0.0911350022344,
        0.432266625063,
        0.151128919964,
        0.000168908906579,
    ],
}
#: RooFFTConvPdf of a Landau and a Gaussian on 300 cache bins, per ``(strategy, fraction, order)``:
#: at -10, -2.1, 3.3, 7.05 and 29.9 the value and the value normalised, then the integral.
FFT = {
    (0, 0.1, 2): [
        4.94781151095,
        0.000266601301996,
        15.3025463502,
        0.0008245420772,
        1563.12506593,
        0.0842253543485,
        1948.41519114,
        0.104985815575,
        33.2048228317,
        0.0017891645589,
        18558.8422633,
    ],
    (1, 0.3, 1): [
        0.00193896292345,
        8.77387200457e-08,
        509.197030981,
        0.0230413357621,
        2528.25076234,
        0.114404191622,
        1054.93729184,
        0.0477362648845,
        52.9540332909,
        0.00239618769705,
        22099.2843574,
    ],
    (2, 0.25, 0): [
        0.00276720546073,
        1.30190592302e-07,
        324.83736816,
        0.0152828439965,
        2536.7893276,
        0.119350048196,
        1206.09768236,
        0.056744095756,
        38.223080735,
        0.00179830720599,
        21255.0339606,
    ],
    (0, 0.0, 3): [
        18.3880791961,
        0.00108966382068,
        13.9092400328,
        0.000824251161595,
        1421.03887063,
        0.084209700676,
        1771.27060893,
        0.104964171548,
        19.5840950543,
        0.00116053882594,
        16875.0020392,
    ],
}


def close(
    found: list[float], expected: list[float], rtol: float = 1e-8, atol: float = 1e-15
) -> None:
    assert np.allclose(found, expected, rtol=rtol, atol=atol), (found, expected)


@pytest.fixture
def quiet() -> Iterator[None]:
    """Keep the messages RooFit prints out of the way of tests about numbers."""
    service = ROOT.RooMsgService.instance()
    before = service.globalKillBelow()
    service.setGlobalKillBelow(5)
    yield
    service.setGlobalKillBelow(before)


# -- RooKeysPdf -------------------------------------------------------------------


def _weighted() -> tuple[object, object]:
    x = ROOT.RooRealVar("x", "x", 0, 10)
    w = ROOT.RooRealVar("w", "w", 0, 10)
    data = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(x, w), WeightVar="w")
    for value, weight in (
        (0.4, 1.0),
        (1.1, 2.0),
        (2.5, 1.0),
        (2.6, 0.5),
        (3.3, 1.0),
        (5.0, 1.5),
        (7.7, 1.0),
        (9.8, 3.0),
    ):
        x.setVal(value)
        data.add(ROOT.RooArgSet(x), weight)
    return x, data


@pytest.mark.parametrize("mode, rho", list(KEYS))
def test_a_kernel_estimate_is_rooFits_table_interpolated_and_integrated(
    quiet: None, mode: int, rho: float
) -> None:
    x, data = _weighted()
    z = ROOT.RooRealVar("z", "z", -2, 12)
    wider = ROOT.RooFormulaVar("u0", "@0", [z])
    x.setRange("part", 2.2, 6.1)
    x.setRange("tiny", 3.301, 3.305)
    keys = ROOT.RooKeysPdf("k", "k", x, data, mode, rho)
    found = []
    for value in (0.0, 0.37, 2.55, 5.0, 9.99, 10.0):
        x.setVal(value)
        found.append(keys.getVal())
    through = ROOT.RooKeysPdf("ku", "ku", wider, x, data, mode, rho)
    for value in (-1.0, 12.0):
        z.setVal(value)
        found.append(through.getVal())
    x.setVal(4.0)
    nx = ROOT.RooArgSet(x)
    found.append(keys.getVal(nx))
    for rng in (None, "part", "tiny"):
        found.append(
            keys.createIntegral(nx, Range=rng).getVal() if rng else keys.createIntegral(nx).getVal()
        )
    close(found, KEYS[(mode, rho)], rtol=1e-10)


def test_subtracted_reflections_subtract_nothing_as_in_root(quiet: None) -> None:
    x, data = _weighted()
    plain = ROOT.RooKeysPdf("a", "a", x, data, ROOT.RooKeysPdf.NoMirror)
    subtracted = ROOT.RooKeysPdf("b", "b", x, data, ROOT.RooKeysPdf.MirrorAsymBoth)
    xs = np.linspace(0, 10, 41)
    assert np.array_equal(plain.compute({"x": xs}), subtracted.compute({"x": xs}))
    assert subtracted.maxVal() == pytest.approx(
        float(np.max(plain.compute({"x": np.linspace(0, 10, 1001)})))
    )


def test_a_kernel_estimate_defaults_to_no_mirror_and_a_bandwidth_of_one(quiet: None) -> None:
    x, data = _weighted()
    x.setVal(0.37)
    assert ROOT.RooKeysPdf("k", "k", x, data).getVal() == pytest.approx(
        KEYS[(0, 1.0)][1], rel=1e-10
    )
    assert ROOT.RooKeysPdf.MirrorLeftAsymRight == 7 and ROOT.RooKeysPdf.MirrorAsymRight == 6


def test_a_mirror_mode_that_does_not_exist_is_refused(quiet: None) -> None:
    x, data = _weighted()
    with pytest.raises(ValueError, match="no mirror mode 9"):
        ROOT.RooKeysPdf("k", "k", x, data, 9)


# -- interpolated histograms ------------------------------------------------------

#: A 13-bin histogram of 500 Gaussian events (seed 1234): at -5, -4.9, -4.3, -1.1, 0, 0.2, 2.71,
#: 4.8 and 5, the density, the density normalised and the function - for orders 2 and 3.
HIST_1D = {
    2: [
        0,
        0,
        -0.25,
        0,
        0,
        -0.2331,
        0.75153,
        0.00150306,
        0.5781,
        103.229165,
        0.20645833,
        79.40705,
        106.6,
        0.2132,
        82,
        107.88102,
        0.21576204,
        82.9854,
        27.799577,
        0.055599154,
        21.38429,
        0,
        0,
        -0.3648,
        0,
        0,
        -0.5,
    ],
    3: [
        0,
        0,
        -0.25,
        0,
        0,
        -0.2331,
        0.97323135,
        0.0019464627,
        0.7486395,
        108.9820296,
        0.2179640593,
        83.8323305,
        106.6,
        0.2132,
        82,
        111.1375824,
        0.2222751648,
        85.490448,
        27.799577,
        0.055599154,
        21.38429,
        0,
        0,
        -0.3648,
        0,
        0,
        -0.5,
    ],
}
#: 800 events of two Gaussians in 13 by 7 bins: at five points, the density and it normalised.
HIST_2D = {
    1: [0, 0, 0, 0, 46.34175, 0.0579271875, 5.8695, 0.007336875, 0, 0],
    2: [0, 0, 0, 0, 47.69177112, 0.05961471389, 5.417279606, 0.006771599507, 0, 0],
}


def _gaussian_histograms() -> tuple[object, object, object, object]:
    x = ROOT.RooRealVar("x", "x", -5, 5)
    y = ROOT.RooRealVar("y", "y", 0, 4)
    m, s = ROOT.RooRealVar("m", "m", 0.3), ROOT.RooRealVar("s", "s", 1.4)
    gx, gy = ROOT.RooGaussian("gx", "gx", x, m, s), ROOT.RooGaussian("gy", "gy", y, m, s)
    ROOT.RooRandom.randomGenerator().SetSeed(1234)
    x.setBins(13)
    y.setBins(7)
    h1 = ROOT.RooDataHist("h1", "h1", {x}, gx.generate({x}, 500))
    h2 = ROOT.RooDataHist(
        "h2", "h2", {x, y}, ROOT.RooProdPdf("p", "p", [gx, gy]).generate({x, y}, 800)
    )
    return x, y, h1, h2


@pytest.mark.parametrize("order", [2, 3])
def test_a_histogram_is_interpolated_by_rooFits_polynomial_through_the_nearest_centres(
    quiet: None, order: int
) -> None:
    x, _, h1, _ = _gaussian_histograms()
    pdf = ROOT.RooHistPdf("p1", "p1", {x}, h1, order)
    func = ROOT.RooHistFunc("f1", "f1", {x}, h1, order)
    found = []
    for value in (-5.0, -4.9, -4.3, -1.1, 0.0, 0.2, 2.71, 4.8, 5.0):
        x.setVal(value)
        found += [pdf.getVal(), pdf.getVal(ROOT.RooArgSet(x)), func.getVal()]
    close(found, HIST_1D[order], rtol=1e-7)
    assert pdf.getInterpolationOrder() == order and pdf.dataHist() is h1


@pytest.mark.parametrize("order", [1, 2])
def test_a_two_dimensional_histogram_is_interpolated_along_each_variable_in_turn(
    quiet: None, order: int
) -> None:
    x, y, _, h2 = _gaussian_histograms()
    pdf = ROOT.RooHistPdf("p2", "p2", {x, y}, h2, order)
    found = []
    for vx, vy in ((-5.0, 0.0), (-4.9, 0.1), (0.3, 2.2), (1.7, 3.95), (4.99, 1.0)):
        x.setVal(vx)
        y.setVal(vy)
        found += [pdf.getVal(), pdf.getVal(ROOT.RooArgSet(x, y))]
    close(found, HIST_2D[order], rtol=1e-7)
    xs = np.array([0.3, 1.7])
    assert np.allclose(pdf.compute({"x": xs, "y": np.array([2.2, 3.95])}), [found[4], found[6]])


def _seven_bins() -> tuple[object, object]:
    x = ROOT.RooRealVar("x", "x", 0, 10)
    w = ROOT.RooRealVar("w", "w", 0, 10)
    x.setBins(7)
    filled = ROOT.RooDataSet("filled", "filled", ROOT.RooArgSet(x, w), WeightVar="w")
    for i, content in enumerate((3.0, 5.0, 4.0, 8.0, 2.0, 1.0, 6.0)):
        x.setVal((i + 0.5) * 10.0 / 7)
        filled.add(ROOT.RooArgSet(x), content)
    return x, ROOT.RooDataHist("h", "h", ROOT.RooArgSet(x), filled)


@pytest.mark.parametrize(
    "order, expected",
    [
        (1, [0.42, 2.52, 8.0, 5.20000000011, 1.7000000006]),
        (2, [0.0588000000506, 2.11680000006, 8.0, 5.53600000012, 2.00100000086]),
    ],
)
def test_a_cumulative_histogram_is_zero_below_its_range_and_one_above(
    quiet: None, order: int, expected: list[float]
) -> None:
    x, h = _seven_bins()
    func = ROOT.RooHistFunc("f", "f", ROOT.RooArgSet(x), h, order)
    func.setCdfBoundaries(True)
    found = []
    for value in (0.1, 0.6, 5.0, 9.4, 9.9):
        x.setVal(value)
        found.append(func.getVal())
    close(found, expected, rtol=1e-10)
    assert func.getCdfBoundaries()


def test_a_histogram_density_of_a_function_reads_the_histogram_at_the_functions_value(
    quiet: None,
) -> None:
    x, h = _seven_bins()
    y = ROOT.RooRealVar("y", "y", -1, 1)
    u = ROOT.RooFormulaVar("u", "5+5*@0", [y])
    pdf = ROOT.RooHistPdf("p", "p", [u], [x], h, 1)
    found = []
    for value in (-1.0, -0.5, 0.0, 0.93):
        y.setVal(value)
        found.append(pdf.getVal())
    close(found, [2.1, 3.325, 5.6, 4.2], rtol=1e-12)
    assert pdf.analytic_names(frozenset(["y"]), None) == frozenset()


def test_a_histogram_integrates_to_its_sum_of_weights_over_its_full_range_only(quiet: None) -> None:
    x, h = _seven_bins()
    pdf = ROOT.RooHistPdf("p", "p", ROOT.RooArgSet(x), h, 2)
    func = ROOT.RooHistFunc("f", "f", ROOT.RooArgSet(x), h, 0)
    x.setRange("low", 0, 5)
    assert pdf.createIntegral(ROOT.RooArgSet(x)).getVal() == pytest.approx(29.0)
    assert func.createIntegral(ROOT.RooArgSet(x)).getVal() == pytest.approx(29.0 * 10 / 7)
    assert pdf.analytic_names(frozenset(["x"]), "low") == frozenset()
    pdf.setInterpolationOrder(0)
    assert pdf.getInterpolationOrder() == 0


def test_interpolating_in_three_dimensions_is_refused_as_root_refuses_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    x, y, z = (ROOT.RooRealVar(n, n, 0, 1) for n in "xyz")
    for one in (x, y, z):
        one.setBins(2)
    h = ROOT.RooDataHist("h3", "h3", ROOT.RooArgSet(x, y, z))
    pdf = ROOT.RooHistPdf("p3", "p3", ROOT.RooArgSet(x, y, z), h, 1)
    assert (
        "RooDataHist::weight(h3) interpolation in 3 dimensions not yet implemented"
        in capsys.readouterr().out
    )
    assert pdf.getVal() == 0.0


def test_the_interpolation_helpers_follow_roots_arithmetic() -> None:
    xa = np.array([[0.0], [1.0], [3.0]])
    ya = xa**2 + 1
    assert neville(xa, ya, np.array([2.0]))[0] == pytest.approx(5.0)
    uneven = Axis([0.0, 1.0, 3.0, 6.0], False)
    assert list(uneven.numbers(np.array([-1.0, 1.0, 5.9, 7.0]))) == [0, 1, 2, 2]
    assert list(uneven.centres(np.array([0, 2]))) == [0.5, 4.5] and list(uneven.widths()) == [
        1.0,
        2.0,
        3.0,
    ]
    var = ROOT.RooRealVar("v", "v", 0, 6)
    var.setBinning(ROOT.RooBinning(0, 6))
    var.getBinning().addBoundary(2.5)
    rows = clone_rows(var, 1.0, 6.0)
    assert list(rows.edges) == [1.0, 2.5, 6.0] and not rows.uniform
    grid = np.arange(12.0).reshape(4, 3)
    axes = [Axis(np.linspace(0, 4, 5), True), Axis(np.linspace(0, 3, 4), True)]
    found = weights_interpolated(axes, grid, [np.array([1.5]), np.array([1.5])], 1)
    assert found[0] == pytest.approx(grid[1, 1])


# -- RooFFTConvPdf ----------------------------------------------------------------


def _landau_gauss(bins: int = 300) -> tuple[object, ...]:
    t = ROOT.RooRealVar("t", "t", -10, 30)
    ml = ROOT.RooRealVar("ml", "ml", 5.0, -20, 20)
    sl = ROOT.RooRealVar("sl", "sl", 1, 0.1, 10)
    landau = ROOT.RooLandau("lx", "lx", t, ml, sl)
    mg = ROOT.RooRealVar("mg", "mg", 0)
    sg = ROOT.RooRealVar("sg", "sg", 2, 0.1, 10)
    gauss = ROOT.RooGaussian("gauss", "gauss", t, mg, sg)
    t.setBins(bins, "cache")
    return t, landau, gauss, ml, sl, sg


@pytest.mark.parametrize("strategy, fraction, order", list(FFT))
def test_a_convolution_is_rooFits_fft_of_the_two_densities_sampled_on_the_cache_binning(
    quiet: None, strategy: int, fraction: float, order: int
) -> None:
    t, landau, gauss, *_ = _landau_gauss()
    lxg = ROOT.RooFFTConvPdf("lxg", "lxg", t, landau, gauss, order)
    lxg.setBufferStrategy(strategy)
    lxg.setBufferFraction(fraction)
    nt = ROOT.RooArgSet(t)
    found = []
    for value in (-10.0, -2.1, 3.3, 7.05, 29.9):
        t.setVal(value)
        found += [lxg.getVal(), lxg.getVal(nt)]
    found.append(lxg.createIntegral(nt).getVal())
    close(found, FFT[(strategy, fraction, order)], rtol=1e-8)
    assert (lxg.bufferStrategy(), lxg.bufferFraction(), lxg.getInterpolationOrder()) == (
        strategy,
        fraction,
        order,
    )


def test_a_convolution_prints_as_root_prints_it(
    quiet: None, capsys: pytest.CaptureFixture[str]
) -> None:
    t, landau, gauss, *_ = _landau_gauss()
    lxg = ROOT.RooFFTConvPdf("lxg", "lxg", t, landau, gauss, 3)
    lxg.setBufferFraction(0.0)
    lxg.setInterpolationOrder(3)
    t.setVal(29.9)
    lxg.getVal(ROOT.RooArgSet(t))
    lxg.Print()
    assert capsys.readouterr().out == "RooFFTConvPdf::lxg[ lx(t) (*) gauss(t) ] = 19.5841\n"


@pytest.mark.parametrize(
    "low, high, points, expected",
    [
        (
            2,
            12,
            (2.5, 6.0, 7.1, 11.5),
            [0.000116184938551, 0.243957552646, 0.35460576534, 9.89339769013e-05],
        ),
        (
            -12,
            -2,
            (-11.5, -7.1, -6.0, -2.5),
            [0.000118515465206, 0.356025999381, 0.234397019853, 9.65278461249e-05],
        ),
    ],
)
def test_a_range_without_zero_puts_zero_where_root_counts_it_from(
    quiet: None, low: float, high: float, points: tuple[float, ...], expected: list[float]
) -> None:
    s = ROOT.RooRealVar("s", "s", low, high)
    s.setBins(200, "cache")
    centre, offset = (6.0, 1.0) if low > 0 else (-6.0, -1.0)
    g1 = ROOT.RooGaussian("g1", "g1", s, ROOT.RooFit.RooConst(centre), ROOT.RooFit.RooConst(1.0))
    g2 = ROOT.RooGaussian("g2", "g2", s, ROOT.RooFit.RooConst(offset), ROOT.RooFit.RooConst(0.5))
    conv = ROOT.RooFFTConvPdf("c", "c", s, g1, g2)
    found = []
    for value in points:
        s.setVal(value)
        found.append(conv.getVal(ROOT.RooArgSet(s)))
    close(found, expected, rtol=1e-8)


def test_a_cache_over_a_second_observable_holds_a_convolution_for_each_of_its_bins(
    quiet: None,
) -> None:
    xx = ROOT.RooRealVar("xx", "xx", -10, 10)
    mean = ROOT.RooRealVar("mean", "mean", -3, 3)
    sigma = ROOT.RooRealVar("sigma", "sigma", 0.5, 0.1, 10)
    gx = ROOT.RooGaussian("gx", "gx", xx, mean, sigma)
    a = ROOT.RooRealVar("a", "a", 2, 1, 10)
    box = ROOT.RooGenericPdf("mm", "abs(mean)<a", [mean, a])
    xx.setBins(40, "cache")
    mean.setBins(50, "cache")
    model = ROOT.RooFFTConvPdf("model", "model", mean, gx, box)
    model.setCacheObservables(ROOT.RooArgSet(xx))
    model.setBufferFraction(1.0)
    assert [one.GetName() for one in model.cacheObservables()] == ["xx"]
    found = []
    for vx, vm in ((0.3, 0.1), (-4.2, 1.9)):
        xx.setVal(vx)
        mean.setVal(vm)
        found += [
            model.getVal(ROOT.RooArgSet(xx, mean)),
            model.getVal(ROOT.RooArgSet(mean)),
            model.getVal(),
        ]
    # The cache of no normalisation set does not watch ``xx`` - it holds only ``mean`` - and is not
    # refilled when ``xx`` moves: ROOT's value at (-4.2, 1.9) is the convolution at ``xx = 0.3``.
    close(
        found,
        [0.0408434422976, 0.245979489173, 1735.97607283, 2.19797215787e-16, 0.0, 1348.92391831],
        rtol=1e-8,
        atol=1e-13,
    )
    columns = {"xx": np.array([0.3, 0.3]), "mean": np.array([0.1, 0.1])}
    assert np.allclose(model.value(columns, frozenset(["xx", "mean"])), found[0])
    moving = {
        "xx": np.array([0.3, 0.3]),
        "mean": np.array([0.1, 0.1]),
        "sigma": np.array([0.5, 0.7]),
    }
    per_event = model.value(moving, frozenset(["xx", "mean"]))
    assert per_event[0] == pytest.approx(found[0]) and per_event[1] != pytest.approx(found[0])
    assert model.norm({}, frozenset()) == pytest.approx(model.getVal())


def _lines(text: str) -> list[str]:
    """Messages with the addresses of caches taken out, as the tutorial comparison takes them."""
    return [re.sub(r"0x[0-9a-f]+", "ADDR", line) for line in text.splitlines()]


def _cache(name: str, pdf: str, nset: str, code: int) -> str:
    return (
        f"[#1] INFO:Caching -- RooAbsCachedPdf::getCache({name}) creating new cache ADDR "
        f"with pdf {pdf} "
        f"for nset ({nset}) with code {code}"
    )


def _numeric(label: str, var: str) -> str:
    return (
        f"[#1] INFO:NumericIntegration -- RooRealIntegral::init({label}) using numeric integrator "
        f"RooIntegrator1D to calculate Int({var})"
    )


def test_events_are_the_sum_of_an_event_of_each_density_drawn_as_root_draws_them() -> None:
    t, landau, gauss, *_ = _landau_gauss()
    lxg = ROOT.RooFFTConvPdf("lxg", "lxg", t, landau, gauss)
    ROOT.RooRandom.randomGenerator().SetSeed(99)
    data = lxg.generate({t}, 6)
    found = [data.get(i).find("t").getVal() for i in range(6)]
    assert found == pytest.approx(
        [
            8.203651990381092,
            4.741151819512794,
            5.7075196052836,
            4.892948731798947,
            3.7792036225467864,
            4.207450802943649,
        ],
        rel=1e-12,
    )
    assert ROOT.RooRandom.randomGenerator().Rndm() == pytest.approx(0.6406014466192573, rel=1e-15)


def _angular() -> tuple[object, ...]:
    psi = ROOT.RooRealVar("psi", "psi", 0, 3.14159268)
    tpsi = ROOT.RooGenericPdf("Tpsi", "1+sin(2*@0)", [psi])
    gbias = ROOT.RooRealVar("gbias", "gbias", 0.2, 0.0, 1)
    greso = ROOT.RooRealVar("greso", "greso", 0.3, 0.1, 1.0)
    rpsi = ROOT.RooGaussian("Rpsi", "Rpsi", psi, gbias, greso)
    return psi, tpsi, rpsi, gbias, greso


def test_a_convolution_without_its_own_generator_is_sampled_by_two_copies_as_root_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    psi, tpsi, rpsi, *_ = _angular()
    psi.setBins(200, "cache")
    conv = ROOT.RooFFTConvPdf("Mf", "Mf", psi, tpsi, rpsi)
    conv.setBufferFraction(0)
    t, landau, gauss, *_ = _landau_gauss()
    ROOT.RooRandom.randomGenerator().SetSeed(99)
    ROOT.RooFFTConvPdf("lxg", "lxg", t, landau, gauss).generate(
        {t}, 6
    )  # the draws ROOT's run took first
    ROOT.RooRandom.randomGenerator().Rndm()
    data = conv.generate({psi}, 5)
    found = [data.get(i).find("psi").getVal() for i in range(5)]
    assert found == pytest.approx(
        [
            2.950842762691795,
            1.8335763541999395,
            0.8308589952680043,
            0.7631537211684974,
            0.5068772953411109,
        ],
        rel=1e-10,
    )
    assert ROOT.RooRandom.randomGenerator().Rndm() == pytest.approx(0.8876482909545302, rel=1e-15)
    warning = (
        "[#0] WARNING:Eval -- The FFT convolution 'Mf' will run with 200 bins. A decent "
        "accuracy for difficult convolutions is typically only reached with n >= 1000. "
        "Suggest to increase the number "
        "of bins of the observable 'psi'."
    )
    copy = [
        _numeric("Tpsi_Int[psi]", "psi"),
        warning,
        _cache("Mf", "Tpsi_CONV_Rpsi_CACHE_Obs[psi]_NORM_psi", "psi", 0),
    ]
    assert _lines(capsys.readouterr().out) == [
        "[#1] INFO:Eval -- RooRealVar::setRange(psi) new range named 'refrange_fft_Mf' "
        "created with bounds [0,3.14159]",
        *copy,
        *copy,
    ]


def test_a_fit_and_a_plot_each_make_a_copy_with_its_own_cache(
    capsys: pytest.CaptureFixture[str],
) -> None:
    t, landau, gauss, ml, sl, sg = _landau_gauss()
    lxg = ROOT.RooFFTConvPdf("lxg", "lxg", t, landau, gauss)
    ROOT.RooRandom.randomGenerator().SetSeed(7)
    data = lxg.generate({t}, 2000)
    result = lxg.fitTo(data, PrintLevel=-1, Save=True)
    assert [ml.getVal(), sl.getVal(), sg.getVal()] == pytest.approx(
        [4.979168736515985, 1.039070547049952, 1.8119074824446724], rel=1e-9
    )
    assert result.minNll() == pytest.approx(5642.687184916707, rel=1e-12)
    frame = t.frame()
    lxg.plotOn(frame)
    curve = frame.getObject(0)
    assert [float(curve.interpolate(v)) for v in (-5.0, 3.0, 8.0, 20.0)] == pytest.approx(
        [6.006933297581122e-07, 0.030577322779734045, 0.033973190203487616, 0.0024329101166529693],
        rel=1e-8,
    )
    lines = [line for line in _lines(capsys.readouterr().out) if "Caching" in line]
    assert lines == [_cache("lxg", "lx_CONV_gauss_CACHE_Obs[t]_NORM_t", "t", 0)] * 2


def test_a_convolution_of_a_function_of_the_observable_is_normalised_numerically(
    capsys: pytest.CaptureFixture[str],
) -> None:
    psi, tpsi, rpsi, gbias, greso = _angular()
    cpsi = ROOT.RooRealVar("cpsi", "cos(psi)", -1, 1)
    psif = ROOT.RooFormulaVar("psif", "acos(cpsi)", [cpsi])
    conv = ROOT.RooFFTConvPdf("Mf", "Mf", psif, psi, tpsi, rpsi)
    conv.setBufferFraction(0)
    nc = ROOT.RooArgSet(cpsi)
    found = []
    for value in (-0.95, -0.3, 0.2, 0.9):
        cpsi.setVal(value)
        found += [conv.getVal(nc), conv.getVal()]
    close(
        found,
        [
            0.127836317935,
            7345974.425,
            0.371819045894,
            21366175.4812,
            0.721770467669,
            41475751.8198,
            0.634020929221,
            36433320.407,
        ],
        rtol=1e-9,
    )
    assert conv.createIntegral(nc).getVal() == pytest.approx(57463908.0949, rel=1e-9)
    ROOT.RooRandom.randomGenerator().SetSeed(5)
    data = conv.generate({cpsi}, 300)
    capsys.readouterr()
    conv.fitTo(data, PrintLevel=-1)
    assert [gbias.getVal(), greso.getVal()] == pytest.approx([0.167728819, 0.2273490162], rel=1e-7)
    lines = _lines(capsys.readouterr().out)
    assert lines[:2] == [
        _numeric("Tpsi_Int[psi]", "psi"),
        _cache("Mf", "Tpsi_CONV_Rpsi_CACHE_Obs[cpsi]_NORM_cpsi", "cpsi", 0),
    ]
    assert lines[-3:] == [
        _numeric("Mf_Int[cpsi]", "cpsi"),
        _numeric("Tpsi_Int[psi]", "psi"),
        _cache("Mf", "Tpsi_CONV_Rpsi_CACHE_Obs[cpsi]", "", 1),
    ]
    assert conv.printMetaArgs() == "Tpsi(psi) (*) Rpsi(psi) "


def test_rooFits_complaints_about_binnings_and_buffers_are_its_own(
    capsys: pytest.CaptureFixture[str],
) -> None:
    u = ROOT.RooRealVar("u", "u", 0, 10)
    binning = ROOT.RooBinning(0, 10)
    binning.addBoundary(2.0)
    binning.addBoundary(7.0)
    u.setBinning(binning)
    gu = ROOT.RooGaussian("gu", "gu", u, ROOT.RooFit.RooConst(5.0), ROOT.RooFit.RooConst(1.0))
    hu = ROOT.RooGaussian("hu", "hu", u, ROOT.RooFit.RooConst(0.0), ROOT.RooFit.RooConst(1.0))
    conv = ROOT.RooFFTConvPdf("cu", "cu", u, gu, hu)
    conv.setBufferFraction(-1)
    assert capsys.readouterr().out.splitlines() == [
        "[#0] ERROR:Caching -- The internal binning of variable u is not uniform. "
        "The numerical FFT will likely "
        "yield wrong results.",
        "[#0] ERROR:InputArguments -- RooFFTConvPdf::setBufferFraction(cu) fraction "
        "should be greater than or "
        "equal to zero",
    ]
    assert conv.bufferFraction() == 0.1


def test_a_density_that_does_not_draw_the_observable_safely_is_sampled(quiet: None) -> None:
    from xrdroot.roofit.pdfs.fftgen import ConvolutionContext

    t, landau, gauss, *_ = _landau_gauss()
    shifted = ROOT.RooFormulaVar("m2", "0.1*@0", [t])
    unsafe = ROOT.RooGaussian("unsafe", "unsafe", t, shifted, ROOT.RooFit.RooConst(1.0))
    generic = ROOT.RooGenericPdf("generic", "1+0*@0", [t])
    assert isinstance(
        ROOT.RooFFTConvPdf("a", "a", t, landau, gauss).gen_context(frozenset(["t"])),
        ConvolutionContext,
    )
    for other in (unsafe, generic):
        conv = ROOT.RooFFTConvPdf("b", "b", t, landau, other)
        assert conv._draws(other) is False
    assert copies.state_of(landau) is None


def test_the_buffer_bookkeeping_is_scanPdfs() -> None:
    assert scan_layout(100, -10.0, 30.0, 0.1, 10.0) == (5, 110, 30 + 25)
    assert scan_layout(10, 2.0, 12.0, 0.2, 0.0, extend=False) == (1, 12, 10)
    values = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert buffered(values, 2, 1).tolist() == [[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0]]
    assert buffered(values, 1, 2).tolist() == [[1.0, 1.0, 2.0, 3.0, 4.0, 4.0]]
    assert buffered(values, 1, 0) is values


def test_a_kernel_estimate_integrates_within_one_bin_of_its_table(quiet: None) -> None:
    x, data = _weighted()
    x.setRange("one", 3.305, 3.31)
    keys = ROOT.RooKeysPdf("k", "k", x, data)
    assert keys.createIntegral(ROOT.RooArgSet(x), Range="one").getVal() == pytest.approx(
        0.000434711036772, rel=1e-10
    )
    wide = ROOT.RooRealVar("wide", "wide", 0, 20)
    wide.setRange("far", 11, 12)
    beyond = ROOT.RooKeysPdf("kw", "kw", wide, x, data)
    assert beyond.createIntegral(ROOT.RooArgSet(wide), Range="far").getVal() == 0.0


def test_a_density_of_none_of_the_caches_observables_has_unit_normalisation(quiet: None) -> None:
    t, landau, *_ = _landau_gauss()
    y = ROOT.RooRealVar("y", "y", 0, 1)
    flat = ROOT.RooUniform("flat", "flat", ROOT.RooArgSet(y))
    conv = ROOT.RooFFTConvPdf("c", "c", t, landau, flat)
    nt = ROOT.RooArgSet(t)
    t.setVal(5.0)
    assert conv.getVal(nt) == pytest.approx(conv.getVal() / conv.norm({}, frozenset(["t"])))
    assert conv.norm({}, frozenset(["t"])) == pytest.approx(conv.createIntegral(nt).getVal())


def test_the_copy_in_use_is_the_innermost_that_has_the_node() -> None:
    node = object()
    with copies.within({id(node): "outer"}), copies.within({}):
        assert copies.state_of(node) == "outer"
    assert copies.state_of(node) is None
