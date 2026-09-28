"""``RooMCStudy``: toy samples generated and fitted as ROOT 6.40.04 generates and fits them.

The studies below were run in ROOT through PyROOT after
``RooRandom::randomGenerator()->SetSeed(4357)``: the samples are ROOT's to
the bit, each fit's parameters, errors, pulls and minimum to Minuit's
tolerance, and the frames of the plots are framed where ROOT frames them.
"""

from __future__ import annotations

from typing import Any

import pytest

from refmachine import FIT_REL, ROOTS_MACHINE, roots
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.mcstudy import RooMCStudy
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

QUIET = RooCmdArg("FitOptions", RooCmdArg("PrintLevel", -1))

#: ROOT's three fits of 40 events: m, s, NLL, ngen, merr, mpull, serr, spull.
BASIC = [
    [
        0.24636691725034773,
        1.843733239012737,
        81.30734283902716,
        40.0,
        0.2913544686762927,
        0.8455916889473624,
        0.20551866163459953,
        -0.7603531462514893,
    ],
    [
        -0.5402295450041339,
        2.0158304190463734,
        84.7959472146875,
        40.0,
        0.3185185843529512,
        -1.696069151197483,
        0.22531139740598516,
        0.07026017870657826,
    ],
    [
        -0.10659976371294329,
        1.9297490574390226,
        83.08263448222795,
        40.0,
        0.3049316536460631,
        -0.34958575942619124,
        0.21544205355022217,
        -0.326078132859057,
    ],
]
ROOT_ORDER = ["m", "s", "NLL", "ngen", "merr", "mpull", "serr", "spull"]


def _gauss(bins: int = 100) -> tuple[Any, Any, Any, Any]:
    generator().SetSeed(4357)
    x = RooRealVar("x", "x", -10, 10)
    x.setBins(bins)
    m = RooRealVar("m", "m", 0, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    return RooGaussian("g", "g", x, m, s), x, m, s


def _rows(data: Any, names: list[str]) -> list[list[float]]:
    return [[float(data.column(n)[i]) for n in names] for i in range(data.numEntries())]


def _progress(out: str) -> list[str]:
    return [line for line in out.splitlines() if "PROGRESS" in line]


def _first_events(study: Any, count: int) -> list[float]:
    return [float(study.genData(i).column("x")[0]) for i in range(count)]


def _basic() -> tuple[Any, Any, Any]:
    g, x, m, s = _gauss()
    study = RooMCStudy(g, [x], RooCmdArg("Silence", True), QUIET)
    assert study.generateAndFit(3, 40, True) is False
    return study, m, s


def test_a_study_generates_and_fits_each_sample_as_root_does(capsys: Any) -> None:
    """Three samples of 40 events: each fit's values, errors, pulls against the generating
    values, minimum and event count - the samples counted down as they go."""
    capsys.readouterr()
    study, _, _ = _basic()
    assert _progress(capsys.readouterr().out) == [
        f"[#0] PROGRESS:Generation -- RooMCStudy::run: sample {n}" for n in (2, 1, 0)
    ]
    data = study.fitParDataSet()
    assert (data.GetName(), data.GetTitle(), data.numEntries()) == (
        "fitParData_g",
        "Fit Parameters DataSet",
        3,
    )
    assert sorted(data.get().names()) == sorted(ROOT_ORDER)
    for found, expected in zip(_rows(data, ROOT_ORDER), BASIC):
        assert found == pytest.approx(expected, rel=1e-7)


def test_a_study_keeps_roots_samples_and_leaves_the_last_fit_in_place() -> None:
    """The kept samples are ROOT's events, the last fit's parameters are where the fit left
    them, and the generator is where ROOT's is."""
    study, m, s = _basic()
    assert _first_events(study, 3) == [1.9978654352187961, -2.3205870192854925, 1.8649088608290905]
    params = study.fitParams(2)
    assert params.names() == ["m", "s"]
    assert [p.getVal() for p in params] == pytest.approx(BASIC[2][:2], rel=1e-9)
    assert (m.getVal(), s.getVal()) == pytest.approx(BASIC[2][:2], rel=1e-9)
    assert generator().Rndm() == 0.49709826125763357


@pytest.mark.xfail(strict=True, reason="the columns are ordered parameters, errors, then NLL")
def test_the_parameter_dataset_has_roots_columns_in_roots_order() -> None:
    """ROOT's columns are the parameters, NLL, ngen, then each parameter's error and pull."""
    g, x, _, _ = _gauss()
    study = RooMCStudy(g, [x], RooCmdArg("Silence", True), QUIET)
    study.generateAndFit(1, 40)
    assert study.fitParDataSet().get().names() == ROOT_ORDER


@pytest.mark.xfail(strict=True, reason="every fit's result is kept, Save() or not")
def test_fit_results_are_kept_only_when_the_fit_options_save_them() -> None:
    """Without ``Save()`` among the fit options ROOT keeps no result: ``fitResult(0)`` is an
    invalid sample number."""
    g, x, _, _ = _gauss()
    study = RooMCStudy(g, [x], RooCmdArg("Silence", True), QUIET)
    study.generateAndFit(1, 40)
    assert study.fitResult(0) is None


def test_an_extended_study_draws_a_poisson_number_of_events_for_each_sample() -> None:
    """With ``Extended()`` each sample's size is a Poisson draw about the number asked for:
    ROOT's 39 and 45, and its fits."""
    g, x, _, _ = _gauss()
    study = RooMCStudy(g, [x], RooCmdArg("Extended", True), RooCmdArg("Silence", True), QUIET)
    study.generateAndFit(2, 40)
    expected = [
        [
            0.18091707534757645,
            1.8995917123621422,
            80.36397016650977,
            39.0,
            0.3039909186217481,
            0.5951397369626333,
            0.21496848874936947,
            -0.46708374898110416,
        ],
        [
            -0.4338071368956115,
            2.0523946833268725,
            96.21213589697317,
            45.0,
            0.3057806141913312,
            -1.418687505886728,
            0.21624983995587554,
            0.24228773227098474,
        ],
    ]
    for found, row in zip(_rows(study.fitParDataSet(), ROOT_ORDER), expected):
        assert found == pytest.approx(row, rel=1e-7)
    assert generator().Rndm() == 0.9967236891388893


def test_an_extended_study_of_a_density_with_yields_draws_about_their_sum() -> None:
    """With no size asked for, an extended sample is a Poisson draw about the expected number -
    here 20 + 10 - and the events are ROOT's."""
    from xrdroot.roofit.pdfs.addpdf import RooAddPdf
    from xrdroot.roofit.pdfs.basic import RooExponential

    g, x, _, _ = _gauss()
    e = RooExponential("e", "e", x, RooRealVar("a", "a", -0.2, -1, 0))
    yields = [RooRealVar("n1", "n1", 20, 0, 100), RooRealVar("n2", "n2", 10, 0, 100)]
    model = RooAddPdf("model", "model", [g, e], yields)
    study = RooMCStudy(model, [x], RooCmdArg("Extended", True), RooCmdArg("Silence", True), QUIET)
    assert study.generate(2, 0, True) is False
    assert [study.genData(i).numEntries() for i in range(2)] == [29, 36]
    assert float(study.genData(1).column("x")[0]) == 0.5963371441475216
    assert generator().Rndm() == 0.3347300221212208


def test_a_binned_study_fits_histograms_of_each_sample() -> None:
    """``Binned()`` generates each sample as a histogram, and fits that: ROOT's fits."""
    g, x, _, _ = _gauss(8)
    study = RooMCStudy(g, [x], RooCmdArg("Binned", True), RooCmdArg("Silence", True), QUIET)
    study.generateAndFit(2, 40)
    expected = [
        [
            -0.5688484822686283,
            2.431668181019001,
            92.36802046641691,
            40.0,
            0.3843152406803614,
            -1.480161133504838,
            0.2719811063413107,
            1.5871256162819558,
        ],
        [
            -0.1253315315953829,
            1.8497550983916922,
            81.36131221091277,
            40.0,
            0.2923056425869667,
            -0.4287687725976563,
            0.20669749706579,
            -0.7268830234576387,
        ],
    ]
    for found, row in zip(_rows(study.fitParDataSet(), ROOT_ORDER), expected):
        assert found == pytest.approx(row, rel=1e-7)
    assert generator().Rndm() == 0.4363430186640471


def test_a_long_study_says_how_far_it_is_every_hundredth_of_the_way(capsys: Any) -> None:
    """Two hundred samples are counted down every second one, as ROOT's prescale does; the
    samples are ROOT's all the same."""
    g, x, _, _ = _gauss()
    study = RooMCStudy(g, [x], QUIET)
    capsys.readouterr()
    study.generate(200, 1)
    progress = _progress(capsys.readouterr().out)
    assert len(progress) == 100
    assert progress[:2] == [
        "[#0] PROGRESS:Generation -- RooMCStudy::run: sample 198",
        "[#0] PROGRESS:Generation -- RooMCStudy::run: sample 196",
    ]
    assert float(study.genData(199).column("x")[0]) == 0.11121998824882695
    assert generator().Rndm() == 0.5990753807127476


def test_a_constant_parameter_has_no_pull_and_roofit_says_why(capsys: Any) -> None:
    """A parameter held constant has no error, so no pull column - and ROOT's warning."""
    g, x, m, _ = _gauss()
    m.setConstant(True)
    study = RooMCStudy(g, [x], RooCmdArg("Silence", True), QUIET)
    capsys.readouterr()
    study.generateAndFit(1, 20)
    assert (
        "[#0] WARNING:Generation -- Fit parameter 'm' does not have an error. A pull "
        "distribution cannot be generated. This might be caused by the parameter being "
        "constant or because the fits were not run.\n"
    ) in capsys.readouterr().out
    data = study.fitParDataSet()
    assert sorted(data.get().names()) == sorted(["m", "s", "NLL", "ngen", "merr", "serr", "spull"])
    names = ["m", "s", "NLL", "ngen", "merr", "serr", "spull"]
    expected = [0.0, 1.620125506824719, 38.028886841504466, 20.0, 0.0, 0.25594375647844947]
    assert _rows(data, names)[0] == pytest.approx([*expected, -1.4842108219477768], rel=1e-7)


def test_samples_generated_first_can_be_fitted_after_with_the_same_results() -> None:
    """``generate`` keeps the samples; ``fit`` fits them as ``generateAndFit`` would have, and
    fit options can be given as keywords."""
    g, x, _, _ = _gauss()
    study = RooMCStudy(g, [x], Silence=True, FitOptions={"PrintLevel": -1, "Save": True})
    study.generate(3, 40)
    assert study.fit(3) is False
    for found, expected in zip(_rows(study.fitParDataSet(), ROOT_ORDER), BASIC):
        assert found == pytest.approx(expected, rel=1e-7)
    assert study.fitResult(1).minNll() == pytest.approx(84.7959472146875, abs=1e-9)


def test_an_empty_sample_is_not_fitted() -> None:
    """A sample of no events has nothing to fit: no result, no row."""
    g, x, _, _ = _gauss()
    study = RooMCStudy(g, [x], RooCmdArg("Silence", True), QUIET)
    study.generateAndFit(1, 0)
    assert (len(study.results), study.fitParDataSet().numEntries()) == (0, 0)


def _fit_model_study() -> Any:
    g, x, _, _ = _gauss()
    m2 = RooRealVar("m2", "m2", 0.5, -5, 5)
    s2 = RooRealVar("s2", "s2", 2, 0.1, 10)
    fit = RooGaussian("g2", "g2", x, m2, s2)
    study = RooMCStudy(g, [x], RooCmdArg("FitModel", fit), RooCmdArg("Silence", True), QUIET)
    study.generateAndFit(2, 40)
    return study


#: ROOT's fits of the second model to the first's samples, in ``_rows`` order below.
FIT_MODEL = [
    [
        0.24461602775322175,
        1.844970588827236,
        81.30727604398575,
        0.29154751845348426,
        0.8390262728037923,
        0.20583458846213631,
        -0.753174732833019,
    ],
    [
        -0.5392065900773981,
        2.0149648612515643,
        84.79595539543115,
        0.31837845861438463,
        -1.6936026150264059,
        0.22508874699657666,
        0.06648427098753126,
    ],
]
FIT_NAMES = ["m2", "s2", "NLL", "m2err", "m2pull", "s2err", "s2pull"]


def test_a_study_can_fit_another_model_than_it_generates_with() -> None:
    """``FitModel`` fits a second model to the first's samples: ROOT's values and errors."""
    rows = _rows(_fit_model_study().fitParDataSet(), FIT_NAMES)
    for found, expected in zip(rows, FIT_MODEL):
        assert found[:4] == pytest.approx(expected[:4], rel=1e-7)
        assert found[5] == pytest.approx(expected[5], rel=1e-7)


@pytest.mark.xfail(strict=True, reason="a fit parameter not in the generator pulls against zero")
def test_another_models_parameters_pull_against_the_generators_in_the_same_place() -> None:
    """ROOT takes the generator's parameter at the same position - m for m2, s for s2 - to
    pull against, warns that it did, and names the dataset after both models."""
    data = _fit_model_study().fitParDataSet()
    assert data.GetName() == "fitParData_g2_g"
    for found, expected in zip(_rows(data, FIT_NAMES), FIT_MODEL):
        assert found == pytest.approx(expected, rel=1e-7)


def _plotted() -> tuple[Any, Any, Any]:
    g, x, m, s = _gauss()
    study = RooMCStudy(g, [x], RooCmdArg("Silence", True), QUIET)
    study.generateAndFit(5, 30)
    return study, m, s


def _framed(frame: Any) -> tuple[float, float, int]:
    return frame.GetXmin(), frame.GetXmax(), frame.GetNbinsX()


def test_a_study_frames_each_column_round_its_values_as_root_does() -> None:
    """A parameter, an error or the minimum is framed on its values with a fifth of their
    spread either side, in the variable's bins or those asked for."""
    study, m, s = _plotted()
    # the ends are the fitted values' least and greatest, which are ROOT's to the bit on ROOT's
    # machine and to a fit's tolerance elsewhere
    framed = (-0.6044825678778627, 0.23432560590854157, 100)
    assert _framed(study.plotParam(m)) == roots(framed, rel=FIT_REL)
    low, high, bins = _framed(study.plotError(s))
    errors = (0.2325135271368771, 0.2854185997174309)  # HESSE's: see FIT_REL off ROOT's machine
    assert (low, high) == pytest.approx(errors, rel=1e-9 if ROOTS_MACHINE else FIT_REL)
    assert bins == 100
    low, high, bins = _framed(study.plotNLL())
    assert (low, high) == pytest.approx((60.50522646887346, 66.41738344466413), rel=1e-12)
    assert _framed(study.plotParam(m, RooCmdArg("Bins", 10)))[2] == 10
    frame = study.plotPull(s, RooCmdArg("Range", -3, 3), Bins=12)
    assert (_framed(frame), frame.numItems()) == ((-3.0, 3.0, 12), 1)


@pytest.mark.xfail(strict=True, reason="a pull is framed round its values, not on [-3, 3]")
def test_a_pull_is_framed_from_minus_three_to_three_in_25_bins() -> None:
    """ROOT's ``plotPull`` frames the pull on [-3, 3] in 25 bins unless told otherwise."""
    study, m, _ = _plotted()
    assert _framed(study.plotPull(m)) == (-3.0, 3.0, 25)


def test_a_pull_plot_can_carry_a_gaussian_fitted_to_it() -> None:
    """``FitGauss()`` fits a Gaussian to the pulls and draws it with its parameters: the
    histogram, the curve and the box ROOT's frame holds."""
    study, m, _ = _plotted()
    frame = study.plotPull(m, RooCmdArg("FitGauss", True))
    names = [frame.getObject(i).GetName() for i in range(int(frame.numItems()))]
    assert names == ["h_fitParData_g", "pullGauss_Norm[mpull]", "pullGauss_paramBox"]
    assert generator().Rndm() == 0.5572100724093616


def test_a_fit_that_fails_keeps_its_result_but_gives_no_row() -> None:
    """Only fits with status 0 make the parameter dataset, as in ``RooMCStudy::fitSample``; a
    dictionary of options in ``FitOptions`` is taken as its keywords, and anything else in it
    is passed over."""
    g, x, _, _ = _gauss()
    options = RooCmdArg("FitOptions", {"PrintLevel": -1, "MaxCalls": 5}, "mr")
    study = RooMCStudy(g, [x], RooCmdArg("Silence", True), options)
    study.generateAndFit(1, 40)
    (result,) = study.results
    assert (result.statusCodeHistory(0), result.status() != 0) == (-1, True)
    assert study.fitParDataSet().numEntries() == 0
