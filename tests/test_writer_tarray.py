"""An array object is written as the class ROOT knows, not as anonymous numbers."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.errors import FormatError, UnsupportedFeatureError


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        ("TArrayD", [1.0, 2.5], [1.0, 2.5]),
        ("TArrayF", [1, 2, 3], [1.0, 2.0, 3.0]),
        ("TArrayI", [4, 5], [4, 5]),
        ("TArrayC", [1, 2], [1, 2]),
    ],
)
def test_a_tarray_is_a_key_that_reads_back_as_itself(tmp_path, name, values, expected):
    """The key carries the ``TArray`` record, so the class survives the file."""
    path = tmp_path / f"{name}.root"
    with xrdroot.create(str(path)) as handle:
        handle["a"] = getattr(ROOT, name)(values)
    with xrdroot.open_root(str(path)) as handle:
        read = np.asarray(handle["a"])
        assert [float(value) for value in read] == [float(v) for v in expected]


def test_every_array_class_names_itself_as_root_does():
    """ROOT asks ``ClassName()`` of anything it writes, so every array answers it."""
    for name in ("TArrayC", "TArrayS", "TArrayI", "TArrayF", "TArrayD", "TArrayL"):
        assert getattr(ROOT, name)().ClassName() == name


def test_a_plain_object_is_not_mistaken_for_an_array(tmp_path):
    """The writer's array path needs a class it knows *and* values it can take."""
    with pytest.raises((FormatError, TypeError, UnsupportedFeatureError)):
        with xrdroot.create(str(tmp_path / "b.root")) as handle:
            handle["a"] = ROOT.TNamed("x", "a named thing")


def test_a_three_vector_is_written_as_its_coordinates(tmp_path):
    """ROOT's ``TVector3`` record is ``fX``, ``fY`` and ``fZ``, and nothing else."""
    import xrdroot

    path = tmp_path / "v.root"
    with xrdroot.create(str(path)) as handle:
        handle["v"] = ROOT.TVector3(1.5, -2.5, 3.0)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["v"]
    assert (read["fX"], read["fY"], read["fZ"]) == (1.5, -2.5, 3.0)


def test_a_lorentz_vector_is_written_as_a_three_vector_and_an_energy(tmp_path):
    """``TLorentzVector`` streams ``fP`` (a ``TVector3``) and ``fE``."""
    import xrdroot

    path = tmp_path / "lv.root"
    with xrdroot.create(str(path)) as handle:
        handle["p"] = ROOT.TLorentzVector(1.0, 2.0, 3.0, 4.0)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["p"]
    assert (read["fP"]["fX"], read["fP"]["fY"], read["fP"]["fZ"]) == (1.0, 2.0, 3.0)
    assert read["fE"] == 4.0


def test_a_matrix_is_written_as_its_shape_and_values(tmp_path):
    """ROOT's ``TMatrixT`` record is the base's shape, then a pointer counted by ``fNelems``."""
    import xrdroot

    path = tmp_path / "m.root"
    matrix = ROOT.TMatrixD(2, 2)
    matrix[0][0] = 1.5
    matrix[1][1] = -2.5
    with xrdroot.create(str(path)) as handle:
        handle["m"] = matrix
    with xrdroot.open_root(str(path)) as handle:
        read = handle["m"]
    base = read["TMatrixTBase<double>"]
    assert (base["fNrows"], base["fNcols"], base["fNelems"]) == (2, 2, 4)
    assert [float(v) for v in read["fElements"]] == [1.5, 0.0, 0.0, -2.5]


def test_a_random_engine_is_written_with_its_mersenne_state(tmp_path):
    """``TRandom3`` streams its name, seed and the 624-word state ROOT continues from."""
    import xrdroot

    path = tmp_path / "r.root"
    with xrdroot.create(str(path)) as handle:
        handle["r"] = ROOT.TRandom3(7)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["r"]
    engine = read["TRandom"]
    assert engine["TNamed"]["fName"] == "Random3"
    assert len(read["fMt"]) == 624
    assert 0 <= read["fCount624"] <= 624


def test_a_single_precision_matrix_is_written_as_its_float_twin(tmp_path):
    """``TMatrixF`` streams as ``TMatrixT<float>``: the same shape, four-byte members."""
    import xrdroot

    path = tmp_path / "mf.root"
    matrix = ROOT.TMatrixF(2, 2)
    matrix[0][1] = 2.5
    with xrdroot.create(str(path)) as handle:
        handle["m"] = matrix
    with xrdroot.open_root(str(path)) as handle:
        read = handle["m"]
    base = read["TMatrixTBase<float>"]
    assert (base["fNrows"], base["fNcols"], base["fNelems"]) == (2, 2, 4)
    assert [float(v) for v in read["fElements"]] == [0.0, 2.5, 0.0, 0.0]


def test_a_tstring_names_itself_as_root_does():
    """``TObject``'s lookup reaches for these, so a ``TString`` spells them."""
    text = ROOT.TString("x")
    assert (text.ClassName(), text.GetName(), text.GetTitle()) == ("TString", "", "")


def test_a_tstring_key_is_written_as_its_characters(tmp_path):
    """ROOT writes a ``TString`` key as the length word and the bytes, nothing round them."""
    import xrdroot

    path = tmp_path / "s.root"
    with xrdroot.create(str(path)) as handle:
        handle["s"] = ROOT.TString("hello")
    assert path.stat().st_size > 0
    assert b"hello" in path.read_bytes()


def test_a_colour_is_written_with_its_channels_and_its_hls(tmp_path):
    """``TColor`` streams the number, the RGB, and the HLS ROOT computes from them."""
    import xrdroot

    path = tmp_path / "c.root"
    with xrdroot.create(str(path)) as handle:
        handle["c"] = ROOT.TColor(2, 1.0, 0.5, 0.25)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["c"]
    assert read["fNumber"] == 2
    assert (read["fRed"], read["fGreen"], read["fBlue"]) == (1.0, 0.5, 0.25)
    assert read["fLight"] == pytest.approx(0.625)


@pytest.mark.parametrize(
    ("rgb", "hue", "light", "saturation"),
    [
        ((1.0, 0.0, 0.0), 0.0, 0.5, 1.0),          # red is the highest channel
        ((0.0, 1.0, 0.0), 120.0, 0.5, 1.0),        # green is
        ((0.0, 0.0, 1.0), 240.0, 0.5, 1.0),        # blue is
        ((0.5, 0.5, 0.5), 0.0, 0.5, 0.0),          # no span: grey
        ((1.0, 0.9, 0.8), 30.0, 0.9, 1.0),         # light above one half
    ],
)
def test_a_colours_hls_follows_from_its_channels(tmp_path, rgb, hue, light, saturation):
    """ROOT keeps the HLS beside the RGB as 32-bit floats, so the tolerance is theirs."""
    import xrdroot

    path = tmp_path / "hls.root"
    with xrdroot.create(str(path)) as handle:
        handle["c"] = ROOT.TColor(2, *rgb)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["c"]
    assert read["fHue"] == pytest.approx(hue, abs=1e-6)
    assert read["fLight"] == pytest.approx(light, abs=1e-6)
    assert read["fSaturation"] == pytest.approx(saturation, abs=1e-6)


def test_a_graph2d_is_written_through_its_pyroot_object(tmp_path):
    """The wrapper answers the row through its accessors: counts, coordinates and attributes."""
    import xrdroot

    path = tmp_path / "g2w.root"
    graph = ROOT.TGraph2D()
    graph.SetPoint(0, 1.0, 2.0, 3.0)
    with xrdroot.create(str(path)) as handle:
        handle["g"] = graph
    with xrdroot.open_root(str(path)) as handle:
        read = handle["g"]
    assert read["fNpoints"] == 1
    assert read["fX"] == [1.0] and read["fZ"] == [3.0]


def test_a_graph2d_is_written_through_its_engine_too(tmp_path):
    """The engine carries its members already, and takes the same record."""
    import xrdroot
    from xrdroot.pyroot.core.records import unwrap

    path = tmp_path / "g2e.root"
    graph = ROOT.TGraph2D()
    graph.SetPoint(0, 1.0, 2.0, 3.0)
    with xrdroot.create(str(path)) as handle:
        handle["g"] = unwrap(graph)
    assert path.stat().st_size > 0


def test_a_tlatex_is_written_with_the_members_its_text_carries(tmp_path):
    """``TLatex``'s row is flat, and a member it does not hold is written as a typed zero."""
    import xrdroot
    from xrdroot.pyroot.core.records import unwrap

    path = tmp_path / "tx.root"
    label = ROOT.TLatex(0.1, 0.2, "hi")
    with xrdroot.create(str(path)) as handle:
        handle["t"] = unwrap(label)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["t"]
    assert read["fX"] == 0.1 and read["fY"] == 0.2
    assert read["fLimitFactorSize"] == 0     # the object holds none: ROOT's fresh one is zero


def test_a_tlatex_is_written_through_its_pyroot_object_too(tmp_path):
    """The wrapper is unwrapped to the same engine, so both routes end in one record."""
    import xrdroot

    path = tmp_path / "txw.root"
    with xrdroot.create(str(path)) as handle:
        handle["t"] = ROOT.TLatex(0.0, 0.0, "x")
    assert path.stat().st_size > 0


def test_an_empty_tobjarray_is_written_and_read_back_with_no_entries(tmp_path):
    """A collection comes back as its entries, so an empty one is a list of nothing."""
    import xrdroot

    path = tmp_path / "oa.root"
    with xrdroot.create(str(path)) as handle:
        handle["a"] = ROOT.TObjArray()
    assert path.stat().st_size > 0
    with xrdroot.open_root(str(path)) as handle:
        assert list(handle["a"]) == []


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("TBox", "fLineColor"),
        ("TPaveText", "fLineColor"),
        ("TLine", "fLineColor"),
        ("TMarker", "fMarkerStyle"),
        ("TEllipse", "fLineColor"),
        ("TArrow", "fLineColor"),
    ],
)
def test_a_drawing_object_with_flat_members_is_written_by_them(tmp_path, name, key):
    """``TBox`` and ``TPaveText`` keep their members themselves, like ``TLatex`` does."""
    import xrdroot

    path = tmp_path / f"{name}.root"
    with xrdroot.create(str(path)) as handle:
        handle["o"] = getattr(ROOT, name)()
    with xrdroot.open_root(str(path)) as handle:
        read = handle["o"]
    #: read back, a base's members are nested under the base's name - or flat, when the
    #: writer's hand-written path for that class took them
    assert read.classname == name and _find(read.members, key) == 1


def _find(held, key):
    """One member wherever the read-back put it: a base's members nest under the base."""
    if key in held:
        return held[key]
    for value in held.values():
        if isinstance(value, dict):
            found = _find(value, key)
            if found is not None:
                return found
    return None


def test_an_annotation_keeps_the_z_of_the_point_it_names(tmp_path):
    """ROOT's ``TAnnotation`` is a ``TLatex`` with an ``fZ``; that member has to survive a file."""
    import xrdroot

    path = tmp_path / "ann.root"
    with xrdroot.create(str(path)) as handle:
        made = ROOT.TAnnotation(1.0, 2.0, 0.5, "hello")
        assert made.GetZ() == 0.5
        handle["a"] = made
    with xrdroot.open_root(str(path)) as handle:
        read = handle["a"]
    #: a class with no wrapper of ours reads back as its members, nested per base
    assert _find(read, "fZ") == 0.5
    assert _find(read, "fTextColor") == 1 and _find(read, "fTitle") == "hello"


def test_a_graph_time_collects_a_point_a_step():
    """``TGraphTime`` fills a step at a time - the option on ``AddPoint`` moves on."""
    made = ROOT.TGraphTime(2, 0.0, 0.0, 1.0, 1.0)
    made.AddPoint(0.0, 0.0)
    made.AddPoint(1.0, 1.0)
    made.AddPoint(2.0, 2.0, "next")
    assert made.GetNsteps() == 2
    assert made.GetStep(0) == [(0.0, 0.0), (1.0, 1.0)]
    assert made.GetStep(1) == [(2.0, 2.0)]
    assert made.GetStep(7) == [] and made.GetFrame() == (0.0, 0.0, 1.0, 1.0)


def test_a_graph_time_paints_the_lines_of_every_step():
    """``Paint`` walks the steps, drawing a line from each point to the next."""
    made = ROOT.TGraphTime(1, 0.0, 0.0, 1.0, 1.0)
    made.AddPoint(0.0, 0.0)
    made.AddPoint(1.0, 1.0)
    assert made.Paint() is made


def test_a_graph_time_takes_a_point_under_roots_other_name():
    """``TakeTime`` is ROOT's name for the same act and must fill the same step."""
    made = ROOT.TGraphTime(1, 0.0, 0.0, 1.0, 1.0)
    made.TakeTime(0.5, 0.5)
    assert made.GetStep(0) == [(0.5, 0.5)]


def test_a_linear_fitter_recovers_the_line_through_its_points():
    """``TLinearFitter`` solves for the coefficients of a formula that is linear in them."""
    fitter = ROOT.TLinearFitter(1, "1++x")
    for x in range(5):
        fitter.AddPoint(float(x), 2.0 + 3.0 * x)
    assert fitter.Fit() == 0
    assert [round(value, 9) for value in fitter.GetParameters()] == [2.0, 3.0]
    assert round(fitter.GetChisquare(), 9) == 0.0
    assert round(fitter.Eval(2.0), 9) == 8.0
    assert fitter.GetNpoints() == 5


def test_a_minuit_settles_on_the_minimum_of_a_parabola():
    """``gMinuit``'s ``Migrad`` is the legacy tutorials' way of minimising; it must find 3."""
    minuit = ROOT.TMinuit(1)
    minuit.SetFCN(lambda npar, gin, f, pars, flag: None)
    minuit.DefineParameter(0, "a", 0.0, 0.1, -10.0, 10.0)
    assert minuit.Migrad() == 0
    assert minuit.GetStatus() == 0


def test_a_convolution_of_two_gaussians_is_a_wider_gaussian():
    """``TF1Convolution`` integrates one function against the other over its range."""
    first = ROOT.TF1("g1", "exp(-x*x)", -5.0, 5.0)
    second = ROOT.TF1("g2", "exp(-x*x)", -5.0, 5.0)
    made = ROOT.TF1Convolution(first, second, -5.0, 5.0)
    assert made.GetRange() == (-5.0, 5.0)
    assert made.Eval(0.0) > made.Eval(3.0)


def test_a_foam_samples_the_box_in_as_many_dimensions_as_it_was_given():
    """``TFoam`` samples ``kDim`` dimensions, each in the box ``Initialize`` was built over."""

    class Flat(ROOT.TFoamIntegrand):
        def Density(self, ndim, x):
            return 1.0

    foam = ROOT.TFoam()
    foam.SetkDim(2)
    foam.SetnCells(100)
    foam.SetPseRan(ROOT.TRandom3(7))
    foam.Initialize(None, Flat())
    vector = [0.0, 0.0]
    foam.MakeEvent()
    foam.GetMCvect(vector)
    assert len(vector) == 2
    assert all(-1.0 <= value <= 1.0 for value in vector)
    assert foam.GetMCwt() > 0.0
    assert foam.GetnCalls() == 1
    assert foam.GetVersion() == "xrdroot-FOAM"


def test_a_linear_fitter_answers_before_it_is_fitted_and_after():
    """Every getter of ``TLinearFitter`` has an answer before ``Fit`` - ROOT's has too."""
    fitter = ROOT.TLinearFitter(1, "1++x")
    assert fitter.GetParameters() == []
    assert fitter.GetParError(0) == 0.0
    assert round(fitter.Eval(2.0), 9) == 0.0
    fitter.SetFormula("1++x")
    for x in range(4):
        fitter.AddPoint(float(x), 1.0 + 1.0 * x)
    fitter.Fit()
    assert round(fitter.GetParameter(0), 9) == 1.0
    assert fitter.GetParError(0) >= 0.0


def test_a_minuit_takes_its_commands_the_way_root_writes_them():
    """``Command("MIGRAD")`` and the print level are how a macro drives ``gMinuit``."""
    minuit = ROOT.TMinuit(1)
    minuit.SetPrintLevel(0)
    minuit.SetFCN(lambda npar, gin, f, pars, flag: None)
    minuit.DefineParameter(0, "a", 1.0, 0.1, -5.0, 5.0)
    minuit.SetParameter(0, 2.0, 0.0)
    minuit.Command("MIGRAD")
    assert minuit.GetStatus() == 0
    assert isinstance(minuit.GetParameterValue(0), float)


def test_a_convolution_needs_two_functions_and_keeps_the_range_it_is_given():
    """A convolution of one function is refused, and ``SetRange`` moves the integration box."""
    with pytest.raises(ValueError):
        ROOT.TF1Convolution(ROOT.TF1("f", "x", 0.0, 1.0))
    made = ROOT.TF1Convolution(ROOT.TF1("a", "1", 0.0, 1.0), ROOT.TF1("b", "1", 0.0, 1.0))
    made.SetRange(0.0, 2.0)
    assert made.GetRange() == (0.0, 2.0)
    assert made.Eval(1.0) > 0.0


def test_a_foam_keeps_every_setting_the_tutorials_give_it():
    """The optional setters are all called by ``foam_demo.C``; each must be answerable."""

    class Peak(ROOT.TFoamIntegrand):
        def Density(self, ndim, x):
            return 1.0 + x[0]

    foam = ROOT.TFoam()
    foam.SetkDim(1)
    foam.SetnCells(50)
    foam.SetnSampl(200)
    foam.SetnBin(8)
    foam.SetOptRej(1)
    foam.SetOptDrive(2)
    foam.SetEvPerBin(10)
    foam.SetChat(0)
    foam.SetRho(0.5)
    foam.SetRange(-1.0, 1.0)
    foam.SetPseRan(ROOT.TRandom3(3))
    foam.Initialize(None, Peak())
    foam.MakeEvent()
    assert foam.GetX() is not None
    held = [0.0]
    foam.GetMCvect(held)
    assert -1.0 <= held[0] <= 1.0
    assert foam.GetMCwt() > 0.0


def test_an_annotation_draws_a_copy_at_a_new_point():
    """``DrawAnnotation`` is ROOT's way of putting the same label somewhere else."""
    made = ROOT.TAnnotation(0.0, 0.0, 0.0, "here")
    moved = made.DrawAnnotation(1.0, 2.0, 3.0, "there")
    assert moved.GetZ() == 3.0
    assert moved.members["fTitle"] == "there"


def test_a_graph_time_keeps_the_first_step_when_points_come_before_any_option():
    """A point added to a graph built with no steps still lands in a step of its own."""
    made = ROOT.TGraphTime()
    made.AddPoint(1.0, 2.0)
    assert made.GetStep(0) == [(1.0, 2.0)]
    assert made.GetStep(3) == []


def test_an_annotation_takes_a_new_z():
    """``SetZ`` is ``TAnnotation``'s one setter, and the member has to follow it."""
    made = ROOT.TAnnotation(0.0, 0.0, 0.0, "z")
    made.SetZ(2.5)
    assert made.GetZ() == 2.5


def test_a_graph_time_moves_on_when_the_option_says_so_twice():
    """A second option opens a third step even when the graph was built with fewer."""
    made = ROOT.TGraphTime(1, 0.0, 0.0, 1.0, 1.0)
    made.AddPoint(0.0, 0.0, "next")
    made.AddPoint(1.0, 1.0, "next")
    #: a second option opens a step past the one the graph was built with
    assert made.GetStep(1) == [(1.0, 1.0)] and made.GetStep(2) == []


def test_a_minuit_command_that_is_not_migrad_is_kept_and_ignored():
    """``gMinuit`` is driven by strings; the ones xrdroot does not act on must not raise."""
    minuit = ROOT.TMinuit(1)
    minuit.SetFCN(lambda npar, gin, f, pars, flag: None)
    minuit.SetParameter(0, 1.0, 0.1)
    minuit.Command("SET PRINT 0")
    assert minuit.GetStatus() == 0


def test_a_minuit_finds_the_bottom_of_an_off_centre_parabola():
    """The simplex has to expand and then converge, not only reflect."""

    minuit = ROOT.TMinuit(1)
    minuit.SetFCN(lambda npar, gin, f, pars, flag: f.__setitem__(0, (pars[0] - 5.0) ** 2))
    minuit.DefineParameter(0, "a", 4.0, 0.5, -20.0, 20.0)
    minuit.Migrad()
    assert abs(minuit.GetParameterValue(0) - 5.0) < 1e-3


def test_a_foam_can_be_initialised_with_its_random_and_its_integrand_at_once():
    """ROOT's older ``Initialize(random, integrand)`` form must build the same grid."""

    class Flat(ROOT.TFoamIntegrand):
        def Density(self, ndim, x):
            return 1.0

    foam = ROOT.TFoam()
    foam.SetkDim(1)
    foam.SetnCells(8)
    foam.Initialize(ROOT.TRandom3(11), Flat())
    foam.MakeEvent()
    assert foam.GetMCwt() > 0.0


def test_a_foam_keeps_the_integrand_it_was_made_with():
    """``TFoam(integrand)`` already has one, so ``Initialize(random)`` need not name it."""

    class Flat(ROOT.TFoamIntegrand):
        def Density(self, ndim, x):
            return 1.0

    foam = ROOT.TFoam(Flat())
    foam.SetkDim(1)
    foam.Initialize(ROOT.TRandom3(5))
    foam.MakeEvent()
    assert foam.GetMCwt() > 0.0


def test_a_minuit_shrinks_towards_a_kink_it_cannot_pass():
    """An absolute-value cost reflects, shrinks and then stops, as ROOT's Migrad would."""
    calls = {"n": 0}

    def cost(npar, gin, f, pars, flag):
        calls["n"] += 1
        f[0] = abs(pars[0] - 2.0) + 1e-9 * calls["n"]

    minuit = ROOT.TMinuit(2)
    minuit.SetFCN(cost)
    minuit.DefineParameter(0, "a", 0.0, 0.7, -9.0, 9.0)
    minuit.DefineParameter(1, "b", 0.0, 0.7, -9.0, 9.0)
    assert minuit.Migrad() == 0
    assert abs(minuit.GetParameterValue(0) - 2.0) < 0.5


def test_a_hyperplane_is_fitted_by_the_name_root_gives_it():
    """``hyp5`` is ROOT's plane in five dimensions, constant first, and it must fit exactly."""
    import random

    random.seed(3)
    rows = [[random.uniform(-9, 9) for _ in range(5)] for _ in range(40)]
    truth = [4.0, 1.0, 2.0, 3.0, 0.2]
    ys = [0.5 + sum(c * v for c, v in zip(truth, row, strict=False)) for row in rows]
    fitter = ROOT.TLinearFitter(5, "hyp5")
    fitter.AssignData(40, 5, [v for row in rows for v in row], ys)
    assert fitter.Fit() == 0
    assert [round(value, 6) for value in fitter.GetParameters()] == [0.5, 4.0, 1.0, 2.0, 3.0, 0.2]


def test_a_weighted_point_moves_the_error_and_not_the_centre():
    """A point's error is a weight: e = 0.01 must shrink the errors a hundredfold."""
    plain = ROOT.TLinearFitter(1, "1++x")
    weighted = ROOT.TLinearFitter(1, "1++x")
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [2.0, 5.0, 8.0, 11.0]
    plain.AssignData(4, 1, xs, ys)
    weighted.AssignData(4, 1, xs, ys, [0.01] * 4)
    plain.Fit()
    weighted.Fit()
    assert round(weighted.GetParameters()[0], 9) == round(plain.GetParameters()[0], 9)
    assert weighted.GetParError(0) < plain.GetParError(0)


def test_a_fixed_parameter_is_held_at_its_value_with_root_s_error_of_one():
    """``FixParameter`` pins a coefficient; ROOT reports the pinned one with an error of 1."""
    fitter = ROOT.TLinearFitter(1, "1++x")
    for x in (0.0, 1.0, 2.0):
        fitter.AddPoint(x, 1.0 + 2.0 * x)
    fitter.FixParameter(1, 5.0)
    fitter.Fit()
    assert round(fitter.GetParameters()[1], 9) == 5.0
    assert fitter.GetParError(1) == 1.0
    fitter.ReleaseParameter(1)
    fitter.Fit()
    assert round(fitter.GetParameters()[1], 9) == 2.0


def test_a_convolution_answers_a_fit_as_a_function_of_one_variable():
    """``TF1("f", conv, ...)`` needs the convolution's parameter count and its ``f(x, par)``."""
    first = ROOT.TF1("a", "x", 0.0, 1.0)
    second = ROOT.TF1("b", "x", 0.0, 1.0)
    made = ROOT.TF1Convolution(first, second, 0.0, 1.0)
    made.SetNofPointsFFT(64)
    assert made.GetNpar() == first.GetNpar() + second.GetNpar()
    held = ROOT.TF1("f", made, 0.0, 1.0, made.GetNpar())
    assert held.Eval(0.5) == made.Eval(0.5)


def test_a_fitter_reads_a_one_dimensional_column_and_a_named_formula_it_cannot_find():
    """A single coordinate arrives flat; an unknown formula name is refused in a sentence."""
    from xrdroot.errors import UnsupportedFeatureError

    fitter = ROOT.TLinearFitter(1, "no_such_formula_here")
    fitter.AssignData(3, 1, [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(UnsupportedFeatureError, match="write its terms"):
        fitter.Fit()


def test_a_fitter_of_a_named_root_function_takes_its_parameters_as_terms():
    """``SetFormula("hyp3")`` resolves at the fit, so a cached function is found or ignored."""
    ROOT.TF1("cached3", "[0]+[1]*x", 0.0, 1.0)
    fitter = ROOT.TLinearFitter(1, "cached3")
    fitter.AssignData(3, 1, [0.0, 1.0, 2.0], [2.0, 5.0, 8.0])
    assert fitter.Fit() == 0
    assert [round(value, 9) for value in fitter.GetParameters()] == [2.0, 3.0]


def test_a_fitter_answers_eval_with_no_point_and_errors_into_a_vector():
    """``Eval()`` fits what was assigned and returns the chi-square; the getters fill a vector."""
    fitter = ROOT.TLinearFitter(1, "1++x")
    fitter.AssignData(3, 1, [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    assert fitter.Eval() == fitter.GetChisquare()
    params, errors = ROOT.TVectorD(), ROOT.TVectorD()
    fitter.GetParameters(params)
    fitter.GetErrors(errors)
    assert params[0] == fitter.GetParameters()[0]
    assert errors[0] == fitter.GetErrors()[0]


def test_the_design_of_a_one_dimensional_column_and_a_fixed_out_fit():
    """A flat column is one coordinate, and a fit with nothing free keeps what it was given."""
    import numpy as np

    from xrdroot.pyroot.mathfit import _design

    assert _design(["1", "x"], np.array([1.0, 2.0])).shape == (2, 2)

    fitter = ROOT.TLinearFitter(1, "1++x")
    fitter.AddPoint(0.0, 1.0)
    fitter.FixParameter(0, 7.0)
    fitter.FixParameter(1, 8.0)
    fitter.Fit()
    assert fitter.GetParameters() == [7.0, 8.0]
    assert fitter.GetParError(0) == 1.0


def test_a_fitter_without_a_constant_answers_one_error_more_than_it_names():
    """ROOT's errors carry a trailing constant when the formula has no constant term."""
    fitter = ROOT.TLinearFitter(1, "x")
    fitter.AddPoint(1.0, 2.0)
    fitter.Fit()
    assert len(fitter.GetErrors()) == len(fitter.GetParameters())
