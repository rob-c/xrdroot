"""``ROOT::Minuit2``'s own interface: functions, parameters, applications, errors and contours.

The call counts and values are what ROOT 6.40's Minuit2 gives for the same
function from the same start (its ``Quad4FMain``): iminuit binds the very
same C++.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT

pytest.importorskip("iminuit")


class Quad(ROOT.FCNBase):
    """``Quad4F``: a quadratic bowl in four parameters."""

    def __call__(self, p):
        x, y, z, w = p[0], p[1], p[2], p[3]
        return (1.0 / 70.0) * (21 * x * x + 20 * y * y + 19 * z * z - 14 * x * z - 20 * y * z) \
            + w * w


class QuadGrad(ROOT.FCNGradientBase):
    """``Quad4FGrad``: the same bowl, with its derivatives."""

    def __call__(self, p):
        return Quad()(p)

    def Gradient(self, p):
        x, y, z, w = p[0], p[1], p[2], p[3]
        return [(42 * x - 14 * z) / 70, (40 * y - 20 * z) / 70, (38 * z - 14 * x - 20 * y) / 70,
                2 * w]


class Parabola(ROOT.FCNBase):
    """``(x - 3)^2 + (y + 1)^2``, with an error level that can be changed."""

    def __init__(self):
        self.up = 1.0

    def __call__(self, p):
        return (p[0] - 3) ** 2 + (p[1] + 1) ** 2

    def Up(self):
        return self.up

    def SetErrorDef(self, up):
        self.up = up


def _four():
    upar = ROOT.MnUserParameters()
    for name in "xyzw":
        upar.Add(name, 1.0, 0.1)
    return upar


def test_migrad_finds_the_minimum_in_the_calls_roots_minuit2_takes():
    minimum = ROOT.MnMigrad(Quad(), _four())()
    assert minimum.IsValid() and minimum.NFcn() == 74 and minimum.Fval() < 1e-8
    graded = QuadGrad()
    found = ROOT.MnMigrad(graded, _four())()
    assert found.NFcn() == 52 and found.UserState().Error("x") == pytest.approx(2)
    ROOT.MnHesse()(graded, found)
    assert found.NFcn() == 75 and found.HasValidCovariance() and found.HasAccurateCovar()
    text = str(found)
    assert "Function calls: 75" in text and "    0 |          x |  free   |" in text
    assert found.UserCovariance()(0, 0) == pytest.approx(4) and found.UserCovariance().Nrow() == 4
    with pytest.raises(xrdroot_errors().UnsupportedFeatureError, match="FunctionMinimum"):
        ROOT.MnHesse(2)(graded, _four())


def xrdroot_errors():
    import xrdroot.errors

    return xrdroot.errors


def test_parameters_are_fixed_limited_and_released_by_name_or_number():
    upar = ROOT.MnUserParameters([1.0, 2.0], [0.1, 0.2])
    assert [upar.GetName(0), upar.Name(1), upar.Index("p1")] == ["p0", "p1", 1]
    upar.Add("c", 5.0)
    upar.Add("l", 0.5, 0.1, 0.0, 1.0)
    upar.SetLimits("p0", -5, 5)
    upar.SetLowerLimit(1, -10)
    upar.SetUpperLimit(1, 10)
    upar.RemoveLimits(1)
    upar.SetValue("p1", 2.5)
    upar.SetError(1, 0.3)
    upar.Fix(1)
    assert list(upar.Params()) == [1.0, 2.5, 5.0, 0.5] and list(upar.Errors())[1] == 0.3
    assert upar.VariableParameters() == 2 and len(upar) == 4
    kinds = [(p.IsFixed(), p.HasLimits(), p.IsConst()) for p in upar.Parameters()]
    assert kinds == [(False, True, False), (True, False, False), (False, False, True),
                     (False, True, False)]
    upar.Release("p1")
    first = upar.Parameter("p0")
    assert (first.Number(), first.GetName(), first.Value(), first.Error()) == (0, "p0", 1.0, 0.1)
    assert (first.HasLowerLimit(), first.HasUpperLimit(), first.LowerLimit(),
            first.UpperLimit()) == (True, True, -5, 5)
    assert upar.Value(1) == 2.5 and upar.Error("p1") == 0.3
    assert "limited" in str(upar) and "const" in str(upar)


def _minimum(fcn):
    migrad = ROOT.MnMigrad(fcn, [0.0, 0.0], [0.1, 0.1], ROOT.MnStrategy(2))
    assert migrad.Strategy().Strategy() == 2 and migrad.Parameters().Value(0) == 0.0
    migrad.Fix("p1")
    once = migrad()
    assert once.UserState().Value(0) == pytest.approx(3) and once.UserState().Value(1) == 0.0
    migrad.State().Release(1)
    return migrad(500, 0.01)


def test_an_application_keeps_its_state_from_one_call_to_the_next():
    minimum = _minimum(Parabola())
    state = minimum.UserState()
    assert state.Value(1) == pytest.approx(-1) and state.IsValid() and state.HasCovariance()
    assert state.Fval() == pytest.approx(0, abs=1e-12) and state.Edm() < 1e-6
    assert state.NFcn() == minimum.NFcn() and state.Covariance()(1, 1) == pytest.approx(1)
    assert minimum.Up() == minimum.ErrorDef() == 1.0
    assert minimum.HasValidParameters() and minimum.HasPosDefCovar() and minimum.HasCovariance()
    flags = (minimum.HasMadePosDefCovar(), minimum.HesseFailed(), minimum.IsAboveMaxEdm(),
             minimum.HasReachedCallLimit())
    assert flags == (False, False, False, False)
    assert minimum.UserParameters().Value(0) == pytest.approx(3)


def test_minos_works_from_a_minimum_at_the_functions_error_level_now():
    fcn = Parabola()
    minos = ROOT.MnMinos(fcn, _minimum(fcn))
    pair = minos(0)
    assert (pair.first, pair.second) == (pytest.approx(-1), pytest.approx(1))
    fcn.SetErrorDef(4.0)
    error = minos.Minos(1)
    assert (error.Lower(), error.Upper()) == (pytest.approx(-2), pytest.approx(2))
    assert (error.IsValid(), error.LowerValid(), error.UpperValid()) == (True, True, True)
    assert error.Parameter() == 1 and error.Value() == pytest.approx(-1) and error.NFcn() > 0
    assert (error.AtLowerLimit(), error.AtUpperLimit()) == (False, False)
    assert "valid" in str(error)
    assert minos.Lower(0) == pytest.approx(-2) and minos.Upper(0) == pytest.approx(2)


def test_contours_go_round_a_minimum_with_the_minos_errors_of_their_parameters():
    fcn = Parabola()
    contours = ROOT.MnContours(fcn, _minimum(fcn))
    assert len(contours(0, 1, 8)) == 8
    cont = contours.Contour(0, 1, 6)
    assert (cont.Xpar(), cont.Ypar(), len(cont()), len(cont.Points())) == (0, 1, 6, 6)
    assert cont.XMinosError().Upper() == pytest.approx(1) and cont.YMinosError().IsValid()
    assert str(cont).startswith("Contours # of function calls")


def test_other_applications_minimise_too():
    fcn = Parabola()
    start = ROOT.MnUserParameters([0.0, 0.0], [0.5, 0.5])
    for application in (ROOT.MnMinimize, ROOT.MnFumiliMinimize):
        assert application(fcn, start)().UserState().Value(0) == pytest.approx(3, abs=1e-2)
    assert ROOT.MnSimplex(fcn, start)().UserState().Value(0) == pytest.approx(3, abs=0.2)
    assert ROOT.Minuit2.MnMigrad is ROOT.MnMigrad and "Minuit2" in repr(ROOT.Minuit2)


def test_a_scan_steps_one_parameter_and_keeps_the_lowest_point(capsys):
    fcn = Parabola()
    scan = ROOT.MnScan(fcn, ROOT.MnUserParameters([0.0, 0.0], [0.5, 0.5]),
                       ROOT.MnUserCovariance(), 0)
    points = scan.Scan(0, 5, -4.0, 4.0)
    assert len(points) == 6 and scan.State().Value(0) == 2.0  # the first of two lowest
    assert len(scan.Scan(1)) == 42
    limited = ROOT.MnUserParameters([0.0], [1.0])
    limited.SetLimits(0, -1.0, 0.5)
    assert ROOT.MnScan(Line(), limited).Scan(0, 3)[-1].first == 0.5
    ROOT.MnPlot()(1.0, 2.0, points)
    ROOT.MnPlot()(points)
    assert "minimum at (1, 2)" in capsys.readouterr().out


def test_the_modular_minimisers_take_values_and_errors_or_parameters():
    fcn = Parabola()
    for minimizer in (ROOT.VariableMetricMinimizer(), ROOT.CombinedMinimizer(),
                      ROOT.SimplexMinimizer(), ROOT.FumiliMinimizer()):
        found = minimizer.Minimize(fcn, [[0.0, 0.0], [0.1, 0.1]])
        assert found.UserState().Value(0) == pytest.approx(3, abs=0.2)
    start = ROOT.MnUserParameters([0.0, 0.0], [0.5, 0.5])
    assert ROOT.ScanMinimizer().Minimize(fcn, start, 1, 200, 0.1).Fval() >= 0


class Line(ROOT.FCNBase):
    def __call__(self, p):
        return p[0]


def test_a_failed_migrad_falls_back_to_the_simplex():
    class Cliff(ROOT.FCNBase):
        def __call__(self, p):
            return abs(p[0]) ** 0.5

    found = ROOT.MnMinimize(Cliff(), [1.0], [0.5])(20)
    assert found.UserState().Value(0) == pytest.approx(0, abs=0.5)


def test_fumili_chi2_sets_the_model_parameters_to_each_position():
    class Model(ROOT.ParametricFunction):
        def __init__(self):
            super().__init__(1)

        def __call__(self, x):
            return x[0] * self.par[0] + x[1]

    model = Model()
    assert model.NumberOfParameters() == 1 and model.GetParameters() == [0.0]
    model.SetParameters([2.0])
    chi2 = ROOT.FumiliStandardChi2FCN(model, [3.0, 5.0, 7.0], [1.0, 2.0, 3.0], [1.0, 0.0, 4.0])
    assert chi2([2.0, 1.0]) == 0.0 and chi2([2.0, 2.0]) == pytest.approx(1 + 1 + 0.25)
    assert ROOT.ParametricFunction([1.0, 2.0]).GetParameters() == [1.0, 2.0]
    assert ROOT.MnMigrad(chi2, [1.0, 0.0], [0.1, 0.1])().UserState().Value(0) == pytest.approx(2)


def test_the_bases_say_what_a_subclass_must_give():
    with pytest.raises(NotImplementedError):
        ROOT.FCNBase()([1.0])
    with pytest.raises(NotImplementedError):
        ROOT.FCNGradientBase().Gradient([1.0])
    base = ROOT.FCNBase()
    base.SetErrorDef(2)
    assert (base.Up(), base.ErrorDef(), base.HasGradient()) == (1.0, 1.0, False)
    migrad = ROOT.MnMigrad(Parabola(), [0.0, 0.0], [0.1, 0.1])
    with pytest.raises(AttributeError):
        migrad.__getattr__("_missing")
    assert ROOT.MnUserCovariance().Nrow() == 0 and str(ROOT.MnUserCovariance([[1.0]])).strip()
    state = ROOT.MnUserParameterState()
    assert (state.Fval(), state.Edm(), state.Covariance().Nrow()) == (0.0, 0.0, 0)


def test_a_strategy_is_set_by_level():
    assert ROOT.MnStrategy(ROOT.MnStrategy(2)).Strategy() == 2
    strategy = ROOT.MnStrategy()
    strategy.SetHighStrategy()
    assert int(strategy) == 2
    strategy.SetLowStrategy()
    assert strategy.Strategy() == 0
    strategy.SetMediumStrategy()
    assert strategy.Strategy() == 1
    precision = ROOT.MnMachinePrecision()
    assert precision.Eps() > 0 and precision.Eps2() > precision.Eps()


def test_the_printer_says_what_its_level_lets_it(capsys):
    assert ROOT.MnPrint.SetGlobalLevel(2) == 0 and ROOT.MnPrint.GlobalLevel() == 2
    printer = ROOT.MnPrint("Test")
    printer.Info("shown")
    printer.Debug("hidden")
    assert printer.SetLevel(3) == 2 and printer.Level() == 3
    printer.Debug("now")
    printer.Warn("warned")
    printer.Error("erred")
    ROOT.MnPrint.SetGlobalLevel(0)
    assert capsys.readouterr().out.split("\n")[:2] == ["Test: shown", "Test: now"]
