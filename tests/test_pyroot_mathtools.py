"""``ROOT.Math``'s functors and tools, ``ROOT.Fit.Fitter``, and names given by keyword."""

from __future__ import annotations

import types

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import mathtools


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_functors_wrap_python_functions():
    one = ROOT.Math.Functor1D(lambda x: x * x - 1)
    many = ROOT.Math.Functor(lambda xx: xx[0] + xx[1], 2)
    expect(
        (one(3), 8),
        (one.NDim(), 1),
        (many([1, 2]), 3),
        (many.NDim(), 2),
    )
    grad = ROOT.Math.GradFunctor1D(lambda x: x * x, lambda x: 2 * x)
    assert grad.Derivative(3) == 6
    grad2 = ROOT.Math.GradFunctor(lambda xx: xx[0] * xx[1], lambda xx, i: xx[1 - i], 2)
    assert grad2.Derivative([2, 3], 0) == 3


def test_the_integrator_is_gsls_rule_first():
    integrator = ROOT.Math.Integrator()
    integrator.SetFunction(ROOT.Math.Functor1D(lambda x: x * x - 1))
    expect(
        (integrator.Integral(0, 3), 5.999999999999999),
        (integrator.Status(), 0),
        (bool(integrator.Error() < 1e-12), True),
    )
    wiggly = ROOT.Math.IntegratorOneDim(ROOT.Math.Functor1D(lambda x: np.sqrt(abs(x))))
    expect(
        (wiggly.Integral(-1, 1), pytest.approx(4 / 3, rel=1e-8)),
        (mathtools.kronrod(lambda x: 1.0, 0.0, 2.0)[0], pytest.approx(2.0)),
    )


def test_the_root_finder_brackets_or_follows_a_derivative():
    finder = ROOT.Math.RootFinder(ROOT.Math.RootFinder.kGSL_NEWTON)
    finder.SetFunction(ROOT.Math.GradFunctor1D(lambda x: x * x - 1, lambda x: 2 * x), 3)
    expect(
        (bool(finder.Solve()), True),
        (finder.Root(), 1.0),
    )
    bracket = ROOT.Math.RootFinder()
    bracket.SetFunction(ROOT.Math.Functor1D(lambda x: x - 0.25), 0, 1)
    expect(
        (bool(bracket.Solve()), True),
        (bracket.Root(), pytest.approx(0.25)),
        (finder.Name(), "newton"),
        (bool(finder.Iterations() > 0), True),
        (bracket.Name(), "BrentRootFinder"),
        (ROOT.Math.RootFinder(ROOT.Math.RootFinder.kMODAB).Name(), "ModABRootFinder"),
    )
    # ``exp`` has no root and Newton's step never shrinks, so it takes the steps it is
    # given and says how many that was.
    stuck = ROOT.Math.RootFinder(ROOT.Math.RootFinder.kGSL_NEWTON)
    stuck.SetFunction(ROOT.Math.GradFunctor1D(np.exp, np.exp), 1.0)
    assert stuck.Solve(3) is True
    assert stuck.Iterations() == 3


def test_the_fitter_minimises_and_prints_as_root_does(capsys):
    def rosenbrock(xx):
        return 100 * (xx[1] - xx[0] ** 2) ** 2 + (1 - xx[0]) ** 2

    fitter = ROOT.Fit.Fitter()
    assert fitter.FitFCN(ROOT.Math.Functor(rosenbrock, 2), np.zeros(2))
    fitter.Result().Print(ROOT.std.cout)
    out = capsys.readouterr().out
    expect(
        (bool("MinFCN" in out), True),
        (bool("Chi2 " not in out), True),
        (bool("Par_1" in out), True),
        (fitter.Result().Parameter(0), pytest.approx(1.0, abs=1e-3)),
    )

    def gradient(xx, i):
        if i == 0:
            return 2 * (200 * xx[0] ** 3 - 200 * xx[0] * xx[1] + xx[0] - 1)
        return 200 * (xx[1] - xx[0] ** 2)

    graded = ROOT.Fit.Fitter()
    graded.FitFCN(ROOT.Math.GradFunctor(rosenbrock, gradient, 2))
    assert graded.Result().Parameter(1) == pytest.approx(1.0, abs=1e-3)


def test_a_minimizer_from_the_factory_prints_minuit2s_lines(capsys):
    minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2", "")
    minimizer.SetMaxFunctionCalls(100000)
    minimizer.SetMaxIterations(1000)
    minimizer.SetTolerance(0.001)
    minimizer.SetPrintLevel(1)
    minimizer.SetStrategy(1)
    minimizer.SetFunction(ROOT.Math.Functor(lambda v: (v[1] - v[0] ** 2) ** 2 + (1 - v[0]) ** 2, 2))
    minimizer.SetVariable(0, "x", -1.0, 0.01)
    minimizer.SetLimitedVariable(1, "y", 1.2, 0.01, -10, 10)
    expect(
        (bool(minimizer.Minimize()), True),
        (minimizer.Status(), 0),
        (minimizer.NDim(), 2),
    )
    lines = capsys.readouterr().out.splitlines()
    expect(
        (
            lines[0],
            "Minuit2Minimizer: Minimize with max-calls 100000 "
            "convergence for edm < 0.001 strategy 1",
        ),
        (lines[1], "Minuit2Minimizer : Valid minimum - status = 0"),
        (bool(lines[2].startswith("FVAL  = ")), True),
        (bool(lines[5].startswith("x\t  = ")), True),
        (bool("\t +/-  " in lines[5]), True),
        (minimizer.X()[0], pytest.approx(1.0, abs=0.01)),
        (bool(minimizer.MinValue() < 1e-4), True),
        (len(minimizer.Errors()), 2),
        (bool(minimizer.Edm() >= 0), True),
        (bool(minimizer.NCalls() > 0), True),
    )
    graded = ROOT.Math.Minimizer()
    graded.SetFunction(
        ROOT.Math.GradFunctor(lambda v: (v[0] - 2) ** 2, lambda v, i: 2 * (v[0] - 2), 1)
    )
    graded.SetVariable(0, "a", 0.0, 0.1)
    expect(
        (bool(graded.Minimize()), True),
        (graded.X()[0], pytest.approx(2.0, abs=1e-4)),
    )
    graded._found = types.SimpleNamespace(
        valid=False,
        fval=0.0,
        fmin=types.SimpleNamespace(edm=0.0),
        nfcn=1,
        values=[0.0],
        errors=[0.0],
    )
    assert graded.Status() == 3
    graded._report(["a"])
    assert "Invalid minimum - status = 3" in capsys.readouterr().out


def test_a_histogram_fills_from_arrays_as_pyroot_lets_it():
    h = ROOT.TH1D("h", "", 4, 0, 4)
    expect(
        (h.Fill(np.array([0.5, 1.5, 1.5])), -1),
        (h.GetBinContent(2), 2),
    )
    h.Fill(np.array([2.5]), np.array([3.0]))
    expect(
        (h.GetBinContent(3), 3),
        (h.GetEntries(), 4),
    )
    h2 = ROOT.TH2D("h2", "", 2, 0, 2, 2, 0, 2)
    h2.Fill(np.array([0.5, 1.5]), np.array([0.5, 0.5]))
    expect(
        (h2.GetBinContent(1, 1), 1),
        (h2.GetBinContent(2, 1), 1),
    )


def test_a_file_that_would_not_open_lists_nothing(capsys):
    ROOT.TFile("missing.root").ls()
    assert capsys.readouterr().out == ""


def test_std_streams_write_what_they_are_given(capsys):
    ROOT.std.cout << "a" << 1 << ROOT.std.endl
    ROOT.std.cerr << "b"
    ROOT.std.cout.flush()
    out = capsys.readouterr()
    expect(
        (out.out, "a1\n"),
        (out.err, "b"),
        (repr(ROOT.std.cout), "<std::cout>"),
    )


def test_histograms_take_roots_arguments_by_name():
    h = ROOT.TH1D("h", "t", nbinsx=20, xlow=0.0, xup=10.0)
    expect(
        (h.GetNbinsX(), 20),
        (h.GetXaxis().GetXmax(), 10),
    )
    v = ROOT.TH1D(name="v", title="t", nbinsx=3, xbins=[0.0, 1.0, 2.0, 5.0])
    assert v.GetXaxis().GetXmax() == 5
    two = ROOT.TH2F("two", "t", 10, 0, 1, nbinsy=4, ylow=-1, yup=1)
    expect(
        (two.GetNbinsY(), 4),
        (two.GetYaxis().GetXmin(), -1),
    )
    with pytest.raises(TypeError, match="no argument 'nbins'"):
        ROOT.TH1D("bad", "t", nbins=3)


def test_a_formula_may_use_a_formula_made_before_it():
    ROOT.TFormula("shape", "x*x")
    ROOT.TFormula("scaled", "[0]*x")
    uses = ROOT.TF1("uses", "[0] + [1]*shape + scaled", 0, 1)
    uses.SetParameters(1, 2, 3)
    expect(
        (uses.Eval(0.5), pytest.approx(1 + 2 * 0.25 + 3 * 0.5)),
        (uses.GetNpar(), 3),
        (ROOT.TF1("plain", "gaus", -1, 1).GetNpar(), 3),
    )


def test_a_file_hands_out_objects_by_subscript(tmp_path):
    with ROOT.TFile("s.root", "RECREATE") as out:
        ROOT.TH1D("h", "", 1, 0, 1)
        out.Write()
    with ROOT.TFile("s.root") as back:
        assert back["h"].GetNbinsX() == 1
        with pytest.raises(KeyError, match="has no object 'nope'"):
            back["nope"]


def test_a_fitter_set_up_first_minimises_from_its_parameters_settings():
    pytest.importorskip("iminuit")

    def bowl(xx):
        return (xx[0] - 3) ** 2 + 4 * (xx[1] + 1) ** 2 + 1

    fitter = ROOT.Fit.Fitter()
    with pytest.raises(ValueError, match="none was given"):
        fitter.FitFCN()
    fitter.SetFCN(ROOT.Math.Functor(bowl, 2), np.array([1.0, 1.0]))
    config = fitter.Config()
    settings = config.ParSettings(0)
    settings.SetStepSize(0.01)
    settings.SetName("a")
    settings.SetValue(2.0)
    config.ParSettings(1).SetLimits(-5, 5)
    assert (settings.StepSize(), settings.Name(), settings.Value(), config.NPar()) == (
        0.01, "a", 2.0, 2)
    assert fitter.FitFCN()
    assert fitter.Result().Parameter(0) == pytest.approx(3.0, abs=1e-2)  # MIGRAD's edm 0.01
    assert fitter.Result().ParName(0) == "a"
    config.ParSettings(0).Fix()
    assert config.ParamsSettings()[0].IsFixed()
    fitter.FitFCN()
    assert fitter.Result().Parameter(0) == pytest.approx(2.0)
    config.ParSettings(0).Release()
    assert not config.ParSettings(0).IsFixed()
    config.SetParamsSettings(2)
    assert config.ParSettings(1).Value() == 0.0


def test_minos_gives_the_errors_either_side_of_a_parameter():
    pytest.importorskip("iminuit")
    minimizer = ROOT.Math.Minimizer()
    minimizer.SetFunction(ROOT.Math.Functor(lambda v: (v[0] - 1) ** 2 + 2 * (v[1] - 3) ** 2, 2))
    minimizer.SetVariable(0, "x", -1.0, 0.01)
    minimizer.SetVariable(1, "y", 1.0, 0.01)
    assert minimizer.Minimize()
    low, high = np.zeros(1), np.zeros(1)
    expect(
        (bool(minimizer.GetMinosError(0, low, high)), True),
        (low[0], pytest.approx(-1.0, rel=1e-4)),
        (high[0], pytest.approx(1.0, rel=1e-4)),
        # 2*(y-3)^2 rises by one at 1/sqrt(2), as ROOT's MINOS finds.
        (bool(minimizer.GetMinosError(1, low, high)), True),
        (high[0], pytest.approx(np.sqrt(0.5), rel=1e-4)),
    )
    # A bare number is handed over by value: there is nowhere to write, and nothing is written.
    assert minimizer.GetMinosError(0, 0.0, 0.0)


def test_minos_says_so_when_it_finds_no_error():
    pytest.importorskip("iminuit")
    minimizer = ROOT.Math.Minimizer()
    # A linear direction never converges, so there is no minimum for MINOS to start from.
    minimizer.SetFunction(ROOT.Math.Functor(lambda v: (v[0] - 1) ** 2 + 1e-6 * v[1], 2))
    minimizer.SetVariable(0, "x", 0.0, 0.01)
    minimizer.SetVariable(1, "y", 0.0, 0.01)
    assert not minimizer.Minimize()
    low, high = np.array([7.0]), np.array([7.0])
    expect(
        (bool(minimizer.GetMinosError(1, low, high)), False),
        (low[0], 0.0),
        (high[0], 0.0),
    )
    minimizer._found = types.SimpleNamespace(
        valid=True,
        parameters=("a",),
        merrors={"a": types.SimpleNamespace(is_valid=False, lower=-2.0, upper=2.0)},
        minos=lambda name: None,
    )
    assert not minimizer.GetMinosError(0, low, high)
    assert (low[0], high[0]) == (0.0, 0.0)
    minimizer._found.merrors = {}  # MINOS says nothing at all of this parameter
    assert not minimizer.GetMinosError(0, low, high)
    # No minimisation done at all - as after Clear - is ROOT's False too, with zeros.
    minimizer.Clear()
    assert not minimizer.GetMinosError(0, low, high)
    assert not minimizer.GetMinosError(0)


def test_a_cleared_minimizer_keeps_its_variables_and_has_no_minimum():
    pytest.importorskip("iminuit")
    minimizer = ROOT.Math.Minimizer()
    minimizer.SetFunction(ROOT.Math.Functor(lambda v: (v[0] - 2) ** 2, 1))
    minimizer.SetVariable(0, "a", 0.0, 0.1)
    assert minimizer.Minimize()
    minimizer.Clear()
    expect(
        (minimizer.NDim(), 1),
        (minimizer.MinValue(), 0.0),
        (minimizer.Edm(), 0.0),
        (minimizer.NCalls(), 0),
        (minimizer.Status(), 0),
        (list(minimizer.X()), []),
        (list(minimizer.Errors()), []),
    )
    # The same minimiser takes a new function and runs again, as ROOT's test programs reuse one.
    minimizer.SetFunction(ROOT.Math.Functor(lambda v: (v[0] - 4) ** 2, 1))
    assert minimizer.Minimize()
    assert minimizer.X()[0] == pytest.approx(4.0, abs=1e-3)


def test_the_chi_square_named_as_a_template_makes_the_same_chi_square():
    """``ROOT::Fit::Chi2FCN<IMultiGenFunction>``: C++'s argument written out, and dropped."""
    data = ROOT.Fit.BinData()
    data.Add(1.0, 2.0, 0.5)
    function = ROOT.Math.WrappedMultiTF1(ROOT.TF1("f", "[0]*x", 0, 1))
    made = ROOT.Fit.Chi2FCN[ROOT.Math.IMultiGenFunction](data, function)
    assert isinstance(made, ROOT.Fit.Chi2Function)
    assert made.NDim() == 1


def test_the_genetic_minimizer_finds_a_minimum_inside_the_limits():
    minimizer = ROOT.Math.GeneticMinimizer()
    minimizer.SetFunction(
        ROOT.Math.Functor(lambda v: (v[0] - 2.0) ** 2 + (v[1] + 1.0) ** 2, 2)
    )
    minimizer.SetLimitedVariable(0, "a", 0.0, 0.1, -10.0, 10.0)
    minimizer.SetLimitedVariable(1, "b", 0.0, 0.1, -10.0, 10.0)
    assert minimizer.Minimize()
    assert list(minimizer.X()) == pytest.approx([2.0, -1.0], abs=0.5)
    assert minimizer.MinValue() < 0.5
    assert minimizer.NCalls() > 0
    expect((minimizer.Edm(), 0.0), (minimizer.Status(), 0), (minimizer.NDim(), 2))


def test_the_genetic_minimizer_searches_around_an_unbounded_variables_start():
    """A variable given no limits is searched within the spread of its step of its start."""
    minimizer = ROOT.Math.GeneticMinimizer()
    minimizer.SetFunction(ROOT.Math.Functor(lambda v: v[0] ** 2, 1))
    minimizer.SetVariable(0, "a", 3.0, 0.5)
    assert minimizer.Minimize()
    assert 2.5 <= minimizer.X()[0] <= 3.5
    assert minimizer.X()[0] == pytest.approx(2.5, abs=0.1)


def test_the_genetic_minimizers_settings_are_its_own_and_its_minimum_is_dropped():
    parameters = ROOT.Math.GeneticMinimizerParameters()
    parameters.fPopSize, parameters.fNsteps, parameters.fSeed = 4, 2, 7
    minimizer = ROOT.Math.GeneticMinimizer()
    minimizer.SetParameters(parameters)
    assert minimizer.Parameters() is parameters
    expect((minimizer.MinValue(), 0.0), (minimizer.Status(), 3), (list(minimizer.X()), []))
    # No function at all, then a function but no variable: nothing to minimise either way.
    assert not minimizer.Minimize()
    minimizer.SetFunction(ROOT.Math.Functor(lambda v: v[0] ** 2, 1))
    assert not minimizer.Minimize()
    minimizer.SetVariable(0, "a", 1.0, 0.1)
    assert minimizer.Minimize()
    minimizer.Clear()
    expect((minimizer.MinValue(), 0.0), (minimizer.NCalls(), 0), (minimizer.Status(), 3),
           (list(minimizer.X()), []))  # fmt: skip


def test_a_fit_handed_its_function_starts_where_that_functions_parameters_stand():
    """``Fit(d, f)``: the model given with the data, as ROOT's own fit programs call it."""
    data = ROOT.Fit.BinData(1, 1)
    for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
        data.Add(x, 1.0 + 2.0 * x, 0.1)
    fitter = ROOT.Fit.Fitter()
    assert fitter.Fit(data, ROOT.Math.WrappedMultiTF1(ROOT.TF1("g", "[0] + [1]*x", -3, 3)))
    result = fitter.Result()
    assert (result.Parameter(0), result.Parameter(1)) == pytest.approx((1.0, 2.0), abs=1e-4)


def test_a_fit_with_neither_a_model_nor_a_function_says_which_it_wants():
    with pytest.raises(ValueError, match="fits the model SetFunction gave it"):
        ROOT.Fit.Fitter().Fit(ROOT.Fit.BinData())


def test_the_genetic_minimizers_seed_and_its_cycles_are_the_search_it_makes():
    """``GeneticMinimizer(seed)`` and ``SetRandomSeed``: the same populations, drawn again."""
    seeded = ROOT.Math.GeneticMinimizer(111)
    assert seeded.Parameters().fSeed == 111
    seeded.SetRandomSeed(7)
    assert seeded.Parameters().fSeed == 7
    # Two searches from the same seed and the same starting point are the same search.
    runs = []
    for _ in range(2):
        minimizer = ROOT.Math.GeneticMinimizer(3)
        minimizer.SetFunction(ROOT.Math.Functor(lambda v: (v[0] - 1.0) ** 2 + v[1] ** 2, 2))
        minimizer.SetLimitedVariable(0, "a", 0.0, 0.1, -5.0, 5.0)
        minimizer.SetLimitedVariable(1, "b", 0.0, 0.1, -5.0, 5.0)
        assert minimizer.Minimize()
        runs.append((list(minimizer.X()), minimizer.MinValue()))
    assert runs[0] == runs[1]
    assert runs[0][0] == pytest.approx([1.0, 0.0], abs=0.2)


def test_the_gsl_solvers_take_a_c_function_and_the_pointer_gsl_hands_it():
    """``ROOT::Math::Roots::Brent`` and its siblings are GSL's own algorithms: each takes a
    ``GSLFuncPointer`` - ``double f(double x, void *p)`` - with the ``void *p`` GSL passes
    to every call, and answers ``Name`` with the name GSL gives that algorithm."""
    def root(x, params):
        return x - params

    for kind, name in ((ROOT.Math.Roots.Brent, "brent"),
                       (ROOT.Math.Roots.Bisection, "bisection"),
                       (ROOT.Math.Roots.FalsePos, "falsepos")):
        solver = kind()
        solver.SetFunction(root, 0.25, 0.0, 1.0)
        expect(
            (bool(solver.Solve()), True),
            (solver.Root(), pytest.approx(0.25)),
            (solver.Name(), name),
            (bool(solver.Iterations() > 0), True),
        )


def test_a_gsl_solver_that_takes_the_derivative_uses_it_or_the_pair_that_gives_both():
    """``Roots::Newton``/``Secant``/``Steffenson`` are ``GSLRootFinderDeriv``: they take the
    function, its derivative, and the ``fdf`` GSL can be given instead - a function that
    writes both through the two pointers it is handed."""
    def square(x, params):
        return x * x - 5

    def slope(x, params):
        return 2 * x

    def both(x, params, y, dy):
        y[0], dy[0] = x * x - 5, 2 * x

    for kind in (ROOT.Math.Roots.Newton, ROOT.Math.Roots.Secant, ROOT.Math.Roots.Steffenson):
        solver = kind()
        solver.SetFunction(square, slope, None, None, 5.0)
        expect(
            (bool(solver.Solve()), True),
            (solver.Root(), pytest.approx(np.sqrt(5.0))),
            (solver.Name(), kind.algorithm),
        )
    paired = ROOT.Math.Roots.Newton()
    paired.SetFunction(square, None, both, None, 5.0)
    assert paired.Solve()
    assert paired.Root() == pytest.approx(np.sqrt(5.0), rel=1e-12)
    # ``exp`` has no root, so Newton's step never shrinks: it takes the steps it is given.
    stuck = ROOT.Math.Roots.Secant()
    stuck.SetFunction(np.exp, np.exp, None, None, 1.0)
    assert stuck.Solve(3)
    assert stuck.Iterations() == 3


def test_the_gsl_solver_base_is_root_maths_own_named_class():
    """``GSLRootFinder`` and ``GSLRootFinderDeriv`` are ``ROOT::Math``'s, not only the
    namespace ``Roots`` holds, because ROOT's own macros name them there."""
    assert ROOT.Math.Roots is not None
    assert issubclass(ROOT.Math.Roots.Brent, ROOT.Math.GSLRootFinder)
    assert issubclass(ROOT.Math.Roots.Newton, ROOT.Math.GSLRootFinderDeriv)
    assert ROOT.Math.GSLRootFinder().Name() == "GSL"
    assert issubclass(ROOT.Math.GSLRootFinder, ROOT.Math.IRootFinderMethod)
    assert issubclass(ROOT.Math.GSLRootFinderDeriv, ROOT.Math.IRootFinderMethod)
