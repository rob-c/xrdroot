"""``RooFit::bindFunction`` and ``bindPdf``: functions and ``TF1``s as RooFit nodes.

What ROOT 6.40 prints of each binding is checked line for line - a function
a macro binds by the name RooFit registered for it, one from Python by its
address - and the values are the bound function's own at the variables'.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.bindings import bind_function, bind_pdf


def _printed(node: object, capsys: pytest.CaptureFixture[str]) -> str:
    node.Print()  # type: ignore[attr-defined]
    return re.sub(r"0x[0-9a-f]+", "0x<addr>", capsys.readouterr().out.strip())


def test_a_function_bound_from_python_is_shown_by_its_address(
    capsys: pytest.CaptureFixture[str],
) -> None:
    x = ROOT.RooRealVar("x", "x", -3, 3)
    erf = ROOT.RooFit.bindFunction("erf", ROOT.TMath.Erf, x)
    assert _printed(erf, capsys) == (
        "RooCFunction1Binding<double,double>::erf[ function=(0x<addr>) x=x ] = 0"
    )
    x.setVal(0.5)
    assert erf.getVal() == math.erf(0.5)
    assert list(erf.compute({"x": np.array([0.1, 0.2])})) == [math.erf(0.1), math.erf(0.2)]


def test_a_function_a_macro_binds_is_shown_by_roofits_name_for_it_and_its_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    x, y = ROOT.RooRealVar("x", "x", 0.1, 0.9), ROOT.RooRealVar("y", "y", 0.5, 0.1, 3)
    scope = {"__name__": "__cint__", "ROOT": ROOT, "bind": bind_function, "x": x, "y": y}
    exec("made = bind('c', ROOT.TMath.Gamma, x); beta = bind('n', ROOT.TMath.Beta, x, y)", scope)
    assert _printed(scope["made"], capsys) == (
        "RooCFunction1Binding<double,double>::c[ function=TMath::Gamma z=x ] = 1.77245"
    )
    assert _printed(scope["beta"], capsys) == (
        "RooCFunction2Binding<double,double,double>::n[ function=TMath::Beta p=x q=y ] = 3.14159"
    )
    exec("other = bind('q', ROOT.Math.beta_pdf, x, y, y)", scope)
    assert "function=(0x<addr>) x=x y=y z=y" in _printed(scope["other"], capsys)


def test_a_bound_density_is_normalised_over_its_variable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    x2 = ROOT.RooRealVar("x2", "x2", 0, 0.999)
    a, b = ROOT.RooRealVar("a", "a", 5, 0, 10), ROOT.RooRealVar("b", "b", 2, 0, 10)
    beta = ROOT.RooFit.bindPdf("beta", ROOT.Math.beta_pdf, x2, a, b)
    assert _printed(beta, capsys) == (
        "RooCFunction3PdfBinding<double,double,double,double>::beta[ function=(0x<addr>) "
        "x=x2 y=a z=b ] = 0.934689"
    )
    # beta_pdf integrates to one over [0, 1]; to 0.999 it misses 1 - I(0.999; 5, 2) of it.
    assert beta.getVal(ROOT.RooArgSet(x2)) == pytest.approx(0.934689 / 0.99998, rel=1e-5)


def test_a_tf1_tf2_or_their_parameters_are_bound_as_root_binds_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    x3 = ROOT.RooRealVar("x3", "x3", 0.01, 20)
    rfa1 = ROOT.RooFit.bindFunction(ROOT.TF1("fa1", "sin(x)/x", 0, 10), x3)
    assert _printed(rfa1, capsys) == (
        "RooTFnBinding::fa1[ TFn={fa1=sin(x)/x} obs=(x3) params=() ] = -0.0547936"
    )
    u, v = ROOT.RooRealVar("u", "u", 1.0, 0, 3), ROOT.RooRealVar("v", "v", 2.0, 0, 3)
    scale = ROOT.RooRealVar("scale", "scale", 3.0, 0, 10)
    plane = ROOT.RooFit.bindPdf(ROOT.TF2("plane", "[0]*x*y", 0, 3, 0, 3), [u, v], [scale])
    assert plane.ClassName() == "RooTFnPdfBinding" and plane.getVal() == 6.0
    assert list(plane.compute({"u": np.array([1.0, 2.0])})) == [6.0, 12.0]


def test_a_function_of_more_than_four_numbers_is_refused() -> None:
    xs = [ROOT.RooRealVar(f"x{i}", "x", 0, 1) for i in range(5)]
    with pytest.raises(TypeError, match="one to four variables"):
        bind_pdf("five", lambda *values: 1.0, *xs)
