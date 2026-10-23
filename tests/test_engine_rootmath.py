"""ROOT math names retain their identity while engine formulas initialize independently."""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import xrdroot
from xrdroot.function import Function, legendres, mathformula, rootmath
from xrdroot.function.library import CALLS


@pytest.mark.parametrize("name", rootmath.__all__)
def test_scalar_math_functions_keep_their_pyroot_compatibility_identity(name):
    legacy = importlib.import_module("xrdroot.pyroot.core.rmath")
    expected = getattr(rootmath, name)
    assert getattr(legacy, name) is expected
    assert name in legacy.__all__
    assert pickle.loads(f"cxrdroot.pyroot.core.rmath\n{name}\n.".encode()) is expected


@pytest.mark.parametrize("name", ["legendre", "assoc_legendre", "sph_legendre"])
def test_the_old_legendre_module_reexports_the_same_scalar_functions(name):
    legacy = importlib.import_module("xrdroot.pyroot.core.legendres")
    assert getattr(legacy, name) is getattr(legendres, name) is getattr(rootmath, name)


def test_pyroot_keeps_its_configuration_and_vector_wrappers_in_the_legacy_namespace():
    legacy = importlib.import_module("xrdroot.pyroot.core.rmath")
    genvector = importlib.import_module("xrdroot.pyroot.core.genvector")
    defaults = importlib.import_module("xrdroot.fit.defaults")
    assert legacy.MinimizerOptions._defaults is defaults.DEFAULTS
    assert legacy.MinimizerOptions.__module__ == "xrdroot.pyroot.core.rmath"
    assert legacy.MathMoreLibrary.__module__ == "xrdroot.pyroot.core.rmath"
    for name in genvector.__all__:
        assert getattr(legacy, name) is getattr(genvector, name)
    assert legacy.__all__ == sorted(set(legacy.__all__))
    assert {"MinimizerOptions", "MathMoreLibrary", "WrappedMultiTF1"} <= set(legacy.__all__)


def test_legacy_formula_helpers_share_the_engine_registry_without_replacing_existing_calls():
    legacy = importlib.import_module("xrdroot.pyroot.core.mathformula")
    for name in ("_arity", "_functions", "_vectorised", "register"):
        assert getattr(legacy, name) is getattr(mathformula, name)
    before = dict(CALLS)
    legacy.register()
    assert CALLS.keys() == before.keys()
    assert all(CALLS[name] is call for name, call in before.items())
    expected = {name for name in rootmath.__all__ if name[:1].islower()}
    assert mathformula._functions().keys() == expected
    assert {f"ROOT::Math::{name}" for name in expected} <= CALLS.keys()


def test_registered_scalar_formulas_broadcast_and_keep_the_scalar_function_values():
    model = Function("normal", "ROOT::Math::normal_pdf(x, [0], [1])", range=(-5, 5))
    model.set_parameters(1.5, 0.2)
    coordinates = np.array([-0.5, 0.0, 0.5])
    expected = [rootmath.normal_pdf(float(value), 1.5, 0.2) for value in coordinates]
    assert model(coordinates).tolist() == expected
    polynomial = Function("legendre", "ROOT::Math::legendre([0], x)", range=(-1, 1))
    polynomial.set_parameters(3)
    assert polynomial(coordinates).tolist() == [legendres.legendre(3, x) for x in coordinates]


def test_formula_registration_and_roofit_normalization_work_without_pyroot_or_the_translator(
    tmp_path,
):
    program = """
        import importlib
        import math
        import sys

        blocked = {"pyroot", "cint", "canvas", "plot", "tree", "interp", "file", "writer"}

        class BlockAdapters:
            def find_spec(self, fullname, path=None, target=None):
                parts = fullname.split(".")
                if parts[0] == "xrdroot" and len(parts) > 1 and parts[1] in blocked:
                    raise AssertionError("The engine imported " + fullname)

        sys.meta_path.insert(0, BlockAdapters())
        from xrdroot.function import Function, mathformula, rootmath
        from xrdroot.function.library import CALLS
        from xrdroot.roofit.collections import RooArgSet
        from xrdroot.roofit.pdfs.chisquare import RooNonCentralChiSquare
        from xrdroot.roofit.pdfs.gamma import RooGamma
        from xrdroot.roofit.variables import RooRealVar

        expected = {"ROOT::Math::" + name for name in rootmath.__all__ if name[:1].islower()}
        assert expected <= CALLS.keys()
        calls = dict(CALLS)
        mathformula.register()
        assert all(CALLS[name] is call for name, call in calls.items())
        function = Function("f", "ROOT::Math::normal_pdf(x, 1.5, 0.2)", range=(-5, 5))
        assert float(function(0.5)) == rootmath.normal_pdf(0.5, 1.5, 0.2)
        polynomial = Function("p", "ROOT::Math::legendre(3, x)", range=(-1, 1))
        assert float(polynomial(0.4)) == rootmath.legendre(3, 0.4)
        x = RooRealVar("x", "x", 2.0, 0.0, 20.0)
        k = RooRealVar("k", "k", 3.0)
        zero = RooRealVar("zero", "zero", 0.0)
        chi = RooNonCentralChiSquare("chi", "chi", x, k, zero)
        assert chi.getVal() == rootmath.chisquared_pdf(2.0, 3.0)
        assert math.isfinite(chi.getVal(RooArgSet(x)))
        shape = RooRealVar("shape", "shape", 2.5)
        scale = RooRealVar("scale", "scale", 1.5)
        gamma = RooGamma("gamma", "gamma", x, shape, scale, zero)
        assert math.isfinite(gamma.getVal(RooArgSet(x)))
        assert all("xrdroot." + name not in sys.modules for name in blocked)
        """
    source = str(Path(xrdroot.__file__).parent.parent)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((source, environment.get("PYTHONPATH", "")))
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
