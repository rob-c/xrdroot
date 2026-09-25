"""Functions read from the files ROOT wrote, checked against ROOT's own values, and written back.

ROOT samples a function over its range into ``fSave`` when it writes one it
cannot write as a formula, and sometimes when it can: ``tgme.root``'s
``pol1`` fit carries a hundred and one of ROOT's own evaluations of it, and
``tformula.root``'s ``func2`` is ``[0] + [1]*x`` as C++, sampled. Evaluating
the formulas here must give those numbers. Writing goes the other way: a
``TF1`` ROOT wrote is written back byte for byte, and a histogram with its
fits attached reads back with them.
"""

from __future__ import annotations

import io
import math
import pathlib

import numpy as np
import pytest

from xrdroot import Function, Graph, Histogram, UnsupportedFeatureError, create, open_root
from xrdroot.kinds import dress
from xrdroot.writer import _payload

DATA = pathlib.Path(__file__).parent / "data"
#: Both vintages of the same functions: ROOT 6.24's TFormula 13, and 6.34's 14.
VINTAGES = ("tformula", "tformula-v14")


def opened(name):
    return open_root(str(DATA / f"{name}.root"))


def sampled_at(function):
    """The points ROOT sampled a function at, and what it saved there."""
    save = np.asarray(function.members["fSave"])
    low, high = save[-2], save[-1]
    points = low + (high - low) / (len(save) - 3) * np.arange(len(save) - 2)
    return points, save[:-2]


def written(**objects) -> bytes:
    buf = io.BytesIO()
    with create(buf) as out:
        for name, obj in objects.items():
            out[name] = obj
    return buf.getvalue()


# -- reading ---------------------------------------------------------------------------------


@pytest.mark.parametrize("vintage", VINTAGES)
def test_every_tf1_root_wrote_reads_as_a_function(vintage):
    with opened(vintage) as root:
        line, code, convolution, total = (root[f"func{n}"] for n in (1, 2, 3, 4))
        assert (line.formula, line.parameter_names, list(line.parameters)) == (
            "[p0]+[p1]*x",
            ("p0", "p1"),
            [10.0, 20.0],
        )
        assert line.fit_result["chi2"] == 0.2 and line.fit_result["npfits"] == 101
        assert code.formula is None and code.parameter_names == ("p0", "p1")
        assert convolution.range == (0.0, 5.0) and convolution.npar == 4
        assert total.npar == 6
        fconv = root["fconv"]
        assert fconv["fFunction2"].parameter_names == ("Constant", "Mean", "Sigma")
        assert fconv["fFunction2"].fixed == (True, False, False)
        assert [held.name for held in root["fnorm"]["fFunctions"]] == ["func1", "func2"]


def test_a_formula_root_fitted_evaluates_to_the_values_root_saved_for_it():
    with opened("tgme") as root:
        fit = root["mg"].members["fFunctions"][0]
    assert (fit.name, fit.formula) == ("pol1", "([p0]+[p1]*x)")
    points, theirs = sampled_at(fit)
    np.testing.assert_allclose(fit.evaluate(points), theirs, rtol=1e-12, atol=0)
    result = fit.fit_result
    assert (result["ndf"], result["npfits"]) == (8, 10)
    assert result["chi2"] == pytest.approx(36.43636834438324, rel=1e-15)
    assert list(result["errors"]) == pytest.approx([0.33687817, 0.18821588], rel=1e-7)


@pytest.mark.parametrize("vintage", VINTAGES)
def test_the_formula_of_a_function_root_compiled_gives_the_values_root_sampled(vintage):
    with opened(vintage) as root:
        code = root["func2"]
    points, theirs = sampled_at(code)
    same = Function("same", "[0] + [1]*x", range=code.range, parameters=code.parameters)
    np.testing.assert_allclose(same.evaluate(points), theirs, rtol=1e-12, atol=0)
    np.testing.assert_allclose(code.evaluate(points), theirs, rtol=1e-12)  # read straight back


def test_a_function_of_code_is_the_straight_line_between_its_samples():
    with opened("tformula") as root:
        convolution = root["func3"]
    points, theirs = sampled_at(convolution)
    np.testing.assert_allclose(convolution.evaluate(points), theirs, rtol=1e-12)
    halfway = 0.5 * (points[:-1] + points[1:])
    np.testing.assert_allclose(convolution.evaluate(halfway), 0.5 * (theirs[:-1] + theirs[1:]))
    assert convolution(-1.0) == 0.0 and convolution(5.5) == 0.0
    assert math.isnan(convolution(math.nan))


def test_a_normalised_function_divides_by_the_integral_root_saved():
    with opened("tformula") as root:
        line = root["func1"]
        saved_normalised = root["fnorm"]["fFunctions"][1]
    assert line.normalized and line.members["fNormIntegral"] == pytest.approx(1100)
    assert line(1.0) == pytest.approx(30 / 1100, rel=1e-15)
    assert line.integral(0, 10) == pytest.approx(1.0, rel=1e-12)
    # ROOT saved the normalised values, and they are not divided again.
    assert saved_normalised(1.0) == pytest.approx(30 / 1100, rel=1e-12)


def test_the_functions_an_efficiency_carries_come_back_as_functions():
    with opened("tconfidence-level") as root:
        (fit,) = root["eff"].members["fFunctions"]
    assert (fit.name, fit.title, fit.parameter_names) == (
        "f1",
        "gaus",
        ("Constant", "Mean", "Sigma"),
    )


def test_a_function_whose_code_is_gone_and_saved_nothing_is_refused_when_evaluated():
    with opened("tformula") as root:
        members = root["func2"].members
    members["fSave"] = np.zeros(0)
    gone = dress("TF1", members)
    with pytest.raises(UnsupportedFeatureError, match="holds no saved values"):
        gone(1.0)
    members["fSave"] = np.array([1.0, 2.0, 3.0, 1.0, 1.0])
    with pytest.raises(UnsupportedFeatureError, match="bins of the histogram"):
        dress("TF1", members)(1.0)


def test_a_formula_this_cannot_evaluate_still_reads_and_falls_back_on_its_samples():
    f = Function("f", "[0]*x", parameters=[2])
    f.members["fFormula"]["fFormula"] = "ROOT::Math::chisquared_pdf(x,[p0])"
    kept = dress("TF1", f.members)
    with pytest.raises(UnsupportedFeatureError, match=r"chisquared_pdf.*no saved values"):
        kept(1.0)
    kept.members["fSave"] = np.array([0.0, 2.0, 0.0, 1.0])
    assert dress("TF1", kept.members)(0.5) == 1.0


def test_members_read_in_other_shapes_are_made_arrays_to_change_in_place():
    f = Function("f", "[0]+[1]*x", parameters=[1, 2])
    f.members["fFormula"]["fClingParameters"] = [1.0, 2.0]
    f.members["fParErrors"] = [0.5]
    f.members["fFormula"]["fParams"] = {"p0": 0, "p1": 1, "ghost": 7}
    again = dress("TF1", f.members)
    assert isinstance(again.parameters, np.ndarray) and again(1.0) == 3.0
    assert list(again.parameter_errors) == [0.5, 0.0]
    assert again.parameter_names == ("p0", "p1")
    bare = dress("TF1", {**Function.from_callable("c", lambda x, p: x, 0).members, "fParams": None})
    assert list(bare.parameters) == [] and bare.parameter_names == ()


def test_a_layout_older_than_root_6_stays_the_dictionary_it_was_read_as():
    old = {"TFormula": {"fNpar": 1}, "fNpx": 100}
    assert dress("TF1", old) is old
    also_old = {"fExpr": ["x"]}
    assert dress("TFormula", also_old) is also_old
    assert dress("TF1", "TF1") == "TF1"  # a class stepped over is its name


def test_a_tformula_on_its_own_is_a_function_too():
    formula = dress("TFormula", Function("g", "gaus", parameters=[1, 0, 2]).members["fFormula"])
    assert formula.classname == "TFormula" and formula.name == "g"
    assert formula(2.0) == pytest.approx(math.exp(-0.5))
    assert formula.range == (0.0, 1.0)
    with open_root(io.BytesIO(written(formula=formula))) as back:
        again = back["formula"]
    assert again.classname == "TFormula" and again(2.0) == formula(2.0)
    assert formula.copy("h").name == "h"


def test_a_tf2_and_a_tf3_come_back_from_their_members():
    plane = Function("plane", "[0]*x+y", range=((0, 1), (0, 2)), parameters=[3])
    again = dress("TF2", plane.members)
    assert again.range == ((0.0, 1.0), (0.0, 2.0)) and again(1.0, 1.0) == 4.0
    cube = dress("TF3", Function("cube", "x*y*z").members)
    assert cube.dimensions == 3 and cube(1, 2, 3) == 6
    grid = Function.from_callable("grid", lambda p, q: p[:, 0], 0, dimensions=2)
    grid._model = None  # as read from a file, without the code
    with pytest.raises(UnsupportedFeatureError, match="a grid this reader does not interpolate"):
        grid(1.0, 1.0)


# -- writing ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["func1", "func2", "func3"])
def test_a_tf1_root_wrote_is_written_back_byte_for_byte(name):
    with opened("tformula") as root:
        theirs = root._key(name).payload(root._source)
        classname, ours, _used = _payload(root[name])
    assert classname == "TF1"
    if name == "func1":
        # ROOT wrote fAllParametersSetted as 0x99, a bool it never initialised;
        # read as true, it is written as the one ROOT means by true.
        at = theirs.index(b"\x40\x34" + bytes(6) + b"\x99") + 8  # after 20.0, the last parameter
        theirs = theirs[:at] + b"\x01" + theirs[at + 1 :]
    assert ours == theirs


def test_a_file_describes_the_function_classes_exactly_as_the_donor_does():
    with opened("tgme") as root:
        theirs = root._source.streamers()
    with open_root(io.BytesIO(written(f=Function("f", "gaus")))) as back:
        ours = back._source.streamers()
    for classname in ("TF1", "TFormula", "TF1Parameters"):
        for member, described in ours[classname].items():
            other = theirs[classname][member]
            assert (described.title, described.stype, described.typename) == (
                other.title,
                other.stype,
                other.typename,
            )
    assert "TF1AbsComposition" in ours


def test_a_histogram_with_its_fits_is_written_and_read_back_with_them():
    with opened("tgme") as root:
        root_fit = root["mg"].members["fFunctions"][0]
    h = Histogram.book("h", (20, -5, 5))
    h.fill(np.linspace(-4, 4, 200))
    fit = Function("fit", "gaus", range=(-5, 5), parameters=[10, 0, 2])
    fit.parameter_errors = [1, 0.1, 0.05]
    fit.fit_result = {"chi2": 12.5, "ndf": 17, "npfits": 20}
    model = Function.from_callable(
        "model", lambda x, p: p[0] * x * x, 1, range=(0, 2), parameters=[3]
    )
    h.attach(fit)
    h.functions.append(root_fit)
    h.attach(model)
    with open_root(io.BytesIO(written(h=h))) as back:
        again = back["h"].functions
    assert [f.name for f in again] == ["fit", "pol1", "model"]
    assert again[0].fit_result["chi2"] == 12.5 and list(again[0].parameter_errors) == [1, 0.1, 0.05]
    np.testing.assert_array_equal(again[1].parameters, root_fit.parameters)
    assert again[2].formula is None
    assert again[2](1.0) == pytest.approx(3.0) and again[2](1.01) == pytest.approx(
        3 * 1.01**2, rel=1e-3
    )


def test_a_graph_with_a_fit_is_written_and_read_back_with_it():
    graph = Graph.new("g", [1, 2, 3], [2, 4, 6])
    assert graph.functions == []
    graph.attach(Function("line", "pol1", range=(0, 4), parameters=[0, 2]))
    with open_root(io.BytesIO(written(g=graph))) as back:
        (line,) = back["g"].functions
    assert line(3.0) == 6.0


def test_a_list_of_functions_with_something_else_in_it_is_refused_by_what_that_is():
    h = Histogram.book("h", (2, 0, 1))
    h.attach(Function("f", "x"))
    h.functions.append("a box of statistics")
    with pytest.raises(UnsupportedFeatureError, match="holding str"):
        written(h=h)


def test_a_function_this_has_no_donor_for_is_refused_by_name():
    with pytest.raises(UnsupportedFeatureError, match="a TF2 is not a class"):
        written(f=Function("f", "x*y"))
    linear = Function("f", "x")
    linear.members["fFormula"]["fLinearParts"] = ["x"]
    with pytest.raises(UnsupportedFeatureError, match="the parts of a linear fit"):
        written(f=linear)


def test_the_list_of_functions_is_made_when_an_object_has_none():
    from xrdroot.function.attached import listed

    core = {"fFunctions": None}
    assert listed(core) == [] and core["fFunctions"] == []
    core = {"fFunctions": ("a",)}
    assert listed(core) == ["a"]
