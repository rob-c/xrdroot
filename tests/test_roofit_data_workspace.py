"""RooFit's workspace and its factory language, held to what ROOT 6.40 printed.

The import messages, the ``Print`` of a workspace and the objects the
factory builds were printed by ROOT itself through PyROOT for the same
calls. The engine's workspace holds the objects it is given rather than
clones of them, so where ROOT's answer depends on the clone the test says so
and pins only what the two share.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.factory import Factory, split
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.messages import service
from xrdroot.roofit.pdfs.basic import RooGaussian, ref
from xrdroot.roofit.variables import RooRealVar
from xrdroot.roofit.workspace import RooWorkspace


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


class Named:
    """Something a workspace keeps that is neither a node nor data: a ``TNamed``."""

    def GetName(self) -> str:
        return "note"


def gaussian() -> tuple[RooGaussian, RooRealVar, RooRealVar, RooRealVar]:
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "m", 0, -1, 1)
    s = RooRealVar("s", "s", 1, 0.1, 5)
    return RooGaussian("g", "g", x, m, s), x, m, s


def test_importing_a_model_imports_every_node_saying_so_as_root_does(capsys: Any) -> None:
    """``import(g)`` takes in the density and what it is made of, each named in ROOT's words."""
    w = RooWorkspace("w", "my workspace")
    g, x, *_ = gaussian()
    w.Import(g)
    w.Import(g)  # the second time, nothing new
    assert capsys.readouterr().out == (
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing RooGaussian::g\n"
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing RooRealVar::x\n"
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing RooRealVar::m\n"
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing RooRealVar::s\n"
    )
    assert (w.var("x") is x, w.pdf("g") is g, w["m"].GetName(), w.arg("nothing")) == (
        False,
        False,
        "m",
        None,
    )
    assert w.pdf("g").servers()[0] is w.var("x")  # the copy is made of the workspace's copies
    x.setVal(3.0)  # rf510: what is changed outside after the import stays outside
    assert w.var("x").getVal() == 0.0


def test_importing_a_dataset_and_renaming_it_is_said_as_root_says_it(capsys: Any) -> None:
    """``import(d)`` and ``import(o, Rename("d2"))``: ROOT's lines, and the data found by name."""
    w = RooWorkspace("w")
    x = RooRealVar("x", "x", -10, 10)
    d = RooDataSet("d", "d", x)
    d.add_columns({"x": [1.0]})
    w.Import(d, "a string is not an option")
    other = RooDataSet("o", "o", x)
    w.Import(other, Rename="d2")
    assert capsys.readouterr().out == (
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing dataset d\n"
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing dataset o\n"
        "[#1] INFO:ObjectHandling -- RooWorkSpace::import(w) changing name of dataset from  o to "
        "d2\n"
    )
    assert (w.data("d") is d, w.embeddedData("d2").GetName(), w.obj("d") is d) == (
        True,
        "d2",
        True,
    )
    assert ([one.GetName() for one in w.allData()], w.var("x").GetName()) == (["d", "d2"], "x")


def test_a_collection_is_imported_one_member_at_a_time(capsys: Any) -> None:
    """A set, a list or a Python list is each of its members imported; ``Silence`` quiets it."""
    w = RooWorkspace("w")
    c = RooCategory("c", "c")
    c.defineType("A")
    x2 = RooRealVar("x2", "x2", 0, 1)
    w.Import(x2, Silence=True)
    assert w.Import(RooArgSet([c, x2])) is False
    w.Import([RooRealVar("y", "y", 0, 1)], Silence=True)
    w.Import((ref(2.0),))
    assert capsys.readouterr().out == (
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing RooCategory::c\n"
    )
    assert sorted(one.GetName() for one in w.components()) == ["2", "c", "x2", "y"]


def test_an_object_that_is_not_a_node_is_kept_by_name() -> None:
    """A ``TNamed`` or the like is kept aside and handed back by ``obj`` and ``genobj``."""
    w = RooWorkspace("w")
    note = Named()
    w.Import(note)
    assert (w.obj("note") is note, w.genobj("note") is note, w["note"] is note) == (
        True,
        True,
        True,
    )
    with pytest.raises(KeyError, match="workspace w has no object called 'nothing'"):
        w["nothing"]


def test_the_kinds_of_contents_are_listed_apart() -> None:
    """``allVars``, ``allCats``, ``allPdfs``, ``allFunctions``: ROOT's lists for the same model."""
    w = RooWorkspace("w")
    g, x, m, _ = gaussian()
    c = RooCategory("c", "c")
    c.defineType("A")
    f = RooFormulaVar("f", "f", "x*m+1", [x, m])
    for one in (g, c, f, ref(2.0)):
        w.Import(one, Silence=True)
    names = [sorted(one.GetName() for one in found)
             for found in (w.allVars(), w.allCats(), w.allPdfs(), w.allFunctions())]  # fmt: skip
    assert names == [["m", "s", "x"], ["c"], ["g"], ["f"]]
    assert (w.function("f").GetName(), w.cat("c").GetName(), w.catfunc("c") is w.cat("c")) == (
        "f",
        "c",
        True,
    )
    assert w.function("f").servers()[0] is w.var("x")  # imported after g: its x, not a new one


def test_a_workspace_prints_each_kind_of_content_under_its_heading(capsys: Any) -> None:
    """``Print``: variables, densities, functions, datasets and named sets, laid out as ROOT's."""
    w = RooWorkspace("w", "my workspace")
    g, x, m, s = gaussian()
    w.Import(g, Silence=True)
    w.factory("expr::f('x*m+1', x, m)")
    d = RooDataSet("d", "d", x)
    w.Import(d)
    w.defineSet("obs", "x")
    w.defineSet("pars", RooArgSet([m, s]))
    capsys.readouterr()
    w.Print()
    assert capsys.readouterr().out == (
        "\nRooWorkspace(w) my workspace contents\n\n"
        "variables\n---------\n(m,s,x)\n\n"
        "p.d.f.s\n-------\nRooGaussian::g[ x=x mean=m sigma=s ] = 1\n\n"
        'functions\n--------\nRooFormulaVar::f[ actualVars=(x,m) formula="x*m+1" ] = 1\n\n'
        "datasets\n--------\nRooDataSet::d(x)\n\n"
        "named sets\n----------\nobs:(x)\npars:(m,s)\n\n"
    )


def test_an_empty_workspace_prints_its_heading_alone(capsys: Any) -> None:
    """A workspace made with a name alone is titled by it: ROOT's ``w2 contents``."""
    w2 = RooWorkspace("w2")
    w2.Print()
    assert capsys.readouterr().out == "\nRooWorkspace(w2) w2 contents\n\n"
    assert (w2.GetName(), w2.GetTitle(), w2.ClassName()) == ("w2", "w2", "RooWorkspace")


def test_named_sets_are_made_from_names_or_objects_and_extended_by_name() -> None:
    """``defineSet``, ``extendSet`` and ``set`` keep the workspace's own members."""
    w = RooWorkspace("w")
    g, _, m, _ = gaussian()
    w.Import(g, Silence=True)
    assert w.defineSet("obs", "x,nothing") is False
    assert w.defineSet("pars", [m, RooRealVar("free", "free", 0, 1)]) is False
    assert w.extendSet("more", "s") is False
    w.extendSet("pars", "nope")  # nothing of that name: the set is left as it was
    assert [one.GetName() for one in w.set("obs")] == ["x"]
    assert [one.GetName() for one in w.set("pars")] == ["m", "free"]
    assert [one.GetName() for one in w.set("more")] == ["s"]
    assert w.set("none") is None


def test_a_snapshot_saves_parameter_values_and_loads_them_back() -> None:
    """``saveSnapshot`` then ``loadSnapshot``: ROOT's True and the saved value again."""
    w = RooWorkspace("w")
    g, _, m, _ = gaussian()
    w.Import(g, Silence=True)
    assert w.saveSnapshot("snap", "m,s,nothing") is False
    w.saveSnapshot("listed", RooArgSet([m]))
    w.var("m").setVal(0.5)
    loaded = (w.loadSnapshot("snap"), w.var("m").getVal(), w.loadSnapshot("none"))
    assert loaded == (True, 0.0, False)
    saved = w.getSnapshot("listed")
    assert ([one.GetName() for one in saved], saved.find("m").getVal()) == (["m"], 0.0)
    assert len(w.getSnapshot("none")) == 0


def test_a_snapshot_importing_values_takes_the_given_parameters_not_the_workspaces() -> None:
    """``saveSnapshot(name, params, true)``, as ``rf510_wsnamedsets.C`` saves its fits.

    The workspace holds copies of what it imported, so the model fitted
    outside it moved its own parameters only: the snapshot takes theirs.
    """
    w = RooWorkspace("w")
    g, _, m, _ = gaussian()
    w.Import(g, Silence=True)
    m.setVal(0.25)
    m.setError(0.125)
    m.setConstant(True)
    w.saveSnapshot("theirs", RooArgSet([m]), True)
    w.saveSnapshot("ours", RooArgSet([m]))
    theirs, ours = w.getSnapshot("theirs").find("m"), w.getSnapshot("ours").find("m")
    assert (theirs.getVal(), theirs.getError(), theirs.isConstant()) == (0.25, 0.125, True)
    assert (ours.getVal(), ours.isConstant(), w.var("m").getVal()) == (0.0, False, 0.0)
    w.loadSnapshot("theirs")
    assert (w.var("m").getVal(), w.var("m").isConstant()) == (0.25, True)


def test_renaming_all_nodes_on_import_gives_each_its_suffix(capsys: Any) -> None:
    """``RenameAllNodes("v2")`` renames the non-fundamental nodes in ROOT's words."""
    w = RooWorkspace("w")
    x = RooRealVar("x", "x", -10, 10)
    m = RooRealVar("m", "m", 0, -1, 1)
    g = RooGaussian("g", "g", x, m, ref(2.0))
    w.Import(g, Silence=True)
    capsys.readouterr()
    w.Import(g, RenameAllNodes="v2")
    assert capsys.readouterr().out == (
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) Resolving name conflict in workspace "
        "by changing name of imported node  g to g_v2\n"
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing RooGaussian::g_v2\n"
    )
    renamed = w.pdf("g_v2")
    assert (g.GetName(), renamed is g, renamed.servers()[0] is w.var("x")) == ("g", False, True)


def printed(capsys: Any, obj: Any) -> str:
    """What ``obj.Print()`` writes."""
    capsys.readouterr()
    obj.Print()
    return str(capsys.readouterr().out)


def test_the_factory_makes_variables_and_constants_as_root_prints_them(capsys: Any) -> None:
    """``x[-10,10]``, ``m[0.5,-1,1]``, ``c5[5]``: ROOT's ranges, values and constancy."""
    w = RooWorkspace("w")
    assert printed(capsys, w.factory("x[-10,10]")) == "RooRealVar::x = 0  L(-10 - 10) \n"
    assert printed(capsys, w.factory("m[0.5,-1,1]")) == "RooRealVar::m = 0.5  L(-1 - 1) \n"
    assert printed(capsys, w.factory(" c5[5] ")) == "RooRealVar::c5 = 5 C  L(-INF - +INF) \n"
    assert w.factory("x[1,2]") is w.var("x")  # already there: the one there
    assert w.factory("x") is w.var("x")
    assert capsys.readouterr().out == ""  # the factory imports silently


def test_the_factory_makes_categories_from_labels_with_or_without_indices(capsys: Any) -> None:
    """``cat[A=0,B=5]`` and ``tag[up,down]``: ROOT's first states and their indices."""
    w = RooWorkspace("w")
    assert printed(capsys, w.factory("cat[A=0,B=5]")) == "RooCategory::cat = A(idx = 0)\n\n"
    assert printed(capsys, w.factory("tag[up,down]")) == "RooCategory::tag = up(idx = 0)\n\n"
    assert (w.cat("cat").lookupIndex("B"), w.cat("tag").lookupIndex("down")) == (5, 1)
    assert (w.factory("none[]").ClassName(), len(w.cat("none").states())) == ("RooCategory", 0)


def test_the_factory_makes_densities_by_class_name_as_root_prints_them(capsys: Any) -> None:
    """``Gaussian::g(...)``, ``Exponential``, ``Polynomial`` with a list: ROOT's lines."""
    w = RooWorkspace("w")
    w.factory("x[-10,10]")
    w.factory("m[0.5,-1,1]")
    assert printed(capsys, w.factory("Gaussian::g(x, m, s[1.5,0.1,5])")) == (
        "RooGaussian::g[ x=x mean=m sigma=s ] = 0.945959\n"
    )
    assert printed(capsys, w.factory("Exponential::e(x, tau[-0.2,-1,0])")) == (
        "RooExponential::e[ x=x c=tau ] = 1\n"
    )
    assert printed(capsys, w.factory("Polynomial::poly(x, {a0[0.1], a1[0.2]})")) == (
        "RooPolynomial::poly[ x=x coefList=(a0,a1) ] = 1\n"
    )
    assert printed(capsys, w.factory("Gaussian::gy(y[-5,5], 0, 1)")) == (
        "RooGaussian::gy[ x=y mean=0 sigma=1 ] = 1\n"
    )
    unnamed = w.factory("Gaussian(x, m, s)")
    assert (unnamed.GetName(), unnamed.getVal()) == (
        "Gaussian_x_m_s",
        pytest.approx(0.945959, 1e-6),
    )


def test_the_factory_reads_an_enumerator_of_the_class_as_its_value(capsys: Any) -> None:
    """``Decay::dec(..., SingleSided)``: ``SingleSided`` is ``RooDecay``'s, as ROOT reads it."""
    w = RooWorkspace("w")
    made = w.factory("Decay::dec(t[0,10], tau[1.5], TruthModel::tm(t), SingleSided)")
    assert printed(capsys, made) == "RooDecay::dec[ t=t tau=tau ] = 0.035674\n"


def test_the_factorys_sums_make_add_pdfs_as_root_prints_them(capsys: Any) -> None:
    """``SUM::model(f*g, e)`` with a fraction, ``SUM::ext(n*g, n*e)`` with yields: ROOT's lines."""
    w = RooWorkspace("w")
    w.factory("Gaussian::g(x[-10,10], m[0.5,-1,1], s[1.5,0.1,5])")
    w.factory("Exponential::e(x, tau[-0.2,-1,0])")
    assert printed(capsys, w.factory("SUM::model(fsig[0.3,0,1]*g, e)")) == (
        "RooAddPdf::model[ fsig * g + [%] * e ] = 0.983788/1\n"
    )
    assert printed(capsys, w.factory("SUM::ext(nsig[10,0,100]*g, nbkg[20,0,100]*e)")) == (
        "RooAddPdf::ext[ nsig * g + nbkg * e ] = 0.981986/1\n"
    )
    assert w.factory("SUM(fsig*g, e)").GetName() == "SUM_fsig_g_e"


def model() -> RooWorkspace:
    """The factory's model of ROOT's run: ``g`` and ``e`` in ``x``, ``gy`` and ``gc`` in ``y``."""
    w = RooWorkspace("w")
    for spec in (
        "Gaussian::g(x[-10,10], m[0.5,-1,1], s[1.5,0.1,5])",
        "Exponential::e(x, tau[-0.2,-1,0])",
        "a[1,0,3]",
        "Gaussian::gy(y[-5,5], 0, 1)",
        "Gaussian::gc(y, x, 2)",
    ):
        w.factory(spec)
    return w


def test_the_factorys_products_and_formulas_have_roots_values() -> None:
    """``PROD``, conditional ``PROD``, ``EXPR``, ``expr``, ``sum`` and ``prod``: ROOT's values."""
    w = model()
    values = [
        w.factory(spec).getVal()
        for spec in (
            "PROD::pr(g, e)",
            "PROD::cond(gc|y, gy)",
            "PROD::braced(gc|{y}, gy)",
            "EXPR::ep('x*x+a', x, a)",
            'EXPR::quoted("x*(x>0)+1", x)',
            "expr::fv('x+a', {x, a})",
            "sum::sm(a, m, 2.5)",
            "prod::pd(a, m)",
        )
    ]
    assert values == pytest.approx([0.945959, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 0.5], 1e-6)
    assert (w.pdf("ep").ClassName(), w.function("fv").ClassName()) == (
        "RooGenericPdf",
        "RooFormulaVar",
    )
    assert w.factory("GENERIC::gen('a', a)").ClassName() == "RooGenericPdf"


def test_the_factorys_simultaneous_density_takes_a_density_per_state() -> None:
    """``SIMUL::sim(cat, A=g, B=e)``: ROOT's value, ``g``'s, in the first state."""
    w = model()
    w.factory("cat[A=0,B=5]")
    sim = w.factory("SIMUL::sim(cat, A=g, B=e)")
    assert (sim.ClassName(), sim.getVal()) == ("RooSimultaneous", pytest.approx(0.945959, 1e-6))
    assert (sim.getPdf("A") is w.pdf("g"), sim.getPdf("B") is w.pdf("e")) == (True, True)


def test_the_workspace_of_the_factorys_model_lists_its_variables_as_root_does(
    capsys: Any,
) -> None:
    """The variables line of ``Print``: the factory's constants are not among them, as in ROOT."""
    w = model()
    w.factory("SUM::model(fsig[0.3,0,1]*g, e)")
    w.factory("tag[up,down]")
    w.Print()
    assert "variables\n---------\n(a,fsig,m,s,tag,tau,x,y)\n\n" in capsys.readouterr().out


def test_the_factory_takes_another_name_for_a_class() -> None:
    """``$Typedef(Gaussian, Gaus)`` then ``Gaus::g2(...)``: a Gaussian by another name."""
    w = model()
    assert w.factory("$Typedef(Gaussian, Gaus)") is True
    g2 = w.factory("Gaus::g2(x, 1, 2)")
    assert (g2.ClassName(), g2.getVal()) == ("RooGaussian", pytest.approx(0.882497, 1e-6))


def test_the_factory_builds_strings_numbers_and_lists_as_they_are() -> None:
    """A quoted string is its text, a number a constant, a braced list a ``RooArgList``."""
    w = model()
    factory = Factory(w)
    found = factory.build("{x, m, 1e-1}")
    assert isinstance(found, RooArgList)
    assert [one.GetName() for one in found] == ["x", "m", "0.1"]
    assert (factory.build("'text'"), factory.build("-.5").getVal()) == ("text", -0.5)
    kept = factory.keep(RooRealVar("kept", "kept", 1.0))
    assert w.var("kept") is kept


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("x + y", "cannot make sense of 'x \\+ y'"),
        ("nothing", "cannot make sense of 'nothing'"),
        ("Nonsense::n(x)", "the factory has no class Nonsense"),
        ("FCONV::fc(x, g, e)", "the factory's FCONV operator is not here yet"),
        ("int::i(g, x)", "the factory's int operator is not here yet"),
    ],
)
def test_what_the_factory_cannot_build_is_refused_by_name(spec: str, message: str) -> None:
    """A spec the factory cannot read, or one it cannot build yet, is refused saying which."""
    w = model()
    with pytest.raises(UnsupportedFeatureError, match=message):
        w.factory(spec)


def test_arguments_are_split_at_commas_outside_brackets_and_quotes() -> None:
    """``split`` keeps a call, a list, a range and a quoted formula whole."""
    text = "a, 'b,c', f(d, e), [1,2], {g, h}, \"(,\""
    assert split(text) == ["a", "'b,c'", "f(d, e)", "[1,2]", "{g, h}", '"(,"']
    assert split("f*g", "*") == ["f", "g"]
    assert split("a, ") == ["a"]
