"""``RooAbsArg``: a node's name, its inputs, the walks over its graph, attributes and copies.

The printed lines were compared with ROOT 6.40.04: ``RooGaussian::g[ x=x
mean=mean sigma=sigma ] = 0.882497`` for ``Print()``, the compact tree for
``Print("t")`` (addresses aside), and the lists of variables, parameters,
observables and components each walk hands back.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from xrdroot.roofit.arg import Proxy, RooAbsArg, _observable_names
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.real import RooAbsReal
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def gaussian() -> tuple[RooRealVar, RooRealVar, RooRealVar, RooGaussian]:
    x = RooRealVar("x", "the x", 1, -10, 10)
    m = RooRealVar("mean", "m", 0, -1, 1)
    s = RooRealVar("sigma", "s", 2, 0.1, 5)
    return x, m, s, RooGaussian("g", "a gauss", x, m, s)


class Summed(RooAbsReal):
    """A node with a single input and a list of inputs, as a sum has."""

    def __init__(self, name: str, first: Any, terms: Any) -> None:
        super().__init__(name, name)
        self.first = self._proxy("first", first)
        self.terms = self._list_proxy("terms", terms)

    def compute(self, ctx: Any) -> Any:
        return self.first.getVal() + sum(one.getVal() for one in self.terms)


class Hidden(RooAbsReal):
    """A node whose only input is one that ``Print`` does not show."""

    def __init__(self, name: str, inner: Any) -> None:
        super().__init__(name, name)
        self._proxy("!inner", inner)


def unaddressed(text: str) -> str:
    return re.sub(r"0x[0-9a-f]+", "@", text)


def test_a_list_input_prints_its_members_in_brackets() -> None:
    """A sum's coefficients print as ``coefList=(a,b)``: the list form of a proxy."""
    a, b = RooRealVar("a", "a", 1.0), RooRealVar("b", "b", 2.0)
    assert Proxy("coefs", RooArgList(a, b), many=True).text() == "coefs=(a,b)"
    assert Proxy("one", a).text() == "one=a"
    assert Proxy("coefs", RooArgList(a, b), many=True).args() == [a, b]


def test_the_names_and_titles_of_a_node_can_be_changed_together() -> None:
    """``SetNameTitle`` renames and retitles, as ``TNamed`` does."""
    node = RooAbsArg("n", "t")
    node.SetNameTitle("renamed", "retitled")
    assert (node.GetName(), node.GetTitle()) == ("renamed", "retitled")
    assert (node.printName(), node.printTitle()) == ("renamed", "retitled")
    assert RooAbsArg("n", None).GetTitle() == ""


def test_a_node_knows_its_class_and_the_classes_it_inherits_from() -> None:
    """``InheritsFrom`` takes a class name or a class, and ``IsA`` is the class itself."""
    *_, g = gaussian()
    assert g.ClassName() == "RooGaussian"
    assert g.printClassName() == "RooGaussian"
    assert g.InheritsFrom("RooAbsPdf")
    assert g.InheritsFrom(RooAbsReal)
    assert not g.InheritsFrom("RooRealVar")
    assert g.IsA() is RooGaussian
    assert repr(g) == "<RooGaussian::g>"


def test_a_gaussian_prints_its_inputs_and_value_as_root_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Print()`` shows each proxy as ``name=arg`` and the value to six figures."""
    *_, g = gaussian()
    g.Print()
    assert capsys.readouterr().out == "RooGaussian::g[ x=x mean=mean sigma=sigma ] = 0.882497\n"


def test_the_tree_of_a_gaussian_marks_each_input_as_a_value_server(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Print("t")`` is the compact tree: the node, then its inputs indented with ``/V-``."""
    *_, g = gaussian()
    g.Print("t")
    assert unaddressed(capsys.readouterr().out) == (
        "@ RooGaussian::g = 0.882497 [Auto,Dirty] \n"
        "  @/V- RooRealVar::x = 1\n"
        "  @/V- RooRealVar::mean = 0\n"
        "  @/V- RooRealVar::sigma = 2\n"
    )
    g.printCompactTree()
    assert unaddressed(capsys.readouterr().out).startswith("@ RooGaussian::g = 0.882497")


def test_the_component_tree_prints_only_the_nodes_that_are_not_variables(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``printComponentTree`` skips the leaves, and a name pattern or a depth cuts it short."""
    x, m, _s, g = gaussian()
    outer = Summed("outer", g, [m, RooConstVar("c", "c", 3.0)])
    outer.printComponentTree()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Summed::outer[ first=g terms=(mean,c) ] = 3.8825"
    assert lines[1] == "  RooGaussian::g[ x=x mean=mean sigma=sigma ] = 0.882497"
    outer.printComponentTree("", "g", 999)
    assert capsys.readouterr().out == "  RooGaussian::g[ x=x mean=mean sigma=sigma ] = 0.882497\n"
    outer.printComponentTree("", None, 1)
    assert len(capsys.readouterr().out.splitlines()) == 1
    x.printComponentTree()
    assert capsys.readouterr().out == ""


def test_an_input_hidden_from_print_leaves_empty_brackets() -> None:
    """A proxy whose name starts with ``!`` is kept out of the printed arguments."""
    x, *_ = gaussian()
    assert Hidden("h", x).printArgs() == "[ ]"
    assert RooAbsArg("bare").printArgs() == ""
    assert unaddressed(RooAbsArg("bare").printTree("")) == "@ RooAbsArg::bare = \n"


def test_the_servers_of_a_node_are_each_listed_once_in_declared_order() -> None:
    """A variable used by two inputs is one server, and ``numProxies`` counts the inputs."""
    x, m, *_ = gaussian()
    node = Summed("s", x, [m, x])
    assert [one.GetName() for one in node.servers()] == ["x", "mean"]
    assert node.numProxies() == 2
    assert node.findServer("mean") is m
    assert node.findServer(x) is x
    assert node.findServer("nothing") is None
    assert node.isDerived()
    assert not node.isFundamental()


def test_the_walks_hand_back_the_variables_parameters_and_observables_root_does() -> None:
    """Constants are no variables, and the parameters are the variables less the observables."""
    x, _m, _s, g = gaussian()
    c = RooConstVar("c", "c", 3.0)
    node = Summed("f2", x, [c])
    assert [one.GetName() for one in node.getVariables()] == ["x"]
    assert [one.GetName() for one in g.getParameters(RooArgSet(x))] == ["mean", "sigma"]
    assert [one.GetName() for one in g.getObservables(RooArgSet(x))] == ["x"]
    assert [one.GetName() for one in g.getParameters()] == ["mean", "sigma", "x"]
    assert [one.GetName() for one in g.getObservables()] == []
    assert [one.GetName() for one in g.getComponents()] == ["g"]


def test_the_parameters_asked_for_into_a_set_are_added_to_it_sorted_as_root_sorts_them() -> None:
    """``getParameters(observables, outputSet)`` fills the caller's set and returns false, as
    ROOT does: what the set held stays, the rest is added, and the whole is sorted by name."""
    x, _m, s, g = gaussian()
    held = RooArgSet(s)
    assert g.getParameters(RooArgSet(x), held) is False
    assert [one.GetName() for one in held] == ["mean", "sigma"]
    everything = RooArgSet()
    assert g.getParameters(None, everything, True) is False
    assert [one.GetName() for one in everything] == ["mean", "sigma", "x"]


def test_what_a_node_depends_on_is_its_variables_and_constants_below_it() -> None:
    """``dependsOn`` looks through the graph; ``dependents`` names the variables, once."""
    x, m, _s, g = gaussian()
    c = RooConstVar("c", "c", 3.0)
    node = Summed("f2", x, [c])
    assert node.dependsOn(c) and g.dependsOn(x) and not g.dependsOn(c)
    assert not g.dependsOn(x, ignoreArg=x)
    assert g.dependsOnValue(RooArgSet(m))
    assert g.dependents() == frozenset({"x", "mean", "sigma"})
    assert g.dependents() is g.dependents()


def test_the_observables_of_a_dataset_are_its_columns() -> None:
    """A dataset stands for its columns wherever observables are asked for."""

    class Columns:
        def __init__(self, columns: RooArgSet) -> None:
            self.columns = columns

        def get(self) -> RooArgSet:
            return self.columns

        def numEntries(self) -> int:
            return 0

    x, m, _s, g = gaussian()
    assert _observable_names(Columns(RooArgSet(x, m))) == ["x", "mean"]
    assert _observable_names(x) == ["x"]
    assert _observable_names(None) == []
    assert [one.GetName() for one in g.getParameters(Columns(RooArgSet(x)))] == ["mean", "sigma"]


def test_a_cached_walk_is_forgotten_when_a_node_is_renamed() -> None:
    """The walks are remembered, but a new name must show in the next one."""
    _x, _m, s, g = gaussian()
    assert g.dependents() == frozenset({"x", "mean", "sigma"})
    s.SetName("width")
    assert g.dependents() == frozenset({"x", "mean", "width"})
    assert [one.GetName() for one in g.leaves()] == ["x", "mean", "width"]


def test_a_node_shared_by_two_inputs_is_walked_once() -> None:
    """In a diamond - two Gaussians of the same ``x`` - ``x`` is one leaf, not two."""
    x, m, _s, g = gaussian()
    other = RooGaussian("g2", "g2", x, m, RooRealVar("w", "w", 1.0))
    top = Summed("top", g, [other])
    assert [one.GetName() for one in top.getComponents()] == ["top", "g", "g2"]
    assert [one.GetName() for one in top.getVariables()] == ["mean", "sigma", "w", "x"]


def test_boolean_attributes_can_be_set_cleared_and_listed() -> None:
    """``Constant`` is the attribute that makes a node constant."""
    node = RooAbsArg("n")
    node.setAttribute("Constant")
    node.setAttribute("Other", True)
    assert node.isConstant()
    assert node.attributes() == {"Constant", "Other"}
    node.setAttribute("Constant", False)
    assert not node.getAttribute("Constant")
    assert not node.isConstant()


def test_string_attributes_can_be_set_replaced_and_removed() -> None:
    """Setting ``None`` removes a string attribute, as removing it does."""
    node = RooAbsArg("n")
    node.setStringAttribute("key", 5)
    assert node.getStringAttribute("key") == "5"
    node.setStringAttribute("key", None)
    assert node.getStringAttribute("key") is None
    node.setStringAttribute("key", "v")
    node.removeStringAttribute("key")
    node.removeStringAttribute("absent")
    assert node.getStringAttribute("key") is None


def test_a_clone_shares_its_inputs_but_not_its_attributes() -> None:
    """``clone`` copies the node and its proxies; the inputs stay the same objects."""
    x, _m, _s, g = gaussian()
    g.setAttribute("tagged")
    g.setStringAttribute("k", "v")
    made = g.clone("copy")
    assert made.GetName() == "copy"
    assert made is not g
    assert made.servers()[0] is x
    made.setAttribute("tagged", False)
    made.setStringAttribute("k", "w")
    assert g.getAttribute("tagged") and g.getStringAttribute("k") == "v"
    assert g.Clone().GetName() == "g"


def test_a_tree_clone_copies_every_node_under_it() -> None:
    """``cloneTree`` is deep: its variables are new objects of the same names."""
    x, _m, _s, g = gaussian()
    made = g.cloneTree("deep")
    assert made.GetName() == "deep"
    assert made.servers()[0] is not x
    assert made.servers()[0].GetName() == "x"
    assert g.cloneTree().GetName() == "g"
