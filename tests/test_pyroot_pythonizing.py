"""``@ROOT.pythonization``: Python run on a C++ class the first time a script looks it up."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot import pythonizing
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _fresh():
    """Each test's pythonizors and lookups are its own; ``ProcessLine`` is found afresh."""
    pythonizors, seen = list(pythonizing.PYTHONIZORS), dict(pythonizing.SEEN)
    yield
    pythonizing.PYTHONIZORS[:] = pythonizors
    pythonizing.SEEN.clear()
    pythonizing.SEEN.update(seen)
    troot.set_line_processor(None)


def test_a_class_looked_up_after_its_pythonizor_is_registered_is_pythonized_once():
    ROOT.gInterpreter.ProcessLine("class PyzLater { };")
    seen = []
    ROOT.pythonization("PyzLater", ns="Elsewhere")(lambda klass: seen.append("elsewhere"))

    @ROOT.pythonization("PyzLater")
    def pythonize(klass, name):
        seen.append((klass.__cpp_name__, name))

    assert ROOT.PyzLater is ROOT.PyzLater
    assert seen == [("PyzLater", "PyzLater")]


def test_a_class_looked_up_before_is_pythonized_as_its_pythonizor_is_registered():
    ROOT.gInterpreter.ProcessLine("namespace PyzNS { class PyzEarly { }; class Other { }; }")
    early = ROOT.PyzNS.PyzEarly
    ROOT.gInterpreter.ProcessLine("class PyzEarly { };")
    ROOT.PyzEarly  # noqa: B018 - the same name at the top, which is another class
    ROOT.PyzNS.Other  # noqa: B018
    seen = []

    @ROOT.pythonization(["PyzEarly", "PyzEarly"], ns="PyzNS::")
    def pythonize(klass):
        seen.append(klass)

    assert seen == [early] and early.__cpp_name__ == "PyzNS::PyzEarly"


def test_a_prefix_matches_every_class_starting_with_it():
    ROOT.gInterpreter.ProcessLine("class PyzPrefixA { }; ")
    ROOT.gInterpreter.ProcessLine("class PyzPrefixB { }; ")
    seen = []
    ROOT.pythonization(("PyzPrefix",), is_prefix=True)(lambda klass, name: seen.append(name))
    ROOT.PyzPrefixA, ROOT.PyzPrefixB  # noqa: B018
    assert seen == ["PyzPrefixA", "PyzPrefixB"]


def test_a_template_is_pythonized_as_its_instances_not_itself():
    ROOT.gInterpreter.ProcessLine("template <class T> class PyzTemplate { };")
    ROOT.gInterpreter.ProcessLine("struct PyzArg { };")
    template = ROOT.PyzTemplate
    instance = template["int"]
    assert instance is template["int"] and issubclass(instance, template)
    assert template[ROOT.PyzArg, float].__cpp_name__ == "PyzTemplate<PyzArg,double>"
    assert template[ROOT.TH1F].__cpp_name__ == "PyzTemplate<TH1F>"
    seen = []
    ROOT.pythonization("PyzTemplate", is_prefix=True)(lambda klass, name: seen.append(name))
    assert seen == ["PyzTemplate<int>", "PyzTemplate<PyzArg,double>", "PyzTemplate<TH1F>"]


def test_a_template_in_a_namespace_keeps_its_namespace():
    ROOT.gInterpreter.ProcessLine("namespace PyzTNS { template <class T> class Box { }; }")
    assert ROOT.PyzTNS.Box[3].__cpp_name__ == "PyzTNS::Box<3>"


def test_names_are_refused_with_a_namespace_or_of_the_wrong_type():
    with pytest.raises(ValueError, match="namespace definition found"):
        ROOT.pythonization("NS::A")
    with pytest.raises(TypeError, match="must be string or iterable of strings"):
        ROOT.pythonization([1])


def test_a_pythonizor_takes_the_class_and_perhaps_its_name_and_nothing_else():
    with pytest.raises(TypeError, match="wrong number of parameters"):
        ROOT.pythonization("A")(lambda klass, name, other: None)


def test_what_is_no_class_and_private_names_are_handed_back_as_they_are():
    assert pythonizing.looked_up(5, "five") == 5
    assert pythonizing.looked_up(ROOT.TH1F, "_hidden") is ROOT.TH1F
    assert "__cpp_name__" not in vars(pythonizing.looked_up(ROOT.TH1F, "TH1F"))
