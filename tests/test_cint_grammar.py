"""Corners of C++'s grammar tutorials use: aliases, qualified members, statements at the top."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source


def test_a_namespace_alias_names_the_namespace_it_stands_for(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    namespace ROOT { namespace Demo { struct Map { int n = 3; }; } }
    namespace Top = ROOT::Demo;
    void t() {
      namespace D = ROOT::Demo;
      D::Map *m = new D::Map; Top::Map k;
      printf("%d %d\\n", m->n, k.n);
    }
    """
    text = translate(source, "t.C")
    assert "m = Map()" in text and "k = Map()" in text
    run_source(source, "t.C", root=fake())
    assert capsys.readouterr().out == "3 3\n"


def test_a_class_and_the_function_of_its_name_are_both_there(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = """
    struct Show { enum { kOne = 1 }; int n = Show::kOne; };
    void Show() { struct Show s; printf("%d\\n", s.n); }
    """
    text = translate(source, "Show.C")
    assert "class Show:" in text and "def Show_function():" in text
    run_source(source, "Show.C", root=fake())
    assert capsys.readouterr().out == "1\n"


def test_a_member_named_through_its_class_is_the_member() -> None:
    text = translate("void t() { h[0]->TF1::GetXaxis()->SetTitle(\"x\"); }", "t.C")
    assert "ROOT.h[0].GetXaxis().SetTitle('x')" in text


def test_an_unnamed_template_value_parameter_chooses_overloads_and_is_passed_nothing() -> None:
    source = """template <typename V, std::enable_if_t<IsContainer<V>::value, int> = 0>
    int count(const V &v) { return v.size(); }"""
    assert "def count(v):" in translate(source, "t.C")
    assert "def g():" in translate("template <int> void g() {}", "t.C")


def test_a_method_called_outside_any_function_runs_as_the_module_loads() -> None:
    source = "RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);\nKnown;\nvoid t() {}\n"
    text = translate(source, "t.C")
    assert "\nROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)\n" in text


def test_a_chrono_duration_and_a_range_cast_take_template_arguments() -> None:
    source = """void t() {
      std::this_thread::sleep_for(std::chrono::duration<double, std::nano>(500));
      for (auto *v : dynamic_range_cast<FlexibleInterpVar *>(ws->allFunctions())) {}
    }"""
    text = translate(source, "t.C")
    assert "ROOT.std.chrono.duration['double', 'std::nano'](500)" in text
    assert "dynamic_range_cast['FlexibleInterpVar*']" in text
