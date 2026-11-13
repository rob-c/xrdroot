"""``T{a, b}``: a temporary built from a braced list - a container's items, a class's arguments."""

from __future__ import annotations

import xrdroot.pyroot as ROOT
from xrdroot.pycint import translate


def expression(source: str) -> str:
    text = translate("void t(int b) {\nauto v = " + source + ";\n}\n", "t.C")
    compile(text, "t.C", "exec")
    return text


def test_a_braced_container_temporary_holds_its_items_rather_than_sizing_itself():
    assert "([b, b * 10])" in expression("ROOT::VecOps::RVec<int>{ b, b * 10 }")
    assert "([1.5])" in expression("std::vector<double>{1.5}")


def test_a_braced_class_temporary_still_takes_its_items_as_constructor_arguments():
    assert "TVector2(1, b)" in expression("TVector2{1, b}")


def test_a_braced_rvec_in_a_define_holds_its_items_per_entry():
    frame = ROOT.RDataFrame(3).Define("b", "(int) rdfentry_")
    found = frame.Define("v", "ROOT::VecOps::RVec<int>{ b, b * 10 }").AsNumpy(["v"])["v"]
    assert [list(v) for v in found] == [[0, 0], [1, 10], [2, 20]]


def test_a_std_string_joined_to_an_integer_writes_it_in_decimal_as_tstring_does():
    text = expression('std::string("a[") + b + "]"')
    assert "str('a[') + str(b) + ']'" in text
    assert "+ str(" not in expression('std::string("a") + \'c\'')
    assert "+ str(" not in expression("std::string(\"a\") + true")
