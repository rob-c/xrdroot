"""What a macro's class tells a tree about itself, and parameters braced by default.

A class's data members are written into its translation as ROOT's
dictionary would describe them - the type as written, ``Double32_t`` and
all, the trailing comment that is the member's title, and the version its
``ClassDef`` gives - so a branch of its objects can be described in a file.
"""

from __future__ import annotations

from xrdroot.cint import translate
from xrdroot.cint.execute import run_source

SOURCE = r"""
class Event : public TObject {
public:
   Double32_t fE;  //[0,100,16] the energy
   Int_t fN;       // how many
   float fA[3];
   ClassDef(Event, 7)
};
"""


def test_a_class_carries_its_members_as_written_and_its_classdef_version():
    text = translate(SOURCE, "t.C")
    assert (
        "_cxx_layout_ = ('Event', ('TObject',), (('fE', 'Double32_t', '[0,100,16] the energy', "
        "()), ('fN', 'Int_t', 'how many', ()), ('fA', 'float', '', (3,))))"
    ) in text
    assert "def Class_Version" in text


def test_a_vector_parameter_defaulted_with_braces_is_a_vector():
    source = "int t(std::vector<bool> opt = {1, 0, 1}) { return opt.size() + opt[1]; }"
    assert "ROOT.std.vector['bool']([1, 0, 1])" in translate(source, "t.C")
    assert run_source(source, "t.C") == 3


def test_a_pointer_member_and_a_template_member_are_spelled_as_written():
    text = translate("struct S { TH1F *h; std::vector<int> v; };", "t.C")
    assert "('h', 'TH1F*', '', ())" in text and "('v', 'std::vector<int>', '', ())" in text
