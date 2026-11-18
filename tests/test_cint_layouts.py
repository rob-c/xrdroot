"""What a macro's class tells a tree about itself, and parameters braced by default.

A class's data members are written into its translation as ROOT's
dictionary would describe them - the type as written, ``Double32_t`` and
all, the trailing comment that is the member's title, and the version its
``ClassDef`` gives - so a branch of its objects can be described in a file.
"""

from __future__ import annotations

from xrdroot.pycint import translate
from xrdroot.pycint.execute import run_source

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


def test_an_array_length_written_as_an_enumerator_is_the_number_it_holds():
    """``enum {kSize = 10}; Int_t fA[kSize];`` is an array of ten, and ROOT's dictionary says
    ten - ``Event.h`` in roottest's ``io/bigevent`` declares one that way."""
    source = r"""
    class Item : public TObject {
    public:
       enum {kSize = 10, kOther, kThird};
       Int_t fA[kSize];
       Int_t fB[kOther];
       Int_t fC[Item::kThird];
       ClassDef(Item, 1)
    };
    """
    text = translate(source, "t.C")
    assert "('fA', 'Int_t', '', (10,))" in text
    assert "('fB', 'Int_t', '', (11,))" in text
    assert "('fC', 'Int_t', '', (12,))" in text


def test_an_array_length_nothing_can_count_at_compile_time_is_left_unknown():
    """A length worked out at run time is no length a file's layout can say, so it is
    ``None`` in the layout, which the writer refuses by name."""
    text = translate("struct S { int n; Int_t fA[n]; };", "t.C")
    assert "('fA', 'Int_t', '', (None,))" in text


def test_an_enumerator_whose_value_is_no_number_is_left_out_of_the_layout():
    """``enum {kA = sizeof(int), kB};`` has no value this can count at compile time, so neither
    is a length any member may have and neither is in the table."""
    text = translate("struct S { enum {kA = sizeof(int), kB}; Int_t f; };", "t.C")
    assert "kA" not in text.split("_cxx_layout_")[1].split("\n")[0]
    assert "('f', 'Int_t', '', ())" in text


def test_a_template_argument_written_as_a_number_is_the_number_in_the_layout():
    """``std::array<int,4>`` spells its length, and one whose length is an expression - which
    the layout cannot say - is ``?``: the writer refuses what it cannot lay out."""
    text = translate("struct S { std::array<int,4> a; std::array<float,2*3> b; };", "t.C")
    assert "('a', 'std::array<int, 4>', '', ())" in text
    assert "('b', 'std::array<float, ?>', '', ())" in text
