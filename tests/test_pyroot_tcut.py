"""``TCut``: a selection that is its expression, combined as ROOT's operators combine cuts."""

from __future__ import annotations

from xrdroot.pyroot.tcut import TCut


def test_a_cut_of_one_argument_is_named_cut_and_titled_its_expression():
    cut = TCut("x > 0")
    assert (cut.GetName(), cut.GetTitle(), cut.ClassName()) == ("CUT", "x > 0", "TCut")
    assert (str(cut), cut.Data(), cut.IsNull(), bool(cut)) == ("x > 0", "x > 0", False, True)


def test_a_cut_of_no_arguments_is_null_and_still_true():
    cut = TCut()
    assert (cut.GetName(), cut.IsNull(), bool(cut)) == ("", True, True)
    assert TCut("name", "y < 1").GetName() == "name"


def test_cuts_combine_as_roots_operators_join_their_expressions():
    a, b = TCut("a"), TCut("b")
    assert str(a + b) == str(a & b) == "(a)&&(b)"
    assert str(a | b) == "(a)||(b)"
    assert str(a * "w") == "(a)*(w)"
    assert str(~a) == "!(a)"
    assert str("c" + a) == "(c)&&(a)"


def test_an_empty_cut_gives_way_to_the_other_in_every_combination():
    assert str(TCut("") + TCut("a")) == "a"
    assert str(TCut("a") + TCut("")) == "a"


def test_a_cut_grows_in_place_under_its_assigning_operators():
    cut = TCut("a")
    cut += "b"
    cut *= "w"
    assert str(cut) == "((a)&&(b))*(w)"


def test_cuts_are_equal_and_hash_alike_when_their_expressions_are():
    assert TCut("a") == TCut("x", "a")
    assert TCut("a") == "a"
    assert len({TCut("a"), TCut("a")}) == 1
