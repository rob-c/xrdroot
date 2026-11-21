"""``TTreeFormula``: ROOT's expression of a tree's branches, asked one entry at a time.

Every answer here is what ROOT 6.40 gives for the same tree: a formula it cannot
compile reports ``Bad numerical expression`` and stays invalid - ``GetNdim()`` 0,
its values 0 - rather than raising, which is what the roottest treeformula macros
check; an instance past the last element of a dynamic array is ``nan``; and a
formula added to ``gROOT``'s list of functions under a name may be called by that
name inside another.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from tests.pyrootsupport import expect, fresh

from xrdroot import FormulaError
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.formula import compile_formula
from xrdroot.formula.select import TreeNames, one_entry
from xrdroot.pyroot.core.files import TFile
from xrdroot.pyroot.core.troot import gROOT
from xrdroot.pyroot.stl import std
from xrdroot.pyroot.trees import TTree, TTreeFormula
from xrdroot.pyroot.trees.formula import refused_expression, registered_function

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def vectors(tmp_path):
    """A tree of one vector branch and one number, filled as ``execMinMaxIf`` fills one."""
    yield from _vectors(tmp_path)


def _vectors(tmp_path):
    """The filling, under the clean session ``fresh`` makes around it."""
    session = fresh(tmp_path)
    next(session)
    v = std.vector["int"]()
    i = np.zeros(1, dtype=np.int32)
    tree = TTree("t", "vectors")
    tree.Branch("v", v)
    tree.Branch("i", i, "i/I")
    for row, number in (([1, 2], 3), ([4], 9)):
        v.clear()
        for each in row:
            v.push_back(each)
        i[0] = number
        tree.Fill()
    try:
        yield tree
    finally:
        session.close()


def test_a_formula_gives_one_value_per_element_of_the_entry_the_tree_read(vectors):
    formula = TTreeFormula("f", "v", vectors)
    expect((formula.GetNdim(), 1), (formula.GetNdata(), 2), (formula.GetName(), "f"),
           (formula.GetTitle(), "v"), (formula.GetTree(), vectors))  # fmt: skip
    expect((formula.EvalInstance(0), 1.0), (formula.EvalInstance(1), 2.0))
    vectors.GetEntry(1)
    expect((formula.EvalInstance(0), 4.0), (formula.GetNdata(), 1))


def test_an_instance_past_the_last_element_of_a_dynamic_array_is_not_a_number(vectors):
    formula = TTreeFormula("f", "v", vectors)
    vectors.GetEntry(0)
    assert formula.EvalInstance(1) == 2.0
    assert np.isnan(formula.EvalInstance(2))  # what ``execOutOfBounds`` asks for


def test_a_formula_reads_entry_zero_before_any_entry_has_been_read(vectors):
    assert TTreeFormula("f", "v", vectors).EvalInstance(0) == 1.0


def test_an_expression_of_one_value_per_entry_has_no_instance_past_the_first(vectors):
    formula = TTreeFormula("f", "v[0]+i", vectors)
    vectors.GetEntry(1)
    expect((formula.GetNdata(), 1), (formula.EvalInstance(0), 13.0),
           (bool(np.isnan(formula.EvalInstance(1))), True))  # fmt: skip


def test_the_number_a_formula_holds_is_the_tree_s_and_an_expression_of_it_is_not(vectors):
    vectors.GetEntry(0)
    assert TTreeFormula("a", "v", vectors).IsInteger() is True
    assert TTreeFormula("b", "v[0]", vectors).IsInteger() is True
    assert TTreeFormula("c", "Entry$", vectors).IsInteger() is True
    assert TTreeFormula("d", "v[0]+1", vectors).IsInteger() is False
    assert TTreeFormula("e", "i>0", vectors).IsInteger() is False
    assert TTreeFormula("f", "v", vectors).IsString() is False


def test_the_multiplicity_says_whether_the_formula_varies_within_an_entry(vectors):
    vectors.GetEntry(0)
    expect((TTreeFormula("a", "v", vectors).GetMultiplicity(), 1),
           (TTreeFormula("b", "i", vectors).GetMultiplicity(), 0))  # fmt: skip


def test_a_formula_that_does_not_compile_is_reported_and_left_invalid(vectors, capfd):
    formula = TTreeFormula("bad", "v[nope]", vectors)
    reported = capfd.readouterr().err.splitlines()
    assert reported == [
        'Error in <TTreeFormula::Compile>:  Bad numerical expression : "nope"',
        'Error in <TTreeFormula::Compile>:  Bad numerical expression : "v[nope]"',
    ]
    expect((formula.GetNdim(), 0), (formula.GetNdata(), 1), (formula.EvalInstance(0), 0.0),
           (formula.IsInteger(), False), (formula.GetMultiplicity(), 0))  # fmt: skip


def test_a_name_no_branch_has_is_reported_without_the_expression_it_stands_in(capfd):
    refused = FormulaError("'nope' in 'nope+1' is not a branch", name="nope")
    refused_expression(refused, "nope+1")
    assert capfd.readouterr().err.splitlines() == [
        'Error in <TTreeFormula::Compile>:  Bad numerical expression : "nope"'
    ]


def test_a_mistake_that_names_no_name_is_reported_as_the_expression_written(capfd):
    refused_expression(FormulaError("'v +' could not be parsed"), "v +")
    assert capfd.readouterr().err.splitlines() == [
        'Error in <TTreeFormula::Compile>:  Bad numerical expression : "v +"'
    ]


def test_a_formula_named_as_a_function_is_called_by_that_name_from_another(vectors):
    # ``runsubform``'s shape: the inner formula is listed under its own name first.
    inner = TTreeFormula("positive()", "i>5", vectors)
    gROOT.GetListOfFunctions().Add(inner)
    caller = TTreeFormula("caller", "positive()", vectors)
    for entry, wanted in ((0, 0.0), (1, 1.0)):
        vectors.GetEntry(entry)
        expect((inner.EvalInstance(0), wanted), (caller.EvalInstance(0), wanted))


def test_nothing_is_registered_under_a_name_no_formula_was_listed_under(vectors):
    assert registered_function("nosuch") is None
    with pytest.raises(FormulaError, match="not a function ROOT's formulas know"):
        compile_formula("nosuch()", TreeNames(vectors._backing()))


def test_a_registered_name_is_a_function_of_no_arguments(vectors):
    gROOT.GetListOfFunctions().Add(TTreeFormula("counted()", "i", vectors))
    with pytest.raises(FormulaError, match="counted takes 0 arguments"):
        compile_formula("counted(i)", TreeNames(vectors._backing()))


def test_a_formula_is_an_expression_over_a_tree_and_needs_one():
    with pytest.raises(TypeError, match="was given no tree"):
        TTreeFormula("f", "x")


def test_another_tree_may_be_read_and_the_values_are_that_tree_s(vectors):
    formula = TTreeFormula("f", "i", vectors)
    other = TTree("other", "another tree")
    other.Branch("i", np.zeros(1, dtype=np.int32), "i/I")
    formula.SetTree(other)
    expect((formula.GetTree(), other), (formula.EvalInstance(0), 0.0))  # fmt: skip


def test_a_tree_with_no_entries_has_no_value_and_no_number(vectors):
    empty = TTree("empty", "no entries")
    empty.Branch("i", np.zeros(1, dtype=np.int32), "i/I")
    formula = TTreeFormula("f", "i", empty)
    expect((formula.EvalInstance(0), 0.0), (formula.GetNdata(), 1))  # fmt: skip


def test_the_parts_of_roofs_answer_this_does_not_give_are_refused_in_a_sentence(vectors):
    formula = TTreeFormula("f", "v", vectors)
    for name in ("EvalObject", "EvalClass", "EvalStringInstance"):
        with pytest.raises(UnsupportedFeatureError, match=f"{name} "):
            getattr(formula, name)()
    with pytest.raises(UnsupportedFeatureError, match="fExpr"):
        formula.Print()
    with pytest.raises(AttributeError, match="ROOT's TTreeFormula has GetLeaf"):
        formula.GetLeaf(0)


def test_what_a_tree_reads_and_what_it_does_not_ask_again(vectors):
    formula = TTreeFormula("f", "v", vectors)
    expect((formula.IsQuickLoad(), False), (formula.GetNpar(), 0), (formula.GetNcodes(), 1),
           (formula.ClassName(), "TTreeFormula"))  # fmt: skip
    formula.EvalInstance(0)
    formula.ResetLoading()
    formula.UpdateFormulaLeaves()
    formula.Notify()
    formula.SetQuickLoad(True)
    formula.SetAxis(None)
    formula.SetName("renamed")
    expect((formula.EvalInstance(0), 1.0), (formula.EvalInstance64(0), 1),
           (formula.GetName(), "renamed"), (formula.IsQuickLoad(), False))  # fmt: skip


def test_a_formula_answers_a_whole_object_s_member_as_scan_and_draw_do():
    with open_root_tree() as tree:
        tree.GetEntry(0)
        member = TTreeFormula("f", "tlv.fE", tree)
        assert member.EvalInstance(0) == 20.0


class open_root_tree:
    """The tree of whole objects ``tests/data`` holds, opened and closed as a context."""

    def __enter__(self):
        self._file = TFile.Open(str(DATA / "object-branches-6.40.root"))
        return self._file.Get("t")

    def __exit__(self, *what):
        self._file.Close()
        return False


def test_one_entry_is_read_as_a_batch_of_one_with_its_own_entry_number():
    with open_root_tree() as tree:
        formula = compile_formula("Entry$", TreeNames(tree._backing()))
        values, valid = one_entry(tree._backing(), formula, 3)
        assert values.tolist() == [3] and valid.tolist() == [True]


def test_the_parsed_expression_is_what_the_formula_is_made_of(vectors):
    formula = compile_formula("v[0]", TreeNames(vectors._backing()))
    assert isinstance(formula.root, object)
    assert formula.branches == ("v",)


def test_a_fixed_size_array_keeps_the_shape_its_title_declares():
    with open_root_tree() as tree:
        formula = compile_formula("h1.fXaxis.fNbins", TreeNames(tree._backing()))
        values, _ = one_entry(tree._backing(), formula, 0)
        assert np.asarray(values).tolist() == [4]


def test_a_column_that_does_not_compile_is_printed_empty_as_roofs_scan_prints_it(vectors, capfd):
    text = vectors._backing().scan("v[nope]:i", on_error=refused_expression, file=None)
    assert text.splitlines()[1] == "*    Row   *   v[nope] *         i *"
    assert text.splitlines()[3] == "*        0 *           *         3 *"
    assert capfd.readouterr().err.splitlines()[0].endswith('"nope"')


def test_a_scan_of_a_tree_prints_the_bad_column_empty_and_the_good_one_beside_it(vectors, capfd):
    # ROOT's ``TTree::Scan`` makes a ``TTreeFormula`` of each column and prints the table
    # with a blank under a name that did not compile, which is what a macro sees.
    assert vectors.Scan("v[nope]:i") == 2
    printed = capfd.readouterr()
    assert printed.out.splitlines()[3] == "*        0 *           *         3 *"
    assert printed.err.splitlines()[1] == (
        'Error in <TTreeFormula::Compile>:  Bad numerical expression : "v[nope]"'
    )


def test_a_blank_column_beside_a_looping_one_still_gives_a_row_per_element(vectors, capfd):
    text = vectors._backing().scan("v[nope]:v", on_error=refused_expression, file=None)
    rows = [line for line in text.splitlines() if line.startswith("*        0 *")]
    assert rows == ["*        0 *        0 *           *         1 *",
                    "*        0 *        1 *           *         2 *"]
    capfd.readouterr()


def test_a_scan_told_of_no_reporting_still_raises_the_bad_expression(vectors):
    with pytest.raises(FormulaError, match="not a branch"):
        vectors._backing().scan("v[nope]")
