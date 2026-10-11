"""``ROOT.RDF``'s helpers and the frame's PyROOT pieces the dataframe tutorials reach for.

``MakeTrivialDataFrame`` (df010), ``Not`` and ``PassAsVec`` (df020), a
macro's ``Aggregate`` (df023), ``RNode`` of a node (df025), ``Stats`` that
print as ``TStatistic`` (df031), ``Describe().Print()`` (df033), and
``*result == n`` comparing the value. ROOT 6.40's answers are in each test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.rdf.frame import RDFDescription


def _macros(function: Any) -> Any:
    """``function`` as a macro's translation defines it, which the frame calls per entry."""
    function.__module__ = "__cint__"
    return function


def test_a_trivial_frame_counts_its_entries_in_col0() -> None:
    """``RTrivialDS``: ``col0`` is each entry's number - the odd ones, skipping the even."""
    frame = ROOT.RDF.MakeTrivialDataFrame(5)
    assert frame.AsNumpy(["col0"])["col0"].tolist() == [0, 1, 2, 3, 4]
    assert frame.GetColumnType("col0") == "ULong64_t"
    odd = ROOT.RDF.MakeTrivialDataFrame(6, True)
    assert odd.AsNumpy(["col0"])["col0"].tolist() == [1, 3, 5]


def test_not_negates_a_macros_filter_entry_by_entry_and_a_batch_at_once() -> None:
    frame = ROOT.RDataFrame(4).Define("x", "int(rdfentry_)")
    is_one = _macros(lambda a: a == 1)
    assert frame.Filter(ROOT.RDF.Not(is_one), ["x"]).Count().GetValue() == 3
    batched = ROOT.RDF.Not(lambda x: x > 1)
    assert batched(np.array([0, 2])).tolist() == [True, False]


def test_pass_as_vec_hands_the_values_over_as_one_vector() -> None:
    """``PassAsVec<2, int>(f)``: ``f`` is given ``std::vector<int>{one, two}``."""
    frame = ROOT.RDataFrame(3).Define("one", "1").Define("two", "2")
    check = _macros(lambda v: v.size() == 2 and v[0] == 1 and v[1] == 2)
    passed = ROOT.RDF.PassAsVec[2, "int"](check)
    assert frame.Filter(passed, ["one", "two"]).Count().GetValue() == 3


def test_a_macros_aggregate_folds_entry_by_entry_and_merges_a_vector_of_partials() -> None:
    """df023: the product of 1 to 5, 120, with ``EnableImplicitMT(2)`` in ROOT."""
    frame = ROOT.RDataFrame(5).Define("x", "rdfentry_ + 1.")

    def merger(accumulators: Any) -> None:
        for at in range(1, accumulators.size()):
            accumulators[0] *= accumulators[at]

    folded = frame.Aggregate(_macros(lambda acc, x: acc * x), _macros(merger), "x", 1.0)
    assert folded.GetValue() == 120.0
    plain = frame.Aggregate(lambda acc, x: acc + x.sum(), lambda a, b: a + b, "x", 0.0)
    assert plain.GetValue() == 15.0


def test_an_rnode_of_a_node_is_that_node() -> None:
    node = ROOT.RDataFrame(8).Filter("rdfentry_ > 3")
    again = ROOT.RDF.RNode(node)
    assert again.Count().GetValue() == 4
    assert again._inner is node._inner


def test_a_result_compares_as_its_value() -> None:
    """``R__ASSERT(*count == n)``: the value compared, as C++ compares what it points at."""
    count = ROOT.RDataFrame(3).Count()
    assert count == 3 and not count != 3
    assert count < 4 and count <= 3 and count > 2 and count >= 3
    with pytest.raises(TypeError):
        hash(count)


def test_stats_print_as_tstatistic_with_its_whole_effective_entries(capsys: Any) -> None:
    """df031: ``Mean = 40.8 +- 12.86``: ROOT's error is over ``(int) W^2/W2``, 22 here."""
    frame = ROOT.RDataFrame(256).Define("v", "rdfentry_").Define("w", "1./(v+1)")
    weighted = frame.Stats("v", "w")
    assert weighted.GetNeff() == 22
    assert weighted.GetMeanErr() == pytest.approx(12.859890995362191, rel=1e-12)
    weighted.Print()
    assert capsys.readouterr().out == (
        " OBJ: TStatistic\t  \t Mean = 40.8 +- 12.86 \t RMS = 60.318 \t Count = 256 \t "
        "Min = 0 \t Max = 255\n"
    )
    assert weighted.GetName() == ""


def test_a_description_prints_itself() -> None:
    described = ROOT.RDataFrame(2).Describe()
    assert isinstance(described, RDFDescription)
    assert described.AsString() == str(described)


def test_a_description_print_writes_the_text(capsys: Any) -> None:
    described = ROOT.RDataFrame(2).Describe()
    described.Print()
    assert capsys.readouterr().out == f"{described}\n"


def test_get_class_names_an_objects_class() -> None:
    """df001: ``ROOT::GetClass(b1Vec.GetPtr())->GetName()`` is ``vector<double>``."""
    taken = ROOT.RDataFrame(2).Define("x", "1.0").Take("x").GetValue()
    assert ROOT.GetClass(taken).GetName() == "vector<double>"
    assert ROOT.GetClass(ROOT.TH1D("gc", "", 1, 0, 1)).GetName() == "TH1D"
    assert ROOT.GetClass(ROOT.std.vector["int"]()).GetName() == "vector<int>"
    assert repr(ROOT.GetClass(ROOT.std.vector["int"]())) == "<TClass vector<int>>"
    with pytest.raises(TypeError, match="no dictionary"):
        ROOT.GetClass(object())


def test_a_macros_vector_merger_is_given_the_partials_two_at_a_time() -> None:
    """Batches merged in turn: ``merger({a, b})`` leaves the answer in the first."""
    from xrdroot.pyroot.rdf.entrywise import _vector_merger

    def merger(accumulators: Any) -> None:
        accumulators[0] += accumulators[1]

    assert _vector_merger(merger)(1.5, 2.0) == 3.5
    assert _vector_merger(merger)(1, 2) == 3
