"""What RooFit's classes take from a macro's C++: numbers, ``{{key, value}}`` maps and ``=``.

``RooArgSet(-0.1, 0.004)`` holds the constants ROOT makes of the numbers
(rf311); ``RooCategory c("c", "c", {{"A", 0}})``, ``Import({{"a", d}})`` and
``Slice({{&cat, "B0"}})`` are maps written as C++ initializers (rf401, rf501,
rf708); ``x = 3;`` of a ``RooRealVar`` sets it, as its ``operator=`` does
(rf402). And a dataset ``merge``\\ s another's columns beside its own; a
binned one finds the bin holding a point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xrdroot.cint import translate
from xrdroot.cint.runtime import assign_into
from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def test_numbers_in_a_collection_are_the_constants_root_makes_of_them() -> None:
    found = RooArgSet(-0.1, 0.004)
    assert [one.GetName() for one in found] == ["-0.1", "0.004"]
    assert all(isinstance(one, RooConstVar) for one in found)
    assert RooArgList(2, True).at(0).getVal() == 2.0 and RooArgList(2, True).at(1) is True


def test_a_maps_initializer_is_a_map() -> None:
    a = RooCategory("a", "a", [["SampleA", 0], ["SampleB", 1]])
    assert a.states() == {"SampleA": 0, "SampleB": 1}
    data = object()
    assert RooCmdArg("Import", [["physics", data]]).value(0) == {"physics": data}
    assert RooCmdArg("Slice", [(a, "SampleB")]).value(0) == {a: "SampleB"}
    assert RooCmdArg("Import", "physics", data).args == ("physics", data)
    assert RooCmdArg("Import", [1, 2, 3]).value(0) == [1, 2, 3]


def test_assigning_to_a_variable_or_a_category_sets_it() -> None:
    x = RooRealVar("x", "x", 0, -10, 10)
    y = RooRealVar("y", "y", 2.5)
    assign_into(x, 3)
    assert x.getVal() == 3.0
    assign_into(x, y)
    assert x.getVal() == 2.5
    c = RooCategory("c", "c", {"Plus": 1, "Minus": -1})
    assign_into(c, "Minus")
    assert c.getLabel() == "Minus"
    assign_into(c, 1)
    assert c.getLabel() == "Plus"


def test_the_translator_sets_a_variable_by_assignment_rather_than_rebinding_it() -> None:
    text = translate('void t() { RooRealVar x("x", "x", 0, 1); x = 0.5; RooRealVar *p = &x; '
                     "p = nullptr; }", "t.C")  # fmt: skip
    assert "assign_into(x, 0.5)" in text and "p = None" in text


def test_merge_adds_the_columns_of_datasets_of_as_many_events(capsys: Any) -> None:
    x, y = RooRealVar("x", "x", 0, 10), RooRealVar("y", "y", 0, 10)
    left, right = RooDataSet("l", "l", [x]), RooDataSet("r", "r", [y, x])
    for value in (1.0, 2.0):
        x.setVal(value)
        y.setVal(2 * value)
        left.add([x])
        right.add([x, y])
    assert left.merge(right) is False
    assert [one.GetName() for one in left.get()] == ["x", "y"]
    assert left.column("y").tolist() == [2.0, 4.0] and left.column("x").tolist() == [1.0, 2.0]
    right.add([x, y])
    capsys.readouterr()
    assert left.merge(right) is True
    assert "datasets have different size" in capsys.readouterr().out


def test_a_binned_dataset_finds_the_bin_a_point_is_in() -> None:
    x = RooRealVar("x", "x", 0, 4)
    x.setBins(4)
    data = RooDataSet("d", "d", [x])
    for value in (0.5, 2.5, 2.7):
        x.setVal(value)
        data.add([x])
    binned = RooDataHist("h", "h", [x], data)
    x.setVal(2.2)
    assert binned.get(RooArgSet(x)).find("x").getVal() == 2.5
    assert binned.weight() == 2.0 and binned.weight(RooArgSet(x)) == 2.0
    assert binned.get({x}) is not None and np.isclose(binned.get(1).find("x").getVal(), 1.5)
    x.setRange(-5, 5)
    x.setVal(-3)
    assert binned.get(RooArgSet(x)) is None
