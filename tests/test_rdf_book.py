"""``RDataFrame::Book``: a custom action written as ROOT's ``Detail::RDF::RActionImpl``.

ROOT's own tests write a helper class deriving from ``RActionImpl<Helper>`` and book it;
the framework calls the helper's ``Initialize``, then ``Exec`` once per entry - with the
slot first, then the booked columns - then ``Finalize``, and the result is what the
helper's ``GetResultPtr`` points at.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


class Counter(ROOT.Detail.RDF.RActionImpl["Counter"]):
    """``CounterHelper`` of ``CounterHelper.h``: counts the entries it is told about."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []
        self.finalized = 0
        self.result = 0

    def GetResultPtr(self) -> int:
        """What ``Finalize`` left: a ``shared_ptr<int>`` reads as the number it points at."""
        return self.result

    def Initialize(self) -> None:
        self.calls.clear()

    def Exec(self, slot: int) -> None:
        self.calls.append(int(slot))
        self.result += 1

    def Finalize(self) -> None:
        self.finalized += 1

    def GetActionName(self) -> str:
        return "Counter"


class ColumnSum(ROOT.Detail.RDF.RActionImpl["ColumnSum"]):
    """A helper whose ``Exec`` takes the slot and one booked column."""

    def __init__(self) -> None:
        self.total = 0.0
        self.slots: set[int] = set()

    def GetResultPtr(self) -> float:
        return self.total

    def Initialize(self) -> None:
        self.total = 0.0

    def Exec(self, slot: int, value: float) -> None:
        self.slots.add(int(slot))
        self.total += float(value)

    def Finalize(self) -> None:
        return None


def test_ra_action_impl_is_one_base_for_every_spelling_of_the_helper() -> None:
    assert ROOT.Detail.RDF.RActionImpl["Counter"] is ROOT.Detail.RDF.RActionImpl
    assert issubclass(Counter, ROOT.Detail.RDF.RActionImpl)
    def looked_up() -> object:
        return ROOT.Detail.RDF.Nowhere  # type: ignore[attr-defined]

    with pytest.raises(AttributeError, match=r"ROOT has Detail\.RDF\.Nowhere"):
        looked_up()


def test_a_booked_helper_is_told_each_entry_between_initialize_and_finalize() -> None:
    frame = ROOT.RDataFrame(5)
    helper = Counter()
    booked = frame.Book(helper, {})
    assert booked.GetValue() == 5
    assert helper.calls == [0] * 5
    assert helper.finalized == 1


def test_a_booked_column_reaches_the_helpers_exec() -> None:
    frame = ROOT.RDataFrame(4).Define("x", "rdfentry_ + 0.5")
    helper = ColumnSum()
    booked = frame.Book(helper, {"x"})
    assert booked.GetValue() == pytest.approx(0.5 + 1.5 + 2.5 + 3.5)
    assert helper.slots == {0}


def test_a_frame_booked_twice_with_helpers_runs_one_loop_for_both() -> None:
    frame = ROOT.RDataFrame(3).Define("x", "1.0")
    first, second = ColumnSum(), ColumnSum()
    one = frame.Book(first, {"x"})
    two = frame.Book(second, {"x"})
    assert one.GetValue() == 3.0 and two.GetValue() == 3.0
