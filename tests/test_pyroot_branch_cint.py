"""Running a macro as cling runs one: a null pointer streamed as ``0``, HistFactory's functions
found unqualified.

Under PyROOT, compiled RooFit code streams a null pointer as ``0x0``; in a
macro cling runs, as ``0``. The runner says which while a macro runs and
puts the setting back after - even when the macro fails. And
``MakeModelAndMeasurementFast(meas)`` is found without its namespace, as
argument-dependent lookup finds it in C++.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.cint import execute
from xrdroot.cint.execute import MacroError, run_source
from xrdroot.cint.runtime.root import USED, RootProxy
from xrdroot.roofit.printing import NULL_POINTER


def test_a_null_pointer_is_zero_while_a_macro_runs_and_0x0_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside the macro's function the setting is ``0``; outside, PyROOT's ``0x0`` again."""
    real = execute._entry

    def entry(*given: object) -> object:
        found = real(*given)  # type: ignore[arg-type]
        return lambda: (NULL_POINTER[0], found())

    monkeypatch.setattr(execute, "_entry", entry)
    assert run_source("int t() { return 7; }\n", "t.C") == ("0", 7)
    assert NULL_POINTER[0] == "0x0"


def test_the_null_pointer_setting_is_put_back_when_a_macro_fails() -> None:
    """A macro that throws leaves the setting as it found it."""
    with pytest.raises(MacroError):
        run_source('void t() { throw std::runtime_error("no"); }\n', "t.C")
    assert NULL_POINTER[0] == "0x0"
    assert run_source("int t() { return 1; }\n", "t.C", call=False)["t"]() == 1


def test_histfactorys_functions_are_found_without_their_namespace() -> None:
    """``MakeModelAndMeasurementFast`` is ``RooStats::HistFactory``'s, looked up unqualified."""
    assert "RooStats.HistFactory" in USED
    found = RootProxy().MakeModelAndMeasurementFast
    assert found is ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast


def test_a_macro_without_the_function_its_file_names_runs_to_nothing() -> None:
    """``t.C`` defining only ``other()`` has no entry to call: running it gives ``None``."""
    assert run_source("int other() { return 1; }\n", "t.C") is None
    assert NULL_POINTER[0] == "0x0"
