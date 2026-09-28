"""A PyROOT script's standard output into a pipe: Python's buffer and C's, in ROOT's order."""

from __future__ import annotations

import io
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from xrdroot import stdio
from xrdroot.roofit import cout


@contextmanager
def piped(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """``sys.stdout`` a pipe, block-buffered as Python makes one; what came out, after."""
    read, write = os.pipe()
    stream = io.TextIOWrapper(io.BufferedWriter(io.FileIO(write, "w")), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", stream)
    out: list[str] = []
    try:
        yield out
    finally:
        monkeypatch.undo()
        stream.close()
        with io.FileIO(read, "r") as pipe:
            out.append(pipe.readall().decode())


def split_run(monkeypatch: pytest.MonkeyPatch, body: Callable[[], None]) -> str:
    with piped(monkeypatch) as out:
        with stdio.split():
            body()
    return out[0]


def test_a_minimisation_prints_before_the_scripts_own_earlier_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumericalMinimization.py: ROOT's output has Minuit2's lines first, the script's last."""
    from xrdroot import pyroot as ROOT

    def script() -> None:
        minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2", "")
        minimizer.SetMaxFunctionCalls(1000000)
        minimizer.SetTolerance(0.001)
        minimizer.SetPrintLevel(1)
        f = ROOT.Math.Functor(lambda v: (v[1] - v[0] ** 2) ** 2 + (1 - v[0]) ** 2, 2)
        print("f(-1,1.2) = ", f([-1.0, 2.0]))
        minimizer.SetFunction(f)
        minimizer.SetVariable(0, "x", -1.0, 0.01)
        minimizer.SetVariable(1, "y", 1.2, 0.01)
        minimizer.Minimize()
        print("done")

    lines = split_run(monkeypatch, script).splitlines()
    assert lines[:2] == [
        "Minuit2Minimizer: Minimize with max-calls 1000000 convergence for edm < 0.001 strategy 1",
        "Minuit2Minimizer : Valid minimum - status = 0",
    ]
    assert lines[-2:] == ["f(-1,1.2) =  5.0", "done"]


def test_a_histograms_print_is_printf_and_follows_the_scripts_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TH1::Print`` is never flushed: C's 4 KiB buffer goes out when full, the rest last.

    ROOT printed this very sequence - 400 summaries, each followed by a line of
    the script's - as 36864 bytes of summaries (nine full buffers), the script's
    2800 bytes, then the last 336 bytes of summaries.
    """
    from xrdroot import pyroot as ROOT

    def script() -> None:
        h = ROOT.TH1F("h" * 48, "h", 10, 0, 1)
        for i in range(400):
            h.Print()
            print(f"py{i:04d}")

    out = split_run(monkeypatch, script)
    summary = "TH1.Print Name  = " + "h" * 48 + ", Entries= 0, Total sum= 0\n"
    summaries = summary * 400
    python = "".join(f"py{i:04d}\n" for i in range(400))
    assert out == summaries[:36864] + python + summaries[36864:]


def test_a_cut_flow_report_is_printed_after_the_scripts_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """df004_cutFlowReport.py: ROOT's ``Printf`` lines come after ``All stats:``."""
    from xrdroot.rdf.report import CutFlowReport, CutInfo

    def script() -> None:
        print("All stats:")
        CutFlowReport([CutInfo("Cut1", 24, 50)]).Print()
        CutFlowReport([]).Print()

    assert split_run(monkeypatch, script) == (
        "All stats:\n"
        "Cut1      : pass=24         all=50         -- eff=48.00 % cumulative eff=48.00 %\n"
    )


def test_roofits_unflushed_line_waits_for_the_next_flushed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rf602_chi2fit.py's order: a line ended ``"\\n"`` waits; ``std::endl``'s takes it along."""

    def script() -> None:
        cout.write_unflushed("held\n")
        print("mine")
        cout.line("flushed")
        cout.write_unflushed("last\n")

    assert split_run(monkeypatch, script) == "held\nflushed\nmine\nlast\n"


def test_flushing_is_the_writers_own_buffer_and_the_rest_is_pythons_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A script's ``flush`` empties Python's buffer; xrdroot's, C's; attributes are Python's."""

    root_side: dict[str, Any] = {"__name__": "xrdroot.somewhere"}
    exec("import sys\ndef flush():\n    sys.stdout.flush()\n", root_side)

    def script() -> None:
        print("first")
        sys.stdout.flush()
        stdio.printf("c side\n")
        print("second")
        root_side["flush"]()
        assert sys.stdout.encoding == "utf-8"

    assert split_run(monkeypatch, script) == "first\nc side\nsecond\n"


def test_on_a_terminal_or_a_stream_with_no_file_nothing_is_split(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Line buffering on both sides keeps a terminal's order; a test's capture has no pipe."""
    with stdio.split():
        assert not stdio.active()
        stdio.printf("plain\n")
        with stdio.unflushed():
            stdio.cwrite("still plain\n")
    assert capsys.readouterr().out == "plain\nstill plain\n"
    read, write = os.pipe()
    with io.TextIOWrapper(io.FileIO(write, "w")) as terminal:
        monkeypatch.setattr(os, "isatty", lambda fd: True)
        monkeypatch.setattr(sys, "stdout", terminal)
        with stdio.split():
            assert not stdio.active()
        monkeypatch.undo()
    os.close(read)
