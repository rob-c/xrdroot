"""Lines RooFit ends with ``"\\n"`` rather than ``std::endl``: held until C++ next flushes."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from xrdroot.roofit import cout


class Stream:
    """A Python stream that is not a terminal: a pipe's, as the tutorials run into."""

    def __init__(self) -> None:
        self.text = ""

    def write(self, text: str) -> None:
        self.text += text

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def under_pyroot(monkeypatch: pytest.MonkeyPatch) -> Stream:
    stream = Stream()
    monkeypatch.setitem(sys.modules, "ROOT", types.ModuleType("xrdroot.pyroot"))
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "__stdout__", stream)
    monkeypatch.setattr(cout, "PENDING", [])
    monkeypatch.setattr(cout.atexit, "register", lambda fn: None)
    return stream


def test_an_unflushed_line_goes_out_with_the_next_flushed_one(
    monkeypatch: pytest.MonkeyPatch, capfd: Any
) -> None:
    """``createChi2``'s line waits, then leaves with the next ``std::endl``'s, in order."""
    under_pyroot(monkeypatch)
    cout.STREAM.write_unflushed("held\n")
    cout.STREAM.write_unflushed("and held\n")
    assert capfd.readouterr().out == ""
    cout.line("flushed")
    assert capfd.readouterr().out == "held\nand held\nflushed\n"


def test_what_is_still_held_at_the_end_follows_pythons_own_output(
    monkeypatch: pytest.MonkeyPatch, capfd: Any
) -> None:
    """At exit Python's buffer goes first, then what C++ still held - the tutorials' order."""
    under_pyroot(monkeypatch)
    cout.write_unflushed("last\n")
    cout._at_exit()
    cout._at_exit()  # nothing more is held the second time
    assert (capfd.readouterr().out, cout.PENDING) == ("last\n", [])


def test_without_pyroot_an_unflushed_line_is_printed_at_once(capsys: Any) -> None:
    """In a test, or a translated macro, there is no C++ buffer to wait in."""
    cout.write_unflushed("now\n")
    assert capsys.readouterr().out == "now\n"
