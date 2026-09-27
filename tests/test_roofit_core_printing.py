"""``RooPrintable`` and ``std::cout``: the parts of a printed line, and where it goes.

Every RooFit object prints through the same assembly of address, class,
name, arguments, value, extras and title; these tests take the parts one at
a time on an object that names each part after itself. Numbers print as
``std::cout`` prints a ``double``: six significant figures, ``-nan`` for a
NaN with its sign bit set, as ROOT 6.40.04 does.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from xrdroot.roofit import cout
from xrdroot.roofit.printing import (
    PRECISION,
    RooPrintable,
    g,
    kAddress,
    kArgs,
    kClassName,
    kExtras,
    kInline,
    kName,
    kSingleLine,
    kStandard,
    kTitle,
    kTreeStructure,
    kValue,
    kVerbose,
    style_of,
)


class Parts(RooPrintable):
    """A printable whose every part is its own name, so each can be seen in place."""

    def printName(self) -> str:
        return "name"

    def printTitle(self) -> str:
        return "title"

    def printClassName(self) -> str:
        return "Class"

    def printArgs(self) -> str:
        return "[args]"

    def printValue(self) -> str:
        return "value"

    def printExtras(self) -> str:
        return "extras"

    def printAddress(self) -> str:
        return "@"

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        return f"multi {contents} {verbose} {indent!r}"


def test_a_double_prints_to_six_significant_figures() -> None:
    """``os << value``: ``%g`` at six digits, or the digits asked for."""
    assert g(0.8824969025845955) == "0.882497"
    assert g(1234567.0) == "1.23457e+06"
    assert g(2.0) == "2"
    assert g(3.14159265, 3) == "3.14"


def test_a_nan_prints_with_its_sign_as_cpp_does() -> None:
    """glibc and libc++ print a NaN with the sign bit set as ``-nan``."""
    assert g(float("nan")) == "nan"
    assert g(-float("nan")) == "-nan"


def test_the_precision_of_cout_can_be_changed_for_the_rest_of_the_process() -> None:
    """RooFit's verbose minimisation leaves ``std::cout`` at four digits; printing follows."""
    PRECISION[0] = 4
    try:
        assert g(0.8824969025845955) == "0.8825"
    finally:
        PRECISION[0] = 6


def test_the_option_letters_pick_the_style_in_roots_order() -> None:
    """``v`` wins over ``s``, which wins over ``i``, then ``t``; nothing is one line."""
    assert style_of("V") == kVerbose
    assert style_of("sv") == kVerbose
    assert style_of("s") == kStandard
    assert style_of("i") == kInline
    assert style_of("t") == kTreeStructure
    assert style_of("") == kSingleLine
    assert style_of(None) == kSingleLine


def test_a_bare_printable_prints_nothing_but_its_address() -> None:
    """The defaults of every part are empty; the address is ``os << this``."""
    bare = RooPrintable()
    assert bare.printName() == bare.printTitle() == bare.printClassName() == ""
    assert bare.printArgs() == bare.printValue() == bare.printExtras() == ""
    assert bare.printAddress() == hex(id(bare))
    assert bare.printMultiline(kName, False, "") == ""
    assert bare.defaultPrintContents("") == kName | kValue
    assert bare.defaultPrintStyle("v") == kVerbose
    assert str(bare) == " = "


def test_a_class_without_a_tree_says_so_as_root_does() -> None:
    """``Print("t")`` on a class that has no tree printing is ROOT's complaint."""
    assert RooPrintable().printStream(kName, kTreeStructure) == (
        "Tree structure printing not implement for class RooPrintable\n"
    )


def test_the_standard_and_verbose_styles_print_over_several_lines() -> None:
    """``s`` and ``v`` hand the object's own multi-line printing the contents and the indent."""
    parts = Parts()
    assert parts.printStream(kName, kStandard, "  ") == "multi 1 False '  '"
    assert parts.printStream(kName, kVerbose) == "multi 1 True ''"


def test_every_part_takes_its_place_in_one_line() -> None:
    """Address, class, name, arguments, value, extras, title - with ROOT's separators."""
    everything = kAddress | kClassName | kName | kArgs | kValue | kExtras | kTitle
    assert Parts().printStream(everything, kSingleLine, "> ") == (
        '> @ Class::name[args] = value extras "title"\n'
    )
    assert Parts().printStream(kName | kValue, kInline, "> ") == "name = value"


def test_a_part_printed_alone_has_no_separator() -> None:
    """Each part on its own is bare: no ``::``, no `` = ``, no quotes."""
    parts = Parts()
    alone = {kAddress: "@", kClassName: "Class", kValue: "value", kExtras: "extras"}
    for contents, text in alone.items():
        assert parts.printStream(contents, kInline) == text
    assert parts.printStream(kTitle, kInline) == "title"


def test_names_are_right_aligned_while_a_width_is_set() -> None:
    """A collection sets the width its members' names print in."""
    RooPrintable.name_length[0] = 7
    try:
        assert Parts().printStream(kName, kInline) == "   name"
    finally:
        RooPrintable.name_length[0] = 0


def test_print_writes_the_default_contents_in_the_style_asked_for(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``Print()`` is ``printStream`` of the class's default contents, to standard output."""
    Parts().Print()
    Parts().Print("s")
    assert capsys.readouterr().out == "name = value\nmulti 5 False ''"


class Stream:
    """A standard output that remembers what was written and says whether it is a terminal."""

    def __init__(self, tty: bool) -> None:
        self.tty = tty
        self.text = ""

    def write(self, text: str) -> None:
        self.text += text

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return self.tty


def pyroot(monkeypatch: pytest.MonkeyPatch, stream: Any) -> None:
    """Make it look as if a PyROOT script is running with ``stream`` as its own output."""
    monkeypatch.setitem(sys.modules, "ROOT", types.ModuleType("xrdroot.pyroot"))
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "__stdout__", stream)


def test_under_pyroot_into_a_pipe_cout_writes_past_pythons_buffer(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A PyROOT script into a pipe gets RooFit's lines before its own buffered ones."""
    stream = Stream(tty=False)
    pyroot(monkeypatch, stream)
    cout.line("direct")
    assert stream.text == ""
    assert capfd.readouterr().out == "direct\n"


def test_under_pyroot_on_a_terminal_cout_is_pythons_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a terminal the order is the same either way, so Python's stream is used."""
    stream = Stream(tty=True)
    pyroot(monkeypatch, stream)
    cout.STREAM.write("on the terminal")
    cout.STREAM.flush()
    assert stream.text == "on the terminal"


def test_a_root_module_that_is_not_xrdroots_leaves_cout_as_pythons_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only xrdroot's own ``ROOT`` changes where RooFit prints."""
    stream = Stream(tty=False)
    monkeypatch.setitem(sys.modules, "ROOT", types.ModuleType("ROOT"))
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "__stdout__", stream)
    cout.line()
    assert stream.text == "\n"
