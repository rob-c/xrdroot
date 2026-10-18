"""Annotated pave text files compared with ROOT 6.40.04 batch controls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh

#: Values captured on ROOT 6.40.04, including its float-sized text attributes.
REFERENCE = json.loads((Path(__file__).parent / "data/pave-readfile-6.40.json").read_text())
#: Input text and the optional line selection used for the native controls.
CASES = [
    ("all", "one\n\nthree\n", "", 50, 0),
    ("append", "one\ntwo\n", "+", 50, 0),
    ("range", "zero\none\ntwo\nthree\n", "", 2, 1),
    ("unterminated", "one\ntwo", "", 50, 0),
    ("commands", "+SetTextColor(2)\none\n+SetTextColor(4)\n+SetTextAlign(13)\n"
     "+SetTextFont(42)\n+SetTextSize(.17)\n+SetTextAngle(12.5)\ntwo\n", "", 50, 0),
    ("unknown", "one\n+SetTextUnknown(2)\n+SetTextColor(2\nlast\n", "", 50, 0),
    ("semicolon", "one\n", "", 50, 0),
    ("empty", "", "", 50, 0),
]


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _record(pave):
    return {
        "align": pave.GetTextAlign(),
        "lines": [{"text": line.GetTitle(), "align": line.GetTextAlign(),
                   "color": line.GetTextColor(), "font": line.GetTextFont(),
                   "size": line.GetTextSize(), "angle": line.GetTextAngle()}
                  for line in pave.GetListOfLines()],
    }


@pytest.mark.parametrize("name,text,option,nlines,start", CASES)
def test_a_pave_reads_the_same_lines_and_attributes_as_root(name, text, option, nlines, start,
                                                         tmp_path):
    path = tmp_path / "text.txt"
    path.write_text(text)
    pave = ROOT.TPaveText()
    pave.AddText("old")
    pave.ReadFile(str(path) + (";" if name == "semicolon" else ""), option, nlines, start)
    expected = next(row for row in REFERENCE if row["name"] == name)
    # These are lexical values and IEEE float conversions, independent of platform libm.
    assert _record(pave) == {key: value for key, value in expected.items() if key != "name"}


@pytest.mark.parametrize("filename", ["", "missing"])
def test_an_empty_or_missing_file_clears_the_pave_and_sets_left_alignment(filename, capsys):
    pave = ROOT.TPaveText()
    pave.AddText("old")
    pave.ReadFile(filename)
    assert _record(pave) == {"align": 12, "lines": []}
    assert capsys.readouterr().err == (
        "Error in <TPaveText::ReadFile>: illegal file name missing\n" if filename else "")


def test_an_annotated_line_does_not_consume_the_selected_line_count(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("one\n+SetTextColor(4)\ntwo\nthree\n")
    pave = ROOT.TPaveText()
    pave.ReadFile(str(path), "", 2)
    assert [line.GetTitle() for line in pave.GetListOfLines()] == ["one", "two"]
    assert pave.GetLine(0).GetTextColor() == 4


def test_an_unreadable_appended_file_keeps_its_existing_lines(capsys):
    pave = ROOT.TPaveText()
    pave.AddText("old")
    pave.ReadFile("missing", "+")
    assert [line.GetTitle() for line in pave.GetListOfLines()] == ["old"]
    assert "illegal file name missing" in capsys.readouterr().err


@pytest.mark.parametrize("name,text,nlines", [
    ("invalid_after_valid", "one\n+SetTextColor(4)\n+SetTextFont()\ntwo\n", 50),
    ("crlf", "one\r\ntwo\r\n", 50),
    ("range_command", "one\n+SetTextColor(4)\ntwo\nthree\n", 2),
])
def test_control_lines_and_carriage_returns_follow_roots_file_reader(name, text, nlines, tmp_path):
    path = tmp_path / "text.txt"
    path.write_bytes(text.encode())
    pave = ROOT.TPaveText()
    pave.ReadFile(str(path), "", nlines)
    path = Path(__file__).parent / "data/pave-readfile-extra-6.40.json"
    reference = json.loads(path.read_text())
    expected = next(row["lines"] for row in reference if row["name"] == name)
    assert [{"text": line.GetTitle(), "color": line.GetTextColor(), "font": line.GetTextFont()}
            for line in pave.GetListOfLines()] == expected
