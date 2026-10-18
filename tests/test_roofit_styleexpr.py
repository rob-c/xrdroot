"""String styles expose the same expression results as ROOT's interpreter."""

from __future__ import annotations

import pytest

from xrdroot.roofit.cmdargs import RooCmdArg


@pytest.mark.parametrize("command", ["LineStyle", "FillStyle", "MarkerStyle"])
@pytest.mark.parametrize(("text", "value", "display"), [
    ("kDashed", 2, "(ELineStyle) (kDashed) : (unsigned int) 2"),
    ("kDotted", 3, "(ELineStyle) (kDotted) : (unsigned int) 3"),
    ("kFullCircle", 20, "(EMarkerStyle) (kFullCircle) : (unsigned int) 20"),
    ("kRed", 632, "(EColor) (kRed) : (unsigned int) 632"),
    (" kDashed ", 2, "(ELineStyle) (kDashed) : (unsigned int) 2"),
    ("kDashed + 0", 2, "(int) 2"),
    ("kDashed+1", 3, "(int) 3"),
    ("2", 2, "(int) 2"),
    ("3004", 3004, "(int) 3004"),
    ("kDashed;", 2, ""),
])
def test_style_expression_values_and_printing_match_roots_interpreter(command, text, value,
                                                                      display, capsys):
    assert RooCmdArg(command, text).value() == value
    assert capsys.readouterr().out == (display + "\n" if display else "")


@pytest.mark.parametrize(("text", "value"), [("-", 1), ("--", 2), (":", 3), ("-.", 4)])
def test_matplotlib_line_styles_bypass_interpreter_display(text, value, capsys):
    assert RooCmdArg("LineStyle", text).value() == value
    assert capsys.readouterr().out == ""


def test_color_name_conversion_does_not_display_an_interpreter_result(capsys):
    assert RooCmdArg("LineColor", "kRed+1").value() == 633
    assert capsys.readouterr().out == ""
