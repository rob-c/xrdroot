"""``ROOT::Split``, ``ROOT::Join``, ``ROOT::Round`` and the environment-flag readers.

ROOT's own text helpers, as ``ROOT/StringUtils.hxx`` and ``FoundationUtils.hxx`` define
them: the tests here are the ones ROOT's ``testStringUtils.cxx`` and
``FoundationUtilsTests.cxx`` ask, plus the shapes the translator reaches for.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


def test_split_cuts_at_every_delimiter_and_keeps_the_empty_pieces() -> None:
    assert ROOT.Split("a,b,c", ",") == ["a", "b", "c"]
    assert ROOT.Split("a,,c", ",") == ["a", "", "c"]
    assert ROOT.Split(",,a,,,", ",") == ["", "", "a", "", "", ""]
    assert ROOT.Split("", ",") == [""]
    assert ROOT.Split("a;b,c", ";,") == ["a", "b", "c"]


def test_split_skips_the_empty_pieces_when_it_is_asked_to() -> None:
    assert ROOT.Split("a,,c", ",", True) == ["a", "c"]
    assert ROOT.Split(",,,", ",", True) == []
    assert ROOT.Split("", "", True) == []


def test_join_puts_the_separator_between_the_pieces() -> None:
    assert ROOT.Join(",", ["apple", "orange"]) == "apple,orange"
    assert ROOT.Join("", ["apple", "orange"]) == "appleorange"
    assert ROOT.Join(",", ["apple", "", "banana"]) == "apple,,banana"
    assert ROOT.Join(",", []) == ""
    assert ROOT.Join(",", ["apple"]) == "apple"


@pytest.mark.parametrize(
    ("value", "error", "cutoff", "delim", "text"),
    [
        (0.000000014, 0.000000024, 1, "#pm", "(10#pm20)*1e-9"),
        (110.0, 0.24, 1, "#pm", "110.0#pm0.2"),
        (120.0, 0.24, 2, "#pm", "120.00#pm0.24"),
        (-140.0, 0.95, 1, "#pm", "-140.0#pm1.0"),
        (150.0, 0.114, 1, "#pm", "150.00#pm0.11"),
        (170.0, 0.195, 1, "#pm", "170.0#pm0.2"),
        (-190.0, 0.95, 0, "#pm", "-190#pm1"),
        (-0.001, 9.5, 1, "#pm", "-0#pm10"),
        (24.0, 19.4, 1, "#pm", "24#pm19"),
        (-25.0, 19.5, 1, "#pm", "-30#pm20"),
        (-25.0, 21, 9, "#pm", "-25#pm21"),
        (1.0, 101.4, 1, "#pm", "0#pm100"),
        (-1.0, 109.4, 1, "#pm", "-0#pm110"),
        (-31.0, 119.5, 1, "#pm", "-30#pm120"),
        (-3030.0, 195, 1, "#pm", "-3000#pm200"),
        (-1200.0, 2000, 1, "#pm", "(-1#pm2)*1e3"),
        (101.0, 2000, 2, "#pm", "(0.1#pm2.0)*1e3"),
        (-30000.0, 2000000000.0, 2, "#pm", "(-0.0#pm2.0)*1e9"),
        (-30000.0, 1000000000.0, 0, "#pm", "(-0#pm1)*1e9"),
        (110.0, 0.24, 1, "+-", "110.0+-0.2"),
    ],
)
def test_round_prints_the_value_and_its_error_with_the_digits_roof_rounds_to(
    value: float, error: float, cutoff: int, delim: str, text: str
) -> None:
    assert ROOT.Round(value, error, cutoff, delim) == text


def test_a_round_with_no_error_to_round_to_prints_every_digit() -> None:
    assert ROOT.Round(110.0, 0.0) == "110.000000"
    assert ROOT.Round(-2.0, -1.0) == "-2.000000"


@pytest.mark.parametrize("text", ["0", "false", "False", "OFF", "1", "true", "On"])
def test_any_of_the_six_flags_can_be_read_as_a_boolean(text: str) -> None:
    assert ROOT.CanConvertEnvValueToBool(text) is True


def test_a_word_that_is_no_flag_cannot_be_read_as_one() -> None:
    assert ROOT.CanConvertEnvValueToBool("") is False
    assert ROOT.CanConvertEnvValueToBool("yes") is False


def test_a_flag_converts_to_the_boolean_it_names_in_any_case() -> None:
    assert ROOT.ConvertEnvValueToBool("TruE") is True
    assert ROOT.ConvertEnvValueToBool("oN") is True
    assert ROOT.ConvertEnvValueToBool("FalSe") is False
    assert ROOT.ConvertEnvValueToBool("oFf") is False
    with pytest.raises(ValueError, match="'yes' is not a boolean"):
        ROOT.ConvertEnvValueToBool("yes")
