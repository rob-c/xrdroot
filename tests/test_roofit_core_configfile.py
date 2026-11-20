"""``RooArgSet::writeToFile`` and ``readFromFile``: rf505's ASCII configuration files.

ROOT 6.40 writes one line per member - ``f =  0.50733 +/- 0.020971 L(0 - 1) `` -
at five significant figures by the error's precision, and reads the constructs
of the format back: sections, ``include``, ``echo``, ``abort``, ``if``/``else``/
``endif``, the deprecated ``P(...)`` and ``F(...)``, and ROOT's own diagnostics.
Every expectation here was observed from ROOT 6.40.04 reading the same text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.configfile import read_text, write_text
from xrdroot.roofit.variables import RooConstVar, RooRealVar


def _fitted() -> RooArgSet:
    """rf505's parameters, at the values ROOT's own fit of its toy sample found."""
    params = _params()
    for name, value, error in (
        ("f", 0.507326, 0.0209708),
        ("m", 0.0064018, 0.053686),
        ("p0", 0.00735091, 0.00783124),
        ("s", 0.965161, 0.0470524),
    ):
        params.find(name).setVal(value)
        params.find(name).setError(error)
    return params


def _params() -> RooArgSet:
    """The set rf505 reads: its four parameters, as the tutorial starts them."""
    return RooArgSet(
        [
            RooRealVar("f", "f", 0.5, 0.0, 1.0),
            RooRealVar("m", "m", 0.0, -10, 10),
            RooRealVar("p0", "p0", 0.01, 0.0, 1.0),
            RooRealVar("s", "s", 1.0, -10, 10),
        ],
        "parameters",
    )


def _read(collection: Any, text: str, *args: Any) -> bool:
    """``readFromFile``'s reading of ``text``, as the file would hold it."""
    return read_text(collection, text, *args)


# -- what is written ---------------------------------------------------------------


def test_a_fitted_set_is_written_as_root_writes_rf505s_file(tmp_path: Path) -> None:
    path = tmp_path / "rf505_asciicfg_example.txt"
    _fitted().writeToFile(str(path))
    assert path.read_text() == (
        "f =  0.50733 +/- 0.020971 L(0 - 1) \n"
        "m =  0.0064018 +/- 0.053686 L(-10 - 10) \n"
        "p0 =  0.0073509 +/- 0.0078312 L(0 - 1) \n"
        "s =  0.96516 +/- 0.047052 L(-10 - 10) \n"
    )


def test_a_variable_without_a_range_is_written_with_roots_infinities() -> None:
    free = RooRealVar("q", "q", 1.5)
    free.setError(-1.0)
    assert write_text(RooArgSet(free)) == "q =  1.5000 C L(-INF - +INF) \n"


def test_a_constant_variable_below_a_hundred_bins_is_written_with_its_bins() -> None:
    cut = RooRealVar("cut", "cut", 0.5, 0.0, 1.0, "GeV")
    cut.setConstant(True)
    cut.setBins(50)
    assert write_text(RooArgSet(cut)) == "cut =  0.50000 C L(0 - 1) B(50) // [GeV]\n"


def test_an_asymmetric_error_is_written_in_parentheses_with_a_space_after_the_sign() -> None:
    var = RooRealVar("a", "a", 1.0, 0.0, 10.0)
    var.setVal(1.23456)
    var.setAsymError(-0.1, 0.2)
    assert write_text(RooArgSet(var)) == "a =  1.2346 +/-  (-0.10000, 0.20000) L(0 - 10) \n"


def test_a_category_is_written_by_its_label() -> None:
    kind = RooCategory("kind", "kind")
    kind.defineType("Plus", 1)
    kind.setLabel("Plus")
    assert write_text(RooArgSet(kind)) == "kind = Plus\n"


def test_a_variable_that_has_no_form_in_the_format_is_refused_by_name() -> None:
    const = RooConstVar("c", "c", 2.0)
    with pytest.raises(UnsupportedFeatureError) as refused:
        write_text(RooArgSet(const))
    assert "'c' is a RooConstVar, which has no form in them here" in str(refused.value)
    with pytest.raises(UnsupportedFeatureError) as reading:
        read_text(RooArgSet(const), "c = 2\n")
    assert "'c' is a RooConstVar, which has no form in them here" in str(reading.value)


def test_a_file_that_cannot_be_written_is_reported_as_root_reports_it(
    tmp_path: Path, capsys: Any
) -> None:
    _params().writeToFile(str(tmp_path / "nowhere" / "f.txt"))
    assert capsys.readouterr().out == (
        f"[#0] ERROR:InputArguments -- RooArgSet::writeToFile(parameters) error opening file "
        f"{tmp_path / 'nowhere' / 'f.txt'}\n"
    )


# -- what is read ------------------------------------------------------------------


def test_a_written_set_reads_back_to_the_values_it_was_written_with() -> None:
    original = _fitted()
    written = write_text(original)
    copy = RooArgSet([one.clone(one.GetName()) for one in original], "parameters")
    for one in copy:
        one.setVal(0.0)
        one.setError(-1.0)
    assert read_text(copy, written) is False
    for before, after in zip(original, copy, strict=True):
        assert after.getVal() == pytest.approx(before.getVal(), rel=1e-4)
        assert after.getError() == pytest.approx(before.getError(), rel=1e-4)


def test_a_set_reads_a_line_an_argument_and_sets_the_value_it_names() -> None:
    params = _params()
    assert _read(params, "f = 0.3\nm = 0.7\n") is False
    assert (params.find("f").getVal(), params.find("m").getVal()) == (0.3, 0.7)


def test_a_line_reads_the_value_the_error_the_constancy_the_range_and_the_bins() -> None:
    params = _params()
    assert _read(params, "f = 0.5 +/- 0.1 C B(50)\n") is False
    f = params.find("f")
    assert (f.getVal(), f.getError(), f.isConstant(), f.getBins()) == (0.5, 0.1, True, 50)


def test_a_range_reads_a_constant_variable_free_unless_c_said_it_was_constant() -> None:
    params = _params()
    params.find("f").setConstant(True)
    assert _read(params, "f = 0.5 C L(-5 - 5)\n") is False
    assert (params.find("f").isConstant(), params.find("f").getMin()) == (True, -5.0)
    assert _read(params, "f = 0.5 L(-5 - 5)\n") is False
    assert (params.find("f").isConstant(), params.find("f").getMax()) == (False, 5.0)


def test_an_asymmetric_error_reads_from_parentheses() -> None:
    params = _params()
    assert _read(params, "f = 0.5 +/- (0.1, 0.3)\n") is False
    f = params.find("f")
    assert (f.getAsymErrorLo(), f.getAsymErrorHi()) == (0.0, 0.3)


def test_an_error_signs_characters_may_be_written_with_blanks_between_them() -> None:
    for sign in ("+/-", "+/ -", "+ /-", "+ / -"):
        params = _params()
        assert _read(params, f"f = 0.5 {sign} 0.3\n") is False
        assert params.find("f").getError() == pytest.approx(0.3)


def test_a_line_that_carries_no_value_leaves_the_value_alone() -> None:
    params = _params()
    assert _read(params, "f = L(-5 - 5)\n") is False
    assert (params.find("f").getVal(), params.find("f").getMin()) == (0.5, -5.0)


def test_the_deprecated_plot_and_fit_ranges_are_read_and_warned_about(capsys: Any) -> None:
    params = _params()
    capsys.readouterr()
    assert _read(params, "f = 0.5 P(-5 - 5 : 20)\n") is False
    assert _read(params, "f = 0.5 F(-5 - 5 : 20)\n") is False
    assert capsys.readouterr().out == (
        "[#0] WARNING:Eval -- RooRealVar::readFromStream(f) WARNING: plot range deprecated, "
        "removed P(...) token\n"
        "[#0] WARNING:Eval -- RooRealVar::readFromStream(f) WARNING: F(lo-hi:bins) token "
        "deprecated, use L(lo-hi) B(bins)\n"
    )


def test_a_deprecated_fit_range_let_a_constant_variable_free() -> None:
    params = _params()
    params.find("f").setConstant(True)
    assert _read(params, "f = 0.5 F(-5 - 5 : 20)\n") is False
    assert params.find("f").isConstant() is False


def test_a_token_out_of_its_place_ends_the_line_with_a_warning(capsys: Any) -> None:
    params = _params()
    capsys.readouterr()
    assert _read(params, "f = 0.5 L 3\n") is False
    assert params.find("f").getVal() == 0.5
    assert capsys.readouterr().out == (
        "[#0] WARNING:InputArguments -- RooRealVar::readFromStream(f): parse error, "
        "expected '(', got '3'\n"
    )


@pytest.mark.parametrize(
    "text",
    [
        "f = 0.5 L(0 3)\n",
        "f = 0.5 L(0 - 3\n",
        "f = 0.5 B(3\n",
        "f = 0.5 P(0 - 3 20)\n",
        "f = 0.5 P 0\n",
        "f = 0.5 B 3\n",
    ],
)
def test_a_shape_that_does_not_fit_its_letter_leaves_the_rest_of_the_line(text: str) -> None:
    params = _params()
    assert _read(params, text) is False
    assert params.find("f").getBins() == 100


def test_a_value_is_read_as_root_reads_a_number_and_infinity_is_not_one() -> None:
    params = _params()
    assert _read(params, "f = 1e-1\n") is False
    assert params.find("f").getVal() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Conversion to floating point failed"):
        _read(params, "f = 0.5 L(-INF - +INF)\n")
    with pytest.raises(ValueError, match="Conversion to floating point failed"):
        _read(params, "f = nonsense\n")


def test_a_category_reads_a_states_number_or_its_label() -> None:
    kind = RooCategory("kind", "kind")
    kind.defineType("Plus", 1)
    kind.defineType("Minus", -1)
    params = RooArgSet(kind)
    assert _read(params, "kind = Minus\n") is False
    assert kind.getLabel() == "Minus"
    assert _read(params, "kind = 1\n") is False
    assert kind.getLabel() == "Plus"
    assert _read(params, "kind = 7\n") is True  # ROOT answers an unset state as an error
    assert kind.getLabel() == "Plus"


# -- the constructs of the format ---------------------------------------------------


def test_only_the_section_asked_for_is_read_and_a_name_may_hold_a_space() -> None:
    text = "[Section1]\nf = 0.1\n[Section 2]\nf = 0.2\n[Section3]\nf = 0.3\n"
    params = _params()
    assert _read(params, text, "", "Section 2") is False
    assert params.find("f").getVal() == 0.2
    assert _read(params, text, "", "Section3") is False
    assert params.find("f").getVal() == 0.3


def test_a_section_that_is_not_the_one_asked_for_is_left_alone() -> None:
    params = _params()
    assert _read(params, "[Other]\nf = 0.3\n", "", "Wanted") is False
    assert params.find("f").getVal() == 0.5


def test_every_section_is_read_when_no_section_is_asked_for() -> None:
    params = _params()
    assert _read(params, "[A]\nf = 0.1\n[B]\nf = 0.2\n") is False
    assert params.find("f").getVal() == 0.2


def test_an_included_file_is_read_where_the_statement_stands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    (tmp_path / "more.txt").write_text("m = 0.25\n")
    monkeypatch.chdir(tmp_path)
    params = _params()
    capsys.readouterr()
    assert _read(params, "include more.txt\nf = 0.3\n") is False
    assert (params.find("m").getVal(), params.find("f").getVal()) == (0.25, 0.3)
    assert capsys.readouterr().out == (
        "[#1] INFO:InputArguments -- RooArgSet::readFromStream(parameters): processing "
        "include file more.txt\n"
    )


def test_an_include_of_a_section_reads_the_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "more.txt").write_text("[Wanted]\nm = 0.25\n[Other]\nf = 0.3\n")
    monkeypatch.chdir(tmp_path)
    params = _params()
    assert _read(params, "[Wanted]\ninclude more.txt\n", "", "Wanted") is False
    assert (params.find("m").getVal(), params.find("f").getVal()) == (0.25, 0.3)


def test_an_include_outside_the_wanted_section_keeps_the_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "more.txt").write_text("m = 0.25\n")
    monkeypatch.chdir(tmp_path)
    params = _params()
    assert _read(params, "include more.txt\n", "", "Wanted") is False
    assert params.find("m").getVal() == 0.0


def test_an_include_that_cannot_be_read_ends_the_reading(capsys: Any) -> None:
    params = _params()
    capsys.readouterr()
    assert _read(params, "include nowhere.txt\nf = 0.3\n") is True
    assert _read(params, "include\nf = 0.3\n") is True
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): cannot open "
        "include file nowhere.txt\n"
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): no filename "
        "found after include statement\n"
    )
    assert params.find("f").getVal() == 0.5


def test_an_echo_is_printed_and_an_abort_stops_the_reading(capsys: Any) -> None:
    params = _params()
    capsys.readouterr()
    assert _read(params, "echo hello world\nf = 0.3\n") is False
    assert params.find("f").getVal() == 0.3
    assert _read(params, "abort now\nf = 0.1\n") is True
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): >> hello world\n"
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): USER ABORT\n"
    )
    assert params.find("f").getVal() == 0.3


def test_a_conditional_reads_the_branch_its_condition_selects() -> None:
    params = _params()
    text = "if (f>0.1)\nf = 0.2\nelse\nf = 0.9\nendif\n"
    assert _read(params, text) is False
    assert params.find("f").getVal() == 0.2


def test_an_else_if_chain_reads_the_first_branch_that_is_true() -> None:
    chain = "if (f>0.6)\nf = 0.2\nelse if (f>0.4)\nf = 0.3\nelse\nf = 0.9\nendif\n"
    for start, wanted in ((0.5, 0.3), (0.5, 0.3), (0.1, 0.9)):
        params = _params()
        params.find("f").setVal(start)
        assert _read(params, chain) is False
        assert params.find("f").getVal() == wanted
    params = _params()
    params.find("f").setVal(0.7)
    assert _read(params, chain) is False
    assert params.find("f").getVal() == 0.2


def test_an_unclosed_conditional_ends_the_reading_with_roots_message(capsys: Any) -> None:
    params = _params()
    capsys.readouterr()
    assert _read(params, "if (f>0.6)\nf = 0.2\n") is True
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): missing 'endif'\n"
    )


def test_an_unmatched_ender_says_so_and_only_an_endif_ends_the_reading(capsys: Any) -> None:
    params = _params()
    capsys.readouterr()
    assert _read(params, "endif\nf = 0.3\n") is True
    assert _read(params, "else\nf = 0.4\n") is False
    assert _read(params, "else while\nf = 0.4\n") is True
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): unmatched 'endif'\n"
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): unmatched 'else'\n"
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): unmatched 'else'\n"
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): syntax error: "
        "'else while'\n"
    )


def test_an_else_is_skipped_when_the_branch_before_it_was_read() -> None:
    params = _params()
    params.find("f").setVal(0.9)
    assert _read(params, "if (f>0.6)\nf = 0.2\nelse\nf = 0.4\nendif\n") is False
    assert params.find("f").getVal() == 0.2


def test_a_line_without_its_equals_sign_is_an_error(capsys: Any) -> None:
    params = _params()
    capsys.readouterr()
    assert _read(params, "f\n") is True
    message = capsys.readouterr().out
    assert message.startswith(
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): missing '=' sign: 0x"
    )


def test_a_name_the_set_does_not_hold_is_ignored_and_said_so_when_verbose(capsys: Any) -> None:
    params = _params()
    capsys.readouterr()
    assert _read(params, "zzz = 3\nf = 0.3\n") is False
    assert capsys.readouterr().out == ""
    params.find("f").setVal(0.5)
    before = params.find("m").getVal()
    params.find("m").setAttribute("Dynamic")
    assert _read(params, "m = 3\n", "", None, True) is False
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooArgSet::readFromStream(parameters): argument m not in "
        "list, ignored\n"
    )
    assert params.find("m").getVal() == before


def test_the_flag_marks_the_arguments_lines_of_the_file_set() -> None:
    params = _params()
    assert _read(params, "f = 0.3\n", "READ") is False
    assert len(params.selectByAttrib("READ", False)) == 3
    assert [one.GetName() for one in params.selectByAttrib("READ", True)] == ["f"]


def test_comments_and_blank_lines_are_not_read() -> None:
    params = _params()
    assert _read(params, "# a comment\n\nf = 0.3 // and a trailing one\n") is False
    assert params.find("f").getVal() == 0.3


def test_a_line_continued_by_a_double_backslash_continues_the_value() -> None:
    params = _params()
    assert _read(params, "f = 0.3 \\\\\n C\n") is False
    assert (params.find("f").getVal(), params.find("f").isConstant()) == (0.3, True)


def test_a_file_that_cannot_be_read_is_reported_as_root_reports_it(
    tmp_path: Path, capsys: Any
) -> None:
    params = _params()
    capsys.readouterr()
    assert params.readFromFile(str(tmp_path / "nowhere.txt")) is True
    assert capsys.readouterr().out == (
        f"[#0] ERROR:InputArguments -- RooArgSet::readFromFile(parameters) error opening file "
        f"{tmp_path / 'nowhere.txt'}\n"
    )


def test_a_set_reads_its_file_and_marks_what_it_read(tmp_path: Path) -> None:
    path = tmp_path / "p.txt"
    path.write_text("f = 0.4\n")
    params = _params()
    assert params.readFromFile(str(path), "READ") is False
    assert params.find("f").getAttribute("READ") is True


def test_a_file_whose_last_line_is_continued_reads_that_line() -> None:
    params = _params()
    assert _read(params, "f = 0.3 \\\\") is False
    assert params.find("f").getVal() == 0.3


def test_an_error_whose_sign_is_not_written_is_a_parse_error() -> None:
    params = _params()
    assert _read(params, "f = 0.5 + 3\n") is False
    assert params.find("f").getError() == 0.0


def test_an_asymmetric_error_that_does_not_fit_its_parentheses_is_a_parse_error() -> None:
    params = _params()
    for text in ("f = 0.5 +/- (0.1 0.3)\n", "f = 0.5 +/- (0.1, 0.3\n"):
        assert _read(params, text) is False
        assert params.find("f").getAsymErrorHi() == 0.0


def test_a_deprecated_fit_range_of_the_wrong_shape_is_a_parse_error() -> None:
    params = _params()
    params.find("f").setConstant(True)
    assert _read(params, "f = 0.5 F 3\n") is False
    assert _read(params, "f = 0.5 F(0 5 : 20)\n") is False
    assert params.find("f").isConstant() is True


def test_a_deprecated_fit_range_leaves_an_already_constant_variable_as_it_was() -> None:
    params = _params()
    params.find("f").setConstant(True)
    assert _read(params, "f = 0.5 C F(-5 - 5 : 20)\n") is False
    assert params.find("f").isConstant() is True
