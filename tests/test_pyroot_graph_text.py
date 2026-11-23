"""``TGraph::SaveAs``'s text files, and the constructors that read them back.

ROOT writes a graph's points as ``.csv``, ``.tsv`` or ``.txt`` - a comma, a tab and a space
between the columns - and reads one of those back with ``TGraph("points.csv")``. The files
here are what the 6.40 oracle writes for the same graph, which is what makes a spreadsheet
and ROOT's own reader agree about one.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot import graph as engine
from xrdroot.errors import UnsupportedFeatureError

#: The points ROOT's own ``test_TGraph_SaveAs`` is written over.
X = [1.0, 2.0, 3.0, 4.0, 5.0]
Y = [5.0, 4.0, 3.0, 2.0, 1.0]
EX = [0.2, 0.3, 0.4, 0.5, 0.6]
EY = [0.6, 0.5, 0.4, 0.3, 0.2]
EXL = [0.25, 0.35, 0.45, 0.55, 0.65]
EXH = [0.15, 0.25, 0.35, 0.45, 0.55]
EYL = [0.67, 0.57, 0.47, 0.37, 0.27]
EYH = [0.53, 0.43, 0.33, 0.23, 0.13]


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _plain():
    return ROOT.TGraph(len(X), X, Y)


def _even():
    return ROOT.TGraphErrors(len(X), X, Y, EX, EY)


def _asymm():
    return ROOT.TGraphAsymmErrors(len(X), X, Y, EXL, EXH, EYL, EYH)


def _text(name):
    with open(name, encoding="utf-8") as handle:
        return handle.read()


def test_a_graph_writes_its_points_with_the_delimiter_its_name_names():
    _plain().SaveAs("points.csv")
    _plain().SaveAs("points.tsv")
    _plain().SaveAs("points.txt")
    expect(
        (_text("points.csv"), "1,5\n2,4\n3,3\n4,2\n5,1\n"),
        (_text("points.tsv"), "1\t5\n2\t4\n3\t3\n4\t2\n5\t1\n"),
        (_text("points.txt"), "1 5\n2 4\n3 3\n4 2\n5 1\n"),
    )


def test_a_value_is_written_as_c_plus_plus_writes_it_six_digits_and_no_trailing_zeros():
    ROOT.TGraph(2, [1.0, 2.5], [5.0, 1.0 / 3.0]).SaveAs("points.csv")
    expect((_text("points.csv"), "1,5\n2.5,0.333333\n"))


def test_the_class_says_which_columns_a_graph_writes():
    _even().SaveAs("even.csv")
    _asymm().SaveAs("asymm.csv")
    expect(
        (_text("even.csv").splitlines()[0], "1,0.2,5,0.6"),
        (_text("asymm.csv").splitlines()[0], "1,0.25,0.15,5,0.67,0.53"),
    )


def test_the_option_says_which_columns_to_write_whatever_the_class_is():
    _asymm().SaveAs("none.csv", "noerrors")
    _asymm().SaveAs("even.csv", "errors")
    _even().SaveAs("asymm.csv", "asymmerrors")
    expect(
        (_text("none.csv").splitlines()[0], "1,5"),
        (_text("even.csv").splitlines()[0], "1,0.206155,5,0.60407"),
        (_text("asymm.csv").splitlines()[0], "1,0.2,0.2,5,0.6,0.6"),
    )


def test_a_plain_graph_asked_for_errors_writes_the_ones_root_answers_with():
    """``TGraph::GetErrorX`` is -1 for a graph keeping none, and that is what ROOT writes."""
    _plain().SaveAs("even.csv", "errors")
    expect((_text("even.csv").splitlines()[0], "1,-1,5,-1"))


def test_asroot_puts_the_point_before_its_errors_and_native_is_the_same_word():
    _even().SaveAs("a.csv", "asroot")
    _even().SaveAs("b.csv", "native")
    expect(
        (_text("a.csv"), "1,5,0.2,0.6\n2,4,0.3,0.5\n3,3,0.4,0.4\n4,2,0.5,0.3\n5,1,0.6,0.2\n"),
        (_text("b.csv"), _text("a.csv")),
    )


def test_a_title_option_names_a_column_per_value_the_points_holding_the_axes_titles():
    _even().SaveAs("plain.csv", "title")
    graph = _even()
    graph.GetHistogram().GetXaxis().SetTitle("energy")
    graph.GetHistogram().GetYaxis().SetTitle("counts")
    graph.SaveAs("named.csv", "title asroot")
    expect(
        (_text("plain.csv").splitlines()[0], "# x\tex\ty\tey"),
        (_text("named.csv").splitlines()[0], "# energy\tcounts\tex\tey"),
    )


def test_an_empty_graph_writes_a_file_of_no_lines():
    ROOT.TGraph().SaveAs("empty.csv")
    expect((_text("empty.csv"), ""))


def test_a_graph_reads_back_the_points_a_text_file_of_its_own_holds():
    for extension in ("csv", "tsv", "txt"):
        _even().SaveAs(f"even.{extension}", "asroot")
        back = ROOT.TGraphErrors(f"even.{extension}", "")
        expect(
            (back.GetN(), len(X)),
            (list(back.GetX()), X),
            (list(back.GetY()), Y),
            (round(back.GetErrorX(0), 5), EX[0]),
            (round(back.GetErrorY(0), 5), EY[0]),
            (back.GetName(), "Graph"),
            (back.GetTitle(), f"even.{extension}"),
        )


def test_an_asymmetric_graph_reads_the_two_sides_of_each_bar():
    _asymm().SaveAs("asymm.csv", "asroot")
    back = ROOT.TGraphAsymmErrors("asymm.csv", "")
    expect(
        (back.GetN(), len(X)),
        (round(back.GetErrorXlow(0), 5), EXL[0]),
        (round(back.GetErrorXhigh(0), 5), EXH[0]),
        (round(back.GetErrorYlow(0), 5), EYL[0]),
        (round(back.GetErrorYhigh(0), 5), EYH[0]),
    )


def test_a_column_a_line_does_not_have_is_zero_as_root_leaves_it():
    with open("two.csv", "w", encoding="utf-8") as handle:
        handle.write("1,5\n2,4\n")
    back = ROOT.TGraphErrors("two.csv", "")
    expect((back.GetN(), 2), (back.GetErrorX(0), 0.0), (back.GetErrorY(1), 0.0))


def test_a_line_of_fewer_than_two_numbers_is_no_point():
    with open("notes.csv", "w", encoding="utf-8") as handle:
        handle.write("# a header line\n1,5\n\n4\n2,4\n")
    expect((ROOT.TGraph("notes.csv", "").GetN(), 2))


def test_a_file_that_cannot_be_opened_is_a_graph_of_no_points_after_an_error(capsys):
    back = ROOT.TGraphErrors("nowhere.csv", "")
    printed = capsys.readouterr().err
    expect(
        (back.GetN(), 0),
        ("TGraphErrors::TGraphErrors" in printed, True),
        ("nowhere.csv" in printed, True),
    )


def test_a_format_or_an_option_that_names_delimiters_is_refused_by_name():
    for args in (("points.csv", "%lg %lg"), ("points.csv", "", " , ")):
        with pytest.raises(UnsupportedFeatureError, match="reads the text files ROOT writes"):
            ROOT.TGraph(*args)


def test_a_name_that_is_no_text_format_is_saved_as_a_root_file(capsys):
    _plain().SaveAs("points.dat")
    printed = capsys.readouterr()
    expect(
        (ROOT.TFile("points.dat").Get("Graph") is not None, True),
        ("ROOT file points.dat has been created" in printed.out + printed.err, True),
    )


def test_every_extension_root_saves_as_text_is_read_with_its_own_delimiter():
    expect(
        (engine.text_delimiter("a.CSV"), ","),
        (engine.text_delimiter("a.Tsv"), "\t"),
        (engine.text_delimiter("a.txt"), " "),
        (engine.text_delimiter("a.root"), None),
    )


def test_a_file_with_nothing_in_it_is_columns_of_no_points():
    with open("blank.csv", "w", encoding="utf-8") as handle:
        handle.write("# nothing here\n")
    columns = engine.read_text("blank.csv", 4, ",")
    expect(([len(column) for column in columns or []], [0, 0, 0, 0]))


def test_a_file_that_is_not_there_is_no_columns_at_all():
    expect((engine.read_text("nowhere.csv", 2, ","), None))


def test_the_lines_a_mode_writes_are_the_columns_it_names():
    values = {"x": np.array([1.0]), "y": np.array([2.0]),
              "ex": np.array([0.5]), "ey": np.array([0.25])}  # fmt: skip
    expect(
        (engine.text_lines(values, "even", ","), ["1,0.5,2,0.25"]),
        (engine.text_lines(values, "even", ",", as_root=True), ["1,2,0.5,0.25"]),
        (engine.text_lines(values, "none", ",", header=True, titles=("t", "u")), ["# t\tu", "1,2"]),
    )


def test_the_bar_a_column_writes_is_root_s_own_answer_and_minus_one_without_one():
    plain, asymm = _plain(), _asymm()
    expect(
        (list(plain._error_column("ex"))[:2], [-1.0, -1.0]),
        (round(float(asymm._error_column("ex")[0]), 5), 0.20616),
        (round(float(asymm._error_column("exl")[0]), 5), EXL[0]),
    )


def test_a_bent_graph_reads_the_six_columns_its_base_class_reads():
    _asymm().SaveAs("asymm.csv", "asroot")
    back = ROOT.TGraphBentErrors("asymm.csv", "")
    expect((back.GetN(), 5), (round(back.GetErrorXlow(1), 5), EXL[1]))
