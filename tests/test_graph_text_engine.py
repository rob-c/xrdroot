"""A graph's three text formats, and the numbers a graph answers one at a time.

``TGraph::SaveAs`` writes a ``.csv``, a ``.tsv`` or a ``.txt`` - a comma, a tab or a space
between the values - and ``TGraph(filename, format)`` reads one back. Round 81 made that path
byte-identical to the 6.40 oracle's for ten files; this file pins the engine underneath it, so a
change to the delimiter, the column order or the parser fails here in a second rather than in a
harness run.

The other half is the graph's own arithmetic: ``Eval`` between points and past the ends,
``Integral`` as the polygon's area, the mean and the spread of either axis, and a layered graph's
``GetErrorY``. Those are ``graphmath``'s answers, which round 146 measured against ROOT; what is
tested here is that the graph hands them the right graph and the right axis.

Everything is a plain NumPy array and a dictionary, and the one file this reads is written by the
test itself, so nothing outside the ``hist`` component's own isolation scope is touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrdroot.graph import Graph, read_text, text_delimiter, text_lines

#: Every column ROOT's three formats can write, so one set of values serves all of them.
POINTS = {
    "x": np.array([1.0, 2.0]),
    "y": np.array([3.5, 4.0]),
    "ex": np.array([0.1, 0.2]),
    "ey": np.array([0.3, 0.4]),
    "exl": np.array([0.1, 0.2]),
    "exh": np.array([0.5, 0.6]),
    "eyl": np.array([0.7, 0.8]),
    "eyh": np.array([0.9, 1.0]),
}


def _written(*, lines: tuple[str, ...]) -> Path:
    """A ``.csv`` file holding exactly these lines, which is what the reader is handed."""
    import tempfile

    made = Path(tempfile.mkdtemp()) / "points.csv"
    made.write_text("\n".join(lines) + "\n")
    return made


def test_the_extension_names_the_delimiter_and_does_not_care_about_case():
    """``TGraph::SaveAs`` writes a comma for ``.csv``, a tab for ``.tsv`` and a space for
    ``.txt``; any other name is a ROOT file, which the caller writes another way. The extension
    is read without regard to case, as ROOT's is."""
    assert text_delimiter("g.csv") == ","
    assert text_delimiter("g.tsv") == "\t"
    assert text_delimiter("g.txt") == " "
    assert text_delimiter("G.TSV") == "\t"  # a case-preserving filesystem may carry either
    assert text_delimiter("g.root") is None
    assert text_delimiter("g.csv.bak") is None  # the extension is the last thing in the name


def test_a_point_comes_before_its_bars_in_root_s_order_and_after_them_in_a_spreadsheet_s():
    """``asroot``/``native`` writes x, y and then the bars, which is the order ROOT reads with
    ``TGraphErrors``' own constructor; a spreadsheet's reader takes the point's own errors right
    after the point, so that is the other order - and it is the default."""
    assert text_lines(POINTS, "even", ",", as_root=True) == ["1,3.5,0.1,0.3", "2,4,0.2,0.4"]
    assert text_lines(POINTS, "even", ",") == ["1,0.1,3.5,0.3", "2,0.2,4,0.4"]
    asymmetric = ["1\t3.5\t0.1\t0.5\t0.7\t0.9", "2\t4\t0.2\t0.6\t0.8\t1"]
    assert text_lines(POINTS, "asymm", "\t", as_root=True) == asymmetric
    assert text_lines(POINTS, "asymm", "\t") == [
        "1\t0.1\t0.5\t3.5\t0.7\t0.9",
        "2\t0.2\t0.6\t4\t0.8\t1",
    ]
    assert text_lines(POINTS, "none", " ") == ["1 3.5", "2 4"]  # a graph keeping no bars


def test_the_header_names_a_column_a_value_and_carries_the_axes_titles():
    """``title`` writes a header line naming a column per value, with the axes' titles in the
    point's own two columns and the bars' own names elsewhere. It is written with **tabs**
    whatever the delimiter is, because a spreadsheet reads these files and has been handed one."""
    assert text_lines(POINTS, "even", ",", header=True, titles=("E", "N")) == [
        "# E\tex\tN\tey",  # the titles sit in the point's two columns of that order
        "1,0.1,3.5,0.3",
        "2,0.2,4,0.4",
    ]
    untitled = text_lines(POINTS, "none", " ", header=True)
    assert untitled[0] == "# \t"  # the two columns are the axes, and no titles were given


def test_a_number_is_written_as_the_ostream_writes_it():
    """C++'s ``ostream`` gives six significant digits and no ``.0`` after a whole one, which is
    what every one of ROOT's three text files holds - ``3`` and not ``3.0``."""
    assert text_lines({"x": np.array([3.0]), "y": np.array([0.00012345678])}, "none", " ") == [
        "3 0.000123457"
    ]


def test_every_line_is_read_for_the_numbers_it_holds_and_one_below_two_is_skipped():
    """A line gives the first ``columns`` numbers it holds, a column it does not have is zero,
    and a line of fewer than two numbers is skipped - which is what a comment, a header and a
    line holding a word are, and what ROOT skips rather than failing the whole file."""
    assert [column.tolist() for column in read_text(_written(lines=("1,2,3,4",)), 4, ",")] == [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ]
    assert [column.tolist() for column in read_text(_written(lines=("1,2",)), 4, ",")] == [
        [1.0],
        [2.0],
        [0.0],
        [0.0],
    ]
    for skipped in ("8", "# x,y,ex,ey", "1,nope,3"):
        assert [
            column.tolist() for column in read_text(_written(lines=(skipped,)), 4, ",")
        ] == [[], [], [], []]


def test_a_text_file_is_read_column_by_column_and_a_missing_one_is_no_columns(tmp_path: Path):
    """``TGraph(filename, format)`` reads a line per point into the columns it was asked for. A
    file that cannot be opened is answered with ``None`` - ROOT gives back a graph of no points
    and an error line rather than refusing - and a file of nothing but comments is a column of no
    values."""
    one = tmp_path / "points.csv"
    one.write_text("1,2\n3,4\n# comment\n5,6,7\n8\n")
    assert [column.tolist() for column in read_text(one, 2, ",")] == [
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
    ]  # the comment and the one-number line are skipped, and the four-value line is cut short
    assert [column.tolist() for column in read_text(one, 4, ",")] == [
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
        [0.0, 0.0, 7.0],
        [0.0, 0.0, 0.0],
    ]
    assert read_text(tmp_path / "missing.csv", 2, ",") is None
    empty = tmp_path / "empty.csv"
    empty.write_text("# only a comment\n")
    assert [column.tolist() for column in read_text(empty, 2, ",")] == [[], []]


def test_a_graph_answers_its_own_arithmetic_on_the_axis_it_is_given():
    """The graph's methods are ``graphmath``'s, handed this graph and the axis asked for:
    ``Eval`` between the points and past the ends, ``Integral`` the polygon's area, and the mean
    and spread of x or of y."""
    found = Graph.new("g", [0, 1, 2], [0, 10, 40])
    assert found.eval([0.5, 3.0]).tolist() == [5.0, 70.0]
    assert found.integral() == 10.0  # the trapezoids under 0 -> 10 -> 40 over one bin each
    assert found.mean() == 1.0 and found.rms() == pytest.approx(np.sqrt(2 / 3))
    assert found.mean(1) == pytest.approx(50 / 3) and found.mean(1) == found.mean(axis=1)
    assert found.rms(1) == pytest.approx(np.sqrt((0 + 100 + 1600) / 3 - (50 / 3) ** 2))
    found.sort()
    assert np.asarray(found.x).tolist() == [0.0, 1.0, 2.0]  # sorting a sorted graph moves nothing


def test_a_layered_graph_adds_its_bars_up_by_its_own_mode():
    """``TGraphMultiErrors::GetErrorY(i)`` is what the layers at a point sum to, by the graph's
    ``fSumErrorsMode`` - ``kOnlyFirst`` unless it was told otherwise."""
    found = Graph.from_layers("m", [1, 2], [3, 4], layers=[([0.1, 0.1], [0.2, 0.2])])
    assert found.sum_errors_mode == 0  # kOnlyFirst, its own default
    assert found.error_y(0) == pytest.approx(np.sqrt(0.1**2 + 0.2**2) / np.sqrt(2))


def test_two_lists_that_are_not_the_same_length_are_not_points():
    """A graph is a list of points, so x and y have to be as long as each other; the refusal says
    how long each was."""
    with pytest.raises(ValueError, match="2 x values and 1 y values are not points"):
        Graph.from_layers("bad", [1, 2], [3])
