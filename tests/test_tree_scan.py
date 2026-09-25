"""``TTree::Scan``: the table ROOT prints, character for character.

Every expected table is what ``TTreePlayer::Scan`` and
``TTreeFormula::PrintValue`` print for the same entries, worked out from
their code: the eleven-star corner, ``* %8lld `` for the row, a column
``%9.9g`` wide - or as wide as its name, up to twenty - numbers too wide for
it trimmed before their exponent, and an ``Instance`` column whenever any
expression, the selection included, loops over a collection.
"""

from __future__ import annotations

import io
import pathlib

import pytest

from xrdroot import chain, open_root

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def flat():
    with open_root(str(DATA / "small-flat-tree.root")) as f:
        yield f["tree"]


def _table(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines()) + "\n"


def test_a_scan_of_numbers_and_strings_with_a_selection(flat):
    assert flat.scan("Int32:Float64:Str", "Int32 < 3") == _table("""
        ************************************************
        *    Row   *     Int32 *   Float64 *       Str *
        ************************************************
        *        0 *         0 *         0 *   evt-000 *
        *        1 *         1 *         1 *   evt-001 *
        *        2 *         2 *         2 *   evt-002 *
        ************************************************
        ==> 3 selected entries
    """)


def test_a_collection_prints_a_row_per_element_and_a_blank_for_none(flat):
    assert flat.scan("N:SliceFloat64", entries=4) == _table("""
        **************************************************
        *    Row   * Instance *         N * SliceFloat64 *
        **************************************************
        *        0 *        0 *         0 *              *
        *        1 *        0 *         1 *            1 *
        *        2 *        0 *         2 *            2 *
        *        2 *        1 *         2 *            2 *
        *        3 *        0 *         3 *            3 *
        *        3 *        1 *         3 *            3 *
        *        3 *        2 *         3 *            3 *
        **************************************************
    """)


def test_numbers_too_wide_lose_digits_before_their_exponent_and_names_are_cut(flat):
    varexp = "Int32 * 123456789:Float64 / 7:SliceFloat64[3]:Sum$(SliceFloat64) + 0.5"
    assert flat.scan(varexp, "Int32 > 10", entries=14) == _table("""
        ***************************************************************************************
        *    Row   * Int32 * 123456789 * Float64 / 7 * SliceFloat64[3] * Sum$(SliceFloat64... *
        ***************************************************************************************
        *       11 *         1.358e+09 *  1.57142857 *                 *                 11.5 *
        *       12 *         1.481e+09 *  1.71428571 *                 *                 24.5 *
        *       13 *         1.604e+09 *  1.85714286 *                 *                 39.5 *
        ***************************************************************************************
        ==> 3 selected entries
    """)


def test_a_selection_that_loops_takes_every_column_down_its_loop(flat):
    scanned = flat.scan("SliceFloat64:Iteration$:Int32", "SliceFloat64 > 0 && Iteration$ < 2",
                        entries=4)
    assert scanned == _table("""
        ***************************************************************
        *    Row   * Instance * SliceFloat64 * Iteration$ *     Int32 *
        ***************************************************************
        *        1 *        0 *            1 *          0 *         1 *
        *        2 *        0 *            2 *          0 *         2 *
        *        2 *        1 *            2 *          1 *         2 *
        *        3 *        0 *            3 *          0 *         3 *
        *        3 *        1 *            3 *          1 *         3 *
        ***************************************************************
        ==> 5 selected entries
    """)


def test_width_is_root_s_colsize_and_precision_its_precision(flat):
    assert flat.scan("Int32:Float64 / 3", "Int32 == 1", width=5) == _table("""
        ****************************
        *    Row   * Int32 * Fl... *
        ****************************
        *        1 *     1 * 0.333 *
        ****************************
        ==> 1 selected entry
    """)
    assert "*        0.33 *" in flat.scan("Float64 / 3", "Int32 == 1", precision=2)


def test_an_empty_expression_is_the_first_eight_columns_and_a_star_is_all(flat):
    first = flat.scan("", entries=1).splitlines()
    assert first[1].split("*")[3:11] == [
        f" {name:>{max(9, len(name))}} "
        for name in flat.readable()[:8]
    ]
    assert len(first) == 3 + 10 + 1  # the array among them makes a row an element
    every = flat.scan(entries=1).splitlines()[1]
    assert all(name in every for name in flat.readable())


def test_a_scan_is_written_to_a_file_as_it_goes(flat):
    out = io.StringIO()
    text = flat.scan("Int32", "Int32 == 7", file=out)
    assert out.getvalue() == text and text.endswith("==> 1 selected entry\n")
    assert flat.scan("Int32", first_entry=98).count("\n") == 3 + 2 + 1


def test_a_selection_with_no_value_selects_nothing(flat):
    scanned = flat.scan("SliceFloat64", "SliceFloat64[8] > 0", entries=10)
    assert scanned.splitlines()[3:] == [
        "*        9 *        0 *            9 *",
        "*        9 *        1 *            9 *",
        "*        9 *        2 *            9 *",
        "*        9 *        3 *            9 *",
        "*        9 *        4 *            9 *",
        "*        9 *        5 *            9 *",
        "*        9 *        6 *            9 *",
        "*        9 *        7 *            9 *",
        "*        9 *        8 *            9 *",
        "**************************************",
        "==> 9 selected entries",
    ]
    assert "==> 0 selected entries" in flat.scan("Int32", "SliceFloat64[20] > 0")


def test_a_chain_is_scanned_across_its_files():
    files = [str(DATA / "chain.flat.1.root"), str(DATA / "chain.flat.2.root")]
    with chain("tree", files) as events:
        rows = events.scan("Entry$:LocalEntry$", "LocalEntry$ == 0").splitlines()[3:5]
        assert rows == [
            "*        0 *         0 *           0 *",
            "*        5 *         5 *           0 *",
        ]


def test_what_a_scan_cannot_mean_is_refused(flat):
    with pytest.raises(ValueError, match="negative"):
        flat.scan("Int32", first_entry=-1)
    with pytest.raises(ValueError, match="at least one character"):
        flat.scan("Int32", width=0)
