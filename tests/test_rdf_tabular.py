"""CSV files and SQLite queries as a frame's columns, typed and read as ROOT's sources do."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import Any

import numpy as np
import pytest

from xrdroot import RDataFrame
from xrdroot.rdf.csvsource import CsvOptions, from_csv, split
from xrdroot.rdf.sqlitesource import from_sqlite


def _csv(tmp_path: Any, text: str, **options: Any) -> Any:
    path = tmp_path / "t.csv"
    path.write_text(text)
    return from_csv(str(path), CsvOptions(**options))


def test_a_line_is_split_as_rcsvds_splits_it() -> None:
    assert split('1997,Ford,"Super, ""luxurious"" truck"') == [
        "1997", "Ford", 'Super, "luxurious" truck']  # fmt: skip
    assert split("a,,NaN,nan,") == ["a", "nan", "nan", "nan", "nan"]
    assert split('x;"";y', ";") == ["x", '"', "y"]  # a doubled quote is a quote, even here
    assert split(' 1 , "a" ') == [" 1 ", " a "]


def test_a_columns_type_is_the_look_of_its_first_value(tmp_path: Any) -> None:
    source = _csv(tmp_path, "i,d,e,b,s,n,late\n1,2.5,.5e3,true,x,nan,\n-2,3.,1e5,false,y,,7\n")
    assert source.types == {"i": "Long64_t", "d": "double", "e": "double", "b": "bool",
                            "s": "std::string", "n": "double", "late": "Long64_t"}  # fmt: skip
    values = {name: source.read(name, 0, 2).tolist() for name in source.names()}
    assert values["i"] == [1, -2] and values["e"] == [500.0, 100000.0]
    assert values["b"] == [True, False] and values["s"] == ["x", "y"]
    assert np.isnan(values["n"]).all() and values["late"] == [0, 7]
    assert 'Column "late" of type Long64_t contains empty cell(s)' in source.warning
    assert (len(source), source.describe(), source.cxx_type("d")) == (
        2, "CSV data source", "double")


def test_options_name_type_trim_and_skip_as_roots_do(tmp_path: Any) -> None:
    text = "junk\n# a comment\n  1 2 x  # trailing\n\n  3 4 y\nfooter\n"
    source = _csv(tmp_path, text, fHeaders=False, fDelimiter=" ", fLeftTrim=True,
                  fRightTrim=True, fSkipFirstNLines=1, fSkipLastNLines=1, fComment="#",
                  fColumnTypes={"Col1": "D"})  # fmt: skip
    assert (source.names(), source.types["Col1"]) == (["Col0", "Col1", "Col2"], "double")
    named = _csv(tmp_path, "a,b\n1,true\n", fColumnNames=["x", "y"], fColumnTypes={"y": "O"})
    assert (named.names(), named.read("y", 0, 1).tolist()) == (["x", "y"], [True])
    boolean = _csv(tmp_path, "b\nnan\ntrue\n", fColumnTypes={"b": "O"})
    assert "hence `false` is stored" in boolean.warning


def test_a_csv_file_that_cannot_be_read_is_refused(tmp_path: Any) -> None:
    refusals = {"a,b\n1,2,3\n": "has 3 fields where the file has 2",
                "a\n": "Could not read the header"}  # fmt: skip
    for text, reason in refusals.items():
        with pytest.raises(ValueError, match=reason):
            _csv(tmp_path, text)
    for options, reason in (({"fColumnTypes": {"z": "D"}}, "no column with name"),
                            ({"fColumnTypes": {"a": "Q"}}, "Type alias 'Q'"),
                            ({"fColumnNames": ["p", "q"]}, "passed 2 column names"),
                            ({"fSkipLastNLines": 9}, "too many footer lines"),
                            ({"fColumnTypes": {"a": "D"}}, "is not a number"),
                            ({"fColumnTypes": {"a": "L"}}, "is not a number")):  # fmt: skip
        with pytest.raises(ValueError, match=reason):
            _csv(tmp_path, "a\nx\n", **options)


def test_a_frame_reads_a_csv_file_as_its_columns(tmp_path: Any) -> None:
    frame = RDataFrame(_csv(tmp_path, "x,y\n1,2.5\n2,3.5\n3,4.5\n"))
    assert frame.Filter("x > 1").Sum("y").GetValue() == 8.0


def _database(tmp_path: Any) -> str:
    path = str(tmp_path / "t.sqlite")
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE t (a INTEGER, b FLOAT, c TEXT, d BLOB, e REAL)")
        db.execute("INSERT INTO t VALUES (1, 2.5, 'x', x'0102', 1.0)")
        db.execute("INSERT INTO t VALUES (NULL, NULL, NULL, NULL, NULL)")
        db.execute("INSERT INTO t VALUES ('7', '0.5', 3, 'ab', 2)")
    return path


def test_a_query_is_typed_by_its_declared_types_then_its_first_row(tmp_path: Any) -> None:
    path = _database(tmp_path)
    source = from_sqlite(path, "SELECT a, b, c, d, a + 1 AS f, 1.5 AS g, 'z' AS h, NULL AS i, "
                               "x'00' AS j FROM t;")  # fmt: skip
    assert source.types == {"a": "Long64_t", "b": "double", "c": "std::string",
                            "d": "std::vector<unsigned char>", "f": "Long64_t", "g": "double",
                            "h": "std::string", "i": "void*",
                            "j": "std::vector<unsigned char>"}  # fmt: skip
    read = {name: source.read(name, 0, 3) for name in source.names()}
    assert (read["a"].tolist(), read["b"].tolist(), list(read["c"])) == (
        [1, 0, 7], [2.5, 0.0, 0.5], ["x", "", "3"])
    assert [list(v) for v in read["d"]] == [[1, 2], [], [97, 98]]
    assert (list(read["i"]), source.describe()) == ([None, None, None], "RSqliteDS")
    with pytest.raises(ValueError, match="Unexpected column decl type 'REAL'"):
        from_sqlite(path, "SELECT e FROM t")
    assert from_sqlite(path, "SELECT a FROM t WHERE a > 99").types == {"a": "Long64_t"}


def test_text_is_read_as_the_number_it_starts_with() -> None:
    from xrdroot.rdf.sqlitesource import _as

    assert (_as("12abc", "Long64_t"), _as(b"2.5", "double"), _as(b"hi", "std::string"),
            _as(None, "Long64_t"), _as("abc", "double")) == (12, 2.5, "hi", 0, 0.0)  # fmt: skip


class _Response:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.data


def test_a_url_is_fetched_whole_before_it_is_read(tmp_path: Any, monkeypatch: Any) -> None:
    import urllib.request

    from xrdroot import remote

    database = (tmp_path / "t.sqlite")
    _database(tmp_path)
    served = {"https://h/t.csv": b"x\n1\n2\n", "http://h/t.sqlite": database.read_bytes()}
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, **_: _Response(served[url]))
    assert len(from_csv("https://h/t.csv", CsvOptions())) == 2
    assert len(from_sqlite("http://h/t.sqlite", "SELECT a FROM t")) == 3
    assert remote._context() is not None
    monkeypatch.setitem(__import__("sys").modules, "certifi", None)
    assert remote._context() is not None
