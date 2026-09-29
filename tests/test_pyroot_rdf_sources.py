"""``RDF.FromCSV``, ``FromSqlite`` and ``MakeLazyDataFrame``, and a macro's lambdas in a frame."""

from __future__ import annotations

import sqlite3
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.cint.execute import run_source

#: A small ROOT file of the tests' own.
DATA = Path(__file__).parent / "data"


def test_a_csv_file_is_a_frame_by_roots_arguments(tmp_path: Any, capsys: Any) -> None:
    (tmp_path / "a.csv").write_text("x,y,n\n1,2.5,\n2,3.5,4\n")
    (tmp_path / "b.dat").write_text("1 a\n2 b\n")
    default = ROOT.RDF.FromCSV(str(tmp_path / "a.csv"))
    spaced = ROOT.RDF.MakeCsvDataFrame(str(tmp_path / "b.dat"), False, ord(" "), -1,
                                       [("Col0", ord("D"))])  # fmt: skip
    options = ROOT.RDF.RCsvDS.ROptions()
    options.fColumnTypes = {"x": "D"}
    chosen = ROOT.RDF.FromCSV(str(tmp_path / "a.csv"), options)
    expect((default.Sum("y").GetValue(), 6.0), (list(default.GetColumnNames()), ["n", "x", "y"]),
           (spaced.GetColumnType("Col0"), "double"),
           (list(spaced.Take("Col1").GetValue()), ["a", "b"]),
           (chosen.GetColumnType("x"), "double"))  # fmt: skip
    assert "Warning in <RCsvDS>: Column \"n\" of type Long64_t" in capsys.readouterr().err


def test_an_sqlite_query_and_taken_results_are_frames(tmp_path: Any) -> None:
    path = str(tmp_path / "t.sqlite")
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE t (a INTEGER, s TEXT)")
        db.executemany("INSERT INTO t VALUES (?, ?)", [(1, "p"), (2, "q"), (3, "r")])
    frame = ROOT.RDF.FromSqlite(path, "SELECT * FROM t")
    also = ROOT.RDF.MakeSqliteDataFrame(path, "SELECT a FROM t WHERE a > 1")
    taken = frame.Take["Long64_t"]("a")
    lazy = ROOT.RDF.MakeLazyDataFrame(("a", taken), ("s", frame.Take("s")), ("d", [0.5, 1.5, 2.5]))
    expect((frame.Count().GetValue(), 3), (also.Sum("a").GetValue(), 5.0),
           (lazy.Sum("d").GetValue(), 4.5), (lazy.GetColumnType("a"), "Long64_t"),
           (lazy.GetColumnType("s"), "std::string"), (lazy.GetColumnType("d"), "double"))


def test_a_macros_lambdas_are_called_once_per_entry(tmp_path: Any, capsys: Any) -> None:
    (tmp_path / "m.csv").write_text("m,k\n3.0,1\n3.1,2\n5.0,3\n")
    macro = f"""
    using namespace ROOT::RDF;
    void m() {{
       auto df = FromCSV("{tmp_path / 'm.csv'}");
       double low = 2.95, high = 3.25;
       auto cut = [low, high](double m) {{ return m < high && m > low; }};
       auto twice = [](Long64_t k) {{ return 2 * k; }};
       auto both = df.Filter(cut, {{"m"}}).Define("k2", twice, {{"k"}});
       both.Foreach([](Long64_t k2) {{ printf("%lld\\n", k2); }}, {{"k2"}});
       auto h = both.Histo1D<double>({{"h", "", 10, 0, 10}}, "k2");
       printf("%g\\n", h->GetEntries());
    }}
    """
    run_source(macro, "m.C")
    assert capsys.readouterr().out == "2\n4\n2\n"


def test_a_jagged_column_reaches_a_macros_lambda_as_an_rvec() -> None:
    from xrdroot.pyroot.rdf.entrywise import entrywise
    from xrdroot.tree import Jagged

    def sizes(v: Any) -> int:
        return int(v.size())

    sizes.__module__ = "__cint__"
    batch = entrywise(sizes)(Jagged(np.array([1.0, 2.0, 3.0]), np.array([0, 2, 3])))
    assert batch.tolist() == [2, 1]


class _Response:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.data


def test_a_root_file_named_by_a_url_is_fetched_and_read(monkeypatch: Any) -> None:
    served = (DATA / "simple.root").read_bytes()
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, **_: _Response(served))
    f = ROOT.TFile.Open("http://h/simple.root")
    assert f is not None and f.GetListOfKeys().GetSize() > 0
    f.Close()
