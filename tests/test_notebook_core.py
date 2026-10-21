"""Header tables remain available when graphics cannot be imported."""

from __future__ import annotations

import sys
from pathlib import Path

from xrdroot import chain, open_root

DATA = Path(__file__).parent / "data"


def test_tree_directory_ntuple_and_chain_tables_do_not_import_graphics(monkeypatch):
    monkeypatch.setitem(sys.modules, "xrdroot.plot", None)
    monkeypatch.setitem(sys.modules, "xrdroot.plot.notebook", None)
    with open_root(DATA / "small-flat-tree.root") as root:
        assert "ROOTFile" in root._repr_html_()
        assert "TTree" in root["tree"]._repr_html_()
    with open_root(DATA / "rntuple/ntpl001_staff_rntuple_v1-0-0-0.root") as root:
        assert "RNTuple" in root["Staff"]._repr_html_()
    with chain("tree", [str(DATA / "chain.1.root"), str(DATA / "chain.2.root")]) as joined:
        assert "Chain" in joined._repr_html_()
        assert joined._counts is None
