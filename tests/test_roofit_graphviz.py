"""``graphVizTree`` and ``printCompactTree`` to a file, held to what ROOT 6.40 wrote for rf206.

``tests/data/rf206_model.dot`` is ROOT's own ``model.graphVizTree`` of the
rf206 model, through cling. Its nodes come in ``treeNodeServerList``'s
order, which this writes too; its edges in the order of the objects'
addresses, which no model fixes, so they are compared as a set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xrdroot.pyroot as ROOT
from xrdroot.cint.runtime.streams import ostringstream

DATA = Path(__file__).parent / "data"


def rf206() -> Any:
    x = ROOT.RooRealVar("x", "x", 0, 10)
    mean = ROOT.RooRealVar("mean", "mean of gaussians", 5)
    sigma1 = ROOT.RooRealVar("sigma1", "width of gaussians", 0.5)
    sigma2 = ROOT.RooRealVar("sigma2", "width of gaussians", 1)
    sig1 = ROOT.RooGaussian("sig1", "Signal component 1", x, mean, sigma1)
    sig2 = ROOT.RooGaussian("sig2", "Signal component 2", x, mean, sigma2)
    sig1frac = ROOT.RooRealVar("sig1frac", "fraction of component 1 in signal", 0.8, 0.0, 1.0)
    sig = ROOT.RooAddPdf("sig", "Signal", [sig1, sig2], [sig1frac])
    a0 = ROOT.RooRealVar("a0", "a0", 0.5, 0.0, 1.0)
    a1 = ROOT.RooRealVar("a1", "a1", 0.2, 0.0, 1.0)
    bkg1 = ROOT.RooChebychev("bkg1", "Background 1", x, [a0, a1])
    alpha = ROOT.RooRealVar("alpha", "alpha", -1)
    bkg2 = ROOT.RooExponential("bkg2", "Background 2", x, alpha)
    bkg1frac = ROOT.RooRealVar("bkg1frac", "fraction of component 1 in background", 0.2, 0, 1)
    bkg = ROOT.RooAddPdf("bkg", "Signal", [bkg1, bkg2], [bkg1frac])
    bkgfrac = ROOT.RooRealVar("bkgfrac", "fraction of background", 0.5, 0.0, 1.0)
    return ROOT.RooAddPdf("model", "g1+g2+a", [bkg, sig], [bkgfrac])


def _split(text: str) -> tuple[list[str], set[str]]:
    lines = text.split('"];\n')
    edges = {line for line in text.splitlines() if " -> " in line}
    return lines[:-1], edges


def test_a_models_graph_is_roots_node_for_node_and_edge_for_edge(tmp_path: Path) -> None:
    model = rf206()
    model.graphVizTree(str(tmp_path / "m.dot"))
    mine = (tmp_path / "m.dot").read_text()
    roots = (DATA / "rf206_model.dot").read_text()
    assert _split(mine) == _split(roots)
    assert mine.startswith('digraph "model"{\n') and mine.endswith("}\n")


def test_a_graph_goes_to_a_stream_and_can_be_labelled_by_titles() -> None:
    model = rf206()
    stream = ostringstream()
    model.graphVizTree(stream, " | ", True)
    assert '"bkgfrac" [ color=blue, label="RooRealVar | fraction of background | 0.5"];' in (
        stream.str()
    )


def test_a_category_node_is_labelled_without_a_value(tmp_path: Path) -> None:
    c = ROOT.RooCategory("c", "c")
    c.defineType("A", 0)
    x = ROOT.RooRealVar("x", "x", 0, 1)
    sim = ROOT.RooSimultaneous("s", "s", {"A": ROOT.RooUniform("u", "u", x)}, c)
    stream = ostringstream()
    sim.graphVizTree(stream)
    assert '"c" [ color=blue, label="RooCategory\nc"];' in stream.str()


def test_a_compact_tree_goes_to_the_file_named(tmp_path: Path, capsys: Any) -> None:
    model = rf206()
    capsys.readouterr()
    model.printCompactTree("", str(tmp_path / "tree.txt"))
    assert capsys.readouterr().out == ""
    written = (tmp_path / "tree.txt").read_text()
    assert written.splitlines()[0].endswith("RooAddPdf::model = 0.582695/1 [Auto,Clean] ")
