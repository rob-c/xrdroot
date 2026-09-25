"""What a notebook shows: pictures that never raise, and tables that read no data."""

from __future__ import annotations

import html
import sys

import xrdroot
from plotting import DATA, cube, curve, gauss, points, tidy  # noqa: F401
from xrdroot import open_root
from xrdroot.plot import set_backend
from xrdroot.plot.notebook import figure, table


def test_a_histogram_shows_as_a_small_svg_with_its_repr_beside_it():
    made = gauss()
    assert "<svg" in made._repr_html_()
    bundle = made._repr_mimebundle_()
    assert bundle["text/plain"] == repr(made) and "<svg" in bundle["text/html"]


def test_graphs_functions_profiles_and_efficiencies_show_as_pictures():
    with open_root(f"{DATA}/tprofile.root") as root:
        profile = root["p1d"]
    with open_root(f"{DATA}/tefficiency.root") as root:
        efficiency = root["eff1"]
    for thing in (points(), curve(), profile, efficiency):
        assert "<svg" in thing._repr_html_()


def test_with_plotly_set_a_notebook_shows_plotlys_html_and_with_text_characters():
    set_backend("plotly")
    assert "cdn.plot.ly" in gauss()._repr_html_()
    set_backend("text")
    assert gauss()._repr_html_().startswith("<pre>h\nx [GeV]")


def test_showing_never_raises_it_falls_back_to_characters_then_to_the_repr(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert figure(gauss()).startswith("<pre>h\n")
    assert figure(cube()) == f"<pre>{html.escape(repr(cube()))}</pre>"


def test_a_table_escapes_what_it_is_given():
    made = table("a <b>", ("x",), [("<script>",)])
    assert "&lt;script&gt;" in made and "a &lt;b&gt;" in made and made.startswith("<table")


def test_a_tree_shows_its_branches_and_types_read_from_its_header():
    with open_root(f"{DATA}/small-flat-tree.root") as root:
        tree = root["tree"]
        shown = tree._repr_html_()
    assert f"{len(tree)} entries" in shown and "Int32" in shown and "int32" in shown
    assert "variable" in shown


def test_a_directory_shows_every_key_its_cycle_and_class():
    with open_root(f"{DATA}/graphs.root") as root:
        shown = root._repr_html_()
    assert "TGraphAsymmErrors" in shown and "<td" in shown and "ROOTFile" in shown
    with open_root(f"{DATA}/dirs-6.14.00.root") as root:
        inner = root[root.keys()[0]]
        assert "Directory" in inner._repr_html_()


def test_an_rntuple_shows_its_fields_python_and_cxx_types():
    with open_root(f"{DATA}/rntuple/ntpl001_staff_rntuple_v1-0-0-0.root") as root:
        shown = root["Staff"]._repr_html_()
    assert "RNTuple" in shown and "C++ type" in shown and "std::int32_t" in shown


def test_a_chain_shows_its_files_and_their_entries_once_counted():
    with xrdroot.chain("tree", [f"{DATA}/chain.1.root", f"{DATA}/chain.2.root"]) as joined:
        uncounted = joined._repr_html_()
        assert "chain.1.root" in uncounted and uncounted.count('"></td>') == 2
        assert joined.counts
        assert joined._repr_html_().count('"></td>') == 0
