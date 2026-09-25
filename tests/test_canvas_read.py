"""Reading ROOT's canvases: the canvas, its pads, and what each pad drew.

``tcanvas.root`` is a canvas ROOT wrote: one pad drawing a graph with
``"alp"``, saved without being drawn. Everything else is written by
``tests/canvases.py`` from the layouts the classes' headers declare - nested
pads, the drawing classes, the option beside each entry of a pad's list - so
these check that a layout is read faithfully, not that it is ROOT's.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import xrdroot
from canvases import HEAP, NDC, canvas, nest, pad, write
from crafted import Out
from xrdroot import Canvas, Graph, Histogram, UnsupportedFeatureError, open_root
from xrdroot.buffer import Buffer, Listed
from xrdroot.canvas import Pad, Primitive
from xrdroot.canvas.model import lookup
from xrdroot.canvas.streamer import read_canvas

DATA = __file__.rsplit("/", 1)[0] + "/data"


def test_roots_own_canvas_reads_as_a_canvas_of_its_pad_and_graph():
    with open_root(f"{DATA}/tcanvas.root") as f:
        c = f["c1"]
    assert (type(c), c.classname, c.name, c.title) == (Canvas, "TCanvas", "c1", "c1-title")
    assert (c.width, c.height, repr(c)) == (296, 372, "<TCanvas 'c1' 296x372 of 0 pads>")
    ((graph, option),) = c.primitives
    assert (type(graph), option, list(graph.y)) == (Graph, "alp", [0.0, 2.0, 4.0, 1.0, 3.0])
    assert (c.pads, list(c.walk())) == ([], [c])
    assert c.margins == pytest.approx((0.1, 0.1, 0.1, 0.1))
    assert (c.range, c.frame) == ((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0))
    assert (c.logx, c.logy, c.logz, c.painted, c.grid, c.ticks) == (
        False, False, False, False, (False, False), (0, 0),
    )  # fmt: skip


def test_roots_own_canvas_keeps_what_its_window_was():
    with open_root(f"{DATA}/tcanvas.root") as f:
        c = f["c1"]
    assert (c.canvas["fXsizeReal"], c.canvas["kMenuBar"]) == (15.0, True)
    assert c.canvas["fCatt"]["fTitleFromTop"] == pytest.approx(1.2)


def test_a_canvas_keeps_the_option_each_thing_was_drawn_with(tmp_path):
    h = Histogram.book("h", (4, 0.0, 4.0))
    text = nest("TLatex", fTitle="#alpha", fX=0.5, fY=0.5, fBits=HEAP | NDC)
    inner = pad("c_1", [("TH1D", h.members, "hist same"), ("TLatex", text, "")])
    path = write(
        tmp_path / "c.root", ("c", canvas("c", [("TPad", inner, ""), ("TH1D", h.members, "e1")]))
    )
    with open_root(str(path)) as f:
        c = f["c"]
    (sub, _), (top, option) = c.primitives
    assert (type(sub), type(top), option, c.pads) == (Pad, Histogram, "e1", [sub])
    assert [p.name for p in c.walk()] == ["c", "c_1"]
    (h_again, drawn), (latex, _) = sub.primitives
    assert (drawn, h_again.name, type(latex), latex.classname) == (
        "hist same", "h", Primitive, "TLatex",
    )  # fmt: skip
    assert (latex["fTitle"], latex.ndc, latex.get("nothing", 7)) == ("#alpha", True, 7)
    assert latex.get("fTextSize") == pytest.approx(0.05)
    assert (repr(latex), repr(sub)) == ("<TLatex>", "<TPad 'c_1' of 2 primitives>")
    assert sub.place == (0.0, 0.0, 1.0, 1.0)


def test_a_list_read_anywhere_keeps_its_options_beside_it():
    listed = Listed([1, 2], ["a", "b"])
    assert listed == [1, 2]
    assert listed.options == ["a", "b"]
    assert Listed().options == []
    with open_root(f"{DATA}/tcanvas.root") as f:
        held = f["c1"].members["fPrimitives"]
    assert isinstance(held, Listed)
    assert held.options == ["alp"]


def test_a_named_primitive_says_its_name():
    named = Primitive("TPave", {"TBox": {"TObject": {"fBits": 0}}, "fName": "title"})
    assert repr(named) == "<TPave 'title'>"
    assert named.name == "title"
    assert not named.ndc


def test_a_member_is_found_however_deep_in_its_bases_it_is():
    h = Histogram.book("h", (4, 0.0, 4.0))
    assert lookup(h, "fLineColor") == h.members["TH1"]["TAttLine"]["fLineColor"]
    assert lookup({"TBase": {"fDeep": 3}}, "fDeep") == 3
    assert lookup(None, "fLineColor", 5) == 5
    assert lookup(Primitive("TLine", {"fX1": 2.0}), "fX1") == 2.0


def _canvas_bytes(version: int) -> Buffer:
    """A ``TCanvas`` record of ``version``, as ``TCanvas::Streamer`` writes one."""
    out = Out()
    at = out.start(version)
    out.string("$DISPLAY")
    out.pack("i?ffff", 0, True, 0, 0, 15, 20)
    out.pack("ii", 1, 2)
    if version > 2:
        out.pack("II", 604, 528)
    out.pack("II", 600, 500)
    out.pack("??h?", True, True, 5, True)
    if version >= 2:
        out.pack("?", False)
    if version >= 4:
        out.pack("?", True)
    if version >= 2:
        out.pack("?", True)
    out.end(at)
    return Buffer(bytes(out.data))


@pytest.mark.parametrize("version", [1, 2, 3, 8])
def test_a_canvas_reads_the_members_its_version_wrote_and_no_others(version):
    read = read_canvas(lambda name: lambda buf: {"read": name})
    row = read(_canvas_bytes(version))
    assert row["TPad"] == {"read": "TPad"}
    assert row["fCatt"] == {"read": "TAttCanvas"}
    assert (row["fCw"], row["fCh"], row["fXsizeReal"]) == (600, 500, 15.0)
    assert (row["fWindowWidth"], row["fWindowHeight"]) == (
        (604, 528) if version > 2 else (600, 500)
    )
    assert ("kShowEventStatus" in row) == (version >= 2)
    assert ("kAutoExec" in row) == (version >= 4)
    assert row["fHighLightColor"] == 5


def test_a_canvas_whose_pad_the_file_does_not_describe_is_refused_by_name(tmp_path):
    from canvases import KNOWN
    from crafted import UNDESCRIBED, craft

    path = craft(
        tmp_path / "c.root", [("TCanvas", "c", canvas("c", []))], KNOWN, UNDESCRIBED | {"TPad"}
    )
    with open_root(str(path)) as f, pytest.raises(UnsupportedFeatureError, match="TCanvas"):
        f["c"]


def test_a_class_a_canvas_holds_that_the_file_does_not_describe_comes_back_as_its_name(tmp_path):
    path = write(tmp_path / "c.root", ("c", canvas("c", [("TUndescribed", {"fWhatever": 1}, "x")])))
    with open_root(str(path)) as f:
        assert f["c"].primitives == [("TUndescribed", "x")]


def test_the_colours_a_canvas_saved_are_kept_apart_from_what_it_draws(tmp_path):
    colors = [
        ("TColor", nest("TColor", fNumber=2, fRed=0.0, fGreen=1.0, fBlue=0.0)),
        ("TColor", nest("TColor", fNumber=3, fRed=0.5, fGreen=0.5, fBlue=0.5)),
    ]
    palette = [("TColor", nest("TColor", fNumber=3, fRed=0.5, fGreen=0.5, fBlue=0.5))]
    listed = [("TObjArray", colors, ""), ("TObjArray", palette, "")]
    path = write(tmp_path / "c.root", ("c", canvas("c", listed)))
    with open_root(str(path)) as f:
        c = f["c"]
    assert c.primitives == []
    assert len(c.colors) == 2
    table = c.palette()
    assert table.rgb(2) == (0.0, 1.0, 0.0)
    assert table.palette == [3]


def test_a_graph_read_from_a_canvas_writes_to_a_new_file_and_reads_back(tmp_path):
    with open_root(f"{DATA}/tcanvas.root") as f:
        graph = f["c1"].primitives[0][0]
    with xrdroot.create(str(tmp_path / "out.root")) as out:
        out["g"] = graph
        out["h"] = Histogram.book("h", (4, 0.0, 4.0))
    with open_root(str(tmp_path / "out.root")) as again:
        assert list(again["g"].y) == list(graph.y)
        assert again.classnames() == {"g": "TGraph", "h": "TH1D"}


def test_a_histogram_with_a_box_among_its_functions_still_reads_as_before():
    with open_root(f"{DATA}/embedded-tbox.root") as f:
        h = f["h1"]
    (box,) = h.functions
    assert isinstance(box, Primitive)
    assert box.classname == "TBox"
    assert box["fX2"] == 10.0
    np.testing.assert_array_equal(h.values().sum(), 5)


def test_nothing_warns_when_a_canvas_is_only_read(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with open_root(f"{DATA}/tcanvas.root") as f:
            f["c1"]
