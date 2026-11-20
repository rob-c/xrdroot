"""Saving live canvases as pictures, books and macros, comparing pictures, and ROOT's namespace."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session, gaussian  # noqa: F401
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.graphics import compare, output


def _canvas():
    c = ROOT.TCanvas("c", "c", 300, 200)
    gaussian().Draw()
    return c


@pytest.mark.parametrize("suffix", ["png", "pdf", "svg", "eps", "jpg", "gif", "ps"])
def test_save_as_writes_the_format_its_suffix_names(tmp_path, suffix, capsys):
    path = tmp_path / f"c.{suffix}"
    _canvas().SaveAs(str(path))
    assert path.stat().st_size > 0
    assert f"{suffix} file {path} has been created" in capsys.readouterr().err


def test_a_gif_is_a_gif_whichever_matplotlib_draws_it(tmp_path):
    """matplotlib before 3.11 writes no GIF, so Pillow writes it from the PNG, on every one."""
    from PIL import Image

    path = tmp_path / "c.gif"
    _canvas().SaveAs(str(path))
    assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    with Image.open(path) as image:
        assert image.size == (296, 172)


def test_a_png_is_the_canvas_size_in_pixels(tmp_path):
    from matplotlib.image import imread

    path = tmp_path / "c.png"
    _canvas().Print(str(path))
    assert imread(str(path)).shape[:2] == (172, 296)


def test_print_takes_its_format_from_its_option_and_a_pad_saves_alone(tmp_path):
    c = _canvas()
    c.Divide(2)
    path = tmp_path / "pad.out"
    c.cd(1).Print(str(path), "png")
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_a_name_is_what_is_saved_when_none_is_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _canvas()
    c.SaveAs()
    c.Print()
    assert (tmp_path / "c.png").exists() and (tmp_path / "c.ps").exists()


def test_a_pad_is_written_as_the_c_macro_that_draws_it_again(tmp_path, capsys):
    """ROOT's ``SaveAs("x.C")`` writes the calls that draw the pad again, not a picture."""
    c = ROOT.TCanvas("mac", "a macro", 300, 200)
    line = ROOT.TLine(0.1, 0.2, 0.9, 0.8)
    line.SetLineColor(2)
    line.Draw("l")
    ROOT.TLatex(0.1, 0.5, 'says "hi"').Draw()
    path = tmp_path / "sg-macro.C"
    c.SaveAs(str(path))
    written = path.read_text()
    assert "void sg_macro()" in written  # the function ``.x`` finds by the file's name
    assert 'TCanvas *canvas = new TCanvas("mac", "a macro", 10, 10, 300, 200);' in written
    assert "new TLine(0.1, 0.2, 0.9, 0.8);" in written
    assert "tline1->SetLineColor(2);" in written and 'tline1->Draw("l");' in written
    assert 'new TLatex(0.1, 0.5, "says \\"hi\\"");' in written
    assert f"macro file {path} has been created" in capsys.readouterr().err


def test_the_macro_of_a_pad_runs_again_and_draws_what_was_saved(tmp_path):
    from xrdroot.pycint.execute import run

    c = ROOT.TCanvas("mac", "a macro", 300, 200)
    ROOT.TLine(0.1, 0.2, 0.9, 0.8).Draw()
    c.SaveAs(str(tmp_path / "again.C"))
    run(tmp_path / "again.C", use_cache=False)
    drawn = [obj for obj, _ in ROOT.gPad.GetCanvas().primitives if obj.ClassName() == "TLine"]
    assert [(obj.members["fX1"], obj.members["fY2"]) for obj in drawn] == [(0.1, 0.8)]


def test_a_macro_literal_is_the_c_that_writes_the_value_again():
    from xrdroot.pyroot.graphics import macro

    assert (macro._literal(True), macro._literal(False)) == ("true", "false")
    assert (macro._literal(2), macro._literal(0.5)) == ("2", "0.5")
    assert macro._literal('a"b\\c') == '"a\\"b\\\\c"'


def test_a_polyline_is_written_point_by_point_as_the_pad_drew_it(tmp_path):
    """ROOT's constructor takes the arrays; a macro's points are the ones set by name."""
    from xrdroot.pycint.execute import run

    c = ROOT.TCanvas("poly", "poly", 300, 200)
    line = ROOT.TPolyLine(2, [0.1, 0.5], [0.2, 0.6])
    line.SetLineWidth(3)
    line.Draw("F")
    path = tmp_path / "sg_poly.C"
    c.SaveAs(str(path))
    written = path.read_text()
    assert "TPolyLine *tpolyline1 = new TPolyLine(2);" in written
    assert "tpolyline1->SetPoint(0, 0.1, 0.2);" in written
    assert "tpolyline1->SetLineWidth(3);" in written and 'tpolyline1->Draw("F");' in written
    run(path, use_cache=False)
    drawn = [o for o, option in ROOT.gPad.GetCanvas().primitives if o.ClassName() == "TPolyLine"]
    assert [(list(o.members["fX"]), list(o.members["fY"])) for o in drawn] == [
        ([0.1, 0.5], [0.2, 0.6])
    ]


def test_a_polyline_is_written_only_as_far_as_its_points_go(tmp_path):
    """A count set past the points it holds writes the points there are, not an IndexError."""
    from xrdroot.pyroot.graphics import macro

    c = ROOT.TCanvas("short", "short", 300, 200)
    line = ROOT.TPolyLine(4, [0.0], [0.0])
    line.members["fN"] = 4
    line.Draw()
    c.SaveAs(str(tmp_path / "short.C"))
    written = (tmp_path / "short.C").read_text()
    assert "new TPolyLine(4);" in written
    assert written.count("SetPoint(") == 1
    assert macro.POINTS["TPolyLine"] == ("fN", "fX", "fY")


def test_a_pad_of_pads_is_written_as_the_pads_it_was_divided_into(tmp_path):
    """``Divide`` makes pads: the macro makes each, enters it, draws what is in it, and leaves."""
    from xrdroot.pycint.execute import run

    c = ROOT.TCanvas("divided", "divided", 400, 300)
    c.Divide(2, 1)
    c.cd(1)
    ROOT.TEllipse(0.5, 0.5, 0.2, 0.2).Draw()
    path = tmp_path / "sg_divided.C"
    c.SaveAs(str(path))
    written = path.read_text()
    assert 'TPad *tpad1 = new TPad("divided_1", "divided_1", 0.01, 0.01, 0.49, 0.99);' in written
    assert "tpad1->cd();" in written and "   canvas->cd();" in written
    order = [written.index(part) for part in ("tpad1->cd();", "TEllipse", "canvas->cd();")]
    assert order == sorted(order)
    run(path, use_cache=False)
    canvas = ROOT.gPad.GetCanvas()
    pads = [o for o, _ in canvas.primitives if o.ClassName() == "TPad"]
    assert [(o.GetName(), o.members["fWNDC"]) for o in pads] == [
        ("divided_1", pytest.approx(0.48)), ("divided_2", pytest.approx(0.48)),
    ]  # fmt: skip
    assert [o.ClassName() for o, _ in pads[0].primitives] == ["TEllipse"]


def test_a_member_only_a_set_of_its_own_reaches_is_written_as_that_set(tmp_path):
    """``TArrow``'s angle has no attribute group's setter, and is still the arrow's own."""
    c = ROOT.TCanvas("arrow", "arrow", 300, 200)
    arrow = ROOT.TArrow(0.1, 0.1, 0.9, 0.9, 0.05, "|>")
    arrow.SetAngle(40)
    arrow.Draw()
    c.SaveAs(str(tmp_path / "arrow.C"))
    written = (tmp_path / "arrow.C").read_text()
    assert 'new TArrow(0.1, 0.1, 0.9, 0.9, 0.05, "|>");' in written
    assert "tarrow1->SetAngle(40);" in written


def test_a_pave_label_and_a_crown_are_written_with_the_arguments_their_constructors_take(tmp_path):
    """A pave's label and a crown's radii are its constructor's, in ROOT's order."""
    c = ROOT.TCanvas("round", "round", 300, 200)
    ROOT.TPaveLabel(0.1, 0.8, 0.9, 0.95, "a label").Draw()
    ROOT.TCrown(0.5, 0.5, 0.2, 0.3, 45.0, 315.0).Draw()
    c.SaveAs(str(tmp_path / "round.C"))
    written = (tmp_path / "round.C").read_text()
    assert 'new TPaveLabel(0.1, 0.8, 0.9, 0.95, "a label", "br");' in written
    assert "new TCrown(0.5, 0.5, 0.2, 0.3, 45, 315);" in written


def test_a_pad_drawing_a_class_with_no_known_constructor_is_refused_by_name(tmp_path):
    c = ROOT.TCanvas("wrapped", "wrapped", 300, 200)
    ROOT.TH1F("h", "h", 10, -4.0, 4.0).Draw()
    with pytest.raises(UnsupportedFeatureError, match="A pad drawing a TH1F is not yet written"):
        c.SaveAs(str(tmp_path / "c.C"))
    with pytest.raises(UnsupportedFeatureError, match=r"knows no constructor for this one"):
        c.Print(str(tmp_path / "c"), "cxx")


def test_a_format_this_does_not_write_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="asks for 'root', which this does not write"):
        _canvas().SaveAs(str(tmp_path / "c.root"))
    with pytest.raises(ValueError, match="'no format'"):
        _canvas().SaveAs(str(tmp_path / "c"))


def test_a_pdf_book_is_opened_filled_and_closed_as_roots_brackets_say(tmp_path):
    c = _canvas()
    book = tmp_path / "book.pdf"
    c.Print(f"{book}[")
    c.Print(str(book))
    c.Print(str(book))
    c.Print(f"{book}]")
    assert str(book) not in output.BOOKS and book.read_bytes().count(b"/Type /Page") >= 2
    other = tmp_path / "other.pdf"
    c.Print(f"{other}(")
    c.Print(f"{other})")
    assert other.exists() and not output.BOOKS
    c.Print(f"{other}]")  # closing a book not open is nothing
    lone = tmp_path / "lone.pdf"
    c.Print(f"{lone})")
    assert lone.exists()
    c.Print(f"{tmp_path / 'left.pdf'}[")
    output.close_books()
    assert not output.BOOKS
    c.Print(f"{tmp_path / 'blank.pdf'}[")
    c.Print(f"{tmp_path / 'blank.pdf'}]")
    # a book no page was added to is no file, whichever matplotlib - and none warns of it
    assert not (tmp_path / "left.pdf").exists() and not (tmp_path / "blank.pdf").exists()


def test_compare_images_is_one_for_the_same_picture_and_less_for_another(tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    _canvas().SaveAs(str(first))
    c = ROOT.TCanvas("d", "d", 300, 200)
    ROOT.TLatex(0.2, 0.5, "different").Draw()
    c.SaveAs(str(second))
    assert compare.compare_images(first, first) == pytest.approx(1.0)
    assert compare.compare_images(first, second) < 0.9
    assert ROOT.compare_images(str(first), np.ones((50, 50))) < 1.0


def test_compare_images_works_on_arrays_and_tiny_pictures():
    ones = np.ones((4, 4, 3), dtype=np.uint8) * 255
    assert compare.compare_images(ones, ones) == 1.0
    assert compare.compare_images(np.zeros((4, 4)), np.ones((4, 4))) == 0.0
    rng = np.random.default_rng(3)
    noise = rng.random((64, 64))
    assert compare._ssim(noise, noise) == pytest.approx(1.0)


def test_compare_images_uses_scikit_image_when_there_is_one(monkeypatch):
    import types

    fake = types.ModuleType("skimage.metrics")
    fake.structural_similarity = lambda a, b, data_range: 0.25
    monkeypatch.setitem(sys.modules, "skimage", types.ModuleType("skimage"))
    monkeypatch.setitem(sys.modules, "skimage.metrics", fake)
    assert compare.compare_images(np.ones((20, 20)), np.ones((20, 20))) == 0.25
    assert compare.compare_images(np.ones((4, 4)), np.ones((4, 4))) == 1.0  # too small for it


def test_compare_images_falls_back_to_numpy_without_scikit_image(monkeypatch):
    monkeypatch.setitem(sys.modules, "skimage.metrics", None)
    assert compare.compare_images(np.ones((20, 20)), np.ones((20, 20))) == pytest.approx(1.0)


def test_the_namespace_has_the_graphics_and_refuses_what_it_lacks_by_name():
    assert ROOT.TCanvas is ROOT.graphics.TCanvas and ROOT.kBird == 57
    assert "TCanvas" in ROOT.__all__ and "gStyle" in ROOT.__all__
    with pytest.raises(AttributeError, match=r"ROOT has TNothing; xrdroot\.pyroot does not yet"):
        ROOT.TNothing  # noqa: B018


def test_a_module_not_on_this_branch_is_passed_over(monkeypatch):
    monkeypatch.setattr(ROOT, "SUBMODULES", ["graphics", "absent"])
    ROOT._gather()
    assert "TCanvas" in ROOT.__all__


def test_the_default_draw_hook_remembers_what_was_drawn():
    core = importlib.import_module("xrdroot.pyroot.core")
    from xrdroot.pyroot.graphics import hook

    core.set_draw_hook(core._remember)
    try:
        core.draw_hook("thing", "opt")
        assert core.DRAWN[-1] == ("thing", "opt")
    finally:
        hook.install()


def _pie_canvas():
    c = ROOT.TCanvas("pie", "pie", 400, 400)
    pie = ROOT.TPie("pie1", "a pie", 3, [0.2, 1.1, 0.6], [2, 3, 4])
    pie.SetCircle(0.5, 0.45, 0.3)
    pie.SetEntryRadiusOffset(1, 0.1)
    pie.SetEntryFillStyle(0, 3030)
    pie.Draw("3d rsc")
    return c


def test_a_pie_is_written_as_the_circle_and_the_slices_the_pad_gave_it(tmp_path):
    """A pie's constructor is told only how many slices it has; ROOT's macro sets each one."""
    from xrdroot.pyroot.graphics import macro

    path = tmp_path / "pie.C"
    _pie_canvas().SaveAs(str(path))
    written = path.read_text()
    assert 'TPie *tpie1 = new TPie("pie1", "a pie", 3);' in written
    assert "tpie1->SetCircle(0.5, 0.45, 0.3);" in written
    assert "tpie1->GetSlice(1)->SetValue(1.1);" in written
    assert "tpie1->GetSlice(1)->SetRadiusOffset(0.1);" in written
    assert "tpie1->GetSlice(0)->SetFillStyle(3030);" in written
    assert written.index("SetCircle(") < written.index("GetSlice(0)")  # the pie, then its slices
    assert macro.PIECES["TPie"][0] == "fPieSlices"


def test_the_macro_of_a_pie_draws_the_slices_the_pie_was_saved_with(tmp_path):
    from xrdroot.pycint.execute import run

    path = tmp_path / "pie.C"
    _pie_canvas().SaveAs(str(path))
    run(path, use_cache=False)
    drawn = [o for o, _ in ROOT.gPad.GetCanvas().primitives if o.ClassName() == "TPie"]
    assert [(one.members["fRadiusOffset"], one.members["fFillStyle"]) for one in
            drawn[0].members["fPieSlices"]] == [(0.0, 3030), (0.1, 1001), (0.0, 1001)]  # fmt: skip


def test_a_pie_of_no_slices_is_written_as_the_count_it_has(tmp_path):
    """A pie made without values has no slice to give anything to, and is still written."""
    c = ROOT.TCanvas("empty", "empty", 300, 300)
    ROOT.TPie("none", "no values", 0).Draw()
    c.SaveAs(str(tmp_path / "empty.C"))
    written = (tmp_path / "empty.C").read_text()
    assert 'TPie *tpie1 = new TPie("none", "no values", 0);' in written
    assert "GetSlice(" not in written


def test_a_pie_made_from_a_histogram_is_written_by_the_slices_the_bins_gave_it(tmp_path):
    hist = ROOT.TH1F("h", "h", 3, 0.0, 3.0)
    for at, value in enumerate((1.0, 2.0, 3.0), start=1):
        hist.SetBinContent(at, value)
        hist.GetXaxis().SetBinLabel(at, f"bin{at}")
    c = ROOT.TCanvas("pieh", "pieh", 300, 300)
    ROOT.TPie(hist).Draw()
    c.SaveAs(str(tmp_path / "pieh.C"))
    written = (tmp_path / "pieh.C").read_text()
    assert 'TPie *tpie1 = new TPie("h", "h", 3);' in written
    assert 'tpie1->GetSlice(2)->SetTitle("bin3");' in written
    assert "tpie1->GetSlice(2)->SetValue(3);" in written
