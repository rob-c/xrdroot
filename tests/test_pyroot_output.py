"""Saving live canvases as pictures and books, comparing pictures, and the ROOT namespace."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session, gaussian  # noqa: F401
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
