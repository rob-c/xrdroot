"""Which face each of ROOT's fonts is drawn in here, and how a string is measured in it.

ROOT ships its fonts; here each is the first installed of faces with the
same widths, found among the faces matplotlib knows, in the style asked
for, from a TrueType file. These tests give that search faces of their own
- a bold one named by its weight's word, an OpenType one, a collection -
to see which it takes, and what it falls back to when none is there. A
Symbol character the installed Symbol has no glyph for is drawn from
matplotlib's STIX faces; a character no face has is left out of a string's
measure, as ``TTF::LayoutGlyphs`` leaves it; a string of blanks draws
nothing, as ``TASImage::DrawText`` draws nothing of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib import font_manager

from test_canvas_draw import make, prim
from xrdroot.canvas import fonts


@pytest.fixture
def installed(monkeypatch):
    """Faces of the test's own, listed by matplotlib before those really installed."""
    real = list(font_manager.fontManager.ttflist)

    def install(*entries):
        fonts.face.cache_clear()
        monkeypatch.setattr(font_manager.fontManager, "ttflist", [*entries, *real])

    yield install
    monkeypatch.undo()
    fonts.face.cache_clear()


def entry(fname, style="normal", weight=400):
    """One face of TeX Gyre Heros, ROOT's 4x, as matplotlib lists it."""
    return font_manager.FontEntry(fname=fname, name="TeX Gyre Heros", style=style, weight=weight)


def test_a_face_named_bold_or_heavy_by_its_weight_is_bold_and_one_named_normal_is_not(installed):
    installed(
        entry("/fonts/a-heavy.ttf", weight="heavy"),
        entry("/fonts/b-bold-italic.ttf", style="italic", weight="Bold"),
        entry("/fonts/c-regular.ttf", weight="normal"),
    )
    assert fonts.face(62) == "/fonts/a-heavy.ttf"
    assert fonts.face(72) == "/fonts/b-bold-italic.ttf"
    assert fonts.face(42) == "/fonts/c-regular.ttf"


def test_only_a_collections_regular_face_and_no_opentype_file_is_taken(installed):
    installed(
        entry("/fonts/a-regular.otf"),
        entry("/fonts/b-bold.ttc", weight=700),
        entry("/fonts/c-regular.ttc"),
    )
    assert fonts.face(42) == "/fonts/c-regular.ttc"
    assert not fonts.face(62).startswith("/fonts/")  # the next face with a bold file


def test_with_no_truetype_file_of_its_faces_a_font_is_matplotlibs_dejavu_in_its_style(
    monkeypatch,
):
    fonts.face.cache_clear()
    monkeypatch.setattr(fonts, "TRUETYPE", ())
    try:
        assert Path(fonts.face(52)).name == "DejaVuSans-Oblique.ttf"
        assert Path(fonts.face(132)).name == "DejaVuSans.ttf"
    finally:
        fonts.face.cache_clear()


def test_a_symbol_the_symbol_face_has_not_is_drawn_from_the_stix_face_that_has_it():
    brackets = "⟩⎛"  # the angle bracket and a big bracket's piece, which no Symbol has both of
    stix = font_manager.findfont(font_manager.FontProperties(family="STIXSizeOneSym"))
    assert fonts.face_of(122, brackets) == stix
    assert fonts.face_of(122, "\uffff") == fonts.face(122)  # nothing has it: its own face


def test_a_character_no_face_has_is_left_out_of_a_strings_measure():
    assert fonts.extent("a\uffffb", 42, 20) == fonts.extent("ab", 42, 20)
    assert fonts.extent("\uffff", 42, 20) == fonts.Extent(0, 0, 0, 0, 0)


def test_a_string_of_blanks_draws_nothing():
    blanks = prim("TText", fTitle="   ", fX=0.5, fY=0.5)
    assert list(make([(blanks, "")]).plot().axes[0].texts) == []
