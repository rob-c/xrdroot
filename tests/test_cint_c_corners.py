"""C's corners tutorials lean on: char arrays by the char, a C string's tail, sizeof arrays."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source


def output(capsys: pytest.CaptureFixture[str], source: str) -> str:
    run_source(source, "t.C", root=fake())
    return capsys.readouterr().out


def test_a_char_array_given_its_chars_is_the_string_they_spell_to_the_first_nul(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = r"""{ char raw[16] = {0}; char ab[8] = {'a', 'b', 0, 'c'};
    printf("[%s] [%s] %d\n", raw, ab, (int)strlen(ab)); }"""
    assert output(capsys, source) == "[] [ab] 2\n"


def test_a_c_string_plus_a_number_is_its_tail(capsys: pytest.CaptureFixture[str]) -> None:
    source = r"""{ const char *dir = "/a/b/name.C"; const char *slash = strrchr(dir, '/');
    const char *base = slash + 1; printf("%s\n", base); }"""
    assert output(capsys, source) == "name.C\n"


def test_sizeof_an_array_of_pointers_or_rows_counts_its_elements(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = r"""
    const char *blurbs[] = {"Love", "Peace", "ROOT"};
    const double colors[][4] = {{1., 0., 0., 1.}, {0., 1., 0., 1.}};
    const int n_blurbs = sizeof(blurbs) / sizeof(char *);
    const unsigned n_colors = sizeof colors / sizeof colors[0];
    void t() { printf("%d %u\n", n_blurbs, n_colors); }
    """
    assert output(capsys, source) == "3 2\n"


def test_cs_named_types_and_unnamed_parameters_of_every_type_declare_prototypes() -> None:
    source = """
    int ReadChunk(FILE *, unsigned int);
    time_t stamps[3];
    void t() { time_t *later = stamps + 1; FILE *f = nullptr; }
    """
    text = translate(source, "t.C")
    assert "stamps = array('long', 3)" in text and "later = stamps[1:]" in text
    assert "f = None" in text


def test_an_array_roots_method_writes_into_is_handed_over_as_it_is() -> None:
    text = translate("void t() { double lo[3], hi[3]; view->GetRange(lo, hi); }", "t.C")
    assert "ROOT.view.GetRange(lo, hi)" in text
