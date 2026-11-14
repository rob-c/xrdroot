"""C's corners tutorials lean on: char arrays by the char, a C string's tail, sizeof arrays."""

from __future__ import annotations

import pytest

from cintfake import fake
from xrdroot.pycint import translate
from xrdroot.pycint.execute import run_source


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


def test_sizeof_of_roots_classes_is_what_root_lays_them_out_in(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """As ROOT 6.40.04 on macOS ARM64 prints it: its classes by themselves, as bases whose
    tail padding a derived class uses (``TAttLine``), and as members."""
    source = r"""
    class EventData : public TObject { public: int n; };
    class L : public TAttLine { public: short s; };
    class M : public TNamed { public: double d[3]; char c; };
    class P { public: TH1F h; TArrayD a; };
    void sz() {
       printf("%zu %zu %zu %zu %zu %zu %zu %zu\n", sizeof(TObject), sizeof(TH1F),
              sizeof(EventData), sizeof(L), sizeof(M), sizeof(P), sizeof(TMatrixD),
              sizeof(TVector3));
    }
    """
    run_source(source, "sz.C")
    assert capsys.readouterr().out == "16 1000 24 16 96 1024 264 40\n"


def test_a_global_pointer_stored_into_a_variable_is_what_it_points_at_then(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``TDirectory *d = gDirectory;`` keeps the directory current then, as ROOT 6.40.04
    prints it: ``gDirectory`` itself follows what is current."""
    from xrdroot.cppstd.cells import pointer_now

    monkeypatch.chdir(tmp_path)
    source = r"""
    void gd() {
       TFile f("gd.root", "RECREATE");
       TDirectory *d = gDirectory;
       gROOT->cd();
       TDirectory *e;
       e = gDirectory;
       printf("%s %d %d\n", d->GetName(), e == gROOT, d == gROOT);
    }
    """
    run_source(source, "gd.C")
    assert capsys.readouterr().out == "gd.root 1 0\n"
    assert pointer_now(3) == 3
