"""A loop filling a histogram from a random draw, written as the draws at once and one Fill.

Python turning a million times through ``h->Fill(r.Gaus())`` is what a
tutorial's minute goes on. A generator draws ``n`` at once in the stream's
own order, and ``Fill`` of arrays is ROOT's bookkeeping entry by entry, so
the loop is written as those two and comes out the same to the last bit.
Anything the arrays could not keep the same - a second draw, the counter in
the body, another call, a histogram the translator cannot be sure of - is
written as the loop it was.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import arith

OBJECTS = (
    'TH1F *h = new TH1F("h", "h", 10, 0, 10); TProfile *p = new TProfile("p", "p", 10, 0, 10);'
    ' TH1F *g = new TH1F("g", "g", 10, 0, 10); TRandom3 r(1); Float_t px, py; double qx, qy;'
    " int k = 0;\n"
)


def body(statements: str) -> str:
    source = f"void t() {{\n{OBJECTS}for (int j = 0; j < 1000; j++) {{ {statements} }}\n}}\n"
    text = translate(source, "t.C")
    compile(text, "t.C", "exec")
    return text


@pytest.mark.parametrize(
    ("statements", "fragment"),
    [
        ("float x = r.Gaus(1, 2); h->Fill(x, 3);", "x = f32(r.Gaus(1, 2, n=count_1))"),
        ("h->Fill(gRandom->Rndm());", "h.Fill(ROOT.gRandom.Rndm(n=count_1))"),
        ("double x = r.Exp(2.5); double y = -x * 2 + 1; h->Fill(y, x / 2);", "h.Fill(y, x / 2)"),
        ("h->Fill(r.Integer(10));", "h.Fill(r.Integer(10, n=count_1))"),
        ("h->Fill(r.Integer(10) * 2.0, m);", "h.Fill(r.Integer(10, n=count_1) * 2.0, ROOT.m)"),
        ("double x = r.Gaus(); h->Fill(x / 2);", "h.Fill(x / 2)"),
        ("float x = r.Gaus(); float y = x * x; h->Fill(y);", "y = f32(x * x)"),
        ("h->Fill(r.Gaus() + n);", "h.Fill(r.Gaus(n=count_1) + ROOT.n)"),
        ("double a = 2.0; h->Fill(r.Gaus() + a);", "h.Fill(r.Gaus(n=count_1) + a)"),
        ("double x = r.Gaus(); h->Fill(-x / 2, (x + 1) / 2);", "h.Fill(-x / 2, (x + 1) / 2)"),
        ("double x = r.Gaus(); h->Fill(x); g->Fill(x * x);", "g.Fill(x * x)"),
        ("r.Rannor(px, py); h->Fill(px - 2); g->Fill(py + 2);", "h.Fill(f32(px_2 - 2))"),
        ("r.Rannor(px, py); double z = px * py; g->Fill(z);", "z = f32(px_2 * py_3)"),
        ("gRandom->Rannor(qx, qy); h->Fill(qx, qy);",
         "qx_2, qy_3 = ROOT.gRandom.Rannor(n=count_1)"),
    ],
)
def test_a_fill_loop_of_one_draw_is_written_as_arrays(statements: str, fragment: str) -> None:
    text = body(statements)
    assert "count_1 = len(range(0, 1000))" in text
    assert fragment in text
    assert "for j" not in text


@pytest.mark.parametrize(
    "statements",
    [
        "double x = r.Gaus(); double y = r.Gaus(); h->Fill(x, y);",
        "h->Fill(r.Gaus(j));",
        "double x = r.Gaus(); h->Fill(TMath::Abs(x));",
        "h->Fill(3);",
        "double x = r.Gaus(); h->Fill(2);",
        "int k = r.Integer(10); h->Fill(k);",
        "double x = r.Gaus(); if (x > 0) h->Fill(x);",
        "double x = r.Gaus(); p->Fill(x, x);",
        "double x = r.Gaus(); k->Fill(x);",
        "double x = r.Gaus(); h->Fill((int)x);",
        "double x = r.Gaus(); h->Fill(x); h->Fill(x);",
        "double x = r.Gaus(r.Rndm()); h->Fill(x);",
        'double x = r.Gaus(); h->Fill("a", x);',
        "h->Fill(r.Integer(10) * 2);",
        "h->Fill(r.Gaus() / 2);",
        "h->Fill(r.Gaus() / n);",
        "double x; x = r.Gaus(); h->Fill(x);",
        "h->Fill(rng->Gaus());",
        "h->Fill(rr[0].Gaus());",
        "double x = r.Gaus(); h->Fill(x); h->Fill(x * x);",
        "r.Rannor(px, py); h->Fill(px); h->Fill(py);",
        "r.Rannor(px, px); h->Fill(px);",
        "r.Rannor(px, py); double z = r.Gaus(); h->Fill(px + z);",
        "r.Rannor(px, py); h->Fill(j);",
        "r.Rannor(px, py); h->Fill(3);",
        "r.Rannor(px, k); h->Fill(px);",
        "r.Rannor(px, py);",
    ],
)
def test_a_loop_the_arrays_would_not_keep_the_same_stays_a_loop(statements: str) -> None:
    """Two draws, the counter, another call, no draw or one unused, an integral temporary,
    a test, a profile, an unknown receiver, a cast, two fills, a draw inside a draw,
    a label, integer arithmetic on a draw, a division the emitter could not see is
    floating, a temporary declared without its value, a generator of no known type or not
    named plainly, one
    histogram filled twice, ``Rannor`` into one variable twice, with a second draw, with
    the counter or nothing of the pair filled, into an integer, or alone."""
    text = body(statements)
    assert "for j in range(0, 1000):" in text
    assert "n=" not in text


MACRO = """
void t() {
   TH2I h("h", "h", 50, 0, 100, 4, 0, 4);
   TRandom3 r(4357);
   for (int b = 0; b < 3; b++)
      for (int j = 0; j < %d; j++) { float x = r.Gaus(40 + b * 8, 10 + b); h.Fill(x, b, %s); }
   printf("%%.17g %%.17g %%.17g %%.17g %%d\\n", h.GetEntries(), h.GetMean(), h.GetRMS(),
          h.GetSumOfWeights(), h.GetBinContent(20, 2));
}
"""


@pytest.mark.parametrize("turns", [0, 5, 2000])
def test_the_arrays_fill_what_the_loop_would_to_the_last_bit(
    capsys: pytest.CaptureFixture[str], turns: int
) -> None:
    """The same macro with the counter in the weight (``1.5 + 0 * j`` is 1.5) stays a loop:
    the bins, the entries and the moments are its, printed to the digit."""
    run_source(MACRO % (turns, "1.5"), "t.C", root=ROOT)
    run_source(MACRO % (turns, "1.5 + 0 * j"), "t.C", root=ROOT)
    arrays, loop = capsys.readouterr().out.splitlines()
    assert arrays == loop
    assert arrays.startswith(f"{3 * turns} ")


PAIRED = """
void t() {
   TH1F h("h", "h", 40, -5, 5);
   TH1D g("g", "g", 40, -5, 5);
   TRandom3 r(4357);
   Float_t px, py;
   for (int j = 0; j < %d; j++) { r.Rannor(px, py); h.Fill(px - 2); g.Fill(py + 2, %s); }
   printf("%%.17g %%.17g %%.17g %%.17g %%.17g %%.17g\\n", h.GetMean(), h.GetRMS(), g.GetMean(),
          g.GetRMS(), px, py);
}
"""


@pytest.mark.parametrize("turns", [0, 3, 500])
def test_the_pair_drawn_at_once_fills_what_the_loop_would_and_is_left_as_it_was(
    capsys: pytest.CaptureFixture[str], turns: int
) -> None:
    """Both histograms' moments to the digit, and ``px``, ``py`` the last pair - or, for no
    turns, what they were."""
    run_source(PAIRED % (turns, "1.0"), "t.C", root=ROOT)
    run_source(PAIRED % (turns, "1.0 + 0 * j"), "t.C", root=ROOT)
    arrays, loop = capsys.readouterr().out.splitlines()
    assert arrays == loop
    if not turns:
        assert arrays.endswith(" 0 0")


def test_rannor_draws_n_pairs_at_once_in_the_streams_order() -> None:
    one, many = ROOT.TRandom3(7), ROOT.TRandom3(7)
    pairs = [one.Rannor() for _ in range(4)]
    first, second = many.Rannor(n=4)
    assert (first.tolist(), second.tolist()) == ([a for a, _ in pairs], [b for _, b in pairs])


def test_a_single_precision_store_rounds_an_array_elementwise() -> None:
    assert arith.f32(0.1) == float(np.float32(0.1))
    stored = arith.f32(np.array([0.1, 1e-45, 3.0]))
    assert stored.dtype == np.float64
    assert stored.tolist() == [float(np.float32(0.1)), float(np.float32(1e-45)), 3.0]


def test_rndm_draws_n_at_once_in_the_streams_order() -> None:
    one, many = ROOT.TRandom3(7), ROOT.TRandom3(7)
    assert [one.Rndm() for _ in range(5)] == many.Rndm(n=5).tolist()
    assert isinstance(one.Rndm(1), float)
