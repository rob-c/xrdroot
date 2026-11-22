"""What the tutorial harness asked of the core next: function lists, templated ``Get``, 2D draws."""

from __future__ import annotations

import sys
import types

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_histograms_list_of_functions_is_its_own_and_keeps_what_is_added():
    h = ROOT.TH1D("h", "", 10, -3, 3)
    h.FillRandom("gaus", 200)
    h.Fit("gaus", "Q")
    listed = h.GetListOfFunctions()
    box, line = ROOT.TNamed("stats", ""), ROOT.TNamed("line", "")
    listed.Add(box)
    listed.AddFirst(line)
    extra = ROOT.TF1("extra", "x", 0, 1)
    h.GetListOfFunctions().AddLast(extra)
    h.GetListOfFunctions().AddFirst(ROOT.TF1("first", "x", 0, 1))
    again = h.GetListOfFunctions()
    expect(
        (again.GetSize(), 5),
        (again.FindObject("stats") is box, True),
        ([item.GetName() for item in again], ["first", "gaus", "extra", "line", "stats"]),
        (len(h._xrd.functions), 3),
    )
    assert again.Remove(box) is box
    assert again.Remove(extra) is extra
    assert again.Remove(ROOT.TNamed("nothing", "")) is None
    again.Clear()
    expect((h.GetListOfFunctions().GetSize(), 0), (h._xrd.functions, []))


def test_a_graphs_list_of_functions_is_its_own_too():
    g = ROOT.TGraph(3, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    g.GetListOfFunctions().Add(ROOT.TNamed("label", ""))
    g.Fit("pol1", "Q")
    names = [item.GetName() for item in g.GetListOfFunctions()]
    expect((names, ["pol1", "label"]), (g.GetFunction("pol1") is not None, True))


def test_get_takes_its_class_in_brackets_as_a_cpp_template_does():
    out = ROOT.TFile("t.root", "RECREATE")
    ROOT.TH1D("h", "", 1, 0, 1)
    out.Write()
    out.Close()
    back = ROOT.TFile("t.root")
    h = back.Get["TH1D"]("h")
    expect(
        (h is back.Get("h"), True),
        (back.GetKey("h").ReadObj[ROOT.TH1D]() is h, True),
        (ROOT.gROOT.Get["TH1"]("nothing"), None),
    )
    assert ROOT.TDirectory.Get.__name__ == "Get"
    back.Close()


def test_a_histogram_names_the_stat_overflows_flag_and_the_interpreters_error_codes():
    """``TH1::EStatOverflows`` is the per-histogram flag that overrides
    ``TH1::SetStatOverflows`` - ``kConsider`` counts the under- and overflow bins in the
    statistics - and ``gInterpreter`` reports one of ``TInterpreter::EErrorCode``."""
    expect(
        (ROOT.TH1.kIgnore, 0),
        (ROOT.TH1.kConsider, 1),
        (ROOT.TH1.kNeutral, 2),
        (ROOT.gInterpreter.kNoError, 0),
        (ROOT.TInterpreter.kProcessing, 99),
        (ROOT.gInterpreter.Declare("int forty_two() { return 42; }"), True),
    )


def test_axes_that_may_grow_and_bars_are_noted():
    h = ROOT.TH1F("h", "", 3, 0, 3)
    expect((h.SetCanExtend(ROOT.TH1.kAllAxes), 0), (h.CanExtendAllAxes(), True))
    h2 = ROOT.TH2F("h2", "", 2, 0, 2, 2, 0, 2)
    h2.SetCanExtend(ROOT.TH1.kXaxis)
    h.SetBarWidth(0.4)
    h.SetBarOffset(0.1)
    expect(
        (h2.CanExtendAllAxes(), False),
        (h.GetBarWidth(), 0.4),
        (h.GetBarOffset(), 0.1),
        (h._xrd._core["fBarWidth"], 400),
    )


def test_macros_are_loaded_once_by_the_interpreter(monkeypatch):
    seen = []
    execute = types.ModuleType("xrdroot.pycint.execute")
    execute.process_line = seen.append
    monkeypatch.setitem(sys.modules, "xrdroot.pycint.execute", execute)
    try:
        assert not ROOT.gInterpreter.IsLoaded("m.C")
        ROOT.gInterpreter.LoadMacro("m.C")
        assert ROOT.gInterpreter.IsLoaded("m.C")
        ROOT.gInterpreter.ExecuteMacro("n.C")
    finally:
        troot.set_line_processor(None)
    assert seen == [".L m.C", ".x n.C"]


def test_a_two_dimensional_histogram_fills_at_random_as_root_does():
    ROOT.gRandom.SetSeed(5)
    ROOT.TF2("f2", "exp(-0.5*(x*x+y*y))", -3, 3, -3, 3)
    h = ROOT.TH2D("h", "", 10, -3, 3, 10, -3, 3)
    h.FillRandom("f2", 2000)
    expect(
        (h.GetEntries(), 2000),
        (round(h.GetMean(1), 6), -0.0066),
        (round(h.GetStdDev(2), 6), 1.028238),
        (h.GetBinContent(5, 5), 124),
        (h.GetBinContent(6, 6), 103),
    )
    g = ROOT.TH2D("g", "", 10, -3, 3, 10, -3, 3)
    g.FillRandom(h, 500)
    expect((g.GetEntries(), 500), (round(g.GetMean(1), 6), 0.011745))
    g.FillRandom(ROOT.TH2D("empty", "", 2, 0, 1, 2, 0, 1), 10)
    g.FillRandom("f2", 0, ROOT.TRandom3(1))
    assert g.GetEntries() == 500


def test_a_sequence_is_walked_by_c_plus_plus_s_random_access_iterator():
    """``ROOT::TSeq``'s ``begin()``/``end()`` are C++ random-access iterators: ``+``/``-`` by
    a count, the difference of two for how far apart they are, ``--``/``-=`` in place,
    ``it[n]`` a value n further on, and ``*it`` the value it is at - which a ``TSeq`` works
    out rather than looks up, so ``c[3]`` is the fourth value on from ``c``."""
    seq = ROOT.TSeqI(0, 6, 2)
    start = seq.begin()
    expect(
        (seq.end() - start, 3),
        (seq.end() - start, int(seq.size())),
        ((start + 3).value, 6),
        ((start + 3)[0], 6),
        ((start + 3)[-1], 4),
        (start[3], 6),
        ((start + 1) == start + 1, True),
        ((start + 1) == seq.begin() + 1, True),
        ((start + 1) != start, True),
        ((start + 1) > start, True),
        (start < start + 1, True),
        (start >= start, True),
        (start <= start, True),
        (bool(start == "a place that is no place"), False),
        (repr(start), "<TSeq iterator at 0>"),
    )
    moved = start + 2
    moved -= 1
    expect((moved == start + 1, True), ((moved - 1).value, 0))
    moved += 1
    expect((moved.value, 4), (moved - start, 2))
    expect((ROOT.TSeqI([7, 8, 9]).value_at(2), 2), (ROOT.TSeqI(3).value_at(2), 2))
    expect((ROOT.TSeqUL(0, 4).size(), 4), (ROOT.TSeqU(0, 4).begin().value, 0))


def test_sequences_classes_by_name_and_thread_safety_as_root_names_them():
    seq = ROOT.TSeqI(2, 8, 2)
    ROOT.EnableThreadSafety()
    ROOT.ROOT.EnableThreadSafety()
    expect(
        (list(seq), [2, 4, 6]),
        ((len(seq), seq.size(), seq.front(), seq.back(), seq[1]), (3, 3, 2, 6, 4)),
        (list(ROOT.ROOT.TSeqL(3)), [0, 1, 2]),
        (ROOT.gROOT.GetClass("TH1F").InheritsFrom("TH1"), True),
    )


def test_a_bitset_is_set_from_the_bytes_of_an_array_and_given_back_as_bytes():
    """``TBits::Set(nbits, array)`` takes the *bytes* an array of any width holds - an array
    of ``Char_t``, ``Short_t`` or ``Int_t`` is one buffer however wide its elements are -
    and makes the bitset exactly ``nbits`` long; ``Get`` copies the bytes back out."""
    import numpy as np

    chars = np.array([0, 1, 3, 7, 15, 31, 63, 127], dtype=np.int8)
    bits = ROOT.TBits()
    bits.Set(64, chars)
    expect(
        (bits.GetNbits(), 64),
        # The first byte is 0, so bits 0-7 are clear; the second is 1, so bit 8 is set.
        ([bits.TestBitNumber(bit) for bit in range(9)], [False] * 8 + [True]),
    )
    buffer = np.zeros(8, dtype=np.int8)
    bits.Get(buffer)
    assert buffer.tolist() == chars.tolist()
    # An array of wider elements is its bytes in order, which is what C's memory holds:
    # ``{0x0307, 0x033F}`` is ``07 03 3F 03``, so byte 2 is 0x3F and bits 16..21 are set.
    wider = np.array([0x0307, 0x033F], dtype=np.int16)
    short_bits = ROOT.TBits()
    short_bits.Set(32, wider)
    back = np.zeros(4, dtype=np.int8)
    short_bits.Get(back)
    expect(
        (short_bits.GetNbits(), 32),
        (back.tolist(), [7, 3, 0x3F, 3]),
        (short_bits.TestBitNumber(0), True),
        (short_bits.TestBitNumber(8), True),
        (short_bits.TestBitNumber(21), True),
        (short_bits.TestBitNumber(22), False),
        (short_bits.TestBitNumber(31), False),
    )


def test_a_bitsets_bytes_go_into_an_array_of_any_element_width() -> None:
    """ROOT's ``TBits::Get`` has an overload per element type and each fills ``GetNbytes()``
    *bytes*, so an array of ``Short_t`` takes them two at a time and one of ``Int_t`` four -
    the memory an array of that width is, not one element per byte."""
    import numpy as np

    bits = ROOT.TBits()
    bits.Set(32, np.array([7, 3, 63, 3], dtype=np.int8))
    shorts = np.zeros(2, dtype=np.int16)
    bits.Get(shorts)
    assert shorts.tolist() == [0x0307, 0x033F]
    as_list = [0, 0, 0, 0]
    bits.Get(as_list)
    assert as_list == [7, 3, 63, 3]
