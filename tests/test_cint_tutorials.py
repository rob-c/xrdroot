"""ROOT's own tutorial macros, translated and run end to end against a fake ROOT.

The macros are in ``tests/data/cint``, copied unchanged from ROOT 6.40.04.
What they print is compared with what they print under ROOT; what they do
to ROOT's objects is read back from the fake's record of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cintfake import fake
from xrdroot.cint import translate_file
from xrdroot.cint.execute import load, run
from xrdroot.cint.runtime import ROOT

#: Where the copied tutorials are.
TUTORIALS = Path(__file__).parent / "data" / "cint"


def test_permute_prints_every_permutation_of_four_and_of_five_with_repeats(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The macro's third part searches three million permutations of ten
    # digits, which a test has no time for; its first two are the same code.
    root = fake()
    namespace = load(TUTORIALS / "math" / "permute.C", root=root)
    with ROOT.bind(root):
        assert namespace["permuteSimple1"]() == 0
        assert namespace["permuteSimple2"]() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[3:5] == ["abcd", "abdc"]
    assert "Found 24 permutations = 4!" in lines
    assert lines[-2:] == ["cbbaa", "Found 30 permutations = 5!/(2! 2!)"]


def test_gr001_computes_its_points_and_reads_the_rest_from_a_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = fake(str(TUTORIALS))
    run(TUTORIALS / "visualisation" / "graphs" / "gr001_simple.C", root=root, use_cache=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == " i 0 0.000000 1.986693 "
    assert lines[19] == " i 19 1.900000 8.632094 "
    assert lines[20] == " i 0 -3.000000 -0.989992 "
    assert len(lines) == 40
    graphs = [thing for thing in root.made if thing.rest and thing.name == 20]
    assert len(graphs) == 2
    assert ("Draw", ("ACP",)) in graphs[0].calls


def test_chebyshev_builds_eleven_functions_and_a_legend_entry_for_each() -> None:
    root = fake()
    run(TUTORIALS / "math" / "ChebyshevPol.C", root=root, use_cache=False)
    functions = [thing for thing in root.made if thing.name == "f1"]
    assert [f.title for f in functions] == [f"cheb{n}" for n in range(11)]
    assert functions[3].calls[:2] == [("SetParameter", (3, 1)), ("SetLineColor", (619,))]
    assert ("Draw", ("",)) in functions[0].calls
    assert ("Draw", ("same",)) in functions[1].calls
    legend = root.made[0]
    assert [call[1][1] for call in legend.calls if call[0] == "AddEntry"][-1] == "N=10"


def test_legendre_builds_five_functions_through_an_array_of_pointers() -> None:
    root = fake()
    run(TUTORIALS / "math" / "Legendre.C", root=root, use_cache=False)
    functions = [thing for thing in root.made if thing.name == "L_0"]
    assert len(functions) == 5
    assert ("SetParameters", (4, 0.0)) in functions[4].calls
    assert ("SetMaximum", (1,)) in functions[0].calls


def test_hist000_fills_a_stack_histogram_and_writes_it_to_the_file() -> None:
    root = fake()
    run(TUTORIALS / "hist" / "hist000_TH1_first.C", root=root, use_cache=False)
    histogram = root.made[0]
    assert histogram.values == [1, 2, 3, 3, 3, 4, 3, 2, 1, 0]
    (written,) = root.TFile.opened
    assert written.calls == [("WriteObject", (histogram, "histogram"))]


def test_hist001_fills_randomly_and_writes_the_histogram() -> None:
    root = fake()
    run(TUTORIALS / "hist" / "hist001_TH1_fillrandom.C", root=root, use_cache=False)
    histogram = root.made[0]
    assert histogram.calls == [("FillRandom", ("gaus", 10000))]
    assert root.TFile.opened[0].name == "fillrandom.root"


def test_every_copied_tutorial_translates_into_python_that_compiles() -> None:
    for path in sorted(TUTORIALS.rglob("*.C")):
        compile(translate_file(path), str(path), "exec")
