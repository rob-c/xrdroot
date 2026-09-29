"""The Reader: methods read back from their weight files, and asked about one event at a time."""

from __future__ import annotations

from array import array

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import VARIABLES, classify, session, weights
from xrdroot.tmva import TMVAError
from xrdroot.tmva.log import CONFIG

__all__ = ["session"]

#: A factory that says nothing.
QUIET = "!V:Silent:AnalysisType=Classification"


class Cell:
    """What the translator makes of ``&var``: something holding a ``.value``."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value


class Text(str):
    """A ``TString``: a string with ``Data()``."""

    def Data(self) -> str:
        return str(self)


def _reader(options: str = "!Color:Silent") -> tuple[object, list[Cell]]:
    reader = ROOT.TMVA.Reader(options)
    cells = [Cell() for _ in VARIABLES]
    for variable, cell in zip(VARIABLES, cells):
        reader.AddVariable(variable, cell)
    return reader, cells


def test_a_reader_answers_what_the_factory_tested_for_the_same_event(session):
    factory = classify([(ROOT.TMVA.Types.kLD, "LD", "")], QUIET, output="")
    reader, cells = _reader()
    reader.BookMVA("LD method", weights("LD"))
    tested = factory.booked["dataset"][0]
    first = tested.loader.dataset().test.values[0]
    for cell, value in zip(cells, first):
        cell.value = value
    assert reader.EvaluateMVA("LD method") == pytest.approx(float(tested.test_values[0]), 1e-6)
    assert reader.EvaluateMVA(Text("LD method")) == reader.EvaluateMVA(list(first), "LD method")
    assert reader.GetMVAError() == 0.0  # left alone, as a Reader without Error leaves it
    assert reader.FindMVA("LD method") is reader.FindCutsMVA("LD method")
    assert reader.FindMVA("nothing") is None


def test_a_reader_says_what_it_books_and_rebuilds_its_data_set_once(session, capsys):
    classify([(ROOT.TMVA.Types.kFisher, "Fisher", "")], QUIET, output="")
    reader = ROOT.TMVA.Reader(list(VARIABLES), "Color")
    assert CONFIG.use_color
    CONFIG.use_color = False
    reader.AddSpectator("spec", array("f", [0.0]))
    reader.BookMVA(weights("Fisher"))
    reader.EvaluateMVA([0.1, 0.2, 0.3, 0.4], weights("Fisher"))
    reader.EvaluateMVA([0.1, 0.2, 0.3, 0.4], weights("Fisher"))
    printed = capsys.readouterr().out
    assert f'Booking "{weights("Fisher")}" of type "Fisher"' in printed
    assert 'Booked classifier "Fisher" of type: "Fisher"' in printed
    assert printed.count("Rebuilding Dataset Default") == 1


def test_a_reader_asked_for_a_method_it_has_not_booked_names_those_it_has(session):
    classify([(ROOT.TMVA.Types.kLD, "LD", "")], QUIET, output="")
    reader, _ = _reader()
    reader.BookMVA("LD", weights("LD"))
    with pytest.raises(TMVAError, match='you looked for "BDT" while the available methods are: LD'):
        reader.EvaluateMVA("BDT")


def test_the_probability_and_rarity_come_from_the_output_densities(session):
    classify([(ROOT.TMVA.Types.kLikelihood, "L", "CreateMVAPdfs")], QUIET, output="")
    reader, cells = _reader()
    reader.BookMVA("L", weights("L"))
    for cell in cells:
        cell.value = np.float32(0.5)
    value = reader.EvaluateMVA("L")
    assert reader.GetProba("L") == reader.GetProba("L", 0.5, value)
    assert 0.0 <= reader.GetProba("L", 0.2) <= 1.0
    assert reader.GetRarity("L") == reader.GetRarity("L", value)
    assert 0.0 <= reader.GetRarity("L") <= 1.0


def test_a_method_without_output_densities_warns_and_answers_no_probability(session, capsys):
    classify([(ROOT.TMVA.Types.kLD, "LD", "")], QUIET, output="")
    reader, _ = _reader("!Color")
    reader.BookMVA("LD", weights("LD"))
    assert reader.GetProba("LD") == -1.0
    assert reader.GetRarity("LD") == 0.0
    printed = capsys.readouterr().out
    assert "<GetProba> MVA PDFs for Signal and Background don't exist" in printed
    assert "<GetRarity> Required MVA PDF" in printed


def test_a_cuts_method_is_asked_at_the_signal_efficiency_given(session):
    cuts = "!H:!V:FitMethod=MC:SampleSize=2000:EffSel"
    classify([(ROOT.TMVA.Types.kCuts, "Cuts", cuts)], QUIET, output="")
    reader, cells = _reader()
    reader.BookMVA("Cuts", weights("Cuts"))
    for cell in cells:
        cell.value = 1.0
    assert reader.EvaluateMVA("Cuts", 0.9) in (0.0, 1.0)
    assert reader.FindCutsMVA("Cuts").test_signal_eff == 0.9
    reader.EvaluateMVA("Cuts")
    assert reader.FindCutsMVA("Cuts").test_signal_eff == -1.0


def test_a_reader_asked_for_errors_gets_the_none_a_method_without_them_gives(session):
    classify([(ROOT.TMVA.Types.kLD, "LD", "")], QUIET, output="")
    reader = ROOT.TMVA.Reader(list(VARIABLES), "!Color:Silent:Error")
    reader.BookMVA("LD method", weights("LD"))
    reader.EvaluateMVA([0.1, 0.2, 0.3, 0.4], "LD method")
    assert reader.GetMVAError() == -1.0
