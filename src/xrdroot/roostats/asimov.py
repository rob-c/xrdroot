"""``AsymptoticCalculator::GenerateAsimovData``: the data a model expects, bin by bin.

Each bin of the observables - their binnings, the first slowest - gets the
density at its centre times its volume times the expected number of
events, as a weight; bins expecting nothing are skipped, and said so. A
product's factors that do not depend on the observables are dropped first;
a simultaneous model's states each make theirs, indexed by the category.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, INFO, WARNING, log

__all__ = ["GenerateAsimovData", "SetPrintLevel", "print_level"]

#: ``fgPrintLevel``: 0 quiet, 1 each dataset printed, 2 every bin said.
_LEVEL = [1]


def SetPrintLevel(level: int) -> None:
    _LEVEL[0] = int(level)


def print_level() -> int:
    return _LEVEL[0]


def _fill(pdf: Any, observables: list[Any], data: Any, index: int, volume: float) -> None:
    """``FillBins``: the bins of the observables from ``index`` on, recursively."""
    var = observables[index]
    names = frozenset(one.GetName() for one in observables)
    expected = pdf.expected(names)
    for i in range(var.getBins()):
        var.setBin(i)
        if index < len(observables) - 1:
            _fill(pdf, observables, data, index + 1, volume * var.getBinWidth(i))
            continue
        value = float(pdf.getVal(RooArgSet(observables))) * volume * var.getBinWidth(i)
        if value * expected <= 0:
            what = "has negative expected events! Please check your inputs." if (
                value * expected < 0) else "has zero expected events - skip it"  # fmt: skip
            log(None, WARNING, "InputArguments", f"AsymptoticCalculator::FillBins(): Bin {i} of "
                f"{var.GetName()} {what}")  # fmt: skip
        else:
            data.add(RooArgSet(observables), value * expected)
    var.setBin(0)  # the bins' values reset, as FillBins leaves them


def _single(pdf: Any, observables: Any, weight: Any, category: Any = None) -> Any:
    """``GenerateAsimovDataSinglePdf``: one density's Asimov data."""
    from ..roofit.cmdargs import RooCmdArg
    from ..roofit.data.dataset import RooDataSet

    own = list(pdf.getObservables(observables))  # the density's own, which it reads
    if not pdf.canBeExtended():
        from .asimovcount import counting_asimov_data

        return counting_asimov_data(pdf, RooArgSet(own), category)
    columns = [*own, weight] + ([category] if category is not None else [])
    if category is not None:
        index = category.getCurrentIndex()
        data = RooDataSet(f"AsimovData{index}", f"combAsimovData{index}", columns,
                          RooCmdArg("WeightVar", weight))  # fmt: skip
    else:
        data = RooDataSet("AsimovData", "AsimovData", columns, RooCmdArg("WeightVar", weight))
    _fill(pdf, own, data, 0, 1.0)
    if _LEVEL[0] >= 1:
        data.Print()
    if math.isnan(data.sumEntries()):
        log(None, ERROR, "Generation", "sum entries is nan")
        return None
    return data


def _observable_part(pdf: Any, observables: Any) -> Any:
    """A product's factors of the observables - the one, if one - or the density itself."""
    if not pdf.InheritsFrom("RooProdPdf"):
        return pdf
    from ..roofit.pdfs.prodpdf import RooProdPdf

    kept = [one for one in pdf.pdfList() if one.dependsOn(observables)]
    if len(kept) == 1:
        return kept[0]
    return RooProdPdf("observableProdPdf", "observableProdPdf", kept)


def GenerateAsimovData(pdf: Any, observables: Any) -> Any:
    """The Asimov data of ``pdf`` in ``observables``, weighted by ``binWeightAsimov``."""
    from ..roofit.cmdargs import RooCmdArg
    from ..roofit.data.dataset import RooDataSet
    from ..roofit.variables import RooRealVar

    weight = RooRealVar("binWeightAsimov", "binWeightAsimov", 1, 0, 1.0e30)
    if _LEVEL[0] > 1:
        log(None, INFO, "Generation", " Generate Asimov data for observables")
    found = _observable_part(pdf, observables)
    if not found.InheritsFrom("RooSimultaneous"):
        return _single(found, observables, weight)
    category = found.indexCat()
    parts: dict[str, Any] = {}
    states = category.states()
    for label in sorted(states, key=lambda one: states[one]):  # by index, as setIndex(i) walks
        category.setLabel(label)
        part = _single(found.getPdf(label), observables, weight, category)
        if part is None:
            log(None, ERROR, "Generation", "Error generating an Asimov data set for pdf "
                f"{found.getPdf(label).GetName()}")  # fmt: skip
            return None
        parts[label] = part
    columns = [*as_list(observables), weight, category]
    return RooDataSet("asimovDataFullModel", "asimovDataFullModel", columns,
                      RooCmdArg("Index", category), RooCmdArg("Import", parts),
                      RooCmdArg("WeightVar", weight))  # fmt: skip
