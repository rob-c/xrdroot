"""The end of a HistFactory channel: its total expectation, its model, its sets and its data.

``MakeTotalExpected`` sums each sample's shape - the product of its
functions and the bin width - times its scale factors in a binned
``RooRealSumPdf``; the model is the constraints times it, conditional on
the observables; the observed data a weighted dataset of the bins' centres,
and the Asimov data what the model expects.
"""

from __future__ import annotations

from typing import Any

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgList, RooArgSet
from ..roofit.messages import FATAL, INFO, PROGRESS, service
from .model import _hf
from .systematics import HistFactoryError

__all__ = ["dataset", "finish", "log_fatal", "total_expected"]


def log_fatal(text: str) -> None:
    """``cxcoutFHF``, then ``hf_exc``."""
    _hf(FATAL, text)
    raise HistFactoryError(f"HistFactory - {text}")


def _shape_name(first: str) -> str:
    """A sample's product of shape functions is named after its first: ``..._shapes``."""
    for marker in ("Hist_alpha", "nominal"):
        at = first.find(marker)
        if at >= 0:
            return first[:at] + "shapes"
    return first


def total_expected(ws: Any, name: str, scales: list[Any], funcs: list[list[Any]]) -> None:
    """``MakeTotalExpected``: ``<channel>_model``, the samples' shapes times their scales."""
    from ..roofit.functions import RooProduct
    from ..roofit.pdfs.histfactory import RooBinWidthFunction
    from ..roofit.pdfs.realsum import RooRealSumPdf
    from .terms import emplace

    first = funcs[0][0]
    if not first.InheritsFrom("RooHistFunc"):
        first = first.nominal
    width = emplace(ws, RooBinWidthFunction, f"{name}_binWidth", first, True)
    coefs, shapes = RooArgSet(), RooArgSet()
    for scale, own in zip(scales, funcs):
        coefs.add(scale)
        own.append(width)  # so always a product, which ROOT tests for one function and never is
        product = _shape_name(own[0].GetName())
        ws.Import(RooProduct(product, own[0].GetTitle(), own), RooCmdArg("RecycleConflictNodes"))
        shapes.add(ws.function(product))
    total = RooRealSumPdf(name, name, list(shapes), list(coefs), True)
    total.setForceNumInt(True)
    total.setAttribute("GenerateBinned")
    total.setAttribute("BinnedLikelihood")
    ws.Import(total, RooCmdArg("RecycleConflictNodes"))


def dataset(data: Any, hist: Any, ws: Any, names: list[str]) -> None:
    """``ConfigureHistFactoryDataset``: each bin's centre, weighted by its content."""
    axes = [hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis()][: len(names)]
    observables = ws.set("observables")

    def fill(level: int, index: tuple[int, ...]) -> None:
        axis = axes[level]
        for i in range(1, axis.GetNbins() + 1):
            ws.var(names[level]).setVal(axis.GetBinCenter(i))
            if level == len(names) - 1:
                data.add(observables, float(hist.GetBinContent(*index, i)), 0.0)
            else:
                fill(level + 1, (*index, i))

    fill(0, ())


def _hf_info_active() -> bool:
    from ..roofit.messages import TOPICS

    return bool(service().isActive(None, TOPICS["HistFactory"], INFO))


def _constraints(ws: Any, state: dict[str, Any]) -> list[Any]:
    """The channel's constraint terms - one missing from the workspace is fatal."""
    constraints = [ws.arg(one) for one in state["constraints"]]
    if any(one is None for one in constraints):
        missing = next(n for n, one in zip(state["constraints"], constraints) if one is None)
        log_fatal(f"Error: Cannot find arg set: {missing} in workspace: {ws.GetName()}")
    return constraints


def _model(ws: Any, channel: Any, constraints: list[Any], likelihood: list[Any],
           observables: Any) -> Any:  # fmt: skip
    """``model_<channel>``: the constraints times the channel's likelihood of its observables."""
    from ..roofit.pdfs.prodpdf import RooProdPdf

    _hf(PROGRESS, "\n-----------------------------------------\n\timport model into workspace\n"
        "-----------------------------------------\n")  # fmt: skip
    model = RooProdPdf(f"model_{channel.GetName()}", "product of Poissons across bins for a "
                       "single channel", constraints,
                       RooCmdArg("Conditional", likelihood, observables))  # fmt: skip
    data_hist = channel.GetData().GetHisto()
    if data_hist is not None and data_hist.GetTitle():
        model.SetTitle(data_hist.GetTitle())
    ws.Import(model, RooCmdArg("RecycleConflictNodes"))
    return model


def _data(ws: Any, channel: Any, names: list[str]) -> None:
    """The observed data, ``obsData``, and each additional dataset - one without a name fatal."""
    from ..roofit.data.dataset import RooDataSet

    data_hist = channel.GetData().GetHisto()
    if data_hist is not None:
        obs = RooDataSet("obsData", "", ws.set("observables"), RooCmdArg("WeightVar", "weightVar"))
        dataset(obs, data_hist, ws, names)
        ws.Import(obs)
    for extra in channel.GetAdditionalData():
        if not extra.GetName():
            log_fatal(f"Error: Additional Data histogram for channel: {channel.GetName()} has no "
                      "name! The name always needs to be set for additional datasets, either via "
                      'the "Name" tag in the XML or via RooStats::HistFactory::Data::SetName().')
        more = RooDataSet(extra.GetName(), "", ws.set("observables"),
                          RooCmdArg("WeightVar", "weightVar"))  # fmt: skip
        dataset(more, extra.GetHisto(), ws, names)
        ws.Import(more)


def finish(ws: Any, config: Any, channel: Any, names: list[str], state: dict[str, Any]) -> None:
    """The channel's sets, ``model_<channel>`` and its ``ModelConfig``, its Asimov and observed
    data - and the workspace printed, if HistFactory says what it does."""
    from ..roostats import asimov

    constraints = _constraints(ws, state)
    likelihood = [ws.arg(f"{channel.GetName()}_model")]
    ws.defineSet("constraintTerms", RooArgSet(constraints))
    ws.defineSet("likelihoodTerms", RooArgSet(likelihood))
    observables = RooArgList([ws.var(n) for n in names])
    ws.defineSet("observables", ",".join(names))
    ws.defineSet("observablesSet", ",".join(names))
    model = _model(ws, channel, constraints, likelihood, observables)
    config.SetPdf(ws.pdf(model.GetName()))
    config.SetObservables(observables)
    config.SetGlobalObservables(ws.set("globalObservables"))
    ws.Import(config, config.GetName())
    asimov.SetPrintLevel(1 if _hf_info_active() else 0)
    made = asimov.GenerateAsimovData(ws.pdf(model.GetName()), observables)
    ws.Import(made, RooCmdArg("Rename", "asimovData"))
    _data(ws, channel, names)
    if _hf_info_active():
        ws.Print()
