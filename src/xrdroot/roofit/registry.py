"""Every RooFit class this package has, by ROOT's name - for the factory and for ``import ROOT``.

The workspace factory makes a ``Gaussian::g(...)`` by finding ``RooGaussian``
here, and :mod:`xrdroot.pyroot.roofit` gives ``ROOT`` the same names, so
the two never disagree about what exists.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["MODULES", "classes", "find"]

#: Each module of the engine, and the names of ROOT's classes it defines.
MODULES = {
    "arg": ["RooAbsArg"],
    "binning": ["RooAbsBinning", "RooBinning", "RooRangeBinning", "RooUniformBinning"],
    "categories": [
        "RooAbsCategory",
        "RooBinningCategory",
        "RooCategory",
        "RooMappedCategory",
        "RooMultiCategory",
        "RooSuperCategory",
        "RooThresholdCategory",
    ],
    "cmdargs": ["RooCmdArg", "RooLinkedList"],
    "collections": ["RooAbsCollection", "RooArgList", "RooArgSet"],
    "data.datahist": ["RooDataHist"],
    "data.dataset": ["RooDataSet"],
    "data.store": ["RooAbsData"],
    "fitting.minimizer": ["RooMinimizer"],
    "fitting.nll": ["RooNLLVar"],
    "fitting.profile": ["RooProfileLL"],
    "fitting.result": ["RooFitResult"],
    "functions": ["RooAddition", "RooFormulaVar", "RooPolyVar", "RooProduct"],
    "integral": ["RooRealIntegral"],
    "matrix": ["TMatrixDSym", "TVectorD"],
    "mcstudy": ["RooMCStudy"],
    "messages": ["RooMsgService"],
    "pdf": ["RooAbsPdf"],
    "pdfs.addmodel": ["RooAddModel"],
    "pdfs.addpdf": ["RooAddPdf", "RooRecursiveFraction"],
    "pdfs.anaconv": ["RooAbsAnaConvPdf"],
    "pdfs.basic": ["RooChebychev", "RooExponential", "RooGaussian", "RooPolynomial", "RooUniform"],
    "pdfs.bdecays": ["RooBCPEffDecay", "RooBCPGenDecay", "RooBDecay"],
    "pdfs.decays": ["RooBMixDecay", "RooDecay"],
    "pdfs.extend": ["RooExtendPdf"],
    "pdfs.fftconv": ["RooFFTConvPdf"],
    "pdfs.gaussmodel": ["RooGaussModel"],
    "pdfs.generic": ["RooGenericPdf"],
    "pdfs.histpdf": ["RooHistFunc", "RooHistPdf"],
    "pdfs.keys": ["RooKeysPdf"],
    "pdfs.realsum": ["RooRealSumPdf"],
    "pdfs.multivar": ["RooMultiVarGaussian"],
    "pdfs.histfactory": [
        "FlexibleInterpVar",
        "ParamHistFunc",
        "PiecewiseInterpolation",
        "RooBinWidthFunction",
    ],
    "pdfs.resolution": ["RooResolutionModel", "RooTruthModel"],
    "pdfs.prodpdf": ["RooProdPdf"],
    "pdfs.simultaneous": ["RooSimultaneous"],
    "pdfs.shapes": [
        "RooArgusBG",
        "RooBifurGauss",
        "RooBreitWigner",
        "RooCBShape",
        "RooLandau",
        "RooLognormal",
        "RooPoisson",
    ],
    "plot.curve": ["RooCurve"],
    "plot.frame": ["RooPlot"],
    "plot.hist": ["RooHist"],
    "printing": ["RooPrintable"],
    "real": ["RooAbsReal"],
    "rng": ["RooRandom"],
    "variables": ["RooAbsRealLValue", "RooConstVar", "RooRealVar"],
    "workspace": ["RooWorkspace"],
}


def classes() -> dict[str, Any]:
    """Every class, by name."""
    found: dict[str, Any] = {}
    for module, names in MODULES.items():
        loaded = importlib.import_module(f"xrdroot.roofit.{module}")
        for name in names:
            found[name] = getattr(loaded, name)
    return found


def find(name: str) -> Any:
    """The class ``name`` - or ``Roo<name>``, as the factory lets it be spelt - or ``None``."""
    table = classes()
    return table.get(name) or table.get(f"Roo{name}")
