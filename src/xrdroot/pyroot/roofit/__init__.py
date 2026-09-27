"""RooFit under ``import ROOT``: :mod:`xrdroot.roofit`'s classes by ROOT's names.

``ROOT.RooRealVar``, ``ROOT.RooGaussian``, ``ROOT.RooFit.Save()``,
``ROOT.RooMsgService.instance()`` are the engine's own classes and
functions: RooFit's names are the engine's names already, so this module
only gathers them, and connects the two things the engine leaves to the
kit - what drawing a frame puts on the pad (the ``Draw`` hook of
:mod:`xrdroot.pyroot.core`), and what a frame's axis is (``TAxis``, when the
core part has one).
"""

from __future__ import annotations

import importlib
from typing import Any

from ...roofit.plot import frame as _frame
from ..core import draw_hook
from .commands import RooFit

#: The engine's modules, and the names each gives ``ROOT``.
EXPORTS = {
    "arg": ["RooAbsArg"],
    "binning": ["RooAbsBinning", "RooBinning", "RooRangeBinning", "RooUniformBinning"],
    "cmdargs": ["RooCmdArg", "RooLinkedList"],
    "collections": ["RooAbsCollection", "RooArgList", "RooArgSet"],
    "data.dataset": ["RooDataSet"],
    "data.store": ["RooAbsData"],
    "fitting.minimizer": ["RooMinimizer"],
    "fitting.nll": ["RooNLLVar"],
    "fitting.result": ["RooFitResult"],
    "functions": ["RooAddition", "RooFormulaVar", "RooPolyVar", "RooProduct"],
    "matrix": ["TMatrixDSym", "TVectorD"],
    "messages": ["RooMsgService"],
    "pdf": ["RooAbsPdf"],
    "pdfs.addpdf": ["RooAddPdf", "RooRecursiveFraction"],
    "pdfs.generic": ["RooGenericPdf"],
    "pdfs.prodpdf": ["RooProdPdf"],
    "pdfs.basic": ["RooChebychev", "RooExponential", "RooGaussian", "RooPolynomial", "RooUniform"],
    "plot.curve": ["RooCurve"],
    "plot.frame": ["RooPlot"],
    "plot.hist": ["RooHist"],
    "printing": ["RooPrintable"],
    "real": ["RooAbsReal"],
    "rng": ["RooRandom"],
    "variables": ["RooAbsRealLValue", "RooConstVar", "RooRealVar"],
}

__all__ = ["RooFit"]


def _gather() -> None:
    for module, names in EXPORTS.items():
        found = importlib.import_module(f"xrdroot.roofit.{module}")
        for name in names:
            if name in ("TMatrixDSym", "TVectorD") and name in globals():
                continue  # pragma: no cover - the core part's own
            globals()[name] = getattr(found, name)
            __all__.append(name)


def _axis() -> Any:
    """The core part's ``TAxis`` over a frame's axis members, or the engine's stand-in."""
    try:
        from ..core.axes import TAxis  # the swap point: the core part's TAxis, when it is there
    except ImportError:
        return _frame.Axis
    return TAxis._of  # pragma: no cover - once the core part is merged


_gather()
_frame.set_drawer(draw_hook)
_frame.set_axis(_axis())
