"""``ROOT::Math::MinimizerOptions``' process-wide defaults, which RooStats reads.

A RooStats calculator minimises with ``DefaultStrategy()``, ``DefaultTolerance()``
and ``DefaultPrintLevel()``, and a script may set them first -
``ROOT.Math.MinimizerOptions.SetDefaultStrategy(0)`` - so the defaults are one
table, here, that ``ROOT.Math.MinimizerOptions`` sets and the engine reads.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DEFAULTS", "default", "minimizer_algo", "minimizer_type", "set_minimizer"]

#: ``MinimizerOptions``' defaults, as a ROOT session starts with them.
DEFAULTS: dict[str, Any] = {
    "Minimizer": "",  # unset until asked for: then .rootrc's ``Root.Fitter``, Minuit2
    "Algorithm": "Migrad",
    "Tolerance": 0.01,
    "Precision": -1.0,
    "PrintLevel": 0,
    "MaxFunctionCalls": 0,
    "MaxIterations": 0,
    "Strategy": 1,
    "ErrorDef": 1.0,
}


def default(key: str) -> Any:
    """``MinimizerOptions::Default<key>()``."""
    return DEFAULTS[key]


def minimizer_type() -> str:
    """``DefaultMinimizerType``: Minuit2 - ``Root.Fitter`` - from the first time it is asked."""
    if not DEFAULTS["Minimizer"]:
        DEFAULTS["Minimizer"] = "Minuit2"
    return str(DEFAULTS["Minimizer"])


def minimizer_algo() -> str:
    """``DefaultMinimizerAlgo``: Migrad - but asked before any minimizer was, when none is set
    yet, it forgets Migrad for good, as ROOT's does."""
    if DEFAULTS["Algorithm"] == "Migrad" and DEFAULTS["Minimizer"] not in ("Minuit", "Minuit2"):
        DEFAULTS["Algorithm"] = ""
    return str(DEFAULTS["Algorithm"])


def set_minimizer(kind: Any, algo: Any = None) -> None:
    """``SetDefaultMinimizer(type, algo)``: Migrad the algorithm of Minuit's if none is named."""
    if kind is not None:
        DEFAULTS["Minimizer"] = str(kind)
    if algo is not None:
        DEFAULTS["Algorithm"] = str(algo)
    if not DEFAULTS["Algorithm"] and DEFAULTS["Minimizer"] in ("Minuit", "Minuit2"):
        DEFAULTS["Algorithm"] = "Migrad"
