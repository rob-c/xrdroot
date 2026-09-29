"""``ROOT::Math::MinimizerOptions``' process-wide defaults, which RooStats reads.

A RooStats calculator minimises with ``DefaultStrategy()``, ``DefaultTolerance()``
and ``DefaultPrintLevel()``, and a script may set them first -
``ROOT.Math.MinimizerOptions.SetDefaultStrategy(0)`` - so the defaults are one
table, here, that ``ROOT.Math.MinimizerOptions`` sets and the engine reads.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DEFAULTS", "default"]

#: ``MinimizerOptions``' defaults, as a ROOT session starts with them.
DEFAULTS: dict[str, Any] = {
    "Minimizer": "Minuit2",
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
