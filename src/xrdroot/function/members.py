"""The members a ``TF1``, ``TF2``, ``TF3`` and ``TFormula`` are written with.

A function built here is held the way ROOT would write it - a ``TF1``'s
members, with its ``TFormula`` behind the ``fFormula`` pointer, or its
parameters in a ``TF1Parameters`` for one defined by code - so that what is
read from a file and what is made by hand are the one shape, and writing
either is writing its members. The defaults are what ROOT gives a function
it has just made: the red line of width two it draws a fit with, a hundred
points to draw it by, and ``-1111`` - ROOT's "not set" - for the plotting
limits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .shapes import param_order

__all__ = [
    "BITS",
    "NOT_GLOBAL",
    "CLASSES",
    "function_members",
    "formula_members",
    "parameters_members",
    "first_of",
    "ranges_of",
]

#: The bits a freshly made object carries: on the heap, and not deleted.
BITS = 0x03000000
#: ``TFormula::kNotGlobal``, which a ``TF1`` sets on the formula it owns.
NOT_GLOBAL = 1 << 10
#: The function classes by dimension.
CLASSES = {1: "TF1", 2: "TF2", 3: "TF3"}
#: ``TF1::EFType``: a formula, or a function of code.
FORMULA_TYPE = 0
CODE_TYPE = 1
#: ROOT's number of points to draw a function by, in one dimension and in more.
NPX = 100
NPX_GRID = 30
#: What ROOT writes for a plotting limit nobody set.
UNSET = -1111.0
#: The axis letters, and the member of each class that holds a range's ends.
LIMITS = (("fXmin", "fXmax"), ("fYmin", "fYmax"), ("fZmin", "fZmax"))


def _named(name: str, title: str, bits: int) -> dict[str, Any]:
    return {"fName": name, "fTitle": title, "fUniqueID": 0, "fBits": bits}


def formula_members(
    name: str, title: str, text: str, names: tuple[str, ...], values: Any, ndim: int
) -> dict[str, Any]:
    """A ``TFormula``: the expanded text, its parameters' names and values."""
    ordered = sorted(range(len(names)), key=lambda index: param_order(names[index]))
    return {
        "TNamed": _named(name, title, BITS | NOT_GLOBAL),
        "fClingParameters": np.array(values, dtype=np.float64),
        "fAllParametersSetted": True,
        "fParams": {names[index]: index for index in ordered},
        "fFormula": text,
        "fNdim": ndim,
        "fLinearParts": [],
        "fVectorized": False,
    }


def parameters_members(names: tuple[str, ...], values: Any) -> dict[str, Any]:
    """A ``TF1Parameters``, which a function of code keeps its parameters in."""
    return {"fParameters": np.array(values, dtype=np.float64), "fParNames": list(names)}


def _tf1(
    name: str,
    title: str,
    limits: list[tuple[float, float]],
    npar: int,
    formula: dict[str, Any] | None,
    parameters: dict[str, Any] | None,
) -> dict[str, Any]:
    ndim = len(limits)
    return {
        "TNamed": _named(name, title, BITS),
        "TAttLine": {"fLineColor": 2, "fLineStyle": 1, "fLineWidth": 2},
        "TAttFill": {"fFillColor": 19, "fFillStyle": 0},
        "TAttMarker": {"fMarkerColor": 1, "fMarkerStyle": 1, "fMarkerSize": 1.0},
        "fXmin": float(limits[0][0]),
        "fXmax": float(limits[0][1]),
        "fNpar": npar,
        "fNdim": ndim,
        "fNpx": NPX if ndim == 1 else NPX_GRID,
        "fType": FORMULA_TYPE if formula is not None else CODE_TYPE,
        "fNpfits": 0,
        "fNDF": 0,
        "fChisquare": 0.0,
        "fMinimum": UNSET,
        "fMaximum": UNSET,
        "fParErrors": np.zeros(npar),
        "fParMin": np.zeros(npar),
        "fParMax": np.zeros(npar),
        "fSave": np.zeros(0),
        "fNormalized": False,
        "fNormIntegral": 0.0,
        "fFormula": formula,
        "fParams": parameters,
        "fComposition": None,
    }


def function_members(
    name: str,
    title: str,
    limits: list[tuple[float, float]],
    npar: int,
    formula: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A ``TF1``, or the ``TF2`` or ``TF3`` built on one, over ``limits``, one pair per axis."""
    members = _tf1(name, title, limits, npar, formula, parameters)
    if len(limits) > 1:
        members = {
            "TF1": members,
            "fYmin": float(limits[1][0]),
            "fYmax": float(limits[1][1]),
            "fNpy": NPX_GRID,
            "fContour": np.zeros(0),
        }
    if len(limits) > 2:
        members = {
            "TF2": members,
            "fZmin": float(limits[2][0]),
            "fZmax": float(limits[2][1]),
            "fNpz": NPX_GRID,
        }
    return members


def first_of(members: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``TF1``, ``TF2`` and ``TF3`` layers of a function's members, ``TF1`` first."""
    layers = [members]
    while "TF1" in layers[0] or "TF2" in layers[0]:
        layers.insert(0, layers[0]["TF1"] if "TF1" in layers[0] else layers[0]["TF2"])
    return layers


def ranges_of(layers: list[dict[str, Any]], ndim: int) -> list[tuple[float, float]]:
    """Each axis's range, read from whichever layer holds it."""
    found = []
    for axis in range(ndim):
        low, high = LIMITS[axis]
        holder = next(layer for layer in layers if low in layer)
        found.append((float(holder[low]), float(holder[high])))
    return found
