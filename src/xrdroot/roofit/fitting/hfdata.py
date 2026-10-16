"""Native binning and data storage cannot invoke Python model callbacks during a fit."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..binning import RooAbsBinning, RooBinning, RooRangeBinning, RooUniformBinning
from ..collections import RooArgList, RooArgSet
from ..data.datahist import RooDataHist
from ..data.dataset import RooDataSet
from . import hfnative as native

#: Fixed binnings read numeric fields, without evaluating parameterized range functions.
BINNINGS = (RooAbsBinning, RooBinning, RooRangeBinning, RooUniformBinning)
#: The collection iterators read an ordinary list in its stored order.
COLLECTIONS = (RooArgList, RooArgSet)
#: The data stores retain concrete NumPy columns and weights.
DATA = (RooDataHist, RooDataSet)


def sequence(value: Any, kind: type[Any], item: Any) -> bool:
    return type(value) is kind and all(item(one) for one in value)


def mapping(value: Any, item: Any) -> bool:
    return type(value) is dict and all(type(key) is str and item(one)
                                       for key, one in value.items())


def members(value: Any) -> list[Any]:
    if not native.instance(value, COLLECTIONS) or type(value._list) is not list:
        raise TypeError("A compiled model requires a native argument collection.")
    return value._list


def endpoint(value: Any) -> bool:
    return native.number(value, finite=False) and not math.isnan(float(value))


def binning(value: Any) -> bool:
    if not native.instance(value, BINNINGS):
        return False
    return (endpoint(value._low) and endpoint(value._high)
            and type(value._bins) is int and value._bins > 0
            and type(value._uniform) is bool
            and sequence(value._edges, list, endpoint))


def variable(value: Any) -> bool:
    return (native.number(value._val) and native.number(value._error)
            and sequence(value._asym, tuple, native.number) and len(value._asym) == 2
            and binning(value._binning) and mapping(value._shared, binning))


def store(value: Any) -> bool:
    return (native.instance(value, DATA) and native.array(value._weights, floating=True)
            and mapping(value._columns, native.array))


def edges(value: Any) -> bool:
    return (type(value) is list and all(native.array(edge, floating=True)
                                       and bool(np.all(np.isfinite(edge))) for edge in value))


def interpolation(value: Any) -> bool:
    return (native.number(value._nominal) and native.number(value._boundary)
            and sequence(value._low, list, native.number)
            and sequence(value._high, list, native.number)
            and sequence(value._codes, list, lambda code: type(code) is int))
