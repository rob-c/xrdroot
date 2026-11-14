"""UHI's serialization of ROOT's histograms: ``h._to_uhi_()``, ``TH1D(ir)`` and
``TH1D._from_uhi_(ir)``.

A histogram becomes UHI's intermediate representation, the dictionary
``uhi.io.json``, ``uhi.io.zip`` and ``uhi.io.hdf5`` write, and is made again
from one: ROOT's class and name under ``writer_info``, each axis regular -
or, binned unevenly, by its edges - and the bins, flow and all, with the
squared weights of a histogram keeping them. A histogram made from one goes
into no directory and takes the name ``writer_info`` gives it, or a new one.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from ....errors import UnsupportedFeatureError
from .indexing import _temporarily_disable_add_directory
from .plotting import PlottableAxisBase, _hasWeights, _inner, _shape
from .tags import _get_axis

__all__: list[str] = []


def _axis_to_dict(root_axis: Any, uhi_axis: PlottableAxisBase) -> dict[str, Any]:
    """One ``TAxis``: regular, or by its edges when its bins are uneven."""
    traits = uhi_axis.traits
    flags = {
        "underflow": traits.underflow, "overflow": traits.overflow, "circular": traits.circular
    }
    if root_axis.IsVariableBinSize():
        edges = [root_axis.GetBinLowEdge(b) for b in range(1, root_axis.GetNbins() + 2)]
        return {"type": "variable", "edges": np.array(edges, dtype=np.float64), **flags}
    return {
        "type": "regular",
        "lower": root_axis.GetXmin(),
        "upper": root_axis.GetXmax(),
        "bins": root_axis.GetNbins(),
        **flags,
    }


def _axis_from_dict(axis_dict: dict[str, Any]) -> list[Any]:
    """The constructor's arguments for one axis: a count and two ends, or a count and edges."""
    axis_type = axis_dict["type"]
    if axis_type == "regular":
        return [axis_dict["bins"], axis_dict["lower"], axis_dict["upper"]]
    if axis_type == "variable":
        edges = np.asarray(axis_dict["edges"], dtype=np.float64)
        return [len(edges) - 1, edges]
    raise ValueError(f"Unsupported axis type for conversion to ROOT: {axis_type}")


def _profile_cells(hist: Any, name: str) -> np.ndarray[Any, Any]:
    """One of a profile's per-bin sums shaped as its bins, flow and all: its weights,
    ``fBinEntries``, or their squares, ``fBinSumw2`` - the weights themselves for a
    profile filled without any, where every weight was one."""
    engine = hist._xrd
    held = engine._writable(engine.members, name)
    if held is None:
        held = engine._writable(engine.members, "fBinEntries")
    return np.asarray(held).reshape(_shape(hist), order="F").copy()


def _storage_to_dict(hist: Any) -> dict[str, Any]:
    """The bins, and what kind of storage holds them.

    A profile's is UHI's weighted mean: the means, their variances, and the
    weights and squared weights in each bin. Any other histogram's is
    weighted when it keeps the squares of its weights, else of integers or
    of doubles as its class keeps its bins.
    """
    storage: dict[str, Any] = {"values": np.array(hist.values(flow=True))}
    if hist.kind == "MEAN":
        storage["type"] = "weighted_mean"
        storage["variances"] = hist.variances(flow=True)
        storage["sum_of_weights"] = _profile_cells(hist, "fBinEntries")
        storage["sum_of_weights_squared"] = _profile_cells(hist, "fBinSumw2")
    elif _hasWeights(hist):
        storage["type"] = "weighted"
        storage["variances"] = hist.variances(flow=True)
    else:
        storage["type"] = "int" if hist.ClassName()[-1] in "CSIL" else "double"
    return storage


def _generate_unique_hist_name(prefix: str = "h_uhi", nhex: int = 8) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:nhex]}"


def _get_ctor_args(self: Any, uhi_dict: dict[str, Any]) -> list[Any]:
    """The constructor's arguments: the name ``writer_info`` gives, or a new one, as name and
    title, then each axis."""
    name = uhi_dict.get("writer_info", {}).get("ROOT", {}).get("name")
    if name is None:
        name = _generate_unique_hist_name()
    ctor_args: list[Any] = [name, name]
    for axis_dict in uhi_dict["axes"]:
        ctor_args.extend(_axis_from_dict(axis_dict))
    return ctor_args


def _set_histogram_storage_from_dict(hist: Any, storage_dict: dict[str, Any]) -> None:
    """The bins set from the storage, and its squared weights where it has them."""
    if storage_dict.get("type") in ("mean", "weighted_mean"):
        raise UnsupportedFeatureError(
            "A profile cannot be made from UHI's mean storage yet: its means alone do not say "
            "what was summed to make them. Fill a TProfile from the original values instead."
        )
    hist[...] = storage_dict["values"]
    if storage_dict.get("type") != "weighted":
        return
    hist.Sumw2()
    variances = np.asarray(storage_dict["variances"], dtype=np.float64)
    squares = hist.GetSumw2().reshape(_shape(hist), order="F")
    flow = variances.shape == squares.shape
    squares[_inner(flow, hist.GetDimension())] = variances
    hist.ResetStats()


def _is_uhi_dict(obj: object) -> bool:
    return isinstance(obj, dict) and "uhi_schema" in obj


def _validate_uhi_dict_for_root_ctor(uhi_dict: dict[str, Any]) -> None:
    version = uhi_dict.get("uhi_schema")
    if version != 1:
        raise ValueError(f"Unsupported UHI schema version: {version}")
    if "axes" not in uhi_dict or "storage" not in uhi_dict:
        raise ValueError("Invalid UHI dict: missing required keys 'axes' and 'storage'.")


def _TH1_Constructor(self: Any, uhi_dict: dict[str, Any]) -> None:
    """``TH1D(ir)``: the histogram UHI's dictionary describes, kept in no directory."""
    from xrdroot.pyroot.core.histcore import Booked

    _validate_uhi_dict_for_root_ctor(uhi_dict)
    with _temporarily_disable_add_directory():
        Booked.__init__(self, *_get_ctor_args(self, uhi_dict))
    _set_histogram_storage_from_dict(self, uhi_dict["storage"])


def _to_uhi_(self: Any) -> dict[str, Any]:
    """The histogram as UHI's intermediate representation."""
    from xrdroot.pyroot import __version__

    return {
        "uhi_schema": 1,
        "writer_info": {
            "ROOT": {"version": __version__, "class": self.ClassName(), "name": self.GetName()}
        },
        "axes": [
            _axis_to_dict(_get_axis(self, i), self.axes[i]) for i in range(self.GetDimension())
        ],
        "storage": _storage_to_dict(self),
    }


def _from_uhi_(cls: Any, uhi_dict: dict[str, Any]) -> Any:
    """``TH1D._from_uhi_(ir)``: a new histogram of this class from UHI's dictionary."""
    if not _is_uhi_dict(uhi_dict):
        raise ValueError("Input is not a valid UHI dictionary.")
    _validate_uhi_dict_for_root_ctor(uhi_dict)
    return cls(uhi_dict)
