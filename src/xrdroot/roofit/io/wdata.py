"""Vector stores retain dataset observables separately from workspace parameters."""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from .records import named, put, record
from .stream import Streamed
from .wnodes import Nodes


def dataset(nodes: Nodes, data: Any) -> Streamed:
    if data.ClassName() != "RooDataSet" or data.isWeighted():
        raise UnsupportedFeatureError(
            "Writing this workspace dataset is not implemented. "
            "Save an unweighted RooDataSet with real-valued observables instead."
        )
    variables = list(data.get())
    if any(one.ClassName() != "RooRealVar" for one in variables):
        raise UnsupportedFeatureError(
            "Writing categorical workspace datasets is not implemented. "
            "Save a RooDataSet with real-valued observables instead."
        )
    made = record("RooDataSet")
    named(made, data.GetName(), data.GetTitle())
    put(made, "storageType", 2)
    put(made, "_vars", nodes.listed(variables, owned=True))
    put(made, "_varsNoWgt", nodes.listed(variables))
    put(made, "_dstore", _store(nodes, data, variables))
    globs = data.getGlobalObservables()
    if globs is not None:
        put(made, "_globalObservables", nodes.listed(list(globs), owned=True))
    return made


def _store(nodes: Nodes, data: Any, variables: list[Any]) -> Streamed:
    made = record("RooVectorDataStore")
    named(made, data.GetName(), data._store_title or data.GetTitle())
    put(made, "_vars", nodes.listed(variables))
    put(made, "_varsww", nodes.listed(variables))
    put(made, "_realStoreList", [_column(nodes, data, one) for one in variables])
    put(made, "_sumWeight", float(data.numEntries()))
    put(made, "_doDirtyProp", True)
    return made


def _column(nodes: Nodes, data: Any, variable: Any) -> Streamed:
    made = record("RooVectorDataStore::RealVector")
    put(made, "_vec", list(map(float, data.column(variable.GetName()))))
    put(made, "_nativeReal", nodes.node(variable))
    put(made, "_real", nodes.node(variable))
    return made
