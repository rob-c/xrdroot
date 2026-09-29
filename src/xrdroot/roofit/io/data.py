"""The datasets of a workspace read from a file: ``RooDataSet`` and ``RooDataHist``.

A dataset's events are in its store - ``RooVectorDataStore``, a column per
real variable and per category - and its weights are the column of its
weight variable; a binned dataset keeps its bins' weights, and their
squares, as arrays of its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from .build import Builder, maker, name_of, title_of, values_of
from .stream import Streamed

__all__: list[str] = []


def _variables(builder: Builder, record: Streamed, member: str) -> list[Any]:
    held = record.get(member)
    return [] if held is None else builder.nodes(held.get("_list"))


@maker("RooDataSet")
def _dataset(builder: Builder, record: Streamed) -> Any:
    """The events of the vector store: a column per variable, and the weight variable's."""
    from ..data.dataset import RooDataSet

    store = record.get("_dstore")
    if store is None or store.cls != "RooVectorDataStore":
        kind = "no store" if store is None else f"a {store.cls}"
        raise UnsupportedFeatureError(f"the dataset {name_of(record)!r} is kept in {kind}; "
                                      "datasets are read from RooVectorDataStore")  # fmt: skip
    weight = record.get("_wgtVar")
    weight_name = name_of(weight) if weight is not None else ""
    variables = _variables(builder, record, "_varsNoWgt")
    options = {"WeightVar": weight_name} if weight_name else {}
    listed = variables + ([builder.node(weight)] if weight_name else [])
    made = RooDataSet(name_of(record), title_of(record), listed, **options)
    columns: dict[str, Any] = {}
    for vector in (store.get("_realStoreList") or []) + (store.get("_realfStoreList") or []):
        columns[name_of(vector.get("_nativeReal"))] = values_of(vector.get("_vec"))
    for vector in store.get("_catStoreList") or []:
        columns[name_of(vector.get("_cat"))] = np.asarray(vector.get("_vec"), dtype=np.float64)
    made._columns = {one.GetName(): columns[one.GetName()] for one in made.get()}
    made._store_title = str(store.get("TNamed", {}).get("fTitle", ""))
    if weight_name:
        made._weights = columns[weight_name]
    globs = record.get("_globalObservables")
    if globs is not None:
        made.setGlobalObservables(builder.nodes(globs.get("_list")))
    return made


@maker("RooDataHist")
def _datahist(builder: Builder, record: Streamed) -> Any:
    """The bins' weights and their squares; the variables with the binnings they were made
    with."""
    from ..data.datahist import RooDataHist

    made = RooDataHist(name_of(record), title_of(record), _variables(builder, record, "_vars"))
    weights = values_of(record.get("_wgt"))
    made._weights = weights
    sumw2 = values_of(record.get("_sumw2"))
    made._sumw2 = sumw2 if len(sumw2) == len(weights) else weights.copy()
    return made
