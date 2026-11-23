"""Workspace equality compares persisted definitions and data rather than Python identities."""

from __future__ import annotations

from typing import Any

from .graph import Graph


def state(obj: Any) -> dict[str, Any]:
    return Graph().project(obj)


def binning(obj: Any) -> dict[str, Any]:
    return {"class": type(obj).__name__, "name": obj.GetName(),
            "low": obj.lowBound(), "high": obj.highBound(), "bins": obj.numBins(),
            "boundaries": [] if obj.isUniform() else obj.array(),
            "parameterized": obj.isParameterized()}


def node(obj: Any, owned: dict[str, Any] | None = None) -> dict[str, Any]:
    found: dict[str, Any] = {"class": obj.ClassName(), "state": state(obj)}
    found["proxies"] = [
        {"name": proxy.name, "many": proxy.many, "shape": proxy.shape,
         "targets": [(one.ClassName(), one.GetName()) for one in proxy.args()]}
        for proxy in obj._proxies
    ]
    if owned is not None:
        found["owned_targets"] = [[owned.get(one.GetName()) is one for one in proxy.args()]
                                  for proxy in obj._proxies]
    found.update(_properties(obj))
    return found


def _properties(obj: Any) -> dict[str, Any]:
    found: dict[str, Any] = {
        "force_numeric_integration": bool(getattr(obj, "_force_num_int", False))}
    if hasattr(obj, "getBinning"):
        found["binning"] = binning(obj.getBinning())
        found["named_binnings"] = {name: binning(value) for name, value in obj._shared.items()}
    config = getattr(obj, "_num_int_config", None)
    if config is not None:
        found["integration"] = configuration(config)
    return found


def configuration(config: Any) -> dict[str, Any]:
    return {
        "epsAbs": config.epsAbs(), "epsRel": config.epsRel(),
        "counter": config.printEvalCounter(),
        "methods": {name: node(value) for name, value in config._methods.items()},
        "sections": {section.GetName(): {arg.GetName(): node(arg) for arg in section}
                     for section in config._sections},
    }


def dataset(data: Any) -> dict[str, Any]:
    return {
        "class": data.ClassName(), "name": data.GetName(), "title": data.GetTitle(),
        "variables": {one.GetName(): node(one) for one in data.get()},
        "columns": data.columns(), "weights": data.weights(),
        "weights_squared": data.weights_squared(), "weighted": data.isWeighted(),
    }


def workspace(value: Any) -> dict[str, Any]:
    found = {
        "name": value.GetName(), "title": value.GetTitle(),
        "reference_graph": Graph().project(value),
        "nodes": {name: node(arg, value._nodes) for name, arg in value._nodes.items()},
        "data": {name: dataset(data) for name, data in value._data.items()},
        "embedded_data": {name: dataset(data) for name, data in value._embedded.items()},
    }
    found.update(_groups(value))
    return found


def _groups(value: Any) -> dict[str, Any]:
    return {
        "sets": {name: [(arg.ClassName(), arg.GetName()) for arg in group]
                 for name, group in value._sets.items()},
        "snapshots": {name: {arg.GetName(): node(arg) for arg in args}
                      for name, args in value._snapshots.items()},
        "generic": {name: {"class": obj.ClassName(), "state": state(obj)}
                    for name, obj in value._generic.items()},
    }
