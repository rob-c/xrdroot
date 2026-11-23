"""Compare saved definitions and alias relationships without dropping unfamiliar state."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ..arg import Proxy, RooAbsArg
from ..pdf import RooAbsPdf

#: Allocation and graph-walk bookkeeping is rebuilt when ROOT reopens a model.
ARG_TRANSIENT = frozenset({"_serial", "_configuration_version", "_walk_cache", "_dependents_cache"})
#: ROOT declares its normalisation context and private generator configuration transient.
PDF_TRANSIENT = frozenset({"_last_norm", "_norm_cache", "_norms_made", "_num_gen_config"})


def transient(owner: Any, name: str) -> bool:
    return ((isinstance(owner, RooAbsArg) and name in ARG_TRANSIENT)
            or (isinstance(owner, RooAbsPdf) and name in PDF_TRANSIENT))


def unsupported(value: Any) -> Any:
    raise UnsupportedFeatureError(
        f"Comparing persisted workspace state of type {type(value).__name__} is not "
        "implemented. Use supported RooFit definitions instead of discarding this state."
    )


class Graph:
    """Stable traversal labels retain shared objects and cycles, including workspace bindings."""

    def __init__(self) -> None:
        self.identities: dict[int, int] = {}
        self.objects: list[dict[str, Any]] = []

    def project(self, value: Any) -> dict[str, Any]:
        root = self.encode(value)
        return {"root": root, "objects": self.objects}

    def encode(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return self.encode(value.item())
        return self.reference(value)

    def reference(self, value: Any) -> dict[str, int]:
        identity = id(value)
        if identity not in self.identities:
            self.identities[identity] = len(self.objects)
            held: dict[str, Any] = {"class": type(value).__name__}
            self.objects.append(held)  # Register before traversing a back reference.
            held["state"] = self.contents(value)
        return {"reference": self.identities[identity]}

    def contents(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return unsupported(value)
            return {"dtype": str(value.dtype), "values": value}
        if isinstance(value, dict):
            return {key: self.encode(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [self.encode(one) for one in value]
        if isinstance(value, (set, frozenset)):
            return self.members(value)
        return self.fields(value)

    def members(self, values: Any) -> list[Any]:
        if not all(isinstance(one, (str, int, float, bool)) for one in values):
            return unsupported(values)
        return [self.encode(one) for one in sorted(values, key=repr)]

    def fields(self, value: Any) -> dict[str, Any]:
        if isinstance(value, Proxy):
            return {name: self.encode(getattr(value, name)) for name in Proxy.__slots__}
        module = type(value).__module__
        if not module.startswith(("xrdroot.roofit.", "xrdroot.roostats.")):
            return unsupported(value)
        return {name: self.encode(held) for name, held in sorted(vars(value).items())
                if not transient(value, name)}
