"""The engine's objects, made of the records :mod:`.stream` read out of a workspace.

Each RooFit class has a maker here that reads the members its streamer
wrote - a proxy's ``_arg``, a list proxy's ``_list``, a variable's
``_value`` and binning - and calls the engine's constructor with them, so
what comes out is the same model the file holds, sharing nodes as the file
shares them. A class with no maker is refused by name: a model with a piece
missing would evaluate to something, and that something would be wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ..shared import PROPERTIES, PROPERTIES_LOCK, SharedBinnings
from .stream import Streamed

__all__ = ["Builder", "MAKERS"]

#: A maker for each class, by ROOT's name: what it is made of, into an engine object.
MAKERS: dict[str, Callable[[Builder, Streamed], Any]] = {}


def maker(*names: str) -> Callable[[Callable[[Builder, Streamed], Any]], Any]:
    def register(make: Callable[[Builder, Streamed], Any]) -> Any:
        for name in names:
            MAKERS[name] = make
        return make

    return register


class Builder:
    """What turns records into objects, each once: nodes shared in the file are shared here."""

    def __init__(self) -> None:
        self.made: dict[int, Any] = {}
        self.validated_schema: set[int] = set()
        self.shared_binnings: dict[tuple[Any, ...], SharedBinnings] = {}
        self._building = False

    def node(self, record: Any) -> Any:
        """The engine's object for ``record``, made the first time it is asked for."""
        if self._building:
            return self._node(record)
        with PROPERTIES_LOCK:
            return self._transaction(record)

    def _transaction(self, record: Any) -> Any:
        saved = (self.made.copy(), self.validated_schema.copy(), self.shared_binnings.copy())
        self._building = True
        try:
            made = self._node(record)
            PROPERTIES.update(self.shared_binnings)
            return made
        except BaseException:
            # Makers can register an unfinished workspace so its children can refer back.
            # A refused graph must never leave those incomplete objects reusable.
            self.made, self.validated_schema, self.shared_binnings = saved
            raise
        finally:
            self._building = False

    def _node(self, record: Any) -> Any:
        if record is None:
            return None
        found = self.made.get(id(record))
        if found is not None:
            return found
        make = MAKERS.get(record.cls)
        if make is None:
            raise UnsupportedFeatureError(
                f"the workspace holds a {record.cls} ({name_of(record)!r}), a class this RooFit "
                "engine does not have, so the model it is part of cannot be read"
            )
        from ..pdf import UNCHECKED
        from .schema import validate

        validate(record, self.validated_schema)
        UNCHECKED[0], was = True, UNCHECKED[0]
        try:
            made = make(self, record)
        finally:
            UNCHECKED[0] = was
        self.made[id(record)] = made
        if hasattr(made, "_attributes"):
            from .links import validate as proxy_links

            made._persisted_proxy_order = proxy_links(record)
            dress(made, record)
        config = record.get("_specIntegratorConfig")
        if config is not None:
            made.setIntegratorConfig(self.node(config))
        return made

    def nodes(self, records: Any) -> list[Any]:
        return [self.node(one) for one in records or ()]

    def proxy(self, record: Streamed, member: str) -> Any:
        """What the proxy ``member`` points at."""
        held = record.get(member)
        return None if held is None else self.node(held.get("_arg"))

    def listed(self, record: Streamed, member: str) -> list[Any]:
        """What the list proxy - or set - ``member`` holds, in order."""
        held = record.get(member)
        return [] if held is None else self.nodes(held.get("_list"))


def name_of(record: Any) -> str:
    """A ``TNamed``'s name, wherever in the record's bases it is."""
    named = record.get("TNamed") if isinstance(record, Streamed) else None
    return str(named.get("fName", "")) if isinstance(named, dict) else ""


def title_of(record: Streamed) -> str:
    named = record.get("TNamed")
    return str(named.get("fTitle", "")) if isinstance(named, dict) else ""


def dress(node: Any, record: Streamed) -> None:
    """A node's attributes as they were written: its boolean and string attributes."""
    for name in record.get("_boolAttrib") or ():
        node.setAttribute(str(name))
    for key, value in (record.get("_stringAttrib") or {}).items():
        node.setStringAttribute(str(key), str(value))
    if record.get("_forceNumInt"):
        node.setForceNumInt(True)
    _unit(node, record.get("_unit"))
    _labels(node, record)


def _labels(node: Any, record: Streamed) -> None:
    label, norm_range = record.get("_label"), record.get("_normRange")
    if label:
        node.setPlotLabel(str(label))
    if norm_range:
        node.setNormRange(str(norm_range))


def _unit(node: Any, unit: Any) -> None:
    if unit and hasattr(node, "setUnit"):
        node.setUnit(str(unit))


def values_of(items: Any) -> np.ndarray[Any, Any]:
    return np.asarray(list(items or ()), dtype=np.float64)
