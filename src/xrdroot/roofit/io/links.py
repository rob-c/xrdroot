"""A proxy registry must resolve to the actual persisted proxy objects, not just names."""

from __future__ import annotations

from typing import Any

from ...errors import FormatError
from .stream import Streamed

#: Native object containers physically own dynamically allocated proxy declarations.
PROXY_CONTAINERS = frozenset({"TList", "THashList", "RooLinkedList"})


def is_proxy(value: Any) -> bool:
    return isinstance(value, Streamed) and ("Proxy<" in value.cls or value.cls.endswith("Proxy"))


def contained(value: Any) -> Any:
    if isinstance(value, Streamed):
        # An argument referenced by a server/client collection has its own declarations.
        return value.m.values() if value.cls in PROXY_CONTAINERS else ()
    if isinstance(value, dict):
        return value.values()
    return value if isinstance(value, (list, tuple)) else ()


def collection_proxies(value: Any, seen: set[int]) -> list[tuple[str, Streamed]]:
    if id(value) in seen:
        return []
    seen.add(id(value))
    if is_proxy(value):
        named = value.get("TNamed") or {}
        return [(str(named.get("fName", "")), value)]
    return [proxy for held in contained(value) for proxy in collection_proxies(held, seen)]


def proxies(record: Streamed) -> list[tuple[str, Streamed]]:
    found = []
    for name, held in record.m.items():
        if isinstance(held, Streamed) and held.cls == "":
            found.extend(proxies(held))
        elif is_proxy(held):
            found.append((name, held))
        else:
            found.extend(collection_proxies(held, set()))
    return found


def identity(proxy: Streamed) -> dict[str, Any]:
    named = proxy.get("TNamed")
    return named.get("TObject", {}) if named is not None else proxy.get("TObject", {})


def declarations(record: Streamed) -> dict[int, tuple[str, dict[str, Any]]]:
    found: dict[int, tuple[str, dict[str, Any]]] = {}
    objects: dict[int, Streamed] = {}
    for name, proxy in proxies(record):
        target = identity(proxy)
        uid = target.get("fUniqueID", 0)
        if not uid:  # Multiple unregistered, zero-UID proxy members are legitimate.
            continue
        if uid in objects and objects[uid] is not proxy:
            raise FormatError(
                "A persisted workspace proxy identifier has ambiguous declared targets."
            )
        if proxy.get("_owner") is not record:
            raise FormatError("A persisted workspace proxy owner is not its declaring object.")
        objects[uid] = proxy
        found.setdefault(uid, (name, target))
    return found


def validate(record: Streamed) -> list[str] | None:
    registry = record.get("_proxyList")
    if registry is None:
        return None
    declared = declarations(record)
    names = []
    for uid in registry.get("uids") or ():
        if uid == 0:
            names.append("<null>")
            continue
        if uid not in declared:
            raise FormatError("A persisted workspace proxy identifier has no declared target.")
        name, target = declared[uid]
        if target.get("pid") != registry.get("pid"):
            raise FormatError(
                "A persisted workspace proxy belongs to a different process identifier."
            )
        names.append(name)
    return names
