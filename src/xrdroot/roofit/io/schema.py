"""Custom, versioned wire records are checked throughout the persisted object graph."""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from .layouts import LAYOUTS
from .stream import Streamed

#: These classes stream their wire fields by hand instead of their declared members.
CUSTOM_LIMITS = {"RooRealVar": 10, "RooAbsBinning": 2, "RooLinkedList": 3,
                 "RooRefArray": 1, "RooWorkspace::CodeRepo": 2, "TList": 5, "THashList": 5}


def real_fields(version: int) -> set[str]:
    fields = {"RooAbsRealLValue", "_error", "_asymErrLo", "_asymErrHi"}
    if version == 1:
        fields.add("fit")
    if version >= 2:
        fields.add("_binning")
    if version >= 3:
        fields.add("_sharedProp")
    return fields


def custom_fields(classname: str, version: int) -> set[str]:
    if classname == "RooRealVar":
        return real_fields(version)
    if classname in ("TList", "THashList"):
        return {"items", "options"} | ({"TObject", "fName"} if version > 3 else set())
    if classname == "RooLinkedList":
        return {"TObject", "items"} | ({"_name"} if 1 < version < 4 else set())
    return {
        "RooAbsBinning": {"TNamed", "RooPrintable"},
        "RooRefArray": {"TObject", "fName", "fLowerBound", "pid", "uids"},
        "RooWorkspace::CodeRepo": {"files", "relations", "extras"},
    }[classname]


def declared(record: Streamed, classname: str) -> set[str] | None:
    if record.version is None:  # Synthetic test/engine records have no wire header.
        return None
    if classname in CUSTOM_LIMITS:
        if not 1 <= record.version <= CUSTOM_LIMITS[classname]:
            raise UnsupportedFeatureError(
                f"Reading persisted {classname} version {record.version} is not implemented. "
                "Use a supported workspace layout instead of discarding its fields."
            )
        return custom_fields(classname, record.version)
    layout = LAYOUTS.get(classname)
    return None if layout is None else {member[0] for member in layout[1]}


def members(record: Streamed) -> None:
    classname = record.cls or record.m.get("__class__", "")
    fields = declared(record, classname)
    if fields is None:
        return
    extra = set(record.m) - fields - {"__class__"}
    if classname == "RooWorkspace::CodeRepo" and record.version == 1 and record.get("extras"):
        extra.add("extras")  # The decoder supplies only an empty default for version one.
    if extra:
        raise UnsupportedFeatureError(
            f"Reading persisted {classname} members {', '.join(sorted(extra))} is not "
            "implemented. Use a supported workspace layout instead of discarding its fields."
        )


def visit(value: Any, seen: set[int]) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Streamed):
        members(value)
        values = value.m.values()
    elif isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return
    for held in values:
        visit(held, seen)


def validate(record: Streamed, visited: set[int] | None = None) -> None:
    seen = set() if visited is None else set(visited)
    visit(record, seen)
    if visited is not None:
        visited.update(seen)  # A failed validation must not cache a partially checked graph.
