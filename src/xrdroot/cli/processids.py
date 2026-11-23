"""Fresh process UUIDs may be renamed only while preserving their equivalence classes."""

from __future__ import annotations

import uuid
from typing import Any

from ..file import Directory


def identity(title: str) -> str | None:
    try:
        return str(uuid.UUID(title))
    except ValueError:
        return None


def aliases(directory: Directory) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in sorted(directory.keys()):
        if directory.key(name).classname == "TProcessID":
            title = directory[name]["TNamed"]["fTitle"]
            uid = identity(title)
            if uid is not None:
                found.setdefault(uid, name)
    return found


def normalized(value: dict[str, Any], names: dict[str, str]) -> dict[str, Any]:
    title = value["TNamed"]["fTitle"]
    uid = identity(title)
    if uid is None:
        return value
    return {**value, "TNamed": {**value["TNamed"], "fTitle": "<process " + names[uid] + ">"}}
