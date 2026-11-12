"""Native records keep identity separate from the model's mutable Python objects."""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

from ...errors import UnsupportedFeatureError
from .layouts import LAYOUTS
from .stream import BASIC, Streamed, _clean, _template, base


def put(record: Streamed, name: str, value: Any) -> None:
    """Replace a persisted member in its declaring class, including a base class."""
    if name in record.m:
        record.m[name] = value
        return
    for held in record.m.values():
        if isinstance(held, Streamed) and held.cls == "":
            try:
                put(held, name, value)
                return
            except KeyError:
                pass
    raise KeyError((record.cls, name))


def named(record: Streamed, name: str, title: str) -> None:
    if record.cls == "TNamed":
        put(record, "fName", name)
        put(record, "fTitle", title)
    else:
        put(record, "TNamed", {"fName": name, "fTitle": title, "TObject": {}})


def record(classname: str, *, inherited: bool = False) -> Streamed:
    if classname not in LAYOUTS:
        raise UnsupportedFeatureError(
            f"Writing the workspace class {classname} is not implemented. "
            "Save a workspace containing supported RooFit classes instead."
        )
    made = base(classname) if inherited else Streamed(classname)
    made.version = LAYOUTS[classname][0]
    for name, kind, typename, length in LAYOUTS[classname][1]:
        made.m[name] = _default(name, kind, typename, length)
    _custom_defaults(made, classname)
    return made


def _default(name: str, kind: int, typename: str, length: int) -> Any:
    if kind == 0:
        return record(name, inherited=True)
    if kind in (66, 67):
        return {} if kind == 66 else {"fName": "", "fTitle": "", "TObject": {}}
    if kind in BASIC:
        return False if kind == 18 else 0
    if kind - 20 in BASIC:
        return [0] * length
    return _held_default(kind, typename)


def _held_default(kind: int, typename: str) -> Any:
    if kind == 65:
        return ""
    if kind in (61, 62, 63):
        return record(typename.rstrip("*"))
    if kind in (64, 69):
        return None
    text = _clean(typename)
    if text in ("string", "TString"):
        return ""
    head, _args = _template(text)
    return {} if head in ("map", "unordered_map", "multimap") else []


def _custom_defaults(made: Streamed, classname: str) -> None:
    if classname == "RooRealVar":
        made.m["_sharedProp"] = record("RooRealVarSharedProperties")
    elif classname == "RooRefArray":
        made.m.update(TObject={}, fName="", fLowerBound=0, pid=0, uids=[])
    elif classname == "RooWorkspace::CodeRepo":
        made.m.update(files=[], relations=[], extras=[])
    elif classname == "RooLinkedList":
        # The custom wire count comes from items, not the C++ bookkeeping members.
        del made.m["_hashThresh"], made.m["_size"]
        made.m.update(TObject={}, items=[], _name="")
    elif classname == "TUUID":
        _uuid(made)


def _uuid(made: Streamed) -> None:
    low, mid, high, clock_high, clock_low, node = uuid.uuid4().fields
    made.m.update(
        fTimeLow=low, fTimeMid=mid, fTimeHiAndVersion=high,
        fClockSeqHiAndReserved=clock_high, fClockSeqLow=clock_low,
        fNode=list(node.to_bytes(6, "big")),
    )


def collection(classname: str, items: list[Any], name: str = "", owned: bool = False) -> Streamed:
    made = record(classname)
    put(made, "_list", items)
    put(made, "_name", name)
    put(made, "_ownCont", owned)
    return made


def linked(items: list[Any]) -> Streamed:
    made = record("RooLinkedList")
    made.m["items"] = items
    return made


def refcounts(items: list[Streamed]) -> Streamed:
    counts = Counter(items)
    made = record("RooSTLRefCountList<RooAbsArg>")
    made.m.update(_storage=list(counts), _refCount=list(counts.values()))
    return made
