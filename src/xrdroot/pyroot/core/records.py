"""ROOT's core objects as the writer takes them: a class and its members.

A ``TNamed``, a ``TObjString``, a bare ``TObject`` and a ``TList`` of any of
them are no xrdroot object of their own - they are ROOT's bookkeeping, and
here they are the pyroot classes of those names - so writing one hands the
writer a :class:`~xrdroot.writer.Record`: the class, and its members as a
reader would have read them back. A list's members are what it holds, each
made the same way, or the xrdroot object a pyroot one stands for.

Only the classes themselves are made into records: a macro's class derived
from ``TNamed`` has members of its own, which this does not know of.
"""

from __future__ import annotations

from typing import Any

from ...writer import Record
from .wrapping import from_members, register_members, unwrap, wrap

__all__ = ["recorded"]

#: ``TObject::kIsOnHeap | kNotDeleted``, the bits every object made with ``new`` carries.
HEAP_BITS = 0x03000000


def _tobject(obj: Any) -> dict[str, int]:
    return {"fUniqueID": int(obj.GetUniqueID()), "fBits": HEAP_BITS | obj._bitword()}


def _named(obj: Any) -> dict[str, Any]:
    return {"TObject": _tobject(obj), "fName": obj.GetName(), "fTitle": obj.GetTitle()}


def _objstring(obj: Any) -> dict[str, Any]:
    return {"TObject": _tobject(obj), "fString": obj.GetName()}


def _held(obj: Any) -> list[Any]:
    return [recorded(item) for item in obj]


#: How each class is made into members, by the name ROOT gives it.
MEMBERS = {
    "TObject": _tobject,
    "TNamed": _named,
    "TObjString": _objstring,
    "TList": _held,
    "THashList": _held,
}


def recorded(obj: Any) -> Any:
    """What the writer is handed for ``obj``: a record of one of ROOT's core classes, the
    xrdroot object a pyroot one stands for, or ``obj`` itself."""
    made = unwrap(obj)
    if made is not obj:
        return made
    members = MEMBERS.get(_classname(obj))
    if members is None:
        return obj
    name = obj.GetName() if _classname(obj) in ("TList", "THashList") else ""
    return Record(obj.ClassName(), members(obj), obj.GetTitle(), name)


def _classname(obj: Any) -> str:
    """The class a pyroot object of ROOT's own is - and nothing for anything else, such as a
    macro's class derived from one of them."""
    module = type(obj).__module__
    if not module.startswith("xrdroot.pyroot."):
        return ""
    return str(obj.ClassName())


# -- read back ----------------------------------------------------------------------------


def _identity(made: Any, members: Any) -> Any:
    """``made`` given the identifier and bits its record kept."""
    tobject = members if "fBits" in members else members.get("TObject", {})
    made.SetUniqueID(int(tobject.get("fUniqueID", 0)))
    made._store_bits(int(tobject.get("fBits", 0)) & ~HEAP_BITS)
    return made


def _named_from(members: Any) -> Any:
    from .objects import TNamed

    return _identity(TNamed(members.get("fName", ""), members.get("fTitle", "")), members)


def _objstring_from(members: Any) -> Any:
    from .collections import TObjString

    return _identity(TObjString(members.get("fString", "")), members)


def _tobject_from(members: Any) -> Any:
    from .objects import TObject

    return _identity(TObject(), members)


def _entry(item: Any, classname: str) -> Any:
    """One object a list read holds, as the namespace has it."""
    from .files import TOther
    from .objects import TObject

    made = wrap(item)
    if isinstance(made, TObject) or not isinstance(made, (dict, list)):
        return made
    return from_members(classname, made) or TOther(classname or "TObject", made)


def _list_from(cls: Any) -> Any:
    def made(items: Any) -> Any:
        held = cls()
        named = getattr(items, "classes", None) or [""] * len(items)
        for item, classname in zip(items, named, strict=True):
            held.Add(_entry(item, classname))
        return held

    return made


def _register() -> None:
    from .collections import THashList, TList

    register_members("TNamed", _named_from)
    register_members("TObjString", _objstring_from)
    register_members("TObject", _tobject_from)
    register_members("TList", _list_from(TList))
    register_members("THashList", _list_from(THashList))


_register()
