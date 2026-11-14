"""ROOT's core objects as the writer takes them: a class and its members.

A ``TNamed``, a ``TObjString``, a bare ``TObject``, a ``TList`` of any of
them, a ``TRef``, a ``TRefArray`` and a ``TBits`` are no xrdroot object of
their own - they are ROOT's bookkeeping, and here they are the pyroot classes
of those names - so writing one hands the writer a
:class:`~xrdroot.writer.Record`: the class, and its members as a reader would
have read them back. A list's members are what it holds, each made the same
way, or the xrdroot object a pyroot one stands for.

An object a ``TRef`` may point at, and every reference, is written with the
index of its process among the file's ``ProcessID`` records, which the writer
adds to the file the first time a process is met.

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
#: ``TObject::kIsReferenced``: a ``TRef`` may point at the object.
IS_REFERENCED = 1 << 4


def _process_index(pid: Any, writer: Any) -> int:
    """The index of ``pid`` - the session's current process when there is none - in the file
    ``writer`` writes into."""
    if pid is None:
        from .references import TProcessID

        pid = TProcessID.GetPID()
    return int(writer.process_index(pid.GetTitle()))


def _tobject(obj: Any, writer: Any = None) -> dict[str, int]:
    found = {"fUniqueID": int(obj.GetUniqueID()), "fBits": HEAP_BITS | obj._bitword()}
    if writer is not None and obj.TestBit(IS_REFERENCED):
        from .references import TProcessID

        found["pidf"] = _process_index(TProcessID.GetProcessWithUID(obj, obj), writer)
    return found


def _named(obj: Any, writer: Any) -> dict[str, Any]:
    return {"TObject": _tobject(obj, writer), "fName": obj.GetName(), "fTitle": obj.GetTitle()}


def _objstring(obj: Any, writer: Any) -> dict[str, Any]:
    return {"TObject": _tobject(obj, writer), "fString": obj.GetName()}


def _held(obj: Any, writer: Any) -> list[Any]:
    return [recorded(item, writer) for item in obj]


def _tref(obj: Any, writer: Any) -> dict[str, Any]:
    plain = {"fUniqueID": obj.GetUniqueID(), "fBits": HEAP_BITS | obj._bitword()}
    return {"TObject": plain, "pidf": _process_index(obj.GetPID(), writer)}


def _trefarray(obj: Any, writer: Any) -> dict[str, Any]:
    plain = {"fUniqueID": obj.GetUniqueID(), "fBits": HEAP_BITS | obj._bitword()}
    uids = obj._uids[: obj.GetEntriesFast()]
    return {
        "TObject": plain,
        "fName": obj.GetName(),
        "fLowerBound": obj.GetLowerBound(),
        "pidf": _process_index(obj.GetPID(), writer),
        "fUIDs": uids,
    }


def _tbits(obj: Any, writer: Any) -> dict[str, Any]:
    return {"TObject": _tobject(obj, writer), **obj._members()}


#: How each class is made into members, by the name ROOT gives it.
MEMBERS = {
    "TObject": _tobject,
    "TNamed": _named,
    "TObjString": _objstring,
    "TList": _held,
    "THashList": _held,
    "TRef": _tref,
    "TRefArray": _trefarray,
    "TBits": _tbits,
}
#: The classes that, written, need a file to say which of its processes they name.
PROCESSED = ("TRef", "TRefArray")


def recorded(obj: Any, writer: Any = None) -> Any:
    """What the writer is handed for ``obj``: a record of one of ROOT's core classes, the
    xrdroot object a pyroot one stands for, or ``obj`` itself. ``writer`` is the file being
    written into, which an object a reference names, and a reference, are numbered in."""
    made = unwrap(obj)
    if made is not obj:
        return made
    classname = _classname(obj)
    members = MEMBERS.get(classname)
    if members is None or (writer is None and classname in PROCESSED):
        return obj
    name = obj.GetName() if classname in ("TList", "THashList") else ""
    return Record(classname, members(obj, writer), obj.GetTitle(), name)


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


def _bits_from(members: Any) -> Any:
    from .bits import TBits

    return _identity(TBits._from_members(members), members)


def _entry(item: Any, classname: str) -> Any:
    """One object a list read holds, as the namespace has it."""
    from .files import TOther
    from .objects import TObject

    made = wrap(item)
    if isinstance(made, TObject) or not isinstance(made, (dict, list)):
        return made
    found = from_members(classname, made)
    return TOther(classname or "TObject", made) if found is None else found


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
    register_members("TBits", _bits_from)


_register()
