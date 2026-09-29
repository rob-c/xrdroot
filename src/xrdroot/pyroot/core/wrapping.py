"""The xrdroot object each pyroot class stands for, and the wrapper each xrdroot object gets.

``TFile.Get`` hands back ROOT's classes, not xrdroot's, so every object read
must find the pyroot class that stands for it: by its ROOT class name first
(a ``TH1F`` is a ``TH1F``), then by the kind of xrdroot object it is. A
part of the kit with classes of its own - trees, canvases - adds them with
:func:`register`, and an object nobody claims comes back as it was read.

A wrapper made for an object already wrapped is the one made before, so
``f.Get("h") is f.Get("h")``, as ROOT's pointers are the same pointer.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable

__all__ = ["register", "register_members", "from_members", "wrap", "unwrap", "adopt"]

#: Wrappers by ROOT class name.
BY_CLASSNAME: dict[str, Callable[[Any], Any]] = {}
#: Wrappers by xrdroot type, tried in the order registered.
BY_TYPE: list[tuple[type, Callable[[Any], Any]]] = []
#: The wrapper already made for each xrdroot object, while both live.
#: Held weakly, so a wrapper nobody holds goes, and with it the entry.
_MADE: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()


def register(
    classnames: Any = (), kind: type | None = None, *, factory: Callable[[Any], Any]
) -> None:
    """Wrap objects of ``classnames`` - or of the xrdroot type ``kind`` - with ``factory``."""
    for name in [classnames] if isinstance(classnames, str) else classnames:
        BY_CLASSNAME[name] = factory
    if kind is not None:
        BY_TYPE.append((kind, factory))


def _factory(obj: Any) -> Callable[[Any], Any] | None:
    found = BY_CLASSNAME.get(str(getattr(obj, "classname", "")))
    if found is not None:
        return found
    return next((made for kind, made in BY_TYPE if isinstance(obj, kind)), None)


def wrap(obj: Any) -> Any:
    """The pyroot object standing for ``obj``: the same one each time, or ``obj`` if none does."""
    if obj is None or hasattr(obj, "_xrd"):
        return obj
    held = _MADE.get(id(obj))
    if held is not None and held._xrd is obj:
        return held
    factory = _factory(obj)
    if factory is None:
        return obj
    made = factory(obj)
    remember(obj, made)
    return made


def remember(obj: Any, wrapper: Any) -> None:
    """Note ``wrapper`` as what stands for ``obj``, forgotten when the wrapper goes."""
    _MADE[id(obj)] = wrapper


def unwrap(obj: Any) -> Any:
    """The xrdroot object a pyroot one stands for; anything else as it is."""
    return getattr(obj, "_xrd", obj)


def adopt(cls: Any, xrd: Any) -> Any:
    """A ``cls`` standing for ``xrd`` without running its constructor - what reading makes."""
    made = cls.__new__(cls)
    made._adopted(xrd)
    remember(xrd, made)
    return made


#: Makers of pyroot objects from the members of a record read, by ROOT class name: for the
#: classes the reader hands back as their members, which a part of the kit makes again.
FROM_MEMBERS: dict[str, Callable[[dict[str, Any]], Any]] = {}


def register_members(classname: str, factory: Callable[[dict[str, Any]], Any]) -> None:
    """Make a record of ``classname`` read as members into ``factory``'s object."""
    FROM_MEMBERS[classname] = factory


def from_members(classname: str, members: Any) -> Any:
    """The object ``factory`` makes of a record's members, or ``None`` if none is registered."""
    factory = FROM_MEMBERS.get(classname)
    return factory(members) if factory is not None and isinstance(members, dict) else None
