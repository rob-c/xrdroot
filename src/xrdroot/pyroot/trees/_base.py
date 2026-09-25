"""What the trees stand on until ``xrdroot.pyroot.core`` is beside them.

A tree is a ``TNamed`` with ROOT's line, fill and marker attributes; it goes
into the directory that is current when it is made, and what its ``Draw``
fills is kept in ``gDirectory`` and drawn on the current pad. All of that is
``core``'s and ``graphics``'s to own, so this module keeps the smallest
stand-in for each, in one place, for them to replace:

* :class:`_TObjectLike` is the base every class here derives from - to be
  ``core``'s ``TNamed`` with ``TAttLine``, ``TAttFill`` and ``TAttMarker``.
* :data:`hooks` holds four functions ``core`` sets when it is imported:
  ``directory()`` (the writable directory a new tree goes into - ``None``,
  a tree in memory, until a file is open), ``registry()`` (the mapping of
  name to xrdroot object that stands for ``gDirectory`` when ``Draw`` fills
  ``>>h``), ``wrap(obj)`` (the PyROOT object an xrdroot histogram or graph is
  handed back as) and ``draw(obj, option)`` (``core.draw_hook``).

A tree made while no file is open is kept in memory, as ROOT keeps one, and
is written by ``Write`` to whatever ``SetDirectory`` gave it.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

__all__ = ["_TObjectLike", "hooks", "ListOf"]

#: The attributes ``TAttLine``, ``TAttFill`` and ``TAttMarker`` give a tree, at ROOT's defaults.
ATTRIBUTES: dict[str, Any] = {
    "LineColor": 1,
    "LineStyle": 1,
    "LineWidth": 1,
    "FillColor": 0,
    "FillStyle": 1001,
    "MarkerColor": 1,
    "MarkerStyle": 1,
    "MarkerSize": 1.0,
}


class _Hooks:
    """The four places ``core`` plugs itself in; see the module's docstring."""

    def __init__(self) -> None:
        #: What ``Draw`` asked to be drawn, and with what option, while nothing draws.
        self.drawn: list[tuple[Any, str]] = []
        #: ``gDirectory``'s objects, while there is no ``gDirectory``.
        self.objects: dict[str, Any] = {}
        self.directory: Callable[[], Any] = lambda: None
        self.registry: Callable[[], MutableMapping[str, Any]] = lambda: self.objects
        self.wrap: Callable[[Any], Any] = lambda obj: obj
        self.draw: Callable[[Any, str], None] = lambda obj, option: self.drawn.append(
            (obj, option)
        )


#: Where ``core`` and ``graphics`` connect: see the module's docstring.
hooks = _Hooks()


class _TObjectLike:
    """``TNamed``, with the line, fill and marker attributes a tree has.

    To be replaced by ``core``'s classes: this is only what the trees call.
    ``SetLineColor(2)`` and ``GetLineColor()`` and the rest are kept and given
    back, since a tree's own drawing is ``graphics``'s to do with them.
    """

    #: The C++ class this stands for, which ``ClassName`` says.
    _classname = "TObject"

    def __init__(self, name: str = "", title: str = "") -> None:
        self._name = str(name)
        self._title = str(title)
        self._attributes = dict(ATTRIBUTES)

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def SetTitle(self, title: str = "") -> None:
        self._title = str(title)

    def SetNameTitle(self, name: str, title: str) -> None:
        self.SetName(name)
        self.SetTitle(title)

    def ClassName(self) -> str:
        return self._classname

    def InheritsFrom(self, classname: str) -> bool:
        """Whether this is a ``classname``, by the C++ names of this class and its bases."""
        return any(getattr(kind, "_classname", None) == classname for kind in type(self).__mro__)

    def __getattr__(self, name: str) -> Any:
        verb, attribute = name[:3], name[3:]
        if verb in ("Set", "Get") and attribute in ATTRIBUTES:
            if verb == "Get":
                return lambda: self._attributes[attribute]
            return lambda value: self._attributes.__setitem__(attribute, value)
        raise AttributeError(
            f"ROOT's {self._classname} has {name}, or does not; xrdroot.pyroot's does not"
        )


class ListOf(list):  # type: ignore[type-arg]
    """``TObjArray``, as far as a tree's lists need one: a list with ROOT's methods.

    To be replaced by ``core``'s ``TObjArray``.
    """

    def GetEntries(self) -> int:
        return len(self)

    GetEntriesFast = GetEntries
    GetSize = GetEntries

    def GetLast(self) -> int:
        return len(self) - 1

    def At(self, index: int) -> Any:
        return self[index] if 0 <= index < len(self) else None

    UncheckedAt = At

    def FindObject(self, name: str) -> Any:
        return next((item for item in self if item.GetName() == name), None)

    def First(self) -> Any:
        return self.At(0)

    def Last(self) -> Any:
        return self.At(len(self) - 1)
