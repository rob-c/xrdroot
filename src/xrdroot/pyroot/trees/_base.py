"""What the trees stand on: the core's classes, and the hooks the core sets.

A tree is a ``TNamed`` with ROOT's line, fill and marker attributes; it goes
into the directory that is current when it is made, and what its ``Draw``
fills is kept in ``gDirectory`` and drawn on the current pad. All of that is
``core``'s and ``graphics``'s to own, and this module is where they meet:

* :class:`_TObjectLike` is the base every class here derives from: the
  core's ``TNamed`` with ``TAttLine``, ``TAttFill`` and ``TAttMarker``.
* :data:`hooks` holds four functions the core sets when it is imported
  (:mod:`xrdroot.pyroot.core.treelinks`):
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

from ..core.objects import TAttFill, TAttLine, TAttMarker, TNamed, TObject

__all__ = ["_TObjectLike", "hooks", "ListOf"]


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
        self.draw: Callable[[Any, str], None] = lambda obj, option: self.drawn.append((obj, option))


#: Where ``core`` and ``graphics`` connect: see the module's docstring.
hooks = _Hooks()


class _TObjectLike(TNamed, TAttLine, TAttFill, TAttMarker):
    """``TNamed``, with the line, fill and marker attributes a tree has: the core's own.

    ``ClassName`` and ``InheritsFrom`` go by :attr:`_classname`, the C++ class
    each class here stands for, so a ``TNtuple`` says it is one; the rest -
    ``SetLineColor`` and its kin, ``Print``, ``ls``, bits - is ``TNamed``'s.
    """

    #: The C++ class this stands for, which ``ClassName`` says.
    _classname = "TObject"

    def __init__(self, name: str = "", title: str = "") -> None:
        TNamed.__init__(self, str(name), str(title))

    def ClassName(self) -> str:
        return self._classname

    def InheritsFrom(self, classname: Any) -> bool:
        """Whether this is a ``classname``, by the C++ names of this class and its bases."""
        wanted = classname.GetName() if hasattr(classname, "GetName") else str(classname)
        mine = any(getattr(kind, "_classname", None) == wanted for kind in type(self).__mro__)
        return mine or TNamed.InheritsFrom(self, wanted)

    def __getattr__(self, name: str) -> Any:
        """A method ROOT's class may have and this one has not, refused by name."""
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"ROOT's {self._classname} has {name}, or does not; xrdroot.pyroot's does not"
        )


class ListOf(list, TObject):  # type: ignore[type-arg]
    """``TObjArray``, as a tree's lists are: a Python list, and a ``TObject`` of ROOT's."""

    def ClassName(self) -> str:
        return "TObjArray"

    def GetName(self) -> str:
        return "TObjArray"

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


def _connect() -> None:
    """Take the core's hooks, if the core finished loading before the trees did."""
    from ..core import treelinks

    treelinks.connect()


_connect()
