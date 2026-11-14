"""The host: where a translation finds every name it did not declare, looked up late.

A translation spells each name the program uses without declaring it as an
attribute of one object, the dialect's host - ``CXX.std.vector`` for plain
C++. A :class:`HostNamespace` is that object. It asks for the namespace it
looks names up in the first time one is wanted - so a translation can be
imported, read and kept without the library loading - or uses whatever
namespace :meth:`HostNamespace.bind` was handed, which is how a test runs a
program against a small namespace of its own. ``std`` is the standard
library's containers, from :mod:`xrdroot.cppstd.containers` unless the host
is given another, and ``testing`` is GoogleTest's.

What the programs run so far declared at their top is kept in
:data:`DECLARED`, as an interpreter keeps it, so that a line run later finds
a function or class an earlier program defined.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

__all__ = ["HostNamespace", "CXX", "DECLARED"]

#: What the programs run so far declared at their top - functions and classes - by name, as
#: an interpreter keeps them: a line run later finds them.
DECLARED: dict[str, Any] = {}

#: An empty namespace: what a host with no library of its own looks names up in.
EMPTY = types.ModuleType("<host>")


def _empty() -> Any:
    return EMPTY


def _standard() -> Any:
    """The standard library's ``std``: its containers and the rest of :mod:`xrdroot.cppstd`."""
    from ...cppstd.containers import std

    return std


def _testing() -> Any:
    """GoogleTest's ``testing``, which a unit test is written against."""
    from .gtest import testing

    return testing


class HostNamespace:
    """A namespace looked up late: bound to a module or to any object with attributes.

    ``default`` gives the namespace looked in when nothing is bound - an empty
    one unless given; ``used`` names the namespaces in it whose names a program
    uses unqualified, looked in when the namespace itself has no such name;
    ``provided`` gives more names the runtime itself provides; ``std`` gives
    the namespace that is the standard library.
    """

    def __init__(
        self,
        default: Callable[[], Any] = _empty,
        used: tuple[str, ...] = (),
        provided: Mapping[str, Callable[[], Any]] | None = None,
        std: Callable[[], Any] = _standard,
    ) -> None:
        self._bound: list[Any] = []
        #: The classes and modules found through a used namespace, by namespace and name.
        self._resolved: dict[tuple[int, str], Any] = {}
        self._default = default
        self._namespaces = used
        self._provided = {"testing": _testing, **(provided or {})}
        self._std = std

    def namespace(self) -> Any:
        """What names are looked up in now: the innermost :meth:`bind`, else the default."""
        if self._bound:
            return self._bound[-1]
        return self._default()

    @contextmanager
    def bind(self, namespace: Any) -> Iterator[Any]:
        """Look names up in ``namespace`` until the ``with`` block ends."""
        self._bound.append(namespace)
        try:
            yield namespace
        finally:
            self._bound.pop()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        namespace = self.namespace()
        found = self._resolved.get((id(namespace), name))
        if found is not None:
            return found
        try:
            return getattr(namespace, name)
        except AttributeError:
            if name != "std":
                return self._used(namespace, name)
            # ``std`` is what a translated loop asks for most - ``std::abs`` per element -
            # so it is remembered like :meth:`_used` remembers its finds: not looking it
            # up again spares the namespace's own search, per turn, inside the loop.
            module = self._std()
            self._resolved[(id(namespace), name)] = module
            return module

    def _used(self, namespace: Any, name: str) -> Any:
        """``name`` from a used namespace, declared or provided - or the refusal the
        namespace's own lookup gave.

        Anything callable found in a used namespace is remembered, since the
        search before it is what a program's inner loop would otherwise do at
        every call; the finds are stable members of the library's own modules.
        What a program declares is not remembered, since the next program may
        declare it again.
        """
        for used in self._namespaces:
            found = getattr(_member(namespace, used), name, None)
            if found is not None:
                if inspect.ismodule(found) or callable(found):
                    self._resolved[(id(namespace), name)] = found
                return found
        if name in DECLARED:
            return DECLARED[name]
        if name in self._provided:  # remembered: every assertion of a test looks it up
            provided = self._provided[name]()
            self._resolved[(id(namespace), name)] = provided
            return provided
        return getattr(namespace, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """``g = v``: a program assigning one of the library's globals - or, for a global
        that stands for something else, such as the current directory, telling it what to
        stand for now through its ``__assign__``."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        namespace = self.namespace()
        held = getattr(namespace, name, None)
        assign = getattr(type(held), "__assign__", None)
        if assign is not None:
            assign(held, value)
            return
        setattr(namespace, name, value)

    def __repr__(self) -> str:
        return "<the host: the names a translated program did not declare>"


def _member(namespace: Any, dotted: str) -> Any:
    """``A.B`` of a namespace: each name in turn, or ``None`` where one is missing."""
    for name in dotted.split("."):
        namespace = getattr(namespace, name, None)
    return namespace


#: The host of plain C++: the standard library, GoogleTest, and what programs declared.
CXX = HostNamespace()
