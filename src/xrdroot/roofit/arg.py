"""``RooAbsArg``: a node of a RooFit model - its name, what it is made of, and its attributes.

A model is a graph: a Gaussian is made of ``x``, ``mean`` and ``sigma``, a
sum of its components and their fractions. Each node keeps its inputs as
named *proxies* in the order its class declares them, which is what
``Print`` shows (``[ x=x mean=mean sigma=sigma ]``) and what the walks here
follow: :meth:`RooAbsArg.getVariables` for the leaves, ``getParameters`` for
those that are not the data's, ``getComponents`` for the nodes.

This is also where a node's ``TObject`` behaviour lives - ``GetName``,
``ClassName``, ``InheritsFrom`` - kept here rather than taken from
:mod:`xrdroot.pyroot.core`, so that the engine stands on xrdroot alone; a
``TObject`` of the pyroot layer can take its place as the base without
anything else changing.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from .collections import RooArgList, RooArgSet, as_list
from .printing import (
    RooPrintable,
    address,
    kArgs,
    kClassName,
    kName,
    kValue,
)
from . import cout

__all__ = ["Proxy", "RooAbsArg", "graph_changed"]

#: How many times any node's name or inputs have changed: what the cached walks are checked against.
_GRAPH = [0]


def graph_changed() -> None:
    """Forget every cached walk: a node was renamed, or given another input."""
    _GRAPH[0] += 1


class Proxy:
    """One named input of a node: a single argument, or a list of them."""

    __slots__ = ("name", "target", "many", "shape")

    def __init__(self, name: str, target: Any, many: bool = False, shape: bool = False) -> None:
        self.name = name
        self.target = target
        self.many = many
        self.shape = shape

    def args(self) -> list[Any]:
        return list(self.target) if self.many else [self.target]

    def text(self) -> str:
        """``RooAbsProxy::print``: ``name=arg``, or ``name=(a,b)`` for a list."""
        if self.many:
            return f"{self.name}=(" + ",".join(one.GetName() for one in self.target) + ")"
        return f"{self.name}={self.target.GetName()}"


class RooAbsArg(RooPrintable):
    """A node of a model: named, titled, made of other nodes, with attributes."""

    def __init__(self, name: Any = "", title: Any = "") -> None:
        self._name = str(name)
        self._title = str(title) if title is not None else ""
        self._proxies: list[Proxy] = []
        self._attributes: set[str] = set()
        self._strings: dict[str, str] = {}

    # -- TObject ------------------------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetName(self, name: str) -> None:
        self._name = str(name)
        graph_changed()

    def SetTitle(self, title: str) -> None:
        self._title = str(title)

    def SetNameTitle(self, name: str, title: str) -> None:
        self.SetName(name)
        self.SetTitle(title)

    def ClassName(self) -> str:
        return type(self).__name__

    def InheritsFrom(self, name: Any) -> bool:
        wanted = name if isinstance(name, str) else getattr(name, "__name__", str(name))
        return any(klass.__name__ == wanted for klass in type(self).__mro__)

    def IsA(self) -> Any:
        return type(self)

    def __repr__(self) -> str:
        return f"<{self.ClassName()}::{self._name}>"

    # -- inputs -------------------------------------------------------------------

    def _proxy(self, name: str, target: Any, shape: bool = False) -> Any:
        """Declare one input called ``name``; the argument itself is returned."""
        self._proxies.append(Proxy(name, target, False, shape))
        graph_changed()
        return target

    def _list_proxy(self, name: str, targets: Any) -> RooArgList:
        """Declare a list of inputs called ``name``."""
        made = RooArgList(as_list(targets))
        self._proxies.append(Proxy(name, made, True))
        graph_changed()
        return made

    def servers(self) -> list[Any]:
        """What this node is made of, each once, in the order its inputs were declared."""
        found: list[Any] = []
        for proxy in self._proxies:
            for one in proxy.args():
                if not any(one is seen for seen in found):
                    found.append(one)
        return found

    def numProxies(self) -> int:
        return len(self._proxies)

    def isFundamental(self) -> bool:
        return False

    def isDerived(self) -> bool:
        return True

    def _walk(self) -> Iterator[RooAbsArg]:
        """This node and every node under it, each once, depth first."""
        seen: list[RooAbsArg] = []
        stack: list[RooAbsArg] = [self]
        while stack:
            node = stack.pop(0)
            if any(node is one for one in seen):
                continue
            seen.append(node)
            yield node
            stack[0:0] = node.servers()

    def _walked(self) -> list[RooAbsArg]:
        """:meth:`_walk`, remembered until the graph changes."""
        cached = self.__dict__.get("_walk_cache")
        if cached is None or cached[0] != _GRAPH[0]:
            cached = (_GRAPH[0], list(self._walk()))
            self.__dict__["_walk_cache"] = cached
        return cached[1]

    def leaves(self) -> list[Any]:
        """The fundamental nodes under this one: its variables and constants."""
        return [node for node in self._walked() if node.isFundamental()]

    def getVariables(self, stripDisconnected: bool = True) -> RooArgSet:
        return RooArgSet(
            [one for one in self.leaves() if not one.InheritsFrom("RooConstVar")]
        ).sorted_copy()

    def getParameters(self, observables: Any = None, stripDisconnected: bool = True) -> RooArgSet:
        """The variables that are not ``observables`` - a set, or a dataset's columns."""
        exclude = set(_observable_names(observables))
        return RooArgSet(
            [one for one in self.getVariables() if one.GetName() not in exclude]
        ).sorted_copy()

    def getObservables(self, observables: Any = None, valueOnly: bool = True) -> RooArgSet:
        """The variables that are among ``observables``."""
        wanted = set(_observable_names(observables))
        return RooArgSet([one for one in self.leaves()
                          if one.GetName() in wanted and not one.InheritsFrom("RooConstVar")])

    def getComponents(self) -> RooArgSet:
        return RooArgSet([node for node in self._walk() if not node.isFundamental()])

    def dependsOn(self, other: Any, ignoreArg: Any = None, valueOnly: bool = False) -> bool:
        names = {one.GetName() for one in as_list(other)}
        return any(node.GetName() in names for node in self._walk() if node is not ignoreArg)

    dependsOnValue = dependsOn

    def dependents(self) -> frozenset[str]:
        """The names of every variable this node depends on."""
        cached = self.__dict__.get("_dependents_cache")
        if cached is None or cached[0] != _GRAPH[0]:
            cached = (_GRAPH[0], frozenset(one.GetName() for one in self.leaves()))
            self.__dict__["_dependents_cache"] = cached
        return cached[1]

    def findServer(self, name: Any) -> Any:
        wanted = name if isinstance(name, str) else name.GetName()
        return next((one for one in self.servers() if one.GetName() == wanted), None)

    # -- attributes ---------------------------------------------------------------

    def setAttribute(self, name: str, value: bool = True) -> None:
        if value:
            self._attributes.add(str(name))
        else:
            self._attributes.discard(str(name))

    def getAttribute(self, name: str) -> bool:
        return str(name) in self._attributes

    def attributes(self) -> set[str]:
        return set(self._attributes)

    def setStringAttribute(self, key: str, value: Any) -> None:
        if value is None:
            self._strings.pop(str(key), None)
        else:
            self._strings[str(key)] = str(value)

    def getStringAttribute(self, key: str) -> Any:
        return self._strings.get(str(key))

    def removeStringAttribute(self, key: str) -> None:
        self._strings.pop(str(key), None)

    def isConstant(self) -> bool:
        return self.getAttribute("Constant")

    # -- copies -------------------------------------------------------------------

    def clone(self, newname: Any = None) -> Any:
        """A copy of this node sharing its inputs, as ``RooAbsArg::clone`` makes one."""
        made = copy.copy(self)
        made._proxies = [Proxy(p.name, p.target, p.many, p.shape) for p in self._proxies]
        made._attributes = set(self._attributes)
        made._strings = dict(self._strings)
        made._copy_state(self)
        if newname:
            made._name = str(newname)
        return made

    def Clone(self, newname: Any = None) -> Any:
        return self.clone(newname)

    def _copy_state(self, other: Any) -> None:
        """Give a fresh copy its own copies of whatever state its class keeps."""

    def cloneTree(self, newname: Any = None) -> Any:
        """A deep copy of this node and everything under it, the top renamed if asked."""
        memo: dict[int, Any] = {}
        made = copy.deepcopy(self, memo)
        if newname:
            made._name = str(newname)
        return made

    # -- printing -----------------------------------------------------------------

    def printName(self) -> str:
        return self._name

    def printTitle(self) -> str:
        return self._title

    def printClassName(self) -> str:
        return self.ClassName()

    def printArgs(self) -> str:
        shown = [proxy.text() for proxy in self._proxies if not proxy.name.startswith("!")]
        meta = self.printMetaArgs()
        if not shown and not meta:
            return "" if not self._proxies else "[ " + meta + "]"
        return "[ " + "".join(one + " " for one in shown) + meta + "]"

    def printMetaArgs(self) -> str:
        return ""

    def defaultPrintContents(self, option: Any) -> int:
        return kName | kClassName | kValue | kArgs

    def printTree(self, indent: str) -> str:
        return self.compact_tree("", None)

    def compact_tree(self, indent: str, client: Any) -> str:
        """``printCompactTree``: this node and, indented, everything it is made of."""
        text = indent + address(self)
        if client is not None:
            text += "/V-"
        text += f" {self.ClassName()}::{self._name} = {self.printValue()}"
        if self._proxies:
            text += f" [Auto,{self.state_word()}] "
        text += "\n"
        for server in self.servers():
            text += server.compact_tree(indent + "  ", self)
        return text

    def state_word(self) -> str:
        """``Dirty`` or ``Clean``, as ROOT's value cache is when the tree is printed."""
        return "Dirty"

    def printCompactTree(self, indent: str = "", filename: Any = None, namePat: Any = None,
                         client: Any = None) -> None:  # fmt: skip
        
        cout.write(self.compact_tree(str(indent), None))

    def printComponentTree(self, indent: str = "", namePat: Any = None, nLevel: int = 999) -> None:
        if nLevel == 0 or self.isFundamental() or self.InheritsFrom("RooConstVar"):
            return
        if not namePat or str(namePat) in self._name:
            
            cout.write(str(indent))
            self.Print()
        for server in self.servers():
            server.printComponentTree(str(indent) + "  ", namePat, nLevel - 1)


def _observable_names(observables: Any) -> list[str]:
    """The names in ``observables``: a set of variables, one variable, or a dataset."""
    if observables is None:
        return []
    columns = getattr(observables, "get", None)
    if callable(columns) and hasattr(observables, "numEntries"):
        return [one.GetName() for one in columns()]
    return [one.GetName() for one in as_list(observables)]
