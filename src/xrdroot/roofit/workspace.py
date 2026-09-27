"""``RooWorkspace``: a named collection of a model's pieces, its datasets, and the factory.

``w.Import(model)`` takes in the model and every node it is made of, each
once by name - saying so for each, as RooFit does - and ``w["x"]``,
``w.pdf("model")``, ``w.var("x")``, ``w.data("d")`` hand them back.
``w.factory("Gaussian::g(x[-10,10],m[0],s[1])")`` builds pieces from
RooFit's factory language (:mod:`.factory`) and imports them.

The workspace holds the objects imported, not copies of them: a variable
changed outside is changed inside. ROOT's workspace holds clones, which
matters only to a macro that changes one and expects the other to stay put.
"""

from __future__ import annotations

from typing import Any

from .cmdargs import commands
from .collections import RooArgSet, as_list
from .messages import INFO, log
from .printing import RooPrintable
from . import cout

__all__ = ["RooWorkspace"]


class RooWorkspace(RooPrintable):
    """A workspace: nodes by name, datasets, named sets, snapshots."""

    def __init__(self, name: Any = "", title: Any = "", *args: Any) -> None:
        self._name = str(name)
        self._title = str(title) if isinstance(title, str) and title else str(name)
        self._nodes: dict[str, Any] = {}
        self._data: dict[str, Any] = {}
        self._sets: dict[str, RooArgSet] = {}
        self._snapshots: dict[str, list[tuple[str, float]]] = {}
        self._generic: dict[str, Any] = {}
        #: The factory's ``$Typedef`` names for classes.
        self._aliases: dict[str, str] = {}

    # -- TObject ------------------------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def ClassName(self) -> str:
        return "RooWorkspace"

    # -- importing ----------------------------------------------------------------

    def Import(self, obj: Any, *args: Any, **kwargs: Any) -> bool:
        """``import``: a node and all it is made of, a dataset, or each of a collection."""
        options = commands([a for a in args if not isinstance(a, str)], kwargs)
        if hasattr(obj, "numEntries"):
            return self._import_data(obj, options)
        if isinstance(obj, (list, tuple, set)) or hasattr(obj, "_list"):
            return all([self.Import(one, *args, **kwargs) for one in as_list(obj)])
        if not hasattr(obj, "servers"):
            self._generic[obj.GetName()] = obj
            return False
        return self._import_node(obj, options)

    def _import_node(self, top: Any, options: Any) -> bool:
        from .editing import rename_all

        suffix = options.get("RenameAllNodes")
        if suffix:
            rename_all(top, str(suffix), self._nodes)
        silent = bool(options.get("Silence", 0, False))
        for node in top._walk():
            if node.GetName() in self._nodes:
                continue
            self._nodes[node.GetName()] = node
            if not silent and not node.InheritsFrom("RooConstVar"):
                log(self, INFO, "ObjectHandling", f"RooWorkspace::import({self._name}) importing "
                    f"{node.ClassName()}::{node.GetName()}")  # fmt: skip
        return False

    def _import_data(self, data: Any, options: Any) -> bool:
        log(self, INFO, "ObjectHandling", f"RooWorkspace::import({self._name}) importing dataset "
            f"{data.GetName()}")  # fmt: skip
        new = options.get("Rename")
        if new:
            log(self, INFO, "ObjectHandling", f"RooWorkSpace::import({self._name}) changing name of "
                f"dataset from  {data.GetName()} to {new}")  # fmt: skip
            data.SetName(str(new))
        self._data[data.GetName()] = data
        for var in data.get():
            self._nodes.setdefault(var.GetName(), var)
        return False

    # -- the factory --------------------------------------------------------------

    def factory(self, spec: str) -> Any:
        from .factory import Factory

        return Factory(self).build(str(spec))

    # -- getting ------------------------------------------------------------------

    def arg(self, name: str) -> Any:
        return self._nodes.get(str(name))

    var = function = pdf = cat = catfunc = arg

    def data(self, name: str) -> Any:
        return self._data.get(str(name))

    embeddedData = data

    def obj(self, name: str) -> Any:
        return self._nodes.get(str(name)) or self._data.get(str(name)) or self._generic.get(str(name))

    genobj = obj

    def __getitem__(self, name: str) -> Any:
        found = self.obj(name)
        if found is None:
            raise KeyError(f"workspace {self._name} has no object called {name!r}")
        return found

    def _kind(self, cls: str) -> RooArgSet:
        return RooArgSet([n for n in self._nodes.values() if n.InheritsFrom(cls)])

    def allVars(self) -> RooArgSet:
        return self._kind("RooRealVar")

    def allCats(self) -> RooArgSet:
        return self._kind("RooCategory")

    def allPdfs(self) -> RooArgSet:
        return self._kind("RooAbsPdf")

    def allFunctions(self) -> RooArgSet:
        return RooArgSet([n for n in self._nodes.values() if _is_function(n)])

    def allData(self) -> list[Any]:
        return list(self._data.values())

    def components(self) -> RooArgSet:
        return RooArgSet(list(self._nodes.values()))

    # -- named sets and snapshots -------------------------------------------------

    def defineSet(self, name: str, content: Any, importMissing: bool = False) -> bool:
        items = [self._nodes.get(n) for n in str(content).split(",")] if isinstance(content, str) else (
            [self._nodes.get(one.GetName(), one) for one in as_list(content)])
        self._sets[str(name)] = RooArgSet([one for one in items if one is not None])
        return False

    def extendSet(self, name: str, names: str) -> bool:
        found = self._sets.setdefault(str(name), RooArgSet())
        for one in str(names).split(","):
            if one in self._nodes:
                found.add(self._nodes[one])
        return False

    def set(self, name: str) -> Any:
        return self._sets.get(str(name))

    def saveSnapshot(self, name: str, params: Any, importValues: bool = False) -> bool:
        items = self._resolve(params)
        self._snapshots[str(name)] = [(one.GetName(), one.getVal()) for one in items]
        return False

    def loadSnapshot(self, name: str) -> bool:
        for key, value in self._snapshots.get(str(name), []):
            self._nodes[key].setVal(value)
        return str(name) in self._snapshots

    def getSnapshot(self, name: str) -> Any:
        from .variables import RooRealVar

        return RooArgSet([RooRealVar(k, k, v) for k, v in self._snapshots.get(str(name), [])])

    def _resolve(self, params: Any) -> list[Any]:
        if isinstance(params, str):
            return [self._nodes[n] for n in params.split(",") if n in self._nodes]
        return as_list(params)

    # -- printing -----------------------------------------------------------------

    def Print(self, option: str = "") -> None:
        """``RooWorkspace::Print``: each kind of content under its heading, sorted by name."""
        cout.write(f"\nRooWorkspace({self._name}) {self._title} contents\n\n")
        nodes = list(self._nodes.values())
        variables = RooArgSet([n for n in nodes if n.InheritsFrom("RooRealVar") or n.InheritsFrom("RooCategory")])
        if len(variables):
            cout.write("variables\n---------\n" + variables.sorted_copy().printValue() + "\n\n")
        _section("p.d.f.s\n-------\n", [n for n in nodes if n.InheritsFrom("RooAbsPdf")])
        _section("functions\n--------\n", [n for n in nodes if _is_function(n)])
        if self._data:
            cout.write("datasets\n--------\n")
            for data in self._data.values():
                cout.write(f"{data.ClassName()}::{data.GetName()}{data.get().printValue()}\n")
            cout.write("\n")
        if self._sets:
            cout.write("named sets\n----------\n")
            for key in sorted(self._sets):
                cout.write(f"{key}:{self._sets[key].printValue()}\n")
            cout.write("\n")


def _is_function(node: Any) -> bool:
    kinds = ("RooAbsPdf", "RooConstVar", "RooRealVar", "RooAbsCategory")
    return node.InheritsFrom("RooAbsReal") and not any(node.InheritsFrom(k) for k in kinds)


def _section(heading: str, nodes: list[Any]) -> None:
    if not nodes:
        return
    cout.write(heading)
    for node in sorted(nodes, key=lambda n: n.GetName()):
        node.Print()
    cout.write("\n")
