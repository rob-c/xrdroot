"""``RooWorkspace``: a named collection of a model's pieces, its datasets, and the factory.

``w.Import(model)`` takes in the model and every node it is made of, each
once by name - saying so for each, as RooFit does - and ``w["x"]``,
``w.pdf("model")``, ``w.var("x")``, ``w.data("d")`` hand them back.
``w.factory("Gaussian::g(x[-10,10],m[0],s[1])")`` builds pieces from
RooFit's factory language (:mod:`.factory`) and imports them.

The workspace holds clones, as ROOT's does: a variable changed outside
after the import is not changed inside - rf510 fixes a fraction of its model
after importing it, and fits the workspace's copy free. A node the
workspace already has by name is not copied again: the clone uses it.
"""

from __future__ import annotations

from typing import Any

from . import cout
from .cmdargs import commands
from .collections import RooArgSet, as_list
from .messages import INFO, log
from .printing import RooPrintable, g

__all__ = ["RooWorkspace"]


class RooWorkspace(RooPrintable):
    """A workspace: nodes by name, datasets, named sets, snapshots."""

    def __init__(self, name: Any = "", title: Any = "", *args: Any) -> None:
        self._name = str(name)
        self._title = str(title) if isinstance(title, str) and title else str(name)
        self._nodes: dict[str, Any] = {}
        self._data: dict[str, Any] = {}
        self._sets: dict[str, RooArgSet] = {}
        self._snapshots: dict[str, list[Any]] = {}
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
            return all([self.Import(one, *args, **kwargs) for one in as_list(obj)])  # noqa: C419
        if not hasattr(obj, "servers"):
            return self._import_generic(obj)
        return self._import_node(obj, options)

    def _import_generic(self, obj: Any) -> bool:
        """``import(TObject&)``: a copy of anything else - a ModelConfig, a histogram - by name."""
        clone = getattr(obj, "Clone", None)
        self._generic[obj.GetName()] = clone() if callable(clone) else obj
        return False

    def _import_node(self, top: Any, options: Any) -> bool:
        from .editing import rename_all

        suffix = options.get("RenameAllNodes")
        top = self._cloned(top, renaming=bool(suffix))
        if suffix:
            rename_all(top, str(suffix), self._nodes)
        silent = bool(options.get("Silence", 0, False))
        for node in top._walk():
            if node.GetName() in self._nodes:
                continue
            self._nodes[node.GetName()] = node
            if not silent and not node.InheritsFrom("RooConstVar"):
                log(
                    self,
                    INFO,
                    "ObjectHandling",
                    f"RooWorkspace::import({self._name}) importing "
                    f"{node.ClassName()}::{node.GetName()}",
                )
        return False

    def _cloned(self, top: Any, renaming: bool) -> Any:
        """``top`` and what it is made of, copied - but for the nodes the workspace has by name
        already (only the variables, when the rest is being renamed), which the copy uses."""
        import copy

        kept = {
            id(node): self._nodes[node.GetName()]
            for node in top._walk()
            if node.GetName() in self._nodes and (node.isFundamental() or not renaming)
        }
        return copy.deepcopy(top, kept)

    def _import_data(self, data: Any, options: Any) -> bool:
        log(
            self,
            INFO,
            "ObjectHandling",
            f"RooWorkspace::import({self._name}) importing dataset {data.GetName()}",
        )
        new = options.get("Rename")
        if new:
            log(
                self,
                INFO,
                "ObjectHandling",
                f"RooWorkSpace::import({self._name}) changing name of "
                f"dataset from  {data.GetName()} to {new}",
            )
            data = _renamed(data, str(new))
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
        return (
            self._nodes.get(str(name)) or self._data.get(str(name)) or self._generic.get(str(name))
        )

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
        """A named set of the workspace's nodes - those named in ``content`` - importing any it
        lacks when ``importMissing`` says to."""
        if not isinstance(content, str) and importMissing:
            for one in as_list(content):
                if one.GetName() not in self._nodes:
                    self.Import(one)
        items = (
            [self._nodes.get(n) for n in str(content).split(",")]
            if isinstance(content, str)
            else ([self._nodes.get(one.GetName(), one) for one in as_list(content)])
        )
        self._sets[str(name)] = RooArgSet([one for one in items if one is not None])
        return False

    def removeSet(self, name: str) -> bool:
        """Forget the named set ``name``; ``True`` if there was none."""
        return self._sets.pop(str(name), None) is None

    def argSet(self, names: str) -> RooArgSet:
        """The workspace's nodes named in the comma-separated ``names``."""
        return RooArgSet([self._nodes[n] for n in str(names).split(",") if n in self._nodes])

    def extendSet(self, name: str, names: str) -> bool:
        found = self._sets.setdefault(str(name), RooArgSet())
        for one in str(names).split(","):
            if one in self._nodes:
                found.add(self._nodes[one])
        return False

    def set(self, name: str) -> Any:
        return self._sets.get(str(name))

    def saveSnapshot(self, name: str, params: Any, importValues: bool = False) -> bool:
        """Copies of the parameters - value, error and whether constant - in the workspace's
        order of its nodes.

        The copies are of the workspace's own nodes; with ``importValues`` they
        then take the values of the ``params`` given, as ROOT's ``assign``
        does - how ``rf510_wsnamedsets.C`` saves the fit of the model it
        imported, whose own parameters the workspace holds copies of.
        """
        given = {one.GetName(): one for one in self._resolve(params)}
        chosen = [one for key, one in self._nodes.items() if key in given]
        copies = [one.clone(one.GetName()) for one in chosen]
        for copy in copies if importValues else ():
            theirs = given[copy.GetName()]
            copy.copy_value_from(theirs)
            copy.setConstant(theirs.isConstant())
        self._snapshots[str(name)] = copies
        return False

    def loadSnapshot(self, name: str) -> bool:
        for saved in self._snapshots.get(str(name), []):
            node = self._nodes[saved.GetName()]
            node.copy_value_from(saved)
            node.setConstant(saved.isConstant())
        return str(name) in self._snapshots

    def getSnapshot(self, name: str) -> Any:
        return RooArgSet(list(self._snapshots.get(str(name), [])))

    def _resolve(self, params: Any) -> list[Any]:
        if isinstance(params, str):
            return [self._nodes[n] for n in params.split(",") if n in self._nodes]
        return as_list(params)

    # -- printing -----------------------------------------------------------------

    def _snapshot_lines(self) -> list[str]:
        """``reference_fit = (a0=0.488363 +/- 0.0241765,sigma1=0.5[C])``: each snapshot's line."""
        return [
            f"{name} = ({','.join(_snapshot_value(one) for one in saved)})"
            for name, saved in self._snapshots.items()
        ]

    def Print(self, option: str = "") -> None:
        """``RooWorkspace::Print``: each kind of content under its heading, sorted by name."""
        cout.write(f"\nRooWorkspace({self._name}) {self._title} contents\n\n")
        nodes = list(self._nodes.values())
        _variables_section(nodes)
        _section("p.d.f.s\n-------\n", [n for n in nodes if n.InheritsFrom("RooAbsPdf")])
        _section("functions\n--------\n", [n for n in nodes if _is_function(n)])
        _lines(
            "datasets\n--------\n",
            [f"{d.ClassName()}::{d.GetName()}{d.get().printValue()}" for d in self._data.values()],
        )
        _lines("parameter snapshots\n-------------------\n", self._snapshot_lines())
        _lines(
            "named sets\n----------\n",
            [f"{k}:{self._sets[k].printValue()}" for k in sorted(self._sets)],
        )
        _lines(
            "generic objects\n---------------\n",
            [f"{o.ClassName()}::{o.GetName()}" for o in self._generic.values()],
        )


def _is_function(node: Any) -> bool:
    kinds = ("RooAbsPdf", "RooConstVar", "RooRealVar", "RooAbsCategory")
    return node.InheritsFrom("RooAbsReal") and not any(node.InheritsFrom(k) for k in kinds)


def _snapshot_value(var: Any) -> str:
    """``a0=0.488363 +/- 0.0241765``, ``sigma1=0.5[C]``: a saved parameter as ROOT lists it."""
    if var.isConstant():
        return f"{var.GetName()}={g(var.getVal())}[C]"
    if var.hasError():
        return f"{var.GetName()}={g(var.getVal())} +/- {g(var.getError())}"
    return f"{var.GetName()}={g(var.getVal())}"


def _renamed(data: Any, name: str) -> Any:
    """A copy of ``data`` called ``name``: the caller's own dataset keeps its name."""
    import copy

    made = copy.copy(data)
    made.SetName(name)
    return made


def _variables_section(nodes: list[Any]) -> None:
    """The variables and categories, as one set's value - or nothing, if there are none."""
    kinds = ("RooRealVar", "RooCategory")
    variables = RooArgSet([n for n in nodes if any(n.InheritsFrom(k) for k in kinds)])
    if len(variables):
        cout.write("variables\n---------\n" + variables.sorted_copy().printValue() + "\n\n")


def _lines(heading: str, lines: list[str]) -> None:
    """A heading and its lines, and a blank line - or nothing, for no lines."""
    if lines:
        cout.write(heading + "".join(line + "\n" for line in lines) + "\n")


def _section(heading: str, nodes: list[Any]) -> None:
    if not nodes:
        return
    cout.write(heading)
    for node in sorted(nodes, key=lambda n: n.GetName()):
        node.Print()
    cout.write("\n")


#: ``w->import(...)``: C++ spells ``Import`` as ROOT does; Python cannot name a method ``import``.
setattr(RooWorkspace, "import", RooWorkspace.Import)
