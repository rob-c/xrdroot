"""``RooDataSet``: unbinned events, made empty, imported from a tree or another dataset, or
generated.

``RooDataSet("d", "d", {x, y}, Import(tree), Cut("y>0"), WeightVar("w"))``
takes the tree's columns for the variables, drops the events outside a
variable's range - saying so as ``RooTreeDataStore::loadValues`` does -
then those the cut rejects. A dataset grows by ``add`` - an event of the
variables' current values - and by ``append``, and ``reduce`` makes a
smaller one: fewer variables, fewer events.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cmdargs import Commands, commands
from ..collections import RooArgSet, as_list
from ..messages import INFO, WARNING, log
from ..printing import g
from .store import RooAbsData, copies_of

__all__ = ["RooDataSet"]


def _column_of(source: Any, name: str) -> np.ndarray[Any, Any]:
    """A tree's or a dataset's column ``name``, as floats."""
    if isinstance(source, RooAbsData):
        return np.asarray(source.column(name), dtype=np.float64)
    tree = getattr(source, "_xrd", source)
    return np.asarray(tree.arrays([name])[name], dtype=np.float64)


class RooDataSet(RooAbsData):
    """Unbinned events, each with a value of every variable and, optionally, a weight."""

    def __init__(self, name: Any = "", title: Any = "", *args: Any, **kwargs: Any) -> None:
        variables, rest = _split(args)
        options = commands(rest, kwargs)
        weight = _weight_name(options)
        index = options.get("Index")
        chosen = as_list(variables) + (
            [index]
            if index is not None
            and index.GetName() not in [v.GetName() for v in as_list(variables)]
            else []
        )
        super().__init__(name, title, [v for v in chosen if v.GetName() != weight])
        self._weight_var = None
        if weight:
            self._weight_var = _weight_variable(chosen, weight)
            self._weights = np.zeros(0)
        if "GlobalObservables" in options:
            self.setGlobalObservables(options.get("GlobalObservables"))
        self._fill(options, weight)

    def _fill(self, options: Commands, weight: str) -> None:
        source = options.get("Import")
        if isinstance(source, dict) or isinstance(source, str) or options.every("Import")[1:]:
            from .slices import import_slices

            import_slices(self, options)
        elif source is not None:
            self._import(source, weight)
        cut, rng = options.get("Cut"), options.get("CutRange")
        if cut is not None or rng is not None:
            self._keep(self.mask(cut, rng))

    # -- filling ------------------------------------------------------------------

    def _import(self, source: Any, weight: str) -> None:
        """Take in a tree's or a dataset's events, skipping those a variable cannot hold."""
        n = (
            source.numEntries()
            if isinstance(source, RooAbsData)
            else len(getattr(source, "_xrd", source))
        )
        columns = {one.GetName(): _column_of(source, one.GetName()) for one in self._vars}
        keep = np.ones(n, dtype=bool)
        if not isinstance(source, RooAbsData):
            keep = self._accommodated(columns, n)
        self._columns = {k: v[keep] for k, v in columns.items()}
        if weight:
            wsource = (
                source.weights()
                if isinstance(source, RooAbsData) and source.isWeighted()
                else (_column_of(source, weight))
            )
            self._weights = np.asarray(wsource, dtype=np.float64)[keep]
        elif isinstance(source, RooAbsData) and source.isWeighted():
            self._weights = source.weights()[keep]

    def _accommodated(self, columns: dict[str, Any], n: int) -> np.ndarray[Any, Any]:
        """``RooTreeDataStore::loadValues``: which events every variable can hold, and a word for
        the rest."""
        keep = np.ones(n, dtype=bool)
        invalid = 0
        for i in range(n):
            bad = next(
                (one for one in self._vars if not one.can_hold(columns[one.GetName()][i])), None
            )
            if bad is None:
                continue
            keep[i] = False
            invalid += 1
            if invalid < 5:
                log(
                    self,
                    INFO,
                    "DataHandling",
                    f"RooTreeDataStore::loadValues({self._name}) Skipping "
                    f"event #{i} because {bad.GetName()} cannot accommodate the value "
                    f"{bad.value_text(columns[bad.GetName()][i])}",
                )
            elif invalid == 5:
                log(
                    self,
                    INFO,
                    "DataHandling",
                    f"RooTreeDataStore::loadValues({self._name}) Skipping ...",
                )
        if invalid:
            log(
                self,
                WARNING,
                "DataHandling",
                f"RooTreeDataStore::loadValues({self._name}) Ignored {invalid} out-of-range events",
            )
        return keep

    def _keep(self, keep: np.ndarray[Any, Any]) -> None:
        self._columns = {k: v[keep] for k, v in self._columns.items()}
        if self._weights is not None:
            self._weights = self._weights[keep]

    def add(self, row: Any, weight: float = 1.0, weightError: float = 0.0) -> None:
        """One more event: the values ``row`` holds now, weighing ``weight`` if weights are kept."""
        for one in self._vars:
            found = row.find(one.GetName()) if hasattr(row, "find") else None
            value = (found if found is not None else one).stored_value()
            self._columns[one.GetName()] = np.append(self._columns[one.GetName()], value)
        if self._weight_var is not None:
            self._weights = np.append(self.weights()[: self.numEntries() - 1], float(weight))
        elif weight != 1.0 or weightError != 0.0:
            log(
                self,
                4,
                "InputArguments",
                f"RooDataSet::add(dataset={self._name}) WARNING: You are "
                "adding a weight but no weight variable exists. The weight will be ignored.",
            )

    def add_columns(self, columns: dict[str, Any], weights: Any = None) -> None:
        """Many events at once, a column for each variable, and their weights if weights are
        kept."""
        before = self.numEntries()
        for one in self._vars:
            self._columns[one.GetName()] = np.concatenate(
                [self._columns[one.GetName()], np.asarray(columns[one.GetName()], dtype=np.float64)]
            )
        if self._weights is not None or weights is not None:
            added = self.numEntries() - before
            extra = np.ones(added) if weights is None else np.asarray(weights, dtype=np.float64)
            self._weights = np.concatenate([self.weights()[:before], extra])

    def append(self, other: RooDataSet) -> None:
        self.add_columns(
            {one.GetName(): other.column(one.GetName()) for one in self._vars},
            other.weights() if self.isWeighted() else None,
        )

    def reduce(self, *args: Any, **kwargs: Any) -> Any:
        """A new dataset of fewer variables (``SelectVars``), events (``Cut``, ``CutRange``,
        ``EventRange``)."""
        from .reducing import reduced

        return reduced(self, args, kwargs)

    def binnedClone(self, name: Any = None, title: Any = None) -> Any:
        """``binnedClone``: these events binned in their variables' binnings."""
        from .datahist import RooDataHist

        made = RooDataHist(
            name or f"{self._name}_binned", title or f"{self._title}_binned", list(self._vars), self
        )
        return made

    # -- printing -----------------------------------------------------------------

    def printArgs(self) -> str:
        text = "[" + ",".join(one.GetName() for one in self._vars)
        if self._weight_var is not None:
            text += f",weight:{self._weight_var.GetName()}"
        return text + "]"

    def printValue(self) -> str:
        text = f"{self.numEntries()} entries"
        if self.isWeighted():
            text += f" ({g(self.sumEntries())} weighted)"
        return text

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        text = super().printMultiline(contents, verbose, indent)
        if self._weight_var is not None:
            text += (
                f'{indent}  Dataset variable "{self._weight_var.GetName()}" is interpreted as '
                "the event weight\n"
            )
        return text


def _split(args: tuple[Any, ...]) -> tuple[Any, list[Any]]:
    """The variables a constructor was given, and its options."""
    from ..cmdargs import RooCmdArg

    variables: list[Any] = []
    rest: list[Any] = []
    for arg in args:
        (rest if isinstance(arg, RooCmdArg) else variables).append(arg)
    return (variables[0] if len(variables) == 1 else variables), rest


def _weight_name(options: Commands) -> str:
    found = options.get("WeightVar")
    if found is None:
        return ""
    return found if isinstance(found, str) else found.GetName()


def _weight_variable(chosen: list[Any], name: str) -> Any:
    from ..variables import RooRealVar

    found = next((v for v in chosen if v.GetName() == name), None)
    return found.clone(name) if found is not None else RooRealVar(name, name, 1.0)


def as_set(items: Any) -> RooArgSet:
    return copies_of(as_list(items))
