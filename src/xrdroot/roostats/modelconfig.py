"""``RooStats::ModelConfig``: which of a model's variables are what, kept in its workspace.

A ModelConfig holds names, not objects: the density's name and the names of
named sets it defines in its workspace - ``ModelConfig_POI``,
``ModelConfig_NuisParams``, ``ModelConfig_Observables``... - so that it
survives being written and read with the workspace, and every getter looks
its answer up there. Setting a density or a set imports into the
workspace whatever it lacks, with RooFit's messages held back below errors,
as RooStats holds them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..roofit import cout
from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, INFO, log, log_plain, service

__all__ = ["ModelConfig", "quieted"]

#: Each set a ModelConfig names, and the suffix its workspace set takes.
SETS = {
    "POI": "_POI",
    "NuisParams": "_NuisParams",
    "ConstrainedParams": "_ConstrainedParams",
    "Observables": "_Observables",
    "ConditionalObservables": "_ConditionalObservables",
    "GlobalObservables": "_GlobalObservables",
    "ExternalConstraints": "_ExternalConstraints",
}


@contextmanager
def quieted(level: int = ERROR) -> Iterator[None]:
    """``setGlobalKillBelow(level)`` for the block, and back as it was."""
    before = service().globalKillBelow()
    service().setGlobalKillBelow(level)
    try:
        yield
    finally:
        service().setGlobalKillBelow(before)


class ModelConfig:
    """A model's configuration: its density and its sets of variables, by name in a workspace."""

    def __init__(self, name: Any = None, title: Any = None, ws: Any = None) -> None:
        if name is not None and not isinstance(name, str):  # ModelConfig(ws)
            name, ws = None, name
        if title is not None and not isinstance(title, str):  # ModelConfig(name, ws)
            title, ws = None, title
        self._name = str(name) if name is not None else ""
        self._title = str(title) if title is not None else self._name
        self._ws: Any = None
        self._names: dict[str, str] = dict.fromkeys(
            ("Pdf", "PriorPdf", "ProtoData", "Snapshot", *SETS), ""
        )
        if ws is not None:
            self.SetWS(ws)

    # -- TObject ------------------------------------------------------------------

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def SetTitle(self, title: str) -> None:
        self._title = str(title)

    def ClassName(self) -> str:
        return "RooStats::ModelConfig"

    def InheritsFrom(self, name: Any) -> bool:
        return str(name) in ("RooStats::ModelConfig", "ModelConfig", "TNamed", "TObject")

    def Clone(self, name: Any = "") -> ModelConfig:
        made = ModelConfig(self._name, self._title)
        made._ws, made._names = self._ws, dict(self._names)
        if name:
            made.SetName(str(name))
        return made

    # -- the workspace ------------------------------------------------------------

    def SetWS(self, ws: Any) -> None:
        if self._ws is None:
            self._ws = ws
            return
        log(self, ERROR, "ObjectHandling", "ModelConfig::SetWS(): workspace already set, "
            "not doing anything")  # fmt: skip

    SetWorkspace = SetWS

    def ReplaceWS(self, ws: Any) -> None:
        self._ws = ws

    def GetWS(self) -> Any:
        if self._ws is None:
            log(self, ERROR, "ObjectHandling", "workspace not set")
        return self._ws

    GetWorkspace = GetWS

    # -- setting ------------------------------------------------------------------

    def _define(self, key: str, items: Any, what: str, check: bool = True) -> None:
        """Name a set of the workspace's for ``key`` - ``<name>_POI`` and so on - once each
        member is known to be a variable (``check``); a global observable becomes constant."""
        if isinstance(items, str):  # a list of the workspace's names
            ws = self.GetWS()
            if ws is None:
                return
            items = ws.argSet(items)
        members = RooArgSet(as_list(items))
        if check and not _only_parameters(members, f"ModelConfig::{what}"):
            return
        if key == "GlobalObservables":  # a fit never floats one
            for one in members:
                one.setAttribute("Constant", True)
        ws = self.GetWS()
        if ws is None:
            return
        self._names[key] = self._name + SETS[key]
        _define_in_ws(ws, self._names[key], members)

    def SetParametersOfInterest(self, items: Any) -> None:
        """By names, ``SetParameters``' own: its message names that instead."""
        what = "SetParameters" if isinstance(items, str) else "SetParametersOfInterest"
        self._define("POI", items, what)

    def SetParameters(self, items: Any) -> None:
        self._define("POI", items, "SetParameters")

    def SetNuisanceParameters(self, items: Any) -> None:
        self._define("NuisParams", items, "SetNuisanceParameters")

    def SetConstraintParameters(self, items: Any) -> None:
        self._define("ConstrainedParams", items, "SetConstrainedParameters")

    def SetObservables(self, items: Any) -> None:
        self._define("Observables", items, "SetObservables")

    def SetConditionalObservables(self, items: Any) -> None:
        self._define("ConditionalObservables", items, "SetConditionalObservables")

    def SetExternalConstraints(self, items: Any) -> None:
        """The constraint densities - not variables, so not checked as the other sets are."""
        self._define("ExternalConstraints", items, "SetExternalConstraints", check=False)

    def SetGlobalObservables(self, items: Any) -> None:
        """The global observables, which become constant: a fit never floats one."""
        self._define("GlobalObservables", items, "SetGlobalObservables")

    def SetPdf(self, pdf: Any) -> None:
        self._named_in_ws("Pdf", pdf, "pdf")

    def SetPriorPdf(self, pdf: Any) -> None:
        self._named_in_ws("PriorPdf", pdf, "pdf")

    def SetProtoData(self, data: Any) -> None:
        self._named_in_ws("ProtoData", data, "data")

    def _named_in_ws(self, key: str, obj: Any, kind: str) -> None:
        """Import ``obj`` if the workspace lacks one of its name, and name it for ``key``."""
        ws = self.GetWS()
        if ws is None:
            return
        name = obj if isinstance(obj, str) else obj.GetName()
        if not isinstance(obj, str) and getattr(ws, kind)(name) is None:
            with quieted():
                ws.Import(obj)
        if getattr(ws, kind)(name) is None:
            text = f"{'pdf' if kind == 'pdf' else 'dataset'} {name} does not exist in workspace"
            log(self, ERROR, "ObjectHandling", text)
            raise RuntimeError(text)
        self._names[key] = name

    def SetSnapshot(self, items: Any) -> None:
        """Save the values of ``items`` as the workspace's snapshot ``<name>_<set>_snapshot``."""
        ws = self.GetWS()
        if ws is None:
            return
        members = items if isinstance(items, RooArgSet) else RooArgSet(as_list(items))
        name = self._name + ("_" if self._name else "") + members.GetName()
        self._names["Snapshot"] = name + ("_" if name else "") + "snapshot"  # ModelConfig__snapshot
        ws.saveSnapshot(self._names["Snapshot"], members, True)
        _define_in_ws(ws, self._names["Snapshot"], members)

    # -- getting ------------------------------------------------------------------

    def _set(self, key: str) -> Any:
        ws = self.GetWS()
        return ws.set(self._names[key]) if ws is not None and self._names[key] else None

    def GetParametersOfInterest(self) -> Any:
        return self._set("POI")

    def GetNuisanceParameters(self) -> Any:
        return self._set("NuisParams")

    def GetConstraintParameters(self) -> Any:
        return self._set("ConstrainedParams")

    def GetObservables(self) -> Any:
        return self._set("Observables")

    def GetConditionalObservables(self) -> Any:
        return self._set("ConditionalObservables")

    def GetGlobalObservables(self) -> Any:
        return self._set("GlobalObservables")

    def GetExternalConstraints(self) -> Any:
        return self._set("ExternalConstraints")

    def GetPdf(self) -> Any:
        ws = self.GetWS()
        return ws.pdf(self._names["Pdf"]) if ws is not None and self._names["Pdf"] else None

    def GetPriorPdf(self) -> Any:
        ws = self.GetWS()
        found = self._names["PriorPdf"]
        return ws.pdf(found) if ws is not None and found else None

    def GetProtoData(self) -> Any:
        ws = self.GetWS()
        found = self._names["ProtoData"]
        return ws.data(found) if ws is not None and found else None

    def GetSnapshot(self) -> Any:
        """A copy of the snapshot's values - the workspace's variables left as they were."""
        ws = self.GetWS()
        saved = self._set("Snapshot")
        if saved is None or not len(saved):  # an empty snapshot is none, as in RooStats
            return None
        now = saved.snapshot()
        ws.loadSnapshot(self._names["Snapshot"])
        found = saved.snapshot()
        saved.assign(now)
        return found

    def LoadSnapshot(self) -> None:
        ws = self.GetWS()
        if ws is not None:
            ws.loadSnapshot(self._names["Snapshot"])

    def GuessObsAndNuisance(self, data: Any, printModelConfig: bool = True) -> None:
        """Observables from the data, global observables and nuisances from what is left."""
        observed = data.get() if hasattr(data, "numEntries") else data
        pdf = self.GetPdf()
        if self.GetObservables() is None:
            self.SetObservables(pdf.getObservables(observed))
        if self.GetGlobalObservables() is None:
            self._guess_globals(pdf, observed)
        if self.GetNuisanceParameters() is None:
            self._guess_nuisance(pdf, observed)
        if printModelConfig:  # to the INFO stream, as RooPrintable's default stream is made to be
            log_plain(self, INFO, "InputArguments", self._text())

    def _guess_globals(self, pdf: Any, observed: Any) -> None:
        """The observables not in the data that are not constant: global observables."""
        seen = {one.GetName() for one in pdf.getObservables(observed)}
        rest = [one for one in self.GetObservables() if one.GetName() not in seen]
        rest = [one for one in rest if not one.isConstant()]
        if rest:
            self.SetGlobalObservables(rest)

    def _guess_nuisance(self, pdf: Any, observed: Any) -> None:
        """The free parameters that are not of interest: the nuisance parameters."""
        poi = {one.GetName() for one in self.GetParametersOfInterest() or ()}
        params = [p for p in pdf.getParameters(observed) if p.GetName() not in poi]
        params = [p for p in params if not p.isConstant()]
        if params:
            self.SetNuisanceParameters(params)

    def Print(self, option: str = "") -> None:
        """``ModelConfig::Print``: each set and density it names, in RooStats' order."""
        cout.write(self._text())

    def _text(self) -> str:
        text = f"\n=== Using the following for {self._name} ===\n"
        for label, get in (
            ("Observables:             ", self.GetObservables),
            ("Parameters of Interest:  ", self.GetParametersOfInterest),
            ("Nuisance Parameters:     ", self.GetNuisanceParameters),
            ("Global Observables:      ", self.GetGlobalObservables),
            ("Constraint Parameters:   ", self.GetConstraintParameters),
            ("Conditional Observables: ", self.GetConditionalObservables),
            ("Proto Data:              ", self.GetProtoData),
            ("PDF:                     ", self.GetPdf),
            ("Prior PDF:               ", self.GetPriorPdf),
        ):
            found = get()
            if found is not None:
                text += label + found.printStream(found.defaultPrintContents(""),
                                                  found.defaultPrintStyle(""))  # fmt: skip
        snapshot = self.GetSnapshot()
        if snapshot is not None:
            text += "Snapshot:                \n"
            text += snapshot.printStream(snapshot.defaultPrintContents("v"),
                                         snapshot.defaultPrintStyle("v"))  # fmt: skip
        return text + "\n"

    # -- fitting ------------------------------------------------------------------

    def _options(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """The options ``createNLL`` and ``fitTo`` are given, and the ModelConfig's own."""
        from ..roofit.cmdargs import RooCmdArg

        extra = []
        for key, command in (
            ("ConditionalObservables", "ConditionalObservables"),
            ("GlobalObservables", "GlobalObservables"),
            ("ExternalConstraints", "ExternalConstraints"),
        ):
            found = self._set(key)
            if found is not None:
                extra.append(RooCmdArg(command, found))
        return (*args, *extra)

    def createNLL(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        return self.GetPdf().createNLL(data, *self._options(args), **kwargs)

    def fitTo(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        return self.GetPdf().fitTo(data, *self._options(args), **kwargs)


def _define_in_ws(ws: Any, name: str, members: Any) -> None:
    """``DefineSetInWS``: the named set ``name`` of the workspace's, replacing any it had."""
    if ws.set(name) is not None:
        ws.removeSet(name)
    with quieted():
        ws.defineSet(name, members, True)


def _only_parameters(members: Any, prefix: str) -> bool:
    """``SetHasOnlyParameters``: whether every member is a variable, saying which are not."""
    others = [one for one in members if not one.isFundamental()]
    if others:
        cout.line(f"{prefix} ERROR: specified set contains non-parameters: "
                  f"{RooArgSet(others).printValue()}")  # fmt: skip
    return not others
