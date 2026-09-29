"""The channels combined: ``simPdf`` over ``channelCat``, the combined data, the measurement.

``MakeCombinedModel`` puts each channel's model in a ``RooSimultaneous``
over a category of the channels, their datasets in one indexed by it and
their global observables in one set; ``ConfigureWorkspaceForMeasurement``
sets the parameters of interest, guesses the rest, saves the nominal
values and adds the measurement's Asimov datasets.
"""

from __future__ import annotations

from typing import Any

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgList, RooArgSet
from ..roofit.messages import INFO, PROGRESS, WARNING
from .assemble import _hf_info_active, log_fatal
from .model import _hf

__all__ = ["combined_model", "configure_for_measurement"]


def _consistent(workspaces: list[Any], names: list[str]) -> None:
    """``isChannelDataConsistent``: every channel has the same datasets, but ``asimovData``."""
    reference = {d.GetName() for d in workspaces[0].allData()} - {"asimovData"}
    for i, ws in enumerate(workspaces[1:], 1):
        mine = {d.GetName() for d in ws.allData()} - {"asimovData"}
        missing, extra = sorted(reference - mine), sorted(mine - reference)
        if missing or extra:
            text = ("ERROR: Inconsistent datasets across channel workspaces.\nWorkspace for "
                    f'channel "{names[i]}" does not match the datasets in channel "{names[0]}".\n')
            text += "".join(["  Missing datasets:\n", *(f"    - {n}\n" for n in missing)]
                            if missing else [])  # fmt: skip
            text += "".join(["  Extra datasets:\n", *(f"    - {n}\n" for n in extra)]
                            if extra else [])  # fmt: skip
            log_fatal(text + "All channel workspaces must contain exactly the same datasets.\n")


def combined_model(factory: Any, names: list[str], workspaces: list[Any]) -> Any:
    """``MakeCombinedModel``: the workspace ``combined``."""
    from ..roofit.categories import RooCategory
    from ..roofit.data.dataset import RooDataSet
    from ..roofit.pdfs.simultaneous import RooSimultaneous
    from ..roofit.workspace import RooWorkspace
    from ..roostats import asimov
    from ..roostats.modelconfig import ModelConfig
    from .terms import emplace

    observables = RooArgList()
    for ws in workspaces:
        observables.add(list(ws.obj("ModelConfig").GetObservables()))
    listed = ",".join(one.GetName() for one in observables)
    _hf(INFO, f"full list of observables:\n({listed})")
    glob = RooArgSet()
    pdfs = {}
    for i, (name, ws) in enumerate(zip(names, workspaces)):
        if i == 0 and name[:1].isdigit():
            raise ValueError(f"The first channel name for HistFactory cannot start with a digit. "
                             f"Got {name}")  # fmt: skip
        pdfs[name] = ws.pdf(f"model_{name}")
        glob.add(list(ws.obj("ModelConfig").GetGlobalObservables()), True)
    _hf(PROGRESS, "\n-----------------------------------------\n\tEntering combination\n"
        "-----------------------------------------\n")  # fmt: skip
    combined = RooWorkspace("combined")
    category = emplace(combined, RooCategory, "channelCat", {n: i for i, n in enumerate(names)})
    sim = RooSimultaneous("simPdf", "", dict(sorted(pdfs.items())), category)
    config = ModelConfig("ModelConfig", combined)
    combined.Import(glob)
    combined.defineSet("globalObservables", glob)
    config.SetGlobalObservables(combined.set("globalObservables"))
    combined.defineSet("observables", [*observables, category], True)
    config.SetObservables(combined.set("observables"))
    _consistent(workspaces, names)
    for data in workspaces[0].allData():
        if data.GetName() == "asimovData":
            continue
        parts = {n: ws.data(data.GetName()) for n, ws in zip(names, workspaces)}
        combined.Import(RooDataSet(data.GetName(), "", list(observables),
                                   RooCmdArg("Index", category),
                                   RooCmdArg("WeightVar", "weightVar"),
                                   RooCmdArg("Import", parts)))  # fmt: skip
    if _hf_info_active():
        combined.Print()
    _hf(PROGRESS, "\n-----------------------------------------\n\tImporting combined model\n"
        "-----------------------------------------\n")  # fmt: skip
    combined.Import(sim, RooCmdArg("RecycleConflictNodes"))
    factory._set_values_and_constants(combined)
    config.SetPdf(combined.pdf("simPdf"))
    combined.Import(config, config.GetName())
    _hf(PROGRESS, "\n-----------------------------------------\n\tcreate toy data\n"
        "-----------------------------------------\n")  # fmt: skip
    made = asimov.GenerateAsimovData(combined.pdf("simPdf"), observables)
    if made is None:
        log_fatal("Error: Failed to create combined asimov dataset")
    combined.Import(made, RooCmdArg("Rename", "asimovData"))
    return combined


def configure_for_measurement(model_name: str, ws: Any, measurement: Any) -> None:
    """``ConfigureWorkspaceForMeasurement``: the parameters of interest set in ``ModelConfig``,
    the rest guessed, the nominal values saved, the measurement's Asimov datasets made."""
    from ..roofit.messages import TOPICS, service
    from ..roostats import asimov

    config = ws.obj("ModelConfig")
    if config is None:
        log_fatal(f"Error: Did not find 'ModelConfig' object in file: {ws.GetName()}")
    pois = measurement.GetPOIList()
    if not pois:
        _hf(WARNING, "No Parametetrs of interest are set")
    _hf(INFO, "Setting Parameter(s) of Interest as: " + "".join(f"{p} " for p in pois))
    params = []
    for name in pois:
        found = ws.var(name)
        if found is None:
            _hf(WARNING, f"WARNING: Can't find parameter of interest: {name} in Workspace. Not "
                "setting in ModelConfig.")  # fmt: skip
        else:
            params.append(found)
    config.SetParametersOfInterest(RooArgSet(params))
    pdf = ws.pdf("newSimPdf") or ws.pdf(model_name)
    observables = ws.set("observables")
    if pois:
        show = service().isActive(None, TOPICS["HistFactory"], INFO)
        config.GuessObsAndNuisance(observables, bool(show))
    ws.saveSnapshot("NominalParamValues", ws.allVars())
    for one in measurement.GetAsimovDatasets():
        _hf(PROGRESS, f"Generating additional Asimov Dataset: {one.GetName()}")
        one.ConfigureWorkspace(ws)
        made = asimov.GenerateAsimovData(pdf, observables)
        _hf(PROGRESS, "Importing Asimov dataset")
        ws.Import(made, RooCmdArg("Rename", one.GetName()))
        ws.loadSnapshot("NominalParamValues")
