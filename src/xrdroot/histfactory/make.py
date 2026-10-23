"""``MakeModelAndMeasurementFast``: every channel's workspace, the combined one, their files.

Each channel's workspace is made and configured, and - as ROOT writes it -
its file ``<prefix>_<channel>_<measurement>_model.root`` opened; then the
combined workspace, configured, and ``<prefix>_combined_<measurement>
_model.root``. What xrdroot writes of each file is the measurement's
histograms (:mod:`.output`); the workspace, a class it does not write yet,
is said so on the standard error. The combined workspace is returned.
"""

from __future__ import annotations

import os
from typing import Any

from ..plugins import require
from ..roofit.messages import INFO, PROGRESS, TOPICS, service
from .assemble import log_fatal
from .factory import HistoToWorkspaceFactoryFast
from .model import _hf

__all__ = ["MakeModelAndMeasurementFast"]


def _output_directory(prefix: str) -> None:
    """The directory of the output prefix, made if it is not there."""
    head = prefix.rpartition("/")[0] if "/" in prefix else ""
    if head and not os.path.isdir(head):
        try:
            os.makedirs(head)
        except OSError:
            log_fatal(f"Error: Failed to make output directory: {os.getcwd()}/{head}")


def _written(measurement: Any, filename: str, ws: Any, channel: Any = None) -> None:
    require("histfactory.write_model")(measurement, filename, ws, channel)


def MakeModelAndMeasurementFast(measurement: Any, cfg: Any = None) -> Any:
    """The measurement's combined workspace, each channel's made and written on the way."""
    stream = service().getStream(1)
    stream.removeTopic(TOPICS["ObjectHandling"])
    try:
        return _make(measurement)
    finally:
        stream.addTopic(TOPICS["ObjectHandling"])


def _make(measurement: Any) -> Any:
    from ..roofit.printing import g
    from .combine import configure_for_measurement

    name = measurement.GetName()
    _hf(INFO, f"Making Model and Measurements (Fast) for measurement: {name}")
    error = measurement.GetLumi() * measurement.GetLumiRelErr()
    _hf(INFO, f"using lumi = {g(measurement.GetLumi())} and lumiError = {g(error)} including "
        f"bins between {measurement.GetBinLow()} and {measurement.GetBinHigh()}")  # fmt: skip
    fixed = "".join(f"   {p}\n" for p in measurement.GetConstantParams())
    service_text = f"fixing the following parameters:\n{fixed}"
    _hf(INFO, service_text[:-1])
    _hf(INFO, "Creating the HistoToWorkspaceFactoryFast factory")
    factory = HistoToWorkspaceFactoryFast(measurement)
    _hf(INFO, "Setting preprocess functions")
    factory.SetFunctionsToPreprocess(measurement.GetPreprocessFunctions())
    names, workspaces = [], []
    prefix = measurement.GetOutputFilePrefix()
    for channel in measurement.GetChannels():
        if not channel.CheckHistograms():
            log_fatal(f"MakeModelAndMeasurementsFast: Channel: {channel.GetName()} has "
                      "uninitialized histogram pointers")  # fmt: skip
        _hf(PROGRESS, f"Starting to process channel: {channel.GetName()}")
        names.append(channel.GetName())
        ws = factory.MakeSingleChannelModel(measurement, channel)
        _output_directory(prefix)
        filename = f"{prefix}_{channel.GetName()}_{name}_model.root"
        _hf(INFO, f"Opening File to hold channel: {filename}")
        _written(measurement, filename, ws, channel)
        _hf(PROGRESS, "Successfully wrote channel to file")
        workspaces.append(ws)
    ws = factory.MakeCombinedModel(names, workspaces)
    configure_for_measurement("simPdf", ws, measurement)
    _output_directory(prefix)
    filename = f"{prefix}_combined_{name}_model.root"
    _hf(PROGRESS, f"Writing combined workspace to file: {filename}")
    _written(measurement, filename, ws)
    return ws
