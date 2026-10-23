"""HistFactory's ROOT histogram ownership and file-writing adapters.

The engine retains configuration and model assembly. Public TH1 objects and
TFile's directory, cloning and diagnostic behavior remain here until equivalent
engine protocols can replace them without changing their measured semantics.
"""

from __future__ import annotations

from typing import Any

from ...histfactory.model import _hf
from ...histfactory.output import not_written, write_measurement
from ...histfactory.systematics import HistFactoryError
from ...roofit.messages import ERROR, INFO, PROGRESS


def counting_data(name: str, data: Any) -> Any:
    """One observed count keeps the TH1F constructor's current-directory behavior."""
    from ..core.hists import TH1F

    hist = TH1F(name, name, 1, 0, 1)
    hist.SetBinContent(1, float(data))
    return hist


def counting_sample(name: str, value: float) -> Any:
    """A counting sample's TH1F is detached before its bin is assigned."""
    from ..core.hists import TH1F

    hist = TH1F(name, name, 1, 0, 1)
    hist.SetDirectory(0)
    hist.SetBinContent(1, value)
    return hist


def read_histogram(files: dict[str, Any], InputFile: str, HistoPath: str, HistoName: str) -> Any:
    """``Channel::GetHistogram``: the histogram, read from its file - opened once - and cloned
    out of it, said as ROOT says it."""
    from ..core.files import TFile

    _hf(PROGRESS, f"Getting histogram {InputFile}:{HistoPath}/{HistoName}")
    handle = files.get(InputFile)
    if handle is None:
        handle = TFile.Open(InputFile)
        if handle is None or not handle.IsOpen():
            _hf(ERROR, f"Error: Unable to open input file: {InputFile}")
            raise HistFactoryError(f"HistFactory - {InputFile} cannot be opened")
        files[InputFile] = handle
        _hf(INFO, f"Opened input file: {InputFile}: ")
    where = f"{HistoPath}/{HistoName}" if HistoPath else HistoName
    found = handle.Get(where)
    if found is None:
        _hf(ERROR, f"Histogram '{HistoName}' wasn't found in file '{InputFile}' in directory "
            f"'{HistoPath}'.")  # fmt: skip
        raise HistFactoryError(f"HistFactory - {InputFile} has no histogram {where}")
    made = found.Clone()
    made.SetDirectory(0)
    return made


def write_model(measurement: Any, filename: str, ws: Any, channel: Any = None) -> None:
    """The file of a workspace: the measurement's histograms - of one channel, if given."""
    import copy

    from ..core.files import TFile

    handle = TFile.Open(filename, "RECREATE")
    not_written(f"the RooWorkspace {ws.GetName()}", filename)
    if channel is not None:
        measurement = copy.deepcopy(measurement)
        measurement.GetChannels().clear()
        measurement.GetChannels().append(copy.deepcopy(channel))  # as C++ pushes a copy
        _hf(INFO, "About to write channel measurement to file")
    else:
        _hf(PROGRESS, f"Writing combined measurement to file: {filename}")
    write_measurement(measurement, handle)
    handle.Close()
