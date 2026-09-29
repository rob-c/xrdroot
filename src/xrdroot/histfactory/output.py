"""Writing a measurement's histograms to its output file, as ``Measurement::writeToFile`` does.

Each channel gets a ``<channel>_hists`` directory, with the data in
``data`` and each sample's histograms - nominal, statistical errors as
``statisticalErrors``, systematics - in a directory of its own; each
histogram's configuration then names where it was written. The
``Measurement`` object itself and the workspace are ROOT classes xrdroot
does not write yet: that is said on the standard error, and the rest of
the file is written.
"""

from __future__ import annotations

import sys
from typing import Any

from ..roofit.messages import ERROR, PROGRESS
from .model import _hf
from .systematics import HistFactoryError

__all__ = ["not_written", "write_measurement"]


def not_written(what: str, filename: str) -> None:
    """The word, on the standard error, that ``what`` is not in ``filename``."""
    sys.stderr.write(f"xrdroot: {what} is not written to {filename}: xrdroot does not write "
                     "that class yet; the file holds the measurement's histograms\n")  # fmt: skip


def _path(directory: Any) -> str:
    """``GetDirPath``: the directory's path in its file, and a slash."""
    path = directory.GetPath()
    return str((path.split(":", 1)[1] if ":" in path else path) + "/")


def _written(directory: Any, hist: Any, name: Any = None) -> str:
    directory.WriteTObject(hist, name)
    return str(name or hist.GetName())


def _sample(sample: Any, directory: Any, filename: str) -> None:
    """``Sample::writeToFile``: its nominal, its statistical errors, its systematics' histograms."""
    where = _path(directory)
    sample.SetHistoName(_written(directory, sample.GetHisto()))
    sample.SetInputFile(filename)
    sample.SetHistoPath(where)
    stat = sample.GetStatError()
    if stat.GetUseHisto():
        stat.SetHistoName(_written(directory, stat.GetErrorHist(), "statisticalErrors"))
        stat.SetInputFile(filename)
        stat.SetHistoPath(where)
    for syst in [*sample.GetHistoSysList(), *sample.GetHistoFactorList()]:
        syst.SetHistoNameLow(_written(directory, syst.GetHistoLow()))
        syst.SetHistoNameHigh(_written(directory, syst.GetHistoHigh()))
        for side in ("Low", "High"):
            getattr(syst, f"SetInputFile{side}")(filename)
            getattr(syst, f"SetHistoPath{side}")(where)
    for one in [*sample.GetShapeSysList(), *sample.GetShapeFactorList()]:
        hist = one.GetErrorHist()
        if hist is not None:
            one.SetHistoName(_written(directory, hist))
            one.SetInputFile(filename)
            one.SetHistoPath(where)


def write_measurement(measurement: Any, handle: Any) -> None:
    """Every channel's histograms into ``handle``, said as ROOT says it."""
    filename = handle.GetName()
    for channel in measurement.GetChannels():
        if not channel.CheckHistograms():
            _hf(ERROR, f"Measurement.writeToFile(): Channel: {channel.GetName()} has "
                "uninitialized histogram pointers")  # fmt: skip
            raise HistFactoryError(f"HistFactory - channel {channel.GetName()} lacks histograms")
        folder = handle.mkdir(f"{channel.GetName()}_hists")
        data_dir = folder.mkdir("data")
        data = channel.GetData()
        if data.GetHisto() is not None:
            data.SetHistoName(_written(data_dir, data.GetHisto()))
            data.SetInputFile(filename)
            data.SetHistoPath(_path(data_dir))
        for sample in channel.GetSamples():
            _hf(PROGRESS, f"Writing sample: {sample.GetName()}")
            _sample(sample, folder.mkdir(sample.GetName()), filename)
    _hf(PROGRESS, "Saved all histograms")
    not_written("the Measurement", filename)
    _hf(PROGRESS, "Saved Measurement")
