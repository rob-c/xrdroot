"""HistFactory's ``Measurement``: the channels combined, the parameters of interest, the lumi.

It says what the model is to measure - its parameters of interest, those to
hold constant or set, the luminosity and its uncertainty, how some
systematics are constrained - and holds the channels, each a copy, as ROOT
holds them. ``PrintTree`` prints it as ROOT does.
"""

from __future__ import annotations

import copy
from typing import Any

from ..roofit.messages import ERROR, INFO, WARNING
from ..roofit.printing import g
from .channel import Channel
from .model import _hf
from .shapes import Asimov, PreprocessFunction
from .systematics import HistFactoryError, _write

__all__ = ["Measurement"]


class Measurement:
    """A measurement: its channels and what is measured with them."""

    def __init__(self, name: str = "", title: str = "") -> None:
        self._name, self._title = str(name), str(title)
        self._prefix = ""
        self._poi: list[str] = []
        self._lumi, self._lumi_rel_err = 1.0, 0.1
        self._bin_low, self._bin_high = 0, 1
        self._channels: list[Channel] = []
        self._constant: list[str] = []
        self._values: dict[str, float] = {}
        self._functions: list[PreprocessFunction] = []
        self._asimovs: list[Asimov] = []
        self._syst: dict[str, dict[str, float]] = {k: {} for k in ("gamma", "uniform", "lognorm",
                                                                     "none")}  # fmt: skip

    def __deepcopy__(self, memo: Any) -> Measurement:
        made = copy.copy(self)
        made._channels = [copy.deepcopy(one, memo) for one in self._channels]
        made._poi, made._constant = list(self._poi), list(self._constant)
        made._values = dict(self._values)
        made._functions, made._asimovs = list(self._functions), list(self._asimovs)
        made._syst = {k: dict(v) for k, v in self._syst.items()}
        return made

    def GetName(self) -> str:
        return self._name

    def GetTitle(self) -> str:
        return self._title

    def SetName(self, name: str) -> None:
        self._name = str(name)

    def ClassName(self) -> str:
        return "RooStats::HistFactory::Measurement"

    def SetOutputFilePrefix(self, prefix: str) -> None:
        self._prefix = str(prefix)

    def GetOutputFilePrefix(self) -> str:
        return self._prefix

    # -- parameters ---------------------------------------------------------------

    def SetPOI(self, poi: str) -> None:
        self._poi.insert(0, str(poi))

    def AddPOI(self, poi: str) -> None:
        self._poi.append(str(poi))

    def GetPOI(self, index: int = 0) -> str:
        return self._poi[index]

    def GetPOIList(self) -> list[str]:
        return self._poi

    def AddConstantParam(self, param: str) -> None:
        if param in self._constant:
            _hf(WARNING, f"Warning: Setting parameter: {param} to constant, but it is already "
                "listed as constant.  You may ignore this warning.")  # fmt: skip
            return
        self._constant.append(str(param))

    def ClearConstantParams(self) -> None:
        self._constant.clear()

    def GetConstantParams(self) -> list[str]:
        return self._constant

    def SetParamValue(self, param: str, value: float) -> None:
        if param in self._values:
            _hf(WARNING, f"Warning: Chainging parameter: {param} value from: "
                f"{g(self._values[param])} to: {g(value)}")  # fmt: skip
        _hf(INFO, f"Setting parameter: {param} value to {g(value)}")
        self._values[str(param)] = float(value)

    def GetParamValues(self) -> dict[str, float]:
        return self._values

    def ClearParamValues(self) -> None:
        self._values.clear()

    def AddPreprocessFunction(self, name: str, expression: str, dependencies: str) -> None:
        self._functions.append(PreprocessFunction(name, expression, dependencies))

    def AddFunctionObject(self, function: PreprocessFunction) -> None:
        self._functions.append(function)

    def SetFunctionObjects(self, functions: Any) -> None:
        self._functions = list(functions)

    def GetFunctionObjects(self) -> list[PreprocessFunction]:
        return self._functions

    def GetPreprocessFunctions(self) -> list[str]:
        return [one.GetCommand() for one in self._functions]

    def GetAsimovDatasets(self) -> list[Asimov]:
        return self._asimovs

    def AddAsimovDataset(self, dataset: Asimov) -> None:
        self._asimovs.append(dataset)

    def SetLumi(self, lumi: float) -> None:
        self._lumi = float(lumi)

    def SetLumiRelErr(self, err: float) -> None:
        self._lumi_rel_err = float(err)

    def GetLumi(self) -> float:
        return self._lumi

    def GetLumiRelErr(self) -> float:
        return self._lumi_rel_err

    def SetBinLow(self, low: int) -> None:
        self._bin_low = int(low)

    def SetBinHigh(self, high: int) -> None:
        self._bin_high = int(high)

    def GetBinLow(self) -> int:
        return self._bin_low

    def GetBinHigh(self) -> int:
        return self._bin_high

    def GetInterpolationScheme(self) -> str:
        return ""

    # -- how systematics are constrained ------------------------------------------

    def AddGammaSyst(self, syst: str, uncert: float) -> None:
        self._syst["gamma"][str(syst)] = float(uncert)

    def AddLogNormSyst(self, syst: str, uncert: float) -> None:
        self._syst["lognorm"][str(syst)] = float(uncert)

    def AddUniformSyst(self, syst: str) -> None:
        self._syst["uniform"][str(syst)] = 1.0

    def AddNoSyst(self, syst: str) -> None:
        self._syst["none"][str(syst)] = 1.0

    def GetGammaSyst(self) -> dict[str, float]:
        return self._syst["gamma"]

    def GetLogNormSyst(self) -> dict[str, float]:
        return self._syst["lognorm"]

    def GetUniformSyst(self) -> dict[str, float]:
        return self._syst["uniform"]

    def GetNoSyst(self) -> dict[str, float]:
        return self._syst["none"]

    # -- channels -----------------------------------------------------------------

    def AddChannel(self, channel: Channel) -> None:
        self._channels.append(copy.deepcopy(channel))

    def GetChannels(self) -> list[Channel]:
        return self._channels

    def HasChannel(self, name: str) -> bool:
        return any(one.GetName() == name for one in self._channels)

    def GetChannel(self, name: str) -> Channel:
        for one in self._channels:
            if one.GetName() == name:
                return one
        _hf(ERROR, f"Error: Did not find channel: {name} in measurement: {self._name}")
        raise HistFactoryError(f"HistFactory - the measurement {self._name} has no channel {name}")

    def CollectHistograms(self) -> None:
        for one in self._channels:
            one.CollectHistograms()

    def PrintTree(self, stream: Any = None) -> None:
        """The measurement, its constants and functions, and every channel - as ROOT prints them."""
        text = (f"Measurement Name: {self._name}\t OutputFilePrefix: {self._prefix}\t POI: "
                f"{''.join(self._poi)}\t Lumi: {g(self._lumi)}\t LumiRelErr: "
                f"{g(self._lumi_rel_err)}\t BinLow: {self._bin_low}\t BinHigh: {self._bin_high}"
                "\t ExportOnly: 1\n")  # fmt: skip
        if self._constant:
            text += "Constant Params: " + "".join(f" {p}" for p in self._constant) + "\n"
        if self._functions:
            text += "Preprocess Functions: " + "".join(f" {f.GetCommand()}"
                                                       for f in self._functions) + "\n"  # fmt: skip
        if self._channels:
            text += "Channels:\n" + "".join(one.text() for one in self._channels)
        _write(stream, text)
        _hf(INFO, f"End Measurement: {self._name}")

    def PrintXML(self, directory: str = "", prefix: str = "") -> None:
        from ..errors import UnsupportedFeatureError

        raise UnsupportedFeatureError(
            "Measurement::PrintXML writes HistFactory's XML configuration; xrdroot does not yet - "
            "build the measurement in Python, as it was built here"
        )
