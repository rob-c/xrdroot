"""``TMVA::Reader``: trained methods read back from their weight files and asked about one event.

A macro declares its variables to the reader by address - ``&var1``, which
the translator makes a :class:`~xrdroot.cint.runtime.cells.Cell`, or a
one-element ``array('f')`` from Python - and each ``EvaluateMVA`` reads
them, in single precision as TMVA's ``Float_t`` addresses hold them, and
hands the event to the method as the Factory's testing did. Every method's
weight file this package writes can be read, and TMVA's own for the methods
whose weights xrdroot reads (all but PDE-Foam's foams).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import DataSetInfo, Events
from .log import CONFIG, Logger
from .method import MULTICLASS, REGRESSION, Method
from .options import Options
from .tools import CxxVector
from .variables import VariableInfo
from .weightfile import read_method

__all__ = ["Reader"]


def _read(address: Any) -> float:
    """What an address holds: a cell's ``.value``, or an array's first element."""
    value = address.value if hasattr(address, "value") else address[0]
    return float(np.float32(value))


class Reader:
    """``TMVA::Reader(options)``: ``!Color``, ``Silent`` and ``V`` as TMVA reads them."""

    def __init__(self, *arguments: Any) -> None:
        text = next((str(a) for a in arguments if isinstance(a, str)), "")
        options = Options(text)
        CONFIG.use_color = options.flag("Color", True)
        CONFIG.silent = options.flag("Silent", False)
        self.dsi = DataSetInfo("Default")
        self.addresses: list[Any] = []
        self.spectator_addresses: list[Any] = []
        self.methods: dict[str, Method] = {}
        self.log = Logger("Reader")
        self.last_error = -1.0
        self._rebuilt = False
        for argument in arguments:
            if not isinstance(argument, str):
                for name in argument:
                    self.AddVariable(str(name), None)

    # -- declarations ---------------------------------------------------------------------

    def AddVariable(self, expression: Any, address: Any = None) -> None:
        self.dsi.variables.append(VariableInfo(str(expression)))
        self.addresses.append(address)

    def AddSpectator(self, expression: Any, address: Any = None) -> None:
        self.dsi.spectators.append(VariableInfo(str(expression)))
        self.spectator_addresses.append(address)

    def BookMVA(self, tag: Any, weightfile: Any = None) -> Method:
        """``BookMVA(tag, file)``: the method in the file, known from now on by ``tag``."""
        tag, path = str(tag), str(weightfile if weightfile is not None else tag)
        from .xmlfile import load

        kind = str(load(path).get("Method")).partition("::")[0]
        self.log.info(f'Booking "{tag}" of type "{kind}" from {path}.')
        method = read_method(path, self.dsi, "", self.log)
        self.log.info(f'Booked classifier "{method.name}" of type: "{method.type_name}"')
        self.methods[tag] = method
        return method

    def FindMVA(self, tag: Any) -> Method | None:
        return self.methods.get(str(tag))

    def FindCutsMVA(self, tag: Any) -> Method | None:
        return self.FindMVA(tag)

    # -- the event ------------------------------------------------------------------------

    def _method(self, tag: Any) -> Method:
        method = self.methods.get(str(tag))
        if method is None:
            raise self.log.fatal(
                f'<EvaluateMVA> unknown classifier in map; you looked for "{tag}" while the '
                "available methods are: " + ", ".join(self.methods)
            )
        if not self._rebuilt:
            self._rebuilt = True
            self.log.info(f"Rebuilding Dataset {self.dsi.name}")
        return method

    def _event(self, values: Any = None) -> Events:
        if values is None:
            values = [_read(address) for address in self.addresses]
        row = np.asarray(values, dtype=np.float32).astype(np.float64)[None, :]
        ntargets = max(self.dsi.GetNTargets(), 1)
        return Events(
            row,
            np.zeros((1, ntargets)),
            np.zeros((1, len(self.spectator_addresses))),
            np.zeros(1, dtype=np.int64),
            np.ones(1),
        )

    def _split(self, arguments: tuple[Any, ...]) -> tuple[Any, Any, tuple[Any, ...]]:
        """``(tag, ...)`` or ``(values, tag, ...)``: the values, if given, the tag and the rest."""
        if arguments and not isinstance(arguments[0], str) and not hasattr(arguments[0], "Data"):
            return list(arguments[0]), arguments[1], arguments[2:]
        return None, arguments[0], arguments[1:]

    def EvaluateMVA(self, *arguments: Any) -> float:
        """``EvaluateMVA(tag, aux=0)``, or ``(values, tag, aux)``: the method's output."""
        values, tag, rest = self._split(arguments)
        method = self._method(tag)
        if method.type_name == "Cuts":
            method.test_signal_eff = float(rest[0]) if rest else -1.0
        events = self._event(values)
        output = float(np.ravel(method.mva(events))[0])
        self.last_error = float(np.ravel(method.error(events))[0])
        return output

    def EvaluateRegression(self, *arguments: Any) -> Any:
        """``EvaluateRegression(tag)``: every target; ``(i, tag)``: target ``i``."""
        if arguments and isinstance(arguments[0], (int, np.integer)):
            return float(self.EvaluateRegression(*arguments[1:])[int(arguments[0])])
        values, tag, _ = self._split(arguments)
        method = self._method(tag)
        if method.analysis != REGRESSION:
            raise self.log.fatal(f'<EvaluateRegression> "{tag}" was not trained for regression')
        found = np.ravel(method.mva(self._event(values)))
        return CxxVector(float(np.float32(v)) for v in found)

    def EvaluateMulticlass(self, *arguments: Any) -> Any:
        """``EvaluateMulticlass(tag)``: every class's output; ``(i, tag)``: class ``i``'s."""
        if arguments and isinstance(arguments[0], (int, np.integer)):
            return float(self.EvaluateMulticlass(*arguments[1:])[int(arguments[0])])
        values, tag, _ = self._split(arguments)
        method = self._method(tag)
        if method.analysis != MULTICLASS:
            raise self.log.fatal(f'<EvaluateMulticlass> "{tag}" was not trained for multiclass')
        found = np.ravel(method.mva(self._event(values)))
        return CxxVector(float(np.float32(v)) for v in found)

    def GetMVAError(self) -> float:
        return self.last_error

    def GetProba(self, tag: Any, signal_fraction: float = 0.5, value: float = -9999999) -> float:
        method = self._method(tag)
        if value == -9999999:
            value = float(np.ravel(method.mva(self._event()))[0])
        return float(np.ravel(method.proba(np.array([value]), signal_fraction))[0])

    def GetRarity(self, tag: Any, value: float = -9999999) -> float:
        method = self._method(tag)
        if value == -9999999:
            value = float(np.ravel(method.mva(self._event()))[0])
        return float(np.ravel(method.rarity(np.array([value])))[0])
