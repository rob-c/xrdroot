"""``NumberCountingPdfFactory``: counting experiments in several bins, each with a sideband.

Each bin is a Poisson of the main measurement ``x_i`` for ``s_i + b_i`` -
``s_i`` the master signal strength times the expected signal - times a
Poisson of the sideband ``y_i`` for ``b_i tau_i``; the model is their
product. The data methods set ``tau_i`` from the background and its
relative uncertainty (or take it), and fit the ranges of the observables
and the background to the numbers given.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.messages import DEBUG, ERROR, FATAL, WARNING, log, service

__all__ = ["NumberCountingPdfFactory"]

#: How many sigma above the background its range reaches.
MAX_SIGMA = 8.0


def _killed(level: int) -> None:
    """``setGlobalKillBelow`` as the factory calls it - leaving it at ``DEBUG`` afterwards."""
    service().setGlobalKillBelow(level)


class NumberCountingPdfFactory:
    """Builds the number counting model and its data into a workspace."""

    def AddModel(self, sig: Any, nbins: int, ws: Any, pdfName: str = "CombinedPdf",
                 muName: str = "masterSignal") -> None:  # fmt: skip
        from ..roofit.functions import RooAddition, RooProduct
        from ..roofit.pdfs.prodpdf import RooProdPdf
        from ..roofit.pdfs.shapes import RooPoisson
        from ..roofit.variables import RooRealVar

        master = RooRealVar(muName, "masterSignal", 1.0, 0.0, 3.0)
        factors = []
        for i in range(int(nbins)):
            n = f"_{i}"
            expected = RooRealVar(f"expected_s{n}", f"expected_s{n}", sig[i], 0.0, 2 * sig[i])
            expected.setConstant(True)
            s = RooProduct(f"s{n}", f"s{n}", [master, expected])
            b = RooRealVar(f"b{n}", f"b{n}", 0.5, 0.0, 1.0)
            tau = RooRealVar(f"tau{n}", f"tau{n}", 0.5, 0.0, 1.0)
            tau.setConstant(True)
            splusb = RooAddition(f"splusb{n}", f"s{n}+b{n}", [s, b])
            btau = RooProduct(f"bTau{n}", f"b*tau{n}", [b, tau])
            x = RooRealVar(f"x{n}", f"x{n}", 0.5, 0.0, 1.0)
            y = RooRealVar(f"y{n}", f"y{n}", 0.5, 0.0, 1.0)
            factors.append(RooPoisson(f"sigRegion{n}", f"sigRegion{n}", x, splusb))
            factors.append(RooPoisson(f"sideband{n}", f"sideband{n}", y, btau, True))
        joint = RooProdPdf(pdfName, "joint", factors)
        _killed(ERROR)
        ws.Import(joint)
        _killed(DEBUG)

    def AddExpData(self, sig: Any, back: Any, back_syst: Any, nbins: int, ws: Any,
                   dsName: str = "ExpectedNumberCountingData") -> None:  # fmt: skip
        main = [sig[i] + back[i] for i in range(int(nbins))]
        self.AddData(main, back, back_syst, nbins, ws, dsName)

    def AddExpDataWithSideband(self, sigExp: Any, backExp: Any, tau: Any, nbins: int, ws: Any,
                               dsName: str = "ExpectedNumberCountingData") -> None:  # fmt: skip
        main = [sigExp[i] + backExp[i] for i in range(int(nbins))]
        side = [backExp[i] * tau[i] for i in range(int(nbins))]
        self.AddDataWithSideband(main, side, tau, nbins, ws, dsName)

    @staticmethod
    def SafeObservableCreation(ws: Any, varName: str, value: float,
                               maximum: float | None = None) -> Any:  # fmt: skip
        """The workspace's variable of that name - or a new one up to ``maximum`` - reaching
        and set to ``value``."""
        from ..roofit.variables import RooRealVar

        value = float(value)
        x = ws.var(varName)
        if x is None:
            top = 10.0 * value if maximum is None else float(maximum)
            x = RooRealVar(varName, varName, value, 0.0, top)
        if x.getMax() < value:
            x.setMax(max(x.getMax(), 10 * value))
        x.setVal(value)
        return x

    def AddData(self, mainMeas: Any, back: Any, back_syst: Any, nbins: int, ws: Any,
                dsName: str = "NumberCountingData") -> None:  # fmt: skip
        rows = []
        for i in range(int(nbins)):
            err = float(back_syst[i])
            tau = (1.0 + math.sqrt(1 + 4 * err * err)) / (2.0 * err * err) / back[i]
            rows.append((float(mainMeas[i]), float(back[i]) * tau, tau, float(back[i]), err))
        self._add(ws, dsName, rows)

    def AddDataWithSideband(self, mainMeas: Any, sideband: Any, tauForTree: Any, nbins: int,
                            ws: Any, dsName: str = "NumberCountingData") -> None:  # fmt: skip
        rows = []
        for i in range(int(nbins)):
            tau, side = float(tauForTree[i]), float(sideband[i])
            rows.append((float(mainMeas[i]), side, tau, side / tau, 1.0 / math.sqrt(side)))
        self._add(ws, dsName, rows)

    def _add(self, ws: Any, name: str, rows: list[Any]) -> None:
        """Each bin's ``tau``, observables and background range, and the one-event dataset."""
        from ..roofit.data.dataset import RooDataSet

        observables = []
        for i, (main, side, tau_value, back, syst) in enumerate(rows):
            tau = self.SafeObservableCreation(ws, f"tau_{i}", tau_value)
            log(None, WARNING, "ObjectHandling", f"NumberCountingPdfFactory: changed value of "
                f"{tau.GetName()} to {_g(tau.getVal())} to be consistent with background and "
                "its uncertainty.  Also stored these values of tau into workspace with name . "
                f"{tau.GetName()}{name} if you test with a different dataset, you should adjust "
                "tau appropriately.\n")  # fmt: skip
            _killed(ERROR)
            ws.Import(tau.clone(f"{tau.GetName()}{name}"))
            _killed(DEBUG)
            observables.append(self.SafeObservableCreation(ws, f"x_{i}", main))
            observables.append(self.SafeObservableCreation(ws, f"y_{i}", side))
            b = ws.var(f"b_{i}")
            b.setMax(1.2 * back + MAX_SIGMA * (math.sqrt(back) + back * syst))
            b.setVal(back)
        data = RooDataSet(name, "Number Counting Data", observables)
        data.add(observables)
        _killed(FATAL)
        ws.Import(data)
        _killed(DEBUG)


def _g(value: float) -> str:
    from ..roofit.printing import g

    return g(value)
