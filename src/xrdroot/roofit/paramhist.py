"""``RooParamHistFunc``: a histogram whose bin contents are scale factors, one per bin.

The Barlow-Beeston tutorials fit a shape whose contents are not fixed by the
histogram but *parameterised* by it: each bin's rate is the bin's content times a
gamma parameter of its own, ``<function>_gamma_bin_<i>``, which a constraint
then ties to the histogram (``RooHistConstraint``). A function made with another
one's parameters - ``RooParamHistFunc(..., dh, x, other, True)`` - shares those
gammas, which is how one set of scale factors is applied to a second histogram.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .collections import RooArgList
from .real import Context, RooAbsReal
from .variables import RooRealVar

__all__ = ["RooParamHistFunc"]


class RooParamHistFunc(RooAbsReal):
    """``RooParamHistFunc``: ``histogram.bin(x) * gamma_bin``, one gamma parameter per bin."""

    def __init__(
        self,
        name: Any,
        title: Any,
        dh: Any,
        x: Any,
        paramHistFunc: Any = None,
        _relBins: bool = False,
    ) -> None:
        super().__init__(name, title)
        self.data_hist = dh
        self._x = RooArgList(x if isinstance(x, (list, tuple, RooArgList)) else [x])
        self._proxies_x = [self._proxy(f"x{i}", one) for i, one in enumerate(self._x)]
        weights = np.asarray(dh._grid_weights(), dtype=np.float64).ravel()
        if paramHistFunc is None:
            self._gammas = [
                RooRealVar(f"{name}_gamma_bin_{index}", f"gamma_bin_{index}", 1.0, 0.0, 10.0)
                for index in range(len(weights))
            ]
        else:
            self._gammas = list(paramHistFunc.binParameters())
            if len(self._gammas) != len(weights):
                raise ValueError(
                    f"RooParamHistFunc {name} holds {len(weights)} bins, and the function whose "
                    f"parameters it shares holds {len(self._gammas)}; they have to match."
                )
        self._gamma_proxies = [
            self._proxy(f"gamma{index}", gamma) for index, gamma in enumerate(self._gammas)
        ]
        self._weights = weights

    # --- what it is made of ---------------------------------------------------

    def xList(self) -> RooArgList:
        """``xList``: the observables it is a function of."""
        return self._x

    def binParameters(self) -> RooArgList:
        """The gamma parameters, one per bin, in bin order.

        ROOT spells this ``getParameters``, which this library's base class already
        uses for the parameters a model is fitted on (with the observables taken
        out), so the per-bin scale factors are named for what they are instead.
        """
        return RooArgList(self._gammas)

    def binContents(self) -> np.ndarray[Any, Any]:
        """The histogram's own contents - what the gammas scale."""
        return self._weights.copy()

    # --- its value ------------------------------------------------------------

    def compute(self, ctx: Context) -> Any:
        """The bin's content times its gamma, for every point asked about."""
        columns = {one.GetName(): np.atleast_1d(one.compute(ctx)) for one in self._x}
        found = self.data_hist._bin_of(columns)
        scaled = np.array(
            [float(gamma.compute(ctx)) for gamma in self._gammas], dtype=np.float64
        )
        contents = self._weights * scaled
        clipped = np.where(found < 0, 0, found)
        return np.where(found < 0, 0.0, contents[clipped])

    def __repr__(self) -> str:
        return f"<RooParamHistFunc {self.GetName()!r} of {len(self._gammas)} bins>"
