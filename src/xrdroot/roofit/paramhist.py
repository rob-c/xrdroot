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
from .pdf import RooAbsPdf
from .real import Context, RooAbsReal
from .variables import RooRealVar

__all__ = ["RooHistConstraint", "RooParamHistFunc"]


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


class RooHistConstraint(RooAbsPdf):
    """``RooHistConstraint``: the constraint that ties a histogram's gammas to its contents.

    Every bin of a :class:`RooParamHistFunc` has a gamma scaling it, and the
    histogram says how well each is known: a bin with ``n`` entries lets its gamma
    fluctuate by ``1/sqrt(n)``, which is a Poisson of mean ``gamma n`` observed at
    ``n``. The constraint is the product of those terms over every bin of every
    function given - once per gamma, since two functions may share them - and a bin
    with more than ``threshold`` entries is constrained the Gaussian way, as ROOT
    does for a large count.
    """

    def __init__(self, name: Any, title: Any, phfSet: Any, threshold: int = 1000000) -> None:
        super().__init__(name, title)
        from .collections import as_list

        self._threshold = int(threshold)
        self._terms: list[tuple[Any, float]] = []
        seen: set[int] = set()
        for function in as_list(phfSet):
            for gamma, nominal in zip(
                function.binParameters(), function.binContents(), strict=True
            ):
                if id(gamma) in seen:  # shared scale factors are constrained once
                    continue
                seen.add(id(gamma))
                self._terms.append((gamma, float(nominal)))
        if not self._terms:
            raise ValueError(
                "RooHistConstraint needs at least one RooParamHistFunc with bins to constrain; "
                "it was given nothing that has any."
            )
        self._gamma_proxies = [
            self._proxy(f"gamma{index}", gamma) for index, (gamma, _) in enumerate(self._terms)
        ]

    def binParameters(self) -> Any:
        """The gamma parameters it constrains, in the order taken from the functions."""
        from .collections import RooArgList

        return RooArgList([gamma for gamma, _ in self._terms])

    def compute(self, ctx: Context) -> Any:
        """The product of the per-bin terms: Poisson, or Gaussian above the threshold."""
        from ..random import libm

        total = 1.0
        for gamma, nominal in self._terms:
            if nominal <= 0.0:
                # an empty bin says nothing about its gamma, so it constrains nothing
                continue
            mean = max(float(gamma.compute(ctx)), 1e-12) * nominal
            if nominal >= self._threshold:
                sigma = np.sqrt(mean)
                total = total * float(np.exp(-0.5 * ((nominal - mean) / sigma) ** 2))
            else:
                log_term = -mean + nominal * libm.log(mean) - libm.lgamma(nominal + 1.0)
                total = total * float(libm.exp(log_term))
        return total

    def __repr__(self) -> str:
        return f"<RooHistConstraint {self.GetName()!r} of {len(self._terms)} bins>"
