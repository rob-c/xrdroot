"""The MC-study modules the tutorials attach: the fit's chi-square, and randomised parameters.

``RooMCStudy`` tells every module when the run starts, before each generation and
after each fit. ``RooChi2MCSModule`` uses the last two to measure the fit's
goodness against the sample it was fitted to, and keeps ``chi2``, ``chi2red`` and
``prob`` as extra columns of the study's parameter data set; the randomisation
module draws the generator's parameters before each sample, so the study sees how
the fits behave when the truth moves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..stats import prob as chi2_probability

__all__ = ["RooChi2MCSModule", "RooDLLSignificanceMCSModule", "RooRandomizeParamMCSModule"]


class RooChi2MCSModule:
    """``RooChi2MCSModule``: each fit's chi-square against the sample it was fitted to.

    The chi-square is the textbook one for a binned sample: every bin's residual
    over its Poisson error, summed, with the model's shape scaled to the sample's
    own count - which is what ROOT's module reports as ``chi2``, ``chi2red`` and
    ``prob`` (the chi-square probability for the degrees of freedom left).
    """

    def __init__(self) -> None:
        self.study: Any = None
        self._found: dict[str, list[float]] = {"chi2": [], "chi2red": [], "prob": []}

    def initializeInstance(self) -> bool:
        return True

    def initializeRun(self, numSamples: int) -> bool:
        self._found = {"chi2": [], "chi2red": [], "prob": []}
        return True

    def processAfterFit(self, fitOk: bool) -> bool:
        if not fitOk:
            return True
        chi2, ndf = self._chi2()
        self._found["chi2"].append(chi2)
        self._found["chi2red"].append(chi2 / ndf if ndf > 0 else 0.0)
        self._found["prob"].append(chi2_probability(chi2, ndf) if ndf > 0 else 0.0)
        return True

    def finalizeRun(self) -> dict[str, list[float]]:
        return self._found

    def _chi2(self) -> tuple[float, int]:
        """The residual sum over the sample's bins, and the degrees of freedom left."""
        study = self.study
        sample = getattr(study, "last_sample", None)
        if sample is None or not hasattr(sample, "_grid_weights"):
            return 0.0, 0
        weights = np.asarray(sample._grid_weights(), dtype=np.float64).ravel()
        model = study.fit_model
        names = [one.GetName() for one in study.observables]
        edges = sample._edges()
        shape = self._model_shape(model, names, edges)
        total = float(weights.sum())
        # the fitted model is scaled to the sample's own count, as RooFit's chi2 is
        expected = shape * (total / shape.sum()) if shape.sum() > 0 else shape
        found = (weights > 0) & (expected > 0)
        chi2 = float(((weights[found] - expected[found]) ** 2 / expected[found]).sum())
        free = sum(
            1
            for one in study.fit_params
            if not one.isConstant() and one.hasError(False)
        )
        return chi2, int(found.sum()) - int(free)

    def _model_shape(self, model: Any, names: list[str], edges: list[Any]) -> Any:
        """The model's density at every bin centre, times the bin's volume."""
        grids = np.meshgrid(*[0.5 * (edge[:-1] + edge[1:]) for edge in edges], indexing="ij")
        ctx = dict(zip(names, grids, strict=True))
        density = np.asarray(model.compute(ctx), dtype=np.float64)
        volume = np.prod([np.diff(edge) for edge in edges], axis=0)
        return density * volume


class RooRandomizeParamMCSModule:
    """``RooRandomizeParamMCSModule``: the generator's parameters drawn afresh for each sample.

    ``sampleUniform(param, lo, hi)`` and ``sampleGaussian(param, mean, sigma)``
    register one; the ``sampleSumUniform``/``sampleSumGauss`` forms draw for a whole
    set of parameters at once, adding the drawn offset to each. Before every sample
    the registered draws are made and written into the parameters, and the values
    drawn are kept as columns of the study's parameter data set.
    """

    def __init__(self) -> None:
        self.study: Any = None
        self._draws: list[tuple[str, Any, str, float, float]] = []
        self._found: dict[str, list[float]] = {}

    def sampleUniform(self, param: Any, lo: float, hi: float) -> None:
        """``sampleUniform``: this parameter drawn uniformly from ``[lo, hi)`` each sample."""
        self._draws.append(("one", param, "uniform", float(lo), float(hi)))

    def sampleGaussian(self, param: Any, mean: float, sigma: float) -> None:
        """``sampleGaussian``: this parameter drawn normally each sample."""
        self._draws.append(("one", param, "gauss", float(mean), float(sigma)))

    def sampleSumUniform(self, params: Any, lo: float, hi: float) -> None:
        """``sampleSumUniform``: one uniform offset, added to every parameter in the set."""
        self._draws.append(("set", params, "uniform", float(lo), float(hi)))

    def sampleSumGauss(self, params: Any, mean: float, sigma: float) -> None:
        """``sampleSumGauss``: one normal offset, added to every parameter in the set."""
        self._draws.append(("set", params, "gauss", float(mean), float(sigma)))

    def initializeInstance(self) -> bool:
        for kind, target, *_ in self._draws:
            for one in self._targets(kind, target):
                self._found.setdefault(f"{one.GetName()}_gen", [])
        return True

    def initializeRun(self, numSamples: int) -> bool:
        for values in self._found.values():
            values.clear()
        return True

    def processBeforeGen(self, sampleNum: int) -> bool:
        from .rng import generator

        rng = generator()
        for kind, target, which, first, second in self._draws:
            if which == "uniform":
                drawn = rng.Rndm() * (second - first) + first
            else:
                drawn = rng.Gaus(first, second)
            for one in self._targets(kind, target):
                one.setVal(drawn if kind == "one" else one.getVal() + drawn)
                self._found[f"{one.GetName()}_gen"].append(float(one.getVal()))
        return True

    def processAfterFit(self, fitOk: bool) -> bool:
        return True

    def finalizeRun(self) -> dict[str, list[float]]:
        return self._found

    def _targets(self, kind: str, target: Any) -> list[Any]:
        from .collections import as_list

        return [target] if kind == "one" else list(as_list(target))


class RooDLLSignificanceMCSModule:
    """``RooDLLSignificanceMCSModule``: how far each fit is from a null value of one parameter.

    ROOT's module re-fits the model with the parameter held at the null hypothesis
    and keeps the difference of the two minima - ``dll_nullhypo_<param>`` - together
    with its significance, ``sig_nullhypo_<param>``, which is the square root of
    twice that difference for a one-sided test.
    """

    def __init__(self, param: Any, nullHypoValue: float = 0.0) -> None:
        self.study: Any = None
        self.param = param
        self.null = float(nullHypoValue)
        self._name = param if isinstance(param, str) else param.GetName()
        self._found: dict[str, list[float]] = {
            f"dll_nullhypo_{self._name}": [],
            f"significance_nullhypo_{self._name}": [],
        }

    def initializeInstance(self) -> bool:
        return True

    def initializeRun(self, numSamples: int) -> bool:
        for values in self._found.values():
            values.clear()
        return True

    def processAfterFit(self, fitOk: bool) -> bool:
        study = self.study
        if not fitOk or study is None or not study.results:
            return True
        from .cmdargs import RooCmdArg

        target = (
            study.fit_model.getParameters(study.observables)[self._name]
            if isinstance(self.param, str)
            else self.param
        )
        start = target.getVal()
        target.setVal(self.null)
        target.setConstant(True)
        try:
            result = study.fit_model.fitTo(
                study.last_sample, RooCmdArg("Save", True), RooCmdArg("PrintLevel", -1)
            )
        finally:
            target.setConstant(False)
            target.setVal(start)
        difference = float(result.minNll() - study.results[-1].minNll())
        self._found[f"dll_nullhypo_{self._name}"].append(difference)
        self._found[f"significance_nullhypo_{self._name}"].append(
            float(np.sqrt(max(2.0 * difference, 0.0)))
        )
        return True

    def finalizeRun(self) -> dict[str, list[float]]:
        return self._found
