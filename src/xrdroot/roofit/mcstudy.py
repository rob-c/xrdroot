"""``RooMCStudy``: generate many samples from a model, fit each, and collect what the fits found.

``RooMCStudy(model, {x}, Extended=True, FitOptions=...)`` makes the model's
generator once - so its numerical sampler is set up once, as RooFit's is -
and ``generateAndFit(n)`` draws, for each of ``n`` samples counting down, a
Poisson number of events (if extended) and the events, fits the model to
them starting from the parameters it started with, and keeps each fit's
parameters, errors, pulls and minimum in ``fitParDataSet()``.
"""

from __future__ import annotations

from typing import Any

from .cmdargs import RooCmdArg, commands
from .collections import RooArgSet, as_list
from .messages import PROGRESS, SERVICE, WARNING, log
from .rng import generator
from .variables import RooRealVar

__all__ = ["RooMCStudy"]


class RooMCStudy:
    """A study of a model by repeated generation and fitting."""

    def __init__(self, model: Any, observables: Any, *args: Any, **kwargs: Any) -> None:
        options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
        self.gen_model = model
        self.fit_model = options.get("FitModel") or model
        self.observables = as_list(observables)
        self.silence = bool(options.get("Silence", 0, False))
        self.extended = bool(options.get("Extended", 0, False))
        self.binned = bool(options.get("Binned", 0, False))
        self.fit_options = _fit_options(options)
        self.gen_params = [p for p in model.getParameters(self.observables)]
        self.gen_init = [(p, p.getVal()) for p in self.gen_params]
        self.fit_params = list(self.fit_model.getParameters(self.observables))
        self.fit_init = [p.clone(p.GetName()) for p in self.fit_params]
        self._generator = None if self.binned else _generator(model, self.observables)
        self.results: list[Any] = []
        self.samples: list[Any] = []
        self.rows: list[dict[str, float]] = []
        self._data: Any = None

    # -- running ------------------------------------------------------------------

    def generateAndFit(self, nSamples: int, nEvtPerSample: int = 0, keepGenData: bool = False,
                       asciiFilePat: Any = None) -> bool:  # fmt: skip
        return self._run(True, True, int(nSamples), int(nEvtPerSample), keepGenData)

    def generate(self, nSamples: int, nEvtPerSample: int = 0, keepGenData: bool = False,
                 asciiFilePat: Any = None) -> bool:  # fmt: skip
        return self._run(True, False, int(nSamples), int(nEvtPerSample), True)

    def fit(self, nSamples: int, *args: Any) -> bool:
        for sample in list(self.samples)[:nSamples]:
            self._fit_sample(sample)
        return False

    def _run(self, generate: bool, fit: bool, samples: int, events: int, keep: bool) -> bool:
        before = SERVICE.globalKillBelow()
        if self.silence:
            SERVICE.setGlobalKillBelow(PROGRESS)
        prescale = int(samples / 100) if samples > 100 else 1
        try:
            while samples:
                samples -= 1
                if samples % prescale == 0:
                    log(self.fit_model, PROGRESS, "Generation", f"RooMCStudy::run: sample {samples}")
                sample = self._sample(events)
                if keep:
                    self.samples.append(sample)
                if fit:
                    self._fit_sample(sample)
            if fit:
                self._check_pulls()
        finally:
            SERVICE.setGlobalKillBelow(before)
        self._data = None
        return False

    def _sample(self, events: int) -> Any:
        """One sample: the generator's parameters reset, a Poisson number of events if extended."""
        for par, value in self.gen_init:
            par.setVal(value)
        count = events
        if self.extended:
            expected = self.gen_model.expected(frozenset(o.GetName() for o in self.observables))
            count = int(generator().Poisson(events or expected))
        if self.binned:
            return self.gen_model.generateBinned(self.observables, count)
        return self._generator.sample(count, f"{self.gen_model.GetName()}Data")

    def _fit_sample(self, sample: Any) -> None:
        """``fitSample``: the fit parameters reset to their start, the fit, and its row if good."""
        for par, start in zip(self.fit_params, self.fit_init):
            par.copy_value_from(start)
        if sample.sumEntries() <= 0:
            return
        options = [*self.fit_options, RooCmdArg("Save", True),
                   RooCmdArg("PrintLevel", -1 if self.silence else 1)]  # fmt: skip
        result = self.fit_model.fitTo(sample, *options)
        self.results.append(result)
        if result.status() == 0:
            row = {p.GetName(): p.getVal() for p in self.fit_params}
            row.update({f"{p.GetName()}err": p.getError() for p in self.fit_params})
            row.update({"NLL": result.minNll(), "ngen": sample.sumEntries()})
            self.rows.append(row)

    def _check_pulls(self) -> None:
        """``calcPulls``' warning for each parameter without an error, which has no pull."""
        for par in self.fit_params:
            if not par.hasError(False):
                warn(self.fit_model, f"Fit parameter '{par.GetName()}' does not have an error. A pull "
                     "distribution cannot be generated. This might be caused by the parameter being "
                     "constant or because the fits were not run.")  # fmt: skip

    # -- what was found -----------------------------------------------------------

    def fitResult(self, index: int) -> Any:
        return self.results[int(index)]

    def genData(self, index: int) -> Any:
        return self.samples[int(index)]

    def fitParams(self, index: int) -> RooArgSet:
        row = self.rows[int(index)]
        return RooArgSet([RooRealVar(p.GetName(), p.GetTitle(), row[p.GetName()]) for p in self.fit_params])

    def fitParDataSet(self) -> Any:
        """``fitParDataSet``: every good fit's parameters, errors, pulls, minimum and event count."""
        if self._data is None:
            from .mcstudydata import parameter_data

            self._data = parameter_data(self)
        return self._data

    def plotParam(self, param: Any, *args: Any, **kwargs: Any) -> Any:
        from .mcstudydata import plot_column

        return plot_column(self, param.GetName(), args, kwargs, symmetric=False)

    def plotError(self, param: Any, *args: Any, **kwargs: Any) -> Any:
        from .mcstudydata import plot_column

        return plot_column(self, f"{param.GetName()}err", args, kwargs, symmetric=False)

    def plotPull(self, param: Any, *args: Any, **kwargs: Any) -> Any:
        from .mcstudydata import plot_pull

        return plot_pull(self, param, args, kwargs)

    def plotNLL(self, *args: Any, **kwargs: Any) -> Any:
        from .mcstudydata import plot_column

        return plot_column(self, "NLL", args, kwargs, symmetric=False)


def _fit_options(options: Any) -> list[RooCmdArg]:
    """The options of ``FitOptions(...)``: its arguments, or a keyword dictionary of them."""
    from .cmdargs import make

    found: list[RooCmdArg] = []
    for arg in options.args("FitOptions"):
        if isinstance(arg, dict):
            found.extend(make(key, value) for key, value in arg.items())
        elif isinstance(arg, RooCmdArg):
            found.append(arg)
    return found


def _generator(model: Any, observables: list[Any]) -> Any:
    from .generation.generate import Generator

    return Generator(model, observables)


def warn(obj: Any, text: str) -> None:
    log(obj, WARNING, "Generation", text)
