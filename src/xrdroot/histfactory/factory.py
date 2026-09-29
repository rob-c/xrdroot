"""``HistoToWorkspaceFactoryFast``: a channel's model, then the channels' simultaneous one.

A channel's workspace has the luminosity and its constraint, then for each
sample its overall systematics' ``FlexibleInterpVar``, its normalisation
factors, its nominal ``RooHistFunc`` - moved by its shape systematics -
its statistical gammas' ``ParamHistFunc``, its shape factors; the channel's
statistical constraints; the ``RooRealSumPdf`` of the samples' shapes times
their scale factors; and ``model_<channel>``, the constraints times it,
with the observed data - each named, set and said as ROOT's factory names,
sets and says it.
"""

from __future__ import annotations

from typing import Any

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgList, RooArgSet
from ..roofit.messages import INFO, PROGRESS, WARNING
from .assemble import finish as _finish
from .assemble import log_fatal
from .assemble import total_expected as _total_expected
from .model import _hf
from .systematics import HistFactoryError
from .terms import constraint_terms, emplace, get_or_create

__all__ = ["HistoToWorkspaceFactoryFast"]


class HistoToWorkspaceFactoryFast:
    """Builds HistFactory's workspaces from a measurement."""

    def __init__(self, measurement: Any, config: Any = None) -> None:
        self._fix = list(measurement.GetConstantParams())
        self._values = dict(measurement.GetParamValues())
        self._lumi = measurement.GetLumi()
        self._lumi_error = measurement.GetLumi() * measurement.GetLumiRelErr()
        self._functions = measurement.GetPreprocessFunctions()
        self._obs: list[str] = []

    def SetFunctionsToPreprocess(self, functions: list[str]) -> None:
        self._functions = list(functions)

    def MakeSingleChannelModel(self, measurement: Any, channel: Any) -> Any:
        """The channel's workspace, configured for the measurement."""
        from .combine import configure_for_measurement

        ws = self.MakeSingleChannelWorkspace(measurement, channel)
        configure_for_measurement(f"model_{channel.GetName()}", ws, measurement)
        return ws

    def MakeCombinedModel(self, names: Any, workspaces: Any = None) -> Any:
        """The combined workspace - of channel workspaces made already, or of a measurement's
        channels, made now and configured for it."""
        from .combine import combined_model, configure_for_measurement

        if workspaces is not None:
            return combined_model(self, list(names), list(workspaces))
        measurement = names
        factory = HistoToWorkspaceFactoryFast(measurement)
        made, channels = [], []
        for channel in measurement.GetChannels():
            if not channel.CheckHistograms():
                log_fatal(f"MakeModelAndMeasurementsFast: Channel: {channel.GetName()} has "
                          "uninitialized histogram pointers")  # fmt: skip
            channels.append(channel.GetName())
            made.append(factory.MakeSingleChannelModel(measurement, channel))
        ws = combined_model(factory, channels, made)
        configure_for_measurement("simPdf", ws, measurement)
        return ws

    def _set_values_and_constants(self, ws: Any) -> None:
        """The measurement's parameter values, then its constants, set in ``ws`` - and said."""
        from ..roofit.messages import ERROR
        from ..roofit.printing import g

        for name, value in sorted(self._values.items()):
            var = ws.var(name)
            if var is None:
                _hf(ERROR, f"could not find variable {name} could not set its value")
                continue
            var.setVal(value)
            _hf(INFO, f"setting {name} to the value: {g(value)}")
        for name in self._fix:
            var = ws.var(name)
            if var is None:
                _hf(ERROR, f"could not find variable {name} could not set it to constant")
                continue
            var.setConstant()
            _hf(INFO, f"setting {name} constant")

    def _lumi_terms(self, ws: Any, names: list[str]) -> None:
        """``Lumi`` - and its Gaussian constraint about ``nominalLumi``, if it is uncertain."""
        from ..roofit.pdfs.basic import RooGaussian
        from ..roofit.variables import RooRealVar

        lumi = get_or_create(ws, RooRealVar, "Lumi", self._lumi, 0.0, 10 * self._lumi)
        if self._lumi_error != 0:
            nominal = emplace(ws, RooRealVar, "nominalLumi", self._lumi, 0.0,
                              self._lumi + 10.0 * self._lumi_error)  # fmt: skip
            emplace(ws, RooGaussian, "lumiConstraint", lumi, nominal, self._lumi_error)
            ws.var("Lumi").setError(self._lumi_error / self._lumi)
            ws.var("nominalLumi").setConstant()
            ws.defineSet("globalObservables", "nominalLumi")
            names.append("lumiConstraint")
        else:
            ws.var("Lumi").setConstant()
            ws.defineSet("globalObservables", RooArgSet())

    def _observable_names(self, hist: Any, channel: str) -> None:
        """``GuessObsNameVec``: ``obs_x_<channel>`` and so on, one per dimension."""
        self._obs = [f"obs_{axis}_{channel}" for axis in "xyz"[: hist.GetDimension()]]

    def _sample(self, ws: Any, meas: Any, channel: str, sample: Any, state: dict[str, Any]) -> None:
        """One sample: its scale factors and its shape's factors, into ``state``."""
        from ..roofit.functions import RooProduct
        from .terms import (
            expected_hist_func,
            interpolation_parameters,
            lin_interp,
            make_gaussian_constraint,
            norm_factor,
            observables,
        )

        epsilon = f"{sample.GetName()}_{channel}_epsilon"
        constraint_terms(ws, meas, "alpha_", epsilon, sample.GetOverallSysList(),
                         state["constraints"])  # fmt: skip
        terms = norm_factor(ws, channel, epsilon, sample)
        nominal = sample.GetHisto()
        prefix = f"{sample.GetName()}_{channel}"
        obs = observables(self._obs, nominal, ws)
        func = expected_hist_func(nominal, ws, prefix, obs)
        shapes: list[Any] = []
        if not sample.GetHistoSysList():
            _hf(INFO, f"{prefix} has no variation histograms ")
            shapes.append(func)
        else:
            params = interpolation_parameters(sample.GetHistoSysList(), ws)
            for param, sys in zip(params, sample.GetHistoSysList()):
                make_gaussian_constraint(param, ws, sys.GetName() in meas.GetUniformSyst(),
                                         state["constraints"])  # fmt: skip
            shapes.append(lin_interp(params, func, ws, sample.GetHistoSysList(),
                                     f"{prefix}_Hist_alpha", obs))  # fmt: skip
        shapes[0].SetTitle(nominal.GetTitle() if nominal.GetTitle() else sample.GetName())
        if sample.GetStatError().GetActivate():
            shapes.append(self._stat_error(ws, channel, sample, state))
        for factor in sample.GetShapeFactorList():
            shapes.append(self._shape_factor(ws, channel, sample, factor))
        if sample.GetShapeSysList():
            shapes.extend(self._shape_sys(ws, channel, sample, state))
        lumi = ws.arg("Lumi")
        if not sample.GetNormalizeByTheory():
            lumi.setVal(meas.GetLumi())
        made = RooProduct(f"{prefix}_scaleFactors", f"{prefix}_scaleFactors", [*terms, lumi])
        ws.Import(made, RooCmdArg("RecycleConflictNodes"))
        state["scales"].append(ws.arg(made.GetName()))
        state["funcs"].append(shapes)

    def _stat_error(self, ws: Any, channel: str, sample: Any, state: dict[str, Any]) -> Any:
        """The sample's statistical uncertainty in the channel's, and ``mc_stat_<channel>``."""
        from ..roofit.pdfs.histfactory import ParamHistFunc
        from .stat import absolute_uncertainty

        _hf(INFO, f"Sample: {sample.GetName()} to be included in Stat Error for channel {channel}")
        name = f"{sample.GetName()}_{channel}_StatAbsolUncert"
        nominal = sample.GetHisto()
        given = sample.GetStatError().GetErrorHist()
        if given is None:
            _hf(INFO, f"Making Statistical Uncertainty Hist for  Channel: {channel} Sample: "
                f"{sample.GetName()}")  # fmt: skip
            error = absolute_uncertainty(name, nominal)
        else:
            error = given.Clone()
            _hf(INFO, f"Using external histogram for Stat Errors for \tChannel: {channel}\t"
                f"Sample: {sample.GetName()}\tError Histogram: {error.GetName()}")  # fmt: skip
            error.Multiply(nominal)
            error.SetName(name)
        state["stat"].append((nominal, error))
        func_name = f"mc_stat_{channel}"
        if ws.function(func_name) is None:
            obs = RooArgList([ws.var(n) for n in self._obs])
            gammas = create_param_set(ws, f"gamma_stat_{channel}", obs, 0.0, 10.0)
            ws.Import(ParamHistFunc(func_name, func_name, obs, gammas),
                      RooCmdArg("RecycleConflictNodes"))  # fmt: skip
        return ws.function(func_name)

    def _shape_factor(self, ws: Any, channel: str, sample: Any, factor: Any) -> Any:
        """``<channel>_<name>_shapeFactor``: free gammas, one per bin, from an initial shape."""
        from ..roofit.pdfs.histfactory import ParamHistFunc

        _hf(INFO, f"Sample: {sample.GetName()} in channel: {channel} to be include a "
            "ShapeFactor.")  # fmt: skip
        name = f"{channel}_{factor.GetName()}_shapeFactor"
        if ws.function(name) is None:
            obs = RooArgList([ws.var(n) for n in self._obs])
            gammas = create_param_set(ws, f"gamma_{factor.GetName()}", obs)
            for gamma in gammas:
                gamma.setVal(factor.GetVal())
                gamma.setMin(factor.GetLow())
                gamma.setMax(factor.GetHigh())
            made = ParamHistFunc(name, name, obs, gammas)
            if factor.GetInitialShape() is not None:
                from ..errors import UnsupportedFeatureError

                raise UnsupportedFeatureError(
                    f"the shape factor {factor.GetName()} has an initial shape, which "
                    "ParamHistFunc::setShape takes; xrdroot's ParamHistFunc does not yet"
                )
            if factor.IsConstant():
                _hf(INFO, f"Setting Shape Factor: {factor.GetName()} to be constant")
                made.setConstant(True)
            ws.Import(made, RooCmdArg("RecycleConflictNodes"))
        return ws.function(name)

    def _shape_sys(self, ws: Any, channel: str, sample: Any, state: dict[str, Any]) -> list[Any]:
        """Each ``<channel>_<name>_ShapeSys``: gammas constrained by the relative errors."""
        from ..roofit.pdfs.histfactory import ParamHistFunc
        from .stat import gamma_constraints, hist_values

        names = []
        for shape in sample.GetShapeSysList():
            _hf(INFO, f"Sample: {sample.GetName()} in channel: {channel} to include a ShapeSys.")
            name = f"{channel}_{shape.GetName()}_ShapeSys"
            names.append(name)
            if ws.function(name) is None:
                obs = RooArgList([ws.var(n) for n in self._obs])
                gammas = create_param_set(ws, f"gamma_{shape.GetName()}", obs, 0.0, 10.0)
                ws.Import(ParamHistFunc(name, name, obs, gammas),
                          RooCmdArg("RecycleConflictNodes"))  # fmt: skip
            func = ws.function(name)
            terms, globs = gamma_constraints(list(func.paramList()),
                                             hist_values(shape.GetErrorHist()), 0.0,
                                             shape.GetConstraintType())  # fmt: skip
            self._constrain(ws, terms, globs, state)
        return [ws.function(n) for n in names]

    def MakeSingleChannelWorkspace(self, measurement: Any, channel: Any) -> Any:
        """The channel's workspace: its model, its data, its ``ModelConfig``."""
        from ..roofit.workspace import RooWorkspace
        from ..roostats.modelconfig import ModelConfig

        template = self._template(channel)
        name = channel.GetName()
        self._observable_names(template, name)
        _hf(PROGRESS, f"\n-----------------------------------------\n\tStarting to process '{name}'"
            f" channel with {len(self._obs)} observables\n"
            "-----------------------------------------\n")  # fmt: skip
        ws = RooWorkspace(name, f"{name} workspace")
        config = ModelConfig("ModelConfig", ws)
        for func in self._functions:
            _hf(INFO, f"will preprocess this line: {func}")
            ws.factory(func)
            ws.Print()
        state: dict[str, Any] = {"constraints": [], "funcs": [], "scales": [], "stat": []}
        self._lumi_terms(ws, state["constraints"])
        for sample in channel.GetSamples():
            self._sample(ws, measurement, name, sample, state)
        if state["stat"]:
            self._stat_constraints(ws, channel, state)
        _total_expected(ws, f"{name}_model", state["scales"], state["funcs"])
        for param in measurement.GetConstantParams():
            var = ws.var(param)
            if var is None:
                _hf(WARNING, f"could not find variable {param} could not set it to constant")
            else:
                var.setConstant()
        _finish(ws, config, channel, self._obs, state)
        return ws

    def _template(self, channel: Any) -> Any:
        """The first sample's histogram - collected, if the channel has not been yet."""
        if not channel.GetSamples():
            raise HistFactoryError(f"HistFactory - the channel {channel.GetName()} has no sample")
        template = channel.GetSamples()[0].GetHisto()
        if template is None:
            channel.CollectHistograms()
            template = channel.GetSamples()[0].GetHisto()
        if not channel.CheckHistograms():
            log_fatal(f"MakeSingleChannelWorkspace: Channel: {channel.GetName()} has "
                      "uninitialized histogram pointers")  # fmt: skip
        return template

    def _stat_constraints(self, ws: Any, channel: Any, state: dict[str, Any]) -> None:
        """The channel's statistical gammas constrained by its relative uncertainty."""
        from .stat import gamma_constraints, hist_values, relative_uncertainty

        name = channel.GetName()
        relative = relative_uncertainty(f"{name}_StatUncert_RelErr", state["stat"])
        func = ws.function(f"mc_stat_{name}")
        names = ",".join(one.GetName() for one in func.paramList())
        _hf(INFO, f"About to create Constraint Terms from: {func.GetName()} params: ({names})")
        config = channel.GetStatErrorConfig()
        kind = config.GetConstraintType()
        _hf(INFO, f"Using {'Gaussian' if kind == 0 else 'Poisson'} StatErrors in channel: {name}")
        terms, globs = gamma_constraints(list(func.paramList()), hist_values(relative),
                                         config.GetRelErrorThreshold(), kind)  # fmt: skip
        self._constrain(ws, terms, globs, state)

    @staticmethod
    def _constrain(ws: Any, terms: list[Any], globs: list[Any], state: dict[str, Any]) -> None:
        """The constraints imported and listed, their global observables named global."""
        for term in terms:
            ws.Import(term, RooCmdArg("RecycleConflictNodes"))
            state["constraints"].append(term.GetName())
        for one in globs:
            ws.set("globalObservables").add(ws.var(one.GetName()))


def create_param_set(ws: Any, prefix: str, obs: Any, low: Any = None, high: Any = None) -> Any:
    """``ParamHistFunc::createParamSet``: ``<prefix>_bin_<i>`` - ``_<i>_<j>`` in two dimensions,
    and so on - one per bin, imported, each one free from zero (and to ``high``)."""
    from ..roofit.variables import RooRealVar

    counts = [one.numBins() for one in obs]
    names = []
    if len(counts) == 1:
        names = [f"{prefix}_bin_{i}" for i in range(counts[0])]
    elif len(counts) == 2:
        names = [f"{prefix}_bin_{i}_{j}" for j in range(counts[1]) for i in range(counts[0])]
    else:
        names = [f"{prefix}_bin_{i}_{j}_{k}" for k in range(counts[2]) for j in range(counts[1])
                 for i in range(counts[0])]  # fmt: skip
    found = RooArgList()
    for name in names:
        gamma = RooRealVar(name, name, 1.0)
        gamma.setMin(0.0)
        gamma.setConstant(False)
        ws.Import(gamma, RooCmdArg("RecycleConflictNodes"))
        found.add(ws.arg(name))
    for gamma in found if low is not None else ():
        gamma.setMin(low)
        gamma.setMax(high)
    return found
