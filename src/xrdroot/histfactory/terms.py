"""The pieces of a HistFactory channel's model, as ``HistoToWorkspaceFactoryFast`` makes them.

Each piece is made and imported into the channel's workspace, and the
workspace's copy is what is used next - ``emplace`` - or the workspace's
own, if it has one by that name - ``getOrCreate``: the observables from the
first histogram's axes, a ``RooHistFunc`` of each sample's nominal
histogram, Gaussian constraints of the systematics' ``alpha`` parameters,
the ``FlexibleInterpVar`` of a sample's overall systematics, the
``PiecewiseInterpolation`` of its shape systematics, its normalisation
factors.
"""

from __future__ import annotations

from typing import Any

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgList
from ..roofit.messages import INFO
from ..roofit.printing import g
from .model import _hf

__all__ = ["emplace", "get_or_create", "make_gaussian_constraint"]

#: The range of every systematic's ``alpha``.
ALPHA_LOW, ALPHA_HIGH = -5.0, 5.0


def emplace(ws: Any, kind: Any, name: str, *args: Any) -> Any:
    """``kind(name, name, *args)`` imported - reusing what the workspace has - and its copy."""
    ws.Import(kind(name, name, *args), RooCmdArg("RecycleConflictNodes"))
    return ws.arg(name)


def get_or_create(ws: Any, kind: Any, name: str, *args: Any) -> Any:
    """The workspace's ``name``, or ``kind(name, name, *args)`` made, imported silently."""
    found = ws.obj(name)
    if found is not None:
        return found
    ws.Import(kind(name, name, *args), RooCmdArg("RecycleConflictNodes"), RooCmdArg("Silence"))
    return ws.obj(name)


def observables(names: list[str], hist: Any, ws: Any) -> RooArgList:
    """``createObservables``: a variable per axis of the histogram - its range and bins - if the
    workspace has none by that name."""
    from ..roofit.binning import RooBinning
    from ..roofit.variables import RooRealVar

    found = RooArgList()
    axes = [hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis()]
    for name, axis in zip(names, axes):
        if ws.var(name) is None:
            obs = emplace(ws, RooRealVar, name, axis.GetXmin(), axis.GetXmax())
            if axis.GetTitle():
                obs.SetTitle(axis.GetTitle())
            obs.setBins(axis.GetNbins())
            if axis.IsVariableBinSize():
                edges = [axis.GetBinLowEdge(i) for i in range(1, axis.GetNbins() + 2)]
                obs.setBinning(RooBinning(axis.GetNbins(), edges))
        found.add(ws.var(name))
    return found


def expected_hist_func(hist: Any, ws: Any, prefix: str, obs: Any) -> Any:
    """``MakeExpectedHistFunc``: the ``RooHistFunc`` ``<prefix>_Hist_alphanominal`` of the
    histogram, over the observables."""
    from ..roofit.data.datahist import RooDataHist
    from ..roofit.pdfs.histpdf import RooHistFunc

    _hf(INFO, f"processing hist {hist.GetName()}")
    name = f"{prefix}_Hist_alphanominal"
    data = RooDataHist(f"{name}DHist", "", obs, hist)
    return emplace(ws, RooHistFunc, name, obs, data, 0)


def make_gaussian_constraint(param: Any, ws: Any, uniform: bool, names: list[str]) -> None:
    """``makeGaussianConstraint``: ``<param>Constraint``, a Gaussian of ``nom_<param>`` - a
    global observable - about the parameter, of width one (a hundred, if uniform)."""
    from ..roofit.pdfs.basic import RooGaussian
    from ..roofit.variables import RooRealVar

    name = param.GetName()
    constraint = f"{name}Constraint"
    if ws.pdf(constraint) is not None:
        return
    sigma = 100.0 if uniform else 1.0
    if uniform:
        _hf(INFO, f"Added a uniform constraint for {name} as a Gaussian constraint with a very "
            "large sigma ")  # fmt: skip
    names.append(constraint)
    var = ws.var(name)
    nominal = emplace(ws, RooRealVar, f"nom_{name}", 0.0, -10.0, 10.0)
    nominal.setConstant()
    emplace(ws, RooGaussian, constraint, var, nominal, sigma)
    var.setError(sigma)
    ws.set("globalObservables").add(nominal)


def constraint_terms(ws: Any, meas: Any, prefix: str, interp_name: str, systematics: list[Any],
                     names: list[str]) -> None:  # fmt: skip
    """``AddConstraintTerms``: each overall systematic's parameter and constraint - Gaussian,
    or as the measurement says - and their ``FlexibleInterpVar`` ``interp_name``, code 5."""
    from ..roofit.pdfs.histfactory import FlexibleInterpVar
    from ..roofit.variables import RooConstVar, RooRealVar

    params, lows, highs = [], [], []
    for sys in systematics:
        name = sys.GetName()
        if name in meas.GetNoSyst():
            _hf(INFO, f"HistoToWorkspaceFast::AddConstraintTerm - skip systematic {name}")
            continue
        if name in meas.GetGammaSyst():
            found = _gamma_constraint(ws, name, meas.GetGammaSyst()[name], names)
            if found is None:
                continue
            params.append(found)
        else:
            alpha = get_or_create(ws, RooRealVar, prefix + name, 0, ALPHA_LOW, ALPHA_HIGH)
            make_gaussian_constraint(alpha, ws, name in meas.GetUniformSyst(), names)
            if name not in meas.GetLogNormSyst():
                params.append(alpha)
            else:
                params.append(_lognormal(ws, name, meas.GetLogNormSyst()[name], alpha))
        lows.append(sys.GetLow())
        highs.append(sys.GetHigh())
    if systematics:
        interp = FlexibleInterpVar(interp_name, "", params, 1.0, lows, highs)
        interp.setAllInterpCodes(5)
        ws.Import(interp)
    else:
        emplace(ws, RooConstVar, interp_name, 1.0)


def _gamma_constraint(ws: Any, name: str, relerr: float, names: list[str]) -> Any:
    """A Gamma-constrained systematic's ``alphaOfBeta_<name>``, ``sqrt(tau) (beta - 1)``, and
    ``beta_<name>Constraint``: a ``RooGamma`` of shape ``nom_beta_<name> + 1`` - a global
    observable - and scale ``1 / tau`` - none, for no uncertainty."""
    from ..roofit.functions import RooAddition, RooPolyVar
    from ..roofit.pdfs.basic import ref
    from ..roofit.pdfs.gamma import RooGamma
    from ..roofit.variables import RooRealVar

    if relerr <= 0:
        _hf(INFO, "HistoToWorkspaceFast::AddConstraintTerm - zero uncertainty assigned - skip "
            f"systematic  {name}")  # fmt: skip
        return None
    tau, root = 1.0 / (relerr * relerr), 1.0 / relerr
    beta = emplace(ws, RooRealVar, f"beta_{name}", 1.0, 0.0, 10.0)
    nominal = emplace(ws, RooRealVar, f"nom_{beta.GetName()}", tau, 0.0, 10.0)
    theta = emplace(ws, RooRealVar, f"theta_{name}", 1.0 / tau)
    alpha = emplace(ws, RooPolyVar, f"alphaOfBeta_{name}", beta,
                    RooArgList([ref(-root), ref(root)]))
    kappa = emplace(ws, RooAddition, f"k_{nominal.GetName()}", RooArgList([nominal, ref(1.0)]))
    gamma = emplace(ws, RooGamma, f"{beta.GetName()}Constraint", beta, kappa, theta, ref(0.0))
    names.append(gamma.GetName())
    nominal.setConstant(True)
    ws.set("globalObservables").add(nominal)
    _hf(INFO, f"Added a gamma constraint for {name}")
    return alpha


def _lognormal(ws: Any, name: str, relerr: float, alpha: Any) -> Any:
    """``alphaOfBeta_<name>``: ``tau (kappa^alpha - 1)`` for a log-normal systematic."""
    from ..roofit.functions import RooFormulaVar
    from ..roofit.variables import RooRealVar

    made = emplace(ws, RooFormulaVar, f"alphaOfBeta_{name}", "x[0]*(pow(x[1],x[2])-1.)",
                   RooArgList([emplace(ws, RooRealVar, f"tau_{name}", 1.0 / relerr),
                               emplace(ws, RooRealVar, f"kappa_{name}", 1.0 + relerr), alpha]))
    _hf(INFO, f"Added a log-normal constraint for {name}")
    return made


def interpolation_parameters(systematics: list[Any], ws: Any) -> RooArgList:
    """``makeInterpolationParameters``: ``alpha_<name>`` of each shape systematic."""
    from ..roofit.variables import RooRealVar

    return RooArgList([get_or_create(ws, RooRealVar, f"alpha_{one.GetName()}", ALPHA_LOW,
                                     ALPHA_HIGH) for one in systematics], "alpha_Hist")  # fmt: skip


def lin_interp(params: Any, nominal: Any, ws: Any, systematics: list[Any], prefix: str,
               obs: Any) -> Any:  # fmt: skip
    """``makeLinInterp``: the nominal function moved to each shape systematic's low and high
    histograms - a ``PiecewiseInterpolation`` of code 4, positive definite."""
    from ..roofit.data.datahist import RooDataHist
    from ..roofit.pdfs.histfactory import PiecewiseInterpolation
    from ..roofit.pdfs.histpdf import RooHistFunc

    lows, highs = [], []
    for j, sys in enumerate(systematics):
        stem = f"{prefix}_{j}"
        low = RooDataHist(f"{stem}lowDHist", "", obs, sys.GetHistoLow())
        high = RooDataHist(f"{stem}highDHist", "", obs, sys.GetHistoHigh())
        lows.append(RooHistFunc(f"{stem}low", "", obs, low, 0))
        highs.append(RooHistFunc(f"{stem}high", "", obs, high, 0))
    interp = PiecewiseInterpolation(prefix, "", nominal, lows, highs, params)
    interp.setPositiveDefinite()
    interp.setAllInterpCodes(4)
    interp.setForceNumInt(True)
    ws.Import(interp, RooCmdArg("RecycleConflictNodes"))
    return ws.arg(prefix)


def norm_factor(ws: Any, channel: str, epsilon: str, sample: Any) -> Any:
    """``CreateNormFactor``: ``<sample>_<channel>_scaleFactors``, the product of the overall
    systematics' factor and each normalisation factor - the terms, for the lumi to join."""
    from ..roofit.variables import RooRealVar

    terms = [ws.arg(epsilon)]
    names = []
    for norm in sample.GetNormFactorList():
        name = norm.GetName()
        if ws.obj(name) is None:
            _hf(INFO, f"making normFactor: {norm.GetName()}")
            emplace(ws, RooRealVar, name, norm.GetVal(), norm.GetLow(), norm.GetHigh())
            ws.var(name).setError(0)
        names.append((name, f"[{g(norm.GetVal())},{g(norm.GetLow())},{g(norm.GetHigh())}]"))
        terms.append(ws.arg(name))
    for name, rng in names:
        if [n for n, _ in names].count(name) > 1:
            _hf(INFO, f'<NormFactor Name ="{name}"> is duplicated for <Sample Name="'
                f'{sample.GetName()}">, but only one factor will be included.  \n Instead, define '
                f'something like\n\t<Function Name="{name}Squared" Expression="{name}*{name}" '
                f'Var="{name}{rng}"> \nin your top-level XML\'s <Measurement> entry and use '
                f'<NormFactor Name="{name}Squared" in your channel XML file.')  # fmt: skip
    return terms
