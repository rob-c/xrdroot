"""What a :class:`~.mcstudy.RooMCStudy` found, as a dataset, and its plots.

The dataset has a column per fitted parameter, ``<p>err`` for its error,
``<p>pull`` for ``(fitted - generated) / error`` (``RooPullVar``), ``NLL``
and ``ngen``. A plot of one column is framed round its values -
``AutoRange(data, 0.2)``, symmetric about the mean for a pull - and a pull
plot may carry a Gaussian fitted to it, as ``fitGaussToPulls`` does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .cmdargs import RooCmdArg, commands
from .variables import RooRealVar

__all__ = ["parameter_data", "plot_column", "plot_pull"]


def _pull(value: float, error: float, generated: float) -> float:
    return (value - generated) / error if error > 0 else 0.0


def parameter_data(study: Any) -> Any:
    from .data.dataset import RooDataSet

    generated = {p.GetName(): v for p, v in study.gen_init}
    variables = []
    pulled = [par for par in study.fit_params if par.hasError(False)]
    for par in study.fit_params:
        variables.append(RooRealVar(par.GetName(), par.GetTitle(), par.getMin(), par.getMax()))
    for par in study.fit_params:
        variables.append(RooRealVar(f"{par.GetName()}err", f"{par.GetTitle()} Error", -1e30, 1e30))
        if par in pulled:
            variables.append(
                RooRealVar(f"{par.GetName()}pull", f"{par.GetTitle()} Pull", -1e30, 1e30)
            )
    variables += [
        RooRealVar("NLL", "-log(Likelihood)", -1e30, 1e30),
        RooRealVar("ngen", "number of generated events", -1e30, 1e30),
    ]
    data = RooDataSet(
        f"fitParData_{study.fit_model.GetName()}", "Fit Parameters DataSet", variables
    )
    columns: dict[str, list[float]] = {v.GetName(): [] for v in variables}
    for row in study.rows:
        for key, value in row.items():
            columns[key].append(value)
        for par in pulled:
            name = par.GetName()
            columns[f"{name}pull"].append(
                _pull(row[name], row[f"{name}err"], generated.get(name, 0.0))
            )
    data.add_columns({k: np.array(v, dtype=np.float64) for k, v in columns.items()})
    return data


def _auto(
    values: np.ndarray[Any, Any], var: Any, margin: float, symmetric: bool
) -> tuple[float, float]:
    """``RooAbsData::getRange``: the values' extent and a margin, symmetric about the mean if
    asked."""
    low, high = float(values.min()), float(values.max())
    if symmetric:
        mean = float(np.mean(values))
        delta = max(high - mean, mean - low) * (1 + margin)
        low, high = mean - delta, mean + delta
    else:
        delta = margin * (high - low)
        low, high = low - delta, high + delta
    return max(low, var.getMin()), min(high, var.getMax())


def plot_column(
    study: Any, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], symmetric: bool
) -> Any:
    """A frame of one column of the study's dataset, and that column plotted on it."""
    data = study.fitParDataSet()
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    var = data.variable(name)
    if symmetric:
        var = RooRealVar(name, var.GetTitle(), -100, 100)
        var.setBins(100)
    if "Range" in options:
        low, high = float(options.get("Range", 0)), float(options.get("Range", 1))
    else:
        low, high = _auto(data.column(name), var, 0.2, symmetric)
    bins = int(options.get("Bins", 0, var.getBins()))
    frame = var.frame(low, high, bins)
    rest = [one for one in options.given if one.name not in ("Bins", "Range", "FitGauss")]
    data.plotOn(frame, *rest)
    return frame


def plot_pull(study: Any, param: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``plotPull``: the pulls, framed symmetrically, with a Gaussian fitted if ``FitGauss``."""
    options = commands([a for a in args if isinstance(a, RooCmdArg)], kwargs)
    frame = plot_column(study, f"{param.GetName()}pull", args, kwargs, symmetric=True)
    if options.get("FitGauss", 0, False):
        _fit_gauss(frame, study.fitParDataSet())
    return frame


def _fit_gauss(frame: Any, data: Any) -> None:
    from .formatting import format_var
    from .workspace import RooWorkspace

    w = RooWorkspace("")
    w.Import(frame.getPlotVar(), Silence=True)
    name = frame.getPlotVar().GetName()
    w.factory(f"Gaussian::pullGauss({name}, pullMean[0.0, -10.0, 10.0], pullSigma[1.0, 0.1, 5.0])")
    mean, sigma, gauss = w.var("pullMean"), w.var("pullSigma"), w.pdf("pullGauss")
    gauss.fitTo(data, RooCmdArg("Minos", False), RooCmdArg("PrintLevel", -1))
    gauss.plotOn(frame)
    label = (
        f"Fit parameters:\n#mu: {format_var(mean, 2, 'ELU')}\n#sigma: {format_var(sigma, 2, 'ELU')}"
    )
    mean.setConstant(True)
    sigma.setConstant(True)
    gauss.paramOn(frame, RooCmdArg("Label", label), RooCmdArg("Layout", 0.60, 0.9, 0.9))
