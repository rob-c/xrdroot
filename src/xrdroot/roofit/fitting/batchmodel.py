"""Independent parameter contexts require built-in analytic model operations."""

import math
from typing import Any

import numpy as np

from .. import evalerrors
from ..binning import RooAbsBinning, RooBinning, RooRangeBinning, RooUniformBinning
from ..categories import RooCategory
from ..data.datahist import RooDataHist
from ..functions import RooProduct
from ..pdfs.basic import RooGaussian
from ..pdfs.histcache import geometry
from ..pdfs.histfactory import FlexibleInterpVar, ParamHistFunc, RooBinWidthFunction
from ..pdfs.histpdf import RooHistFunc
from ..pdfs.prodpdf import RooProdPdf
from ..pdfs.realsum import RooRealSumPdf
from ..pdfs.shapes import RooPoisson
from ..pdfs.simultaneous import RooSimultaneous
from ..variables import RooConstVar, RooRealVar
from .binned import binned_part
from .nll import RooNLLVar

#: Only these built-in operations have independent parameter contexts.
TYPES = (
    RooCategory,
    RooConstVar,
    RooRealVar,
    RooProduct,
    RooGaussian,
    RooPoisson,
    RooHistFunc,
    FlexibleInterpVar,
    ParamHistFunc,
    RooBinWidthFunction,
    RooRealSumPdf,
    RooProdPdf,
    RooSimultaneous,
)
#: Their range metadata is fixed during an integrand call.
BINNINGS = (RooAbsBinning, RooBinning, RooRangeBinning, RooUniformBinning)


def node_supported(node: Any) -> bool:
    if not _native_node(node):
        return False
    if type(node) is RooRealVar:
        return type(node.getBinning()) in BINNINGS and finite(node.getVal())
    if type(node) is RooConstVar:
        return finite(node.compute({}))
    if type(node) is RooHistFunc:
        return _histogram_supported(node)
    return True


def constraints_supported(nll: Any) -> bool:
    for node in nll.constraints:
        names = nll._constrained(node) & node.dependents()
        if type(node) is RooGaussian:
            if names not in (
                frozenset(),
                frozenset([node.x.GetName()]),
                frozenset([node.mean.GetName()]),
            ):
                return False
        elif type(node) is RooPoisson:
            if names not in (frozenset(), frozenset([node.x.GetName()])):
                return False
        else:
            return False
    return True


def eligible(nll: Any, parameters: Any) -> bool:
    if type(nll) is not RooNLLVar or not _native_instance(nll):
        return False
    if not _configured(nll) or evalerrors.active():
        return False
    return (
        _parameter_types(nll, parameters)
        and _weights_supported(nll)
        and model_supported(nll, parameters)
    )


def model_supported(nll: Any, parameters: Any) -> bool:
    if not all(finite(value) for value in nll._global_values.values()):
        return False
    if not _binned_channels(nll):
        return False
    nodes = _model_nodes(nll)
    return (
        _bindings_supported(nodes, parameters)
        and all(node_supported(node) for node in nodes.values())
        and constraints_supported(nll)
    )


def clipped(parameters: Any, points: Any) -> Any:
    """Python min(max(value, low), high) keeps the input's sign when zeros tie."""
    given = {}
    for i, parameter in enumerate(parameters):
        values = np.asarray(points[:, i], dtype=np.float64)
        low, high = parameter.getMin(), parameter.getMax()
        values = np.where(values < low, low, values)
        given[parameter.GetName()] = np.where(values > high, high, values)
    return given


def _nonnegative_interval(node: Any, names: Any) -> Any:
    if type(node) is RooConstVar:
        value = node.getVal()
        return value, value
    if type(node) is RooRealVar:
        if node.GetName() in names:
            return node.getMin(), node.getMax()
        value = node.getVal()
        return value, value
    if type(node) is RooProduct:
        low, high = 1.0, 1.0
        for term in node.terms:
            first, last = _nonnegative_interval(term, names)
            if not 0 <= first <= last < math.inf:
                return -math.inf, math.inf
            low *= first
            high *= last
        return low, high
    return -math.inf, math.inf


def _unit_gaussian_bound(nll: Any, constraint: Any, names: Any) -> bool:
    if type(constraint.sigma) is not RooConstVar or constraint.sigma.getVal() != 1.0:
        return False
    axes = _gaussian_axes(nll, constraint)
    return axes is not None and _covered_domain(axes[0], axes[1], names)


def _poisson_bound(nll: Any, constraint: Any, names: Any) -> bool:
    if type(constraint.x) is not RooRealVar or constraint.x.GetName() in names:
        return False
    over = nll._constrained(constraint) & constraint.dependents()
    if over != frozenset([constraint.x.GetName()]):
        return False
    low, high = _nonnegative_interval(constraint.mean, names)
    if not 0 <= low <= high <= 1e6:
        return False
    count = nll._global_values.get(constraint.x.GetName(), constraint.x.getVal())
    # This full count range makes ROOT's incomplete-gamma tail underflow to zero.
    return bool(
        0 <= count <= 1e6
        and constraint.x.getMin() == 0.0
        and constraint.x.getMax() >= max(1e9, high + 100.0 * math.sqrt(high))
    )


def bounded_posterior(like: Any, parameters: Any) -> bool:
    """Only bounded likelihoods can reorder independent runs without moving warnings.

    Count kernels have a factor-four margin below the cancellation scales.
    Unit Gaussian constraints have normalization greater than one. The
    bound selects batching and never approximates an evaluated value.
    """
    nll = like.nll.func
    if not eligible(nll, parameters) or nll._global_values:
        return False
    if not _prior_supported(like.prior):
        return False
    weights = nll.w * nll.w if nll._weight_squared else nll.w
    if np.any(weights > 1e6):
        return False
    budget = _constraint_budget(nll, frozenset(p.GetName() for p in parameters))
    return budget is not None and _within_warning_bound(
        like.offset + nll._offset_value, len(weights) + budget
    )


def finite(value: Any) -> bool:
    try:
        return type(value) in (float, int, np.float64) and math.isfinite(value)
    except OverflowError:
        return False


def _native_instance(node: Any) -> bool:
    """Native node fields contain data; callbacks need their original call order."""
    return not any(callable(value) for value in node.__dict__.values())


def _native_node(node: Any) -> bool:
    if type(node) not in TYPES or not _native_instance(node):
        return False
    # Binned sums are evaluated raw, so ROOT's forceNumInt flag is irrelevant.
    if type(node) is RooRealSumPdf:
        return True
    return not getattr(node, "_force_num_int", False) and not getattr(node, "_norm_range", None)


def _histogram_supported(node: Any) -> bool:
    return node.order == 0 and type(node.data) is RooDataHist and geometry(node.data) is not None


def _configured(nll: Any) -> bool:
    return bool(nll.binned and not nll.rng and not nll.offset and not nll._copies)


def _parameter_types(nll: Any, parameters: Any) -> bool:
    return all(type(p) is RooRealVar and p.GetName() not in nll.columns for p in parameters)


def _weights_supported(nll: Any) -> bool:
    return bool(len(nll.w) and np.isfinite(nll.w).all() and not (nll.w < 0).any())


def _binned_channels(nll: Any) -> bool:
    parts = list(nll.pdf.channels.values()) if type(nll.pdf) is RooSimultaneous else [nll.pdf]
    return all(binned_part(part) is not None for part in parts)


def _model_nodes(nll: Any) -> dict[int, Any]:
    return {id(node): node for top in [nll.pdf, *nll.constraints] for node in top._walk()}


def _bindings_supported(nodes: dict[int, Any], parameters: Any) -> bool:
    """Contexts set names; scalar bindings set objects. Require one matching object per name."""
    counts: dict[str, int] = {}
    for node in nodes.values():
        if type(node) is RooRealVar:
            name = node.GetName()
            counts[name] = counts.get(name, 0) + 1
    return all(id(p) in nodes and counts.get(p.GetName()) == 1 for p in parameters)


def _gaussian_axes(nll: Any, constraint: Any) -> Any:
    over = nll._constrained(constraint) & constraint.dependents()
    if over == frozenset([constraint.x.GetName()]):
        return constraint.x, constraint.mean
    if over == frozenset([constraint.mean.GetName()]):
        return constraint.mean, constraint.x
    return None


def _covered_domain(variable: Any, other: Any, names: Any) -> bool:
    if type(variable) is not RooRealVar or type(other) not in (RooRealVar, RooConstVar):
        return False
    low, high = _nonnegative_interval(other, names)
    # For unit sigma and width >= 2, the minimum normalizing integral exceeds one.
    return bool(
        math.isfinite(low)
        and math.isfinite(high)
        and variable.getMin() <= low
        and variable.getMax() >= high
        and variable.getMax() - variable.getMin() >= 2.0
    )


def _prior_supported(prior: Any) -> bool:
    from ..pdfs.basic import RooUniform

    if prior is None:
        return True
    pdf = prior.func
    if type(pdf) is not RooUniform or pdf.getForceNumInt() or pdf._norm_range:
        return False
    return all(_wide_variable(node) for node in pdf.servers())


def _wide_variable(node: Any) -> bool:
    return (
        type(node) is RooRealVar and node_supported(node) and node.getMax() - node.getMin() >= 1.0
    )


def _constraint_budget(nll: Any, names: Any) -> int | None:
    """Constraint types have already passed the eligibility checks."""
    count = 0
    for constraint in nll.constraints:
        if type(constraint) is RooGaussian:
            valid = _unit_gaussian_bound(nll, constraint, names)
        else:
            valid = _poisson_bound(nll, constraint, names)
            count += 1
        if not valid:
            return None
    return count


def _within_warning_bound(offset: Any, budget: int) -> bool:
    return bool(math.isfinite(offset) and offset + budget * math.log(4.0) < math.log(1e10) - 1.0)
