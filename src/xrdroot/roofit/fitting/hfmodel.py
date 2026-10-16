"""A native fit has no model callback that could change cached structure between FCN calls."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..arg import Proxy
from ..categories import RooCategory
from ..functions import RooProduct
from ..pdfs.basic import RooGaussian
from ..pdfs.gamma import RooGamma
from ..pdfs.histfactory import FlexibleInterpVar, ParamHistFunc, RooBinWidthFunction
from ..pdfs.histpdf import RooHistFunc
from ..pdfs.prodpdf import RooProdPdf
from ..pdfs.realsum import RooRealSumPdf
from ..pdfs.shapes import RooPoisson
from ..pdfs.simultaneous import RooSimultaneous
from ..variables import RooConstVar, RooRealVar
from . import hfdata, hfnative

#: Walk both live fields and proxies: low-level collection edits need not bump graph versions.
FIELDS = {
    RooRealVar: (), RooConstVar: (), RooCategory: (), RooProduct: ("terms",),
    RooRealSumPdf: ("funcs", "coefs"), FlexibleInterpVar: ("params",),
    ParamHistFunc: ("vars", "gammas"), RooBinWidthFunction: ("hist",),
    RooHistFunc: ("observables", "data", "_hist_vars", "_given"),
    RooProdPdf: ("pdfs",), RooSimultaneous: ("index", "channels"),
    RooGaussian: ("x", "mean", "sigma"), RooPoisson: ("x", "mean"),
    RooGamma: ("x", "gamma", "beta", "mu"),
}
#: Bound planning work; other graphs use the ordinary evaluator.
MAX_NODES = 128


def _items(value: Any) -> list[Any]:
    if type(value) in hfdata.COLLECTIONS:
        return hfdata.members(value)
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise TypeError("Native model field names must be strings.")
        return list(value.values())
    return value if type(value) is list else [value]


def _proxies(node: Any) -> list[Any]:
    if type(node._proxies) is not list:
        raise TypeError("Native proxy storage must be a list.")
    children = []
    for proxy in node._proxies:
        if not hfnative.instance(proxy, (Proxy,)) or type(proxy.many) is not bool:
            raise TypeError("Native model proxies cannot run callbacks.")
        children.extend(hfdata.members(proxy.target) if proxy.many else [proxy.target])
    return children


def _fields(node: Any) -> bool:
    kind = type(node)
    checks: dict[type[Any], Callable[[Any], bool]] = {
        RooRealVar: hfdata.variable, RooConstVar: lambda v: hfnative.number(v._val),
        RooCategory: lambda v: type(v._index) is int and hfdata.mapping(
            v._states, lambda i: type(i) is int),
        FlexibleInterpVar: hfdata.interpolation, ParamHistFunc: lambda v: hfdata.edges(v._edges),
    }
    return kind not in checks or bool(checks[kind](node))


def _metadata(node: Any) -> None:
    if type(node._name) is not str or not hfdata.sequence(
            node._attributes, set, lambda value: type(value) is str):
        raise TypeError("Native model attributes must be strings.")
    if not hfdata.mapping(node._strings, lambda value: type(value) is str):
        raise TypeError("Native model string attributes must be strings.")


def _node(node: Any) -> list[Any]:
    if type(node) in hfdata.DATA:
        if not hfdata.store(node):
            raise TypeError("A compiled fit requires native data storage.")
        return hfdata.members(node._vars)
    if not hfnative.instance(node, tuple(FIELDS)) or not _fields(node):
        raise TypeError("A compiled fit requires native model nodes.")
    _metadata(node)
    children = _proxies(node)
    for field in FIELDS[type(node)]:
        children.extend(_items(getattr(node, field)))
    return children


def _walk(roots: list[Any]) -> bool:
    todo = roots[:]
    seen: set[int] = set()
    while todo:
        node = todo.pop()
        if id(node) in seen:
            continue
        if len(seen) >= MAX_NODES:
            return False
        seen.add(id(node))
        todo.extend(_node(node))
    return True


def model(nll: Any, parameters: list[Any]) -> bool:
    """Recheck mutable model fields at every fit, including edits made since the last fit."""
    try:
        if nll._copies is not None and (type(nll._copies) is not dict or nll._copies):
            return False
        if nll.rng or type(nll.constraints) is not list:
            return False
        if not hfdata.mapping(nll._global_values, hfnative.number):
            return False
        return _walk([nll.pdf, nll.data, *nll.constraints, *parameters])
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
