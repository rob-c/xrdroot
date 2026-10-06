"""The nodes of a model read from a file: densities and functions, by their proxies.

Most classes are their proxies, in their constructor's order - a
``RooGaussian`` is ``x``, ``mean`` and ``sigma`` - and :data:`SIMPLE` says
which; the rest have makers of their own for the lists, flags and numbers
they carry. A ``RooHistFunc`` and a ``RooBinWidthFunction`` read back list
their proxy twice, as ROOT's do - registered by the streamer and again from
the file's list of proxies - so they print as ROOT prints them.
"""

from __future__ import annotations

from typing import Any

from ..arg import Proxy
from ..cmdargs import RooCmdArg
from .build import MAKERS, Builder, maker, name_of, title_of
from .stream import Streamed

__all__ = ["SIMPLE"]

#: Classes made of their proxies alone: the engine's module, and the members in order.
SIMPLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "RooGaussian": ("pdfs.basic", ("x", "mean", "sigma")),
    "RooExponential": ("pdfs.basic", ("x", "c")),
    "RooUniform": ("pdfs.basic", ("x",)),
    "RooLandau": ("pdfs.shapes", ("x", "mean", "sigma")),
    "RooLognormal": ("pdfs.shapes", ("x", "m0", "k")),
    "RooBreitWigner": ("pdfs.shapes", ("x", "mean", "width")),
    "RooExtendPdf": ("pdfs.extend", ("_pdf", "_n")),
    "RooGamma": ("pdfs.gamma", ("x", "gamma", "beta", "mu")),
}


def _simple(module: str, members: tuple[str, ...]) -> Any:
    def make(builder: Builder, record: Streamed) -> Any:
        import importlib

        cls = getattr(importlib.import_module(f"xrdroot.roofit.{module}"), record.cls)
        args = [builder.proxy(record, member) for member in members]
        if record.cls == "RooUniform":
            args = [builder.listed(record, "x")]
        return cls(name_of(record), title_of(record), *args)

    return make


for _name, (_module, _members) in SIMPLE.items():
    MAKERS[_name] = _simple(_module, _members)


def _twice(made: Any) -> Any:
    """The proxy listed again, as a class ROOT registers it for in its streamer prints it."""
    first = made._proxies[0]
    made._proxies.append(Proxy(first.name, first.target, first.many, first.shape))
    return made


@maker("RooPoisson")
def _poisson(builder: Builder, record: Streamed) -> Any:
    from ..pdfs.shapes import RooPoisson

    made = RooPoisson(name_of(record), title_of(record), builder.proxy(record, "x"),
                      builder.proxy(record, "mean"), bool(record.get("_noRounding")))  # fmt: skip
    made.protectNegativeMean(bool(record.get("_protectNegative")))
    return made


@maker("RooPolyVar")
def _polyvar(builder: Builder, record: Streamed) -> Any:
    from ..functions import RooPolyVar

    lowest = int(record.get("_lowestOrder") or 0)
    return RooPolyVar(name_of(record), title_of(record), builder.proxy(record, "_x"),
                      builder.listed(record, "_coefList"), lowest)  # fmt: skip


@maker("RooProduct")
def _product(builder: Builder, record: Streamed) -> Any:
    from ..functions import RooProduct

    terms = builder.listed(record, "_compRSet") + builder.listed(record, "_compCSet")
    return RooProduct(name_of(record), title_of(record), terms)


@maker("RooAddition")
def _addition(builder: Builder, record: Streamed) -> Any:
    from ..functions import RooAddition

    return RooAddition(name_of(record), title_of(record), builder.listed(record, "_set"))


@maker("RooRealSumPdf")
def _real_sum(builder: Builder, record: Streamed) -> Any:
    from ..pdfs.realsum import RooRealSumPdf

    funcs, coefs = builder.listed(record, "_funcList"), builder.listed(record, "_coefList")
    return RooRealSumPdf(name_of(record), title_of(record), funcs, coefs,
                         bool(record.get("_extended")))  # fmt: skip


@maker("RooProdPdf")
def _product_pdf(builder: Builder, record: Streamed) -> Any:
    """The factors, each conditional one with its normalisation set - ``nset`` - or its
    conditional observables - ``cset``."""
    from ..pdfs.prodpdf import RooProdPdf

    pdfs = builder.listed(record, "_pdfList")
    factors: list[Any] = []
    for pdf, nset in zip(pdfs, record.get("_pdfNSetList") or [None] * len(pdfs), strict=False):
        members = builder.nodes(nset.get("_list")) if nset is not None else []
        if not members:
            factors.append([pdf])
            continue
        cset = nset.get("_name") == "cset"
        factors.append(RooCmdArg("Conditional", [pdf], members, cset))
    return RooProdPdf(name_of(record), title_of(record), *factors,
                      float(record.get("_cutOff") or 0.0))  # fmt: skip


@maker("RooSimultaneous")
def _simultaneous(builder: Builder, record: Streamed) -> Any:
    """The index category, and each state's density from the list of the proxies to them."""
    from ..pdfs.simultaneous import RooSimultaneous

    made = RooSimultaneous(name_of(record), title_of(record), builder.proxy(record, "_indexCat"))
    for proxy in (record.get("_pdfProxyList").get("items") or ()):
        made.addPdf(builder.node(proxy.get("_arg")), name_of(proxy))
    return made


@maker("RooHistFunc")
def _hist_func(builder: Builder, record: Streamed) -> Any:
    from ..pdfs.histpdf import RooHistFunc

    data = builder.node(record.get("_dataHist"))
    made = RooHistFunc(name_of(record), title_of(record), builder.listed(record, "_depList"),
                       data, int(record.get("_intOrder") or 0))  # fmt: skip
    made.setCdfBoundaries(bool(record.get("_cdfBoundaries")))
    return _twice(made)


@maker("RooBinWidthFunction")
def _bin_width(builder: Builder, record: Streamed) -> Any:
    from ..pdfs.histfactory import RooBinWidthFunction

    made = RooBinWidthFunction(name_of(record), title_of(record),
                               builder.proxy(record, "_histFunc"),
                               bool(record.get("_divideByBinWidth")))  # fmt: skip
    return _twice(made)


@maker("ParamHistFunc")
def _param_hist(builder: Builder, record: Streamed) -> Any:
    from ..pdfs.histfactory import ParamHistFunc

    return ParamHistFunc(name_of(record), title_of(record), builder.listed(record, "_dataVars"),
                         builder.listed(record, "_paramSet"))  # fmt: skip


@maker("RooStats::HistFactory::FlexibleInterpVar")
def _flexible(builder: Builder, record: Streamed) -> Any:
    from ..pdfs.histfactory import FlexibleInterpVar

    made = FlexibleInterpVar(name_of(record), title_of(record),
                             builder.listed(record, "_paramList"), float(record.get("_nominal")),
                             record.get("_low"), record.get("_high"),
                             record.get("_interpCode"))  # fmt: skip
    made._boundary = float(record.get("_interpBoundary") or 1.0)
    return made
