"""``PdfProposal`` and ``ProposalHelper``: proposals drawn from a density about the chain's point.

A ``PdfProposal`` draws its candidates from a density - generated a cache at
a time - whose parameters follow the chain's current point through the
mappings it is given; ``ProposalHelper`` makes the usual one: a
multivariate Gaussian about the point, of the covariance given (or a fifth
of each range), mixed with a uniform part if asked.
"""

from __future__ import annotations

from typing import Any

from ..roofit.collections import RooArgList, RooArgSet, as_list
from ..roofit.messages import ERROR, log
from .markov import ProposalFunction

__all__ = ["PdfProposal", "ProposalHelper"]


class PdfProposal(ProposalFunction):
    """Candidates drawn from a density, a cache at a time, its parameters mapped to the point."""

    def __init__(self, pdf: Any = None) -> None:
        self._pdf = pdf
        self._map: list[tuple[Any, Any]] = []
        self._master = RooArgSet()
        self._last: Any = None
        self._cache: Any = None
        self._position, self._size = 0, 1

    def SetPdf(self, pdf: Any) -> None:
        self._pdf = pdf

    def GetPdf(self) -> Any:
        return self._pdf

    def SetOwnsPdf(self, owns: bool = True) -> None:
        """Who deletes the density, which Python's memory manages here."""

    def SetCacheSize(self, size: int) -> None:
        if size > 0:
            self._size = int(size)
        else:
            log(None, ERROR, "Eval", "Warning: Requested non-positive cache size: "
                f"{size}. Cache size unchanged.")  # fmt: skip

    def AddMapping(self, proposalParam: Any, update: Any) -> None:
        """``proposalParam`` follows ``update`` - a variable of the chain, usually."""
        params = list(update.getParameters(None))
        self._master.add(params or [update], True)
        self._map.append((proposalParam, update))

    def IsSymmetric(self, x1: Any, x2: Any) -> bool:
        return False

    def _follow(self, x: Any) -> None:
        self._master.assign(x)
        for param, update in self._map:
            param.setVal(update.getVal(x))

    def _generate(self, xPrime: Any) -> None:
        self._cache = self._pdf.generate(list(as_list(xPrime)), self._size)

    def Propose(self, xPrime: Any, x: Any) -> None:
        if self._last is None:
            self._last = RooArgSet(as_list(x)).snapshot()
            self._follow(x)
            self._generate(xPrime)
        moved = False
        if self._map:
            moved = not _equal(self._last, x)
            if moved:
                self._follow(x)
                self._last.assign(x)
        if moved or self._position >= self._size:
            self._generate(xPrime)
            self._position = 0
        proposal = self._cache.get(self._position)
        self._position += 1
        RooArgSet(as_list(xPrime)).assign(proposal)

    def GetProposalDensity(self, x1: Any, x2: Any) -> float:
        self._follow(x2)
        mine = self._pdf.getObservables(x1)
        mine.assign(x1)
        return float(self._pdf.getVal(x1))


class ProposalHelper:
    """Builds a ``PdfProposal``: a Gaussian about the point, perhaps mixed with a uniform one."""

    def __init__(self) -> None:
        self._prop = PdfProposal()
        self._vars: Any = None
        self._pdf: Any = None
        self._cov: Any = None
        self._divisor = 5.0
        self._uniform_frac = -1.0
        self._cache = -1
        self._updates = False

    def SetPdf(self, pdf: Any) -> None:
        self._pdf = pdf

    def SetVariables(self, items: Any) -> None:
        self._vars = RooArgList(as_list(items))

    def SetCovMatrix(self, cov: Any) -> None:
        import numpy as np

        self._cov = np.array(getattr(cov, "matrix", lambda: cov)(), dtype=np.float64)

    def SetWidthRangeDivisor(self, divisor: float) -> None:
        if divisor > 0.0:
            self._divisor = float(divisor)

    def SetUniformFraction(self, fraction: float) -> None:
        self._uniform_frac = float(fraction)

    def SetCacheSize(self, size: int) -> None:
        if size > 0:
            self._cache = int(size)
        else:
            log(None, ERROR, "Eval", "Warning: Requested non-positive cache size: "
                f"{size}. Cache size unchanged.")  # fmt: skip

    def SetUpdateProposalParameters(self, flag: bool) -> None:
        self._updates = bool(flag)

    def SetClues(self, clues: Any) -> None:
        from ..errors import UnsupportedFeatureError

        raise UnsupportedFeatureError(
            "ProposalHelper's clues are a RooNDKeysPdf, which xrdroot's RooFit does not have yet"
        )

    def _gaussian(self) -> Any:
        """``CreatePdf``: ``mvg``, a Gaussian in the variables about ``mu__<name>`` copies."""
        import numpy as np

        from ..roofit.pdfs.multivar import RooMultiVarGaussian

        xs, mus = list(self._vars), []
        for var in xs:
            mu = var.clone(f"mu__{var.GetName()}")
            mus.append(mu)
            if self._updates:
                self._prop.AddMapping(mu, var)
        if self._cov is None:
            widths = [(v.getMax() - v.getMin()) / self._divisor for v in xs]
            self._cov = np.diag(widths)
        return RooMultiVarGaussian("mvg", "MVG Proposal", xs, mus, self._cov)

    def GetProposalFunction(self) -> PdfProposal:
        """The proposal: ``proposalFunction``, the Gaussian - and the uniform part - summed."""
        from ..roofit.pdfs.addpdf import RooAddPdf
        from ..roofit.pdfs.basic import RooUniform, ref

        if self._pdf is None:
            self._pdf = self._gaussian()
        components, coefs = [], []
        if self._uniform_frac > 0.0:
            components.append(RooUniform("uniform", "Uniform Proposal PDF", list(self._vars)))
            coefs.append(ref(self._uniform_frac))
        components.append(self._pdf)
        self._prop.SetPdf(RooAddPdf("proposalFunction", "Proposal Density", components, coefs))
        if self._cache > 0:
            self._prop.SetCacheSize(self._cache)
        return self._prop


def _equal(one: Any, two: Any) -> bool:
    """``Equals``: the same variables, holding the same values."""
    first, second = RooArgSet(as_list(one)), RooArgSet(as_list(two))
    if sorted(first.names()) != sorted(second.names()):
        return False
    return all(v.getVal() == second.getRealValue(v.GetName()) for v in first)
