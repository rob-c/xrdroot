"""``RooExtendedBinding``: a density's expected events as a function of a normalisation set.

ROOT's own test of the class (``roofit/roofitcore/test/testRooExtendedBinding.cxx``)
asks a ``RooRealSumPdf`` over a ``RooUniform`` on ``x`` in ``[-10, 10]`` what it
expects, through the binding: ``1.0`` when the binding holds no observables and
``20.0`` when it holds ``x``. The rest of these tests hold the binding to the
call it makes, not only to the number it returns.
"""

from __future__ import annotations

from typing import Any

from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.pdf import RooAbsPdf
from xrdroot.roofit.pdfs.basic import RooUniform
from xrdroot.roofit.pdfs.extend import RooExtendedBinding
from xrdroot.roofit.pdfs.realsum import RooRealSumPdf
from xrdroot.roofit.real import Context
from xrdroot.roofit.variables import RooRealVar


def uniform_sum() -> tuple[RooRealVar, RooRealSumPdf]:
    """ROOT's test's model: a uniform density in ``x`` over ``[-10, 10]``, as a sum."""
    x = RooRealVar("x", "x", -10, 10)
    return x, RooRealSumPdf("pdf", "pdf", [RooUniform("unif", "unif", x)], [])


class Recorder(RooAbsPdf):
    """A density that remembers the normalisation set ``expectedEvents`` is asked about."""

    def __init__(self) -> None:
        super().__init__("rec", "rec")
        self.asked: list[Any] = []

    def compute(self, ctx: Context) -> Any:
        return 1.0

    def expectedEvents(self, nset: Any = None) -> float:
        self.asked.append(nset)
        return 3.0


def test_an_un_normalised_binding_reports_one_expected_event() -> None:
    """Without observables a density's expected events are its own normalisation, ``1``."""
    _, pdf = uniform_sum()

    assert RooExtendedBinding("ext1", "ext1", pdf).getVal() == 1.0


def test_a_binding_with_a_normalisation_set_reports_the_integral_over_it() -> None:
    """With ``x`` the binding is the density's integral over ``x``, the range's width here."""
    x, pdf = uniform_sum()

    assert RooExtendedBinding("ext2", "ext2", pdf, RooArgSet([x])).getVal() == 20.0


def test_the_binding_passes_the_observables_it_was_built_with() -> None:
    """The set reaches ``expectedEvents`` itself, as ROOT's ``RooSetProxy`` does."""
    x, _ = uniform_sum()
    pdf = Recorder()
    obs = RooArgSet([x])

    RooExtendedBinding("ext", "ext", pdf, obs).getVal()

    assert [one.GetName() for one in pdf.asked[0]] == ["x"]


def test_a_binding_without_observables_asks_about_none() -> None:
    """``evaluate()`` hands ``expectedEvents`` a null set, not an empty one."""
    pdf = Recorder()

    RooExtendedBinding("ext", "ext", pdf).getVal()

    assert pdf.asked == [None]


def test_a_single_observable_stands_for_its_set() -> None:
    """ROOT takes a ``RooArgSet``; a bare variable means the set holding it, as elsewhere here."""
    x, pdf = uniform_sum()

    assert RooExtendedBinding("ext", "ext", pdf, x).getVal() == pdf.expectedEvents(x)


def test_the_binding_depends_on_the_density_and_its_observables() -> None:
    """Both inputs are proxies, so the binding is a client of each and moves with them."""
    x, pdf = uniform_sum()
    binding = RooExtendedBinding("ext", "ext", pdf, x)

    assert [one.GetName() for one in binding.servers()] == ["pdf", "x"]
    assert binding.dependents() == frozenset({"x"})


def test_pyroot_exposes_the_binding_under_roots_name() -> None:
    """``import ROOT`` finds the class through the registry, so the factory's name is ROOT's."""
    import xrdroot.pyroot as ROOT
    from xrdroot.roofit.registry import find

    assert ROOT.RooExtendedBinding is RooExtendedBinding
    assert find("RooExtendedBinding") is RooExtendedBinding
