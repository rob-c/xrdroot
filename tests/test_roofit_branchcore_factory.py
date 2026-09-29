"""The names the workspace factory gives what it makes without one, as ROOT 6.40 gives them.

An object built inside another's argument list is ``<outer>_<n>`` for the
n-th argument - a digit more inside a ``{...}`` list - and one built at the
top level is ``gobj<n>``, a session-wide counter that skips a name already
taken. ``SUM``, ``PROD``, ``expr``, ``sum``, ``prod`` and ``SIMUL`` name
their parts so; ``sum`` of products makes a ``RooAddition`` of products.
Each name below is one ROOT's own ``RooWorkspace::Print`` listed for the same
factory calls.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roofit import factory
from xrdroot.roofit.messages import service
from xrdroot.roofit.workspace import RooWorkspace


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fresh session: the ``gobj`` counter at zero, RooFit's message streams as they start."""
    monkeypatch.setattr(factory, "_GLOBAL", [0])
    service().reset()
    service().setGlobalKillBelow(3)  # WARNING and above: the import lines are not the point
    yield
    service().reset()


def workspace() -> RooWorkspace:
    w = RooWorkspace("w")
    w.factory("x[0,-5,5]")
    return w


def test_a_top_level_object_without_a_name_is_gobj_counting_past_names_taken() -> None:
    """``Gaussian(x,0,1)`` twice is ``gobj0``, ``gobj1``; with ``gobj2`` taken the next is
    ``gobj3``."""
    w = workspace()
    assert w.factory("Gaussian(x, 0, 1)").GetName() == "gobj0"
    assert w.factory("Gaussian(x, 0, 1)").GetName() == "gobj1"
    w.factory("gobj2[1]")
    assert w.factory("Gaussian(x, 1, 2)").GetName() == "gobj3"


def test_an_argument_made_in_place_is_named_after_its_owner_and_position() -> None:
    """``p2(x, {0.3, Gaussian(...)})``'s Gaussian is ``p2_22``: argument 2, member 2."""
    w = workspace()
    w.factory("Polynomial::p2(x, {0.3, Gaussian(x,0,1)})")
    assert w.pdf("p2_22") is not None
    w.factory("Polynomial::p(x, {a0[0.1], a1[0.2]})")
    assert w.var("a0") is not None and w.var("a1") is not None


def test_sum_and_prod_name_their_parts_by_position() -> None:
    """``SUM::sm(f*G, G)`` makes ``sm_1`` and ``sm_2``; ``PROD::pr(G, G|x)`` ``pr_1``, ``pr_2``."""
    w = workspace()
    w.factory("SUM::sm(f[0.5]*Gaussian(x,0,1), Gaussian(x,1,1))")
    w.factory("PROD::pr(Gaussian(x,0,1), Gaussian(y[0,-5,5],x,1)|x)")
    names = ("sm_1", "sm_2", "pr_1", "pr_2")
    assert all(w.pdf(one) is not None for one in names)
    assert w.var("f").getVal() == 0.5


def test_expr_and_simul_name_their_parts_by_position() -> None:
    """``expr::e('x*2', x, G)`` makes ``e_3``; ``SIMUL::si(c, A=G, B=G)`` ``si_2`` and ``si_3``."""
    w = workspace()
    w.factory("expr::e('x*2', x, Gaussian(x,0,1))")
    w.factory("SIMUL::si(c[A,B], A=Gaussian(x,0,1), B=Gaussian(x,1,1))")
    assert all(w.pdf(one) is not None for one in ("e_3", "si_2", "si_3"))
    assert w.function("e").getVal() == 0.0


def test_a_sum_of_products_is_an_addition_of_products_named_as_roofit_names_them() -> None:
    """``sum::su(2*x, 3*m)`` is ``su_[2_x_x] + su_[3_x_m]``; ``prod::pd(x, 2)`` a product."""
    w = workspace()
    w.factory("m[1]")
    su = w.factory("sum::su(2*x, 3*m)")
    assert [t.GetName() for t in su.terms] == ["su_[2_x_x]", "su_[3_x_m]"]
    assert su.getVal() == 3.0
    assert w.factory("prod::pd(x, 2)").printArgs() == "[ x * 2 ]"
    assert w.factory("sum::plain(x, m)").getVal() == 1.0


def test_a_sum_mixing_products_and_plain_terms_is_refused_in_roofits_words() -> None:
    """Either every term of ``sum`` is a product or none is."""
    w = workspace()
    w.factory("m[1]")
    with pytest.raises(UnsupportedFeatureError, match="either all sum terms must be products"):
        w.factory("sum::bad(2*x, m)")


def test_a_list_at_the_top_level_makes_its_members_with_top_level_names() -> None:
    """``{x, z[2], Gaussian(x,0,1)}`` makes ``z`` and ``gobj0`` - no owner to be named after."""
    w = workspace()
    w.factory("{x, z[2], Gaussian(x,0,1)}")
    assert w.var("z").getVal() == 2.0
    assert w.pdf("gobj0") is not None
