"""The densities and distributions ROOT's ``TMath``/``ROOT::Math`` provide, to the oracle's digit.

Every expected number below was printed by the ROOT 6.40.04 oracle with ``%.17g`` -
``ROOT.Math.normal_cdf``, ``chisquared_quantile``, ``landau_pdf`` and the rest - so a reader that
rounds differently shows up here rather than in a tutorial. The comparison uses a relative tolerance
with a comment, because off ROOT's own machine glibc and Apple's libm round ``erf`` and ``lgamma``
in the last bits.

Nothing here opens a file: the functions take numbers, which is what keeps
``xrdroot.function.distributions`` inside the ``hist`` component's own scope.
"""

from __future__ import annotations

import pytest

from xrdroot.function import distributions as dist

#: One per function: the function's name, its arguments, and what the oracle answered.
ORACLE = [
    ("normal_cdf", (1.0,), 0.84134474606854293),
    ("normal_quantile", (0.3,), -0.52440051270804089),
    ("gamma_pdf", (1.5, 2.0, 3.0), 0.10108844328543891),
    ("chi2_cdf", (3.0, 4.0), 0.44217459962892536),
    ("chi2_quantile", (0.3, 4.0), 2.1946984214069833),
    ("student_pdf", (1.0, 5.0), 0.21967979735098056),
    ("student_cdf", (1.0, 5.0), 0.81839126617543867),
    ("student_quantile", (0.3, 5.0), -0.55942964446936083),
    ("f_pdf", (1.0, 3.0, 4.0), 0.34355003127532363),
    ("f_cdf", (1.0, 3.0, 4.0), 0.52105088076757367),
    ("f_quantile", (0.3, 3.0, 4.0), 0.50389668662447662),
    ("beta_pdf", (0.3, 2.0, 3.0), 1.764),
    ("beta_cdf", (0.3, 2.0, 3.0), 0.34829999999999978),
    ("poisson_pdf", (3.0, 2.5), 0.21376301724973648),
    ("poisson_cdf", (3.0, 2.5), 0.75757613313306593),
    ("binomial_pdf", (3.0, 0.4, 10.0), 0.21499084800000015),
    ("binomial_cdf", (3.0, 0.4, 10.0), 0.38228060160000016),
    ("lognormal_pdf", (1.5, 0.5, 0.4), 0.64659150449091518),
    ("lognormal_cdf", (1.5, 0.5, 0.4), 0.40658549994911303),
]


@pytest.mark.parametrize(("name", "arguments", "expected"), ORACLE)
def test_each_distribution_answers_what_root_s_own_does(name, arguments, expected):
    """The value at one point, against the oracle's: 1e-12 relative covers libm's last bits."""
    assert getattr(dist, name)(*arguments) == pytest.approx(expected, rel=1e-12)


def test_invert_walks_a_cdf_to_the_quantile_root_would_give():
    """``invert`` is the bisection the quantiles are built on: fed a normal cdf it finds the value
    ``normal_quantile`` gives.

    It widens its bracket upwards until the bracket holds the answer, so a cdf that never
    **reaches** ``p`` would widen for ever - `W0-277` records that, and no test feeds it one."""
    assert dist.invert(dist.normal_cdf, 0.3, -10.0, 10.0) == pytest.approx(
        -0.52440051270804089, rel=1e-9
    )


def test_a_quantile_of_nothing_or_everything_is_the_end_it_asks_for():
    """A chance of zero is below everything and one is above, which the quantiles answer without
    hunting for a crossing."""
    assert dist.normal_quantile(0.0) < -8.0
    assert dist.normal_quantile(1.0) > 8.0


def test_a_quantile_is_its_cdf_read_backwards():
    """Whatever the family, the quantile of a chance put back through the cdf is that chance - the
    property every interval ROOT quotes is built on."""
    assert dist.chi2_cdf(dist.chi2_quantile(0.3, 4.0), 4.0) == pytest.approx(0.3, rel=1e-9)
    assert dist.f_cdf(dist.f_quantile(0.3, 3.0, 4.0), 3.0, 4.0) == pytest.approx(0.3, rel=1e-9)
    assert dist.student_cdf(dist.student_quantile(0.3, 5.0), 5.0) == pytest.approx(0.3, rel=1e-9)


def test_a_density_outside_its_support_is_zero_and_its_total_is_one():
    """A gamma density below its start and a log-normal's below zero are zero, and a Poisson's
    masses over the integers add up to one."""
    assert dist.gamma_pdf(0.0, 2.0, 3.0) == 0.0
    assert dist.lognormal_pdf(-1.0, 0.5, 0.4) == 0.0
    total = sum(dist.poisson_pdf(float(n), 2.5) for n in range(40))
    assert total == pytest.approx(1.0, abs=1e-12)


def test_the_standard_normal_s_cdf_at_zero_is_the_half_a_fit_is_built_on():
    """A Gaussian's cdf at its mean is a half, one sigma either side covers its 68.27 %, and a mean
    and a sigma move and widen it."""
    assert dist.normal_cdf(0.0) == pytest.approx(0.5)
    width = dist.normal_cdf(1.0) - dist.normal_cdf(-1.0)
    assert width == pytest.approx(0.682689492137086, rel=1e-12)
    assert dist.normal_cdf(0.0, 2.0, 1.0) == pytest.approx(0.3085375387259869, rel=1e-12)
