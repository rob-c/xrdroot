"""``TMath``'s densities, distributions and quantiles, at the digit the oracle prints.
``chisquared_quantile_c``, ``crystalball_function``, ``cyl_bessel_j`` and the rest - which is the
recipe in ``docs/instructions/read-a-whole-root-math-family-off-the-oracle-into-a.md``. A relative
tolerance is used with this comment: off ROOT's own machine glibc and Apple's libm round ``erf`` and
``lgamma`` in the last bits, so bit-exactness is only claimed where ``tests/refmachine`` says so.

Nothing here opens a file: the functions take numbers, which keeps ``xrdroot.function.rootmath``
inside the ``hist`` component's own scope.
"""

from __future__ import annotations

import math

import pytest

from xrdroot.function import rootmath as rm

#: The oracle's answers, one per function: name, arguments, value.
ORACLE = [
    ("gaussian_pdf", (1.0, 2.0, 0.0), 0.17603266338214976),
    ("gaussian_cdf", (1.0, 2.0, 0.0), 0.69146246127401301),
    ("gaussian_cdf_c", (1.0, 2.0, 0.0), 0.30853753872598694),
    ("gaussian_quantile", (0.3, 1.0), -0.52440051270804089),
    ("gaussian_quantile_c", (0.3, 1.0), 0.52440051270804089),
    ("chisquared_pdf", (3.0, 4.0), 0.16734762011132237),
    ("chisquared_cdf", (3.0, 4.0), 0.44217459962892536),
    ("chisquared_cdf_c", (3.0, 4.0), 0.55782540037107464),
    ("chisquared_quantile", (0.3, 4.0), 2.1946984214069833),
    ("chisquared_quantile_c", (0.3, 4.0), 4.8784329665604087),
    ("gamma_pdf", (1.5, 2.0, 3.0), 0.10108844328543891),
    ("gamma_cdf", (1.5, 2.0, 3.0), 0.090204010431049864),
    ("gamma_cdf_c", (1.5, 2.0, 3.0), 0.90979598956895014),
    ("gamma_quantile", (0.3, 2.0, 3.0), 3.2920476321104752),
    ("gamma_quantile_c", (0.3, 2.0, 3.0), 7.3176494498406131),
    ("exponential_pdf", (1.5, 2.0), 0.099574136735727889),
    ("exponential_cdf", (1.5, 2.0), 0.95021293163213605),
    ("exponential_cdf_c", (1.5, 2.0), 0.049787068367863944),
    ("exponential_quantile", (0.3, 2.0), 0.17833747196936617),
    ("exponential_quantile_c", (0.3, 2.0), 0.60198640216296806),
    ("poisson_pdf", (3.0, 2.5), 0.21376301724973648),
    ("poisson_cdf", (3.0, 2.5), 0.75757613313306593),
    ("poisson_cdf_c", (3.0, 2.5), 0.2424238668669341),
    ("binomial_pdf", (3.0, 0.4, 10.0), 0.21499084800000015),
    ("binomial_cdf", (3.0, 0.4, 10.0), 0.38228060160000016),
    ("binomial_cdf_c", (3.0, 0.4, 10.0), 0.61771939839999979),
    ("landau_pdf", (1.0, 1.0, 0.0), 0.14520663713086199),
    ("breitwigner_pdf", (1.0, 2.0, 0.0), 0.15915494309189535),
    ("breitwigner_cdf", (1.0, 2.0, 0.0), 0.75),
    ("breitwigner_cdf_c", (1.0, 2.0, 0.0), 0.25),
    ("cauchy_pdf", (1.0, 1.0, 0.0), 0.15915494309189535),
    ("cauchy_cdf", (1.0, 1.0, 0.0), 0.75),
    ("cauchy_cdf_c", (1.0, 1.0, 0.0), 0.25),
    ("lognormal_pdf", (1.5, 0.5, 0.4), 0.64659150449091518),
    ("lognormal_cdf", (1.5, 0.5, 0.4), 0.40658549994911303),
    ("lognormal_cdf_c", (1.5, 0.5, 0.4), 0.59341450005088692),
    ("tdistribution_pdf", (1.0, 5.0), 0.21967979735098056),
    ("tdistribution_cdf", (1.0, 5.0), 0.81839126617543867),
    ("tdistribution_cdf_c", (1.0, 5.0), 0.18160873382456133),
    ("tdistribution_quantile", (0.3, 5.0), -0.55942964446936083),
    ("tdistribution_quantile_c", (0.3, 5.0), 0.55942964446936083),
    ("fdistribution_pdf", (1.0, 3.0, 4.0), 0.34355003127532363),
    ("fdistribution_cdf", (1.0, 3.0, 4.0), 0.52105088076757367),
    ("fdistribution_cdf_c", (1.0, 3.0, 4.0), 0.47894911923242622),
    ("fdistribution_quantile", (0.3, 3.0, 4.0), 0.50389668662447662),
    ("fdistribution_quantile_c", (0.3, 3.0, 4.0), 1.7214950496159529),
    ("beta_pdf", (0.3, 2.0, 3.0), 1.764),
    ("beta_cdf", (0.3, 2.0, 3.0), 0.34829999999999978),
    ("beta_cdf_c", (0.3, 2.0, 3.0), 0.65169999999999995),
    ("beta_quantile", (0.3, 2.0, 3.0), 0.27238394207510536),
    ("beta_quantile_c", (0.3, 2.0, 3.0), 0.50840475487258441),
    ("uniform_pdf", (0.3, 0.0, 1.0), 1.0),
    ("uniform_cdf", (0.3, 0.0, 1.0), 0.29999999999999999),
    ("uniform_cdf_c", (0.3, 0.0, 1.0), 0.69999999999999996),
    ("crystalball_pdf", (1.0, 1.5, 3.0, 1.0), 0.22769206530292196),
    ("crystalball_cdf", (1.0, 1.5, 3.0, 1.0), 0.85070697080978896),
    ("crystalball_cdf_c", (1.0, 1.5, 3.0, 1.0), 0.14929302919021106),
    ("crystalball_function", (1.0, 1.5, 3.0, 1.0), 0.60653065971263342),
    ("cyl_bessel_j", (1.0, 2.0), 0.57672480775687363),
    ("cyl_bessel_i", (1.0, 2.0), 1.5906368546373291),
    ("assoc_legendre", (2, 1, 0.5), 1.299038105676658),
]


@pytest.mark.parametrize(("name", "arguments", "expected"), ORACLE)
def test_each_tmath_function_answers_what_root_s_own_does(name, arguments, expected):
    """The value at one point, against the oracle's: 1e-12 relative covers libm's last bits."""
    assert getattr(rm, name)(*arguments) == pytest.approx(expected, rel=1e-12)


#: Each family's two tails and the arguments both take.
PAIRS = [
    ("gaussian_cdf", "gaussian_cdf_c", (1.0, 2.0, 0.0)),
    ("chisquared_cdf", "chisquared_cdf_c", (3.0, 4.0)),
    ("gamma_cdf", "gamma_cdf_c", (1.5, 2.0, 3.0)),
    ("exponential_cdf", "exponential_cdf_c", (1.5, 2.0)),
    ("poisson_cdf", "poisson_cdf_c", (3.0, 2.5)),
    ("binomial_cdf", "binomial_cdf_c", (3.0, 0.4, 10.0)),
    ("breitwigner_cdf", "breitwigner_cdf_c", (1.0, 2.0)),
    ("cauchy_cdf", "cauchy_cdf_c", (1.0, 1.0)),
    ("lognormal_cdf", "lognormal_cdf_c", (1.5, 0.5, 0.4)),
    ("tdistribution_cdf", "tdistribution_cdf_c", (1.0, 5.0)),
    ("fdistribution_cdf", "fdistribution_cdf_c", (1.0, 3.0, 4.0)),
    ("beta_cdf", "beta_cdf_c", (0.3, 2.0, 3.0)),
    ("uniform_cdf", "uniform_cdf_c", (0.3, 0.0, 1.0)),
    ("crystalball_cdf", "crystalball_cdf_c", (1.0, 1.5, 3.0, 1.0)),
]


@pytest.mark.parametrize(("lower", "upper", "arguments"), PAIRS)
def test_a_lower_tail_and_its_upper_twin_add_up_to_certainty(lower, upper, arguments):
    """Every family here comes in a pair, ``_cdf`` and ``_cdf_c``, and the two must sum to one: a
    pair that drifts apart is a plotting bug no tutorial would attribute to the maths."""
    assert getattr(rm, lower)(*arguments) + getattr(rm, upper)(*arguments) == pytest.approx(
        1.0, rel=1e-12
    )


def test_a_quantile_reads_its_own_cdf_backwards_and_the_upper_one_mirrors_it():
    """A quantile put through its cdf is the chance it was asked for, and an upper quantile answers
    the chance one less the lower one would: the property every interval is built on."""
    assert rm.chisquared_cdf(rm.chisquared_quantile(0.3, 4.0), 4.0) == pytest.approx(0.3, rel=1e-9)
    assert rm.beta_cdf(rm.beta_quantile(0.3, 2.0, 3.0), 2.0, 3.0) == pytest.approx(0.3, rel=1e-9)
    assert rm.gaussian_quantile_c(0.3, 1.0) == pytest.approx(-rm.gaussian_quantile(0.3, 1.0))


def test_pi_is_called_and_is_the_constant_the_gaussian_s_own_is_built_from():
    """``TMath::Pi`` is a call, not a constant - ROOT writes ``TMath::Pi()`` - and it answers
    ``math.pi`` to the last bit, which is what ``gaussian_pdf`` at zero is built from."""
    assert rm.Pi() == math.pi
    assert rm.gaussian_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
