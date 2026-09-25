"""``RooAdaptiveIntegratorND``: ROOT's adaptive cubature for integrals in two dimensions or more.

RooFit integrates a function of two or more observables numerically with
``ROOT::Math::AdaptiveIntegratorMultiDim``, the Genz-Malik algorithm. A
degree-seven rule is applied to a box, and a degree-five rule on the same
points gives the error estimate. The box with the largest estimated error
is taken off a heap and cut in half across the axis where the fourth
difference was biggest. Both halves are then measured and pushed back.
This goes on until the summed error is small enough, or until the next cut
would use more evaluations than the limit allows.

A number that should match ROOT to the last bit has to be made the way ROOT
makes it. So this is a line-for-line port of ``DoIntegral``. The heap is
kept in ROOT's flat work array and uses ROOT's 1-based offsets, so ties
break the same way. The regions are cut in ROOT's order, and each sum is
added up term by term in the order the C++ adds it. The integrand is called
at the same points in the same order. Given a vectorised integrand, one
region's points go in one call, and the values are then added in that same
order. The two paths give the same bits.

``RooAdaptiveIntegratorND`` sets no absolute tolerance, and it takes the
relative tolerance from ``RooNumIntConfig`` (``1e-7`` by default). Its
evaluation limit depends on the number of dimensions: see
:func:`max_evaluations` and :func:`integrate_nd`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "MAX_EVAL_2D",
    "MAX_EVAL_3D",
    "MAX_EVAL_ND",
    "MAX_WARN",
    "Cubature",
    "adaptive_integral",
    "integrate_nd",
    "max_evaluations",
    "not_converged_warning",
    "suppressed_warnings_summary",
    "suppressing_further_warnings",
]

#: The evaluation limit ``RooAdaptiveIntegratorND`` registers for 2-dimensional integrals.
MAX_EVAL_2D = 100000
#: The evaluation limit ``RooAdaptiveIntegratorND`` registers for 3-dimensional integrals.
MAX_EVAL_3D = 1000000
#: The evaluation limit ``RooAdaptiveIntegratorND`` registers above three dimensions.
MAX_EVAL_ND = 10000000
#: How many precision warnings ``RooAdaptiveIntegratorND`` is configured to print.
MAX_WARN = 5
#: ``RooNumIntConfig``'s default relative tolerance, which RooFit hands the integrator.
DEFAULT_EPS_REL = 1e-7
#: ``IntegratorMultiDimOptions``' default work-space size, used when none is given.
DEFAULT_WK_SIZE = 100000
#: ``IntegratorMultiDimOptions``' default evaluation limit, used when none is given.
DEFAULT_N_CALLS = 100000
#: The message topic RooFit files the precision warnings under.
TOPIC = "NumericIntegration"

#: The rule's node distances from the centre, in units of the half-width: lambda 2, 4 and 5.
XL2, XL4, XL5 = 0.358568582800318073, 0.948683298050513796, 0.688247201611685289
#: The degree-seven weights for the lambda-2 and lambda-4 nodes, divided by 2**n.
W2, W4 = 980.0 / 6561, 200.0 / 19683
#: The degree-five (error) weights for the same nodes, divided by 2**n.
WP2, WP4 = 245.0 / 486, 25.0 / 729

# fmt: off
#: The degree-seven weight of the centre, by dimension from two to fifteen.
WN1 = (
    -0.193872885230909911, -0.555606360818980835, -0.876695625666819078,
    -1.15714067977442459, -1.39694152314179743, -1.59609815576893754,
    -1.75461057765584494, -1.87247878880251983, -1.94970278920896201,
    -1.98628257887517146, -1.98221815780114818, -1.93750952598689219,
    -1.85215668343240347, -1.72615963013768225,
)
#: The degree-seven weight of the lambda-4 axis nodes, by dimension.
WN3 = (
    0.0518213686937966768, 0.0314992633236803330, 0.0111771579535639891,
    -0.00914494741655235473, -0.0294670527866686986, -0.0497891581567850424,
    -0.0701112635269013768, -0.0904333688970177241, -0.110755474267134071,
    -0.131077579637250419, -0.151399685007366752, -0.171721790377483099,
    -0.192043895747599447, -0.212366001117715794,
)
#: The degree-seven weight of the lambda-5 corners, by dimension.
WN5 = (
    0.871183254585174982e-01, 0.435591627292587508e-01, 0.217795813646293754e-01,
    0.108897906823146873e-01, 0.544489534115734364e-02, 0.272244767057867193e-02,
    0.136122383528933596e-02, 0.680611917644667955e-03, 0.340305958822333977e-03,
    0.170152979411166995e-03, 0.850764897055834977e-04, 0.425382448527917472e-04,
    0.212691224263958736e-04, 0.106345612131979372e-04,
)
#: The degree-five weight of the centre, by dimension.
WPN1 = (
    -1.33196159122085045, -2.29218106995884763, -3.11522633744855959,
    -3.80109739368998611, -4.34979423868312742, -4.76131687242798352,
    -5.03566529492455417, -5.17283950617283939, -5.17283950617283939,
    -5.03566529492455417, -4.76131687242798352, -4.34979423868312742,
    -3.80109739368998611, -3.11522633744855959,
)
#: The degree-five weight of the lambda-4 axis nodes, by dimension.
WPN3 = (
    0.0445816186556927292, -0.0240054869684499309, -0.0925925925925925875,
    -0.161179698216735251, -0.229766803840877915, -0.298353909465020564,
    -0.366941015089163228, -0.435528120713305891, -0.504115226337448555,
    -0.572702331961591218, -0.641289437585733882, -0.709876543209876532,
    -0.778463648834019195, -0.847050754458161859,
)
# fmt: on

#: The status ``DoIntegral`` holds while it still has regions to cut.
_DIVIDING = 3


def max_evaluations(ndim: int) -> int:
    """The evaluation limit ``RooAdaptiveIntegratorND`` picks for ``ndim`` dimensions.

    RooFit refuses a one-dimensional integral here, because its 1D
    integrators are better suited to it.
    """
    if ndim < 2:
        raise ValueError(
            "RooAdaptiveIntegratorND::ctor ERROR dimension of function must be at least 2."
        )
    return {2: MAX_EVAL_2D, 3: MAX_EVAL_3D}.get(ndim, MAX_EVAL_ND)


def _axis_points(ctr: list[float], width: list[float]) -> list[list[float]]:
    """The four nodes on each axis, at lambda 2 then lambda 4, below the centre then above it."""
    points = []
    for j, (centre, half) in enumerate(zip(ctr, width)):
        near, far = XL2 * half, XL4 * half
        for coordinate in (centre - near, centre + near, centre - far, centre + far):
            point = list(ctr)
            point[j] = coordinate
            points.append(point)
    return points


def _pair_points(ctr: list[float], width: list[float]) -> list[list[float]]:
    """The lambda-4 nodes in every coordinate plane, in the order ROOT's nested loops visit them."""
    far = [XL4 * half for half in width]
    points = []
    n = len(ctr)
    for j1 in range(n - 1):
        for k in range(j1 + 1, n):
            for first in (ctr[j1] - far[j1], ctr[j1] + far[j1]):
                for second in (ctr[k] - far[k], ctr[k] + far[k]):
                    point = list(ctr)
                    point[j1], point[k] = first, second
                    points.append(point)
    return points


def _corner_points(ctr: list[float], width: list[float]) -> list[list[float]]:
    """The lambda-5 corners, walked as ROOT walks them: a binary counter over the signs.

    The walk is simulated with ROOT's own test, that the flipped offset is
    positive, so even a box of zero width is walked as ROOT walks it.
    """
    offsets = [-XL5 * half for half in width]
    z = [centre + offset for centre, offset in zip(ctr, offsets)]
    points = [list(z)]
    j = 0
    while j < len(ctr):
        offsets[j] = -offsets[j]
        z[j] = ctr[j] + offsets[j]
        if offsets[j] > 0:
            points.append(list(z))
            j = 0
        else:
            j += 1
    return points


def rule_points(ctr: list[float], width: list[float]) -> Any:
    """Every point the rule evaluates for one box, as an ``(npoints, ndim)`` array in ROOT's order.

    The centre comes first, then the axis nodes, the plane nodes and the
    corners: ``2**n + 2n(n+1) + 1`` points in all.
    """
    rows = [list(ctr), *_axis_points(ctr, width), *_pair_points(ctr, width)]
    return np.array(rows + _corner_points(ctr, width), dtype=np.float64)


def _running_sum(values: Any) -> float:
    """Values added one at a time from zero, as ROOT's ``+=`` loops add them."""
    total = 0.0
    for value in values:
        total += float(value)
    return total


@dataclass
class _Sums:
    """The five weighted sums of one box, and the axis with the biggest fourth difference."""

    sum1: float
    sum2: float
    sum3: float
    sum4: float
    sum5: float
    axis: int | None

    def zero(self) -> bool:
        """Whether every sum is zero, which ROOT takes to mean the integral is zero."""
        return (
            self.sum1 == 0
            and self.sum2 == 0
            and self.sum3 == 0
            and self.sum4 == 0
            and self.sum5 == 0
        )


def _sums(values: list[float], n: int) -> _Sums:
    """Add up one box's values in ROOT's order, and find the axis ROOT would cut next.

    The axis is ``None`` when no difference beats zero, which only a NaN
    can cause: ROOT then keeps the axis it chose for the box before.
    """
    sum1 = values[0]
    sum2 = sum3 = difmax = 0.0
    axis = None
    for j in range(n):
        a, b, c, d = values[1 + 4 * j : 5 + 4 * j]
        f2, f3 = a + b, c + d
        sum2 += f2
        sum3 += f3
        dif = abs(7 * f2 - f3 - 12 * sum1)
        if dif >= difmax:
            difmax, axis = dif, j + 1
    pairs = 1 + 4 * n + 2 * n * (n - 1)
    sum4 = _running_sum(values[1 + 4 * n : pairs])
    return _Sums(sum1, sum2, sum3, sum4, _running_sum(values[pairs:]), axis)


def _estimates(sums: _Sums, n: int, width: list[float]) -> tuple[float, float]:
    """The box's degree-seven value and its error, the gap to the degree-five value."""
    rgnvol = 2.0**n
    for half in width:
        rgnvol *= half
    i = n - 2
    rgncmp = rgnvol * (
        WPN1[i] * sums.sum1 + WP2 * sums.sum2 + WPN3[i] * sums.sum3 + WP4 * sums.sum4
    )
    rgnval = WN1[i] * sums.sum1 + W2 * sums.sum2 + WN3[i] * sums.sum3 + W4 * sums.sum4
    rgnval = (rgnval + WN5[i] * sums.sum5) * rgnvol
    return rgnval, abs(rgnval - rgncmp)


@dataclass
class _Limits:
    """The evaluation bounds and work-space size ``DoIntegral`` settles on before it starts."""

    irgnst: int
    irlcls: int
    minpts: int
    maxpts: int
    iwk: int

    @classmethod
    def resolve(cls, n: int, min_pts: int, max_pts: int, size: int) -> _Limits:
        """ROOT's arithmetic: at least one rule, and ten times ``minpts`` over a low maximum."""
        irgnst = 2 * n + 3
        irlcls = 2**n + 2 * n * (n + 1) + 1
        maxpts = max(max_pts, irlcls)
        minpts = min_pts if min_pts >= 1 else irlcls
        if maxpts < minpts:
            maxpts = 10 * minpts
        return cls(irgnst, irlcls, minpts, maxpts, max(size, irgnst * (1 + maxpts // irlcls) // 2))


@dataclass
class _Heap:
    """ROOT's work array: a binary heap of boxes by error, each a record of ``2n + 3`` numbers.

    A record ends at a 1-based offset ``isbrgn``, a multiple of the record
    size. From the end backwards it holds the error, the value, the axis
    to cut, then the centre and half-width of each coordinate in turn.
    """

    irgnst: int
    wk: list[float]
    isbrgn: int
    isbrgs: int

    def _move(self, source: int) -> None:
        """Copy the record ending at ``source`` over the record at ``isbrgn``, and move there."""
        self.wk[self.isbrgn - self.irgnst : self.isbrgn] = self.wk[source - self.irgnst : source]
        self.isbrgn = source

    def sift_down(self, rgnerr: float) -> None:
        """ROOT's label 110: carry a new top box down past every child with a bigger error."""
        while 2 * self.isbrgn <= self.isbrgs:
            child = 2 * self.isbrgn
            if child < self.isbrgs and self.wk[child - 1] < self.wk[child + self.irgnst - 1]:
                child += self.irgnst
            if rgnerr >= self.wk[child - 1]:
                return
            self._move(child)

    def sift_up(self, rgnerr: float) -> None:
        """ROOT's label 140: carry a new last box up past every parent with a smaller error."""
        parent = (self.isbrgn // (2 * self.irgnst)) * self.irgnst
        while parent >= self.irgnst and rgnerr > self.wk[parent - 1]:
            self._move(parent)
            parent = (self.isbrgn // (2 * self.irgnst)) * self.irgnst

    def store(
        self, rgnval: float, rgnerr: float, axis: int, box: tuple[list[float], list[float]]
    ) -> None:
        """ROOT's label 160: write a box's record where the sifting left the gap."""
        end = self.isbrgn
        self.wk[end - 1], self.wk[end - 2], self.wk[end - 3] = rgnerr, rgnval, float(axis)
        for j, (centre, half) in enumerate(zip(*box)):
            self.wk[end - 2 * j - 4], self.wk[end - 2 * j - 5] = centre, half

    def top(self, n: int) -> tuple[float, float, int, list[float], list[float]]:
        """The worst box: its value, error, axis, centre and half-widths."""
        wk, end = self.wk, self.irgnst
        ctr = [wk[end - 2 * j - 4] for j in range(n)]
        width = [wk[end - 2 * j - 5] for j in range(n)]
        return wk[end - 2], wk[end - 1], int(wk[end - 3]), ctr, width


#: An integrand: a point (or, vectorised, an ``(npoints, ndim)`` array of them) to its value(s).
Integrand = Callable[[Any], Any]


class CubatureResult(NamedTuple):
    """What ``DoIntegral`` reports.

    The status is ROOT's: 0 when the tolerance was met (or the integrand
    was zero on the last box when the evaluations ran out), 1 when the
    evaluation limit was reached first, and 2 when the work space was full.
    ``neval`` is what the integral was charged: a whole number of rules.
    """

    value: float
    relerr: float
    status: int
    neval: int


def _evaluator(func: Integrand, vectorized: bool) -> Callable[[Any], list[float]]:
    """Turn the integrand into one call per box, whichever kind it is."""
    if vectorized:
        return lambda points: [float(value) for value in np.asarray(func(points)).ravel()]
    return lambda points: [float(func(point)) for point in points]


@dataclass
class _Run:
    """One ``DoIntegral`` call: the heap, the running totals and the stopping rules."""

    limits: _Limits
    eps_abs: float
    eps_rel: float
    evaluate: Callable[[Any], list[float]]
    heap: _Heap
    result: float = 0.0
    abserr: float = 0.0
    ifncls: int = 0
    axis: int = 0
    last: _Sums = field(default_factory=lambda: _Sums(0.0, 0.0, 0.0, 0.0, 0.0, None))

    def measure(self, ctr: list[float], width: list[float]) -> tuple[float, float]:
        """Apply the rule to one box and add it to the totals, as label 20 does."""
        self.last = _sums(self.evaluate(rule_points(ctr, width)), len(ctr))
        if self.last.axis is not None:
            self.axis = self.last.axis
        rgnval, rgnerr = _estimates(self.last, len(ctr), width)
        self.result += rgnval
        self.abserr += rgnerr
        self.ifncls += self.limits.irlcls
        return rgnval, rgnerr

    def push(self, ctr: list[float], width: list[float]) -> None:
        """Measure a new box and add it at the end of the heap."""
        rgnval, rgnerr = self.measure(ctr, width)
        self.heap.sift_up(rgnerr)
        self.heap.store(rgnval, rgnerr, self.axis, (ctr, width))

    def relerr(self) -> float:
        """The summed error over the magnitude of the value, or the plain error if that is zero."""
        aresult = abs(self.result)
        return self.abserr / aresult if aresult != 0 else self.abserr

    def status(self, relerr: float) -> int:
        """ROOT's stopping tests in ROOT's order: work space, evaluations, then the tolerance.

        ``relerr`` is taken before the tests, because one of them zeroes the result.
        """
        limits, status = self.limits, _DIVIDING
        if self.heap.isbrgs + limits.irgnst > limits.iwk:
            status = 2
        if self.ifncls + 2 * limits.irlcls > limits.maxpts:
            status = 0 if self.last.zero() else 1
            if status == 0:
                self.result = 0.0
        tolerance = relerr < self.eps_rel or self.abserr < self.eps_abs
        return 0 if tolerance and self.ifncls >= limits.minpts else status

    def divide(self, n: int) -> None:
        """Take the worst box off the heap, cut it in two across its axis, and put both back."""
        heap = self.heap
        heap.isbrgn = heap.irgnst
        rgnval, rgnerr, axis, ctr, width = heap.top(n)
        self.abserr -= rgnerr
        self.result -= rgnval
        # ROOT logs "Logic error: idvax0 < 1!" here and carries on with the first axis.
        cut = max(axis, 1) - 1
        width[cut] = 0.5 * width[cut]
        ctr[cut] -= width[cut]
        rgnval, rgnerr = self.measure(ctr, width)
        heap.sift_down(rgnerr)
        heap.store(rgnval, rgnerr, self.axis, (ctr, width))
        ctr[cut] += 2 * width[cut]
        heap.isbrgs += heap.irgnst
        heap.isbrgn = heap.isbrgs
        self.push(ctr, width)


def _box(lows: Any, highs: Any) -> tuple[list[float], list[float]]:
    """The centre and half-widths of the integration box, as ROOT works them out."""
    xmin = [float(x) for x in np.asarray(lows, dtype=np.float64).ravel()]
    xmax = [float(x) for x in np.asarray(highs, dtype=np.float64).ravel()]
    if len(xmin) != len(xmax):
        raise ValueError(
            f"The integral has {len(xmin)} lower limits but {len(xmax)} upper limits;"
            " each dimension needs one of each."
        )
    if not 2 <= len(xmin) <= 15:
        raise ValueError(
            f"AdaptiveIntegratorMultiDim integrates in 2 to 15 dimensions, not {len(xmin)}."
        )
    ctr = [(high + low) * 0.5 for low, high in zip(xmin, xmax)]
    return ctr, [(high - low) * 0.5 for low, high in zip(xmin, xmax)]


def _start(
    n: int, options: tuple[float, float, int | None, int, int]
) -> tuple[_Limits, float, float]:
    """ROOT's constructor defaults for the options left unset, then ``DoIntegral``'s bounds."""
    eps_abs, eps_rel, max_pts, min_pts, size = options
    if max_pts is None:
        max_pts = max_evaluations(n)
    limits = _Limits.resolve(n, min_pts, max_pts or DEFAULT_N_CALLS, size or DEFAULT_WK_SIZE)
    return limits, max(eps_abs, 0.0), eps_rel if eps_rel >= 0 else 1e-9


def adaptive_integral(
    func: Integrand,
    lows: Any,
    highs: Any,
    eps_abs: float = 0.0,
    eps_rel: float = DEFAULT_EPS_REL,
    max_pts: int | None = None,
    min_pts: int = 0,
    size: int = 0,
    vectorized: bool = False,
) -> CubatureResult:
    """``AdaptiveIntegratorMultiDim::Integral(xmin, xmax)``: the integral of ``func`` over a box.

    ``func`` takes a point, a 1-D array of ``ndim`` coordinates, and gives a
    float. With ``vectorized`` it is called once per box with every point
    the rule needs, an ``(npoints, ndim)`` array, and gives their values.

    The defaults are what ``RooAdaptiveIntegratorND`` passes: no absolute
    tolerance, a relative one of ``1e-7``, and an evaluation limit that
    depends on the dimension (see :func:`max_evaluations`). As in ROOT, a
    ``max_pts`` or ``size`` of zero means ``IntegratorMultiDimOptions``'
    default, a negative tolerance means its default, and a ``min_pts`` of
    zero means a single rule.
    """
    ctr, width = _box(lows, highs)
    n = len(ctr)
    limits, eps_abs, eps_rel = _start(n, (eps_abs, eps_rel, max_pts, min_pts, size))
    heap = _Heap(limits.irgnst, [0.0] * (limits.iwk + 10), limits.irgnst, limits.irgnst)
    run = _Run(limits, eps_abs, eps_rel, _evaluator(func, vectorized), heap)
    run.push(ctr, width)
    while True:
        relerr = run.relerr()
        status = run.status(relerr)
        if status != _DIVIDING:
            return CubatureResult(run.result, relerr, status, run.ifncls)
        run.divide(n)


def integrate_nd(
    func: Integrand,
    lows: Any,
    highs: Any,
    eps_rel: float = DEFAULT_EPS_REL,
    vectorized: bool = False,
) -> CubatureResult:
    """An integral as ``RooAdaptiveIntegratorND`` asks for one, with its limit for the dimension."""
    return adaptive_integral(func, lows, highs, 0.0, eps_rel, vectorized=vectorized)


def not_converged_warning(name: str, max_eval: int, relerr: float) -> str:
    """The warning ``RooAdaptiveIntegratorND::integral`` gives when the limit stopped it early.

    RooFit files it under the ``NumericIntegration`` topic (:data:`TOPIC`).
    """
    return (
        f"RooAdaptiveIntegratorND::integral({name}) WARNING: target rel. precision not reached"
        f" due to nEval limit of {max_eval}, estimated rel. precision is {relerr:3.1e}"
    )


def suppressing_further_warnings(name: str) -> str:
    """The notice that follows the last precision warning RooFit is configured to print."""
    return (
        f"RooAdaptiveIntegratorND::integral({name}) Further warnings on target precision are"
        " suppressed conform specification in integrator specification"
    )


def suppressed_warnings_summary(name: str, count: int) -> str:
    """The destructor's count of the precision warnings it did not print, ROOT's spelling kept."""
    return (
        f"RooAdaptiveIntegratorND::dtor({name}) WARNING: Number of suppressed warningings about"
        f" integral evaluations where target precision was not reached is {count}"
    )


class Cubature:
    """``RooAdaptiveIntegratorND``: one integrand over fixed limits, and its precision warnings.

    Each :meth:`integral` is one ``Integral(xmin, xmax)``. A result that hit
    the evaluation limit counts as an error, as RooFit counts it. RooFit's
    constructor reads ``maxWarn`` (5) from its configuration but then sets
    its warning allowance to zero, so ROOT never prints the per-integral
    warning and only reports the count when the integrator is destroyed.
    ``max_warn`` keeps that allowance, zero by default as in ROOT. The
    messages RooFit would print are collected in :attr:`messages`, and
    :meth:`close` adds the destructor's.
    """

    def __init__(
        self,
        func: Integrand,
        lows: Any,
        highs: Any,
        name: str = "",
        eps_rel: float = DEFAULT_EPS_REL,
        vectorized: bool = False,
        max_warn: int = 0,
    ) -> None:
        self.func, self.name, self.vectorized = func, name, vectorized
        self.lows, self.highs = lows, highs
        self.max_eval = max_evaluations(len(np.asarray(lows).ravel()))
        self.eps_rel, self.max_warn = eps_rel, max_warn
        self.n_error = 0
        self.last: CubatureResult | None = None
        self.messages: list[str] = []

    def integral(self) -> float:
        """The integral, counting and perhaps reporting a failure to reach the precision."""
        self.last = adaptive_integral(
            self.func,
            self.lows,
            self.highs,
            0.0,
            self.eps_rel,
            self.max_eval,
            vectorized=self.vectorized,
        )
        if self.last.status == 1:
            self._count_failure(self.last.relerr)
        return self.last.value

    def _count_failure(self, relerr: float) -> None:
        """RooFit's bookkeeping of an integral that stopped at the evaluation limit."""
        self.n_error += 1
        if self.n_error <= self.max_warn:
            self.messages.append(not_converged_warning(self.name, self.max_eval, relerr))
        if self.n_error == self.max_warn:
            self.messages.append(suppressing_further_warnings(self.name))

    def close(self) -> str | None:
        """The destructor's count of suppressed warnings, if any, which also goes in messages."""
        if self.n_error <= self.max_warn:
            return None
        message = suppressed_warnings_summary(self.name, self.n_error - self.max_warn)
        self.messages.append(message)
        return message
