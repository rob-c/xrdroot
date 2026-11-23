"""``TKDE``: ROOT's one-dimensional kernel density estimate.

ROOT's ``TKDE`` (``hist/hist/src/TKDE.cxx``) is a semi-parametric density
estimate: every data point is smeared by a kernel whose width follows
Silverman's rule of thumb, either the same for every point (``kFixed``) or
proportional to ``1/sqrt(density)`` at the point itself (``kAdaptive``). The
data may be used as they are, or first collected into bins - the only way a
sample of a hundred thousand points is cheap - and data pushed against a
declared range may be mirrored back into it, which is what stops the estimate
from sagging at the edges.

The numbers matter, not only the shape: ``gtest/hist/hist/testTKDE`` compares
the estimate with the density the sample came from through chi-square and
Kolmogorov probabilities, so every constant here - the canonical bandwidths
0.7764, 1.7188, ..., the ``(3/(8*sqrt(pi))*n)**-0.2`` factor, the Gaussian
kernel cut off at nine widths - is ROOT's, in ROOT's order of operations.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ...errors import UnsupportedFeatureError
from .objects import TNamed

__all__ = ["TKDE"]

#: ``(2*pi)**-0.5``: the Gaussian kernel's normalisation, as ROOT writes it out.
TWO_PI_INV_ROOT = 0.398942280401432703
#: The narrowest a point's adaptive bandwidth may get, relative to the fixed one.
MIN_WEIGHT_FRACTION = 0.05
#: ``1/sqrt(2*pi)*exp(-0.5)``: the geometric mean of the Gaussian kernel that an
#: adaptive estimate over mirrored data assumes instead of measuring it.
APPROX_GEO_MEAN = 0.241970724519143365
#: The share of the sample that a Gaussian kernel's width is divided by, per
#: Silverman: ``(3/(8*sqrt(pi))*n)**-0.2``.
_KERNEL_RULE = 3.0 / (8.0 * np.sqrt(np.pi))


def _gaussian(x: np.ndarray) -> np.ndarray:
    """The Gaussian kernel at ``x``, zero past nine widths as ROOT's is."""
    inside = (x > -9.0) & (x < 9.0)
    return np.where(inside, TWO_PI_INV_ROOT * np.exp(-0.5 * np.where(inside, x, 0.0) ** 2), 0.0)


def _epanechnikov(x: np.ndarray) -> np.ndarray:
    """``3/4*(1-x*x)`` inside the unit interval, zero outside it."""
    inside = (x > -1.0) & (x < 1.0)
    return np.where(inside, 0.75 * (1.0 - x * x), 0.0)


def _biweight(x: np.ndarray) -> np.ndarray:
    """``15/16*(1-x*x)**2`` inside the unit interval, zero outside it."""
    inside = (x > -1.0) & (x < 1.0)
    return np.where(inside, 15.0 / 16.0 * (1.0 - x * x) ** 2, 0.0)


def _cosine_arch(x: np.ndarray) -> np.ndarray:
    """``pi/4*cos(pi*x/2)`` inside the unit interval, zero outside it."""
    inside = (x > -1.0) & (x < 1.0)
    return np.where(inside, np.pi / 4.0 * np.cos(np.pi / 2.0 * x), 0.0)


#: Each kernel of ROOT's ``EKernelType``, by its enum value.
KERNELS = (_gaussian, _epanechnikov, _biweight, _cosine_arch)
#: ROOT's canonical bandwidths, by kernel, which fix each kernel's smoothing.
CANONICAL_BANDWIDTHS = (0.7764, 1.7188, 2.03617, 1.7663, 1.0)
#: Each kernel's variance, by kernel.
KERNEL_SIGMAS2 = (1.0, 1.0 / 5.0, 1.0 / 7.0, 1.0 - 8.0 / np.pi**2, -1.0)
#: The constructor's option words, and the value each sets, by option kind.
KERNEL_TYPES = ("gaussian", "epanechnikov", "biweight", "cosinearch")
#: ``EIteration``.
ADAPTIVE, FIXED = 0, 1
#: ``EMirror``, in ROOT's order.
MIRRORS = (
    "nomirror",
    "mirrorleft",
    "mirrorright",
    "mirrorboth",
    "mirrorasymleft",
    "mirrorrightasymleft",
    "mirrorasymright",
    "mirrorleftasymright",
    "mirrorasymboth",
)
#: ``EBinning``.
UNBINNED, RELAXED_BINNING, FORCED_BINNING = 0, 1, 2


class TKDE(TNamed):
    """``TKDE``: a kernel density estimate of a one-dimensional sample."""

    #: ``EKernelType``.
    kGaussian, kEpanechnikov, kBiweight, kCosineArch, kUserDefined, kTotalKernels = range(6)
    #: ``EIteration``.
    kAdaptive, kFixed = ADAPTIVE, FIXED
    #: ``EMirror``.
    (
        kNoMirror,
        kMirrorLeft,
        kMirrorRight,
        kMirrorBoth,
        kMirrorAsymLeft,
        kMirrorRightAsymLeft,
        kMirrorAsymRight,
        kMirrorLeftAsymRight,
        kMirrorAsymBoth,
    ) = range(9)
    #: ``EBinning``.
    kUnbinned, kRelaxedBinning, kForcedBinning = UNBINNED, RELAXED_BINNING, FORCED_BINNING

    def __init__(self, events: Any = 0, data: Any = None, *rest: Any) -> None:
        super().__init__()
        self._kernel: np.ndarray | None = None
        self._drawn: Any = None
        self._pdf: Any = None
        self._upper: Any = None
        self._lower: Any = None
        self._graph: Any = None
        weights = rest[0] if len(rest) >= 5 else None
        numbers = rest[1:] if len(rest) >= 5 else rest
        self._instantiate(int(events), data, weights, numbers)

    # -- construction ------------------------------------------------------------------------

    def _instantiate(self, events: int, data: Any, weights: Any, numbers: Any) -> None:
        """ROOT's ``Instantiate``: options, mirroring, binning, then the data itself."""
        xmin, xmax = (float(numbers[0]), float(numbers[1])) if len(numbers) >= 2 else (0.0, 0.0)
        option = str(numbers[2]) if len(numbers) >= 3 else self.DEFAULT_OPTION
        rho = float(numbers[3]) if len(numbers) >= 4 else 1.0
        self._new_data = False
        self._events = np.empty(0)
        self._weights = np.empty(0)
        self._data = np.empty(0)
        self._bin_count = np.empty(0)
        self._n_events = events
        self._n_bins = 1000 if events < 10000 else min(10000, (events // 100) * 10)
        self._use_bins_events = 10000
        self._mean = self._sigma = 0.0
        self._sigma_rob = 0.0
        self._x_min, self._x_max = xmin, xmax
        self._from_data = xmin >= xmax
        self._sum_of_counts = 0.0
        self._adaptive_factor = 1.0
        self._rho = rho
        self._weight_size = 0.0
        self._set_options(option, rho)
        self._check_options()
        self._set_mirror()
        self._set_use_bins()
        self._set_data(data, weights)

    #: The option string ROOT's constructor defaults to.
    DEFAULT_OPTION = "KernelType:Gaussian;Iteration:Adaptive;Mirror:noMirror;Binning:RelaxedBinning"

    def _set_options(self, option: str, rho: float) -> None:
        """The ``KernelType:...;Iteration:...`` words, last first, as ROOT reads them."""
        self._setted = [False, False, False, False]
        self._kernel_type = self.kGaussian
        self._iteration = self.kAdaptive
        self._mirror = self.kNoMirror
        self._binning = self.kRelaxedBinning
        for word in [w for w in option.lower().split(";") if ":" in w]:
            self._one_option(*word.split(":", 1))
        self._rho = rho

    #: How each option word is read: the flag it sets, the words it knows, and ROOT's
    #: value for a word it does not - and the member each kind of word sets.
    OPTIONS: ClassVar[dict[str, tuple[int, tuple[str, ...], int, str]]] = {
        "kerneltype": (0, KERNEL_TYPES, kGaussian, "_kernel_type"),
        "iteration": (1, ("adaptive", "fixed"), ADAPTIVE, "_iteration"),
        "mirror": (2, MIRRORS, kNoMirror, "_mirror"),
        "binning": (
            3,
            ("unbinned", "relaxedbinning", "forcedbinning"),
            RELAXED_BINNING,
            "_binning",
        ),
    }

    def _one_option(self, kind: str, value: str) -> None:
        """One ``kind:value`` word: ROOT keeps its default for a kind or a value it lacks."""
        found = self.OPTIONS.get(kind.strip())
        if found is None:
            return
        slot, known, fallback, member = found
        self._setted[slot] = True
        setattr(self, member, known.index(value.strip()) if value.strip() in known else fallback)

    def _check_options(self) -> None:
        """ROOT's ``CheckOptions``: an impossible option is the default, silently here."""
        if self._iteration not in (ADAPTIVE, FIXED):
            self._iteration = ADAPTIVE
        if not 0 <= self._mirror <= self.kMirrorAsymBoth:
            self._mirror = self.kNoMirror
        if not 0 <= self._binning <= FORCED_BINNING:
            self._binning = RELAXED_BINNING
        if self._rho <= 0.0:
            self._rho = 1.0

    # -- options the setters take -------------------------------------------------------------

    def SetKernelType(self, kern: Any) -> None:
        """``SetKernelType``: the kernel, re-doing the estimate with it."""
        self._kernel_type = int(kern)
        self._check_options()
        self._kernel = None

    def SetIteration(self, iteration: Any) -> None:
        """``SetIteration``: ``kFixed`` or ``kAdaptive``, re-doing the estimate."""
        self._iteration = int(iteration)
        self._check_options()
        self._kernel = None

    def SetMirror(self, mirror: Any) -> None:
        """``SetMirror``: which edges the data are reflected into, re-doing the estimate."""
        self._mirror = int(mirror)
        self._check_options()
        self._set_mirror()
        if self._use_mirroring:
            self._set_mirrored_events()
        self._kernel = None

    def SetBinning(self, binning: Any) -> None:
        """``SetBinning``: whether the data are binned, which the option string also sets."""
        self._binning = int(binning)
        self._check_options()
        self._set_use_bins()

    def SetNBins(self, nbins: Any) -> None:
        """``SetNBins``: how many bins a binned estimate uses."""
        if not int(nbins):
            return
        self._n_bins = int(nbins)
        self._set_use_bins()

    def SetUseBinsNEvents(self, events: Any) -> None:
        """``SetUseBinsNEvents``: from how many events a relaxed estimate bins its data."""
        self._use_bins_events = int(events)
        self._set_use_bins()

    def SetTuneFactor(self, rho: float) -> None:
        """``SetTuneFactor``: a multiplier on every bandwidth - below one smooths less."""
        self._rho = float(rho)
        self._check_options()
        self._kernel = None

    def SetRange(self, xmin: float, xmax: float) -> None:
        """``SetRange``: the range the estimate is drawn over, and that bins the data."""
        if xmin >= xmax:
            return
        self._x_min, self._x_max = float(xmin), float(xmax)
        self._from_data = False
        self._kernel = None

    # -- mirroring and binning --------------------------------------------------------------

    def _set_mirror(self) -> None:
        """Which of ROOT's nine mirrorings are in force: left, right, and each asymmetric."""
        mirror = self._mirror
        self._mirror_left = mirror in (
            self.kMirrorLeft,
            self.kMirrorBoth,
            self.kMirrorLeftAsymRight,
        )
        self._mirror_right = mirror in (
            self.kMirrorRight,
            self.kMirrorBoth,
            self.kMirrorRightAsymLeft,
        )
        self._asym_left = mirror in (
            self.kMirrorAsymLeft,
            self.kMirrorRightAsymLeft,
            self.kMirrorAsymBoth,
        )
        self._asym_right = mirror in (
            self.kMirrorAsymRight,
            self.kMirrorLeftAsymRight,
            self.kMirrorAsymBoth,
        )
        self._use_mirroring = self._mirror_left or self._mirror_right

    def _set_use_bins(self) -> None:
        """Whether the data are binned, and the bins filled afresh when they now are."""
        forced, relaxed = self._binning == FORCED_BINNING, self._binning == RELAXED_BINNING
        self._use_bins = forced or (relaxed and self._n_events >= self._use_bins_events)
        if self._events.size == 0:
            return
        if self._use_bins:
            self._weight_size = self._n_bins / (self._x_max - self._x_min)
            self._set_bin_centres()
            self._set_bin_counts()
        else:
            self._weight_size = 0.0
            self._bin_count = np.empty(0)
            self._data = self._events.copy()
        self._kernel = None

    def _set_bin_centres(self) -> None:
        """``SetBinCentreData``: the centre of each of ``fNBins`` bins over the range."""
        width = (self._x_max - self._x_min) / self._n_bins
        self._data = self._x_min + (np.arange(self._n_bins) + 0.5) * width

    # -- the data ----------------------------------------------------------------------------

    def _set_data(self, data: Any, weights: Any) -> None:
        """ROOT's ``SetData``: the sample, its statistics, and then its bins or mirrors."""
        if data is None:
            return
        self._events = np.asarray(data, dtype=np.float64).ravel()[: self._n_events].copy()
        self._weights = (
            np.empty(0) if weights is None else np.asarray(weights, dtype=np.float64).ravel().copy()
        )
        if self._from_data and self._events.size:
            self._x_min = float(self._events.min())
            self._x_max = float(self._events.max())
        if self._use_bins:
            self._weight_size = self._n_bins / (self._x_max - self._x_min)
            self._set_bin_centres()
        else:
            self._weight_size = 0.0
            self._data = self._events.copy()
        self._compute_data_stats()
        if self._use_mirroring:
            self._set_mirrored_events()
        else:
            self._set_bin_counts()
        self._kernel = None

    def _set_bin_counts(self) -> None:
        """``SetBinCountData``: how much weight each bin holds, and the total in range."""
        inside = (self._events >= self._x_min) & (self._events < self._x_max)
        weights = self._weights if self._weights.size else np.ones(self._events.size)
        self._sum_of_counts = float(weights[inside].sum())
        if self._use_bins:
            self._bin_count = np.bincount(
                self._index(self._events[inside]), weights=weights[inside], minlength=self._n_bins
            ).astype(np.float64)
        elif self._weights.size:
            self._bin_count = self._weights.copy()
        else:
            self._bin_count = np.empty(0)

    def _index(self, x: np.ndarray) -> np.ndarray:
        """``Index``: the bin a value falls in, clamped to the bins there are."""
        return np.clip(
            ((x - self._x_min) * self._weight_size).astype(np.int64), 0, self._data.size - 1
        )

    def _set_mirrored_events(self) -> None:
        """``SetMirroredEvents``: the sample reflected into the range it was drawn over."""
        if self._use_bins:
            self._set_bin_centres()
            self._set_bin_counts()
            self._mirror_bins()
        else:
            self._mirror_events()
        self._kernel = None

    def _mirror_bins(self) -> None:
        """The bins, each reflected about the range's edges, the counts turned round with them."""
        count = self._bin_count
        centre, width = self._data.copy(), self._x_max - self._x_min
        left = (
            np.concatenate([centre - width, centre, np.empty(0)]) if self._mirror_left else centre
        )
        counts = np.concatenate([count[::-1], count, np.empty(0)]) if self._mirror_left else count
        if self._mirror_right:
            left = np.concatenate([left, centre + width])
            counts = np.concatenate([counts, count[::-1]])
        self._data, self._bin_count = left, counts

    def _mirror_events(self) -> None:
        """The sample, each point reflected about an edge, the weights copied with them."""
        original = self._events.copy()
        reflected: list[np.ndarray] = [original]
        if self._mirror_left:
            reflected.insert(0, 2.0 * self._x_min - original)
        if self._mirror_right:
            reflected.append(2.0 * self._x_max - original)
        self._data = np.concatenate(reflected) if len(reflected) > 1 else original.copy()
        self._events = self._data.copy()
        self._set_bin_counts()
        self._events = original

    def _compute_data_stats(self) -> None:
        """``ComputeDataStats``: the mean, the standard deviation and a robust one."""
        if self._events.size == 0:
            return
        weights = self._weights if self._weights.size else np.ones(self._events.size)
        total = float(weights.sum())
        self._mean = float((weights * self._events).sum() / total)
        variance = float((weights * (self._events - self._mean) ** 2).sum() / total)
        self._sigma = float(
            np.sqrt(variance * self._events.size / max(self._events.size - 1.0, 1.0))
        )
        midspread = self._midspread()
        self._sigma_rob = min(self._sigma, midspread / 1.349)

    def _midspread(self) -> float:
        """``ComputeMidspread``: the range between the quartiles, from the sorted sample."""
        if self._events.size < 2:
            return 0.0
        low, high = np.quantile(self._events, [0.25, 0.75], method="linear")
        return float(high - low)

    # -- the estimate ------------------------------------------------------------------------

    def _set_kernel(self) -> None:
        """``SetKernel``: every point's bandwidth, the same one or one per point."""
        n = self._n_bins if self._use_bins else self._n_events
        if not n:
            return
        weight = CANONICAL_BANDWIDTHS[self.kGaussian] * self._sigma_rob * (n * _KERNEL_RULE) ** -0.2
        weight *= (
            self._rho
            * CANONICAL_BANDWIDTHS[self._kernel_type]
            / CANONICAL_BANDWIDTHS[self.kGaussian]
        )
        fixed = np.array([weight], dtype=np.float64)
        self._kernel = fixed if self._iteration != ADAPTIVE else self._adaptive_weights(fixed)

    def _adaptive_weights(self, fixed: np.ndarray) -> np.ndarray:
        """``ComputeAdaptiveWeights``: each point's bandwidth from the density there."""
        self._kernel = fixed
        n = self._data.size
        density = self._kernel_sum(self._data, fixed) / self._sum_of_counts
        weights = np.full(n, fixed[0])
        usable = density > 0.0
        if self._bin_count.size == n:
            usable &= self._bin_count > 0.0
        weights[usable] = fixed[0] / np.sqrt(density[usable])
        weights[usable] = np.maximum(weights[usable], fixed[0] * MIN_WEIGHT_FRACTION)
        weights[~usable] = np.where(density[~usable] > 0.0, weights[~usable], 0.0)
        total = 1.0 + float(np.log(density[usable]).sum())
        self._adaptive_factor = (
            APPROX_GEO_MEAN / self._sigma_rob
            if self._use_mirroring
            else float(np.sqrt(np.exp(total / n)))
        )
        return weights * self._adaptive_factor

    def _ensure_kernel(self) -> np.ndarray:
        """The bandwidths, the estimate made afresh if they are not there yet."""
        if self._kernel is None:
            self._reinit()
        if self._kernel is None:
            raise ValueError("TKDE holds no data, so its density is not defined anywhere")
        return self._kernel

    def _reinit(self) -> None:
        """ROOT's ``ReInit``: an estimate for data filled since, or read from a file."""
        if self._new_data:
            self._init_from_new_data()
            return
        if self._events.size == 0:
            return
        self._set_kernel()

    def _init_from_new_data(self) -> None:
        """ROOT's ``InitFromNewData``: statistics and kernel over the data filled since."""
        self._new_data = False
        self._events = self._data.copy() if not self._use_bins else self._events
        if self._from_data and self._events.size:
            self._x_min = float(self._events.min())
            self._x_max = float(self._events.max())
        self._compute_data_stats()
        self._weight_size = self._n_events / (self._x_max - self._x_min)
        if self._use_mirroring:
            self._set_mirrored_events()
        else:
            self._set_bin_counts()
        self._set_kernel()

    def _kernel_sum(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """The summed kernel at each of ``x``, in ROOT's units - not yet divided by the total."""
        data = self._data
        kernel = KERNELS[self._kernel_type]
        counts = self._bin_count if self._bin_count.size == data.size else np.ones(data.size)
        inverse = (
            1.0 / weights if weights.size == data.size else np.full(data.size, 1.0 / weights[0])
        )
        usable = weights != 0.0 if weights.size == data.size else np.ones(data.size, dtype=bool)
        scale = counts * inverse
        total = np.zeros(np.shape(x), dtype=np.float64)
        for shifted in (0.0, 1.0, 2.0):
            at = self._shifted(data, shifted)
            if at is None:
                continue
            total = total + (kernel((np.asarray(x)[..., None] - at) * inverse) * scale).sum(axis=-1)
        return np.where(usable.any(), total, 0.0)

    def _shifted(self, data: np.ndarray, which: float) -> np.ndarray | None:
        """The data themselves, or reflected about the left or the right edge for ``kAsym*``."""
        if which == 0.0:
            return data
        if which == 1.0:
            return 2.0 * self._x_min - data if self._asym_left else None
        return 2.0 * self._x_max - data if self._asym_right else None

    def __call__(self, x: Any) -> Any:
        """``operator()``: the estimated density at ``x``, or at each of ``x``."""
        weights = self._ensure_kernel()
        values = self._kernel_sum(np.asarray(x, dtype=np.float64), weights)
        result = values / self._sum_of_counts
        return result if np.ndim(x) else float(result)

    def GetValue(self, x: Any) -> Any:
        """``GetValue``: the estimated density at ``x``."""
        return self(x)

    def GetMean(self) -> float:
        """``GetMean``: the mean of the data."""
        if self._new_data:
            self._init_from_new_data()
        return float(self._mean)

    def GetSigma(self) -> float:
        """``GetSigma``: the standard deviation of the data."""
        if self._new_data:
            self._init_from_new_data()
        return float(self._sigma)

    def GetFixedWeight(self) -> float:
        """``GetFixedWeight``: the bandwidth of a ``kFixed`` estimate, else ``-1``."""
        if self._iteration == ADAPTIVE:
            return -1.0
        return float(self._ensure_kernel()[0])

    def GetAdaptiveWeights(self) -> Any:
        """``GetAdaptiveWeights``: every point's bandwidth, or ``None`` for a fixed estimate."""
        if self._iteration != ADAPTIVE:
            return None
        if self._new_data:
            self._init_from_new_data()
        return self._ensure_kernel()

    # -- filling, and the function the estimate is drawn as ---------------------------------

    def Fill(self, data: float, weight: Any = None) -> None:
        """``Fill``: one more point, which the next evaluation makes the estimate over."""
        if self._use_bins:
            return
        self._data = np.append(self._data, float(data))
        if weight is not None or self._weights.size:
            # a weight for one point and none for another is a sample ROOT cannot hold: the
            # points filled without one count as one, and the sample becomes weighted
            if self._weights.size != self._data.size - 1:
                self._weights = np.ones(self._data.size - 1)
            self._weights = np.append(self._weights, 1.0 if weight is None else float(weight))
        self._n_events += 1
        self._new_data = True
        self._kernel = None

    def GetFunction(self, npx: Any = 100, xmin: float = 1.0, xmax: float = 0.0) -> Any:
        """``GetFunction``: the estimate as a ``TF1`` sampled in ``npx`` points."""
        from .funcs import TF1, EAddToList

        low, high = (self._x_min, self._x_max) if xmin >= xmax else (float(xmin), float(xmax))
        name = f"KDEFunc_{self.GetName()}"
        made = TF1(name, self, low, high, 0, 1, EAddToList.kNo)
        if npx > 0:
            made.SetNpx(int(npx))
        made.SetTitle(f"KDE {self.GetTitle()}")
        return made

    def Draw(self, option: str = "") -> None:
        """``Draw``: the estimate's function on the current pad, kept for ``GetDrawnFunction``."""
        self._drawn = self.GetFunction()
        self._drawn.Draw(str(option))

    def GetDrawnFunction(self) -> Any:
        """``GetDrawnFunction``: the ``TF1`` last drawn, whose style can still be changed."""
        return self._drawn

    def GetError(self, x: float) -> float:
        """``GetError``: the pointwise standard deviation of the estimate, refused.

        ROOT integrates the kernel's square over the whole line to get the norm that
        ``f * L2 /(n * bandwidth)`` needs, through its general integrator. That integral,
        the pointwise upper and lower confidence intervals built on it, and the
        Richardson-derivative bias are not computed here: nothing ROOT's own tests ask of
        ``TKDE`` reads them, and a guessed norm would put a plausible number where ROOT's
        is worked out.
        """
        raise UnsupportedFeatureError(
            "TKDE::GetError is not implemented: it needs the integral of the kernel's square "
            "over the whole line, which this does not compute. Use GetValue for the estimated "
            "density, or GetAdaptiveWeights for the bandwidths the estimate was made with"
        )

    def GetBias(self, x: float) -> float:
        """``GetBias``: the leading term of the estimate's bias, refused with ``GetError``."""
        raise UnsupportedFeatureError(
            "TKDE::GetBias is not implemented: it is the estimate's second derivative by "
            "Richardson's method, which this does not compute. Compare the estimate with the "
            "density it comes from instead, as ROOT's own test does"
        )
