# ROOT files

`xrdroot` opens a ROOT file and reads its trees in Python — over `root://`,
`https://`, WebDAV, `s3://` or a local path, with no ROOT and no C++ anywhere
in the way. What it reads comes back as NumPy arrays, and
[goes on](#into-pandas-awkward-arrow-and-polars) to pandas, Awkward, Arrow,
Polars and `hist` in one call. It [writes new files](#writing) too, and
histograms and graphs [draw themselves](#drawing), onto matplotlib axes or into
plain characters. The functions fits are made with [evaluate](#functions) over whole
arrays. [`RDataFrame`](#rdataframe) is ROOT's declarative analysis over
all of it: lazy, in one pass, a batch of entries at a time.

```python
import xrdroot

with xrdroot.open_root("root://eos.example.org//store/events.root") as f:
    tree = f["Events"]
    print(len(tree), tree.keys())
    pt = tree["Muon_pt"].array(0, 10_000)
```

Nothing is downloaded. A tree is a set of *baskets* — compressed blocks of
consecutive entries, one branch at a time — and asking for entries 0 to 10 000
of one branch reads the baskets that hold them and nothing else. That is what
makes it reasonable to walk a hundred-gigabyte file from a laptop: the bytes
that cross the wire are the ones asked for.

## Opening

```python
xrdroot.open_root("/local/path/f.root")  # a path
xrdroot.open_root("davs://dav.example.org/store/f.root")  # any scheme this library speaks
xrdroot.open_root(open("f.root", "rb"))  # an open file, left open
```

A file is a mapping from name to object:

```python
f.keys()  # ['Events', 'metadata']
f.classnames()  # {'Events': 'TTree', 'metadata': 'TH1F'}
f.trees()  # ['Events']
f.tree()  # the only tree, or a KeyError naming the ones there are
f["dir/sub/Events"]  # directories nest, with a path
f["Events;1"]  # an older cycle, when a file kept several
```

`classnames()` is the first thing to try when something will not open: it says
what a name is without reading any of it.

A name that is not a tree is read too, when the file describes the class it
holds — which is the usual case for a C++ class somebody wrote, and for most of
ROOT's own kit as well:

```python
f["tlv"]  # {'TObject': {...}, 'fP': {'fX': 10.0, ...}, 'fE': 40.0}
f["FileSummaryRecord"]  # a std::string key is a str
f["written"]  # a TDatime is a datetime.datetime
f["h1d"]  # a histogram is a Histogram, and a graph a Graph
```

Objects held by pointer are followed, whether the class promises they are there
or writes the name of the class in front of them; a `TList`, `TObjArray`,
`TClonesArray` or `TArray` member is read the way it streams itself rather than
by the members it declares — a `TClonesArray` names the one class it holds at
the front, and a slot never filled comes back as `None` rather than shifting
the rest along; and a container of objects is read one object after another, or
a field at a time when it was written that way — every object's first member,
then every object's second — which is how ROOT writes a container of plain C++
objects and a `TClonesArray` asked to bypass its streamer. A `vector` of pairs
comes back as the list of tuples it is. A class
inside an entry that this reader cannot walk — one the file does not describe,
or one that streams itself some way of its own — comes back as the name of its
class, because the length written in front of it says how to step over it
without touching anything else.

The bytes have to account for themselves: a class that streams itself in some
way of its own leaves the object a different length than the layout says, and
that is refused by name rather than read into a plausible wrong answer.

## Histograms

A `TH1`, `TH2` or `TH3` of any bin type comes back as a `Histogram`: bins,
edges and what is in them, counted the way Python counts.

```python
h = f["h1d"]
h.name, h.title, h.shape  # ('h1d', 'h1d', (10,))
h.values()  # array([  6.6,  72.6, 543.4, ...])
h.errors()  # the same shape, from the sums of squared weights
h.variances(), h.counts()  # squared errors, and the effective number of fills
h.edges()  # 11 edges for 10 bins
h.axes[0].centers()  # where a point would be drawn
h.sum(), h.entries  # 11000.0, 10004.0
len(h), len(h.axes[0])  # 10, 10
```

`values()[0]` is the first bin of the axis, not the underflow. ROOT keeps two
extra bins per axis for what fell off each end, and `values(flow=True)` gives
them back at the ends where ROOT keeps them — so `sum(flow=True)` is everything
that was ever filled and `sum()` is what landed on the axis.

Everything is a NumPy array shaped the way the axes are, x first:
`values()[ix, iy]` is the bin ROOT calls `GetBinContent(ix+1, iy+1)`, and a
three-dimensional one takes a third index.

A histogram filled with weights keeps the sum of their squares, and that is
where `errors()` comes from; one filled without them keeps no such sum, and
then the error on a count of *n* is its square root, which is what ROOT itself
would give. An unevenly binned axis keeps every edge and those are given back
exactly; an evenly binned one keeps none, and they are worked out from the ends.

Every member the histogram was written with is still there under `h.members`,
under the name of the class that declared it, so nothing is hidden by being
tidied away.

A `Histogram` speaks the plotting protocol that `hist`, `boost-histogram` and
`mplhep` share — `kind`, `values()`, `variances()`, `counts()` and `axes`, each
axis a sequence of `(low, high)` bins — so any of those takes one as it stands,
and two methods hand it over outright:

```python
import mplhep

mplhep.histplot(f["h1d"])  # drawn straight from the file
f["h1d"].to_hist()  # a hist.Hist, with Weight storage when it was weighted
values, edges = f["h1d"].to_numpy()  # as numpy.histogram would give them
f["h1d"].density()  # divided by bin width and total, integrating to one
```

### Profiles

A `TProfile`, `TProfile2D` or `TProfile3D` comes back as a `Profile`, which is
a `Histogram` whose bins hold the mean of something rather than a count. The
file keeps no means: it keeps the sum of `w*y` per bin, the sum of `w*y*y`,
and the sum of the weights, and a profile read as a plain histogram would plot
the first of those and be wrong without looking wrong. So `values()` divides
them out:

```python
p = f["p1d"]
p.kind  # 'MEAN', in the plotting protocol's words
p.values()  # the mean in each bin, zero in an empty one
p.bin_entries()  # the sum of the weights in each - GetBinEntries
p.counts()  # the effective number of entries, (sum w)**2 / sum w**2
p.errors()  # the error on each mean, by the profile's own error option
p.errors(error_mode="s")  # or another: '', 's', 'i' or 'g', as SetErrorOption
p.spread()  # the standard deviation of what went into each bin
p.to_hist()  # a hist.Hist of Mean storage, WeightedMean when it was weighted
```

The error is ROOT's `GetBinError`, whichever of the four the profile was saved
with (`p.error_mode`): the error on the mean, the spread, the spread with a
floor of `1/sqrt(12)` for identical integers, or one over the root of the
weights. `variances()` is its square. `sum()` and `density()` refuse, since
a sum of means is not a thing; `sums()` has what the file stored.

### Efficiencies

A `TEfficiency` comes back as an `Efficiency`: the two histograms it was
filled into, and the efficiency and its confidence interval worked out from
them the way ROOT works them out — which is to say, not stored in the file at
all.

```python
eff = f["trigger"]
eff.passed, eff.total  # the two Histograms
eff.values()  # passed / total, zero where nothing was tried
eff.intervals()  # (low, high): by fStatisticOption, at fConfLevel
eff.intervals(level=0.95, method="wilson")
eff.errors()  # (below, above): the error bars, distance to each end
eff.method, eff.level  # ('clopper-pearson', 0.682689492137)
```

The methods are ROOT's: `clopper-pearson` (the default, and exact), `normal`,
`wilson`, `agresti-coull`, and the Bayesian `jeffreys`, `uniform` and
`bayesian` — the last with the Beta prior the object was saved with, bin by bin
when it was given priors per bin. For a Bayesian method `values()` is the
posterior's mean, or its mode when the object says so. None of them can leave
[0, 1], which is the point: dividing two histograms and propagating their
errors gives an interval past one with no width at zero.

The Beta quantile the exact and Bayesian intervals need is worked out here in
plain Python, since SciPy is not a dependency; it agrees with gonum's to about
one part in 10^13. Feldman-Cousins, mid-P, the shortest Bayesian interval and
an efficiency filled with weights are refused by name rather than
approximated.

### Sparse histograms

A `THnSparse` of any content type comes back as a `SparseHistogram`: its axes,
and the bins something fell into, since in ten dimensions the grid would not
fit in any machine.

```python
hn = f["hn"]
hn.axes, hn.shape  # one Axis per dimension, and their bin counts
hn.coordinates()  # one row per filled bin, a column per axis
hn.values(), hn.variances()  # what is in each, in the same order
hn.to_dense()  # the grid, for one small enough to have one
```

Coordinates count from zero the way a `Histogram` does, so `-1` is an
underflow and `len(axis)` an overflow. `to_dense(flow=True)` keeps the flow
bins, and a grid of more than ten million cells is refused.

## Building and filling histograms

A histogram is also something to fill and compute with, the way ROOT's `TH1`
is. Book one, fill it an array at a time, and ask it what ROOT would be asked:

```python
from xrdroot import Efficiency, Histogram, Profile

h = Histogram.book("pt", (100, 0.0, 200.0), title="p_{T};p_{T} [GeV];events")
h.fill(pt)  # arrays or numbers
h.fill(pt, weight=w)  # the sums of squares start at the first weight not one
h.mean(), h.std(), h.mean_error(), h.effective_entries
h.integral(), h.integral(1, 10, width=True), h.integral_error()

h2 = Histogram.book("map", (50, -2.5, 2.5), [0, 10, 20, 50, 100], kind="F")
h2.fill(eta, pt)
h2.projection_x(), h2.profile_x(), h2.rebin(5, 2)

p = Profile.book("pz", (100, -4, 4), value_range=(0, 20), error_option="s")
p.fill(px, pz)  # the coordinates, then the value averaged

eff = Efficiency.book("trigger", (20, 0, 100))
eff.fill(fired, pt)  # whether each entry passed, then where it is
```

| ROOT | xrdroot |
| --- | --- |
| `TH1D h("h", "t", 100, 0, 1)` | `Histogram.book("h", (100, 0, 1), title="t")` |
| `TH1F`, `TH1I`, `TH1S`, `TH1C` | `Histogram.book(..., kind="F")`, `"I"`, `"S"`, `"C"` |
| `TH1D h("h", "", n, xbins)` | `Histogram.book("h", xbins)` — edges as a list or array |
| `TH2D h("h", "", 10, 0, 1, 20, 0, 2)` | `Histogram.book("h", (10, 0, 1), (20, 0, 2))` |
| `TProfile p("p", "", 100, -4, 4, 0, 20, "s")` | `Profile.book("p", (100, -4, 4), value_range=(0, 20), error_option="s")` |
| `TEfficiency e("e", "", 20, 0, 100)` | `Efficiency.book("e", (20, 0, 100))` |
| `TEfficiency(passed, total)` | `Efficiency.from_histograms(passed, total)` |
| `h->Fill(x)`, `h->Fill(x, w)` | `h.fill(x)`, `h.fill(x, weight=w)` — arrays or numbers |
| `p->Fill(x, y, w)` | `p.fill(x, y, weight=w)` |
| `e->Fill(passed, x)`, `FillWeighted` | `e.fill(passed, x)`, `e.fill(passed, x, weight=w)` |
| `GetEntries`, `GetEffectiveEntries` | `h.entries`, `h.effective_entries` |
| `GetMean(1)`, `GetStdDev(2)` | `h.mean(0)`, `h.std(1)` — axes from zero |
| `GetMeanError`, `GetStdDevError` | `h.mean_error()`, `h.std_error()` |
| `GetSkewness`, `GetKurtosis` | `h.skewness()`, `h.kurtosis()`, and their `_error()`s |
| `Integral()`, `Integral(a, b, "width")` | `h.integral()`, `h.integral(a, b, width=True)` |
| `IntegralAndError` | `h.integral_error(...)` |
| `FindBin(x, y)`, `Interpolate(x)` | `h.find_bin(x, y)`, `h.interpolate(x)` — vectorised |
| `GetMaximum`, `GetMaximumBin` | `h.maximum()`, `h.argmax()` — an index into `values()` |
| `h->Add(h2)`, `h->Add(h2, c)` | `h.add(h2)`, `h.add(h2, c)`, or `h + h2`, `h - h2`, `h += h2` |
| `h->Scale(c)`, `h->Scale(c, "width")` | `h.scale(c)`, `h.scale(c, width=True)`, or `h * c`, `h / c` |
| `h->Multiply(h2)`, `h->Divide(h2)` | `h.multiply(h2)`, `h.divide(h2)`, or `h * h2`, `h / h2` |
| `h->Divide(pass, total, 1, 1, "B")` | `passed.divide(total, binomial=True)` |
| `h->Scale(1 / h->Integral())` | `h.normalized()`, `h.normalized(width=True)` for a density |
| `hadd`, `TH1::Merge` | `Histogram.merge([h1, h2, ...])`, `sum([h1, h2])`, `Profile.merge` |
| `h->Reset()`, `h->Clone("c")` | `h.reset()`, `h.copy("c")` |
| `h->Rebin(4)`, `h->Rebin(n, "", xbins)` | `h.rebin(4)`, `h.rebin(xbins)` |
| `h2->Rebin2D(2, 5)` | `h2.rebin(2, 5)` |
| `h2->ProjectionX("", 1, 5)` | `h2.projection_x(y_range=(1, 5))` |
| `h3->Project3D("yx")`, `Project3D("x")` | `h3.projection("xy")`, `h3.projection("x")` |
| `h2->ProfileX()`, `ProfileY()` | `h2.profile_x()`, `h2.profile_y()` |
| `TGraphErrors(h)`, `e->CreateGraph()` | `Graph.from_histogram(h)`, `Graph.from_histogram(e)` |

The numbers are ROOT's, to the bit: the suite runs go-hep's ROOT macros for
`tefficiency.root` and `tprofile.root` again, drawing `gRandom`'s very numbers,
and what comes out matches every bin, square of weights, running sum and entry
count ROOT wrote. That is because the bookkeeping is ROOT's. An entry off the
end of an axis goes to its flow bin — the upper edge of the last bin to the
overflow, and a NaN too, since ROOT's `FindBin` asks `!(x < xmax)` — and
counts towards `entries` but not towards the moments, which is ROOT's default
of leaving the flow out of the statistics. The mean and spread are made from
the running sums the fills added to, and only when those are gone — a
histogram whose bins were set by hand, or whose sums an operation rebuilt —
from the bin centres. Every running total is added in the order ROOT would
have met the entries, rather than pairwise as NumPy would. Integer storage
saturates at ROOT's limits and takes a weight's whole part, and a `TH1F` adds
in single precision.

Filling and ROOT's other in-place methods — `add`, `scale`, `multiply`,
`divide`, `reset` — change the histogram they are called on, as ROOT's do;
the operators, `copy`, `normalized`, `rebin` and the projections make a new
one. The members stay the whole of a histogram's state throughout: filling
changes the arrays and the sums they hold, so what is written is exactly what
was computed. The arithmetic propagates errors as ROOT does and refuses two
histograms binned differently, by name; it refuses profiles too, whose bins
are means — `Profile.merge` adds profiles up as `hadd` does. Rebinning onto
new edges refuses an edge the old axis does not have, since merging bins can
never split one. `projection("xy")` keeps the axes in the order named, which
is the reverse of ROOT's `Project3D` letters.

### Comparing two histograms

ROOT's two tests of whether two histograms show the same thing are here as
ROOT's code has them, down to the order the sums are added in:

```python
data.chi2_test(mc, "UW")  # the p-value: a count against a weighted histogram
data.chi2_test(mc, "UW CHI2/NDF")  # or the chi-square over its degrees of freedom
found = data.chi2_test_full(mc, "UW")  # Chi2TestX: chi2, ndf, igood, p, residuals
data.kolmogorov_test(mc)  # the probability that the shapes agree
data.kolmogorov_test(mc, "M"), data.kolmogorov_test(mc, "UON")
xrdroot.stats.prob(chi2, ndf), xrdroot.stats.kolmogorov_prob(z)
```

| ROOT | xrdroot |
| --- | --- |
| `h1->Chi2Test(h2, "UU")` | `h1.chi2_test(h2)` — `"UU"`, `"UW"`, `"WW"`, `"NORM"`, `"UF"`, `"OF"`, `"P"` |
| `h1->Chi2Test(h2, "CHI2")`, `"CHI2/NDF"` | `h1.chi2_test(h2, "CHI2")`, `"CHI2/NDF"` |
| `h1->Chi2TestX(h2, chi2, ndf, igood, "UW", res)` | `h1.chi2_test_full(h2, "UW")` — a named tuple, `residuals` an array |
| `h1->KolmogorovTest(h2, "UON")` | `h1.kolmogorov_test(h2, "UON")` — `"U"`, `"O"`, `"N"`, `"M"`, `"D"`, `"X"` |
| `TMath::Prob(chi2, ndf)` | `xrdroot.stats.prob(chi2, ndf)` |
| `TMath::KolmogorovProb(z)` | `xrdroot.stats.kolmogorov_prob(z)` |
| `ROOT::Math::inc_gamma(a, x)`, `inc_gamma_c` | `xrdroot.stats.incomplete_gamma(a, x)`, `incomplete_gamma_c` |
| `TH1::SmoothArray(n, xx, ntimes)` | `xrdroot.stats.smooth_array(xx, ntimes)` |

The chi-square is Gagunashvili's, as ROOT's is: `"UU"` for two counts —
the default — `"UW"` for a count against a weighted histogram, the count
first, and `"WW"` for two weighted ones, with an option naming none of them
letting the histograms' own sums say which. A bin empty in both is skipped
and takes a degree of freedom with it; where the weighted comparison would
divide zero by zero, ROOT adds an entry to the count and to its total, for
the rest of the bins too, and so does this. The test on `tutorials/math/chi2test.C`
gives ROOT's documented 21.09 with a p-value of 0.33. `TMath::Prob` is the
incomplete gamma function of Cephes that ROOT's MathCore carries, and
`TMath::KolmogorovProb` CERNLIB's `PROBKL` as ROOT translated it; neither needs
SciPy.

Where ROOT prints an error and returns zero — histograms binned differently,
an empty one, errors of zero on both sides — this refuses with the reason,
since a zero is a p-value and would be read as one. Profiles are refused.
ROOT's warnings, for a test of counts asked of weighted histograms, are
Python's `RuntimeWarning`. Option `"X"` of the Kolmogorov test is ROOT's
procedure — pseudo-experiments drawn from the histogram of more entries, the
fraction straying further than the data — but the draws are NumPy's,
seeded with `TRandom3`'s default of 4357 unless `seed=` says otherwise, so the
number repeats from run to run and is not the one `gRandom` would give. An axis
range set with `SetRange` is not applied: every bin on the axis is compared.

### Indexing and slicing

A histogram is indexed the way [UHI](https://uhi.readthedocs.io/en/latest/indexing.html)
says, as `hist` and `boost-histogram` are — and their `bh.loc`, `bh.rebin`,
`bh.underflow` and `bh.overflow` work here too:

```python
from xrdroot import loc, overflow, rebin, underflow

h[3], h[-1]  # a bin's content, counted from zero
h[underflow], h[overflow], h[loc(1.5)], h[loc(1.5) + 1]
h[2:8], h[loc(0.5) : loc(2.0)]  # a new histogram: what is cut away goes to the flow
h[::rebin(2)], h[2:8:rebin(3)]  # bins merged, those left over to the overflow
h[::sum], h[0:len:sum]  # a number: flow and all, or only the axis
h2[:, sum], h2[{1: sum}], h2[3, :]  # an axis summed away, or one bin of it
h[5] = 7.0; h[...] = values  # SetBinContent, flow too when two longer
```

| ROOT | xrdroot |
| --- | --- |
| `h->GetBinContent(i + 1)` | `h[i]` |
| `h->GetBinContent(0)`, `(nbins + 1)` | `h[underflow]`, `h[overflow]` |
| `h->GetBinContent(h->FindBin(x))` | `h[loc(x)]` |
| `h->SetBinContent(i + 1, v)` | `h[i] = v` |
| `h->Rebin(2)` | `h[::rebin(2)]` |
| `h2->ProjectionX()`, `ProjectionX("", 1, ny)` | `h2[:, sum]`, `h2[:, 0:len:sum]` |
| `h->Integral(0, nbins + 1)`, `Integral()` | `h[::sum]`, `h[0:len:sum]` |

The bookkeeping is ROOT's wherever ROOT has the same operation. An axis
summed away is ROOT's projection with that range, keeping its rules for what
the result's running sums and entries are. A cut or a rebinning is `Rebin`
with bins falling into the flow: the entries stay, since every fill is still
in some bin, and the running sums stay while nothing moved into the flow and
are made from the bins once anything did. Setting bins is `SetBinContent`
bin by bin — one more entry each, the running sums made from the bins, the
squares of the weights left alone. A profile is cut and rebinned by its sums,
so its means come out right, and a bin of it is its mean; summing its bins,
or setting one, is refused.

### A histogram as a distribution

```python
h.cumulative()  # GetCumulative: every bin up to each one
h.cumulative(forward=False, suffix="_eff")  # from each one to the end
h.quantiles([0.25, 0.5, 0.75])  # GetQuantiles
h.smooth(2)  # Smooth: 353QH twice, done two times over, in place
h.sumw2(), h.sumw2(False)  # start or stop keeping the squares of the weights
```

| ROOT | xrdroot |
| --- | --- |
| `h->GetCumulative()`, `GetCumulative(kFALSE, "_eff")` | `h.cumulative()`, `h.cumulative(False, "_eff")` |
| `h->GetQuantiles(n, xq, p)` | `xq = h.quantiles(p)` |
| `h->GetQuantiles(nbins + 1, xq)` | `xq = h.quantiles()` |
| `h->Smooth(ntimes)` | `h.smooth(ntimes)` |
| `h->Sumw2()`, `h->Sumw2(kFALSE)` | `h.sumw2()`, `h.sumw2(False)` |

A cumulative of two or three axes is every bin below and to the left of each,
by the inclusion and exclusion ROOT 6.42 uses, each neighbour read back as it
was stored, so a `TH1F`'s rounds as ROOT's does. A quantile inside a bin is
on the straight line across it, and at a probability the distribution reaches
exactly it is ROOT's choice: past any empty bins at zero, the middle of an
empty stretch elsewhere. Smoothing is HBOOK's `hsmoof` as ROOT translated it —
running medians of three, five and three, a quadratic over the plateaus they
leave, a running mean, and all of it again on what was left over — and it
changes the contents only, as ROOT's does: the errors, the running sums and
the entries still describe what was filled.

## Random numbers

A macro that generates anything — a toy study, a smearing, a histogram filled
from `gRandom` — can only be checked against what it produced with the
generator it was written for. `xrdroot.random` is ROOT's generators, to the
bit: the same seed gives the same `Rndm()` values in the same order, and every
distribution follows ROOT's own algorithm with ROOT's constants, so it takes
the same draws and returns the same numbers. Every call takes `n` for an array
of that many — exactly the numbers `n` calls in ROOT would give, leaving the
generator exactly where they would — and without it gives one.

```python
from xrdroot import gRandom, TRandom3

r = TRandom3(4357)                  # gRandom's own seed
r.rndm(5)                           # five gRandom->Rndm()
smeared = r.gaus(pt, 0.02 * pt)     # one Gaus(pt[i], 0.02*pt[i]) per entry
counts = r.poisson(3.2, n=10_000)
x, y = r.rannor(1000)
```

| ROOT | xrdroot |
| --- | --- |
| `gRandom` | `xrdroot.gRandom`, a `TRandom3` at seed 4357 |
| `TRandom3 r(seed)`, `TRandom2`, `TRandom1(seed, lux)`, `TRandom` | `TRandom3(seed)`, `TRandom2(seed)`, `TRandom1(seed, lux)`, `TRandom(seed)` in `xrdroot.random` |
| `r.Rndm()`, `r.RndmArray(n, a)` | `r.rndm()`, `r.rndm(n)` |
| `Uniform(x1)`, `Uniform(x1, x2)` | `r.uniform(x1)`, `r.uniform(x1, x2)` |
| `Gaus(mean, sigma)` | `r.gaus(mean, sigma)` |
| `Rannor(a, b)` | `a, b = r.rannor()` — the `Float_t` overload's are `np.float32(a)` |
| `Exp(tau)`, `Integer(imax)` | `r.exp(tau)`, `r.integer(imax)` |
| `Poisson(mean)`, `PoissonD(mean)` | `r.poisson(mean)`, `r.poisson_d(mean)` |
| `Binomial(ntot, prob)` | `r.binomial(ntot, prob)` |
| `Landau(mean, sigma)`, `BreitWigner(mean, gamma)` | `r.landau(mean, sigma)`, `r.breit_wigner(mean, gamma)` |
| `Circle(x, y, r)`, `Sphere(x, y, z, r)` | `x, y = r.circle(radius)`, `x, y, z = r.sphere(radius)` |
| `h->GetRandom(rng)` | `rng.from_distribution(h.values(), h.axes[0])` |
| `SetSeed(s)`, `GetSeed()` | `r.set_seed(s)`, `r.get_seed()` |
| the object written to a file | `r.state` — ROOT's data members by name — and `r.set_state(...)`, or pickle it |

Parameters can be arrays: `r.gaus(means, sigmas)` draws one number for each.
The ones that decide how many draws a number takes — Poisson's mean,
Binomial's `ntot` and `prob`, Landau's `sigma` — are one number for the whole
call, since ROOT's algorithm for a mean of 3 and one for a mean of 300 take
their draws differently.

**How exact.** `TRandom3` is the Mersenne Twister; its words are NumPy's
legacy `MT19937`'s, which knows nothing of ROOT, the first few for three seeds
are the ones go-hep's `rrand` pins, and filling histograms from it at seed
4357 reproduces, bin for bin and bit for bit, the files ROOT macros wrote in
the test data (`tefficiency.root`, `tprofile.root` — `Rndm` and `Rannor`).
All four generators are held draw for draw, over thousands of draws taken in
every mixture of one at a time and arrays, to ROOT's C++ transcribed a
statement at a time — `TRandom1` at every luxury level, down to its ring of
24 floats and its carry. The distributions are ROOT's code with ROOT's
arithmetic in ROOT's order; `Gaus`, `Exp`, `Rannor` and `Poisson` reproduce
go-hep's pinned numbers, which come from its own transcription of the same
C++. `log`, `exp`, `sin`, `cos` and `tan` are the C library's, which is what
ROOT calls: NumPy's are used only after they have been seen to agree with it
exactly on this machine, and `LnGamma` is the C library's `lgamma` rather than
Python's own, which differs from it in the last place about half the time.
No ROOT-made numbers for `Gaus`, `Poisson` or `Landau` were to hand, so those
rest on the transcription rather than on ROOT's output.

**How fast.** Everything is an array at a time. `TRandom3` makes words with
NumPy's compiled twister from ROOT's state: ten million draws take about
0.2 s. `TRandom2` steps a few thousand streams side by side, each started
further on by a jump matrix (about 0.6 s for ten million); `TRandom1` runs
RANLUX as the LCG it is on 576-bit numbers, one big-integer multiplication per
24 numbers (about 0.4 s for a million, where ROOT's loop costs 80 ns a draw).
The rejection algorithms — `Gaus`, `Poisson`, `Sphere` — work out, for every
draw, what the number starting there would be and where the next would start,
then follow those links from the first draw by pointer doubling, so they take
exactly ROOT's draws without a Python loop per draw: a million `Gaus` in
about 0.5 s, a million `Poisson(100)` in about 1 s. `Poisson` below a mean of
25 multiplies draws until the product crosses `exp(-mean)`, about mean + 1
draws a count, and every product is formed in ROOT's order before it is
believed, so it costs about 0.2 µs per draw it uses: a million `Poisson(3)` in
about 1 s, `Poisson(24)` in about 5 s. One number at a time goes through
ROOT's loop written out in Python instead — a couple of microseconds for a
`Gaus`. Draws are made ahead and held; `state`, `get_seed` and pickling
report the generator after exactly the draws handed out.

**Seeds.** A seed is ROOT's: an unsigned 32-bit number to a constructor, an
unsigned 64-bit one to `set_seed`, cut to 32 bits where ROOT's member is a
`UInt_t`. A seed of 0 asks ROOT for an unrepeatable stream, taken from a UUID
— `TRandom3` fills its state from a `TRandom2` seeded that way and throws the
first ten draws away — and that is what it does here, from a random UUID; it
is a valid stream but never the same twice, and never ROOT's. `get_seed`
answers what ROOT's `GetSeed` does, which for `TRandom3` is the next word of
state: 4357 only until the first draw. `TRandom1()` with no seed takes the
next of ROOT's table of 215 seeds, as ROOT's default constructor does.

`from_distribution(contents, axis, n)` is `TH1::GetRandom`: the cumulative
integral summed in bin order, the bin found as `TMath::BinarySearch` finds it,
and the number placed on a straight line within it, with an even axis's edges
worked out from its ends the way `TAxis` does. `width=True` is ROOT's
`"width"` option; a histogram with nothing in it gives 0 and draws nothing,
and one with a negative bin is refused, where ROOT would draw from a NaN
integral. ROOT's newer engines — `TRandomMixMax`, `TRandomRanluxpp`,
`TRandomMT64` — are not here.

## Graphs

A `TGraph`, `TGraphErrors`, `TGraphAsymmErrors` or `TGraphMultiErrors` comes
back as a `Graph`, which is a sequence of points:

```python
g = f["tge"]
len(g), g[0]  # 4, (1.0, 2.0)
for x, y in g:
    ...
g.x, g.y  # NumPy arrays, one value per point
g.points()  # [(1.0, 2.0), (2.0, 4.0), ...]
below, above = g.yerr  # the bars either side, or None if none were kept
```

`xerr` and `yerr` are always a pair, low side first, so the same code reads a
graph whose bars are symmetric and one whose are not — a graph that kept one
array for both sides gives that array twice. A `TGraph` proper kept none, and
both are `None`.

A `TGraphMultiErrors` keeps its y errors in layers — statistical and
systematic, say — and `g.layers` is those pairs of bars in the order they were
added. Asking such a graph for `yerr` raises rather than picking a layer or
summing them for you, because how to combine them is physics, not format. For
every other graph `layers` is simply `(yerr,)`, or `()` when none were kept.

A graph works out what ROOT's `TGraph` does from its points:

```python
g.eval(x)  # Eval: straight lines between points, and past the ends along them
g.integral(), g.integral(0, 5)  # the area of the polygon the points make
g.sort()  # in increasing x, error bars and all, in place
g.mean(), g.rms(1)  # GetMean(1), GetRMS(2): of the points, unweighted
```

`eval` walks the points in the order they were added, as ROOT's `Eval`
does without `SetBit(kIsSortedX)`: past an end it extrapolates along the two
points that walk found last, which for points out of order are not always the
two nearest. Sorting keeps points at the same `x` in the order they had, where
ROOT's comparison leaves them to its C++ library.

### Several at once

A `TMultiGraph` comes back as a `MultiGraph`, which is the sequence of the
graphs it holds, and a `THStack` as a `Stack`, the sequence of its histograms,
each in the order they were added; the frame they were drawn in is under
`.members` with everything else.

```python
[g.classname for g in f["mg"]]  # ['TGraph', 'TGraphErrors', 'TGraphAsymmErrors']
sum(h.values() for h in f["stack"])  # what the stack's top edge is
```

Every one of these classes — a histogram, profile, efficiency, sparse
histogram, graph, multigraph, stack or entry list — met *inside* another
object comes back as that class too, the same as it would standing in a key of
its own: the graphs of a `MultiGraph` are `Graph`s, and the two histograms of
an `Efficiency` are `Histogram`s.

## Functions

ROOT's `TF1`, `TF2` and `TF3` — and the `TFormula` inside each — come back
as a `Function`: a formula, a value for each of its parameters, a range for
each of its variables and, once fitted, what the fit left behind. It is
built by hand the way `TF1`'s constructor builds one, read from a file on
its own or among the fits a histogram or graph carries, and evaluated over
whole arrays at once:

```python
from xrdroot import Function

f = Function("peak", "gaus(0) + pol1(3)", range=(0, 10), parameters=[40, 5, 0.5, 2, 0.1])
f(5.0), f(np.linspace(0, 10, 101))  # a number for a number, an array for an array
f.parameter_names  # ('p0', 'p1', 'p2', 'p3', 'p4')
f.gradient(xs)  # (n, npar): exact for sums, products, powers, exp, log, ...
f.integral(0, 10), f.maximum(), f.x_at(20.0), f.derivative(5.0)

with xrdroot.open_root("fitted.root") as file:
    fit = file["h_mass"].functions[0]  # the TF1 ROOT's Fit left on the histogram
    fit.parameters, fit.parameter_errors, fit.fit_result["chi2"]

decay = Function.from_callable("decay", lambda x, p: p[0] * np.exp(-x / p[1]), 2)
```

| ROOT | xrdroot |
| --- | --- |
| `TF1 f("f", "[0]*exp(-x/[tau])", 0, 10)` | `Function("f", "[0]*exp(-x/[tau])", range=(0, 10))` |
| `TF2 f("f", "x*y", 0, 1, 0, 2)` | `Function("f", "x*y", range=((0, 1), (0, 2)))` |
| `TF1 f("f", cppfunction, 0, 10, 2)` | `Function.from_callable("f", fn, 2, range=(0, 10))` — `fn(x, params)` |
| `f->Eval(x)`, `f->EvalPar(x, p)` | `f(x)`, `f.evaluate(x, p)` — arrays, `(n,)` or `(n, dimensions)` |
| `f->SetParameters(...)`, `SetParameter("mean", v)` | `f.set_parameters(...)`, `f.set_parameters(mean=v)`, `f.parameters = [...]` |
| `GetParName(i)`, `SetParNames(...)` | `f.parameter_names`, `f.parameter_names = (...)` |
| `GetParErrors()`, `SetParLimits(i, a, b)` | `f.parameter_errors`, `f.set_limits(i, a, b)`, `f.parameter_limits` |
| `FixParameter(i, v)`, `ReleaseParameter(i)` | `f.fix(i, v)`, `f.release(i)`, `f.fixed` |
| `GetChisquare()`, `GetNDF()`, `GetNumberFitPoints()` | `f.fit_result` — `chi2`, `ndf`, `npfits`, `parameters`, `errors` |
| `GradientPar(x, grad)` | `f.gradient(x)` — every parameter at once |
| `Integral(a, b)` | `f.integral(a, b)` — either end may be infinite |
| `Derivative(x)` | `f.derivative(x)` |
| `GetMaximum()`, `GetMaximumX()`, `GetMinimum()`, `GetMinimumX()` | `f.maximum()`, `f.maximum_x()`, `f.minimum()`, `f.minimum_x()` |
| `GetX(y)` | `f.x_at(y)` |
| `SetNormalized(true)` | `f.normalized = True` |
| `Clone("g")` | `f.copy("g")` |
| `h->GetListOfFunctions()`, `h->Fit(f)` leaving `f` on `h` | `h.functions`, `h.attach(f)` — and a graph's the same |

The language is ROOT 6's: `x`, `y` and `z` — or `x[0]`, `x[1]`, `x[2]` —
parameters by number `[0]` or by name `[mean]`, `^` and `**` for a power,
`TMath`, `<cmath>` and the `ROOT::Math` densities, the constants `pi`, `e`,
`ln10`, `sqrt2`, `infinity` and the rest, and ROOT's predefined shapes with
the parameter meaning and names ROOT gives them: `gaus` and the normalised
`gausn`, `xygaus`, `xyzgaus` and `bigaus`, `expo` and `xyexpo`, `landau` and
`landaun`, `crystalball` and `crystalballn`, `breitwigner`, `pol0` to `polN`
and `cheb0` to `cheb10`. A shape takes where its parameters start,
`gaus(3)`, its variable, `pol1(y, 0)`, or its parameters' names,
`pol1(x, [A], [B])`. The shapes are written out as ROOT writes them —
`gaus` becomes `[Constant]*exp(-0.5*((x-[Mean])/[Sigma])*((x-[Mean])/[Sigma]))`,
exactly the text ROOT keeps in the file — and the densities are ROOT's own,
operation for operation: CERNLIB's rational approximation of the Landau,
the Crystal Ball's two sides and its `n > 1` normalisation, Clenshaw's sum
for a Chebyshev series.

The gradient is exact wherever the formula is built from arithmetic and the
elementary functions, which covers the polynomials, `gaus`, `expo` and any
sum or product of them; where it calls something with no derivative written
down, such as `TMath::Landau`, the parameters inside that call are
differentiated as `TF1::GradientPar` does it — two central differences of
steps `h` and `h/2`, combined, with `h` a hundredth of the parameter's error
— and only those. Integrals are adaptive 21-point Gauss-Kronrod to
`TF1::Integral`'s tolerance of `1e-12`; extrema and `x_at` are
`BrentMinimizer1D`'s scan of `fNpx` points and Brent's search inside the
bracket; the derivative is `RichardsonDerivator`'s, with a step of a
thousandth of the range. A normalised function is divided by its integral,
kept up to date as its parameters change.

A `TF1` of C++ code — a function pointer, a convolution, a normalised sum —
cannot bring its code into the file, and ROOT writes the function sampled
over its range instead. Such a function reads back as those samples, and is
the straight line between the two either side of a point, zero outside the
range, as `TF1::GetSave` makes it; a function made here from Python code is
written the same way, and reads back — here or in ROOT — the same way. The
checks are ROOT's own numbers: `tgme.root`'s `pol1` fit saved a hundred and
one evaluations of itself, and the formula evaluated here gives every one of
them to a part in 10^12.

What is written is a `TF1` of either kind, alone or in a histogram's or
graph's list of functions, in the layouts ROOT 6.24 wrote into `tgme.root`
and `tformula.root`; a `TF1` ROOT wrote is written back byte for byte, but
for the one boolean ROOT left uninitialised and so wrote as `0x99`. What
is refused, by name: writing a `TF2` or `TF3`, which no file here describes
for a writer to copy; evaluating a formula that calls what this does not
know, or a physical constant whose value ROOT has changed between releases;
evaluating a function of code that saved nothing, or saved its samples at
the bins of the histogram it was fitted to; and a `TF1NormSum` or
`TF1Convolution` on its own, which comes back as its members. A `TF1` from
ROOT 5, which was a `TFormula` rather than holding one, comes back as its
members too, so the histogram it hangs off still reads.

## Drawing

ROOT draws through a `TCanvas`, which this library does not carry. What it
has instead is the two ways Python usually looks at data. `plot()` draws onto
matplotlib axes — made on demand, or brought along — and returns them, so
styling and saving carry on where it left off:

```python
ax = f["h1d"].plot()  # steps for 1D, a shaded mesh for 2D
f["tge"].plot(ax=ax, color="crimson")  # points with their error bars
ax.figure.savefig("both.png")
```

matplotlib is not a dependency; `pip install xrdroot[plot]` brings it,
and without it `plot()` refuses with both ways out by name. The other way is
`text()`, which needs nothing installed at all and goes anywhere a string
goes — a terminal, a log file, a CI transcript:

```python
print(f["h1d"].text())  # one line per bin: its edges, a bar and the value
print(f["h2d"].text())  # a shaded grid, y upward
print(f["tge"].text())  # a grid of stars with the axis ends labelled
```

A graph of layered error bars draws every layer over the same points; a
three-dimensional histogram has no honest flat picture and refuses both ways,
saying to slice `values()` down to the two dimensions you want to see.

## Columns

```python
tree.show()  # one line per column: name, type, variable or not
tree.typenames()  # {'Muon_pt': 'float32', 'nMuon': 'int32', ...}
tree.keys()  # every column
tree.readable()  # the ones this reader decodes
tree.unreadable  # {name: why not}, rather than quietly missing
```

A column comes back as one of three things, and which one is knowable in
advance from `tree[name].is_jagged` and `typename`:

| The column | What `array()` gives |
| --- | --- |
| a plain number (`x/F`) | a NumPy array of one value per entry |
| a fixed array (`x[10]/F`) | a NumPy array of shape `(entries, branch.length)` |
| a variable one (`x[n]/F`) | a `Jagged` — rows of different lengths |
| a character leaf (`x/C`) | a `list[str]` |
| a string, `std::string` or `TString` | a `list[str]` |
| an STL container | a `list`, one Python object per entry |

```python
tree["nMuon"].array()  # array([2, 0, 3, ...], dtype=int32)
tree["Muon_pt"].array(0, 1000)  # <Jagged 1000 rows of 2431 float32 values>
jets = tree["Muon_pt"].array()
jets[7]  # array([22.5, 19. ], dtype=float32)
jets.lengths()  # array([2, 0, 3, ...])
jets.content, jets.offsets  # every value, and where each row starts and stops
jets[100:200]  # a Jagged of those rows, sharing the same values
values, width = jets.padded()  # a flat rectangle and its width
```

A `Jagged` is kept the way Awkward Array and Arrow keep a list — one flat
array of values and one of offsets — so `jets.to_awkward()` and
`jets.to_arrow()` hand it over without copying the values. Every column is
decoded in C: a basket's bytes become an array in one pass, and the rows of a
variable column are cut out of it by one mask rather than one slice each.

Entry numbers behave like a Python slice, negatives included:
`branch.array(-1000)` is the last thousand entries.

## Reading a file that does not fit

`iterate` walks the tree in batches, holding one batch rather than one file:

```python
for batch in tree.iterate(["Muon_pt", "Muon_eta"], step=50_000):
    analyse(batch["Muon_pt"], batch["Muon_eta"])
```

`tree.arrays()` is the same for one range, and takes every readable column
when it is not told which.

## Expressions

A name `arrays` is given that is not a branch is a `TTree::Draw` expression —
ROOT's `TTreeFormula` language — and comes back under its own text. `cut` is a
selection in the same language:

```python
batch = tree.arrays(
    ["nJet", "Sum$(jet_pt > 30)", "jet_pt * cosh(jet_eta)"],
    cut="nJet >= 2 && met > 50",
    aliases={"met": "sqrt(met_x*met_x + met_y*met_y)"},
)
batch["Sum$(jet_pt > 30)"]  # array([2., 3., 1., ...]), one per entry
batch["jet_pt * cosh(jet_eta)"]  # a Jagged, one per jet
```

Only the branches the expressions and the cut need are read, once each, and
`iterate` and a chain's `arrays` take the same arguments. The language on its
own is `xrdroot.compile_formula`, for anything built on top of it:

```python
f = xrdroot.compile_formula("Max$(jet_pt)", tree.keys())
f.branches  # ('jet_pt',) — resolved to the tree's own names
f.evaluate(tree.arrays(f.branches))  # array([88., 0., 41.5, ...])
values, valid = f.evaluate_masked(columns)  # where an index ran off the end
```

Everything is evaluated over the whole batch at once, on the flat values and
offsets a `Jagged` is made of: `Sum$(jet_pt > 30)` over a million entries of
six jets each takes about 80 ms, and `jet_pt * cosh(jet_eta)` about 125 ms,
against some 75 ms for the same arithmetic written in NumPy by hand.

| ROOT | Here |
| --- | --- |
| `+ - * /`, `%`, `& \| << >> ~`, `== != < <= > >=`, `&& \|\| !`, `? :` | the same, with C's precedence; arithmetic in `double`, so `3/2` is `1.5`; `%` and the bitwise operators on integers truncated toward zero |
| `x^2` | a power, as ROOT's formulas have always read it — not C's exclusive or |
| `(int)x`, `int(x)`, `(double)`, `(float)`, `(bool)`, `Long64_t`… | C++'s conversions |
| `1`, `0x1f`, `2.5f`, `1e3`, `"text"` | numbers as C++ writes them, and strings to compare a string branch with |
| `TMath::Abs`, `Sqrt`, `Power`, `Exp`, `Log`, `Log10`, trigonometry and its inverses and hyperbolics, `ATan2`, `Min`, `Max`, `Floor`, `Ceil`, `Nint`, `Sign`, `Hypot`, `Erf`, `Erfc`, `Gamma`, `Gaus`, `BreitWigner`, `IsNaN`, `Finite`, `Even`, `Odd`, `Pi()`, `TwoPi()`, `E()`… | the same functions, over whole arrays |
| `abs`, `fabs`, `sqrt`, `pow`, `exp`, `log`, `sin`… `fmod`, `round`, `min`, `max`, `std::` in front of any | C's `<cmath>` |
| `strstr(s, "abc")`, `s == "abc"` | string branches: containment and equality |
| `evt.P3.Px`, `P3.Px`, `friend.x`, `ArrayI16` for `ArrayI16[10]` | branch names; the longest branch a dotted name spells wins |
| `x[0]`, `m[1][2]`, `x[n-1]`, `m[][2]`, `x[]` | an index, any expression; `[]` loops over the dimension |
| `x.size()`, `@x.size()` | how many elements a collection holds; without `@`, of each innermost one |
| `Length$(x)`, `Sum$(x)`, `Min$(x)`, `Max$(x)`, `MinIf$(x, c)`, `MaxIf$(x, c)` | one value per entry over the elements of `x`; `0` for an entry with none |
| `Alt$(x[3], -1)` | `x[3]`, or `-1` where there is no `x[3]` |
| `Entry$`, `Entries$`, `LocalEntry$`, `Iteration$`, `Length$` | where the loop is |

**The implicit loop.** A branch that is a collection makes the expression one
value per *element*: `jet_pt * 2` is a `Jagged` of one value per jet. Every
dimension not given an index is looped over; the dimensions looped over are
matched left to right across the branches, ignoring the ones given an index;
and dimensions matched together share one index and run to the *shortest* of
them, as ROOT documents. A number per entry, or `jet_pt[0]`, goes with every
element. So with `m` a `[3][3]` array and `v` one of five, `m - v` is nine
values, `m[i][j] - v[i]`; `m[][2] - v` is three; and `pt + eta` over three and
two elements is two — where go-hep's port refuses collections of different
lengths, this does what ROOT does. `Formula.per_element` says which kind an
expression is, when it was compiled against a mapping of names to their
dimensions, as `arrays` compiles it.

**What is missing.** `pt[3]` of an entry with two jets has no value. ROOT's
`Draw` leaves such an entry out, and so does `evaluate`: a row of elements
loses the missing ones, and a value per entry is `NaN`. `evaluate_masked`
gives the mask instead, and `Alt$` the fallback. A cut leaves out entries whose
cut is missing too.

**Cuts.** A cut that is one value per entry keeps the entries where it is
nonzero. One that loops — `jet_pt > 30` — is applied element by element, as
`Draw` applies it, to every column that is one value per element: expressions
that loop, variable-length branches and fixed arrays alike, pairing its
elements with theirs up to the shorter of the two. An entry is kept when any
of its elements passes, so the columns still line up.

What it will not do is refused by name: a name no branch has, with the nearest
that do; a member of an object the tree holds whole; a method other than
`size()`; and one of ROOT's rarer loops — a branch indexed by a collection
beside another looped over outside the index, which ROOT runs as two nested
loops.

## Draw and Scan

`tree.draw` is ROOT's `TTree::Draw`, and `tree.scan` its `TTree::Scan`, on a
tree, a tree with friends and a chain alike:

```python
h = tree.draw("jet_pt", "Sum$(jet_pt > 30) > 1")  # a TH1F called htemp
h.entries, h.mean(), h.std(), h.selected  # selected: what Draw returns
h2 = tree.draw("eta:phi>>map(64, -3.2, 3.2, 50, -2.5, 2.5)", "pt > 20")
p = tree.draw("response:pt", "", "prof")  # a TProfile of response in bins of pt
print(tree.scan("run:event:nJet", "nJet > 4", entries=100))
```

The tree is read a batch at a time — `step` entries, a hundred thousand
unless told — so memory is one batch and never the tree. Every axis, the
selection and the weight are evaluated together in one loop, as ROOT's
`TTreeFormulaManager` runs them: their collections are paired element by
element up to the shortest, a number per entry goes with every element, and
`met` drawn with a cut on the jets is filled once for every jet that passes.
The selection's value *multiplies* the weight — a cut gives 0 or 1, a number
is a weight — and a fill of weight zero is not made. What is filled is
filled through `Histogram.fill`, in the order ROOT would meet it, so the
bins, the entries and the moments are ROOT's.

Given no binning, a draw books what ROOT books: a `TH1F` of 100 bins, a
`TH2F` of 40 a side, a `TH3F` of 20, a `TProfile` of 100 or a `TProfile2D`
of 20 by 20, named `htemp`, titled with the expression and `{selection}`,
its axes titled with their expressions. The first `estimate` fills — ROOT's
`TTree::GetEstimate()`, a million — are held back to find the axes with
`THLimitsFinder`, ported statement by statement: the range widened by a
tenth, a bin width rounded to 1, 2 or 5 times a power of ten, and bins a
whole number wide when the expression is a lone integer branch, `Entry$`,
`Length$` or `Iteration$`. After those, a value off an end doubles the axis
until it is on it, as `TH1::ExtendAxis` does for ROOT's own `htemp`, and a
NaN stops a histogram extending at all. So a draw of a billion entries ends
up with the axes ROOT would give it too.

| ROOT | xrdroot |
| --- | --- |
| `t->Draw("x")` | `t.draw("x")` — a `TH1F` named `htemp` |
| `t->Draw("x", "y > 2")`, `t->Draw("x", "w")` | `t.draw("x", "y > 2")`, `t.draw("x", "w")` — the selection is a weight |
| `t->SetWeight(2); t->Draw("x")` | `t.draw("x", weight=2)`, or `weight="w_expr"` |
| `t->Draw("y:x")`, `t->Draw("z:y:x")` | `t.draw("y:x")`, `t.draw("z:y:x")` — the vertical axis first |
| `t->Draw("x>>h(100, 0, 10)")` | the same, or `t.draw("x", bins=(100, 0, 10), name="h")` |
| `t->Draw("y:x>>h(10, 0, 1, 20, -1, 1)")` | the same, or `bins=[(10, 0, 1), (20, -1, 1)]` |
| `t->Draw("x>>h(50)")` | the same, or `bins=50`: 50 bins, the ends found |
| variable bins, `TH1D h(..., n, edges); t->Draw("x>>h")` | `bins=[0, 1, 5, 10]` |
| `t->Draw("x>>+h")` | `t.draw("x>>+h", histograms=d)` — `d` stands for `gDirectory` |
| `t->Draw("y:x", "", "prof")`, `"profs"`, `"profi"`, `"profg"` | the same — a `TProfile`, with its error option |
| `t->Draw("z:y:x", "", "prof")` | the same — a `TProfile2D` |
| `t->Draw("y:x", "", "p")`, `"l"`, `"*"` | the same — the scatter of points, as a `Graph` named `Graph` |
| `t->Draw("x", "", "e")`, `"norm"` | the same: squares of weights kept, and scaled to a sum of one |
| `t->Draw("x", "", "goff")` | every draw; draw onto matplotlib with `ax=` |
| `t->Draw("x", "", "", n, first)` | `t.draw("x", entries=n, first_entry=first)` |
| `t->SetEstimate(n)` | `t.draw(..., estimate=n)` |
| `t->GetSelectedRows()`, `Draw`'s return value | `h.selected` |
| `t->Scan("a:b", "c > 0")` | `print(t.scan("a:b", "c > 0"))` |
| `t->Scan()`, `t->Scan("*")` | `t.scan("")` — the first eight columns — and `t.scan()` — all |
| `t->Scan("a", "", "colsize=12 precision=4")` | `t.scan("a", width=12, precision=4)` |
| `SetScanRedirect`, `SetScanFileName` | `t.scan(..., file=handle)`; the text also comes back |
| `TChain::Draw`, a friend's `t->Draw("f.x")` | `chain.draw(...)`, `t.draw("f.x")` |

The result is a `Histogram`, `Profile` or `Graph` like any other — to fill
more, compute with, plot, or write to a file — carrying `selected`, the
number of fills made, and (for a histogram) `extendable`. `>>name` names what
is filled; with `histograms=` — a dict standing for `gDirectory` — the result
is put in it under its name, `>>name` refills a histogram already there
from nothing, as ROOT resets it, and `>>+name` adds to it: to a histogram of
your own too, whose entries then grow by what was selected. Brackets after a
name that is there make a new one, as ROOT deletes the old. A histogram of
the wrong shape for the expression is refused where ROOT would warn and
replace it.

A `y:x` draw with no option fills a `TH2F`: this is ROOT's `goff`, since
nothing is drawn unless `ax=` is given, and with `goff` ROOT fills the
`htemp` it otherwise leaves empty behind a scatter plot. Options asking for
points or lines — `p`, `l`, `*`, unless `col`, `box` or another binned style
is there too — give the scatter itself, as a `Graph` of every fill in order,
which does hold all of them. `same` does nothing. `para`, `candle`, `gl5d`
and filling an entry list with `>>list` are refused by name, and so is an
expression of strings, which ROOT bins by label.

`scan` prints ROOT's table to the character: the eleven-star corner, the
`*    Row   *` and `* Instance *` columns, each number through `%9.9g`
trimmed before its exponent when too wide, a column as wide as its name from
nine to twenty characters, names too long cut to `...`, and
`==> 3 selected entries` under a selection. An expression over a collection
prints a row per element; the columns go down together only when the
selection loops too, as ROOT synchronises them, and otherwise each runs to
its own length with the ones that run out left blank. It returns the text and
writes it to `file` as it goes; ROOT's pause every fifty rows is not made,
and `"*"` means the columns this reads, under their names here — ROOT spells
a plain leaf `x.x`.

## Chains

A dataset is rarely one file. `xrdroot.chain` reads the tree of one name in
each of many as one tree, entries end to end — which is ROOT's `TChain`:

```python
events = xrdroot.chain("Events", ["run1/*.root", "root://host//store/extra.root"])
len(events)  # the entries of every file together
events["Muon_pt"].array(95_000, 105_000)  # across the boundary of two files
for batch in events.iterate(["Muon_pt", "Muon_eta"], step=100_000):
    ...
events.arrays(["nMuon"], library="pd")
events.close()  # or use it in a with block
```

A source is a path, a URL, or a file already open; a local path with `*`, `?`
or `[` in it is a glob, standing for every file it matches in sorted order. A
chain reads with everything a tree reads with — `keys`, `typenames`, `show`,
`arrays`, `iterate` — and the columns are joined across files as one read
would give them: arrays end to end, `Jagged` rows with each file's offsets
carried on, lists as one list. A batch of `iterate` runs off the end of one
file and into the next, so every batch but the last is `step` long.

Files are opened when first needed and once each. The number of entries needs
every file — `len`, a range counted from the end, or an entry in the tenth file
all need to know how long the nine before it are — and that costs a small read
of each file's header, key list and tree record, no baskets. A column whose
type differs from one file to another is refused by name, and so is one that a
later file does not have. A chain pickles as where its files are, so it can go
to a worker process, which opens them again there.

## Friends

A friend is another tree of the same entries, read beside a tree as though it
were part of it — weights computed afterwards, say, written to a file of their
own:

```python
events = f["Events"]
events.add_friend(g["Weights"], "w")
events["w.nominal"].array(0, 10)  # the friend's column, entry for entry
events["nominal"].array(0, 10)  # the same, when no other tree has the name
events.arrays(["Muon_pt", "w.nominal"])
```

A friend has to have exactly as many entries, since entry `i` of it is read as
entry `i` of the tree. Its columns are there by `alias.branch` — the alias is
the friend's own name unless given — and by their bare names too, when neither
the tree nor another friend has one of the same name; a name two friends share
has to be asked for by alias. A `Chain` can be a friend as well as a tree.

ROOT records the friends a tree was given when it was written, and those are
read back: `tree.friends` finds them the first time it is asked for. A friend
in another file is looked for where ROOT wrote down it was, then beside this
file under the same name, then beside it by the file's last part — usually the
right one, since the friend was shipped with the tree — and a remote file's
friend is looked for beside it on the same server. The files opened for them
close when the tree's file does.

## Entry lists

A selection run over a big tree can be kept as a `TEntryList` — just the
numbers of the entries that passed — or the older `TEventList`. Either comes
back as an `EntryList`, and every read takes one:

```python
kept = f["passed_cuts"]
kept.entries  # array([ 3,  4, 17, ...]): the entry numbers, int64
tree.arrays(["Muon_pt"], entries=kept)  # just those entries
tree["Muon_pt"].array(entries=[7, 2, 90])  # or any entry numbers, in that order
tree["Muon_pt"].array(entries=mask)  # or a mask of one bool per entry
for batch in tree.iterate(step=1000, entries=kept):
    ...
```

Only the baskets holding an entry that was asked for are read. A list made
over a chain keeps one list per tree, in `kept.lists`, each naming its tree
and file; `tree.arrays(entries=kept)` takes the one for the tree it is
reading, a `Chain` takes each file's from it, and `kept.for_tree(name, file)`
finds one by hand. `kept.entries` on such a list refuses, since its numbers
count from the start of each tree rather than from one place.

## Into pandas, Awkward, Arrow and Polars

`arrays` and `iterate` take a `library`, which says what to hand the columns
back as:

```python
tree.arrays(["pt", "eta", "jet_pt"], library="pd")  # a pandas DataFrame
tree.arrays(library="ak")  # an Awkward record array
tree.arrays(library="pa")  # a pyarrow Table, jagged columns as large_list
tree.arrays(library="pl")  # a Polars DataFrame
for frame in tree.iterate(step=100_000, library="pd"):
    ...
```

`np`, the default, is a dict of what the branches gave. A jagged column is a
real list type in Awkward, Arrow and Polars, and a column of arrays — one per
row — in pandas; a fixed-size array column is Arrow's `fixed_size_list`. None
of these libraries is a dependency: each is imported when it is asked for, and
one that is not installed is refused with the `pip install` that fixes it.

## C++ classes and containers

A file written by a physics framework is usually a C++ class per entry, split
by ROOT into one branch per member. Those branches are columns here like any
other, under the names ROOT gave them:

```python
tree.keys()  # ['Muon.pt', 'Muon.eta', 'evt.N', 'evt.StlVecF64', ...]
tree["evt.StlVecF64"].array(0, 100)  # <Jagged 100 rows of ...>
```

The object itself is a branch too, holding nothing at all — every byte of it
is in the members — so asking for it gives back one dictionary per entry, and
an object nested inside it is a dictionary inside that:

```python
tree.groups()  # ['evt', 'P3'] - the objects that were split
tree["evt"].array(1, 2)
# [{'I32': 1, 'Str': 'evt-001', 'P3': {'Px': 0, 'Py': 1.0, 'Pz': 0}, ...}]
```

This reads exactly the same baskets as asking for the members, and costs the
same; it is a shape, not a shortcut. `tree.arrays()` and `tree.iterate()`
leave the split objects out, because their members are already there under
their own names and taking both would read every basket twice. A member this
reader will not decode is absent from the dictionary and named in
`tree["evt"].unreadable`, with the same sentence `tree.unreadable` gives it.

A file written with splitting turned off puts the whole object in one branch,
with nothing under it. That reads as the same dictionary per entry, walked
member by member in the order the class declares them, using the layout the
file's own streamer information gives:

```python
tree.keys()  # ['evt'] - the object, and nothing under it
tree["evt"].array(1, 2)
# [{'Beg': 'beg-001', 'I32': 1, 'P3': {'Px': 0, 'Py': 1.0, 'Pz': 0}, ...}]
```

A class that inherits keeps each base under the base's own name, because a
derived class is allowed to declare a member its base already declared and
flattening the two together would quietly drop one of them. `TObject`, which
nearly every ROOT class inherits, comes back as the identifier and bits it
really is:

```python
tree["p4"].array(0, 1)
# [{'TObject': {'fUniqueID': 0, 'fBits': 50331648},
#   'fP': {'TObject': {...}, 'fX': 0.0, 'fY': 1.0, 'fZ': 2.0}, 'fE': 3.0}]
```

Some classes stream themselves rather than being written out by the file's
streamer information - `TLorentzVector` is one - and then the entry is the
class's own record with a version in front of the members. Both are read, and
so is the older `TBranchObject`, which writes the name of the class in front
of every entry; a branch of that kind holding more than one class stops with a
message naming both rather than reading the one as the other.

The same events written the two ways read back the same values, member for
member. Splitting is still the cheaper way to have written them - a split file
lets you read one member without touching the rest, and an unsplit one cannot -
but neither is a file this reader has to refuse. A class the file's streamer
information does not describe is refused, because its layout is then not
knowable, and so is a member of a kind this reader has no reader for: an
unsplit object is read from first byte to last, so one member it cannot walk
past is the whole entry.

A member that is an STL container comes back as the Python thing it most
nearly is, one per entry:

| In the file | In Python |
| --- | --- |
| `std::vector<double>`, `list`, `deque`, `set` of numbers | a `Jagged` — rows of NumPy arrays |
| a container of strings | a `list[list[str]]` |
| `vector<vector<T>>` | a `list` of `list`s of NumPy arrays |
| `std::map<K, V>`, `unordered_map` | a `list[dict]`, one dict per entry |
| `std::string`, `TString` | a `list[str]` |
| `std::vector<bool>` | rows of 0 and 1, a byte an element, which is how ROOT wrote it |
| `ROOT::VecOps::RVec<T>` | whatever the same `std::vector<T>` gives, which is what it is written as |
| `std::bitset<N>` | rows of 0 and 1, `bs[0]` first, a byte a bit as ROOT wrote it |
| `TDatime` | a `datetime.datetime`, out of the one word it packs itself into |

`Double32_t` and `Float16_t` are floats squeezed into three or four bytes by a
recipe written as `[xmin,xmax,nbits]`, where the ends may be given in units of
`pi`. A branch of its own keeps the recipe in the leaf title; a member of a
split class keeps it in the trailing comment on the declaration, which is in
the file's streamer information — either way it is unpacked back to `float64`,
so `typenames()` says `float64` and nothing about the packing reaches you. A
packed member of a class the file does not describe is refused rather than
read at the default, because the wrong recipe gives plausible wrong numbers.

## Into PyTorch and TensorFlow

Turning what comes out of a tree into tensors is [`xrdml`](https://github.com/rob-c/xrdml),
a package of its own on top of this one: `xrdml.tensors` batches a tree into
PyTorch or TensorFlow a basket at a time, and `xrdml.load` takes a URL
straight to a training loop.

## RDataFrame

`xrdroot.RDataFrame` is ROOT's declarative analysis: describe what to select,
compute and fill, and one event loop does all of it, reading each column once.
The interface is ROOT's, method for method, and the expressions are ROOT's
too — C++, with `ROOT::VecOps` in scope. What is different is what runs.
ROOT calls the code of a `Define` or `Filter` once per entry; here an
expression, or a Python callable, is evaluated over a whole batch of entries
at once, as NumPy arrays and `Jagged` collections, so nothing loops over
entries in Python.

ROOT's `df102_NanoAODDimuonAnalysis`, line for line:

```python
from xrdroot import RDataFrame

df = RDataFrame("Events", "root://eospublic.cern.ch//eos/opendata/cms/Run2012BC_DoubleMuParked_Muons.root")
two = df.Filter("nMuon == 2", "Events with exactly two muons")
pair = two.Filter("Muon_charge[0] != Muon_charge[1]", "Muons with opposite charge")
mass = pair.Define("Dimuon_mass", "InvariantMass(Muon_pt, Muon_eta, Muon_phi, Muon_mass)")
h = mass.Histo1D(("Dimuon_mass", "Dimuon mass;m_{#mu#mu} (GeV);N_{Events}", 30000, 0.25, 300), "Dimuon_mass")
report = df.Report()

h.GetValue().plot()        # the loop runs here, once, for both results
print(report.GetValue())
# Events with exactly two muons: pass=... all=...   -- eff=... % cumulative eff=... %
```

Nothing is read until a result is asked for. `Define`, `Filter`, `Alias` and
`Range` each give a new frame over the same graph, and leave the one they were
called on as it was, so one source feeds several branches of an analysis;
`Count`, `Sum`, `Histo1D` and the other actions book a result and give a
`Result` — ROOT's `RResultPtr` — whose `GetValue()` (or `.value`, `float()`,
iteration, indexing, or any attribute of the value itself) runs the loop for
every result booked so far. A column is computed only for the entries that
reach the node it was defined at, so a `Define` after a `Filter` never sees an
entry the filter rejected.

```python
RDataFrame(tree)                                  # a TTree, a Chain or an RNTuple
RDataFrame("Events", "events.root")               # a tree or RNTuple, by name
RDataFrame("Events", ["a.root", "run2/*.root"])   # several files, globs too
RDataFrame(1_000_000).Define("x", "rdfentry_ * 2.")   # empty entries to define columns on
RDataFrame(tree, workers=4, step=200_000)         # four processes, batches of 200 000
```

A frame made from a name opens its files and closes them with `close()` or a
`with` block; one handed a tree leaves it open.

### Expressions

A string is a C++ expression over the frame's columns, evaluated as C++
would: integers stay integers and divide as C divides them (`7 / 2` is `3`),
`float` stays `float`, `bool` and the small integers are promoted to `int`,
and `^` is exclusive or. A collection is an `RVec`: arithmetic pairs its
elements with another's one for one, a number per entry goes with each of its
entry's elements, comparing gives an `RVec<int>`, and a mask indexes one:

```python
df.Define("good_pt", "Muon_pt[Muon_pt > 30 && abs(Muon_eta) < 2.4]")   # a collection per entry
df.Define("n_good", "good_pt.size()")
df.Define("lead", "n_good > 0 ? good_pt[0] : -1.f")
df.Filter("Sum(Jet_pt > 30) >= 2", "two jets")
df.Define("dr", "DeltaR(Muon_eta[0], Muon_eta[1], Muon_phi[0], Muon_phi[1])")
df.Define("pairs", "Combinations(Muon_pt, 2)")       # Take(Muon_pt, pairs[0]) and pairs[1]
```

What C++ evaluates conditionally is evaluated conditionally: the right side of
`&&` and `||`, and each branch of `? :`, is evaluated only for the entries that
reach it, so `nMuon > 0 && Muon_pt[0] > 30` never looks at the first muon of an
entry with none. Where C++ is undefined — an index past the end, the `Max` of
an empty collection, an integer divided by zero — the expression is refused,
naming the entry, rather than given a made-up value. A name no column has is
refused when `Define` or `Filter` is called, with the nearest names that are
columns.

`ROOT::VecOps` is there by its bare names and with `VecOps::` or
`ROOT::VecOps::` in front: `Sum`, `Product`, `Mean`, `Var`, `StdDev`, `Max`,
`Min`, `ArgMax`, `ArgMin`, `Any`, `All`, `Dot`, `Take` (of positions, of the
first or last `n`, padded with a default), `Nonzero`, `Where`, `Argsort`,
`StableArgsort`, `Sort`, `Reverse`, `Concatenate`, `Drop`, `Enumerate`,
`Range`, `Combinations`, `DeltaPhi`, `DeltaR2`, `DeltaR`, `InvariantMass` and
`InvariantMasses`; so are `<cmath>` and `TMath`, element by element over
collections, casts in all three spellings, and an `RVec`'s `size()`,
`empty()`, `front()`, `back()` and `at(i)`. `rdfentry_` and `rdfslot_` are
columns of every frame. A C++ lambda is refused, naming a Python callable —
which does the same job — as the alternative.

### Python callables

A callable is given its columns a batch at a time — NumPy arrays, `Jagged`
collections, lists of strings — named by `columns`, or by its own parameters
when that is not given, and gives back one value per entry of the batch:

```python
import numpy as np
from xrdroot.rdf import vecops

df.Define("pt2", lambda Muon_pt: vecops.Map(np.square, Muon_pt))
df.Define("r", np.hypot, ["x", "y"])
df.Filter(lambda nMuon: nMuon >= 2, name="two or more")
df.Define("mass", vecops.InvariantMass, ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"])
```

`xrdroot.rdf.vecops` is `ROOT::VecOps` for callables: the same functions by
the same names, on a `Jagged` of every entry's collection at once, giving an
array of one value per entry or another `Jagged`. They are the kernels the
string expressions use, so the numbers are the same either way. Where C++
would be undefined these do not make something up either: `Max` and `Min` put
`default` (NaN unless told) in an empty entry, and `Take` past the end is
refused unless given a default to pad with.

### Results

| Action | Gives |
| --- | --- |
| `Count()` | the number of entries reaching the node |
| `Sum(c)`, `Mean(c)`, `Min(c)`, `Max(c)`, `StdDev(c)` | over every value — every element, of a collection |
| `Stats(c, w)` | a `TStatistic`: `GetN`, `GetMean`, `GetRMS`, `GetMin`, `GetMax`, ... |
| `Histo1D(model, x, w)`, `Histo1D(x)` | a `Histogram`, filled with ROOT's bookkeeping |
| `Histo2D(model, x, y, w)`, `Histo3D(model, x, y, z, w)` | the same, of more axes |
| `Profile1D(model, x, y, w)`, `Profile2D(model, x, y, z, w)` | a `Profile` |
| `Graph(x, y)` | a `Graph` of a point per entry, in entry order |
| `Take(c)`, `AsNumpy(columns, exclude)` | the values: an array, a `Jagged`, a list; a dict of them |
| `Reduce(f, c, init)` | the values folded by `f` — a NumPy ufunc reduces a batch in C |
| `Aggregate(aggregator, merger, c, init)` | `aggregator(acc, batch_values)` per batch, `merger(a, b)` across |
| `Foreach(f, columns)`, `ForeachSlot` | `f` called on every batch, now |
| `Display(columns, rows, elements)` | ROOT's box of the first entries |
| `Report()` | the cut flow of the named filters, printed as ROOT prints it |
| `Snapshot(tree, file, columns)` | the entries and columns written to a file, and a frame over it |

A histogram's model is ROOT's `TH1DModel` as a tuple — `("name", "title",
nbins, low, high)`, or a number of bins and a list of edges per axis, a
profile's followed by the range of values it averages and its error option —
or a `Histogram` or `Profile` already booked, whose binning and kind are used.
Without one, `Histo1D` books 128 bins spanning every value it is filled with.
Any column an action names may be an expression instead, as ROOT's actions do
not allow: `df.Sum("x * 2")`.

`Snapshot` writes as ROOT's does — immediately, with every other result booked
— through `xrdroot.create`, a `TTree` or, with `rntuple=True`, an RNTuple:
numbers, collections and strings; columns as a list or a regular expression,
every one by default; `mode="UPDATE"` to add to a file already there;
`lazy=True` for a `Result` rather than the frame over the file written. A
loop that fails abandons the file rather than leaving half of one.

`GetColumnNames()` (or `.columns`), `GetDefinedColumnNames()`,
`GetColumnType(c)` — ROOT's names for the types, `Float_t` for a tree's
branch, `ROOT::VecOps::RVec<double>` for a defined collection — `HasColumn`,
`GetFilterNames`, `GetNRuns` and `Describe()` say what a frame is, and
`to_pandas()`, `to_awkward()`, `to_arrow()` and `to_polars()` hand its columns
over. Every method has a snake_case spelling too: `define`, `filter`,
`histo1d`, `as_numpy`, `get_column_names`.

### In parallel

`workers=n`, or `xrdroot.EnableImplicitMT(n)` for every frame made after it,
shares the loop across `n` worker processes — processes rather than ROOT's
threads, which is what lets Python in a `Define` run side by side. The entries
are cut into tasks of `step` entries that never straddle two files, a worker
opens the files again for itself (a chain, a tree and an RNTuple all pickle as
where their files are), and each task's partial results come back and are
merged in task order — exactly as one process merges them — so the results
are the same to the last bit however many workers made them. `workers=1`, the
default, runs the same tasks here.

Everything sent to a worker is pickled, so a callable must be a function
defined at the top level of a module, or a NumPy function; a lambda is refused
with that said. `Range`, which counts entries in the order they arrive, and
`Foreach`, which runs for its side effects, are refused with more than one
worker, as ROOT refuses `Range` under implicit multithreading. `RunGraphs`
computes the results of several frames, in one loop for frames made over the
very same tree object.

A dimuon analysis like the one above, with a second histogram and the report,
over a million NanoAOD-like entries in a local file, takes 1.45 s in one
process — the same as reading its six columns with `arrays` and nothing else,
1.47 s, because the vectorised arithmetic is a small part of it — and 1.04 s
with four workers, process start-up included.

### From ROOT

| ROOT | xrdroot |
| --- | --- |
| `ROOT::RDataFrame df("Events", "f.root")` | `df = RDataFrame("Events", "f.root")` |
| `ROOT::RDataFrame df(1000)` | `RDataFrame(1000)` |
| `df.Define("pt2", "pt*pt")` | `df.Define("pt2", "pt*pt")` |
| `df.Define("r", [](float x, float y) { ... }, {"x", "y"})` | `df.Define("r", func, ["x", "y"])` — `func` takes arrays |
| `df.Filter("n > 1", "cut")` | `df.Filter("n > 1", "cut")` |
| `df.Range(100)`, `df.Alias("a", "b")`, `df.Redefine(...)` | the same |
| `df.Histo1D({"h", "t", 100, 0., 1.}, "x", "w")` | `df.Histo1D(("h", "t", 100, 0.0, 1.0), "x", "w")` |
| `df.Profile1D({"p", "", 10, 0., 1., 0., 5., "s"}, "x", "y")` | `df.Profile1D(("p", "", 10, 0.0, 1.0, 0.0, 5.0, "s"), "x", "y")` |
| `auto n = df.Count(); *n` | `n = df.Count(); n.GetValue()` |
| `h->Draw()` | `h.GetValue().plot()` |
| `df.Report()->Print()` | `print(df.Report().GetValue())` |
| `df.Display({"x", "y"}, 5)->Print()` | `print(df.Display(["x", "y"], 5).GetValue())` |
| `df.Snapshot("t", "out.root", {"x"})` | `df.Snapshot("t", "out.root", ["x"])` |
| `ROOT::RDF::RSnapshotOptions opts; opts.fLazy = true` | `df.Snapshot(..., lazy=True)` |
| `ROOT::EnableImplicitMT(4)` | `xrdroot.EnableImplicitMT(4)` |
| `ROOT::RDF::RunGraphs({r1, r2})` | `xrdroot.RunGraphs([r1, r2])` |
| `ROOT.RDataFrame(...).AsNumpy(["x"])` | `df.AsNumpy(["x"]).GetValue()` |
| `ROOT::VecOps::Sum(v)` in C++ | `vecops.Sum(v)` in Python, `Sum(v)` in a string |

What differs from ROOT, and why:

- A callable is called once per batch with arrays, not once per entry with
  numbers; it gives back an array of the batch's length.
- `v.size()` is a `Long64_t` rather than `size_t`, so `v.size() - 1` of an
  empty collection is `-1`, not a wrapped-around unsigned number.
- What C++ leaves undefined is refused, naming the entry, rather than read out
  of whatever memory was there.
- `Sum` adds integers exactly and anything else in `double`, batch by batch,
  with the batches' sums added by `math.fsum`; `StdDev` of fewer than two
  values is zero, as go-hep has it; a `Graph`'s points are in entry order
  however many workers there were.
- `Histo1D` without a model spans exactly the values it was filled with, where
  ROOT rounds the range out to "nice" numbers.
- A loop that fails fails every result it was computing; results booked after
  it compute as usual.
- `Vary`, `DefineSlot`, `FilterAvailable` and string lambdas are not here.

## Writing

`create` makes a new ROOT file anywhere this library can put bytes — a local
path, any URL scheme it writes, or an already-open binary file, used as it is
and left open:

```python
import xrdroot

with xrdroot.create("root://eos.example.org//store/user/me/out.root") as f:
    f["counts"] = xrdroot.Histogram.new("counts", edges, values)
    f["scan"] = xrdroot.Graph.new("scan", xs, ys, yerr=bars)
    f["note"] = "made from run 4711"
    f["weights"] = np.asarray(weights)
    f["events"] = {"pt": pt, "eta": eta}  # a tree, from arrays or any DataFrame
    f["spectrum"] = hist_object  # a hist.Hist, boost histogram or numpy.histogram
```

The file is a mapping from name to object, closed with `with` or `close()`. It
is the real thing — keys, directory, free list, and the streamer information
describing its classes exactly as the ROOT the layouts were harvested from
would — so ROOT, uproot and this library's own reader all read it back by its
own self-description. A name written twice becomes a second cycle of itself,
exactly as in ROOT, and reading back gives the newest.

Records go out as they are made, a few megabytes at a time, so a file far
larger than memory is written in the memory of one basket per column; only the
header at the front, which points at the bookkeeping written last, is held back
and filled in at the close. A `with` block that raises takes back everything it
wrote — the target is cut back to where the file began — on the principle that
no file is better than half a file, and a remote file is opened
persist-on-successful-close, so a process that dies part way leaves the server
nothing. A target that cannot seek, such as a pipe, is the one exception: that
file is kept in memory and written in order at the close.

What can be written is what can be written *correctly*: trees of numbers,
histograms and graphs — read from another file, built from plain numbers, or
made by `hist`, `boost-histogram` or `numpy.histogram` — and the functions
fitted to them, plus strings and
one-dimensional arrays of signed integers or floats, which become the matching
`TArray`. `Histogram.new` takes every bin edge — or a set of edges per axis,
for two or three — the values shaped the way the axes are (two more along an
axis fills its flow bins), and optionally per-bin errors or variances, axis
labels and an entry count; evenly spaced edges are stored the compact way ROOT
stores an even axis. Every `TH1`, `TH2` and `TH3` of chars, shorts, ints,
floats and doubles is written, and so are `TProfile`, `TProfile2D`,
`TProfile3D` and `TEfficiency` — booked and filled here or read from a file;
`Histogram.of(...)` turns any histogram Python has into one of these.
`Graph.new` picks its own class: plain points make a `TGraph`, one bar per
point a `TGraphErrors`, and any `(low, high)` pair of runs a
`TGraphAsymmErrors`.

Anything else is refused by name rather than guessed at — a
`TGraphMultiErrors`, an unsigned array ROOT has no class for, a histogram
whose list of functions holds anything but functions, or an axis with labels
(take them out first, rather than have them silently dropped), a name a
reader could never ask back for —
because a plausible-looking file that ROOT misreads is worse than an error
message.

`create(..., compression="zstd")` chooses the algorithm — zlib unless said
otherwise, or `lzma`, `lz4`, `zstd`, `None` to store raw — and `level` the
effort; an object that did not get smaller is stored raw anyway, exactly as
ROOT does.

### Trees

`f.tree(name, columns)` declares a tree, and `fill` gives it entries:

```python
with xrdroot.create("out.root") as f:
    tree = f.tree("events", {"energy": float, "hits": ("i", 4), "ok": bool})
    for event in run:
        tree.fill(energy=event.energy, hits=event.hits, ok=event.passed)
```

A column is a Python `bool`, `int` or `float`, an [`array`][array] type code
such as `'f'` for a narrower number, anything NumPy calls a type
(`np.float32`, `"uint8"`, a dtype), or a pair of any of those and how many
values every entry holds — `("i", 4)` is four `int32`s per entry. A Python
`int` gets the widest ROOT has, because a Python int has no width of its own
and a column that quietly stopped fitting halfway down the file would be worse.

Rows whose length changes from entry to entry are declared with `None` for the
count, and strings with `str`:

```python
with xrdroot.create("out.root") as f:
    tree = f.tree("events", {
        "jet_pt": ("f", None),       # float32 rows, counted by njet_pt
        "mu_px": ("f", "nmu"),       # two columns sharing one counter, nmu
        "mu_py": ("f", "nmu"),
        "trigger": str,
    })
    tree.fill(jet_pt=[40.5, 22.0], mu_px=[1.5], mu_py=[-0.5], trigger="HLT_Mu20")
```

These are laid out the way ROOT lays out a leaf-list `x[n]/F`: a counter
branch of `int32`s — `n` and the column's name, unless one is named — that
comes just before the first column it counts, a data leaf that points at the
counter's leaf, and baskets that carry a table of where each entry begins
after the values, as ROOT's own do. The counter is never filled by hand: it
is the length of each row, and columns that share one must agree about that
in every entry. A string is a `TLeafC`, its length and then its bytes. `fill`
takes any one-dimensional sequence for a row and a `str` for text; `extend`
takes a `Jagged`, an Awkward Array or a list of arrays for a column of rows,
and a list or NumPy array of strings for text, and packs rows in C the way it
packs everything else.

`extend` is the fast way in: given a mapping of column name to array, it packs
every column in C, checks the shapes and refuses a float going into an integer
column or an integer too wide for one, and puts the entries into exactly the
baskets filling them one by one would have. Given an iterable of mappings it
takes them as entries instead. Writing a table under a name — `f["events"] =
frame` — declares the tree from the arrays' own types and extends it in one
go: a `Jagged`, an Awkward Array of lists, a list of arrays or a frame's
column of lists becomes a column of rows, and strings a column of text.

[array]: https://docs.python.org/3/library/array.html

Entries go out a basket at a time as they gather — `basket_size` bytes of a
column at a time, 32 kB unless said otherwise — so a tree can be far larger
than memory, and reading a range back later costs one read per basket it
touches rather than one read of everything. Each column fills its own baskets
at its own rate, so a wide image column and a one-byte label column do not
force each other's geometry. The tree's own record says where every basket
landed, so it is written when the file closes.

An entry that does not fit its columns is refused whole — nothing is kept for
any column, so the tree is exactly as it was — and it is refused by name:
which column, what it holds, and what arrived instead — a row that is not
flat, a float for an integer column, a string with a NUL in it, which ROOT
would take to end there. Split C++ objects are refused rather than
approximated, on the same principle as the rest of the writer.

The result is a tree laid out the way ROOT lays one out, down to the record
versions and the `fLeaves` references pointing at the very leaves the branches
hold, so ROOT, uproot and this library's own reader all walk it the same way.

### Directories

A name with a `/` in it goes into a directory, and every directory along the
path is made if it is not there yet; `mkdir` makes one outright and hands it
back, with the same mapping, `tree` and `mkdir` as the file itself:

```python
with xrdroot.create("out.root") as f:
    f["runs/4711/pt"] = pt_hist             # runs and runs/4711 are made for it
    calib = f.mkdir("calibration")
    calib["gains"] = gains
    events = calib.tree("events", {"energy": float})
```

Each is ROOT's own `TDirectory`: a key in the directory above it, a record
behind that key, and a key list of its own written at the close, laid out
record for record the way ROOT 6 lays out the directories in
`tests/data/dirs-6.14.00.root`. Reading back is by path, `back["runs/4711/pt"]`,
here and in ROOT and uproot alike. `mkdir` of a directory already there gives
it back rather than making a second; a name already holding an object is
refused as a directory, and a directory's name refused as an object, because
either would hide the other from anything reading the file back.

### Files past 2 GB

ROOT keeps places in a file in four bytes until a file passes 2 GB, and in
eight after — keys, directory records, the free list and the header each have
a wide form for it. A file written here changes over exactly where ROOT does:
a record stays small until it is written past the 2 GB line or belongs to a
directory that was, and the header goes wide once the file ends past it. What
comes out is small keys at the front and wide ones after, as in ROOT's own big
files, and it reads back here, in ROOT and in uproot. Nothing is refused for
size any more.

### Updating a file

`update` opens a ROOT file that is already there — written by ROOT, uproot,
go-hep or this library — to add to it, with the same mapping, trees and
directories as a new one:

```python
with xrdroot.update("root://eos.example.org//store/user/me/out.root") as f:
    f["counts"] = newer_counts        # the next cycle of what was there
    f["runs/4712/pt"] = pt_hist       # into directories old or new
```

What was there stays where it was. New records go on the end; a name written
again becomes its next cycle; at the close the directories that gained
something get new key lists, the streamer information gains whatever classes
the new objects need — the file's own descriptions kept byte for byte, the new
ones added after — and the records those replace join the free list, marked
the way ROOT marks a gap. The header is the last thing written, so until then
the file still says exactly what it said before, and a `with` block that
raises cuts it back to its old length: a failed update leaves the file byte for
byte as it was. `compression` carries on with the file's own setting unless
told otherwise.

A file whose length is not where its header says it ends — still being
written, cut short, or added to by something else — is refused rather than
guessed at, as is one whose top directory has no room for the record an update
rewrites.

### The datasets everyone teaches with

Converting open data into ROOT — MNIST and its family, CIFAR, the UCI
teaching tables, the Hugging Face Hub mirrors and the rest, more than fourteen
hundred sets of them — is [`xrddatasets`](https://github.com/rob-c/xrddatasets),
which is built on this package and publishes the catalogue those files are
served from.

## RNTuple

RNTuple is ROOT 7's successor to the `TTree`: a column of plain values for
every leaf of the schema, cut into compressed pages, grouped into clusters of
consecutive entries, and described in little-endian envelopes of its own
rather than in ROOT's streamer format. A directory lists only its anchor, a
`ROOT::RNTuple` key, and asking for one gives back an `RNTuple`, which is used
the way a tree is:

```python
events = f["Events"]  # <RNTuple 'Events' with 969 fields and 10 entries>
len(events), events.keys(), events.num_clusters
events.show()  # name, Python type and C++ type, one line per field
events.typenames()  # {'nMuon': 'uint32', 'Muon_pt': 'list[float32]', ...}
events.cxx_types()  # {'nMuon': 'ROOT::RNTupleCardinality<std::uint32_t>', ...}
events["Muon_pt"].array(0, 1000)  # <Jagged 1000 rows of 2372 float32 values>
events.arrays(["nMuon", "Muon_pt"], library="ak")
for batch in events.iterate(step=50_000, library="pd"):
    ...
```

A range of entries reads the page lists of the cluster groups it touches and
the pages of the fields asked for in the clusters it touches, nothing else —
the same bargain a tree's baskets make. Every column encoding of the
specification (1.0) is decoded: the plain little-endian numbers; the split
ones, whose pages hold every element's first byte, then every second; delta
for offsets and zigzag for signed integers on top of those; booleans a bit to
an element; half-precision floats; `Real32Trunc`, a float with its low
mantissa bits dropped; and `Real32Quant`, a float spread over the integers its
bits can count within the range the column declares. So is everything around
them: several clusters and cluster groups, a schema that grew while the file
was being written (its entries before a field existed read as zeros), a field
written through two encodings with a different one live in different
clusters, projected fields and their alias columns, blocks split across keys,
and the three versions of the format ROOT has shipped. Envelopes, pages and
the anchor each carry an XXH3 checksum; with the `lz4` extra's `xxhash`
installed every one is checked on the way in and a damaged one refused, and
without it they are passed over, as the LZ4 checksums are.

What comes back is shaped the way a tree's columns are:

| The field | What `array()` gives |
| --- | --- |
| a number or a `bool` | a NumPy array, of the type the C++ says — a `float` stored as `Real16` is still `float32` |
| `std::array<T, N>` of numbers, `std::bitset<N>` | a NumPy array of shape `(entries, N)`, a dimension per nesting |
| `std::vector`, `RVec` or set of numbers | a `Jagged` |
| `ROOT::RNTupleCardinality` | a NumPy array of how many items each entry's collection holds |
| `std::string` | a `list[str]` |
| a class or struct | a `dict` per entry, a key per member — a base class under the name of its class, as a tree's objects keep theirs |
| `std::pair`, `std::tuple` | a `tuple` per entry |
| `std::map`, `std::unordered_map` | a `dict` per entry; a multimap is a list of `(key, value)` pairs, so nothing is lost |
| `std::optional`, `std::unique_ptr` | the value, or `None` |
| `std::variant` | whichever alternative each entry holds, or `None` for none |
| `std::atomic`, an enum | what it wraps |
| anything nested | a list per entry, of the same shapes one level down: a `vector<vector<int>>` is a list of NumPy arrays per entry |

A member of a record is a field of its own under a dotted name,
`events["event.muon.pt"]`, which reads that column alone; so is a record held
in a record. A field ROOT streamed whole with its own streamer, and a column
type newer than the specification, are listed in `unreadable` with the reason
rather than read, and a feature flag this reader does not know refuses the
RNTuple outright, as the specification asks.

### Writing one

`f.rntuple(name, fields)` declares an RNTuple and `extend` and `fill` give it
entries; a table goes in whole with `rntuple=True`:

```python
with xrdroot.create("out.root") as f:
    events = f.rntuple("events", {"n": np.int32, "pt": [np.float32], "tag": str})
    events.extend({"n": counts, "pt": jagged_pt, "tag": tags})
    events.fill(n=2, pt=[10.5, 3.25], tag="last")
    f.write("summary", frame, rntuple=True)  # a dict of arrays or any DataFrame
```

A field is a number — a NumPy type, a C++ name such as `"std::uint16_t"` or
`"double"`, or a Python `bool`, `int` or `float` — or `str` for a
`std::string`, or a vector of a number, spelled `[np.float32]` or
`"std::vector<float>"`. A vector is given as a `Jagged`, an Awkward list, or a
list of rows. Entries gather a cluster at a time — `cluster_size` bytes of
values, 32 MB unless said otherwise — and each column is written in pages of
at most `page_size` bytes, ROOT's own megabyte by default, so an RNTuple far
larger than memory costs one cluster of it. The pages are compressed with the
file's algorithm, each followed by its checksum, in the unlisted `RBlob` keys
ROOT keeps them in; the header, page list, footer and anchor go in when the
file closes, and the file's streamer information describes `ROOT::RNTuple` as
ROOT itself does. The column encodings are ROOT's defaults: split, zigzag and
delta when the file is compressed, and the plain ones when it is not.

The checksums are XXH3, computed by `xxhash` when it is there and by a Python
XXH3 here when it is not, which the test suite holds to the C library's
digests and to the checksums ROOT wrote into the files under `tests/data`. What
this writes is read back by uproot and by go-hep, which checks every checksum.

Records, nested collections, variants and the lossy float encodings are not
written: their layout is well defined, but a writer that gets one subtly wrong
makes files ROOT misreads, and each is refused by name until it is here.

## Compression

Every algorithm ROOT writes with is read here — and written: zlib, lzma, LZ4,
zstd, and the bare-deflate blocks ROOT wrote before 2005 (read only). LZ4 is
decoded and encoded in Python, checksum and all, when nothing else is there,
and by the C codec of the `lz4` extra when it is — about sixty times faster,
with the checksum then checked on the way in, so a damaged block is refused
rather than decoded. zstd uses Python 3.14's own `compression.zstd` where there
is one and the `zstandard` package otherwise, and is the one case where a file
may need something installed.

## Old files

A tree written by ROOT 4 opens like any other. Those files count entries in
doubles and keep their seek points in 32-bit integers, and one small enough
never to have been flushed holds its baskets inside the branch record rather
than out in the file — all of which is read here, so a decade-old Geant4 run
needs no copying forward first. ROOT 3 and older are refused by name.

## What it refuses, and why by name

A plausible misreading of physics data is worse than a refusal, so anything
this reader does not understand is named rather than guessed at. A column that
cannot be decoded is listed in `tree.unreadable` with the reason, and raises
with the same sentence if it is asked for:

```python
>>> tree.unreadable
{'vtx': 'Vertex, which is a C++ type this reader does not decode: this '
        "file's streamer information does not describe its layout, and a "
        'file written split would have its members as branches of their own'}
```

What is named that way:

- a class whose layout the file's streamer information does not describe, so
  that reading it whole would be a guess;
- a member of an unsplit object of a kind this reader has no reader for, which
  stops the entry it is in, because an unsplit entry is read from first byte to
  last and there is no length in front of a member to step over it by;
- a `multimap`, whose duplicate keys a `dict` would silently drop, and a map
  keyed by a container or nested inside one;
- a `pair` anywhere but directly under its own container, and a container of
  pairs written pair by pair, neither of which any file this reader has met
  writes (a map written pair by pair, as a `TFormula` writes its parameter
  names, is read);
- a graph of layered y errors asked for `yerr`, because summing the layers
  would be an answer this reader made up;
- trees written by ROOT 3 or older.

A class the file describes as having no members at all — `TLimit` is one —
reads as the empty `dict` it honestly is, rather than being refused.

## Errors

`xrdroot.ROOTError` is an `XRootDError`, so it is an `OSError` like
everything else this library raises.

| Exception | Means |
| --- | --- |
| `FormatError` | the bytes are not the ROOT format they claim — a truncated download and an HTML error page both look like this |
| `UnsupportedFeatureError` | a valid file using something this reader does not do, named rather than guessed at |
