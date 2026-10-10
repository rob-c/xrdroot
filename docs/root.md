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
| `h->FillRandom("gaus", n)`, `FillRandom(f, n, rng)` | `h.fill_random("gaus", n)`, `h.fill_random(f, n, rng=rng)` |
| `h->FillRandom(h2, n)` | `h.fill_random(h2, n)` |
| `h->Fit("gaus")` and the rest | `h.fit("gaus")` — see [Fitting](#fitting) |

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
fraction straying further than the data — and ROOT's draws: each
pseudo-experiment is `FillRandom` from that histogram, a Poisson count per
bin corrected entry by entry past ten entries a bin and `GetRandom` for each
entry below, taken from `gRandom` unless `rng=` gives another generator. So
it moves `gRandom` on, as ROOT's does, and `rng=TRandom3(seed)` repeats. An axis
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
| `f->GetRandom()`, `f->GetRandom(xmin, xmax)` | `f.get_random()`, `f.get_random(n, range=(xmin, xmax))` |
| `h->FillRandom("gaus", n)` | `h.fill_random("gaus", n)` — from `gRandom` unless `rng=` |
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

`f.get_random(n)` is `TF1::GetRandom`: the cumulative integral at `fNpx`
points over the function's range — of `log10(x)` when the range starts above
zero and spans more than `fNpx` times itself — a parabola fitted to it in each
interval from the integral over the interval and over its first half, the
interval found with `TMath::BinarySearch` and the parabola solved, one
`Rndm()` a number. `range=(xmin, xmax)` is the second `GetRandom`, drawing
`Uniform(pmin, pmax)` between the table entries either side and drawing again
until a number lands inside, one `Rndm()` a try. `h.fill_random(source, n)`
is `TH1::FillRandom`: from a function, its integral over each bin of the
histogram's axis range summed into a table, and each entry placed on the
straight line across the bin its `Rndm()` falls in; from a histogram,
`GetRandom` for each entry, or past ten entries a bin a Poisson count per bin
and single entries added or taken away until there are exactly `n`, the sums
then made again from the bins, as `ResetStats` makes them. `"gaus"` and the
other names are `gROOT`'s standard functions, over `(-1, 1)` with the
parameters `InitStandardFunctions` gives them. Both are ROOT's loops,
statement for statement, and are held draw for draw to them written out one
number at a time; `dirs-6.14.00.root` and `embedded-tbox.root` hold
histograms ROOT filled with `h->FillRandom("gaus", 5)` straight after
starting, which filled here from `TRandom3()` are the same doubles —
contents, entries and running sums. The integrals are `TF1::Integral`'s: in
closed form for `gaus`, `gausn`, `expo`, `landau`, `landaun` and `polN`, as
ROOT's `AnalyticalIntegral` has them — Cephes's error function and CERNLIB's
`DISLAN`, operation for operation — and numerically otherwise, where ROOT's
adaptive GSL integrator and this one agree to their tolerance rather than to
the bit. ROOT's `FillRandom(h, n, rng)` takes the entries it adds from
`gRandom` whatever it was given; here they come from `rng`, which is the same
thing whenever `rng` is `gRandom`. Filling at random is for one axis:
`TH2::FillRandom` integrates a `TF2` over every cell adaptively, and is not
here.

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
| `GetNumber()` | `f.number` — 100 for `gaus`, 300 + N for `polN`; `f.predefined` names the shape |
| `GetRandom()`, `GetRandom(xmin, xmax)` | `f.get_random(n)`, `f.get_random(n, range=(xmin, xmax))` |

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
— and only those. Integrals of `gaus`, `expo`, `landau` and `polN`
are ROOT's closed forms, and of anything else adaptive 21-point Gauss-Kronrod
to `TF1::Integral`'s tolerance of `1e-12`; extrema and `x_at` are
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

## Fitting

`h.fit(model, option)` is `TH1::Fit`, and `g.fit(...)` `TGraph::Fit`, step for
step: the same points taken from the histogram or the graph, the same
starting values for a built-in shape, the same chi-square or likelihood,
MIGRAD set up as ROOT sets it up, a polynomial solved exactly rather than
iterated, and the fitted function left on what was fitted with what ROOT's
`TF1` records. Minuit comes from [iminuit](https://scikit-hep.org/iminuit/),
which is Minuit2's C++: `pip install xrdroot[fit]`. Without it the fits that
are linear in their parameters still work, and the rest refuse, naming the
extra.

```python
r = h.fit("gaus")  # h->Fit("gaus"): prints ROOT's lines unless "Q"
r = h.fit("gaus", "L R")  # a likelihood fit, in the function's range
r = h.fit(f, "S", range=(0, 5), parameters=[100, 2.5, 0.5], fixed={"Mean": 2.5})
r.parameters, r.errors, r.covariance, r.correlation, r.chi2, r.ndf, r.prob
r.parameter("Sigma"), r.error(2), r.minos, r.status, r.valid, r.edm, r.nfev
print(r.summary())  # FitResult::Print, word for word
h.functions[0]  # the fitted TF1, as ROOT hangs it on the histogram

g.fit("pol1")  # a TGraphErrors: its errors; with x errors, the effective variance
mg.fit("pol1", "F")  # a TMultiGraph, all of its graphs at once
p.fit("pol1"), h2.fit("xygaus")  # a TProfile's means; a TH2 with a TF2
h.fit(lambda x, p: p[0] * np.exp(-x / p[1]), npar=2, parameters=[100, 2])

from xrdroot.fit import minimize, unbinned

minimize(lambda p: (p[0] - 1) ** 2 + (p[1] - 2) ** 2, [0, 0], minos=True)  # Minuit itself
unbinned(x, "gaus", parameters=[1, 0, 1], range=(-5, 5))  # an unbinned likelihood
```

| ROOT | xrdroot |
| --- | --- |
| `h->Fit("gaus")` | `h.fit("gaus")` |
| `h->Fit("gaus", "L R")`, `h->Fit(f, "Q", "", 0, 5)` | `h.fit("gaus", "L R")`, `h.fit(f, "Q", (0, 5))` |
| `TFitResultPtr r = h->Fit(f, "S")` | `r = h.fit(f, "S")` — a `FitResult`, whatever the options |
| `r->Parameter(0)`, `r->ParError(0)` | `r.parameter(0)`, `r.error(0)` — or by name |
| `r->LowerError(0)`, `r->UpperError(0)` | `r.lower_error(0)`, `r.upper_error(0)` — Minos's, with `"E"` |
| `r->Chi2()`, `r->Ndf()`, `r->Prob()`, `r->MinFcnValue()` | `r.chi2`, `r.ndf`, `r.prob`, `r.fcn` |
| `r->GetCovarianceMatrix()`, `GetCorrelationMatrix()` | `r.covariance`, `r.correlation` |
| `r->Status()`, `r->IsValid()`, `r->Edm()`, `r->NCalls()` | `r.status`, `r.valid`, `r.edm`, `r.nfev` |
| `r->Print("V")` | `print(r.summary(covariance=True))` |
| `int(h->Fit(...))` | `int(r)` — the status |
| `f->SetParameters(...)`, `SetParLimits`, `FixParameter` before a fit | `parameters=`, `limits=`, `fixed=`, or the `Function`'s own methods |
| `f->GetChisquare()`, `GetNDF()`, `GetNumberFitPoints()` | `f.fit_result` |
| `gr->Fit("pol1")`, `mg->Fit("pol1", "F")` | `gr.fit("pol1")`, `mg.fit("pol1", "F")` |
| `gMinuit`, `ROOT::Math::Minimizer` with an FCN | `xrdroot.fit.minimize(fcn, x0, ...)` |
| `TTree::UnbinnedFit`, `Fitter::LikelihoodFit` on `UnBinData` | `xrdroot.fit.unbinned(data, model, ...)` |

The options are ROOT's letters, read as `FitOptionsMake` reads them — any
order, either case, a word such as `WIDTH` or `MULTI` taken out before its
letters count:

| Option | What it does | Here |
| --- | --- | --- |
| (none) | Neyman chi-square over the bins with entries, errors from the bins | as ROOT |
| `L` | Poisson likelihood with Baker and Cousins's constant, empty bins in; `chi2` is twice the minimum | as ROOT |
| `WL` | the weighted likelihood: errors corrected by the Hessian of the weights squared | as ROOT; plain `L`, with ROOT's warning, for a histogram without `Sumw2` |
| `L MULTI` | the multinomial likelihood: not extended, so the normalisation is to be held | as ROOT |
| `P`, `PW` | Pearson's chi-square, the expected errors — over the bins' weights for `PW` | as ROOT |
| `W`, `WW` | every error one — the empty bins in too for `WW` — and scaled by `sqrt(chi2/ndf)` after | as ROOT |
| `I` | the function's average over each bin | a 21-point Gauss-Kronrod rule per bin (10-point Gauss per axis in more), where ROOT's adaptive integrator agrees to 1e-9 |
| `WIDTH`, `NORMWIDTH` | the function times the bin's width, or its width over the narrowest | as ROOT |
| `R` | the function's range | as ROOT: only the bins whose centres are inside |
| `B` | the parameters and limits given, no guess for a built-in shape | as ROOT |
| `E` | HESSE and MINOS after MIGRAD | as ROOT; not for `WL`, as ROOT has it |
| `M` | look for a better minimum | MIGRAD run again from where it stopped: Minuit2 has no `IMPROVE` |
| `G` | the function's gradient | Minuit's numerical one, as without `G`; it takes the fit off the linear fitter, as in ROOT |
| `F` | Minuit, even for a `polN` | as ROOT |
| `S`, `C`, `SERIAL`, `MULTITHREAD` | a result object; no chi-square for a linear fit; how to run | nothing to do: the result is always an object, the chi-square always there |
| `Q`, `V`, `VV`, `VVV` | quiet, or the covariance too | as ROOT |
| `N`, `0`, `+` | not stored; stored but not drawn (`kNotDraw`); added rather than replacing | as ROOT |
| `U` | the FCN set on `TVirtualFitter` | there is none, so an ordinary fit, which is what ROOT does without one |
| `EX0`, `ROB` (graphs) | no x errors; the robust linear fit | `EX0` as ROOT; `ROB` refused by name |

**What a fit is.** For a histogram the points are its bins inside the axis
range — x outermost, then y — at their centres, or their edges for `I` and
`WIDTH`; a range, given or the function's with `R`, keeps the bins whose
centres are inside it. A bin of no error is left out of a chi-square unless
`WW` or `P`, and given an error of one in a likelihood. For a graph its error
bars decide, as `GetDataType` decides: no errors — fitted with errors of one,
scaled after; errors in y; errors in x as well, and then the effective
variance, `ey^2 + (ex f'(x))^2` with `f'` Richardson's derivative at ROOT's
step; or asymmetric errors, the lower where the function is below the point
and the upper where it is above. A multigraph takes the most elaborate kind
of any of its graphs, which leaves a plain `TGraph` in it with no points to
give, as in ROOT. Every term is capped at `DBL_MAX / n`, and the terms are
added in order.

**Where it starts.** A `gaus` or `landau` starts at the points' weighted mean
and RMS and a height halfway between the largest value and a Gaussian's of
that area — and its width is bounded to `[0, 10 RMS]`, which stays on the
fitted function, as in ROOT; an `expo` at the line through the logarithms of
its two ends; an `xygaus` or `bigaus` at both. Every parameter is then set
up from the function: fixed where `FixParameter` marked it, bounded where it
has limits, and its first step its error if it has one, a tenth of its range
if bounded, 30% of its value otherwise. MIGRAD runs once, with ROOT's
tolerance of 0.01 and strategy 1, no SIMPLEX first, an error definition of
one for a chi-square and a half for a likelihood; the status is
`Minuit2Minimizer::ExamineMinimum`'s. A fit of a name ROOT keeps in
`gROOT` — `"gaus"`, `"pol1"`, `"expo"`, `"landau"`, `"xygaus"` — starts from
that function as ROOT makes it, over `(-1, 1)` with its standard parameters,
which is why `"R"` with one of those names fits over `(-1, 1)`, as in ROOT.
Any other formula is fitted over the object's own range from its
`parameters=`, or zeros.

**Linear least squares.** A function linear in its parameters, fitted by a
chi-square without an option that needs Minuit and without x errors, is
solved exactly, as `TLinearFitter` solves a `polN`: the parameters, the
covariance and the chi-square to rounding, with a fixed parameter's term
moved to the other side. Whether it is linear is asked of the function
itself, by evaluating it, so `"[0]*sin(x) + [1]"` is solved exactly too,
where ROOT would give it to Minuit; the minimum is the one Minuit converges
to.

**What is recorded.** The function fitted — the one given, or ROOT's standard
one for a name — is left with the fitted parameters, their errors in
`fParErrors`, and `fChisquare`, `fNDF` and `fNpfits`; unless `N`, a copy of it
is put in the histogram's or graph's functions, replacing every function
there unless `+`, with its range set to the one drawn — the fit's range, or
the histogram's axis range, or a graph's frame — and sampled into `fSave`
over it as `TF1::Save` samples it: at the bin centres for a histogram over
more than `fNpx` times its lower end, at `fNpx + 1` points otherwise. A file
written with it is one ROOT reads as fitted. A `TF2` fitted to a `TH2` is
recorded the same way, but a histogram holding one cannot yet be written,
since no `TF2` layout is to hand to write it by.

**How close to ROOT.** `tgme.root`'s multigraph was fitted by ROOT 6.24 with
`mg->Fit("pol1", "FQ")`: three graphs, one without errors, one with
symmetric errors in x and y, one asymmetric in y. The chi-square here at
ROOT's parameters is ROOT's to the last bit, and refitted, the parameters
are ROOT's to a part in 10^10, the chi-square to 10^-15 and the errors to
10^-6: Minuit2, from ROOT's own starting point - `gROOT`'s `pol1` at its
standard parameters of one - took ROOT's steps. ROOT from 6.38 starts a
straight line through errors in x from an unweighted least-squares line
(`InitPolynom`); this starts it where 6.24 did, since that is the ROOT the
file came from. The linear fits are the normal equations' answer, to 10^-12.
ROOT's own documented numbers come out: `stress.cxx`'s integral of a fitted
triple Gaussian over `[-8, 6]` from `FillRandom` at seed 65539, 1923.74578
to within its tolerance of 10 (it is 0.9 off), PyROOT's fit of a Python
Gaussian to `FillRandom("gaus", 200000)` with its 96 degrees of freedom, and
the Minuit example `Ifit.C` that PyROOT's tests check, four parameters and
four errors to two decimal places. Where the fit is Minuit's and no ROOT
answer is to hand, the minimum is checked on a grid of the parameters by
brute force, and against the exact linear solution. A `TEfficiency::Fit`,
ROOT's binomial likelihood of an efficiency, is not here.

## Drawing

ROOT draws on a `TCanvas`; one saved in a file draws as it was (see
[Canvases](#canvases)). Everything else draws through the libraries Python
already draws with — matplotlib, plotly and bokeh — or in plain characters, and everything drawn has the
same `plot()`:

```python
ax = f["h_pt"].plot()  # ROOT's default: HIST, or E once it keeps Sumw2
f["h_pt_mc"].plot(option="HIST SAME", color="kRed+1", label="MC")
ax.figure.savefig("pt.pdf")

fig = f["h2"].plot(backend="plotly", option="LEGO2Z")  # turn it round in a notebook
xrdroot.plot.set_backend("bokeh")  # from now on, unless a call says otherwise
print(f["h_pt"].plot(backend="text"))  # a terminal, a log, a CI transcript
```

`obj.plot(ax=None, backend=None, option="", **style)` is the same on a
histogram, profile, efficiency, graph (plain, with errors, asymmetric or in
layers), multigraph, stack and function, and `xrdroot.plot.plot(obj, ...)`
is the same again. What comes back is the backend's own object — matplotlib
`Axes`, a plotly `Figure`, a bokeh `figure`, a `str` — so styling, saving
and laying out carry on with the library's own calls, and `ax=` draws onto
one you already have. Only the text backend needs nothing installed;
`pip install xrdroot[plot]`, `[plotly]`, `[bokeh]` and `[hep]` (mplhep)
bring the others, and one that is missing is refused with the command that
installs it.

What is drawn does not change with the backend. The object becomes a
picture first — its layers of steps, points, bands, curves and shaded
cells, and the frame round them (`xrdroot.plot.picture(obj, option)` shows
it) — and every backend draws that picture. The look is ROOT's, read off
the object: `fLineColor`, `fFillStyle`, `fMarkerStyle` and the rest,
through ROOT's own colour table (indices 0–50, the pretty palette, the
Petroff sets and the colour wheel, `kRed+1` and all, to the bit) and its
markers; a fit hung on a histogram or graph is drawn over it in its red, as
ROOT draws it, unless the fit was made with option `0`.

### Options

Options are ROOT's, in any case and run together as ROOT takes them
(`"E1SAME"`, `"colz"`). One this does not draw is refused by name with why,
and so is one that means nothing for the object — `COLZ` on a 1-D
histogram — rather than being quietly ignored.

| ROOT | What it draws | Here |
| --- | --- | --- |
| *(none)*, 1-D | `HIST`, or `E` once `Sumw2` is kept; a profile `E` | same, with fits drawn |
| *(none)*, 2-D | `COL` | `COLZ`: the colours need their scale |
| *(none)*, graph | `ALP`, or the graph's `fOption` | same |
| *(none)*, efficiency | `AP`, or `COLZ` in 2-D | same |
| *(none)*, TF1 / TF2 | a line / `CONT3` lines | same, at `fNpx` (and `fNpy`) points |
| `HIST` | the outline, and no fits | steps |
| `E`, `E0`, `E1` | error bars; `E0` for empty bins too, `E1` with ticks | same; `X0` drops the x bars |
| `E2` | error boxes | filled boxes and markers |
| `E3`, `E4` | a band through the bars' ends, `E4` smoothed | same |
| `P`, `*`, `L`, `C` | markers, stars, a line, a smooth line | same |
| `B` | bars | same |
| `TEXT`, `TEXTnn` | the values, at `nn` degrees | same |
| `COL`, `COLZ` | shaded cells, with the scale | same; empty cells unpainted |
| `BOX` | a box per cell, as big as its content | same |
| `CONT`, `CONT0`–`CONT4` | filled bands; `CONT1`–`3` lines | same, `levels=` of them |
| `LEGO`, `SURF` (and their numbers) | 3-D blocks, a surface | matplotlib 3-D axes, plotly surfaces; bokeh refuses |
| 3-D histogram, `BOX`, `ISO` | boxes, iso-surfaces | plotly markers, isosurfaces; the others refuse |
| `A`, `2`, `3`, `4`, `X`, `Z` (graphs) | axes, error boxes, bands, no bars, no ticks | same |
| `SAME` | onto the current pad | onto what that backend last drew on |
| `NORM` | scaled to a sum of one | same; refused for a profile's means |
| `FUNC` | the fits alone | same |
| `NOSTACK`, `NOSTACKB` (stacks) | overlaid, side by side | same |
| `PLC`, `PMC`, `PFC` | colours from the palette | same, spread across it |
| `LOGX`, `LOGY`, `LOGZ` | not options: set on the pad | `logx=True`, `logy=True`, `logz=True` |
| `SCAT`, `ARR`, `PIE`, `POL`, `CYL`, `SPH`, `PSR`, `CANDLE`, `VIOLIN`, `TRI`, `SPEC`, `GL…`, `E5`, `E6`, `HBAR`, `PADS` | — | refused, each with why |

### Style keywords

`color`, `linewidth`, `linestyle`, `fill`, `alpha`, `hatch`, `marker`,
`markersize`, `markercolor` and `label` restyle the marks — a colour may be
ROOT's (`2`, `"kAzure-3"`) or any the backend knows, a marker ROOT's style
number (`20`) or a shape; `title`, `xlabel`, `ylabel`, `zlabel`, `logx`,
`logy`, `logz`, `xlim`, `ylim`, `legend` and `grid` set the frame;
`palette` shades a grid (`"bird"`, ROOT's default, `"viridis"`, or a colour
map of the backend's), and `levels` counts contours. Any other keyword is
the backend's own and is handed to its call for the first layer, so
`zorder=3` reaches matplotlib and `opacity=0.5` plotly.

### Ratios, comparisons and stacks

```python
upper, lower = xrdroot.plot.ratio(data, mc, labels=["data", "MC"])  # TRatioPlot
fig = xrdroot.plot.ratio(h, h.functions[0], "diffsig", backend="plotly")  # pulls
ax = xrdroot.plot.compare([h_2016, h_2017, h_2018], norm=True)
ax = xrdroot.plot.stack([ttbar, wjets, qcd], ["tt", "W+jets", "QCD"])
data.plot(ax=ax, option="E SAME", label="data")
```

`ratio` is `TRatioPlot`: the two above, their ratio below on a shared x
axis about a dashed line — `divsym` (the default) with the errors of both as
`TH1::Divide` gives them, `pois` with the interval on a ratio of two counts
as `TGraphAsymmErrors::Divide` gives it, `diff` and `diffsig`. Against a
function it is the residuals or the pulls of a fit. matplotlib gives back
the two axes, plotly one figure of two rows, bokeh a column of two figures.
`compare` overlays several things in colours told apart — ROOT's ten
Petroff colours unless told — with a legend; `stack` piles histograms as
`THStack` does, each filled, the first at the bottom, and takes a stack's
options (`NOSTACK`, `NOSTACKB`). A `THStack` read from a file draws itself
the same way.

### Styles and labels

```python
ax = h.plot(style="ROOT")  # built in: ROOT's ticks, frame and axis titles
upper, lower = xrdroot.plot.ratio(data, mc, style="CMS")  # mplhep's
xrdroot.plot.label(upper, "CMS", "Preliminary", lumi=138, energy=13)
xrdroot.plot.use_style("ATLAS")  # every matplotlib plot from now on
```

`style="ROOT"` needs nothing installed and styles plotly and bokeh too;
`"CMS"`, `"ATLAS"`, `"LHCb"`, `"ALICE"` and the rest are mplhep's, for
matplotlib, and refused with the install command when mplhep is not there.
`label` writes the experiment in bold, its text after it, and the
luminosity and energy on the right, on whichever backend drew the plot.

### Notebooks

Left at the end of a Jupyter cell, a histogram, graph, profile, efficiency,
stack or function shows as its picture — a small SVG, drawn on a figure
pyplot never hears of so it is not shown twice, or plotly's HTML when
plotly is the backend set. Showing never fails a cell: whatever goes wrong
falls back to the picture in characters, and then to the `repr`. A tree, a
chain, an RNTuple and a directory show as a table of their branches, files,
fields or keys — names, types, entries, classes and cycles — made from
what was read when they were opened, without reading a basket.

### Characters

`text()` is the plainest picture of all, and needs nothing:

```python
print(f["h1d"].text())  # one line per bin: its edges, a bar and the value
print(f["h2d"].text())  # a shaded grid, y upward
print(f["tge"].text())  # a grid of stars with the axis ends labelled
```

A 3-D histogram has no honest flat picture in characters or in
matplotlib, and says to slice `values()` down or draw it with plotly.

## Canvases

A `TCanvas` saved in a file reads as a `Canvas`, and draws the way it looked
in ROOT: a matplotlib figure the canvas's size, one axes per pad, and in each
the things the pad drew, each with the option it was drawn with.

```python
c = f["c1"]
c.width, c.height, c.title  # fCw and fCh, in pixels
[pad.name for pad in c.pads]  # the pads inside it, which may hold pads of their own
[(type(obj).__name__, option) for obj, option in c.primitives]  # what it drew, and how
c.save("c1.png")  # or .pdf, .svg - whatever matplotlib writes
fig = c.plot()  # the figure itself, to keep styling
```

**What is read.** The canvas — its size, its window and its flags, which
`TCanvas::Streamer` writes in an order of its own that no file describes —
and the pad under it: where each pad sits (`fXlowNDC`, `fWNDC`...), its
margins, its range in the units of its axes (`fX1` to `fY2`) and the frame's
as last drawn (`fUxmin` to `fUymax`), `fLogx`/`fLogy`/`fLogz`,
`fGridx`/`fGridy`, `fTickx`/`fTicky`, its fill and border. A pad's list of
primitives keeps the option beside each entry — `"hist same"`, `"ap"`,
`"colz"` — as `Pad.primitives`, a list of `(object, option)`; the list
itself, anywhere in any file, is a `list` that carries them as `.options`.
Histograms, profiles, graphs, stacks, multigraphs and functions come back as
the classes they always are. The drawing classes — `TFrame`, `TPave`,
`TPaveText`, `TPavesText`, `TPaveLabel`, `TPaveStats`, `TPaletteAxis`,
`TLegend` and `TLegendEntry`, `TText`, `TLatex`, `TLine`, `TArrow`, `TBox`,
`TWbox`, `TEllipse`, `TArc`, `TCrown`, `TMarker`, `TPolyLine`, `TPolyMarker`,
`TGaxis` and `TColor` — come back as a
`Primitive`: the class, and its members by name however deep in its bases
ROOT keeps them (`text["fTextSize"]`). The colours a canvas saved with it,
its `ListOfColors` and palette, are used to draw it and kept out of
`primitives`.

**What is drawn**, and by what option. The data — histograms, profiles,
graphs, multigraphs, stacks and functions — is drawn as [Drawing](#drawing)
draws it: the option the pad kept, read the same way, the same attributes
read off the object, the same layers, onto the pad's axes. What a pad adds:

| Class | Drawn as |
| --- | --- |
| data | onto the pad's frame, whose range, scales and titles are the pad's; a `COLZ` scale where its `TPaletteAxis` was, or in the pad's right margin; each graph of a multigraph by the option it was added with, and the multigraph's fits over them; in the colours and palette the canvas saved, when it saved any |
| stats box | a saved `TPaveStats` with the lines it was saved with, a name on the left and its value on the right; a histogram saved without one, not drawn `SAME` nor told `kNoStats`, gets `gStyle`'s — its name, entries, mean and standard deviation (`SetOptStat(1111)`) |
| title | the `title` pave a drawn pad saved; one saved undrawn gets its histogram's or graph's title at the top, as `gStyle` puts it |
| `TText`, `TLatex` | at `fX`, `fY` in the axes' units or, `SetNDC`, the pad's fractions; `TLatex`'s `#` mathematics laid out as `TLatex` lays it out — Greek letters and symbols from the Symbol font, `^{}` and `_{}`, `#sqrt` and `#sqrt[n]`, `#frac`, `#splitline`, `#sum` and `#int` with their limits, `#bar` and the other accents drawn as lines, `#left` and `#right` brackets grown to fit, `#it`, `#bf`, `#font`, `#color`, `#scale`, `#kern`, `#lower` — and a string whose braces do not match drawn as nothing, as ROOT draws it |
| `TLine`, `TArrow`, `TBox`, `TEllipse`, `TArc`, `TCrown`, `TMarker`, `TPolyLine`, `TPolyMarker` | as their attributes say; an arrow's head by its `fOption` (`"|>"`, `"<|>"`, `"->-"`...), an ellipse's or a crown's slice by `fPhimin` and `fPhimax`, a polyline filled when drawn `f`, a box outlined only when it is hollow |
| `TGaxis` | as `TGaxis::PaintAxis` draws one, and a frame's axes too: a line graduated from `fWmin` to `fWmax` in round steps as `fNdiv` asks (`THLimitsFinder`'s), logarithmically for `G` in `fChopt`, its ticks on the side `+` or `-` names and its labels opposite (or with them, `=`; none, `U`), a common `×10^n` where the labels would be long, its title at its far end |
| `TPave`, `TPaveText`, `TPaveLabel`, `TLegend` | the box, its border and shadow (on the sides `fOption` names), and its lines stacked in it, or its entries in `fNColumns` columns as wide as their widest labels, each a symbol — `l` line, `p` marker, `f` fill, `e` error bar, `h` a header of its own row — drawn in the style of the thing it stands for |
| `LEGO`, `SURF` | a two-dimensional histogram as `THistPainter::PaintLego` and `PaintSurface` draw it: seen from the pad's `fTheta` and `fPhi` through `TView3D`, its blocks or its mesh drawn front to back with the lines behind them hidden (`TPainter3dAlgorithms`' moving screen), the box's back walls lined at the z axis's divisions, and its three axes along the box's nearer edges |
| three dimensions | a `TH3` drawn `LEGO` or `BOX` as `PaintH3BoxRaster` draws it, a box in each bin as big as the cube root of its share of the highest, hidden lines hidden by the raster screen; drawn `ISO`, the surface where its contents cross their mean, lit as `PaintH3Iso` lights it and filled in 28 shades of its fill colour - found through tetrahedra where ROOT uses its own marching cubes, so its triangles differ a little |
| `CONT1`, `CONT2`, `CONT3` | contour lines as `PaintContour` finds them, cell by cell of the bins' centres at `gStyle`'s twenty levels: each level in its palette colour, its line style, or all in the histogram's own line |

Colours are ROOT's by index, from the table [Drawing](#drawing) uses —
ROOT's own, `kOrange` to `kPink` and all — and a canvas saved with its
colours draws in those. Line styles, marker styles and sizes, fill styles (hollow, solid, the 3000s as
hatches, the 4000s as transparency), fonts (family, italic, bold) and
alignment are ROOT's numbers translated; a size is in pixels for a font of
precision 3 and a fraction of the pad's shorter side otherwise, the figure
drawn at 100 dots to the inch so that one of ROOT's pixels is one of its.
Text left at size 0 in a pave or a legend is sized to fit, as ROOT sizes it.

**How it is drawn**, to be ROOT's picture pixel for pixel where it can. ROOT
draws a canvas saved as a PNG in batch through `TImageDump`, and so does
this, onto matplotlib's Agg: every point is rounded to a whole pixel as
`TImageDump` rounds it, and every line is set pixel by pixel as `TASImage`
sets it — straight runs as runs, slopes by Bresenham's walk, wide lines with
a square brush, dashes a quarter of `TStyle`'s lengths — rather than
stroked. A marker is `TImageDump`'s shape for its style, at its size in whole
pixels. Text is placed as FreeType places it for ROOT: unhinted, at
`TTF`'s size of the pad's shorter side times 0.93376, aligned by the glyphs'
advances and ascents. Saved as a PDF or an SVG, the same lines are stroked.

**Fonts.** ROOT's fonts 4x are TeX Gyre Heros, a Helvetica; 1x to 3x and 13x
FreeSerif, a Times; 8x to 11x FreeMono, a Courier; 12x and 15x its Symbol.
None of them is shipped here, and none is needed: each is the first
TrueType face installed of a list with the same widths - for 4x `TeX Gyre
Heros`, `Nimbus Sans`, `Helvetica`, `Arial`, `Liberation Sans`, `FreeSans`,
then `DejaVu Sans`, which matplotlib always has; the Times and Courier lists
are alike. Helvetica's widths are the ones the picture is laid out to; with
`DejaVu Sans`, which is wider, text is wider and what is sized to fit is
smaller. Greek letters and symbols come from a `Symbol` font when one is
installed (macOS has one with ROOT's advances), else from matplotlib's own
`STIX`.

**What is left out**, with a warning naming every one: a class the file
does not describe, or does but this does not draw — a `TButton`,
anything of a GUI — a three-dimensional histogram, and a function that
cannot be evaluated here. An option ROOT takes that is not drawn here
(`SCAT`, `*H`, the `[]` of an asymmetric graph) is drawn as the object would
be without it; each is said in the warning. `LEGO` and `SURF` are drawn
in three dimensions, and so is a three-dimensional histogram drawn `LEGO`,
`BOX` or `ISO`; one drawn with no option, a scatter of points in ROOT, is not. Writing a canvas is not supported; reading one never
stands in the way of writing what it drew.

`xrdroot.canvas.render(obj, path)` saves a `Canvas`, a `Pad`, or the members
of either as a dictionary, which is what a tool handed any object reaches for.

## A ROOT you can import

`import xrdroot.pyroot as ROOT` is ROOT's own Python namespace - the one a
PyROOT script is written against - over this library. Its classes have
ROOT's names, take ROOT's arguments in ROOT's order with ROOT's defaults, and
print what ROOT prints; underneath, each keeps the xrdroot object it stands
for in `._xrd`, so a `TH1F` is a `Histogram`, a `TGraph` a `Graph`, a `TF1`
a `Function`, and everything written, fitted or drawn is done by the rest of
the library.

```python
import xrdroot.pyroot as ROOT

f = ROOT.TFile("hsimple.root", "RECREATE", "Demo ROOT file with histograms")
hpx = ROOT.TH1F("hpx", "This is the px distribution", 100, -4, 4)
hprof = ROOT.TProfile("hprof", "Profile of pz versus px", 100, -4, 4, 0, 20)
for i in range(25000):
    px, py = ROOT.gRandom.Rannor()              # or Rannor(a, b) into ctypes doubles
    hpx.Fill(px)
    hprof.Fill(px, px * px + py * py)
r = hpx.Fit("gaus", "S")                         # prints ROOT's fit summary
print(r.Parameter(2), hpx.GetFunction("gaus").GetChisquare())
f.Write()                                        # every histogram booked since the file opened
f.ls()
f.Close()
```

**Where things are kept.** As in ROOT, a histogram made while a file is
open is kept in that file's directory - `gDirectory` - and `Write()` writes
what the directory holds; made with no file open, it is kept in `gROOT`,
where `gROOT.FindObject("hpx")` finds it. One made with a name already kept
there replaces it, with ROOT's `Replacing existing TH1F: hpx (Potential
memory leak)` on standard error; `SetDirectory(ROOT.nullptr)` keeps it
nowhere, and `TH1.AddDirectory(False)` stops the keeping altogether. A
`TF1` goes into `gROOT.GetListOfFunctions()`, so `FillRandom("myfunc")` and
`Fit("myfunc")` find it by name. `TFile.Get` hands back ROOT's classes - a
`TH1F` read is a `TH1F` - and keeps a histogram or tree it read, so a second
`Get` is the same object; `f.hpx` is `f.Get("hpx")`, as in PyROOT.
`TFile("x.root", "READ" | "RECREATE" | "UPDATE" | "NEW")` reads with
`open_root`, writes with `create` and adds with `update`, any URL they take,
and a file that will not open is a zombie with ROOT's error, `TFile.Open`
handing back `None`.

**What prints.** `Print` and `ls` print what ROOT prints for the common
cases - `TH1.Print Name  = h, Entries= 1, Total sum= 1` and its bins with
`"all"`, a graph's `x[0]=1, y[0]=2`, `TFile**`/`TFile*` and each `KEY:` line
with `[current cycle]` and `[backup cycle]`, the `Formula based function:`
of a `TF1`, a `TLorentzVector`'s `(x,y,z,t)=(...)`, `TStopwatch`'s `Real
time 0:00:01, CP time 0.990` - and each was checked against ROOT 6.40. ROOT's
messages go to standard error as `Warning in <TROOT::Append>: ...`,
`gErrorIgnoreLevel` obeyed.

**Numbers ROOT's way.** `gRandom` is `xrdroot.random.gRandom`, so the same
seed gives ROOT's draws, and `FillRandom` and `GetRandom` take theirs from it
as ROOT does. Statistics honour an axis's range as ROOT's do - after
`GetXaxis().SetRangeUser(a, b)`, `GetMean` and `Integral` are of the bins in
it - and `UnZoom`, like ROOT's, needs a pad. `TMath` is the whole of ROOT's
commonly used set (`Prob`, `Gaus`, `Landau`, `Poisson`, `BinomialI`,
`StudentQuantile` by Hill's algorithm as ROOT's, `KolmogorovTest`, `Median`,
`RMS` with ROOT's `n - 1`, `Sort`, `BinarySearch`, `Nint` rounding halves to
even...) and `ROOT.Math` MathCore's distributions (`normal_cdf`,
`chisquared_cdf_c`, `tdistribution_quantile`, `gamma_pdf`, `poisson_cdf`,
`beta_quantile`...), all checked against what ROOT 6.40 answers. Every
`ROOT.Math` function, the Legendre polynomials among them, may be written in
a formula too, as C++ lets ROOT's be: `TF1("f", "ROOT::Math::normal_pdf(x,
[0], [1])", -5, 5)`. A histogram filled by label is one of categories, as in
ROOT: a new label takes the next free bin, a full axis doubles, and
`LabelsDeflate` and `LabelsOption("a")` trim and sort it.

| What | ROOT's names here |
| --- | --- |
| objects | `TObject` (`GetName`, `ClassName`, `IsA().GetName()`, `InheritsFrom`, `Clone`, `Copy`, `Draw`, `DrawClone`, `Print`, `ls`, `Write`, `SaveAs`, bits), `TNamed`, `TClass`, the `TAttLine`/`TAttFill`/`TAttMarker`/`TAttText` setters and getters - kept in the xrdroot object's own members - and every `EColor`, style, palette and `kTRUE`/`kFALSE` as ints, so `kRed + 2` is 634 |
| the session | `gROOT` (`SetBatch`, `IsBatch`, `GetListOfFiles`/`Canvases`/`Functions`/`Styles`, `FindObject`, `GetFunction`, `SetStyle`, `ProcessLine` and `gInterpreter.Declare` through `xrdroot.cint`, `Reset`, `GetVersion` as `6.40.04`, `GetTutorialDir` from `$ROOT_TUTORIAL_DIR`), `gDirectory`, `TDirectory.TContext`, `gSystem` (`Load` and `AddIncludePath` succeed and do nothing; `Getenv`, `Setenv`, `Exec`, `AccessPathName` - true when the path is **not** there, as ROOT's is - `mkdir`, `Which`, `BaseName`, `DirName`, `ProcessEvents`, `Sleep`...), `ROOT.ROOT.EnableImplicitMT`, `TSeqI`, `Float_t`/`Int_t` and the other number types as casts, `Info`/`Warning`/`Error`, `SetOwnership`, `nullptr` |
| text and time | `TString` (a string that changes in place: `ReplaceAll`, `Append`, `ToLower`, `Contains`, `Tokenize`, `Form`, `TString.Format`...), `Form` and `Printf` with `printf`'s formats, `TStopwatch`, `TBenchmark`/`gBenchmark`, `TDatime` |
| files | `TFile`, `TDirectoryFile` (`Get`, `GetListOfKeys`, `GetKey`, `mkdir`, `cd`, `GetDirectory`, `Write`, `WriteObject`, `ls`, `Close`, a context manager), `TKey` (`GetName`, `GetClassName`, `GetCycle`, `ReadObj`) |
| histograms | `TH1`/`TH2`/`TH3` in `C`, `S`, `I`, `F` and `D` (arguments by position or by name, `nbinsx=`...), `TProfile`, `TProfile2D`, `TProfile3D`: `Fill`, `FillN`, `Get`/`SetBinContent` and `Error` by global or per-axis bin, `GetBin`, `FindBin`, `GetMean`, `GetStdDev`, `GetRMS`, `GetEntries`, `Integral`, `IntegralAndError`, `GetMaximum`/`MaximumBin`, `SetMaximum`/`Minimum`, `Scale`, `Add`, `Multiply`, `Divide`, `Rebin`, `ProjectionX`/`Y`, `ProfileX`/`Y`, `Project3D`, `GetCumulative`, `Fit`, `GetFunction`, `FillRandom`, `GetRandom`, `KolmogorovTest`, `Chi2Test`, `Smooth`, `Sumw2`, `Reset`, `SetStats`, `Print`; `TAxis` (`SetTitle`, `SetRange`, `SetRangeUser`, `SetBinLabel`, `GetBinCenter`...); `THStack`; UHI's `values()`, `variances()`, `h[...]` |
| graphs | `TGraph`, `TGraphErrors`, `TGraphAsymmErrors` (`SetPoint` past the end grows it, `GetPoint`, `GetX`, `SetPointError`, `Eval`, `Fit`, `GetHistogram` - ROOT's frame, a tenth wider than the points), `TMultiGraph` |
| functions and fits | `TF1`, `TF2`, `TF3`, `TFormula` - from a formula, or from a Python `fn(x, p)` as PyROOT calls it - with `SetParameters`, `SetParNames`, `SetParLimits`, `FixParameter`, `Eval`, `Integral`, `Derivative`, `GetMaximumX`, `GetX`, `GetRandom`, `Moment`; `TFitResultPtr` and `TFitResult` (`Parameter`, `ParError`, `Chi2`, `Ndf`, `Prob`, `GetCovarianceMatrix`, `Print`); `TVirtualFitter.GetFitter().GetConfidenceIntervals`, the latest fit's band; `IntegralError`; `TEfficiency` |
| mathematics | `TMath`, `ROOT.Math` (its distributions, `MinimizerOptions`, and GenVector's `PtEtaPhiMVector`, `PxPyPzEVector`, `XYZVector`... kept in their own coordinates and printed as `operator<<` prints them, with `VectorUtil`), `TVector2`, `TVector3`, `TRotation`, `TLorentzVector`, `TRandom`, `TRandom1`, `TRandom2`, `TRandom3` |
| containers | `TList`, `TObjArray`, `THashList`, `TIter`, `TObjString` |

The namespace is put together from `xrdroot.pyroot.SUBMODULES`: this core,
the trees (`TTree`, `TChain`, `RDataFrame`...), the STL stand-ins, RDF and
the graphics, each a part of the kit that may or may not be installed. A
name ROOT has and none of them does is refused by name - `ROOT has
TGraphSmooth; xrdroot.pyroot does not yet` - so a script says exactly what it
missed.

## Canvases you draw on

`xrdroot.pyroot` is ROOT's own Python namespace, and its graphics are ROOT's
in batch mode: a macro makes a `TCanvas`, divides it, draws into its pads and
saves it, and the picture is drawn by the same code that draws a canvas read
from a file ([Canvases](#canvases)).

```python
import xrdroot.pyroot as ROOT

c = ROOT.TCanvas("c1", "Two views", 800, 400)
c.Divide(2, 1)
c.cd(1)
h.Draw("E1")                        # any object with an ._xrd draws as what it wraps
ROOT.gPad.SetLogy()
legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
legend.AddEntry(h, "data", "lep")
legend.Draw()
c.cd(2)
ROOT.TLatex().DrawLatexNDC(0.2, 0.8, "#sqrt{s} = 13 TeV")
ROOT.gStyle.SetOptStat("nemr")
c.SaveAs("views.png")               # .pdf .svg .eps .ps .jpg .gif too
c.Print("book.pdf[")                # a book of pages, as ROOT makes one
c.Print("book.pdf")
c.Print("book.pdf]")
```

**How a live pad is drawn.** A `TPad` keeps what a saved one holds - its
members by ROOT's names (`fXlowNDC`, `fLeftMargin`, `fLogy`, `fGridx`...) and
its primitives, each with the option it was drawn with - and when it is saved
it becomes the `Canvas` and `Pad` a file would have given: data as the
histogram, graph or function it wraps (`._xrd`), every drawing class as a
`Primitive` of its members. What ROOT adds when it paints a pad is made then,
from `gStyle`, the way `THistPainter` makes it: the `TFrame`; the `title`
pave at `TitleX`/`TitleY` by `TitleAlign`; and a `TPaveStats` after each
histogram shown with its statistics, and each fitted graph with `SetOptFit`,
of the lines `PaintStat` writes (`"Entries = 1000"`), at `StatX`/`StatY`, a
quarter of `StatH` tall per line and 1.8 times as wide for a fit. `Update()`
makes them at once, so `gPad.GetPrimitive("stats")` finds the box to move,
and a box moved keeps its place. The colours a session makes and a palette
other than `kBird` go with the canvas as the colour tables a saved one
carries.

**Drawing.** `Draw` on anything - through the hook the core module's
`TObject.Draw` calls - puts it on the current pad, `gPad`, making ROOT's
default canvas `c1` (700 by 500) when there is none. Data drawn without
`SAME` clears the pad first, as `TH1::Draw` does: a histogram, function,
stack or efficiency always, a graph or multigraph when drawn with `A`; a
graph drawn alone draws its own axes. The frame is the first histogram's, or
a graph's drawn `A`, with an axis's `SetRangeUser` (its `fFirst`/`fLast`)
and `SetMinimum`/`SetMaximum` respected; `GetUxmin()` and the rest give it,
in powers of ten on a logarithmic axis as ROOT does. `DrawFrame` draws an
empty `hframe`, `Range` sets a pad's coordinates for what is drawn with no
frame, and `BuildLegend` makes a legend of what the pad draws.

**Sizes.** A canvas's window is the size it is made - `(w, h)`,
`(x, y, w, h)`, a form number, or `gStyle`'s `CanvasDefW` by `CanvasDefH` -
and its picture, as in ROOT's batch mode, is that less the window's
decoration: 4 pixels narrower and 28 shorter, so 700 by 500 saves as 696 by
472. A negative width asks for the picture itself to be that size.

| What | ROOT's names here |
| --- | --- |
| canvases and pads | `TCanvas`, `TPad`, `gPad`: `cd`, `Divide`, `Draw`, `Clear`, `Close`, `Update`, `Modified`, `Paint`, `SetLogx`/`y`/`z`, `SetGrid`, `SetGridx`/`y`, `SetTicks`, `SetTickx`/`y`, `SetLeftMargin` and the other three, `SetMargin`, `SetFillColor`, `SetFrameFillColor` and the frame's line and border, `SetBorderMode`/`Size`, `Range`, `GetRange`, `GetUxmin`/`Uxmax`/`Uymin`/`Uymax`, `GetFrame`, `DrawFrame`, `BuildLegend`, `GetListOfPrimitives`, `FindObject`, `GetPrimitive`, `GetPad`, `GetMother`, `GetCanvas`, `SaveAs`, `Print`, `ls`; a canvas's `GetWw`/`Wh`, `GetWindowWidth`/`Height`, `SetCanvasSize`, `SetWindowSize` |
| styles | `TStyle`, `gStyle`: a `Set` and `Get` for every field of ROOT's - `SetOptStat` (digits or `"nemruoisk"`), `SetOptFit`, `SetOptTitle`, `SetStatX`..., `SetTitleX`..., `SetTitleFontSize`, `SetLabelSize(size, "xyz")` and every per-axis field, `SetPadTickX`, `SetPalette(number or kBird, kRainBow...)` or colours of one's own; `set_style("Plain")` for `gROOT->SetStyle` - `Modern`, `Plain`, `Classic`, `Default`, `Bold`, `Video`, `Pub`, `ATLAS`, `BELLE2` |
| colours | `TColor(index, r, g, b)`, `TColor.GetColor(r, g, b)` or `("#rrggbb")` (an existing colour when there is one), `GetFreeColorIndex` (1179 at the start, as in ROOT), `CreateGradientColorTable`, `GetColorBright`/`Dark`/`Transparent`; the palettes `kDeepSea` to `kCividis` by name |
| text | `TText`, `TLatex`, `TMathText`: `DrawText`, `DrawLatex`, `DrawLatexNDC`, `SetNDC`, `SetTextAlign`/`Size`/`Font`/`Angle`/`Color` |
| paves and legends | `TPave`, `TPaveText` (`AddText`, `AddLine`, `GetLine`, `SetAllWith`), `TPaveLabel` (`DrawPaveLabel`), `TPaveStats`, `TLegend` (`AddEntry(obj or name, label, "lpfe")`, `SetHeader`, `SetNColumns`, `SetBorderSize`, `SetMargin`), `TLegendEntry` |
| shapes | `TLine`, `TArrow`, `TBox`, `TWbox`, `TEllipse`, `TArc`, `TCrown`, `TMarker`, `TPolyLine`, `TPolyMarker`, `TGaxis`, and their `DrawLine`, `DrawBox`, `DrawArrow`, `DrawEllipse`, `DrawArc`, `DrawMarker`, `DrawPolyLine`, `DrawAxis` |

Every drawing class takes its attributes from `gStyle` when it is made, as
ROOT 6's do - text in font 62, fills in colour 19 - and has the
`TAttLine`, `TAttFill`, `TAttMarker` and `TAttText` methods of its C++ class
and a `Set`/`Get` for each coordinate. A name ROOT has that is not here is
refused by name, `ROOT has TGraphSmooth; xrdroot.pyroot does not yet`.

`xrdroot.pyroot.graphics.compare_images(a, b)` says how alike two pictures
are - files or arrays - as their structural similarity, 1 for the same
picture: scikit-image's when it is installed, else the same formula in
NumPy.

## Geometry

A detector geometry is built as ROOT builds one - a `TGeoManager`, its
materials and media, shapes made into volumes, and volumes placed in
volumes by translations and rotations - and drawn in a pad as ROOT draws
one in batch: a wireframe of every visible volume in the pad's perspective,
each in its volume's line colour, style and width. The picture is pixel for
pixel ROOT 6.40's for its `rootgeom.C` tutorial, and for a tree of the
older `TGeometry` package's shapes, drawn from a macro or read back from
the file ROOT wrote it to.

```python
import xrdroot.pyroot as ROOT

geom = ROOT.TGeoManager("simple1", "Simple geometry")
vacuum = ROOT.TGeoMedium("Vacuum", 1, ROOT.TGeoMaterial("Vacuum", 0, 0, 0))
aluminium = ROOT.TGeoMedium("Al", 2, ROOT.TGeoMaterial("Al", 26.98, 13, 2.7))
top = geom.MakeBox("TOP", vacuum, 270, 270, 120)
geom.SetTopVolume(top)
bar = geom.MakeTubs("bar", aluminium, 5, 15, 5, 90, 270)
bar.SetLineColor(ROOT.kRed)
top.AddNode(bar, 1, ROOT.TGeoCombiTrans(10, 0, 0, ROOT.TGeoRotation("r", 90, 0, 0)))
geom.CloseGeometry()                 # counts nodes and levels, on standard error, as ROOT does
top.Draw()
ROOT.gPad.SaveAs("geometry.png")
```

**What is here.** Every `TGeo` solid ROOT tutorials build - boxes,
trapezoids, `TGeoArb8`, parallelepipeds, tubes and cones and their segments,
cut tubes, spheres, tori, elliptic tubes, paraboloids, hyperboloids,
polycones and polygons, extruded polygons, tessellated solids (from
Wavefront `.obj` files too) and Boolean composites, which are drawn as their
components are; `TGeoTranslation`, `TGeoRotation` (Euler's angles or
GEANT3's, and back), `TGeoCombiTrans`, `TGeoHMatrix`; `TGeoVolume`,
`TGeoVolumeAssembly`, `TGeoNode`, `TGeoIterator` and iterator plugins; the
visible depth and the visualisation options (`SetVisLevel`,
`SetVisOption`, `SetTopVisible`, `VisibleDaughters`); and the old package's
`TGeometry`, `TNode` with its seven visibility codes and `SetBomb`'s
exploded view, `TBRIK` and its kin, `TMaterial`, `TRotMatrix`. A pad's 3-D
view - `TView::CreateView`, `SetRange`, `RotateView`, `Front`/`Top`/`Side`
- is shared with `TPolyLine3D` and `TPolyMarker3D`.

**A scene to turn round.** `xrdroot.geom.backends` hands the same solids to
plotly (`to_plotly`, a figure of shaded meshes for a notebook or an HTML
page) or to pyvista (`to_pyvista`, rendered off-screen; `save(..., "x.png")`
takes a screenshot with hidden surfaces hidden). Neither is needed for a
pad's picture; both are the `geom` extra, `pip install xrdroot[geom]`, and
asking for one that is not installed says so.

**What is refused.** Whatever needs ROOT's navigator - where a point is,
where a ray goes (`RandomRays`), which volumes overlap (`CheckOverlaps`),
parallel worlds, physical nodes - is refused by name, as are GDML import
and export, `TGeoVolume::Divide`, the table of radionuclides (read from a
data file ROOT installs), and writing a `TGeometry`, whose streamer is
ROOT's own code. The widgets that exist only to be clicked - `TControlBar`
and `TSlider` - are refused too; a `TButton` is drawn, as a picture shows
it.

## More of ROOT's classes

**Polar graphs, polygons and cuts.** `TGraphPolar` is drawn on its
`TGraphPolargram` - circles, spokes and labels in degrees, radians or grads
- as ROOT draws it; `TH2Poly` has bins of any polygon (`AddBin`,
`Honeycomb`, filled by point or by name, drawn `COL` with its palette's
range); `TCutG` says which points are inside it as ROOT's crossing test
does, and cuts what is drawn with `[cutname]`; `TExec` runs its line of C++
each time its pad is painted - what it paints with `PaintText` put where it
stands among the pad's primitives - and `TPavesText` is the stacked pave.

**XML.** `TXMLEngine` makes, walks, changes, reads and writes documents as
ROOT's does: nodes, attributes and namespaces by pointer, `nullptr` being
`None`, and a document written byte for byte as `SaveDoc` lays one out.

**FITS files.** `TFITSHDU` opens one unit of a FITS file - by number, by
`EXTNAME`, with a CFITSIO row filter, `"f.fits[1][DATAMAX > 2e-15]"` - read
with NumPy alone: its header records, an image's pixels (as a `TMatrixD`, a
`TImage`, a histogram or rows of a `TVectorD`) and a binary or ASCII table's
columns, fixed-length and variable-length arrays included. `Print` prints
what ROOT's prints, character for character. A table of bit columns
(`TFORM` `X`) and CFITSIO's column and binning filters are refused.

**ROOT 7's histograms.** `ROOT.Experimental.RHist['int']` (or `'double'`,
or `RBinWithError`) fills along `RRegularAxis` and `RVariableBinAxis` axes
as `ROOT::Experimental` does, with its statistics of every fill;
`Experimental.Hist.ConvertToTH1D` makes the `TH1` of it, and
`RDataFrame.Hist` books one on a frame.

**ROOT 7's RNTuple classes.** `RNTupleModel::Create()` and `MakeField<T>`,
`RNTupleWriter::Recreate` and `Append`, `RNTupleParallelWriter` with its
fill contexts (staged clusters too), `RNTupleReader` with `LoadEntry`,
views, collection views, `PrintInfo()` and `Show(i)` as ROOT prints them, and
`RNTupleProcessor` chains and joins - all over the RNTuple reader and writer
below. A field holds a number, a `std::string` or a `std::vector` of a
number; records, nested collections and low-precision floats are refused,
as are `PrintInfo`'s storage details, which are the sizes ROOT's own writer
chose.

**ROOT 7's graphics.** `RCanvas`, its pads, frames and `Divide`, the
primitives `Draw<T>` puts on them - `RLine`, `RBox`, `RText`, `RMarker`,
`RPave`, `RPaveText`, `RFrameTitle`, `RAxisDrawable`, `TObjectDrawable` -
with their attribute groups (`RAttrLine`, `RAttrText`, `RAttrAxis`...),
`RColor`, pad lengths (`0.1_normal - 5_px`) and `RStyle` sheets. ROOT shows
these in a web browser; run in batch there is none, so `Show` and `Update`
show nothing, as in ROOT's batch mode, and `SaveAs` - which ROOT renders
through a headless browser - writes no file and says so. `gEnv`, ROOT's
resources from `.rootrc` files, is there for the settings macros read and set.

**A macro's random numbers.** `std::mt19937`, `std::uniform_real_distribution`
and `std::normal_distribution` draw what libc++ draws, number for number, so
a macro that fills from them fills what ROOT filled.

## Ratio plots

`TRatioPlot` is ROOT's plot of two histograms with their ratio beneath, or of
a fitted histogram with its fit's residuals beneath, and as in ROOT it is a
layout of pads rather than a picture of its own: the upper pad holds the
histograms, the lower - below `SetSplitFraction`, three tenths of the height
by default - what is worked out from them, and a clear pad over both the
axes, so that the two share one x axis. It is built on the live pads, so
`GetUpperPad()` is a `TPad` to draw a legend in, and the canvas saved is
drawn as ROOT draws it.

```python
import xrdroot.pyroot as ROOT

c = ROOT.TCanvas("c", "A ratio example")
rp = ROOT.TRatioPlot(h1, h2)                 # h1 / h2, by TGraphAsymmErrors::Divide "pois"
rp.Draw()
rp.GetLowYaxis().SetNdivisions(505)
rp.GetUpperPad().cd()
ROOT.TLegend(0.3, 0.7, 0.7, 0.85).Draw()

fit = ROOT.TRatioPlot(h, "errfunc")          # h fitted: (h - f) / sqrt(f), with its bands
fit.SetGridlines([-2, -1, 0, 1, 2])
fit.SetConfidenceIntervalColors(ROOT.kBlue, ROOT.kRed)
fit.Draw()
fit.GetLowerRefGraph().SetMinimum(-2)
```

**What the lower pad shows.** By default the two histograms are divided by
`TGraphAsymmErrors::Divide` with the constructor's option passed on
(`"pois"` unless another is given) - a divide that is ROOT's line for line,
every interval with it: Clopper-Pearson, `n`, `w`, `ac`, `midp`, `b(a,b)`
with `mode` and `cen`, `cl=`, `pois`, `e0` and `v`, weighted histograms
counted by their effective entries. `divsym` divides with `TH1::Divide`,
into symmetric errors; `diff` subtracts, and `diffsig` divides each
difference by its bin's error. For one fitted histogram each bin's residual
is shown over its error - with `errasym` the error on the side the function
lies, from `SetBinErrorOption`'s Poisson intervals, with `errfunc` the root
of the function - and the latest fit's one- and two-sigma bands, or those
of the `TFitResult` given, are drawn about it, divided by the same error.
Each mode draws ROOT's dashed reference lines: at 0.7, 1 and 1.3 for a
ratio, 0 for a difference, and -1, 0 and 1 for a significance or residual.

**Its axes.** Each pad's own axes are hidden, and a `TGaxis` is drawn at
each frame's edge, graduated over the frame's range and styled from the
axis it stands for - the upper x axis unlabelled, the lower y axis's ticks
lengthened by the ratio of the frames' heights - with twins on the top and
right when the parent pad has ticks there or its frame no fill. They are
placed afresh whenever the canvas is drawn, so a range or title set after
`Draw` is shown. Where the pads nearly touch (`SetSeparationMargin` below
0.025) the lower axis's top label is hidden, or the upper's bottom one with
`hideup`; `fhideup` and `fhidelow` hide one regardless, `nohide` neither.
Every tutorial's picture is ROOT's to a structural similarity above 0.97.

| What | ROOT's names here |
| --- | --- |
| making one | `TRatioPlot(h1, h2[, option])`, `(stack, h2)`, `(h1, stack)`, `(h1[, option, fitresult])`; the options `divsym`, `diff`, `diffsig`, `errasym`, `errfunc` and `Divide`'s own |
| drawing | `Draw` with `grid`/`nogrid`, `confint`/`noconfint`, `hideup`, `hidelow`, `fhideup`, `fhidelow`, `nohide`; `SetH1DrawOpt`, `SetH2DrawOpt`, `SetGraphDrawOpt`, `SetFitDrawOpt` |
| its parts | `GetUpperPad`, `GetLowerPad`, `GetUpperRefObject`, `GetUpperRefXaxis`/`Yaxis`, `GetLowerRefGraph`, `GetLowerRefXaxis`/`Yaxis`, `GetXaxis`, `GetUpYaxis`, `GetLowYaxis`, `GetCalculationOutputGraph`, `GetConfidenceInterval1`/`2` |
| layout | `SetSplitFraction`, `SetInsetWidth`, `SetSeparationMargin`/`GetSeparationMargin`, `SetLeftMargin`, `SetRightMargin`, `SetUpTopMargin`, `SetUpBottomMargin`, `SetLowTopMargin`, `SetLowBottomMargin` |
| the calculation | `SetGridlines(vector)` or `(array, n)`, `SetConfidenceLevels`, `SetConfidenceIntervalColors` (numbers or `"kBlue"`), `SetC1`, `SetC2`, `SetFitResult` |

`Divide`'s Feldman-Cousins interval (`fc`) and its shortest Bayesian one
(`sh`, or `mode` without `cen`) are refused by name.

## Spectra

`TSpectrum` and `TSpectrum2` are Miroslav Morhac's spectrum processing, as
ROOT's `hist/spectrum` has it: the background under peaks by SNIP clipping,
Markov-chain smoothing, Gold's and Richardson-Lucy's deconvolution,
unfolding through a response matrix, and the high-resolution peak search
that deconvolves by a Gaussian before it looks. Each is ROOT's algorithm
step for step (`xrdroot.spectrum`): a loop ROOT runs channel by channel is
run here over the whole spectrum at once, but with every channel's sums
added in ROOT's order from ROOT's first term, and a loop that feeds on
itself is a loop over Python floats, which are C doubles - so every number
is ROOT 6.40's to the last bit, checked against it on the tutorials'
spectra and on every option: each filter order, window direction and
smoothing width, Compton edges, boosted repetitions, and searches with and
without background removal and smoothing.

```python
import numpy as np
import xrdroot.pyroot as ROOT

s = ROOT.TSpectrum()
source = np.array([h.GetBinContent(i + 1) for i in range(1024)])
s.Background(source, 1024, 20, ROOT.TSpectrum.kBackDecreasingWindow,
             ROOT.TSpectrum.kBackOrder8, True, ROOT.TSpectrum.kBackSmoothing5, True)
dest = np.zeros(1024)
n = s.SearchHighRes(source, dest, 1024, 8, 2, True, 3, True, 3)
print(s.GetPositionX()[:n])            # the peaks, in channels, highest first
n = s.Search(h, 2, "", 0.10)           # on a histogram: bin centres, a TPolyMarker, drawn
hb = s.Background(h, 20, "same")       # h_background, red, drawn over it
```

**Arrays in, answers in place.** Where ROOT takes a `Double_t *` and its
size and leaves its answer there, so do these: a NumPy array - what a
translated macro's `Double_t source[1024]` is - an `array.array` or a list,
filled in place; a `Double_t **` is a list of rows (a macro's `new
Double_t *[n]`, a vector's `data()`) or a two-dimensional array. The
methods that can refuse hand back ROOT's own message - `"Too Large Clipping
Window"` - and `None` otherwise, as ROOT's `const char *` is `nullptr`;
`SearchHighRes` prints ROOT's `Error in <TSpectrum::SearchHighRes>` and
finds nothing, and warns `Peak buffer full` as ROOT does. `GetPositionX`
hands back the array itself, as ROOT hands back its pointer, so a macro
indexes it; `Search` on a histogram works over its axis range, leaves bin
centres and contents there, hangs a `TPolyMarker` of red triangles on the
histogram - replacing the last - and draws it unless told `goff` or
`nodraw`; `Print` prints ROOT's `Number of positions`.

**ROOT's quirks, kept.** The smoothed eighth-order filter takes one of its
terms with the wrong sign, the two-dimensional search smooths a
non-square plane read from where ROOT's copy - at `2 * ssizex_ext` columns
- leaves it, and the one-dimensional search reads its `H'y` shifted by the
response's length: all as ROOT does, because what ROOT prints is what
these print. The one place they part is where ROOT reads memory it never
wrote: the last `length - 1` channels of a Richardson-Lucy deconvolution,
never solved for, are whatever ROOT's heap held (often the previous
deconvolution's leftovers) and are zero here.

**Fitting peaks.** `TSpectrumFit` and `TSpectrum2Fit` are Morhac's peak
fitters, step for step: peaks of one sigma - in two dimensions correlated
Gaussians with a ridge along x and one along y beside each - with tails,
steps and a linear or quadratic background, fitted by AWMI, the algorithm
without matrix inversion, in which each parameter moves along its own
gradient over its own curvature, or by Stiefel and Hestenes' conjugate
gradients on the normal equations. A script calls them as ROOT's:
`SetFitParameters(xmin, xmax, iterations, alpha, kFitOptimChiCounts,
kFitAlphaHalving, kFitPower2, kFitTaylorOrderFirst)`, `SetPeakParameters`,
then `FitAwmi(source)`, which leaves the fitted spectrum in `source`;
`GetPositions`, `GetAmplitudes` and `GetAreas` (`GetVolumes` in two
dimensions) give what was found, with their errors beside them, and
`GetSigma(sigma, sigmaErr)` fills the variables it is handed. Every sum
over the channels is added channel by channel in ROOT's order, with ROOT's
own approximation to `erfc` and the C library's `exp`, so on ROOT's machine
every chi, value and error is ROOT 6.40's to the last bit - but for a
two-dimensional volume error ROOT builds from memory it never wrote, which
is 0 here. ROOT's quirks are kept, because each changes the numbers, and
what ROOT refuses is refused in ROOT's words.

**Transforms.** `TSpectrumTransform` and `TSpectrum2Transform` are Morhac's
fast orthogonal transforms of a spectrum whose length is a power of two:
Haar, Walsh, cosine, sine, Fourier and Hartley, and the mixed
Fourier-Walsh, Fourier-Haar, Walsh-Haar, cosine-Walsh, cosine-Haar,
sine-Walsh and sine-Haar of any degree the spectrum allows, forward and
back. `FilterZonal` sets the coefficients in a region to one value and
`Enhance` multiplies them, each then transforming back. Each is ROOT's own
arithmetic, stage by stage over ROOT's working space, so every coefficient
is ROOT 6.40's to the last bit, and each object changes itself as ROOT's
does: a one-dimensional cosine or sine transform doubles its size, and a
cosine or sine mixed transform raises its degree, every time it runs; the
two-dimensional filter scales what it returns back to the source's sum,
and writes nothing when that sum is zero. What ROOT would do only by
reading memory it never set is refused in a sentence: a second call
reading past the source, or a degree raised until ROOT would divide by zero.

| What | ROOT's names here |
| --- | --- |
| one dimension | `TSpectrum(maxpositions=100)`: `Background` (of an array, with `kBackIncreasingWindow`/`kBackDecreasingWindow`, `kBackOrder2`...`8`, `kBackSmoothing3`...`15` and Compton edges; or of a histogram, with the option words `BackIncreasingWindow`, `BackOrder4`, `nosmoothing`, `BackSmoothing7`, `Compton`, `same`), `SmoothMarkov`, `Deconvolution`, `DeconvolutionRL`, `Unfolding`, `SearchHighRes`, `Search1HighRes`, `Search` (`nobackground`, `nomarkov`, `nodraw`, `goff`), `StaticSearch`, `StaticBackground`, `GetPositionX`/`Y`, `GetNPeaks`, `SetAverageWindow`, `SetDeconIterations`, `SetResolution`, `Print` |
| two dimensions | `TSpectrum2`: `Background` (`kBackSuccessiveFiltering`, `kBackOneStepFiltering`; of a `TH2` with `BackIncreasingWindow`, `BackOneStepFiltering`, `same`), `SmoothMarkov`, `Deconvolution`, `SearchHighRes`, `Search`, `StaticSearch`, `StaticBackground`, `GetPositionX`/`Y`, `Print` |
| transforms | `TSpectrumTransform(size)` and `TSpectrum2Transform(sizeX, sizeY)`: `SetTransformType(kTransformHaar` ... `kTransformSinHaar, degree)`, `SetDirection(kTransformForward` or `kTransformInverse)`, `SetRegion`, `SetFilterCoeff`, `SetEnhanceCoeff`, `Transform(source, dest)`, `FilterZonal`, `Enhance` |
| fitting peaks | `TSpectrumFit(n)`, `TSpectrum2Fit(n)`: `SetFitParameters`, `SetPeakParameters`, `SetBackgroundParameters`, `SetTailParameters`, `FitAwmi`, `FitStiefel`, `GetPositions`, `GetPositionsErrors` (2-D `GetPositionErrors`), `GetAmplitudes`, `GetAmplitudesErrors` (2-D `GetAmplitudeErrors`), `GetAreas`/`GetAreasErrors` (2-D `GetVolumes`/`GetVolumeErrors`), `GetSigma` (2-D `GetSigmaX`/`GetSigmaY`/`GetRo`), `GetBackgroundParameters`, `GetTailParameters`, `GetChi`, the `kFit…` constants |

## RooFit

`ROOT.RooRealVar`, `ROOT.RooGaussian`, `ROOT.RooFit.Save()` and the rest of
RooFit are here too, over an engine of its own, `xrdroot.roofit`, which has
only NumPy, this library's fitting, histograms, graphs and random numbers
under it. A RooFit script runs unchanged, and what it prints is what ROOT
6.40 prints: the same generated events, the same fit, the same
`RooFitResult`, the same messages in the same order.

```python
import xrdroot.pyroot as ROOT

x = ROOT.RooRealVar("x", "x", -10, 10)
mean = ROOT.RooRealVar("mean", "mean of gaussian", 1, -10, 10)
sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 10)
gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

data = gauss.generate({x}, 10000)          # RooRandom's TRandom3: ROOT's events
r = gauss.fitTo(data, Save=True, PrintLevel=-1)
r.Print()                                  # ROOT's table, to the digit
frame = x.frame(Title="Gaussian pdf")
data.plotOn(frame)
gauss.plotOn(frame, LineColor="r")
gauss.paramOn(frame)
frame.Draw()
```

**Why the numbers are ROOT's.**
- **Densities** are computed as RooFit's batch kernels compute them, and normalised by
  RooFit's own closed forms. Where there is no closed form, the integral is numerical, by
  RooFit's integrators ported step for step: Romberg in one dimension, adaptive cubature in
  more. Each is announced where RooFit announces one.
- **Likelihoods** are summed with RooFit's Kahan sum.
  - A value that cannot be had - a negative density, say - is packed as RooFit packs it, and
    scored as it scores it.
  - The minimiser is Minuit2 through iminuit, set up as RooFit sets it up: its step sizes,
    tolerance, strategy, call limit and error level.
- **Generation** draws from `RooRandom`'s own `TRandom3` in RooFit's order, through the same
  generator contexts:
  - a density's own generator where it has one;
  - TFoam, ported bit for bit, where it has none;
  - accept-reject for a density that is conditional on prototype data.
- **Plots** follow `plotOn`'s option list as RooFit reads it, warnings about repeated options
  included. A curve is sampled adaptively, as `RooCurve` samples one, and data carry
  Poisson intervals.

**What it has.**
- **Variables and functions**: `RooRealVar` (named ranges, and ranges with functions for
  ends), `RooConstVar`, `RooCategory` and its derived kinds, `RooFormulaVar`,
  `RooGenericPdf`, `RooPolyVar`, `RooProduct` and `RooAddition`.
- **Densities**: the standard shapes (Gaussian, exponential, polynomial, Chebychev, ARGUS,
  Crystal Ball, Breit-Wigner, Landau, Poisson, gamma, chi-square, non-central chi-square...).
- **Compositions**: sums (`RooAddPdf`, recursive fractions too), products with conditional
  factors, `RooExtendPdf`, `RooSimultaneous`, `RooRealSumPdf`, `RooHistPdf`,
  `RooMultiVarGaussian`, `RooFFTConvPdf`, `RooKeysPdf`, and the resolution models and B
  decays.
- **Data**: `RooDataSet` and `RooDataHist` (imported from trees, histograms or slices;
  reduced, appended, binned, tabulated; with global observables) and `createHistogram`.
- **Generation**: `generate` and `generateBinned`, prototype data (`ProtoData`) included.
- **Fitting**: `fitTo` and its options, `createNLL`, `RooMinimizer` (MIGRAD, HESSE, MINOS)
  and `RooFitResult`, including `createHessePdf`, `randomizePars` and `correlation`.
  Constraint terms are found in products, or given, and normalised over the global
  observables, as RooFit does.
- **Studies and plots**: `RooMCStudy`; `RooPlot` with components, ranges, projection
  ranges, slices, projections over data, error bands, `chiSquare`, `residHist`, `pullHist`
  and `paramOn`.
- **The workspace**: `RooWorkspace` with its factory language (`SUM`, `PROD`, `EXPR`,
  `SIMUL`...).
- **Messages**: `RooMsgService`.

**What it refuses.** A class or method RooFit has that this engine does not is refused by
name, like the rest of the namespace: `ROOT has RooCustomizer; xrdroot.pyroot does not yet`.
The same goes for, among others:
- `RooNDKeysPdf`, `RooEffProd`, `RooLagrangianMorphFunc`, `RooMultiPdf`, `RooParamHistFunc`,
  and the `RooMCStudy` add-on modules;
- `RooAbsPdf.defaultIntegratorConfig` and the graph printers.

Refusing is better than a curve or a fit that is almost ROOT's. A workspace read from a ROOT
file is read into the same engine, class by class; a class it has no maker for - or one whose
C++ code the file holds for ROOT to compile - is refused by name.

## RooStats

`ROOT.RooStats` is RooStats 6.40 over the same engine: the calculators and intervals of a
limit or a discovery, printing what ROOT prints and drawing from `RooRandom`'s `TRandom3` in
ROOT's order, so toys are ROOT's toys.

```python
import xrdroot.pyroot as ROOT

f = ROOT.TFile.Open("example_combined_GaussExample_model.root")   # hist2workspace's
w = f.Get("combined")
data, sb = w.data("obsData"), w.obj("ModelConfig")
poi = sb.GetParametersOfInterest().first()
b = sb.Clone("B")
poi.setVal(0)
b.SetSnapshot(ROOT.RooArgSet(poi))

calc = ROOT.RooStats.AsymptoticCalculator(data, b, sb)
calc.SetOneSided(True)
inverter = ROOT.RooStats.HypoTestInverter(calc)
inverter.UseCLs(True)
inverter.SetFixedScan(6, 0, 3)
result = inverter.GetInterval()
print(result.UpperLimit(), result.GetExpectedUpperLimit(0))
ROOT.RooStats.HypoTestInverterPlot("scan", "CLs scan", result).Draw("CLb 2CL")
```

**What it has.**
- **Models**: `ModelConfig` - kept in its workspace by name, and read back with it - and the
  helpers of `RooStatsUtils` (`FactorizePdf`, `MakeNuisancePdf`, `StripConstraints`,
  `PValueToSignificance`, ...), `NumberCountingPdfFactory` and `NumberCountingUtils`.
- **Intervals**: `ProfileLikelihoodCalculator` and `LikelihoodInterval` (with Minuit's
  contours and `LikelihoodIntervalPlot`); `FeldmanCousins` and `NeymanConstruction` with
  `ConfidenceBelt`; `BayesianCalculator` (numerical integration and scans, its posterior
  plot); `MCMCCalculator`, `MetropolisHastings`, `MarkovChain`, the proposal functions and
  `ProposalHelper`, `MCMCInterval` and `MCMCIntervalPlot`.
- **Tests**: `ProfileLikelihoodTestStat` and the other test statistics, `ToyMCSampler`,
  `FrequentistCalculator`, `HybridCalculator`, `AsymptoticCalculator` (its Asimov data,
  global observables and expected p-values), `HypoTestResult`, `SamplingDistribution`,
  `SamplingDistPlot` and `HypoTestPlot`.
- **Limits**: `HypoTestInverter` - fixed scans and the automatic search - with
  `HypoTestInverterResult` (interpolated and expected limits, `ExclusionCleanup`) and
  `HypoTestInverterPlot`'s observed curve and expected bands.
- **Inspection**: `ProfileInspector`.

**Why the numbers are ROOT's.** The fits are RooFit's, set up as RooStats sets them up - its
retries included. The one-dimensional minimisation, integration and root finding are
MathCore's and GSL's, ported: Brent's minimiser and root finder, QAGS, and Cephes' normal
quantiles. The asymptotic formulae are RooStats' own, step for step.

**What it refuses**, by name: `BernsteinCorrection`, `SPlot`, `HypoTestInverter`'s rebuilt
limit distributions, the spline interpolation of a scan, keys-based MCMC intervals, and
PROOF.

## HistFactory

`ROOT.RooStats.HistFactory` builds a model from histograms, as `hist2workspace` and
`MakeModelAndMeasurementFast` build it: the same `RooWorkspace`, the same `ModelConfig`, the
same printout, on the RooFit engine - not a translation into another format - so everything
RooStats does with a HistFactory model it does here.

```python
import xrdroot.pyroot as ROOT

meas = ROOT.RooStats.HistFactory.Measurement("meas", "meas")
meas.SetOutputFilePrefix("./results/example")
meas.SetPOI("SigXsecOverSM")
meas.AddConstantParam("Lumi")
meas.SetLumi(1.0)
meas.SetLumiRelErr(0.10)
chan = ROOT.RooStats.HistFactory.Channel("channel1")
chan.SetData("data", "data/example.root")
chan.SetStatErrorConfig(0.05, "Poisson")
signal = ROOT.RooStats.HistFactory.Sample("signal", "signal", "data/example.root")
signal.AddOverallSys("syst1", 0.95, 1.05)
signal.AddNormFactor("SigXsecOverSM", 1, 0, 3)
chan.AddSample(signal)
meas.AddChannel(chan)
meas.CollectHistograms()
w = ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast(meas)
```

**What it has**: `Measurement`, `Channel`, `Sample` and `Data`; overall, shape and
normalisation systematics (`OverallSys`, `HistoSys`, `NormFactor`, `ShapeSys`,
`ShapeFactor`, `StatError`) with Gaussian, Poisson, Gamma and log-normal constraints;
`FlexibleInterpVar`, `PiecewiseInterpolation` and `ParamHistFunc`; Asimov datasets;
`PrintTree`. HistFactory workspaces written by ROOT are read, and the output files hold the
measurement itself (`meas`) as ROOT 6.40 streams it, byte for byte, beside the histograms.

**What it refuses**: the XML configuration (`PrintXML`, `hist2workspace` itself), a shape
factor's initial shape, and writing the workspace into the output file - the histograms and
the measurement are written, and the workspace is said not to be.

## TMVA

`ROOT.TMVA.Factory`, `ROOT.TMVA.DataLoader`, `ROOT.TMVA.Reader` and the rest of TMVA are
here, over an engine of its own, `xrdroot.tmva`, with NumPy under it and scikit-learn,
SciPy, XGBoost and - where it has a wheel - PyTorch for the methods that learn
iteratively (`pip install xrdroot[tmva]`). A TMVA macro runs unchanged: its option
strings are read as TMVA reads them, and it prints TMVA's messages and tables, writes
TMVA's output file - `dataset/InputVariables_*`, `Method_<type>/<title>`, `TestTree` and
`TrainTree` - and TMVA's weight files, `dataset/weights/<job>_<title>.weights.xml`.

```python
import xrdroot.pyroot as ROOT

loader = ROOT.TMVA.DataLoader("dataset")
for name in ("var1", "var2", "var3", "var4"):
    loader.AddVariable(name, "F")
loader.AddSignalTree(signal)
loader.AddBackgroundTree(background)
loader.PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V")

output = ROOT.TFile.Open("TMVAC.root", "RECREATE")
factory = ROOT.TMVA.Factory("TMVAClassification", output, "!V:AnalysisType=Classification")
factory.BookMethod(loader, ROOT.TMVA.Types.kBDT, "BDT", "NTrees=850:MaxDepth=3")
factory.TrainAllMethods(); factory.TestAllMethods(); factory.EvaluateAllMethods()
print(factory.GetROCIntegral(loader, "BDT"))

reader = ROOT.TMVA.Reader("!Color:!Silent")      # and back, one event at a time
```

**ROOT to xrdroot.**
- **The data set** - the split, the renormalisation, the correlation matrices - is TMVA's
  to the digit: the events are drawn from TMVA's `TRandom3(SplitSeed)` and shuffled as
  libc++'s `std::shuffle` shuffles them.
- **The transformations** (`I`, `N`, `D`, `P`, `G`, `U` and chains of them, per class or
  for all) are TMVA's, PCA's eigenvectors by the same Householder reduction and QL iterations
  (JAMA's, as `TMatrixDSymEigen` has them), signs and all.
- **The evaluation** is TMVA's: the efficiencies from its 10000-bin cumulative histograms
  and root finder, the ROC integral from `ROCCurve`, the regression's biases, RMS and
  mutual information, the multiclass 1-vs-rest tables and confusion matrices.
- **The Reader** takes `&var` addresses (cells) from a macro and one-element arrays from
  Python; `EvaluateMVA`, `EvaluateRegression`, `EvaluateMulticlass`, `GetProba`,
  `GetRarity`, `GetMVAError` and `FindMVA` are TMVA's. It reads every weight file the
  Factory here writes, and TMVA's own for the methods listed below as read.
- **CrossValidation** splits folds by `SplitExpr` or TMVA's own shuffled draw, trains a
  method per fold and a `CrossValidation` method over them; the Envelope
  `TMVA::Experimental::Classification` is here too.
- **`TMVA::Experimental`**: `RTensor`, `AsTensor`, `RReader`, `Compute`,
  `RStandardScaler`, `RBDT` and `SaveXGBoost`; and `ROOT.Experimental.ML.RDataLoader`.
- **The genetic algorithm** (`GeneticFitter`, `IFitterTarget`, `Interval`) is TMVA's, draw
  for draw.

**The methods, and what trains them.** Where the method is TMVA's closed form or TMVA's
own sampler, it is ported, and its outputs are TMVA's to single precision; where TMVA
trains iteratively, a library does, with TMVA's options mapped onto it.

| Method | Trained by | Against TMVA 6.40 (the tutorials' samples) |
|---|---|---|
| `LD`, `Fisher` (and `Mahalanobis`) | TMVA's sums and inversions | coefficients and outputs exact |
| `Likelihood` (`Spline0`-`Spline2` PDFs, `TransformOutput`) | TMVA's PDFs, smoothing and interpolation | outputs exact, ranking too |
| `Cuts` (`FitMethod=MC`, `GA`; `FMax`/`FMin`/`FSmart`) | TMVA's Monte Carlo and genetic samplers | cuts and efficiencies exact |
| `FDA` (`FitMethod=MC`, `GA`) | TMVA's samplers over the user's formula | MC exact; GA see below |
| `PDERS` (every kernel; `Unscaled`, `MinMax`, `RMS`, `Adaptive`) | TMVA's adaptive box, in single precision | outputs exact, search tree written node for node |
| `PDEFoam` (one foam, two, multiclass, regression) | TMVA's foam, draw for draw | outputs exact |
| `KNN` | brute-force neighbours, TMVA's scaling and kernels | outputs exact |
| `SVM` | scikit-learn `SVC`/`SVR`, TMVA's per-event costs | ROC integral exact to three places |
| `BDT` (`AdaBoost`, `RealAdaBoost`, `Grad`, `Bagging`, `AdaBoostR2`; regression, multiclass) | scikit-learn trees, TMVA's boosting and bagging | ROC 0.888 vs 0.889 |
| `MLP` (`BFGS`, `BP`; Bayesian regulator) | NumPy back-propagation, SciPy's L-BFGS | ROC 0.921 vs 0.919 |
| `DL`/`DNN` (dense layers) | PyTorch, or NumPy's own Adam/SGD without it | ROC 0.921 vs 0.919 |
| `RuleFit` (`RFTMVA`) | a boosted scikit-learn forest, a gradient-directed path | ROC 0.893 vs 0.875 |
| `Category` | its sub-methods, each over its own data set | as its sub-methods |
| `CrossValidation` | its folds' methods | as they |
| `PyRandomForest`, `PyAdaBoost`, `PyGTB` | scikit-learn, as TMVA's PyMVA wraps it | - |

The mapping onto scikit-learn: a BDT's `MaxDepth`, `MinNodeSize` (as
`min_weight_fraction_leaf`), `SeparationType` (`GiniIndex` and `CrossEntropy`; the other
separations are grown as `GiniIndex`), `UseNvars` and
`UseRandomisedTrees` (`max_features`) grow each tree; TMVA's own code does the boosting -
the event weights, the tree weights, `Shrinkage`, `UseBaggedBoost` with TMVA's
`TRandom3` draws - and the trees are written in TMVA's `<BinaryTree>` XML and read back by
xrdroot's own evaluator, which reads TMVA's BDT weight files too.

**Why some are not TMVA's to the digit.** A decision tree grown by scikit-learn splits
where TMVA's `nCuts` grid does not; a network's weights start from another generator; and
TMVA's `FDA_GA` and `TMVAGAexample` draw from a generator seeded from the clock
(`GeneticAlgorithm`'s store is `TRandom3(0)`), so ROOT does not repeat itself either.
TMVA's ROC integral of an output with many equal values depends on how its C++ library's
`std::sort` orders them; xrdroot keeps the order stable, which can move the third decimal
(PDEFoam's 0.830 against 0.829). After reading a weight file TMVA does not process a
method's options again, so a PDERS read back evaluates with the box kernel, whatever was
booked - xrdroot does the same, since that is what TMVA's test outputs are.

**What it refuses.** A method this engine does not have is refused by name when booked or
read: `HMatrix`, `CFMlpANN`, `TMlpANN`, `BoostedFisher` (TMVA's generalised boosting),
`PyKeras` and `PyTorch` (a user's own model file), and SOFIE. So are the options that need
what is not here: `FitMethod=SA` and `MINUIT` and a MINUIT `Converger`, RuleFit's
`RFFriedman`, the `Spline3`, `Spline5` and `KDE` PDFs, PDEFoam's decision-tree cell splitting and kernels, and the DL layers other
than dense (`CONV`, `MAXPOOL`, `RNN`, `LSTM`, `GRU`, `BNORM`...). PDE-Foam writes its foams
beside the weight file as trees of cells. It also reads TMVA's own `_foams.root`, whose
`PDEFoam` objects link their cells by `TRef`, so an Application given ROOT's weight files
answers as ROOT does. `RStandardScaler.Save` and `SaveXGBoost` write trees that
xrdroot reads but ROOT does not. TMVA's standalone `.class.C` files are not written.

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
  numbers; it gives back an array of the batch's length. A C++ macro's
  lambda, which is written for one entry, is called once per entry, as ROOT
  calls it - a collection reaching it as an `RVec`.
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

### From CSV files and SQLite

`ROOT.RDF.FromCSV` and `ROOT.RDF.FromSqlite` make a frame of a CSV file's
columns or of the rows an SQL query gives, typed as ROOT's `RCsvDS` and
`RSqliteDS` type them: a CSV column by the letter given for it (`O` bool,
`D` double, `L` Long64_t, `T` string) or by the look of its first value,
with ROOT's quoting, its `nan` for an empty field and its warning for an
integer column that has one; an SQLite column by its declared type, or by
its first value when it declares none. Either may be named by an
`http://` or `https://` URL, which is fetched whole first - as may a
`TFile` - and `RDF.MakeLazyDataFrame` makes a frame of results `Take` booked.

```python
import xrdroot.pyroot as ROOT

df = ROOT.RDF.FromCSV("muons.csv")                       # readHeaders, delimiter, ... as ROOT's
options = ROOT.RDF.RCsvDS.ROptions()
options.fColumnTypes = {"Run": "D"}
typed = ROOT.RDF.FromCSV("muons.csv", options)
rows = ROOT.RDF.FromSqlite("stats.sqlite", "SELECT Version FROM accesslog")
```

The one type SQLite's own tools do not say is a column its table declares
with no type: ROOT types it by its first value, and here - the declared
types found through a temporary view of the query - it is a blob.

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

### Trees the way ROOT writes them

`xrdroot.pyroot` is `import ROOT` for a PyROOT script or a translated macro,
and its `TTree` is filled the way ROOT's tutorials fill one: an address bound
to each branch, the value at the address changed, then `Fill`, which reads
every address at that moment.

```python
import numpy as np
import xrdroot
from xrdroot.pyroot import TTree, std

px, n, e = np.zeros(1, "f"), np.zeros(1, "i"), np.zeros(10)
hits = std.vector["float"]()
tree = TTree("t1", "a simple tree")
tree.Branch("px", px, "px/F")
tree.Branch("n", n, "n/I")
tree.Branch("e", e, "e[n]/D")           # counted by n, as ROOT's x[n]/F is
tree.Branch("hits", hits)                # a std::vector: rows of different lengths
for i in range(1000):
    px[0], n[0] = np.random.normal(), i % 10
    e[: n[0]] = np.random.exponential(size=n[0])
    hits.assign(np.random.normal(size=i % 4))
    tree.Fill()
with xrdroot.create("tree1.root") as f:
    tree.SetDirectory(f)
    tree.Write()
```

An address is anything that can be read and changed in place: a NumPy array,
an `array.array`, a `ctypes` number or array, a `std.vector` or `std.string`,
or any object with a `.value` - which is what the macro translator makes of
`Float_t px;`. A Python `float` is refused by name, because a tree that read a
copy would fill every entry with the first one's value. A leaf list is
ROOT's: `x/D`, `x[3]/F`, `x[n]/F` counted by an integer branch declared
before it, and `a/I:b:c/F` of several leaves, which are given a struct - a
NumPy structured array, a `ctypes.Structure`, an array of the leaves one after
another, or an object whose attributes they are named after. A branch given
no leaf list takes its type from its address.

Two things are written differently from ROOT, because this writer lays a
tree out a leaf to a branch and rows the leaf-list way: the leaves of a leaf
list are each a branch of their own (`a`, `b` and `c` above), and a
`std::vector` is written as rows with a counter branch of its own
(`nhits`), which ROOT reads as `hits[nhits]/F`. While the tree is being
filled, `Print`, `GetListOfBranches` and `GetBranch` show it as it was
declared.

Reading is `SetBranchAddress` and `GetEntry`, served a thousand entries at a
time so that the entry-by-entry loop costs a read per branch per thousand;
`tree.px` after `GetEntry`, `for event in tree`, `TTreeReader` with
`TTreeReaderValue["float"]` and `TTreeReaderArray["double"]`, `GetLeaf(...).GetValue()`,
`SetBranchStatus`, `AddFriend` (read entry for entry, or by `BuildIndex`),
`SetEntryList`, `CloneTree`, `CopyTree`, `TChain.Add` with wildcards, and
`Draw`, `Scan`, `Show` and `Print` with ROOT's arguments, return values and
text all work the same on a tree being filled and on one read from a file.
`Print`'s byte counts are this writer's baskets', and a tree not yet written
prints its branches as ROOT's "One basket in memory".

`xrdroot.pyroot.std` has the containers a script hands a tree:
`std.vector["float"]()` - the spelling the translator writes for
`std::vector<float> v;`, the same class as `std.vector("float")` and
`std.vector[np.float32]` - with `push_back`, `size`, `[]` and `data()`, a
view of its NumPy storage; `std.map["std::string", "int"]`, `std.pair` and
`std.string`. `ROOT.RDataFrame` is xrdroot's frame taking these trees, with
`AsNumpy` giving an array per entry of a collection, `ROOT.RDF.FromNumpy`,
`ROOT.RDF.RunGraphs`, and `ROOT.RVec` and `ROOT.VecOps` over NumPy.

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

## Merging and copying

`merge` is ROOT's `hadd`, and `copy` is `rootcp` — or, given a cut,
`TTree::CopyTree`. Neither needs ROOT, and both work on anything this
library opens and writes, local or remote.

```python
import xrdroot

xrdroot.merge("all.root", ["run1.root", "run2.root", "run3.root"])
xrdroot.merge("all.root", paths, compression=505, force=True)    # hadd -f505
xrdroot.copy("in.root", "out.root", ["hists/*", "events"])       # rootcp
xrdroot.copy("in.root", "skim.root", "events", cut="nMuon >= 2",
             columns=["nMuon", "Muon_pt"])                       # TTree::CopyTree
```

Every name in every input is merged with the same name in the others, the
way ROOT's `TFileMerger` does it: directories are walked all the way down,
empty ones kept, and a name only a later file holds is taken up too. The
inputs are opened one at a time, in order, and closed before the next, so a
thousand of them cost one open file and the histograms being added up. What
comes back is a `Merged`: what became of each path, how many tree baskets
went across as they were, were packed again or were left where they were, and
how many entries were read and written the slow way.

| What | Merged as | ROOT's |
| --- | --- | --- |
| `TH1`, `TH2`, `TH3` of every storage | bins, squares of weights, running sums and entries added; the same binning required | `TH1::Merge` |
| `TProfile`, `TProfile2D`, `TProfile3D` | the sums of each bin, its weights and their squares added | `TProfile::Merge` |
| `TEfficiency` | passed added to passed, total to total | `TEfficiency::Merge` |
| `TGraph`, `TGraphErrors`, `TGraphAsymmErrors` | the points of each after the last's, error bars of the first graph's kind | `TGraph::Merge` |
| `TMultiGraph` | the graphs of each after the last's | `TMultiGraph::Merge` |
| `TTree`, `TNtuple` | every input's entries one after another; baskets copied as they are where they can be | `TTree::Merge` with `"fast"` |
| `ROOT::RNTuple` | every input's entries one after another, read and written again | `RNTupleMerger` |
| anything else — `TF1`, `TObjString`, a string, a class of your own | carried over from every input as it was, a cycle each, with a `MergeWarning` saying so once | the pass-through for a class with no `Merge` |

Two histograms binned differently are refused by name rather than added bin
by bin into the wrong bins — ROOT's extendable and labelled axes are not
merged here — and so are two trees whose branches differ, two RNTuples whose
fields do, or one name holding different classes in different files. The
refusal names the object and the file, and a merge that raises leaves no
output behind.

Something ROOT does that is worth knowing: an object ROOT has no `Merge`
for is not "kept from the first file" but written from *every* file that has
it, one cycle each, and reading the name gives the last. This does the same.
A multigraph's graphs keep their data but not the draw option each was added
with, which the reader does not keep.

### The fast way

A tree is mostly its baskets, and a basket does not know which file it is
in. So a merged tree's baskets are the inputs' baskets, read and written
again byte for byte behind new keys, and only its `TTree` and `TBranch`
records are new — with each branch's leaf made again as ROOT made it: the
same leaf class, a counter of the same integer type, the largest count and
the longest string its inputs had. Nothing is decompressed, let alone decoded.

That needs every branch to be one this writer makes the same way: a number,
a fixed run of them, a run counted by another branch, or a string, each a
branch of one leaf. A basket goes across as it was when its input was
compressed as the output is, or with `keep_compression` (`hadd -fk`), which is
what leaving `compression` alone means; otherwise it is decompressed and
compressed again, still without decoding an entry. ROOT writes every basket's
key in its wide form so that a basket's table of entry offsets, which counts
from the start of its key, never moves; a basket is copied in the width its
key had for the same reason, and only one whose key has to grow — a
small-keyed basket landing past 2 GB, or a tree copied under a longer name —
has that table moved along, the basket unpacked for it and packed again.
Baskets ROOT kept inside a branch's record become baskets of their own.

A tree with a branch this writer does not make that way — a `TLeafG`, a
packed `Float16_t`, an STL vector ROOT wrote as a `TBranchElement` — goes the
slow way: its entries read a batch at a time and written through
`WritableTree`, as the nearest thing this writer makes. What neither way can
carry — a split object, a leaf list, an unreadable column — is refused by name.
`fast=False` (`hadd -O`) sends every tree the slow way.

Ten files of a million entries each — a float64, an int32, a bool and a
jagged float32, 178 MB between them — merge in about 3 s the fast way, 6 s
when every basket is packed again for another compression, and 31 s the slow
way; the fast way is the time it takes to copy the bytes.

### `hadd` flags

The command line takes `hadd`'s flags as `hadd` spells them:
`xrdroot merge [flags] TARGET SOURCES...`.

| `hadd` | `xrdroot merge` | `xrdroot.merge(...)` |
| --- | --- | --- |
| `-f` | `-f` | `force=True` |
| `-f505`, `-f[0-509]` | `-f505` | `compression=505` (or `"zstd"`, or `("zstd", 5)`) |
| `-fk`, `-fk505` | `-fk`, `-fk505` | `keep_compression=True` |
| `-ff` | `-ff` | `keep_compression=False`, `compression` left alone |
| `-a` | `-a` | `append=True` |
| `-k` | `-k` | `skip_errors=True` |
| `-O` | `-O` | `fast=False` |
| `-T` | `-T` | `trees=False` |
| `-L FILE -Ltype SkipListed` | the same | `skip_keys=[...]` |
| `-L FILE -Ltype OnlyListed` | the same | `only_keys=[...]` |
| `-v LEVEL` | `-v LEVEL` | — |
| `-j N`, `-n N` | taken and ignored | — |

One default differs: with no `-f` setting `hadd` writes ROOT's 101, where this
keeps the first input's setting and copies baskets as they are — `-f101` asks
for `hadd`'s. `-j` is taken and ignored because the inputs are merged in one
process, one at a time; `-n` because only one input is ever open. With `-a`
what the output already holds is the first input: its trees' baskets are
left where they are and pointed at, and each merged object is a new cycle of
its name. `skip_keys` and `only_keys` take names, paths from the top of the
file, or shell patterns of either; a directory named in `only_keys` is taken
whole.

### Copying

`copy(source, destination, keys)` copies what `keys` names — everything at
the top of the file, directories and all, when `None`; a path or shell
pattern, or a list of them; or a mapping of each path to a new one. An object
goes across as the very record it was, and the destination is made to
describe its classes as the source did, so a class this library has no
layout for reads back exactly as it read before. The destination is added to
if it is there, and a name already in it becomes its next cycle; it can also
be a directory of a file being written, which is left open.

A tree goes across as its baskets. `columns` naming branches keeps the fast
way — only those branches' baskets go, with the counters their runs need —
while `cut`, or `columns` as a mapping of name to expression, reads the
entries through the formula engine and writes only those that pass;
`tree_filter`, given each tree's path, says which trees they apply to.

On the command line, `xrdroot cp SOURCE... DEST` names what to take the way
`rootcp` does, `file.root:path` with a shell pattern allowed, and `DEST` may
be `out.root:directory`; `--cut`, `--columns a,b`, `-c 505` and `--recreate`
are the rest.

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

## The shell and command-line tools

`pip install` puts an `xrdroot` command on the path — `python -m xrdroot` is the
same thing — and it is ROOT's prompt and ROOT's command-line kit at once, over
any URL `open_root` takes: a local path, `root://`, `https://`, `s3://`.

```console
$ xrdroot events.root                   # root -l events.root: a prompt, with _file0
$ xrdroot ls -t root://host//events.root
$ xrdroot dump events.root:dir/h        # every bin, point and entry, as text
$ xrdroot diff before.root after.root   # exit status 0 the same, 1 different
$ xrdroot print events.root:h -o h.png  # or .pdf, .svg
$ xrdroot scan events.root:Events "pt:eta" "pt > 30"
$ xrdroot draw events.root:Events pt -o pt.png
$ xrdroot info events.root              # version, compression, UUID, sizes, classes
```

A thing inside a file is `FILE:path`. A URL has colons of its own, so the
path is split off at the last `.root` that a `:` follows —
`root://host:1094//f.root:dir/h` is the file `root://host:1094//f.root` and
the path `dir/h` — and a file whose name does not end in `.root` names its
path with `-k` instead. A refusal is one line on standard error and exit
status 2.

| ROOT | here |
| --- | --- |
| `root -l f.root` | `xrdroot f.root` — `_file0`, `_file1`… as ROOT names them, and a `.py` among them run as a macro; `-q` leaves after |
| `.ls`, `.pwd`, `.cd dir`, `.q` | the same, at the prompt: `print(gROOT.ls())`, `gROOT.cd("dir")`… |
| `.x macro.C(1, 2)` | `.x macro.py(1, 2)` — or `gROOT.macro("macro.py", 1, 2)` — runs the file, then its function of the same name; a `.C` is [translated from C++](#running-root-macros) first |
| `root -b -q macro.C` | `xrdroot run macro.C` |
| `gROOT`, `gDirectory`, `gFile` | `xrdroot.gROOT`, `xrdroot.gDirectory`; the file is `gROOT.cd()`'s answer, or `_file0` |
| `TFile::Open(url)` | `gROOT.open(url)`, which goes into the file as ROOT's does |
| `gROOT->Get("f.root:/dir/h")`, `FindObject("h")` | `gROOT.get("f.root:/dir/h")`, `gROOT["h"]` |
| `TBrowser` | not provided: `xrdroot ls -t` for what a file holds, `dump` for what is in it, `print` or `.plot()` to see it |
| `rootls -t -l` | `xrdroot ls -t -l` |
| `rootprint`, `root-print` | `xrdroot print` |
| `rootdiff`, `root-diff` | `xrdroot diff`, with `--atol`, `--rtol` and `-k` |
| `root-dump` | `xrdroot dump`, with `-n` entries of each tree |
| `TTree::Scan`, `TTree::Draw` | `xrdroot scan`, `xrdroot draw` |
| `hadd`, `rootcp` | the `merge` and `cp` subcommands, where `xrdroot.merge` is installed |

### The prompt

`xrdroot` with no subcommand is a Python prompt — IPython if it is installed,
the standard library's otherwise — holding everything `from xrdroot import *`
brings, and NumPy as `np`. A line starting in its first column with one of
ROOT's dot-commands is the Python it stands for; anything else is Python, so
`h = _file0["h1d"]` and `.ls` sit side by side. `.help` lists the commands.

```text
$ xrdroot tests/data/dirs-6.14.00.root
>>> .cd dir1/dir11
>>> .ls
TDirectoryFile*		dir11	tests/data/dirs-6.14.00.root:/dir1/dir11
  KEY: TH1F	h1;1	h1
>>> gDirectory["h1"].sum()
5.0
```

In IPython or Jupyter, `%load_ext xrdroot` brings the same: the names, the
dot-commands, `%root_ls [dir]`, `%root_open FILE` (the next `_fileN`), and a
`%%root_macro` cell magic that runs its cell as `.x` runs a macro.

### gROOT

`gROOT` is ROOT's session made an object you can ask for — nothing else in the
library looks at it, so a program that never imports it never has one.

```python
from xrdroot import gROOT, gDirectory

f = gROOT.open("dirs.root")        # held open, and the session goes into it
gROOT.cd("dir1/dir11")             # "..", "/dir2", "other.root:/dir" and a Directory work too
gROOT.pwd()                        # 'dirs.root:/dir1/dir11'
print(gROOT.ls())                  # ROOT's TFile** / KEY: listing
gROOT["h1"]                        # here, then memory, then every open file
gROOT.get("dirs.root:/dir1/dir11/h1")
gROOT.add(Histogram.book("h", (10, 0, 1)))   # TH1::AddDirectory, said out loud
gROOT.files                        # every file open in the process, open_root's too
gROOT.close_all()                  # the files gROOT opened; open_root's are their opener's
```

A bare name is looked for where ROOT's `FindObject` looks: the current
directory, then the objects `add` put in memory, then each open file in the
order it was opened. `gDirectory` is whichever directory the session is in
when it is used — the session itself at the top — rather than the one it was
in when it was imported.

### The subcommands

`ls` is a line per key — class, name, title, cycle — walking directories all
the way down; `-t` lists every tree's and RNTuple's columns with their types
and entries, `-l` adds each record's bytes on disk and uncompressed, the ratio,
its date and each column's baskets. `dump` writes every key out: a tree an
entry and a column at a time, `[001][pt]: 42.5` as go-hep's `root-dump` does, a
histogram every bin with its edges and error, a graph every point with its
bars, a function its formula and parameters, anything else as it reads.
`diff` compares names, classes and values — a histogram's edges, contents,
errors and entries, a graph's points and bars, a tree column by column a batch
at a time — and says the first difference in each; numbers are the same
within `--atol` and `--rtol`, exactly unless told. `print` draws through
`.plot()` and saves the figure in the format the file name ends in, a whole
directory at once into `out_<path>.png` files, or prints the `.text()`
picture without `-o`; a `TCanvas` is drawn by `xrdroot.canvas` where that is
installed and refused by name where it is not. `info` is the header: the ROOT
release, the seek width, the file's size, its compression in words, its UUID,
where its free segments and class descriptions are, and every class it
describes.

Every subcommand is a module `xrdroot.cli.<name>` with an `add_parser(subparsers)`
and a `run(args)`, named in the list `xrdroot.cli.COMMANDS`; adding one is
writing the module and adding its name to that list.

## Running ROOT macros

A ROOT macro is C++ as Cling reads it. `xrdroot.cint` translates it into
Python and runs that: `xrdroot run` is `root -b -q`, and `.x` at the prompt
takes a `.C` as readily as a `.py`.

```console
$ xrdroot run hsimple.C
$ xrdroot run 'fit.C(1000, "gaus")'       # .x fit.C(1000, "gaus"): the function fit, called
$ xrdroot run hsimple.C+                  # ACLiC's + runs the same way: nothing to compile
$ xrdroot run hsimple.C --python          # the translation, printed; nothing run
$ xrdroot run script.py                   # a PyROOT script, `import ROOT` being xrdroot.pyroot
```

What the macro's function returns is printed after its output as Cling
prints it - `(int) 3`, `(TCanvas *) 0x7f...`, `(double) 2.7000000` - and is
the exit status, as `root -q` makes it: a value from 0 to 255 as it is,
anything else, a pointer included, as 255. So `hsimple.C`, which returns
its `TFile *`, exits 255, as ROOT's CI expects of it.

```python
from xrdroot import cint

cint.run("hsimple.C")                      # the function named like the file, called
cint.run("fit.C", args=(1000,))
python = cint.translate(open("fit.C").read(), "fit.C")   # readable Python, to keep
cint.process_line('printf("%d\\n", 7 / 2)')              # gROOT->ProcessLine
cint.load("helpers.C")                     # .L: what it defines, nothing called
gROOT.macro("hsimple.C")                   # .x at the prompt
```

A macro runs as ROOT runs it: the function named like the file is called
with the arguments `.x` gave, an unnamed macro — a file that is a single
`{ ... }` block — runs its block, and a file with neither just defines what
it declares. Translations are kept under `$XRDROOT_CINT_CACHE` (by default
`~/.cache/xrdroot/cint`), filed by a hash of the macro, of every local header
it reads, and of the translator itself, so nothing stale is ever run.

### What the Python looks like

```cpp
void count(int n = 10) {
   TH1F *h = new TH1F("h", "h", 10, 0, 1);
   Float_t px, py;
   for (int i = 0; i < n; i++) {
      gRandom->Rannor(px, py);
      h->Fill(px);
   }
   printf("%d entries, %d halves\n", (int)h->GetEntries(), n / 2);
}
```

```python
from xrdroot.cint.runtime import *  # noqa: F403


def count(n=10):
    h = ROOT.TH1F('h', 'h', 10, 0, 1)
    px = Cell(0.0, 'float')
    py = Cell(0.0, 'float')
    for i in range(0, n):
        ROOT.gRandom.Rannor(px, py)
        h.Fill(px.value)
    printf('%d entries, %d halves\n', int(h.GetEntries()), idiv(n, 2))
```

Every name the macro did not declare is ROOT's, spelled `ROOT.<name>` and
looked up in `xrdroot.pyroot` when first used (`ROOT.bind(namespace)` points it
anywhere else); `std::vector<double>` is `ROOT.std.vector['double']`. The
runtime supplies what C++ has and Python lacks:

| C++ | Python |
| --- | --- |
| `7 / 2`, `-7 % 2` of integers | `idiv(7, 2)`, `imod(-7, 2)` — truncating toward zero; `div`/`mod` where the types are not known |
| `int n = x;`, `float f = x;`, `unsigned u = -1;` | `int(x)`, `f32(x)`, `u32(-1)`: C's conversion on every store |
| `&x`, a variable handed to `double&` or to ROOT's `Rannor(px, py)` | a `Cell`, read and written as `.value` — ROOT's address contract |
| `&a[i]`, pointer arithmetic on a number array | a NumPy view `a[i:]` |
| `double a[3] = {1}`, `TH1F *h[4]` | `array('double', 3, [1])` (NumPy, of the declared width), a list of `None` |
| `printf`, `sprintf(buf, ...)`, `Printf`, `Form`, `Info` | the C formatting, exactly: `%u`, `%x` of negatives, `%ld`, `%c`, `%*d` |
| `std::cout << x << std::setw(6) << std::fixed` | `cout << x << setw(6) << fixed`: iostreams' six significant digits, flags and widths |
| `std::string`, `const char*`, `char buf[64]` | `str`, with `.size()`, `.substr()`, `.find()` and `strlen`, `strcmp`, `atof` rewritten |
| `p == 0`, `if (!h)` for a pointer | `p is None`, `if h is None` |
| `std::sort(v.begin(), v.end(), less)`, `std::swap(a, b)` | `sort_range(v, 0, None, less)`, `a, b = b, a` |
| overloaded functions and constructors | `f__1`, `f__2` and an `Overloaded` choosing by count, then type |
| classes, inheritance (from ROOT's too), operators | Python classes; `operator+` is `__add__`, `~T()` is `_destruct`, called by `delete` |
| `switch`, `do`/`while`, `for`, `continue` | `if`/`elif` chains (a one-pass loop when a case falls through), `range` when the bound cannot change |
| lambdas, `[x]` and `[&]` | a `def` before the statement; by-value captures bound as defaults, by-reference ones `nonlocal` |

An error as the macro runs is reported at its C++ line — `hsimple.C:42:
ZeroDivisionError: integer division by zero` — through the translation's
source map.

### What is refused

Anything the translator cannot turn into Python that does the same thing it
refuses, by name and at the C++ line, before running any of it:
`tutorials/foo.C:42: pointer arithmetic on a char* is not something this
translator turns into Python`. Among them: `goto`, placement `new` anywhere
but an element of an array (a `TClonesArray`'s slot), variadic templates,
value parameters of class templates, specialisations of class templates,
bit-fields, user-defined literals other than the library's (ROOT 7's
`0.1_normal`, `20_px`, `80_user` and `std::chrono`'s `100us`), assigning one
character of a `std::string` in place, and inline assembly. A program built
against Qt or SYCL - what `#include <QWidget>` or `<sycl/sycl.hpp>` says a
file is - is refused as a program, not a macro.

Some C++ has a Python that does the same only with the runtime's help:

| C++ | Python |
| --- | --- |
| `m(i, j) = v`, `p.X() = v`, `TMatrixDColumn(A, 0) = 1` | `assign_call`, `assign_method`, `assign_into`: an object's `__setcall__`, its `Set<Name>`, or a copy into it |
| `*p = v` of a pointer to an object, or of a type not known | `store_through(p, v)`: into the cell or the object |
| `static int n = f(x);` in a function, method or lambda | a module-level `Static`, initialised the first time its line is reached |
| a local whose destructor does something | the rest of its block in a `try`, whose `finally` runs `_destruct`s in reverse order |
| `template <unsigned N> f(T (&a)[N])` | `N` a keyword argument, `f<3>(a)` giving it, `len(a)` otherwise |
| `while (std::getline(in, s))`, `while (in >> a >> b)` | each value stored, then the stream tested: `stream_after(in, ...)` |
| `new (clones[i]) T(args)` | `construct_at(clones, i, T(args))` - ROOT's `AddAt` |
| `ostream &operator<<(ostream &, const T &)` | a function `cout << t` calls where `t` is a `T`, registered with the runtime for the macro's classes |
| a class's `operator++()`, `operator++(int)`, `operator*` | `_preinc`, `_postinc`, `_deref`, which `++it`, `it++`, `*it` and a range-for call |
| `std::thread`, `std::mutex`, `std::lock_guard`, `std::atomic`, `std::condition_variable` | the runtime's, over Python's `threading`; a guard gives its mutex back as its scope ends |
| `std::chrono::milliseconds(20)`, `high_resolution_clock::now()`, `duration_cast<T>(d)` | `chrono.milliseconds(20)` and the rest: durations that `count()` in their ticks |

`python tools/cint_survey.py ROOT/tutorials` translates every tutorial macro
and counts what translates, what is refused and why, and what (never, one
hopes) crashes; `--min-percent 98` makes it a ratchet that exits 1 below
that, or on any crash. Against ROOT 6.40.04's 910 macros it translates 900,
98.9%, into Python that compiles. Of the ten left, seven are the Qt and SYCL
programs; the other three are a class template with a value parameter
(`df018_customActions.C`), R's `ROOTR_EXPOSED_CLASS` macro, which only
ROOT-R defines (`math/r/Functor.C`), and a string written one character at a
time (`view3ds.C`).

## Compression

Every algorithm ROOT writes with is read here — and written: zlib, lzma, LZ4,
zstd, and the bare-deflate blocks ROOT wrote before 2005 (read only). LZ4 is
decoded and encoded in Python, checksum and all, when nothing else is there,
and by the C codec of the `lz4` extra when it is — about sixty times faster,
with the checksum then checked on the way in, so a damaged block is refused
rather than decoded. zstd uses Python 3.14's own `compression.zstd` where there
is one and the `zstandard` package otherwise, and is the one case where a file
may need something installed.

## Leaf lists with arrays in them

A branch made from a leaf list — `"n/I:px[n]/F:py[n]/F:q2/F"` — writes each
entry's leaves one after another, so where `py` and `q2` are depends on
`n`. Each leaf of such a branch reads the way ROOT reads it, by walking the
entry from its front with the counters it passes; only a counter that comes
after the array it counts, or a string among the leaves, is refused.

## Split collections, and baskets in another file

A `TClonesArray`, or a `std::vector` of a class, split into members is a
branch per member under one branch that holds how many objects each entry
has. Each member reads as a row per entry, one value per object — an array
member as the run of each object's values in turn, and a string, pointer or
object as a list of one per object — and the branch they hang from reads as
a dictionary per entry of those rows, as any split object does:

```python
>>> esd = xrdroot.open_root("alice_ESDs.root")["esdTree"]
>>> esd["Tracks.fITSncls"].array(0, 1)[0][:5]
array([5, 6, 6, 6, 6], dtype=int32)
```

A branch ROOT was told to write to a file of its own — `TBranch::SetFile`,
which ALICE used for its `ESDfriend` — records that file's name, and its
baskets are read from it, found beside the tree's own file the way a
friend's is.

## Old files

A tree written by ROOT 4 opens like any other. Those files count entries in
doubles and keep their seek points in 32-bit integers, and one small enough
never to have been flushed holds its baskets inside the branch record rather
than out in the file — all of which is read here, so a decade-old Geant4 run
needs no copying forward first. So is the last basket of a tree saved while
it was still being filled, which the branch record keeps after the ones
written out.

Older trees open too, back to the ROOT 2.24 of the H1 files ROOT's
`h1analysis` tutorial reads. A `TTree` of versions 6 to 15 — ROOT 3.02 to
5.08, whose fields changed from release to release — is read the way ROOT
reads it, member by member from the description of `TTree` the file itself
carries; one older than that is read the way `TTree::Streamer` still reads
it by hand, as is the `TBranch` of version 5 under it and the arrays ROOT 2
wrote without a byte count. Only a `TBranch` older than version 5, which kept
no sizes for its baskets, is refused by name.

## Pictures

A `TASImage` — the picture a canvas is saved as, or an image read into ROOT
— streams itself as its name and the PNG it would have saved, and it reads
as an `xrdroot.Image`:

```python
>>> image = xrdroot.open_root("gallery.root")["hsimple.png."]
>>> image.width, image.height, image.array.shape
(696, 472, (472, 696, 4))
>>> image.save("hsimple.png")      # the stored bytes, as they are
>>> image.save("hsimple.jpg")      # the pixels, through matplotlib
```

`.png` is the stored PNG untouched and `.array` is its pixels as RGBA
`uint8`, decoded with nothing but `zlib` and NumPy.

An image ROOT made from numbers — the `galaxy_image` tutorial's NGC 4254 —
is kept as those numbers and the palette that colours them. It reads with
`.values`, the grid top row first, and `.palette`, the stops and the 16-bit
levels at each; `.array` colours it the way libAfterImage does, and is the
same array as ROOT's own `GetArgbArray()` of that image, pixel for pixel.
Its `.png` is those pixels encoded, since the file holds none.

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
- a container of numbers in each object of a split `TClonesArray` or vector
  of a class: the entry holds one object's after another's with nothing to
  say where each ends (numbers, strings, pointers and objects there are read);
- a `TBranch` older than version 5, which kept no
  sizes for its baskets.

A class the file describes as having no members at all — `TLimit` is one —
reads as the empty `dict` it honestly is, rather than being refused.

## Errors

`xrdroot.ROOTError` is an `XRootDError`, so it is an `OSError` like
everything else this library raises.

| Exception | Means |
| --- | --- |
| `FormatError` | the bytes are not the ROOT format they claim — a truncated download and an HTML error page both look like this |
| `UnsupportedFeatureError` | a valid file using something this reader does not do, named rather than guessed at |
