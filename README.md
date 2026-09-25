# xrdroot

The ROOT file format, in pure Python, read and written over any URL
[xrdclient](https://github.com/rob-c/xrdclient) can open.

```python
import xrdroot

with xrdroot.open_root("root://eos.example.org//store/events.root") as f:
    tree = f["Events"]
    for batch in tree.iterate(["pt", "eta"], step=10_000):
        analyse(batch)
```

No ROOT and no C++ — the format itself, decoded into NumPy. Every column comes
back as an array, every histogram speaks the plotting protocol `hist` and
`mplhep` share, and a tree is one keyword away from being a pandas, Awkward,
Arrow or Polars table. Nothing is downloaded either: a tree is read a basket at
a time through `xrdclient`, so a hundred-gigabyte file on the other side of the
world is walked from a laptop and costs the entries you asked for rather than
the file.

```python
frame = tree.arrays(["pt", "eta"], library="pd")    # or "ak", "pa", "pl"
spectrum = f["h_pt"].to_hist()                       # a hist.Hist, flow and all
```

## Install

    pip install git+https://github.com/rob-c/xrdroot

That brings `xrdclient` with it, which is where `root://`, `https://`,
HEP WebDAV and `s3://` come from, and NumPy, which is what everything is read
into. Nothing else is required. `matplotlib` makes histograms and graphs draw
themselves onto axes; the `lz4` extra (`lz4` and `xxhash`) makes LZ4 some sixty
times faster than the pure-Python codec that is always there, and `zstandard`
reads zstd before Python 3.14. pandas, Awkward, pyarrow, Polars and `hist` are
used when asked for and never required.

## What it reads

Keys, directories, trees, baskets and all four of ROOT's compression
algorithms. Split C++ classes — member by member, or the whole object per
entry as a dictionary — and unsplit ones walked straight out of the file's own
layout. STL containers, `std::map`, both kinds of string, `TClonesArray`
however it was told to write itself, `vector<pair>`, and the packed
`Double32_t`/`Float16_t` floats.

Beside the tree, the objects ROOT's own kit writes: a `TH1`, `TH2` or `TH3`
comes back as a `Histogram` with its bins, edges and errors where you would
look for them, and a graph — layered error bars and all — as a `Graph` you can
walk a point at a time, inside another object as well as in a key. Profiles
are a `Profile` of means, a `TEfficiency` an `Efficiency` with ROOT's
confidence intervals worked out, a `THnSparse` a `SparseHistogram`, and a
`TMultiGraph` or `THStack` the sequence of what it holds.

Many files of one tree read as one with `xrdroot.chain`, a tree reads its
friends — the ones ROOT recorded, or any added with `add_friend` — beside its
own columns, and a `TEntryList` picks out the entries to read, costing only
the baskets they are in.

RNTuple, ROOT 7's columnar successor to the tree, reads the same way — every
column encoding of its specification, records, collections, variants and all —
and a table or a field-by-field declaration writes as one too.

The few columns this reader will not decode are refused by name with the
reason, because a plausible misreading of physics data is worse than a
refusal.

## What it writes

`xrdroot.create` makes a new ROOT file anywhere the client can put bytes:
trees, histograms of one, two and three dimensions in every storage ROOT has,
profiles, efficiencies, graphs, strings and arrays of numbers, under every
compression ROOT itself writes. Records go out as they
are made, so a file far larger than memory is written in the memory of a
basket per column.

```python
import xrdroot

with xrdroot.create("root://eos.example.org//store/out.root") as f:
    f["Events"] = {"pt": pt, "eta": eta, "p4": p4}    # arrays, or any DataFrame
    f["h_pt"] = hist.Hist(...)                          # or numpy.histogram(...)
```

A dict of arrays, a pandas or Polars DataFrame or an Arrow table becomes a
tree, packed in C a column at a time; `f.tree(...)` and `fill` write one entry
at a time where that is the natural shape of the loop.

A name with a `/` in it — `f["runs/4711/h_pt"]` — goes into ROOT directories,
made on the way; a file past 2 GB takes ROOT's wide layout as ROOT does; and
`xrdroot.update` opens a file that is already there, ROOT's or anyone's, to
add to it, leaving it byte for byte as it was if the `with` block fails.

## Histograms to fill

A histogram is also something to fill and compute with, the way ROOT's `TH1`
is, with ROOT's bookkeeping to the last bit:

```python
h = xrdroot.Histogram.book("pt", (100, 0.0, 200.0))    # TH1D("pt", "", 100, 0, 200)
h.fill(pt, weight=w)                                   # arrays, not one at a time
h.mean(), h.std(), h.integral(), h.rebin(4), h / other
```

Profiles and efficiencies book and fill the same way, and the arithmetic,
rebinning and projections are ROOT's, errors and all. So are the tests that
compare two histograms, and a histogram indexes the way `hist` does:

```python
data.chi2_test(mc, "UW"), data.kolmogorov_test(mc)     # Chi2Test, KolmogorovTest
h[xrdroot.loc(10.0):xrdroot.loc(50.0)], h2[:, sum]    # UHI slicing, ROOT's bookkeeping
h.cumulative(), h.quantiles([0.5]), h.smooth()         # GetCumulative, GetQuantiles, Smooth
```

The functions fits are made with are ROOT's too: a `TF1` read from a file,
or hung on a histogram by a fit, is a `Function` in ROOT's formula language,
evaluated over whole arrays, differentiated in its parameters and written
back:

```python
f = xrdroot.Function("peak", "gaus(0) + pol1(3)", range=(0, 10))
f(xs), f.gradient(xs), f.integral(0, 10)            # TF1::Eval, GradientPar, Integral
h.attach(f); out["h"] = h                             # written with its fit
```

## Analysis

`RDataFrame` is ROOT's declarative analysis — ROOT's methods, ROOT's C++
expressions with `ROOT::VecOps` — evaluated lazily, in one pass, over whole
batches of entries with NumPy, and across worker processes with results that
are the same to the last bit:

```python
df = xrdroot.RDataFrame("Events", "run*.root")
pair = df.Filter("nMuon == 2", "two muons").Filter("Muon_charge[0] != Muon_charge[1]")
m = pair.Define("m", "InvariantMass(Muon_pt, Muon_eta, Muon_phi, Muon_mass)")
h = m.Histo1D(("m", "dimuon mass", 300, 0.25, 300), "m")
h.GetValue().plot(); print(df.Report().GetValue())
```

## Drawing

Histograms and graphs draw themselves: `.plot()` onto matplotlib axes when
matplotlib is there, `.text()` into plain characters when it is not.

## Where this sits

    xrdclient    the XRootD protocol, files, copies, authentication
      └─ xrdroot        the ROOT file format                    (this package)
           └─ xrdml     trees to tensors, a URL to a training loop
                └─ xrddatasets   open data converted to ROOT, and the site that serves it

## Tests

    pip install -e ".[dev]"
    pytest -q

The format tests read files written by ROOT itself — the fixtures under
`tests/data` come from [go-hep](https://github.com/go-hep/hep), under the
licence kept beside them — because a reader checked only against its own
writer proves nothing.

## Licence

LGPL-3.0-or-later. See [COPYING](COPYING) and [LICENSE](LICENSE).
