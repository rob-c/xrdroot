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
walk a point at a time, inside another object as well as in a key.

RNTuple, ROOT 7's columnar successor to the tree, reads the same way — every
column encoding of its specification, records, collections, variants and all —
and a table or a field-by-field declaration writes as one too.

The few columns this reader will not decode are refused by name with the
reason, because a plausible misreading of physics data is worse than a
refusal.

## What it writes

`xrdroot.create` makes a new ROOT file anywhere the client can put bytes:
trees, histograms of one, two and three dimensions, graphs, strings and arrays
of numbers, under every compression ROOT itself writes. Records go out as they
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
