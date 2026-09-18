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

No ROOT, no `uproot`, no `numpy`, no compiled extension — the format itself,
read from the standard library. Nothing is downloaded either: a tree is read a
basket at a time through `xrdclient`, so a hundred-gigabyte file on the other side of
the world is walked from a laptop and costs the entries you asked for rather
than the file.

## Install

    pip install git+https://github.com/rob-c/xrdroot

That brings `xrdclient` with it, which is where `root://`, `https://`,
HEP WebDAV and `s3://` come from. Nothing else is required. `matplotlib` makes
histograms and graphs draw themselves onto axes; `lz4` and `zstandard` make
those two compression algorithms faster than the pure-Python fallbacks that
are always there.

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

The few columns this reader will not decode are refused by name with the
reason, because a plausible misreading of physics data is worse than a
refusal.

## What it writes

`xrdroot.create` makes a new ROOT file anywhere the client can put bytes:
trees filled entry by entry and flushed a basket at a time, histograms, graphs,
strings and arrays of numbers, under every compression ROOT itself writes —
still from nothing but the standard library.

```python
import xrdroot

with xrdroot.create("root://eos.example.org//store/out.root") as f:
    tree = f.tree("Events", {"pt": "f4", "eta": "f4"})
    for pt, eta in rows:
        tree.fill(pt=pt, eta=eta)
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
