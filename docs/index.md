# xrdroot

The ROOT file format in pure Python, read and written over any URL
[xrdclient](https://github.com/rob-c/xrdclient) can open.

```python
import xrdroot

with xrdroot.open_root("root://eos.example.org//store/events.root") as f:
    tree = f["Events"]
    for batch in tree.iterate(["pt", "eta"], step=10_000):
        analyse(batch)
```

No ROOT and no C++: the format itself, decoded into NumPy and handed on to
pandas, Awkward, Arrow, Polars and `hist` when asked. Nothing is downloaded: a
tree is read a basket at a time over the wire.

See [ROOT files](root.md) for the whole surface — reading, writing, histograms,
graphs and what the reader refuses — and [Maintainability
metrics](maintainability.md) for how this package is kept honest.

## The stack

| Package | What it is |
| --- | --- |
| [`xrdclient`](https://github.com/rob-c/xrdclient) | the XRootD protocol, files, copies, authentication |
| `xrdroot` | this package: the ROOT file format |
| [`xrdml`](https://github.com/rob-c/xrdml) | trees to tensors, a URL to a training loop |
| [`xrddatasets`](https://github.com/rob-c/xrddatasets) | open data converted to ROOT, and the site that serves it |
