# Files and data frames

## io

I/O owns ROOT file reading and writing, directories and keys, streamers, tree
branches and baskets, RNTuple, merging, remote access and file-backed drawing.
The main modules include `file`, `tree`, `streamers`, `interp`, `writer`,
`wtree`, `wupdate`, `rntuple`, `merging`, `drawable` and `treedraw`.
Its direct component dependencies are core and hist.

Internal callers import the engine that owns an operation:

```python
from xrdroot.file import open_root
from xrdroot.writer import create
from xrdroot.wupdate import update
```

The public `xrdroot.open_root`, `create` and `update` facade names continue to
resolve to these operations. Engine modules avoid importing that app-owned
facade to reach one another. `library` and `Jagged` are core primitives even
though the tree reader still exposes compatible public names.

Decoding a canvas, RooWorkspace or HistFactory object needs a higher engine.
These operations use [declared providers](providers.md). Unknown classes retain
their diagnostics; file-specific streamer layouts still take precedence.
A cold read of unrelated data must not load a renderer just to discover that
an unknown name is absent.

Public mutable registries keep their lifecycle contracts. Explicit access to
`kinds.CLASSES` materializes the actual shared dictionary and then respects its
replacements and deletions. Canvas streamer overrides and custom additions
remain visible. Already-loaded-provider lookup preserves these mutations
without importing graphics for arbitrary unknown names.

Remote bytes still use the pinned project transport. File formats, comparison
policy, timeout behavior and written-file semantics are not changed to satisfy
a component boundary.

## rdf

RDF owns engine data-frame execution, expressions, column evaluation and batch
kernels under `xrdroot.rdf`. It depends directly on core, hist and I/O. The ROOT
`RDataFrame` namespace, translated-call adapters and RVec wrappers belong to
`pyroot`.

The split retains worker caps, vector copies, nested value semantics and
assignment behavior. Native row and mask fast paths use exact implementation
provenance, ordinary NumPy storage and unmodified methods. A callback, subclass,
custom dictionary or altered constructor must keep its scalar behavior and
observable call order. Moving a definition requires updating its trusted module
name without weakening the remaining checks.

RDF integration tests can legitimately cover I/O and ROOT adapters together;
those tests do not establish isolated engine imports. Use the separate
[boundary commands](boundaries.md) for that scope.
