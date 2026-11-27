---
title: "Two objects reach the writer, so a payload must serve both routes"
kind: "instruction"
summary: "A class arrives as either the pyroot wrapper or its engine, so a payload must build one row for both and cast numpy scalars to int/float or struct packing fails; cover each route with a test."
tags: [writer, pyroot, graph2d, coverage, io]
created: "2026-11-27T12:35:57.930Z"
updated: "2026-11-27T12:35:57.930Z"
---
Two routes reach the writer for the same class — the pyroot wrapper and the engine behind it (`unwrap(obj)`) — and a payload must serve both. For `TGraph2D` that means one row shape: the engine's own `members` are handed straight to `_object_payload`, while the wrapper answers through accessors. Casting every value to a plain `int` or `float` is not tidiness: numpy scalars reaching the record's `struct` packing raise `struct.error: required argument is not an integer`.

```python
# src/xrdroot/writer.py, _graph2d_payload
members = getattr(value, "members", None)
if members is not None:
    return _object_payload(value)          # the engine: it already has its members
count = int(value.GetN())
row = {
    "fLineColor": int(value.GetLineColor()), ...        # the attribute bases
    "fNpoints": count, "fNpx": int(value.GetNpx()), ...
    "fX": [float(item) for item in value.GetX()], ...   # coordinates, not the array
}
```

## Two API facts found on the way

1. pyroot's `TGraph2D` has **no `GetMaxIter`** — the member falls back to ROOT's default 100000 via `getattr(..., lambda: 100000)()`.
2. `GetX()` takes **no index**: it answers the whole array, so a per-point `GetX(i)` loop is the wrong shape.

```mermaid
flowchart LR
    A["f['g'] = TGraph2D()"] --> B["recorded(obj) -> engine Graph2D"]
    B --> C{"has members?"}
    C -->|yes| D["_object_payload(value)"]
    A2["direct wrapper"] --> C
    C -->|no| E["row from GetN/GetX/GetY/GetZ + attributes"]
    E --> F["cast every value to int/float"]
    D --> G["_record -> file; io_matrix PASS both ways"]
    F --> G
```

## Why the coverage gate forced the second test

One route tested leaves the other's lines uncovered, so the writer sat at 99.75 %. `tests/test_writer_tarray.py` now has a test per route, which is what took the gate back to 100.00 % — and those two tests are also the only guard that both objects keep producing the same record.
