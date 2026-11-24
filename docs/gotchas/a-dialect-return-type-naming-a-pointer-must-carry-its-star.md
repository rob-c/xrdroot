---
title: "A dialect return type that names a pointer must carry its star"
kind: "gotcha"
summary: "`CType(\"Double_t*\")` built from a string has pointer=0, so a range over a graph's own arrays is refused; parse the `*` in `_named_type`."
tags: [translator, dialect, pointer, cppstd, tgraph, is_sorted]
created: "2026-11-24T01:00:56.992Z"
updated: "2026-11-24T01:00:56.992Z"
---
A dialect's `method_returns` entry is read as a *declaration*, not a class name: `"Double_t*"` means a pointer. Without the `*` being parsed, `CType("Double_t*")` has `pointer=0`, so a graph's own arrays are not ranges and `std::is_sorted(graph.GetX(), graph.GetX() + n)` is refused with "an iterator this translator cannot follow back to its container".

`pycxx/infer.py::_class_method_type` used to build the type with `CType(kept)` — the string constructor sets `name` and nothing else. `_named_type` now strips a trailing `*` and sets `pointer=1`, which is what `pycint/dialect.py`'s `RETURNED` table needed for the whole graph family:

```python
**{(kind, name): "Double_t*"
   for kind in ("TGraph", "TGraphErrors", "TGraphAsymmErrors", "TGraphBentErrors",
                "TGraphMultiErrors")
   for name in ("GetX", "GetY", "GetEX", "GetEY", "GetEXlow", "GetEXhigh", "GetEYlow",
                "GetEYhigh", "GetEXlowd", "GetEXhighd", "GetEYlowd", "GetEYhighd")},
```

## The shape of the trap

```mermaid
graph LR
  A[ROOT has std.is_sorted] --> B[translator: the range]
  B --> C[typeof graph.GetX<br/>= CType pointer=0]
  C --> D[refused: an iterator this<br/>translator cannot follow]
  D --> E[_named_type parses the *]
  E --> F[sorted_range(graph.GetX(), 0, n)]
```

Three walls stood between the message and the fix, and only the first was the message: the name (`std::is_sorted`, a `cppstd` range check plus an emitter entry), the *type* (`GetX()`'s return, so the range could be followed), and the class (`TGraphMultiErrors` had no `GetEXlow`, which ROOT overrides from `TGraphAsymmErrors`).

## What to check when an algorithm over a ROOT array is refused

1. `active().method_returns` — is the accessor named at all, with the class's own name as the key?
2. `pycxx.infer`'s `typeof` on the receiver — the lookup needs `owner.name` to be the C++ type, so a local declared `TGraph g(...)` works and an `auto` from an unknown factory does not.
3. A pointer entry needs the `*`: `_named_type` is the only place that turns the string into `pointer=1`.
4. Then whether the *class* has the accessor: ROOT overrides `GetEXlow`/`GetEXhigh` in `TGraphMultiErrors`, and a missing one shows up as `'NoneType' object is not subscriptable` rather than a refusal.
