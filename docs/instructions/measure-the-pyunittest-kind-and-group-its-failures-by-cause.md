---
title: "Measure pyunittest first: 96/129, and its 30 failures are three causes"
kind: "instruction"
summary: "pyunittest first reads 96 PASS / 30 FAIL / 2 UNSUPPORTED / 1 SKIP of 129 in 129 s; its 30 failures are three causes (18 cppyy, 9 pythonizations, 3 dataframe/ntuple). Bank the 96 first — the baseline holds gtest rows only."
tags: [pyunittest, rootsuite, measurement, baseline, cppyy]
created: "2026-11-27T00:01:09.469Z"
updated: "2026-11-27T00:01:09.469Z"
---
The `pyunittest` kind had never been measured in this loop; it now reads **96 PASS / 30 FAIL / 2 UNSUPPORTED / 1 SKIP of 129** in 129 s at `--jobs 4`, and its 30 failures fall into three causes rather than thirty. Measure a kind before touching code in it, and bank its PASSes as a baseline, because `tools/rootsuite/baseline.json` holds gtest rows only — a regression in pyunittest would otherwise go unseen.

```bash
python -m tools.rootsuite run --root "$HOME/src/.xrdroot-scratch/rootsrc" --kind pyunittest \
    --jobs 4 --xrdroot-src "$WT/src" --out "$OUT"
# {"pyunittest": {"PASS": 96, "FAIL": 30, "UNSUPPORTED": 2, "SKIP": 1}, "roottest": {"PASS": 3}}
jq -r '.tests[] | select(.status!="PASS") | "\(.status) \(.id)"' "$OUT/results.json"
```

## The three causes behind the 30

| Cause | Programs |
|---|---|
| cppyy's own binding semantics (`bindings/pyroot/cppyy/cppyy/test_*`) | 18 |
| pythonizations: `AsNumpy`/`MakeNumpy`, `numba`declare, SOFIE, `RooFit`, memory | 9 |
| dataframe/ntuple Python API (`DatasetSpec`, merge results, ntuple entry access) | 3 |

## What to do

Pick by cause, not by program. The eighteen cppyy modules are the biggest single cause and will take several rounds; the pythonization names are individually smaller and share `AsNumpy`. Bank the 96 PASSes first, so the work has a regression guard.

```mermaid
flowchart TD
    A["pyunittest: 129 programs"] --> B["96 PASS (bank these first)"]
    A --> C["30 FAIL"]
    A --> D["2 UNSUPPORTED + 1 SKIP"]
    C --> E["18: cppyy binding semantics"]
    C --> F["9: pythonizations (AsNumpy, numba, SOFIE, RooFit)"]
    C --> G["3: dataframe/ntuple Python API"]
```
