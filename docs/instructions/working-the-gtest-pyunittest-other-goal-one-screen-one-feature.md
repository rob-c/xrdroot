---
title: "Working the gtest/pyunittest/other goal: one screen, one feature behind it"
kind: "instruction"
summary: "Each missing name in the gtest/pyunittest/other goal hides a feature: measure the 591 records in ten minutes, then work screen-by-screen, knowing PyROOT's brace-rendered containers and typed null."
tags: [gtest, pyunittest, goal, pyroot, rendering, measurement]
created: "2026-11-26T00:02:45.862Z"
updated: "2026-11-26T00:02:45.862Z"
---
Working the gtest + pyunittest + `other` goal in this repository is a loop of *remove one screen, meet one feature*: the harness refuses a test with `ROOT has X; xrdroot.pyroot does not yet`, and making `X` exist does not make the test pass — it moves the test one wall deeper. Two consecutive examples, both measured on 25 Nov 2026:

```text
std::true_type  absent -> added (03ac9b82) -> ntuple_type_name, ntuple_show now need RRecordField
ProcInfo_t      absent -> added (1f5a4339) -> kDTreeTest now needs TKDTreeIF
pretty_printing 3 cases failing -> vector text + MakeNullPointer -> PASSES
```

```bash
cd /Users/rcurrie/src/xrdroot && . .venv/bin/activate && export PYTHONPATH=$W/src
W=$HOME/.cache/xrdroot-runs/wt-goal-$(date +%Y%m%d-%H%M)   # worktree add -B goal/… origin/main
python -m tools.rootsuite run --root "$HOME/src/.xrdroot-scratch/rootsrc" -j 6 --xrdroot-src "$W/src" \
  --kind gtest --kind pyunittest --kind other --out "$HOME/.cache/xrdroot-runs/goal-$(date +%Y%m%d-%H%M)"
```

591 records in about ten minutes, so every batch can be measured rather than assumed. Baseline at `c5aabb22`: 228/588 PASS (gtest 93/350, pyunittest 87/129, other 48/109), 360 non-passing and **zero** out-of-scope labels.

## What a test needs that is not obvious from the error

- **PyROOT renders a container as its elements between braces, and `repr` is `str`**: `print(ROOT.std.vector[float]([0, 1]))` is `{ 0, 1 }`, with each element in C++'s default ostream text (`%g` for a float, no `.0`). ROOT's own `pretty_printing` test asserts `assertEqual(s, r)`, so a Python-list rendering fails it.
- **`ROOT.MakeNullPointer("TTree")` is a typed null, not `None`**: false, comparable with another null of the same class, rendered `<TTree object at 0x0>`.
- **`ProcInfo_t`** carries `fCpuUser`/`fCpuSys` in milliseconds and `fMemResident`/`fMemVirtual` in kilobytes, filled by `gSystem.GetProcInfo(info)` from `getrusage` plus the kernel's own figures — `proc_pidinfo`/`libproc` on macOS, `/proc/self/statm` elsewhere.

Per-test causes live in `logs/<kind>/<id>/run.log` inside the run directory (the record's `log` field names a *directory*, not a file).
