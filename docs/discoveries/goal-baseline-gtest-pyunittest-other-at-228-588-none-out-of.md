---
title: "Goal baseline: gtest + pyunittest + other at 228/588, none out of scope"
kind: "discovery"
summary: "At c5aabb22 gtest+pyunittest+other are 228/588 (38.8 %); all 360 non-passing are in scope, with ~30 blocked by missing names alone, 9 by one typedef diagnosis and 4 by the 300 s cap."
tags: [gtest, pyunittest, goal, measurement, clusters]
created: "2026-11-25T23:25:18.982Z"
updated: "2026-11-25T23:25:18.982Z"
---
Measured at `origin/main` = `c5aabb22` (25 Nov, a worktree pinned there, ROOT 6.40.04 as oracle, `-j 6`) the three kinds the goal names sit at **228 of 588 passing (38.8 %)**, and **not one of the 360 non-passing tests carries an out-of-scope label** — so "all of them pass" is a coherent target rather than a category error.

```text
kind         PASS  total    %    FAIL/UNSUPPORTED/TIMEOUT/SKIP
gtest          93    350  26.6     181/74/2/0
pyunittest     87    129  67.4      39/ 2/0/1
other          48    109  44.0       0/59/2/0
--------------------------------------------------------------
all           228    588  38.8     220/135/4/1
```

```bash
W=$HOME/.cache/xrdroot-runs/wt-small-$(date +%Y%m%d-%H%M)   # worktree add -B measure/… origin/main
python -m tools.rootsuite run --root "$HOME/src/.xrdroot-scratch/rootsrc" -j 6 \
  --xrdroot-src "$W/src" --kind gtest --kind pyunittest --kind other \
  --out "$HOME/.cache/xrdroot-runs/smalls-$(date +%Y%m%d-%H%M)"
```

## The 360, by what the logs and reasons say

- **~30 by missing names alone** — `ROOT.Random` (3), `TCpu` (3), `Detail.RDF` (4–5), `_Namespace.MultiProcess` (3), `ProcInfo_t` (2), `Internal.LogicalCPUBandwidthControl` (2), `Internal.RPageSource` (2), `std.true_type` (2), `_Namespace.Minuit2Minimizer` (2), `rmath.WrappedParamFunction` (2) and six more `rmath` names. Mostly aliases and small namespace attributes, the cheapest block in the whole goal.
- **9 by one translator diagnosis** — `tmva/tmva/test/DNN/CNN/TestConvNet.h:372: the name Net_t stands for two…`, i.e. the same `typedef` read two ways inside one header.
- **9 by the build path** — `xrdroot build: …` failures (executable/dictionary/library builds), plus 54 `xrdroot run:` refusals that name their feature.
- **4 by speed** — the 300 s cap (`mathcore-stdsort`, `minuit2_ParallelTest`, `ntuple_largefile1`, `testBulkApiMultiple`).
- **the rest** — a long tail of one gtest case per feature (`Basics` 5, `Sanity` 3, `multiply9x9.simple` 2, `stdRead` 2, `ColTypeNames` 2, `TASImage.FilledCircleOpaque`, `RZip.ZipBufferSizesOld`, `TEnum.UnderlyingType`, …).

Per-test causes are recoverable from `logs/<kind>/<id>/run.log` in the run directory (588 of them); the runner's `logs` field is a directory, not a file, so read `run.log` inside it.
