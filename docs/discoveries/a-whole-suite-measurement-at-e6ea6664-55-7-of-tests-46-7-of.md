---
title: "A whole-suite measurement at e6ea6664: 55.7 % of tests, 46.7 % of tutorials"
kind: "discovery"
summary: "A full run at e6ea6664 gives 1570/2820 tests (55.7 %, 65.4 % of those attempted) and 555/1189 tutorials (46.7 %, 64.1 % of those ROOT runs); five timeouts and ten FLAKY tutorials sit outside those figures."
tags: [measurement, rootsuite, tutorials, percentages, baseline]
created: "2026-11-24T19:49:05.653Z"
updated: "2026-11-24T19:49:05.653Z"
---
A whole-suite run at `origin/main` = `e6ea6664` (ROOT 6.40.04's tree as oracle, this machine, load ~5) reads **1570 of 2820 tests PASS (55.7 %)** and **555 of 1189 tutorials PASS (46.7 %)**. Both were taken with the source pinned to a worktree at that commit, because `main` moves under a long run.

```text
2820 tests      PASS 1570 (55.7%)  FAIL 671  UNSUPPORTED 408  DIFF 155  TIMEOUT 5  SKIP 11
  -> of the 2401 not refused/skipped:            65.4 %
1189 tutorials  PASS  555 (46.7%)  DIFF 131  UNSUPPORTED  90  FAIL 75  SKIP 323
                                   FLAKY 10  ORACLE-FAIL 5
  -> of the 866 ROOT runs:                       64.1 %
  -> of the 851 reachable as promises:           65.2 %
```

```bash
cd /Users/rcurrie/src/xrdroot && . .venv/bin/activate && export PYTHONPATH=$W/src
W=$HOME/.cache/xrdroot-runs/wt-suite-$(date +%Y%m%d-%H%M)   # git worktree add "$W" -B measure/… origin/main
python -m tools.rootsuite run --root "$HOME/src/.xrdroot-scratch/rootsrc" -j 6 \
  --xrdroot-src "$W/src" --out "$HOME/.cache/xrdroot-runs/suite-$(date +%Y%m%d-%H%M)"
python -m tools.tutorials run --tutorials "$HOME/src/.xrdroot-scratch/rootsrc/tutorials" \
  --jobs 8 --xrdroot-src "$W/src" --out "$HOME/.cache/xrdroot-runs/tutorials-$(date +%Y%m%d-%H%M)"
```

## Per kind, and the caveats that belong with the figures

| suite | PASS | total | % |
|---|---|---|---|
| roottest | 1332 | 2173 | 61.3 |
| gtest | 93 | 350 | 26.6 |
| pyunittest | 87 | 129 | 67.4 |
| `other` | 48 | 109 | 44.0 |
| `test/` | 10 | 59 | 16.9 |

- Five tests and six tutorials hit the 300 s cap while two suites ran together (`mathcore-stdsort`, `minuit2_ParallelTest`, `ntuple_largefile1`, `testBulkApiMultiple`, `runtreeCloneTest2`; `df101` ×2, `df102`, `hist102`, `rs401d` ×2) — rerun alone before calling them losses, so 1570/555 are lower bounds.
- The 10 FLAKY tutorials are ones ROOT itself draws differently run to run; 555 + 10 is the old 564 baseline, so the drop against `tools/tutorials/baseline.json` is that reclassification, not a regression.
- `hsimple.C` is ORACLE-FAIL: the oracle now exits **255** on it (`root -b -q hsimple.C` reproduces; a trivial macro exits 0) while still writing `hsimple.root`, so dependants are unaffected but the tutorial cannot be compared.
