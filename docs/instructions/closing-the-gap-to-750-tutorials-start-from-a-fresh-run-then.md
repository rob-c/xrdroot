---
title: "Closing the gap to 750 tutorials: start from a fresh run, then roofit and the canvas"
kind: "instruction"
summary: "750 tutorials needs 186 of ~284 obtainable failures fixed (66 %): start from a fresh whole-suite run, then roofit (92 failures) and canvas images, since the committed 564 baseline is stale."
tags: [tutorials, planning, baseline, flaky, measurement]
created: "2026-11-24T10:10:09.374Z"
updated: "2026-11-24T10:10:09.374Z"
---
750 tutorials is 66 % of everything still obtainable, so the first requirement is a fresh whole-suite run, not more fixes: the committed `tools/tutorials/baseline.json` stood at **564 PASS** while rounds 65–96 changed roofit, the canvas and the harness (`flaky.json`, `record-flaky`), and nobody had re-measured whole.

```text
1189 tutorials
 ├─ 326 SKIP            (GUI 164 of visualisation alone, helpers, not-in-CMake)
 ├─  15 ORACLE-FAIL     (ROOT itself fails here)
 ├─  10 recorded flaky  (tools/tutorials/flaky.json: ROOT's two runs differ in the draw)
 └─ ~848 obtainable ── 564 PASS ── ~284 failing
                                    └─ 750 needs 186 of them fixed  (66 %)
```

```bash
cd /Users/rcurrie/src/xrdroot && . .venv/bin/activate && export PYTHONPATH=$PWD/src
python -m tools.tutorials run \
  --tutorials $HOME/src/.xrdroot-scratch/rootsrc/tutorials \
  --jobs 8 --out $HOME/.cache/xrdroot-runs/tutorials-$(date +%Y%m%d)
python -c "import json,collections; d=json.load(open('$HOME/.cache/xrdroot-runs/tutorials-YYYYMMDD/results.json')); print(collections.Counter(t['status'] for t in d['tutorials']))"
```

## Where the ~284 failures are (committed baseline, by area)

| area | PASS | DIFF | FAIL | UNSUPPORTED | what it needs |
|---|---|---|---|---|---|
| roofit | 147 | 34 | 23 | 35 | message/print fidelity, the missing RooStats classes, Minuit2's stepping — the largest block, 50–65 reachable |
| io | 36 | 12 | 9 | 25 | RNTuple features, `TFile` modes, tree-basket bytes |
| math | 67 | 9 | 12 | 19 | GenVector, `rmath` gaps, one-class tutorials |
| visualisation | 131 | 7 | 3 | 21 | image fidelity only (the 164 SKIPs are GUI) |
| analysis | 47 | 9 | 3 | 13 | RDF: `SaveGraph`, snapshots, NanoAOD timeouts |
| legacy | 44 | 1 | 5 | 16 | `TRolke`, `PhaseSpace`, `TListAndSTL` |
| machine_learning | 12 | 13 | 2 | 0 | TMVA standalone-class text and numerics |
| hist | 76 | 1 | 5 | 2 | contours, `THnSparse`, `TH1::FFT` |

Roofit 55 + canvas 20 + the name tail 45 + exactness 20 ≈ 700; 750 additionally needs part of the Minuit2/TMVA/byte-exact block. Plan for 800 and treat the three subsystems as the campaign.
