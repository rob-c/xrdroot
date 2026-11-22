---
title: "Running the tutorials ROOT vetoes for \"no display\": --also-vetoed, worth 23 PASSes"
kind: "instruction"
summary: "`--also-vetoed \"needs a display (GUI)\"` runs ROOT's headless drawing tutorials and yields 23 real PASSes of 135; the records are marked beyond_ctest and the baseline refuses them."
tags: [tutorials, harness, gui-veto, baseline, measurement]
created: "2026-11-22T13:43:03.229Z"
updated: "2026-11-22T13:43:03.229Z"
---
`tools/tutorials` counts a tutorial as SKIP when ROOT's own `tutorials/CMakeLists.txt` vetoes it — `gui_veto` above all, with 135 entries. That veto exists because ROOT's CI is headless, not because the code cannot run: here ROOT itself exits 0 on 81 of the 135 when run as `root -l -b -q <file>` in the tutorial's own directory, and xrdroot reproduces 70 of those stdouts exactly.

```text
135 gui_veto tutorials
  ├─ 54  ROOT itself fails here (no cocoa/EVE backend)   -> ORACLE-FAIL
  └─ 81  ROOT exits 0
        ├─ 42  xrdroot refuses on purpose (TGMainFrame, TTreeViewer, TSlider, TGL*)
        ├─ 14  FAIL, 2 DIFF
        └─ 23  PASS   <- what the flag is worth, images compared
```

## The flag, and what it is worth

```bash
cd /Users/rcurrie/src/xrdroot && . .venv/bin/activate && export PYTHONPATH=$PWD/src
python -m tools.tutorials run \
  --tutorials /Users/rcurrie/src/.xrdroot-scratch/rootsrc/tutorials \
  --also-vetoed "needs a display (GUI)" --jobs 8 \
  --out /Users/rcurrie/.cache/xrdroot-runs/gui-run \
  --only 'visualisation/cocoa/flower.C'      # repeat --only, or omit for all 135
```

`--also-vetoed REASON` takes a substring of the veto reason (`tools/tutorials/catalog.py`'s `VETO_REASONS`), runs what that reason dropped, and marks each record `beyond_ctest`. Over all 135: **PASS 23** / DIFF 2 / FAIL 14 / UNSUPPORTED 42 / ORACLE-FAIL 54. Comparing stdout alone badly overstates the gain — 70 match on text, but these tutorials write PNGs and the pictures usually differ.

## Why the baseline refuses such a run

`python -m tools.tutorials baseline RUN/results.json` raises when any record carries `beyond_ctest`: the baseline is ROOT's own test set, and these are not in it. Keep them as a second, labelled count and never let them into `tools/tutorials/baseline.json`.
