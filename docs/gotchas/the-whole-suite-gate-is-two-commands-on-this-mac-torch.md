---
title: "The whole-suite gate is two commands on this Mac: torch segfaults under xdist"
kind: "gotcha"
summary: "Run the gate as two pytest commands (torch file and references file apart); a single -n 4 run segfaults a worker and its lost coverage reads as 26 %."
tags: [gate, pytest, coverage, xdist, torch]
created: "2026-11-26T16:11:19.981Z"
updated: "2026-11-26T16:11:19.981Z"
---
The gate command in `AGENTS.md` (`pytest -q -n 4 --cov=xrdroot`) **cannot run as one command on this Mac**: under parallel load `tests/test_tmva_learn_networks.py` trains a torch network and a worker dies with `Fatal Python error: Segmentation fault` inside `torch/nn/functional.py:binary_cross_entropy_with_logits`. The dead worker's coverage data is lost with it, so the same run reports 9 torch failures *and* coverage near 26 % — which reads like a broken change and is not one.

```text
one command   -n 4  -> worker segfaults in torch, 5-9 torch failures, TOTAL 26 %
two commands  A: -n 4 --cov      --ignore test_tmva_learn_networks.py --ignore test_pyroot_references.py
              B:      --cov-append  those two files only          -> TOTAL 100 %
the figure to quote is B's, which is the whole package appended
```

## Why the two files are apart

- `test_tmva_learn_networks.py` trains through torch; alone it takes 8 s and always passes.
- `test_pyroot_references.py` fails only when it shares a worker, because `TProcessID` is process-global.

`-n 0` (one process, everything) dies with `Fatal Python error` at 63–91 %, so a whole-suite run on macOS stays two commands. Linux CI runs the single command and is green. This is `W0-106` in `TODO.md`.
