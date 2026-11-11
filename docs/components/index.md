# Internal components

xrdroot has fifteen internal components in one Python project. They separate
ownership and dependencies while keeping the `xrdroot` distribution, source
root, command-line tools and public imports together.

The refactor has landed as thirty-three gated commits on the branch
`refactor/modularisation-commits-20261021` (head `2418664`, from accepted parent
`8a008d88`). On that tree the component contract reports 857 source modules and
9,788 import references with zero undeclared violations, every one of the
fifteen components imports in a fresh process with higher imports blocked, and
all six gates pass. The measured results and their exact scope are in
[validation status](#validation-status) below. Main is not yet published.

## Install and work in one source tree

Use the existing root project from a checkout or an isolated worktree:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,zstd,docs]"
export PYTHONPATH="$PWD/src"
```

For a smaller installation, choose existing extras for the work you use:

```bash
pip install -e ".[fit,plot]"
pip install -e ".[geom]"
pip install -e ".[tmva]"
```

All examples install the same project. Component names are ownership labels;
they are not separate install targets. NumPy and the pinned xrdclient remain
project dependencies. Other libraries retain their existing extras and lazy
imports. ROOT remains a validation oracle and is never a runtime dependency.

The distribution itself is smoke-tested as a built artifact: CI builds the
sdist and the wheel, installs the wheel into a clean environment and writes and
reads back a ROOT file with it, so what `pip install xrdroot` puts on a machine
is the whole project and not only what the editable install provided.

## Responsibilities and direct dependencies

Dependencies in this table are direct import permissions. A transitive dependency
is available while executing a component, but does not grant a new direct import
edge. The authoritative map is `[tool.xrdroot.components]` in `pyproject.toml`.

| Component | Responsibility | Allowed direct component dependencies |
|---|---|---|
| [cppstd](cppstd.md) | Generic C++ standard-library and value semantics | None |
| [core](primitives.md) | Shared arrays, numerics, errors and display primitives | cppstd |
| [hist](analysis.md#hist) | Histograms, graphs, formulas and fitting | core |
| [spectrum](analysis.md#spectrum) | Spectrum analysis | core |
| [fits](analysis.md#fits) | FITS data engine | core |
| [io](io.md#io) | ROOT files, trees, streamers and RNTuple | core, hist |
| [rdf](io.md#rdf) | Engine data-frame execution | core, hist, io |
| [geom](presentation.md#geom) | Geometry models and scene data | core |
| [graphics](presentation.md#graphics) | Canvas painting and plotting backends | core, hist, geom |
| [roofit](analysis.md#roofit) | Statistical models, data and likelihood fitting | core, hist, cppstd |
| [roostats](analysis.md#roostats) | Inference, intervals and hypothesis tests | core, hist, roofit |
| [histfactory](analysis.md#histfactory) | Channel/sample models and persistence layouts | core, hist, io, roofit, roostats |
| [tmva](analysis.md#tmva) | Learning, evaluation and model persistence | core, hist, io |
| [pyroot](applications.md#pyroot) | ROOT names, translator and session adapters | All thirteen engine components above |
| [app](applications.md#app) | Public package facade and command-line application | All fourteen components above |

Read [boundary checks](boundaries.md) for the enforcement commands and
[providers](providers.md) for separately declared upward dispatch.
[Public ROOT classes](../root.md) remain documented in the existing API guide.

## Validation status

Measured on the integrated branch `refactor/modularisation-commits-20261021` at
`2418664` on 2026-10-25. Each figure is bound to that source; nothing transfers
to another tree.

| Check | Result on `2418664` |
|---|---|
| Component ownership and imports | 857 modules, 9,788 import references, zero undeclared violations; the one declared private `iminuit._core` import is reported and reviewed |
| Fresh-process isolation | all fifteen components import with every higher import blocked (CI `fresh component imports`); `isolate cppstd --tests cppstd` runs its 61 focused tests |
| Six gates | both test modes record 9,814 passed, 1 skipped, 46 expected failures and strict 100% line and branch coverage; Ruff, maintainability, strict mypy and strict docs pass |
| CI | green on the branch, run `40158357628`: five test jobs (Ubuntu 3.10–3.13, macOS 3.12), lint, docs and the built-distribution smoke check |
| Translator survey | 900 of 910 macros translated (98.9%), 10 refused, 0 crashed — the unchanged 98% floor and the no-crashes requirement both pass |
| Global no-PASS-lost | full harness on this tree recorded 553 PASS; the nine tutorials its ratchet flagged were loaded-machine timeouts and all nine pass when rerun alone; the three set-order tutorials pass against ROOT's own recorded orders |

Component debt was burned down to zero in eight reviewed groups (58 → 47 → 41
→ 36 → 32 → 25 → 12 → 8 → … → 0) with no limit raised, no coverage lowered and
no assertion loosened. The committed `tools/tutorials/baseline.json` is
unchanged. Publishing main remains a separate decision and has not happened:
the published `main` is still `fa34ea9`.

Historical inventories, lane candidates and failed attempts keep their original
scope and are not pooled into the figures above. See repository
`MODULARISATION.md`, `AGENTS.md` and the latest durable coordinator checkpoint
for the per-group acceptance records.
