# Internal components

xrdroot has fifteen internal components in one Python project. They separate
ownership and dependencies while keeping the `xrdroot` distribution, source
root, command-line tools and public imports together.

Implementation is authorized and source preparation is in progress on accepted
parent `8a008d88`. These pages describe the prepared component contract and
refactors. They do not certify an integrated tree: the new inventory, isolation,
focused tests and six gates still require source-specific execution and review.
No zero-violation or tutorial-parity result is claimed.

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

The unchanged-parent ordinary full tutorial baseline was launched separately
from source preparation. This documentation packet accepts no final baseline
report or audit; recorded failures remain retained for review. The committed
`tools/tutorials/baseline.json` is unchanged, and the global no-PASS-lost ratchet
remains open, including the three Python set-order obligations whose original
captures were lost.

The earlier current073 translator survey translated 897 of 910 macros, meeting
the 98% floor, but retained three writable-reader crashes. Its no-crashes gate
failed. The separate reader candidate's FIRST full-gate launcher stopped before
the driver; it supplies no reader gate acceptance. Refactoring does not waive
those failures, change the 300-second tutorial limit or replace old evidence.

Acceptance requires the exact integrated source's component inventory and
isolation, strict coverage and all six gates, affected tutorials, translator
survey and global tutorial ratchet. See repository `MODULARISATION.md`,
`AGENTS.md` and the latest durable coordinator checkpoint for source-specific
status. Historical inventories and failed attempts retain their original scope.
