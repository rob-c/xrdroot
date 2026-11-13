# Internal components

xrdroot has seventeen internal components in one Python project. They separate
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
| [pycxx](pycxx.md) | The C++ translator: parser, emitters, runtime, GoogleTest, dialects | cppstd |
| [pyroot](applications.md#pyroot) | ROOT names and session adapters | All thirteen engine components above, pycxx |
| [pycint](pycint.md) | ROOT's dialect of the translator, macro execution, the `cint` shim | pycxx and pyroot with everything below it |
| [app](applications.md#app) | Public package facade and command-line application | All sixteen components above |

Read [boundary checks](boundaries.md) for the enforcement commands and
[providers](providers.md) for separately declared upward dispatch.
[Public ROOT classes](../root.md) remain documented in the existing API guide.

## Validation status

The component graph is enforced on every commit: `tools/components.py check
--strict` must report zero undeclared imports, and CI imports each component
in a fresh process with everything above it blocked. The one declared private
import, `iminuit._core`, is reported and reviewed. The refactor reached zero
debt in small gated groups, with no limit raised, no coverage lowered and no
assertion loosened. Its plan and audit trail are kept in the repository under
`docs/history/`.
