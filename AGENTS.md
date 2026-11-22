# xrdroot — instructions for coding agents

## Goal

xrdroot is a pure-Python, NumPy-backed replacement for CERN ROOT. It reads and
writes ROOT files, does what ROOT's analysis classes do (histograms, fits, RooFit,
RooStats, HistFactory, TMVA, RDataFrame, RNTuple, graphics), and runs ROOT's own
programs without ROOT:

- **PyROOT scripts** run against `xrdroot.pyroot`, a ROOT-compatible namespace
  (`TFile`, `TH1F`, `TCanvas`, `gROOT`, `gStyle`, `RooFit`, `RDataFrame` …).
- **C++ macros and programs** (`.C`, `.cxx`) are translated to Python by
  `xrdroot.cint` and run against the same namespace (`xrdroot run FILE.C`, `.x`,
  `gROOT.ProcessLine`), including GoogleTest programs and `main()`.

**The target is ROOT's whole test suite, not only its tutorials.** ROOT 6.40's
ctest suite has five kinds of test; one Linux CI job runs about 3,820:

| Kind | ~Count | Checked by | Needs ROOT? |
|---|---|---|---|
| `tutorial-*` (`tutorials/`) | 918 | compared with ROOT's own run | yes (oracle) |
| `roottest-*` (`roottest/`) | 2,210 | stdout against committed `.ref` files | no |
| `gtest-*` (each module's `test/`) | 389 | self-checking GoogleTest | no |
| `pyunittests-*` (PyROOT's `test/`) | 109 | self-checking Python tests | no |
| `test-*` (`test/` stress programs) | 64 | self-checking, some `.ref` | no |

`tools/rootsuite` also finds about 110 module test programs outside those
families (`mathcore-*`, `tmva-*` …), reported as kind `other`.

Every test is either PASS, or refused with a clear, full-sentence
`UnsupportedFeatureError` because it is genuinely out of scope: GUI widgets,
Qt, SYCL/CUDA, cling/ACLiC internals, generated C++ dictionaries. Those scope
labels are committed (see "Measuring progress") and reviewed, never used to
hide something xrdroot merely fails today. Real ROOT is a **validation oracle
only**: it must never become a runtime dependency.

Well-maintained data-science libraries may be used: iminuit (Minuit2), SciPy,
scikit-learn, XGBoost, PyTorch, pyhf, matplotlib, plotly, pyvista. NumPy is the
only hard dependency besides the transport (`xrdclient`). Anything else is an
optional extra in `pyproject.toml`, imported lazily, and refused with the
`pip install` that fixes it when missing (see `src/xrdroot/library.py`).

The package will later be split into separate projects along its component
boundaries. Keep those boundaries clean (see "Components").

## Where things are

| Area | Engine | ROOT-compatible layer |
|---|---|---|
| File I/O, trees, streamers | `src/xrdroot/{file,tree,streamers,writer,wtree,wbranch,…}.py` | `pyroot/trees`, `pyroot/core` |
| C++ → Python translator (no ROOT) | `src/xrdroot/pycxx/` (`runtime/` is what translated code calls; `dialect.py` is the interface a library's dialect fills in) | — |
| ROOT's dialect, `.x`/`.L`/`ProcessLine`, the CINT prompt | — | `src/xrdroot/pycint/` (`xrdroot.cint` is its compatibility shim) |
| C++ standard library in Python | `src/xrdroot/cppstd/` | `pyroot/stl.py`, `pyroot/stdlib.py` |
| Histograms, graphs, functions | `hist.py`, `graph*.py`, `function/`, `formula/` | `pyroot/core`, `pyroot/graphics` |
| Fitting | `fit/` | `pyroot/core` |
| Canvas renderer (ROOT's pixels) | `canvas/` (rasteriser, fonts, TLatex layout) | `pyroot/graphics` |
| Plotting with matplotlib/plotly/bokeh | `plot/` | the draw hook in `pyroot/core` |
| RooFit / RooStats / HistFactory | `roofit/`, `roostats/`, `histfactory/` | `pyroot/roofit`, `pyroot/roostats` |
| RDataFrame | `rdf/` | `pyroot/rdf` |
| RNTuple | `rntuple/` | `pyroot/ntuple` |
| TMVA | `tmva/` | `pyroot/tmva.py` |
| Spectra (TSpectrum) | `spectrum/` | `pyroot/spectra` |
| Geometry (TGeo) | `geom/` | `pyroot/geom` |
| ROOT 7 (RHist, RCanvas) | — | `pyroot/histv7`, `pyroot/rcanvas` |
| Linear algebra | — | `pyroot/linalg` |
| Test harnesses (dev only) | `tools/rootsuite/` (all five kinds), `tools/tutorials/` (oracle comparison), `tools/cint_survey.py` | — |

New PyROOT modules must be added to `SUBMODULES` in
`src/xrdroot/pyroot/__init__.py`. Follow how existing areas split an engine
package (plain Python API) from its `pyroot/` wrapper (ROOT's names and behaviour).

## Components

The source is eighteen internal components with an enforced import graph:
`cppstd → core → hist → io → rdf/graphics/roofit/… → pyroot → pycint → app`,
with `pycxx` (the C++ translator) depending on `cppstd` alone, beneath pyroot.
Their ownership and allowed dependencies are in `[tool.xrdroot.components]` in
`pyproject.toml`; `docs/components/` explains each one.

The translator is split in two. `pycxx` translates and runs C++ and knows no
library past the standard one: never put a ROOT name, pattern or import in it.
ROOT's knowledge - typedefs, headers and `R__` macros, class-name patterns,
`TString`, `TTreeReader`, `RVec`, the fill-loop and ntuple hooks, cling's
printing of a returned value - is data and small hooks in `pycint/dialect.py`
and `pycint/emit.py`, read through the `Dialect` interface of
`pycxx/dialect.py`; what translated ROOT code calls is `pycint/runtime/`. pyroot
reaches pycint only through the `interpreter.*` providers.

```bash
python tools/components.py check               # every import, including lazy ones
python tools/components.py isolate cppstd core pycxx pycint  # fresh-process import isolation
python tools/components.py isolate pycxx --tests pycxx  # the translator's tests, no ROOT
python tools/components.py graph
```

A lower component never imports a higher one. When an engine needs something
from above, return plain data or an engine object and let the wrapper build
the ROOT object, pass a callable in, or use a provider in `xrdroot.plugins`,
in that order of preference. Generic C++ behaviour belongs in `cppstd`; ROOT
specifics stay in adapters. The plan that set this up is in `docs/history/`.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,zstd]"          # plus extras: fit, tmva, geom, plot, docs, …
export PYTHONPATH=$PWD/src            # always, especially in a git worktree
```

ROOT's sources, for the tests and tutorials: a sparse clone at the matching tag.

```bash
git clone --filter=blob:none --sparse --depth 1 -b v6-40-04 \
    https://github.com/root-project/root.git rootsrc-6.40
git -C rootsrc-6.40 ls-tree -r --name-only HEAD | grep -oE '^.*/test/' \
    | grep -vE '^(interpreter|roottest|tutorials)/' | sort -u > /tmp/testdirs
git -C rootsrc-6.40 sparse-checkout set tutorials test roottest cmake/modules $(cat /tmp/testdirs)
```

The **ROOT 6.40.04 oracle** is needed only for tutorial comparison. Install it
with micromamba from conda-forge (Homebrew's ROOT does not install). If
micromamba is missing, the standalone binary needs no installer
(`curl -fsSL https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj bin/micromamba`):

```bash
micromamba create -p ~/.local/root-oracle -c conda-forge --override-channels \
    root=6.40.04 python=3.13 pytest numpy pandas
```

About 2.3 GB. `root-config --features` on it is the macOS ARM64 feature set.

ROOT's tutorial outputs are cached in `~/.cache/xrdroot-tutorials/oracle`, so
ROOT only reruns when a tutorial changes (`--refresh-oracle` forces a rerun).

## Measuring progress

**The whole suite** (`tools/rootsuite`; needs only ROOT's sources):

```bash
python -m tools.rootsuite list --root rootsrc-6.40          # what ROOT's CMake declares, per kind
python -m tools.rootsuite run  --root rootsrc-6.40 --jobs 6 --out RUN_DIR [--kind roottest] [--only 'gtest/hist/*']
python -m tools.rootsuite ratchet RUN_DIR/results.json      # fails if a baseline PASS was lost
```

Statuses are PASS, FAIL, DIFF (ran, output differs from the `.ref`),
UNSUPPORTED (xrdroot refused, or lacks the subcommand the test needs),
OUT-OF-SCOPE (a committed scope label), SKIP (the harness cannot express the
test) and TIMEOUT. The suite is configured as on macOS ARM64: the oracle's
`root-config --features` (else the committed macOS ARM64 set; `--features`
overrides), `APPLE`, `CMAKE_SYSTEM_PROCESSOR=arm64`. Scope labels live in
`tools/rootsuite/scope.toml`, each with a full-sentence reason, and only
cover what ROOT cannot run on such a Mac (CUDA, SYCL, X11, io_uring): tests
needing dictionaries, ACLiC, compiled executables, cling or cppyy stay in
scope. The baseline is `tools/rootsuite/baseline.json`.

ROOT's build model maps onto xrdroot's (run in the test's working directory):

| ROOT | xrdroot |
|---|---|
| a dictionary (rootcling, genreflex) | `xrdroot build dictionary --name NAME [--linkdef F] [--selection F] [-I D]... HEADER...` → `libNAME.xrdlib` |
| a test library | `xrdroot build library --name NAME [-I D]... SOURCE...` → `libNAME.xrdlib` |
| a test executable | `xrdroot build executable --name EXE [--library NAME]... [-I D]... SOURCE...` → `./EXE` |
| ACLiC (`.L m.C+`, `ROOTTEST_COMPILE_MACRO`) | `xrdroot build aclic m.C+kc` → `m_C.xrdlib` |
| `root.exe [opts] [-e line]... [files]...` | `xrdroot root` with the same words |
| `hadd`, `rootls`, `rootrm`, `rootmkdir`, `rootmv`, `rootcp`, `rootprint` | `xrdroot hadd`, `ls`, `rm`, `mkdir`, `mv`, `cp`, `print` |

`NAME` is the file ROOT writes less `lib` and its suffix. Fixtures and
`DEPENDS` order tests as ctest does, a test whose fixture setup did not pass
is not run, and the working copy has the build tree's `bin/`, `tutorials/`
and `hsimple.root`.

**Tutorials against the ROOT oracle** (`tools/tutorials`):

```bash
python -m tools.tutorials run --tutorials rootsrc-6.40/tutorials --jobs 4 \
    --oracle-prefix ~/.local/root-oracle --xrdroot-src $PWD/src --out RUN_DIR [--only roofit/roostats]
python -m tools.tutorials ratchet RUN_DIR/results.json
```

It compares stdout, written ROOT files and images with ROOT's run. The
baseline is `tools/tutorials/baseline.json`.

**The translator** has its own ratchet:

```bash
python tools/cint_survey.py rootsrc-6.40/tutorials --min-percent 98
python tools/cint_survey.py rootsrc-6.40/roottest --top 20
```

Rules for all of them:

- The per-test timeout is 300 s. **Never raise it.** Make the code faster. On
  a loaded machine, rerun timeouts alone before calling one a regression.
- Run long jobs in the background with their output under `RUN_DIR`, outside
  the repository.
- **No PASS may be lost.** Update a baseline in the same commit as the
  improvement it records, and only from a real run.

## Gates: every commit must pass all of these (CI runs them)

```bash
pytest -q -p no:cacheprovider -n 4 --cov=xrdroot     # 100% line AND branch coverage
XRDROOT_TEST_OTHER_MACHINE=1 pytest -q -p no:cacheprovider -n 4 --cov=xrdroot
ruff check src tests tools
mypy                                                 # strict
python tools/maintainability.py check
python tools/components.py check
mkdocs build --strict
```

CI runs Linux on Python 3.10–3.13 and macOS on 3.12. Run the gates, then
commit, then push, as separate steps. After pushing, watch CI (`gh run watch`)
and fix it until it's green.

Hard rules:

- **Never raise a limit.** The maintainability limits are CCN ≤ 10,
  cognitive ≤ 15, NPath ≤ 200, Halstead volume ≤ 500, nesting ≤ 4. Don't lower
  coverage, raise the harness timeout, or add `# pragma: no cover` to dodge a
  branch. Split functions instead.
- **Never delete or loosen an assertion to make it pass.** Find out whether the
  code or the test is wrong.
- **Numbers that must match ROOT exactly:**
  - ROOT's reference outputs were printed on darwin x86_64. `tests/refmachine.py`
    defines `ROOTS_MACHINE`: assert bit-exact there.
  - Elsewhere, use a tolerance with a comment explaining the numerics:
    `FIT_REL`, `PROFILE_REL` (1e-9), `NLL_REL` (1e-12), `roots()`. Typical
    causes are glibc vs Apple libm rounding (`lgamma`, `erf`), FMA on arm64, and
    `sum()` before Python 3.12.
  - Route transcendental functions through `xrdroot.random.libm` wherever ROOT
    exactness is claimed. NumPy's ufuncs use AVX-512 code paths on some CI
    runners and round differently from the C library ROOT calls.
- Tests that need an optional package must skip cleanly when it is missing.
- Python 3.10 is the floor; `xrdclient`, a hard dependency, needs it.
- pytest-timeout uses `timeout_method = "thread"`. The signal method
  deadlocked coverage's tracer lock and hung CI; don't change it back.

## Working practice

- The sources of truth are git, CI, and the committed baselines and scope
  labels. Do not keep project state in files outside the repository, and do
  not build receipt, audit-chain, process-ledger or lock machinery around the
  gates: a gate passes or it doesn't, and the harness's `results.json` is the
  evidence.
- Work in small commits, one fix group each. Work in parallel in separate git
  worktrees with `PYTHONPATH=$PWD/src`, on disjoint files where possible.
- When a run fails, find and fix the cause. Don't re-litigate it with process.

## House style

- Docstrings are prose explaining *why*, not restating the signature.
  Constants carry `#:` comments.
- Refusals are full sentences that say what is unsupported and what to do
  instead: `raise UnsupportedFeatureError("…")`.
- Test names are sentences:
  `test_a_url_is_fetched_whole_before_it_is_read`.
- Match the surrounding code's naming, density and idiom. Keep commits small,
  with messages in the existing style (`feat:`, `fix:`, `perf:`, `tests:`,
  `docs:`). Document new user-facing classes in `docs/root.md`.

## What remains, in priority order

Start from the latest `tools/rootsuite` and `tools/tutorials` results. Group
every FAIL, DIFF and UNSUPPORTED by root cause (the missing name or method, the
translator refusal, the failing feature) and count how many tests each cause
blocks. Work from the top. Known items:

1. **Cheap, wide wins across the suite.**
   - Translator refusals that block many roottest and gtest files
     (`tools/cint_survey.py --top 20` on `roottest/` and the module test dirs).
   - The remaining top PyROOT unit-test failure causes.
   - roottest `.ref` differences caused by output formatting rather than physics.
2. **Speed.**
   - RooStats toy studies (FeldmanCousins, HypoTestInverter,
     StandardHypoTestDemo, One/TwoSidedFrequentistUpperLimitWithBands,
     rs101, rs701, StandardBayesianNumericalDemo) and rf704.
   - The NanoAOD analyses near the timeout (df103, df104, df107, hist103), df101,
     ntpl016, the `mt_*`/`mp_*` ntuple fillers, fit2dHist, fitFraction, minuit2FitBench.
   - Profile, then vectorise RooFit likelihood and toy generation. Keep ROOT's
     numerics: same integration rules, same random streams.
3. **RooFit configuration and persistence.**
   - `RooWorkspace::writeToFile` and ascii I/O (an ungated start is on the
     `wip/r2-workspace-write-20261112` branch).
   - `RooLagrangianMorphFunc`; RooSimWSTool SIMCLONE/EDIT (rf513); rf401/rf402.
   - The remaining DIFFs: rf211, rf316, rf608.py.
4. **RDataFrame.** `Snapshot` of `vector<XYZTVector>` (df002); `Book` (rf408);
   string-valued `TTree::Draw` and RDF expressions (hist006, tree501); df029, df030, df040.
5. **Maths.** The remaining GenVector classes (rotations, Polar3DPoint, boosts);
   GoFTest's Anderson–Darling test (hist017, goftest); TestBinomial;
   mathmoreIntegration; pdf007; `TF1Convolution`; `TMinuit` on iminuit.
6. **Missing classes**, each blocking a few tests: `TTask`,
   `TMultiLayerPerceptron`, `THnSparseD`, `TSpider`, `fopen`, `TAnnotation`,
   `gVirtualX` (batch no-ops), `ROOT.Numba`. Implement each, or refuse it in a
   full sentence.
7. **HistFactory and RooStats DIFFs.** hf001's written file; Zbi_Zgamma and
   HybridInstructional (factory `PROJ`); writing a `HypoTestInverterResult`; SPlot (rs301).
8. **Tree and ntuple output DIFFs.** tree107, tree121, tree123, tree201; ntpl007
   and ntpl009 written files; RNTuple fields of records, nested collections and
   low-precision floats; `RNTupleImporter`.
9. **Graphics fidelity.** Most image DIFFs are line widths, font metrics and
   text placement in `canvas/`. Fix them against ROOT's PNGs, pixel by pixel
   where a test already does, as in `test_canvas_draw.py`.
10. **Preparing the split into separate projects.** Make the transport a
    plug-in (local files in plain Python, `xrdclient` optional), give `core` its
    own exception base instead of `xrdclient.errors.XRootDError`, and keep the
    component graph acyclic.

Still wanted from the parity instrumentation: a file round-trip matrix (write
with xrdroot and read with ROOT, and the reverse, for every streamer layout),
a refusal inventory (every `UnsupportedFeatureError` with the tests that hit
it), and Linux checks of exact-value tests before pushing.

## Done means

- Every test of all five kinds is PASS, or OUT-OF-SCOPE/UNSUPPORTED with a
  full-sentence reason for something genuinely out of scope, or SKIP/ORACLE-FAIL
  for reasons outside xrdroot.
- The API-coverage report (`python -m tools.tutorials api-coverage`) shows every
  non-GUI ROOT class the suite uses as implemented.
- All gates pass and CI is green on every platform.
