# xrdroot: state assessment and the route to whole-suite parity

*14 November 2026. Measured on the commit `10e0e58` of `feat/root-testsuite`,
macOS ARM64 (18 cores), CPython 3.14.10, NumPy 2.5.4, against ROOT 6.40.04.*

This is an assessment, not a plan of record: it says what is there, what was
measured, where the remaining work actually is, and which engineering decisions
will decide whether the "whole ROOT test suite in pure Python" goal arrives in
months or in years.

## 0. How the numbers below were obtained

```bash
git clone --filter=blob:none --sparse --depth 1 -b v6-40-04 \
    https://github.com/root-project/root.git /tmp/rootsrc-6.40
git -C /tmp/rootsrc-6.40 ls-tree -r --name-only HEAD | grep -oE '^.*/test/' \
    | grep -vE '^(interpreter|roottest|tutorials)/' | sort -u > /tmp/testdirs
git -C /tmp/rootsrc-6.40 sparse-checkout set tutorials test roottest cmake/modules $(cat /tmp/testdirs)

python3.14 -m venv .venv && .venv/bin/pip install -e ".[dev,zstd,fit]"

python -m pytest -q -n 4                                   # unit suite + gates
python -m pytest -q -n 4 --cov=xrdroot                     # the 100% coverage gate
ruff check src tests tools ; mypy ; python tools/maintainability.py check
python tools/components.py check
mkdocs build --strict

python -m tools.rootsuite list --root /tmp/rootsrc-6.40
python -m tools.rootsuite run  --root /tmp/rootsrc-6.40 --jobs 6 --out RUN1 \
       --kind gtest --kind pyunittest --kind test --kind other      # 650 tests, 555 s
python -m tools.rootsuite run  --root /tmp/rootsrc-6.40 --jobs 6 --out RUN2 --kind roottest
                                                                    # 2173 tests, 876 s
python -m tools.tutorials run --tutorials /tmp/rootsrc-6.40/tutorials --jobs 4 \
       --oracle-prefix ~/.local/root-oracle --out RUN3               # 1189 tutorials
python -m tools.tutorials api-coverage --oracle-prefix ~/.local/root-oracle --out API
```

Caveats, all of which matter for the counts: the committed baselines were taken
in a 3.12/NumPy-1.26 environment (`~/xrdroot-parity/venv312-pin806`), this one is
3.14/NumPy 2.5.4; `libomp` is not installed here, so XGBoost (and therefore
TMVA's RBDT) does not import; the oracle here has no xml/r/unuran build options.
Where my run differs from the committed baseline, the difference is attributed
below rather than assumed to be a regression.

## 1. Headline

| | |
|---|---|
| source | 994 modules, 201 000 lines in `src/xrdroot`, largest file 1 868 lines |
| tests in this repository | 591 files, 11 270 cases, **64 s** with `-n 4`, 99.92 % line+branch coverage locally (the missing 0.08 % is `tmva/rbdt.py`, only reachable through the XGBoost import that this machine cannot make) |
| gates | ruff, `mypy --strict` (995 files), maintainability (27 804 functions, all within CCN ≤ 10 / cognitive ≤ 15 / NPath ≤ 200), component contract (11 519 imports, 0 undeclared), `mkdocs --strict` — **all green** |
| roottest | 1038 / 2173 PASS (47.8 %) |
| gtest | 52 / 350 PASS (14.9 %; 507 of 2020 cases) |
| pyunittest | 78 / 129 PASS (60.5 %; 439 of 515 cases) |
| `test/` stress programs | 4 / 59 PASS |
| module tests (`other`) | 14 / 109 PASS |
| all non-tutorial kinds | **1185 / 2820 PASS (42 %)** — reproduced exactly from `tools/rootsuite/baseline.json` |
| tutorials | baseline 558 PASS / 1189 (47 %); my environment 544 |
| tutorial speed | 5 858 s of tutorial wall time, 3 000 s of it ten timeouts at the 300 s limit, then a dozen tutorials of minutes each; the ROOT-relative ratios this once quoted were retired (see §4) |

The one-sentence verdict: **the architecture is sound and unusually well
engineered; the code is the equal of anything in scientific Python. What stands
between here and "the whole suite passes" is breadth of ROOT's API surface, a
handful of deep engine semantics (I/O/reflection, ownership, RooFit caching),
and a performance tail — not the design.**

## 2. What the codebase actually is

Eighteen internal components with an enforced, acyclic import graph
(`[tool.xrdroot.components]` in `pyproject.toml`, checked by
`tools/components.py`, and each component imported in a fresh process with
everything above it blocked in CI):

| layer | what it owns | size |
|---|---|---|
| `cppstd` | C++ value semantics: containers, strings, streams, `printf`, `<cmath>` | 4 721 |
| `core` | arrays, numerics, statistics, random (ROOT-bit-exact), compression, buffers, display/plot dispatch | 7 600 |
| `hist` | histograms, graphs, profiles, formulas, fitting | 15 709 |
| `io` | ROOT file format, streamers, trees, writers, RNTuple, merging, `TTree::Draw` | 18 083 |
| `rdf` | RDataFrame execution, sources, actions, distributed | 6 662 |
| `graphics` | ROOT's canvas rasteriser + fonts + matplotlib/plotly/bokeh backends | 13 015 |
| `roofit`, `roostats`, `histfactory` | RooFit, RooStats, HistFactory | 38 103 |
| `tmva` | TMVA learners over scikit-learn/SciPy/XGBoost/PyTorch | 15 138 |
| `pycxx` | **the C++ translator**: lexer, preprocessor, parser, type inference, templates, emitter, GoogleTest runner, translation cache | 15 645 |
| `pycint` | ROOT's dialect of that translator, `.x`/`.L`/`ProcessLine`, the CINT prompt, dictionaries, ACLiC | 6 165 |
| `pyroot` | ROOT's names and behaviour — 845 exported names over 38 submodules, 256 files | 48 539 |
| `cppyy`, `app` | `import cppyy` without cppyy; the facade and `xrdroot` CLI (17 subcommands) | 5 360 |

Strengths that should not be traded away:

- **The translator is real.** `pycxx` parses C++ and emits Python: templates are
  instantiated per argument set, `if constexpr` compiles only the branch it
  chooses, NRVO and pointer/cell semantics are modelled, `.o`-free ACLiC
  libraries are `libNAME.xrdlib`. It is the crown jewel, and it is the reason
  "run ROOT's own tests, not only its tutorials" is credible at all.
- **The refusal discipline.** Every gap is a named, full-sentence
  `UnsupportedFeatureError` or `AttributeError("ROOT has X; xrdroot.pyroot does
  not yet")`. That is why the harness can rank causes automatically, and it is
  the single most valuable cultural decision in the repository.
- **Measurement first.** `tools/rootsuite` (5 kinds, CMake-driven discovery,
  fixtures, `.ref` comparison, ratchet) and `tools/tutorials` (oracle-cached
  differential runs, stdout/file/image comparison, timing, API inventory) are
  better than most C++ projects' own harnesses.
- **Boundaries that hold.** 11 519 import edges, zero undeclared, one reviewed
  private import (`iminuit._core`), and providers (`xrdroot.plugins`) rather
  than upward imports for the few dispatch points.
- **Test economy.** 11 270 assertions-heavy cases in one minute is what makes a
  100 % line-and-branch coverage gate survivable.

## 3. Parity: where the failures are, by cause

### 3.1 roottest (2 173 tests, 1 038 PASS)

Causes, from the run's own reasons (a test counts once under its first cause):

| cause | tests | where it lives |
|---|---|---|
| `TClass` reflection: `GetClassInfo` (45), `GetMethod` (4), `GetMissingDictionaries` (4), `GetListOfEnums` (3), `BuildRealData`, dictionaries, `TInterpreter.GetMapfile` … | ~75 | needs a **reflection layer** over `pycint`'s declared classes — the single biggest roottest blocker |
| **byte-exact written-file layout**: `Disk size of hsimpleXXX.root should have been N but is M` (16 tests, `filemerger` and `testSimpleFile`), `Read too few bytes`, `file too small` (2) | 18 | `writer.py`, `wtree.py`, `wpacking.py` — the same defect as the tutorial `ROOT file differs` group |
| C++ object lifetime/ownership and reflection of heap state (`TObjArray` ownership, destructor-on-delete, `kIsOnHeap`, persistence flags) | ~12 | `pycxx/runtime/objects.py`, `cppstd` cells, `interp.py` |
| `TTree::Print`/`TObjString`/dictionary print output the test compares line by line | ~5 | `pyroot/trees`, `interp.py` |
| tree column typing: `int() argument must be …`, `'event' holds float64 values, and these are object`, `.Class`/`.typedef`/`.autodict` not `gROOT->ProcessLine` commands | ~55 | `interp.py`, `pyroot/trees/*`, `pycint.py` line commands |
| `TTreeFormula` vocabulary: `GetNtrack()` etc. — a branch's *member functions* in a formula | ~25 | `formula/`, `treedraw.py` |
| missing ROOT names (`TEnum`, `TTreeFormula`, `TBufferJSON`, `gDebug`, `TClassEdit`, `TSelector`, `std::complex`, `std::forward_list`, `std::multiset`, `vector`, `fgets`, `TPython` …) | 153 | `pyroot/**` — mostly small |
| engine refusals (`TGeoManager::Import` GDML, `TTree::BranchRef`/`TRefTable`, `TArrayI.fArray`, `std::array<int,?>` writing, `TClonesArray` splitting, `DataBlock` layouts) | 151 | `io/`, `geom/`, `interp.py` |
| stdout/stderr differs from the committed `.ref` | 96 | see below |
| cascades (`A fixture it requires was not set up`) | 101 | disappears as the 45 root causes above are fixed |
| translator refusals | **20** | `pycxx` — *not* the bottleneck |
| timeouts | 5 | `io/perf/ReadBuffer*`, `WriteBuffer*`, `testSetAddressLoop` |

Two facts worth internalising: **the translator is not the constraint** (20 of
1 125 failing roottest tests), and **four directories hold 91 % of the failures**:

| area | tests | PASS | FAIL/DIFF/UNSUPPORTED |
|---|---|---|---|
| `root/io` | 770 | 386 | 384 |
| `root/tree` | 293 | 97 | 196 |
| `root/meta` | 365 | 183 | 182 |
| `root/treeformula` | 161 | 39 | 122 |

The 96 DIFFs are real semantics, not formatting noise. Examples read from the
logs:

- `cling/const/run1`: ROOT prints `with const MyClass4*`; xrdroot prints
  `with MyClass4*` — the type-name printer drops `const`.
- `cling/function/runoverloadResolution`: `11554121` then `11654121` vs ROOT's
  `11654121` twice — an overload-resolution error in the translator.
- `cling/other/runsnprintfselect`: the positional format `(%3$i-%1$f)` is
  printed literally instead of being formatted — `%n$` positionals are missing
  from `cppstd`'s `printf`.

### 3.2 gtest (350 programs, 52 PASS; 507/2020 cases)

The weakest kind, and the one whose causes the harness already extracts per
case. Clusters: GoogleTest facilities xrdroot lacks (`ROOT::TestSupport::FileRaii`
blocks 17), translator gaps in test bodies (mutable lambda capture, `std::string::begin()`,
`std::accumulate` with a custom op, `sizeof` of unknown types, "changing this in
the middle of an expression"), and absent namespaces (`ROOT.Detail`,
`ROOT.Internal.RDF/RPageSource/…`, `ROOT.std.true_type`, `ROOT.MultiProcess`).

### 3.3 pyunittest (129 files, 78 PASS; 439/515 cases)

Only 74 cases fail, and they are concentrated:

| cases | cause |
|---|---|
| 24 | `pyroot/trees/addresses.py:77` — `View.get()` calls `.item()` on a NumPy scalar; `test_array_branch` and the `SetBranchAddress` family |
| 7 | `RNTupleWriter` has no `__enter__`/`__exit__` (context-manager protocol) |
| 6 | `xroofit_python`: `TH1F.SetContent` missing |
| 5 | RDF `FromNumpy` reference-count/lifetime tests (deliberate Python-side difference) |
| 4 | `ROOT.Internal.RDF` (`ChangeEmptyEntryRange`), `Detail.RDF.GetMergeableValue`, `RDatasetSpec.REntryRange` |
| 3 | `ROOT.__name__`, `PyConfig.IgnoreCommandLineOptions`, lazy `gDirectory` sentinel |
| 2+1+1 | `TGraphAsymmErrors.GetEXlowd`, `RooStringVar`, `RNTuple` nested vectors |
| 1 | `leftTrim`/`rightTrim`/`skipFirstNLines` missing from `FromCSV` |

That is roughly **a week of work for 300+ additional passing cases** across the
kinds (each of these also unblocks roottest cascades).

### 3.4 `test/` and module (`other`) tests

Dominated by `ROOT.RooUnitTest` (5), element actions (`TElementPosActionD`
etc., 5), `fgets`, GenVector classes not yet written (`RotationZ`, `Matrix4D32`,
`BoostX`, `DisplacementVector2D`), `TMath` names (`IParamMultiFunction`,
`IMultiGenFunction`, `Polynomial`, `WrappedParamFunction`, `GoFTest`,
`IntegrationOneDim`), and TMVA's PyMVA/DL surface.

### 3.5 Tutorials (1 189; baseline 558 PASS, my run 544)

Blockers, ranked by how many tutorials each alone unblocks:

| reason | tutorials |
|---|---|
| **timeout** (df101 C++/py, df103 C++/py, df104, df105 C++/py/rcanvas, df107, hist102, hist103, tmva100) | 10 |
| `stdout` differs (RooFit/RooStats log order and values, `NCalls`, minuit status, TProfile stats lines) | 84 |
| `image differs` (similarity < 0.95; mostly 0.84–0.95, `tmathtext` 0.56, `palettes` 0.74–0.86) | 21 |
| `ROOT file differs` (hsimple's `hprof` errors differ by 2·10⁻⁶ relative, in 2 of 102 bins; tree201's file is 103 kB larger; double32; tree105) | 13 |
| missing `RooLagrangianMorphFunc` | 4 |
| missing `fgets`, `RDF.Experimental.FromSpec`, `RDF.SaveGraph`, `RooJeffreysPrior`, `RooMomentMorphFuncND`, `RooPolyFunc`, `RooSimWSTool`, `RooStats::SPlot`, `TDOMParser`, `TF1Convolution`, `TFoamIntegrand`, `TGraphTime`, `TH1F.SetContent`, `TMultiLayerPerceptron`, `TPrincipal`, `TGLParametricEquation` | 2 each |

25 baseline PASSes were lost in my environment: 4 are the timeouts above; 8 are
RooFit/RooStats stdout DIFFs (an unpinned NumPy/Python changes last digits and
log ordering); 2 are `RNTupleProcessor`; the rest are file/stdout differences.
This is an **environment-reproducibility** finding, not a regression finding:
the parity numbers must be taken in a pinned environment, and CI does not pin
one for these runs (it does not run them at all — see §8).

## 4. Performance

Two different stories.

**Throughput is good.** The tutorial suite's own total is 5 858 s, of which
ten timeouts at the 300 s limit alone account for 3 000 s. Reading a
2-column tree: 34.7 M rows/s. `RDataFrame.Filter/Define/Mean` over 1 M rows:
32.9 M rows/s. `TH1::FillN`: 38.5 M rows/s.

**A note on the ratios this section used to quote.** Every tutorial was once
measured against ROOT's own run and the ratios committed in
`baseline-timings.json`. That measurement was retired on purpose: xrdroot aims
to be faster than ROOT ultimately, so a ratio is a target that moves whenever
either side does. Nothing committed compares the two sides' wall times now, and
`tools/tutorials` ranks the slowest tutorials by xrdroot's own seconds.

**The tail is bad, and it is concentrated:** the ten timeouts, then
`tmva100_DataPreparation.py`, `df101_h1Analysis.C` and `.py`, `df103` and
`hist103`, `permute.C`, `perceptualcolormap.C`, `fit2dHist.C`,
`schroedinger_hydrogen.C`, `rf801_mcstudy.py` and `rs701`, each taking minutes
where the rest of the suite takes seconds.

Three distinct root causes, each with a clear owner:

1. **RDataFrame with C++ functions declared through `gInterpreter.Declare`.**
   The batch evaluator (`rdf/expression.py`, `pyroot/rdf/batched.py`) vectorises
   simple expressions; a declared function falls back to `pyroot/rdf/entrywise.py`
   — one Python call per entry. That is exactly df101/df103/df104/df105/df107,
   hist102/hist103 and tmva100. Vectorising jagged-array `ROOT::VecOps` bodies
   (and translating declared functions into batched form) is the single largest
   speed win available: it is worth about ten tutorials and, with them, the
   larger part of the suite's wall time.
2. **RooStats toys and RooFit likelihoods:** evaluation count, caching and
   integral reuse — `roofit/fitting`, `roostats/toys`.
3. **Canvas text and palettes:** `canvas/fonts`, `canvas/raster`, `palette.py`
   — `perceptualcolormap`, `schroedinger_hydrogen`, `tmathtext` also have the
   *worst image similarity*, so they are the same defect seen twice.

A fourth, cheap one: **the default write compression level does not match
ROOT's**. `compression.LEVELS["zlib"] = 6`, while ROOT's `TFile` default is
101 — zlib **level 1** (verified with the oracle: `GetCompressionSettings() ==
101`, `GetCompressionLevel() == 1`). Writing 1 M rows × 2 columns:
1.8 M rows/s at level 6, 4.0 M rows/s at level 1, 9.1 M rows/s uncompressed.
Matching ROOT's default is one line and a ~2× write speedup, and it removes a
silent file-size difference.

## 5. Architecture: what to keep, what to change

Keep: the component graph, the refusal discipline, the `_xrd` wrapper idiom
(thin ROOT classes over plain engine objects), the translator's split into
`pycxx` (C++ and the standard library only) and `pycint` (ROOT's dialect as
data), the provider registry, and the harnesses.

Change, in order of leverage for the eventual split:

1. **Make the transport a plugin.** `core` still hard-depends on `xrdclient`
   (`errors.py` subclasses `xrdclient.errors.XRootDError`; `file.py` calls
   `xrdclient.io.open_url`; 7 modules total). Before any split, `xrdroot.errors`
   must own its base class and `file.py` must open through a declared provider
   with a plain-`open` default. This is the only real coupling left (AGENTS
   item 10) and it is a day's work.
2. **Give `pyroot` an internal shape that matches the future packages.** It is
   one flat namespace of 845 names over 41 submodules; the directory layout is
   already by area (`core/`, `trees/`, `graphics/`, `rdf/`, `roofit/`, …), so
   what is missing is a per-area `__all__` contract and a rule that no area
   imports another area's internals — the component checker currently sees
   `pyroot` as one bucket.
3. **Generate type stubs and an API reference.** The ROOT name surface is
   dynamic (`__getattr__`), so IDEs and `mypy` users see nothing. The API
   inventory already enumerates ROOT's classes and methods
   (`tools/tutorials/api-coverage`); emitting `pyroot/__init__.pyi` from it (and
   from `SUBMODULES`) would give completion, catch renames, and give the
   API-coverage report a second, independent consumer.
4. **A reflection layer, not 45 stubs.** `TClass.GetClassInfo` is the biggest
   single blocker in the whole non-tutorial suite. Implement `TClass` over what
   `pycint` actually declares (a real dictionary registry: class → members,
   methods, enums, base classes, streamer info), with `GetClassInfo` returning a
   live object once the class is interpolated. That one design also serves
   `TClass::GetMethod`, `GetListOfEnums`, `TBufferJSON`, `TClassEdit`,
   `TEnum`, `TSelector`, `GetMissingDictionaries` and the rest of `root/meta`
   (182 failing tests).
5. **`TTreeFormula` needs to see the tree, not only a name list.** 122 failing
   `root/treeformula` tests are the largest single-area block; most are
   `GetNtrack()`-style member calls on branches and `TTree::Draw` features.
6. **RNTuple is at 14 of 195 classes** in the API inventory: records, nested
   collections, variants, low-precision floats and `RNTupleProcessor` are the
   named gaps. It blocks io/ntuple (15), the RNTuple tutorials and 7 pyunittest
   cases.
7. **Delete the `xrdroot.cint` compatibility shim** at the next major release,
   and stop mirroring ROOT's private Python (`pyroot/_distrdf/*`): those are
   implementation details of ROOT that will change under the test suite.
8. **Cap the translation cache.** `~/.cache/xrdroot/cint` is 771 MB here and
   grows without bound; an LRU with a size limit and `xrdroot cache prune` is a
   small, honest addition.

## 6. Engineering practice: gaps to close

Already modern and good: hatchling + `pyproject` single source, ruff, strict
mypy, `py.typed`, pytest-xdist, coverage with branch, mkdocs-material + strict
build, a wheel-install smoke test, CI matrix 3.10–3.13 + macOS 3.12, component
isolation jobs, maintainability checker.

Missing, in order of how much they would save:

1. **CI does not run the parity harnesses.** The suite and tutorial runs — the
   *actual objective* — are only run by hand. A nightly job (Linux, one kind per
   job, `--jobs 4`, ratchet against the committed baseline) plus a PR job over
   the ~200 cheapest tests would make "no PASS may be lost" enforceable instead
   of aspirational.
2. **The parity environment is not pinned.** Today's 25 "lost" tutorials are an
   environment artefact. A `requirements-parity.txt` (or `uv.lock`) plus a
   documented interpreter pin for the harness runs is required before any
   ratchet result means anything.
3. **No performance gate.** Nothing fails when the tutorial tail grows, and the
   ROOT-relative ratio baseline this once proposed to ratchet has been retired
   on purpose (§4). An absolute ceiling — the 300 s timeout, with
   `tools/tutorials`' ranking of the slowest tutorials by xrdroot's own time —
   and a micro-benchmark of tree read/write/RDF would keep the tail from
   growing.
4. **`ratchet` cannot judge a partial run.** `python -m tools.rootsuite ratchet
   RUN2/results.json` on a `--kind roottest` run reports 148 "regressions" that
   are only kinds that were not run. It must scope itself to `meta.kinds`.
5. **Harness reasons should name the first difference.** `"standard output
   differs from file.ref"` costs a human a log dig per test; 96 roottest DIFFs
   and 84 tutorial DIFFs would triage themselves with
   `refdiff`'s first-differing-line text (the tutorial harness already does
   this per file; the rootsuite judge does not).
6. **Discovery covers 3 714 of ROOT's 3 820 CI tests** (gtest 350/389,
   roottest 2 173/2 210, test 59/64, tutorials 894/918, pyunittest 129/109).
   The gap should be explained by name in the report, not left as a silent 3 %.
7. **The API-coverage report counts GUI libraries.** It is 100 %-plus noise
   (`Gui 0/213`, `Eve 0/197`, `RGL 0/133`). Scope it to the non-GUI libraries
   the suite uses, as the definition of done says, and publish the per-library
   numbers on the docs site.
8. **Dependency hygiene.** `xrdclient` is pinned to a git SHA (and pulls
   `xrdcore`); publish it, pin a version, and add dependabot/renovate. No
   `CHANGELOG`, no `CONTRIBUTING`, no `pre-commit`, no release automation yet.
9. **One exact-value test asserts ROOT's last bit without the
   `ROOTS_MACHINE` guard** (`tests/test_roofit_branch_chisquare_gamma.py::test_the_chisquare_likelihood_takes_the_kernels_digits`:
   13.972970374276391 vs …639 on arm64). It should use `refmachine`'s rule like
   the other 18 files that do.
10. **Housekeeping.** `docs/history/agent-status-2026-10.md` (805 lines of audit
    narrative, including multi-hash acceptance chains) contradicts the working
    practice in AGENTS.md and should be archived or deleted; AGENTS.md's
    "What remains" list is stale (fopen, gVirtualX, TTask, THnSparseD now
    exist) and should be regenerated from the harness reports, which is what it
    claims to be.

## 7. Risks to the goal

| risk | why it matters | mitigation |
|---|---|---|
| **100 % line+branch coverage on 109 000 statements** | it is a real tax on every new ROOT class; the suite is fast enough that it survives, but it pushes work towards test-shaped code and it is why 45 `# pragma: no cover` exist | keep it for engines; consider diff-coverage for the `pyroot` adapters, which are exercised end-to-end by the harness anyway |
| **Breadth of ROOT's API** | 153 roottest tests + most tutorials fail on a *name* that does not exist; there are ~2 000 more in the API inventory that no test yet reaches | the API inventory should drive a generated skeleton (refusing constructors, correct signatures) so names never cost a test |
| **RooFit/RooStats semantics** | 84 tutorial stdout DIFFs are evaluation counts, cache keys and log ordering; these are the hardest to fix and the least visible | treat RooFit output parity as its own workstream with its own ratchet (rf* tutorials) |
| **Reflection/dictionaries** | 182 `root/meta` + 45 `TClass.GetClassInfo` tests, plus ACLiC and `TBufferJSON` | one coherent dictionary registry, built once |
| **Perf tail** | 10 tutorials cannot pass while they exceed 300 s, and the slowest of them takes minutes | batched RDF evaluation first; never raise the timeout |
| **Oracle availability** | the oracle is the only definition of "correct", and a macOS-only 2.3 GB conda install is fragile | document a CI-installable oracle (Linux conda-forge) and version-key the cache (already done) |

## 8. A sequenced route to "the whole suite passes"

Each wave is written to be independently shippable, ratcheted, and measurable in
the existing harnesses. Counts are the tests each wave is expected to move.

**Wave 0 — make the number trustworthy (1 week).** Pin the parity environment;
add the nightly rootsuite+tutorials CI job with ratcheting; make `ratchet`
kind-aware; put the first-difference text in every DIFF reason; re-baseline on
the pinned environment. *No behaviour change; this is what makes every later
"PASS" mean something.*

**Wave 1 — cheap, wide wins (2–3 weeks, ~400 tests).**
`TClass` reflection registry + `GetClassInfo` (~75); the missing ROOT names
that block 3+ tests each (~60); `RNTupleWriter` context manager, `TH1::SetContent`,
`TGraphAsymmErrors.GetEXlowd`, `FromCSV` trim options, `RooStringVar`,
`ROOT.__name__`/`PyConfig`, `Internal.RDF`/`Detail.RDF` (~20 pyunittest cases);
the `addresses.View.get()` scalar bug (24 cases); `printf` `%n$` positionals and
the `const`-qualified type printer (2 roottest DIFFs); zlib default level 1.

**Wave 2 — the tree/IO/meta block (6–10 weeks, ~600 tests).**
`TTreeFormula` member calls and `TTree::Draw` (~122); tree column typing and
`gROOT->ProcessLine` line commands (`~55`); object ownership and heap flags
(~12); `TGeoManager::Import` GDML, `TRefTable`/`BranchRef`, `TArray*::fArray`,
`std::array`/`TClonesArray` splitting (~151 refusals, in descending frequency);
and the **byte-exact layout** of what the writer produces — ROOT's basket
sizes, `TKey` padding, free-segment records and same-level compression — which
is 18 roottest tests, the 13 tutorial `ROOT file differs` results, and the
prerequisite for the round-trip matrix AGENTS.md asks for. This wave is where
the project crosses 60 % of roottest.

**Wave 3 — performance (3 weeks, 10 tutorials + the tail).**
Batched/vectorised evaluation for declared C++ functions and `VecOps` over
jagged arrays (df101/103/104/105/107, hist102/103, tmva100); RooStats toy and
RooFit integral reuse; canvas text/palette speed with image similarity in the
same tests. Exit criterion: no tutorial over 150 s on this machine.

**Wave 4 — RooFit/RooStats parity (ongoing, 84 DIFFs).**
Work through `roofit/roofit` and `roofit/roostats` in dependency order
(caching → normalization sets → integrals → toy ordering), with the tutorial
`.ref` as the specification. Add the missing RooFit classes by popularity
(`RooLagrangianMorphFunc` 4, then the pairs in §3.5).

**Wave 5 — gtest (the weakest kind, 209 programs).**
GoogleTest facilities (`ROOT::TestSupport::*`), then the translator's remaining
test-body gaps (mutable captures, `std::accumulate`, container iterators — each
worth 2–5 programs), then the `ROOT.Internal.*` and `ROOT.MultiProcess`
namespaces.

**Wave 6 — split readiness (parallel, 2 weeks).**
Transport plugin and own error base; per-area `__all__` contracts in `pyroot`;
generated `.pyi`; `cint` shim removal; cache cap; then the sub-project split
along the component graph — in that order, because each of those is a
precondition for a package that can be installed alone.

**Definition of done, made measurable.** Today's statement ("every test PASS,
or OUT-OF-SCOPE/UNSUPPORTED with a full-sentence reason") is right but not yet
checkable in CI. Concretely: (a) the nightly ratchet is green on the pinned
environment, (b) every FAIL/DIFF has been triaged into a cause with a test count
in the report, (c) the API-coverage report — scoped to non-GUI libraries — is at
100 % present for every class the suite touches, (d) no test is
UNSUPPORTED that a named plan item would make pass.

## 9. Two notes on the record

- The repository's own history (`docs/history/agent-status-2026-10.md`) records
  a large amount of gated-commit bookkeeping — audit hashes, per-prefix gate
  proofs, resource unions. The engineering in it is real, but AGENTS.md now says
  not to build that machinery, and the file's presence reads as the opposite.
  The evidence that matters is `results.json` and the ratchets.
- `tools/rootsuite/baseline.json` and `tools/tutorials/baseline.json` reproduce
  exactly on this machine (1 185 / 2 820 and, modulo environment, 544 vs 558).
  That reproducibility is the project's strongest asset; everything in Wave 0
  exists to protect it.
