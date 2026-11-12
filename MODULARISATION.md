# Modularising xrdroot within the project

Status: implementation in progress, 2026-10-22. This revision replaces the
2026-10-18 multi-distribution cutover plan. The user has authorized internal
modularisation in this single project. Work starts from accepted local
`8a008d88` in isolated worktrees. The unchanged-parent full tutorial baseline
runs separately from candidate preparation; each implementation commit still
requires its own validation. The first runtime attempts exposed a facade
import cycle and compatibility-shim typing errors; both are retained and being
corrected. The corrected facade passes all six gates and its 79 affected
tutorial comparisons lose no current-parent PASS. The complete refactor and
ordered component prefixes still need validation. The first group is committed
locally as `720d178d`; global tutorial acceptance and main publication remain
held. The component-contract group is also committed locally as `c0b9d09a`
after all six gates and independent audit. The corrected complete source
`7c01ca5b` passes all fifteen public component imports, 134 component-contract
tests and all 61 isolated cppstd tests. All six gates now pass on this exact
source: both test modes record 9,771 passed, one skip, 46 expected failures
and 100% line and branch coverage. Independent audit `69028760` accepts those
gates; ordered-prefix and tutorial acceptance remain separate. The prior
`ee093e1b` attempt stopped at a test helper's complexity violation; that
failure is preserved, and the successor splits the helper without changing
assertions. The shared-primitives group `6b5581d5` passes all six gates. Its
completed 297 affected comparisons record 219 PASS, 19 DIFF, 25 FAIL, 32
UNSUPPORTED and two ORACLE-FAIL. Eight strict parent PASS obligations remain
lost: Python df103, ntpl004, df014, df015, df029, df030 and both hist039
variants. Captures retain the timeout and remote-data failures; independent
audits accept capture integrity only. Native ROOT also fails to open ntpl004's
original URL with HTTP 401. No data replacement, timeout change or parity
exception is granted. The first joined-report builder fails on a metadata
schema assumption and is preserved; corrected scripts now pass independent
saved execution and review (`eb01f17c`). The first copy-helper raw-file
parsing failure remains preserved. Its reviewed correction successfully copies
the exact gated tree, now committed locally as `83dbc8f3`; all eight tutorial
losses remain open. A local gated implementation commit remains distinct from
accepting tutorial outputs or satisfying definition-of-done item 7. Main
publication stays held.

The next local implementation commit is `8fd86c90`: seven adapters import I/O
engines directly, and measured component debt falls from 58 to 47. Its exact
1,477-file tree matches gated source `f226c8ef`; commit receipt `051fc575`
retains all thirteen failed PASS obligations from its 321-path capture. The
branch is clean and unpublished.

The notebook-provider group is committed locally as `14ecd34a`. Its exact
1,482-file tree matches gated source `0c684cec` and record `80ce91a3`.
All 356 affected captures are independently reviewed; all 217 immediate parent
PASSes survive and thirteen older PASS obligations remain lost. Main is unchanged.

All 25 later source prefixes are prepared and each needs its own validation.
Their first inventory batch completes 16 inventories, then exceeds the fixed
40-second checker bound at printf; eight later inventories remain unrun. Its
failed capture is independently audited and preserved. A fresh quiet full
batch now finishes all 77 stages successfully with all 25 fresh inventories
and the same bound. A saved-auditor raw-log parsing failure is retained; a
narrow validator successor passes actual saved review `aeabe3b7` with 123,872
checks. All measured transitions add zero violations; reviewed debt reaches
zero at session. After explicit allowance review, all 25 baselines are
materialized in separate source-only worktrees (`603327d7`). No partial
first-batch evidence is pooled into it. Every ordered prefix still requires
its own gates and affected comparisons. A draft saved-gate auditor was
rejected before actual use for missing child-proof validation. Its corrected
source now accepts the completed L7 and L2 six-gate captures (`358496b2` and
`c631d9c1`). Both modes pass with strict 100% line and branch coverage: 9,495
tests for L7 and 9,503 for L2. Their affected comparisons and local commits
are recorded below for L7 and remain pending for L2. These source-specific
gates confer no tutorial acceptance.

The persistence prefix also passes its own six gates: both modes record
9,536 passed with strict 100% coverage, independently audited as `cd33607e`.
The math prefix completes all gate stages with exit 0 and 9,622 passed in
each test mode, with strict 100% coverage; saved audit `3d01f570` passes. The
L7 affected capture finishes with all four quiet comparisons failing and
controls recording 217 PASS, 24 DIFF, 32 FAIL, 42 UNSUPPORTED and two
ORACLE-FAIL. Controls audit `dc69b047` passes and retains nine control PASS
losses. The original joined helper fails on missing constants and is preserved;
the corrected helper passes independent source review and saved execution,
then independent reconstruction (`76c10db1`). All 321 paths and thirteen
failed PASS obligations remain recorded. These
captures do not discharge lost PASS obligations. All remaining gate recipes
are prepared, with the unchanged limits and coverage requirements. The next
two full gate jobs, graphics and RooFit, now pass their own six gates and
independent audits `3018ae98`/`47a05814`. Both modes record 9,627/9,643
passed with strict 100% coverage. The notebook/plugin prefix's four quiet
tutorial comparisons now run with both slots reserved. Their actual preflight
passes for 356 selected paths and 358 native captures. All four comparisons
fail: three 300-second timeouts and ntpl004's format error. Independent audit
`778cbf26` accepts capture integrity only and retains every PASS obligation.
The reviewed controls binding succeeds; controls352 and the seventh prefix's
gates now run on separate slots. Both seventh-prefix test modes finish with
9,650 passed and strict 100% coverage. Typing and documentation pass;
independent saved audit `bb5ae33e` accepts the source-specific six gates.
The streams/strings prefix also passes its own six gates and independent audit
`92b9f7f9`, with 9,655 passed and strict 100% coverage in each mode. Fresh
peer-aware checks release arithmetic into slot 1 as `3fc4e65c`. L2 controls
then complete and pass saved audit `a96a0a09`; the original joined 356-path
capture passes independent review `0cf6a002`. All 217 immediate parent PASSes
survive, while thirteen older PASS losses remain open. The copy-only decision
passes independent review and the exact source is committed locally. Arithmetic
also passes its own six gates and audit `2debdeaf`: both modes record 9,658
passed with strict 100% coverage. Both preceding heavy jobs are complete.
Persistence source review `e0d110a2` passes; fresh resource release `a063177e`
starts its original quiet4 with both slots reserved. Its 593-path/596-native
catalogue/cache preflight passes (`71f8ab28`). All four comparisons fail:
three 300-second timeouts and ntpl004's invalid-data error. Independent audit
`ed101084` accepts capture integrity only. Controls binding passes; fresh
checks accept 8,619 known completed births, and release retains 8,621.
Step 10 passes its own six gates and audit `cf799368`, with 9,664 passed and
strict 100% coverage in both modes. Step 11 then passes all six gates and
audit `099b208c`, with 9,664 passed and strict 100% coverage in both modes.
Step 12 now passes its own six gates and independent audit `c0c40bb3`, with
9,664 passed and strict 100% coverage across 92,419 statements and 22,598
branches in both modes. Twelve ordered prefixes are gated. Controls589 remain
in slot 0; fresh peer checks retain 9,080 known births and release step 13
in slot 1 (`71e56698`).
Step 13 also passes its own six gates and audit `31a000d4`: both modes record
9,664 passed, one skip and 46 expected failures, with strict 100% coverage
across 92,435 statements and 22,596 branches. Thirteen ordered prefixes are
gated. Fresh peer checks retain 9,248 known births and release step 14 in
slot 1 (`4581af94`), while controls589 continue in slot 0. The L2 ledger
extension `80435745`, independently reviewed as `443d22f4`, preserves all
266 observed L2 PASSes without adding paths to the 563-path union. Current
L3 capture inputs remain bound to their original immutable ledger.
Source-only review `ab479ced` verifies runtime, packaged assets and tutorial
harness invariance across rows 22–25. ROOT's scope decision `78b6d3a7` selects
no additional tutorial paths for those test, CI and documentation deltas;
all parent PASS obligations, own gates and final full validation remain required.
The next math scope is being widened to the exact 659-path union of the
original math and persistence scopes so every possible immediate-parent PASS
is selected. Its original 405-path proposal remains preserved; no successor
recipe has runtime authority yet. Independent source review `6a6159e2` passes
250 checks for the 659 selected paths and 662-path dependency closure, with
all future predecessor and runtime authorities still NULL.
Proposed successive scopes for steps 05–07 select 688, 688 and 730 paths by
including each predecessor's full selection and the group's original paths.
Their independent source review and runtime authority remain pending.
Conservative 893-path stdlib planning retains 296 separate SKIPs and all 563
existing PASS obligations; sealed sibling inputs pass source review. Each
group still needs its own actual predecessor and preflight evidence. Scope and
global acceptance remain false. Later gate recipes remain held. At arithmetic release CPU speed reports 72%;
limits and acceptance rules are unchanged.
Each later comparison must explicitly retain its immediate local parent's
actual PASS captures alongside the existing baseline and accepted-scope views.
An unaccepted whole scope does not waive its individual PASS obligations.
The final full-harness ratchet retains those earlier PASS obligations as well
as the committed baseline; old execution recipes need exact-source and parent
evidence rebinding before they run.

Latest coordinator status: six local implementation commits end at `4fe45f16`
(persistence providers), with clean exact source `29f56c7a` across 1,484 files
and commit record `aff94b1d`. Its 593 affected comparisons preserve all 266
immediate-parent PASSes and retain thirteen older losses. Independent joined
review `6c4af537` passes; tutorial and global acceptance remain false. The
actual L3 ledger `c5547b42` passes independent review `9a29ef3d`, retaining
all nine views and 563 PASS paths. Fifteen prefixes pass their own six gates.
Step 15 audit `55b92801` accepts 30,258 checks, with 9,673 passed in both modes
and strict 100% coverage across 92,536 statements and 22,596 branches. Its
source is unchanged, processes drained and slot released. The math prefix's
four sequential comparisons finish under controller 68477 and release
`d31bb21e` with 10,263 retained births; all seven stages pass their capture
commands, source stays unchanged, processes drain and both slots release. Final family source review
`2544676b` passes; the new catalogue/cache preflight passes in 58.241 seconds
(`d041841a`) and genuine-client provenance passes. All four outcomes are FAIL: Python/C++ df103 and hist103 time out at
300.193/300.096/300.153 seconds; ntpl004 exits 2 after 6.983 seconds without
timeout. Independent saved audit `3f172a86` passes 17,296 checks for capture
integrity only. All old PASS obligations remain open. Actual controls binding
`5d2e7855` passes independent 78-check review `82549ddc`. A fresh resource
union verifies 10,374 recorded births absent and retains all seven PID-only
obligations; release retains 10,376 births and disjoint slots. Controls FIRST
controller 76032 exits 1 before any stage because its native-reader guard
sees the peer gate job's simultaneous `lsof`. Its failed launch is preserved,
source unchanged, processes drained and slot 0 released. Step16 controller
76088 runs its own six gates in slot 1. A controls retry needs a saved failure
audit and fresh resource authorization. A parse-failed metadata interpreter's unobserved identity remains an
explicit gap (`04101de6`); its exit/error evidence is retained separately and
no identity or absence is invented.
The first math recipe metadata attempt stopped before recipe creation on an
incorrect stage-name expectation; the failure is retained and a finite source
correction and final capture family now pass review. Fixed169 actor review
`b6caef48` finds no additional unmatched identities; the separate metadata gap
remains unresolved, without any complete birth-capture claim.
Scope review `7452b031` and pending-family review `bedfc2f2` pass for rows
05–07, without runtime authority. Earlier status paragraphs are historical;
the latest durable coordinator checkpoint controls active scheduling. Main,
the committed baseline and every publication hold remain unchanged.

## 0. Goal and scope

Refactor xrdroot into components with clear responsibilities and enforceable
dependency boundaries **inside the existing Python project**. Keep one
repository, one root `pyproject.toml`, one installable `xrdroot` distribution,
one `src/xrdroot/` source tree and the existing command-line entry points.

The Python implementation of C++ standard-library behavior is an additional
component alongside core, I/O, histograms, RooFit and the other engines. Its
proposed home is `xrdroot.cppstd`. Here, “stdlib” means C++ facilities
implemented in Python; Python's own standard library is unchanged.

Separating components into multiple Python projects, distributions or
repositories comes later. This phase creates the internal boundaries that
would make that possible. It introduces no per-component packaging metadata,
namespace-package machinery, independent versions or publication steps.

### Definition of done

1. Every source module belongs to one of the fifteen components in section 2.1.
   Existing engine packages can keep their current locations; shared behavior
   moves only where needed to establish the boundary.
2. A component contract checks imports, including imports inside functions,
   and prevents new dependencies outside the declared graph. Existing debt is
   measured on the selected source, then reduced to zero through gated groups.
3. `xrdroot.cppstd` contains the reusable C++ standard-library implementation
   and its supporting value semantics. It imports no ROOT namespace, tree,
   RooFit, translator or transport implementation. ROOT integration lives in
   explicit adapters in higher components.
4. Existing public import paths and object identities remain compatible through
   re-exports where necessary. Installation remains `pip install -e ".[dev,zstd]"`
   and worktrees continue to use `PYTHONPATH=$PWD/src`.
5. Component tests establish their own behavior without higher components being
   pulled in by shared fixtures or import-time side effects. These are tests of
   internal boundaries in the single installed project, not separate-wheel tests.
6. Every implementation commit passes the existing six gates: normal and forced
   OTHER tests with 100% line and branch coverage, Ruff, strict mypy,
   maintainability and strict docs. Python 3.10 remains the floor.
7. Affected tutorials retain their outputs after each group. The full tutorial
   ratchet and translator survey pass at integration; main publication remains
   subject to the global no-PASS-lost rule and green CI.

## 1. Evidence and starting point

The original survey measured `fa34ea9` on 2026-10-18: 815 source files,
168,062 lines, 57 imports outside its proposed component graph, and additional
external-dependency findings. Those are historical planning observations, not
current source counts, gate results or the starting allowance for this revision.
The added `cppstd` boundary changes the graph and requires a fresh measurement.

Current parity status and source-specific acceptance are in `AGENTS.md` and
`~/xrdroot-parity/RECOVERY-STATE.json`'s `latest_coordinator_checkpoint`.
Use those records to select an accepted integration source before beginning
refactoring. Do not harvest, commit, rebase or remove other lanes' worktrees
from the old plan's inventory; their status has changed since that inventory.

The committed tutorial baseline remains `tools/tutorials/baseline.json`.
The October 18 scratchpad run is not a replacement baseline. The original
captures for three Python set-order obligations were lost, and those obligations
remain open under the unchanged no-PASS-lost rule. Refactoring grants no
exception, synthetic replacement or comparison-policy change.

Store before/after source maps, import inventories, reports, logs and oracle
evidence in a unique directory under `~/xrdroot-parity/`. Never rely on `/tmp`
or a session scratchpad for important evidence. Preserve failed attempts.

## 2. Target architecture

### 2.1 Components and allowed dependencies

Component names below describe internal ownership, not distribution names.
“Depends on” lists allowed direct component dependencies, including lazy imports.
The contract must expand the two explicit aggregate entries before checking.
Module assignments are proposed and must be reconciled with the selected source;
line counts from the old plan are deliberately not carried forward.

| Component | Owned areas or proposed changes | Depends on |
|---|---|---|
| `cppstd` | New `cppstd/`: containers, strings, streams, iterators, algorithms, numeric traits, generic C++ arithmetic/reference/value semantics, standard random engines and distributions, chrono and thread facilities already implemented | — |
| `core` | `errors buffer compression stdio limits random stats numerics cxx booking drawspec polybins xmldoc palette delaunay graphsmooth supsmu entries draw display library`; new `plugins jagged rootarrays notebook`; owns `py.typed` | cppstd |
| `hist` | `hist arithmetic compare distribution efficiency filling fillrandom function moments profile reshaping slicing graph graphmath graph2d stacks sparse scatterplot formula fit` | core |
| `spectrum` | `spectrum` | core |
| `fits` | `fitsio` | core |
| `io` | `file streamers objects interp tree leaflist kinds known chain friends winfo writer wbranch wclasses wcolumns wmeasure wobjects wpacking wtree wupdate rntuple merging image remote drawable treedraw scan extending` | core, hist |
| `rdf` | `rdf` | core, hist, io |
| `geom` | `geom`, with pad painting moved to graphics | core |
| `graphics` | `canvas plot`, including geometry pad painting | core, hist, geom |
| `roofit` | `roofit` | core, hist, cppstd |
| `roostats` | `roostats` | core, hist, roofit |
| `histfactory` | `histfactory` | core, hist, io, roofit, roostats |
| `tmva` | `tmva` | core, hist, io |
| `pyroot` | `pyroot cint session`, excluding generic behavior moved to cppstd; ROOT adapters and compatibility re-exports remain here | all thirteen components above |
| `app` | `cli __main__` and the top-level `__init__.py` public facade | all fourteen components above |

NumPy is the only hard dependency needed by `cppstd`. Existing project-wide
dependencies, extras and lazy optional imports stay in the root `pyproject.toml`.
This refactor does not remove xrdclient from the project's hard dependencies or
change exception inheritance merely to simulate independently installed pieces.
It does remove accidental private dependencies such as `xrdclient._compat`
from generic code where Python's public API provides the same behavior.

Key placement decisions:

- Shared `Jagged` and primitive-array descriptions move below I/O so histogram
  formulas need not import the tree reader.
- `library` belongs to core once its `Jagged` dependency moves there.
- `drawable treedraw scan extending` stay with I/O; `fit` stays with hist.
- Geometry modeling stays in geom; pad painting belongs to graphics.
- ROOT's `TRandom` wrappers stay in core. Generic C++ standard random engines
  and distributions move to cppstd, with exact behavior and shared state retained.
- The parser/emitter, ROOT namespace resolution and session remain in pyroot.
  Extracting the generic runtime does not require a standalone translator.
- Each module has one owner. Compatibility shims at old paths belong to their
  old component and depend downwards on the new implementation.

### 2.2 Source and test layout

```text
pyproject.toml                  # existing project metadata, extras and tool configuration
src/xrdroot/
  __init__.py                   # one existing package root; app-owned lazy facade
  py.typed
  cppstd/                       # new internal C++ standard-library component
    ...                         # final module names chosen during the source inventory
  hist.py ...                   # existing engine locations retained where practical
  roofit/ roostats/ rdf/ ...
  pyroot/                       # ROOT wrappers, adapters and compatibility shims
  cint/                         # parser/emitter and ROOT-specific runtime helpers
tests/
  ...                           # existing tests retained; component selection is recorded
  cppstd/                       # focused generic-runtime tests as they are established
tools/ docs/ mkdocs.yml maintainability.json AGENTS.md
```

There is no `packages/` cutover, `pkgutil.extend_path`, multi-root `PYTHONPATH`
or generated collection of `pyproject.toml` files. Mypy and coverage continue
to inspect the complete existing source tree. Check actual file inventory and
coverage attribution after moves rather than asserting an obsolete file count.

Test ownership can initially be recorded in a manifest without moving the
whole suite. Move a test only when that makes its dependencies clearer, keeping
all assertions and collection semantics. Shared fixtures must not force cppstd
tests to import xrdclient, RooFit or RooStats.

### 2.3 The C++ standard-library component

Inventory the generic implementation currently spread across:

- `pyroot/stl.py`, `pyroot/stdlib.py`, `pyroot/bitset.py`, standard random wrappers, and generic
  parts of `pyroot/vectorvalue.py` and `pyroot/vectorassign.py`;
- `cint/runtime/arith.py`, `cells.py`, `references.py`, generic parts of `objects.py`, `strings.py`,
  `streams.py`, `istreams.py`, `printf.py`, `chrono.py` and `threads.py`;
- generic scalar mathematics and chrono literals mixed into runtime helpers.

These are source candidates, not instructions to move each file whole. Resolve
their imports and actual responsibilities before assigning individual symbols.
Keep ROOT units, `RootProxy`, RVec overloads, ntuple filling, tree-reader behavior
and ROOT-specific type aliases in adapters. A generic runtime may accept an
element type or resolver supplied by a caller; it must not discover ROOT by
importing `xrdroot.pyroot` or looking up a hard-coded xrdroot module name.

Required adapter boundaries:

1. **Type resolution.** Replace the STL implementation's opaque-element ROOT
   lookup with an explicit resolver or registered factory. The ROOT adapter
   supplies ROOT classes and aliases. Ordinary standard types work without it.
2. **Vector value semantics.** Generic containers own their copy/assignment,
   reference and storage contracts. ROOT adapters supply RVec and registered
   ROOT value types. Preserve element identity rules, nested copies, reserved
   storage and assignment behavior measured by the native fixtures.
3. **Maths and literals.** Scalar C/C++ maths and standard chrono behavior live
   in cppstd; RVec results and ROOT graphical units stay in ROOT adapters.
4. **Namespace and errors.** Generic facilities accept explicit dependencies
   and expose documented errors. ROOT-facing shims retain existing public
   refusals and names. Neither cppstd nor its import path loads RootProxy.
5. **Streams and state.** Compatibility paths re-export the same `cout`, `endl`,
   stream types, registries and random-engine objects. Moving definitions must
   not create duplicate state or change manipulator/type identity.

Document the supported C++ subset and target numeric model. Preserve current
integer widths, narrowing, float32 stores, promotion, division/remainder,
platform-dependent numerics and standard-library-specific RNG sequences.
Do not present Python object ownership as a native C++ ABI guarantee.

Prove the boundary in a fresh subprocess using the installed single project:
import every cppstd module with higher components and transport imports blocked.
Exercise ordinary containers, references, streams and RNGs without registering
ROOT adapters. Then exercise the ROOT adapters through existing xrdroot tests.
Use native ROOT/C++ fixtures and Linux checks where exact behavior is claimed.

### 2.4 Top-level API and internal extension points

Preserve existing public imports, `__all__`, `dir()` behavior and CLI entry
points. Importing `xrdroot.cppstd` necessarily executes `xrdroot/__init__.py`
first, so its bootstrap must import no higher engine, ROOT namespace or transport.
The app-owned facade resolves public exports lazily; its dependencies belong
to app, not core. L9's inert bootstrap is a prerequisite of strict cppstd
isolation and is its own gated group. All components still ship
together, so no new “pip install xrdroot-<component>” error path is needed.
Optional third-party dependencies retain their existing helpful install messages.

The isolation blocker permits execution of that required parent-package
bootstrap even though the facade is app-owned. This narrow exception permits
no app submodule or higher-component import; resolving a lazy export outside
the selected dependency closure remains blocked.

Where a lower engine needs a higher implementation, prefer, in order:

1. Return an engine object or plain data and let the ROOT wrapper construct
   the public ROOT object.
2. Pass a small callable or resolver explicitly.
3. Use a small internal `xrdroot.plugins` provider table for established
   extension points such as file readers and drawing.

The provider table records component, module and callable. It resolves lazily,
caches only where lifetime rules allow, and has no distribution lookup,
entry-point discovery or absent-component installation behavior in this phase.
The contract checks string targets as well as normal imports. Record each
permitted upward dispatch separately in a reviewed extension-edge allowlist:
caller, extension point, provider component and target callable. Such dispatch
is not an ordinary allowed import edge and does not expand an engine's dependency
closure. Arbitrary dynamic imports or undeclared providers remain violations.
Isolated engine tests inject a fake or downward provider; integration tests
explicitly declare and exercise the higher provider. Both scopes are required.

Candidate extension points retained from the original plan:

| Point | Higher provider | Lower caller |
|---|---|---|
| Notebook plotting and HTML figures | graphics | core display |
| Canvas classes and streamers | graphics | I/O class dressing and decoding |
| RooWorkspace keys | roofit | file reader |
| HistFactory layouts | histfactory | streamer interpretation |
| ModelConfig workspace construction | roostats | RooFit workspace reader |
| Geometry pad painting | graphics | geom adapter |

A `root.class` fallback is a last resort. Record each use as debt with its
call site and reason; do not make engines routinely construct ROOT wrappers.

### 2.5 The component contract

Implement `tools/components.py` with ownership and allowed imports recorded
under `[tool.xrdroot.components]` in the existing root `pyproject.toml`.
The map must support submodules because cppstd is extracted from larger areas.

Checks:

- **C1: ownership and imports.** Every source module has exactly one owner.
  Every import, including inside functions, must follow the declared graph.
  Expand aggregate dependencies and reject dependency cycles.
- **C2: external dependencies.** Record allowed hard and optional imports per
  component. Check lazy optional imports and private third-party dependencies.
  Python's stdlib is distinct from the new cppstd component.
- **C3: adapter targets.** Validate provider-table targets and declared resolver
  registrations against the separate extension-edge allowlist. Reject unlisted
  dispatch; listed dispatch does not permit direct imports of the provider.
  Compatibility shims may point downwards; the generic runtime cannot import
  its ROOT adapters, even conditionally.
- **C4: tests.** Record focused component selections and their support modules;
  integration tests can declare several components. Do not misclassify tests or
  discard assertions to manufacture isolation or coverage.

| Command | Purpose |
|---|---|
| `check` | Report violations by component edge and file/line; reject new violations against the reviewed initial inventory and ratchet resolved debt down to zero |
| `isolate COMPONENT…` | In fresh subprocesses, block imports outside each component's declared dependency closure while importing its modules; includes lazy code paths exercised by focused tests |
| `graph [--dot]` | Report the current component graph and source inventory |

These commands have a prepared implementation, including declared test scopes,
fixture boundaries and reviewed compatibility-module aliases. An independently
reviewed initial inventory measures 70 dependency violations plus one visible,
exactly declared private Minuit import. It is eligible to establish the component
ratchet only on its sealed source and policy. The combined refactor's latest
strict source diagnostic on `33add7a` reports zero undeclared violations across
857 modules and 9,788 import references; the exact private Minuit import remains
visible.
Ruff and bounded AST metric diagnostics also pass. Earlier failures and review
corrections remain preserved. The first complete-tree runtime attempt passes
strict/default component checks, Ruff and maintainability, then stops at 16 shim
re-export mypy errors. The first facade attempt stops at an import cycle in its
fresh-process compatibility probe. Both failures remain preserved; no complete
six-gate acceptance is claimed.
No component packaging or installation commands are needed.
The tool follows the same tests, typing and complexity rules as the project.

## 3. Work breakdown and order

### P0 — select the source and measure boundaries

1. Reconcile the active parity handoff and accepted commits. Preserve other
   candidates and their evidence. Select and seal an integration source;
   no automatic worktree harvesting or cleanup.
2. Save its source inventory, full gate results, tutorial report and ratchet
   outcome, translator survey, import behavior and timings durably. Disclose
   existing failures; they cannot be counted as accepted refactor results.
3. Build the component map and contract tool. Recompute all imports, including
   cppstd's current ROOT/runtime dependencies. The old count of 57 is not a
   fixed acceptance target. Seal the actual starting debt and forbid additions.
4. Record file ownership and explicit handoffs before assigning lanes. Establish
   small provider/resolver interfaces only where the first lane needs them.

### P1 — refactor in place, one gated group at a time

The lanes below describe work groups, not simultaneous execution requirements.
Each group preserves compatibility imports where possible, resolves its named
dependency edges, adds no unrelated violations and reruns affected tutorials.

| Lane | Work and boundaries | Main source areas |
|---|---|---|
| L1: shared primitives | Move `Jagged`, bounds/take helpers and primitive-array descriptions below I/O; update histogram/formula consumers; classify `library` as core; replace private `zip_strict` imports with Python 3.10's `zip(strict=True)` | tree, interp, hist, slicing, writer, formula, library, graph; new jagged/rootarrays |
| L2: notebook display | Move pure HTML/table helpers into core; route plotting/figure behavior through the graphics provider; preserve notebook fallback and public behavior | display, plot/notebook, tree, chain, file, rntuple/reader; new notebook |
| L3: readers and persistence adapters | Resolve canvas classes/streamers, RooWorkspace keys and HistFactory layouts through explicit providers; put ModelConfig construction in roostats; preserve reader precedence and unknown-class diagnostics | kinds, interp, file, canvas, roofit/io, roostats/workspaceio |
| L4: RooFit maths | Move plain ROOT math functions into the hist engine; make formula registration complete without importing pyroot; use cppstd streams for RooFit ASCII I/O after L10's stream move | pyroot/core/rmath, mathformula, function, roofit/pdfs, roofit/data/ascii |
| L5: statistical engines | Replace internal ROOT wrapper construction with engine objects; move public ROOT-object construction into pyroot wrappers; document unavoidable fallbacks | roofit bindings/contour, roostats scans/limits/MCMC, histfactory, matching pyroot wrappers |
| L6: TMVA plotting | Return ROC arrays from the engine; construct graphs, canvases and legends in the ROOT wrapper, preserving public return values | tmva/envelope, evaluating, pyroot/tmva |
| L7: direct engine imports | Replace pyroot imports through the top-level facade with imports from file/writer/wupdate; keep public facade imports working | pyroot files, ntuple, rdf and tree wrappers |
| L8: geometry painting | Move pad painting into canvas; keep scene modeling and optional scene exports in geom without a canvas dependency | geom/paint, view, backends, canvas/paint; new canvas/geompaint |
| L9: top-level imports | Establish an inert facade bootstrap before strict cppstd isolation; resolve exports lazily while preserving names, identity, extension loading, errors and introspection; measure cold imports | xrdroot/__init__, focused fresh-process tests |
| L10: C++ stdlib | Factor generic standard-library/runtime behavior into cppstd in the sequence below; supply explicit ROOT adapters and compatibility re-exports | pyroot STL/stdlib/vector/random facilities, cint/runtime and new cppstd |

L10 is an additional component lane, with several small gated commits rather
than one large move:

1. Inventory symbols and freeze identity/state contracts. Establish the generic
   package without importing the aggregate `cint.runtime` initializer. Coordinate
   L9's inert parent-package bootstrap before claiming cppstd import isolation.
2. Move streams, strings and self-contained helpers. Re-export exact objects
   from old locations. L4 consumes streams here, not a second `cxxstreams`
   implementation in core.
3. Move arithmetic, numeric traits, references and generic value/copy contracts.
   Introduce the minimal registration interface needed by ROOT vector adapters.
4. Move containers, iterators and algorithms; remove opaque-type ROOT lookup
   from generic code. Preserve factories and nested-vector behavior.
5. Move standard random engines/distributions and remaining generic chrono,
   thread and scalar-math facilities. Leave ROOT wrappers, RVec math and units
   in adapters. Keep RNG state and existing exact sequences.
6. Prove fresh-process isolation and native/platform semantics; document the
   supported subset and adapter interfaces. Remove duplicate implementations.

Within each step, choose smaller commits if moving an identity-sensitive cluster
requires separate validation. Do not commit intermediate trees that break gates.
L4 and L10 share stream consumers; L1/L2/L3 share tree and file modules; L10 and
the active reader/vector candidates share runtime semantics. Assign one writer
to overlapping files and merge the prerequisite before the dependent work.

### P2 — strengthen component tests and documentation

1. Classify tests by actual engine/API use and imported helpers. Keep broad
   integration tests explicit rather than forcing every test into one owner.
2. Make generic shared fixtures independent of RooFit, RooStats and transport.
   Move specialized fixtures to appropriately scoped support modules without
   changing their lifetime or behavior.
3. Add component import/isolation checks to CI using the same source tree and
   installed project. Add focused coverage for behavior previously exercised
   only through higher components. Combined coverage remains strict 100%;
   focused component coverage must also meet the agreed strict scope.
4. Document responsibilities, allowed dependencies and examples under
   `docs/components/`; retain `docs/root.md` for public ROOT classes. Examples
   install the single `xrdroot` project with relevant existing extras.
5. Retain one root build and installation smoke check. There is no package
   cutover, namespace-wheel inspection or version synchronization step.

### P3 — integration and acceptance

1. Check zero unresolved component violations and run isolation across all
   fifteen components from the exact proposed tree.
2. Run all six gates on that tree; verify complete source inclusion in typing
   and coverage. Run the translator survey with the unchanged 98% floor and
   no crashes, plus required Linux numerical checks.
3. Run the full tutorial harness and global no-PASS-lost ratchet using unchanged
   comparison policy and 300-second timeout. Investigate timing regressions;
   rerun loaded timeouts alone under the existing policy. Preserve every run.
4. Publish main only when all required acceptance conditions pass. Watch CI
   after any authorized push and repair failures until green. No independent
   component publication is part of this phase.

## 4. Coordination, commits and validation

Use **four agents total: coordinator plus three workers**, with a single serial
integration queue. The machine has **two heavy-job slots**; workers and
coordinator share those slots. Do not schedule four lane test runs plus gates.
Full gates use the existing `-n 4`; focused jobs use the approved smaller scope.
Quiet tutorial/native comparisons may require both slots. Fresh resource checks
precede execution, and long jobs write durable logs under `~/xrdroot-parity/`.

Each lane works in a separate worktree and sets `PYTHONPATH=$PWD/src`. No shared
environment recreation or multi-root editable installation is required. Do not
mutate frozen validation trees or overwrite failed reports.

For each concrete fix/refactor group, the coordinator:

1. Reviews the exact diff, ownership changes and compatibility contracts.
2. Checks the component inventory and absence of new dependency violations.
3. Runs focused tests with strict coverage and the affected tutorial comparisons.
4. Runs all six required gates on the complete candidate tree.
5. Commits that gated tree as `refactor: ...`, then handles the next group.
   Commit, gate and push are separate operations. Do not bundle several groups
   into one commit or claim another source's tests as acceptance.

The global ratchet still controls main publication. A branch CI run does not
waive that rule. Do not update the tutorial baseline to conceal a lost PASS,
raise a timeout/complexity limit, lower coverage or loosen an assertion.

This document revision does not pause tutorial parity. Before starting a
refactoring group, coordinate only the files and integrations it overlaps.
Pause or reschedule an overlapping lane explicitly; there is no blanket
project-wide freeze or automatic change to the active task.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Moving definitions creates distinct stream, factory, registry or RNG objects | Keep one definition and identity-preserving re-exports; test old/new paths and lifecycle behavior |
| Generic vector behavior still imports ROOT indirectly | Check all imports and registration paths; block higher imports in fresh-process cppstd tests |
| Numeric or RNG semantics change during a move | Preserve implementation first; compare native fixtures and platform results before optimizing |
| Shared conftest hides an upward dependency | Split specialized fixtures and exercise focused tests without higher components |
| Providers conceal cycles or add hot-path overhead | Prefer direct downward imports/data/callables; check provider targets; retain timing evidence |
| Lazy facade drops an import side effect or API name | Test fresh-process formulas, public names, identities and introspection; run full gates and tutorials |
| Refactor collides with active parity candidates | Select an accepted parent, reserve overlapping files and use source-specific validation |
| Source moves disappear from coverage or mypy | Compare actual inventories and report inclusion across the single source root |
| Old measurements are mistaken for current acceptance | Seal new source-specific evidence and retain historical reports with their original status |

## 6. Decisions for this phase

1. One repository, one Python project/distribution and one `src/xrdroot` tree.
2. Fifteen internal components, including `cppstd`; app is an internal component,
   not a newly packaged umbrella distribution.
3. Existing public imports remain supported. Logical component ownership does
   not require moving every engine into a new directory.
4. ROOT adapters depend on the generic C++ implementation; generic code never
   imports ROOT or the translator to discover its environment.
5. Tests remain in the root test tree; move or group them only where useful.
6. Existing installation, extras, versions, CLI and runtime dependency policy
   remain in the root project configuration.
7. The user has authorized implementation. Scheduling remains coordinated with
   active parity work and its unchanged gates; source preparation alone does
   not establish acceptance.

## 7. Later work: separate Python projects

After the internal boundaries are proven, write a separate proposal for
independently installable components. That later proposal can decide project
names, repository layout, public package namespaces, build metadata, minimal
dependency environments, version policy, compatibility guarantees and release
automation. None of those packaging choices is fixed by this plan.

The C++ standard-library component is a candidate for an independent project
and other consumers. A standalone parser/emitter is another possible follow-up;
its ROOT-facing runtime adapters would still need an explicit boundary.

Further engine refinements, such as I/O without a histogram dependency or a
different CLI ownership split, can be evaluated separately after current
boundaries and parity are stable.

## 8. Progress log

| Date | Event | Acceptance |
|---|---|---|
| 2026-10-18 | Original multi-distribution draft | Historical plan only; measured against fa34ea9 |
| 2026-10-21 | Revised to internal refactoring; cppstd added as fifteenth component; separate Python projects deferred | Documentation revision only; no lanes started or implementation results claimed |
| 2026-10-21 | Implementation authorized; selected accepted local `8a008d88`, started an unchanged-source full baseline and prepared isolated source candidates across the lanes | Baseline active; candidates untested and uncommitted; no new gate, tutorial or global-ratchet acceptance |
| 2026-10-21 | Parent baseline completed and its capture audit passed; native outputs archived durably | Five committed PASS obligations remain lost; global ratchet is false |
| 2026-10-21 | Lazy facade and import-cycle correction pass all six gates on source `7c6fca40`; both test modes pass 9,351 tests with strict 100% line/branch coverage | Independent audits accept the gates and 79-path scope: 75 PASS, four existing DIFFs, no parent PASS lost. Committed locally as `720d178d` |
| 2026-10-21 | Complete refactor source `5ff9b15d` passes all six gates and its independent audit; both modes record 9,740 passed with strict 100% coverage | Source-specific gate acceptance only; ordered commits and tutorial acceptance remain pending |
| 2026-10-21 | Actual public CLI imports all fifteen components successfully in fresh processes | Its isolated cppstd tests expose a checker false positive for pytest assertion rewriting; a narrow test-caller correction and regressions are being prepared |
| 2026-10-21 | Python 3.10 development-only `tomli` approved for the component checker | Prepared with the component CI group; running sources and shared environments remain unchanged |
| 2026-10-22–23 | Component debt burned to zero in eight reviewed groups (58 → 47 → 41 → 36 → 32 → 25 → 12 → 8 → 1 → 0) with no limit raised and no assertion loosened; the twenty-five ordered prefix gates and their commits landed on `refactor/modularisation-commits-20261021` | Each prefix's six gates recorded their own both-mode counts with strict 100% line and branch coverage; `check --strict` reports zero undeclared violations |
| 2026-10-24 | Writable reader lvalues (`c2e9322`) and the set-order comparison mechanism (`4ddc223`) landed: `TTreeReaderValue` assignments store through the bound branch, and the harness accepts xrdroot only for an exact match with a genuine ROOT output — the cached draw or a recorded order | Six gates on each; translator survey 900/910 (98.9%) with zero crashes; rf102, rf404 and rf406 pass 12/12 stability runs |
| 2026-10-25 | NanoAOD speed group (`2418664`): guarded plain-int element reads and remembered ROOT-name lookups; the full harness on that tree records 553 PASS, and all nine tutorials its ratchet flagged pass when rerun alone | Six gates, CI green, translator survey 98.9% with no crash |
| 2026-10-25 | P3 acceptance packet sealed at `recovery/modularisation-P3-acceptance-33add7a-ROOT-ACTUAL-20261025`; Linux numerics confirmed in a `python:3.12-slim` container (full suite 9,814 passed, identical to macOS) | Branch CI green on nine jobs including the new distribution smoke check |
| 2026-10-25 | **`main` published at `33add7a`** (fast-forward from `fa34ea9`); the eight checks — strict component contract, all fifteen isolations, Ruff, maintainability, mypy, strict docs and both test modes with strict 100% coverage — were re-run on the main checkout before the push | Publication only; no baseline change, no limit raised, no assertion loosened |
| 2026-10-25 | `33add7a`: the shared fixtures act only on loaded state, `core` and `hist` join `cppstd` as declared focused scopes with the root fixtures in play, the component pages state the measured results, and CI builds and installs the wheel as a distribution smoke check | `check --strict` 857 modules / 9,788 imports / zero violations; all fifteen components isolate; both test modes 9,814 passed with strict 100% line and branch coverage |

Original draft preserved at
`~/xrdroot-parity/recovery/MODULARISATION-in-project-revision-20261021/MODULARISATION.before.md`.
