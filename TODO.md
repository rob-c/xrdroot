# xrdroot — parity work queue (agent handoff)

*Written 14 November 2026 from a full, measured run at commit `10e0e58`
(`feat/root-testsuite`). Companion document:
[`docs/history/state-assessment-2026-11.md`](docs/history/state-assessment-2026-11.md) —
the narrative, the numbers and the reasoning. **This file is the queue: what to
do, in what order, in which file, and how to prove it worked.**

---

## 0. Read this first

**Mission.** Make xrdroot run ROOT 6.40's own test suite, in pure Python, with
ROOT used only as a validation oracle. Success is defined at the bottom of
`AGENTS.md` ("Done means") and nothing here changes it.

**Where it stands today.** The numbers below are the assessment's at commit
`10e0e58` (reproduce in §2) with the ledger's wins (§7) added; the last column
is where each kind stands after them.

| kind | at `10e0e58` | now |
|---|---|---|
| roottest | 1038 / 2173 (47.8 %) | **1241 / 2173** (round 17's 1233 plus the ledger's in-area gains since - `io/tree` 11 → 18, `io/json` 4 → 11 - the last of them `io/json/ArrayCompress`, round 42, `/tmp/r42/json`; the last *whole-suite* roottest run is still round 17's, because a loaded run is unquotable, `W0-48`) |
| gtest | 52 / 350 (14.9 %; 507 / 2020 cases) | **61 / 350, measured whole at round 56** (`/tmp/r56/gtest`: 61 PASS / 211 FAIL / 76 UNSUPPORTED / 2 TIMEOUT of 350, `--jobs 3`) - the first whole-kind run since round 53's 56, and the five it gained are rounds 53-56's, **with none lost** |
| pyunittest | 78 / 129 (60.5 %; 439 / 515 cases) | **81 / 129 (470 / 515 cases)** |
| `test/` | 4 / 59 | **9 / 59**: the eight of `/tmp/r45/testkind` (round 45, which names every FAIL and refusal) plus `test/testbits` (`/tmp/r46/bits`, round 46) |
| module tests (`other`) | 14 / 109 | **40 / 109** measured whole at `/tmp/r45/other` (39) plus `root_exe_errorMessage` (`/tmp/r45/errm2`, round 45); the two TIMEOUTs are `mathcore-stdsort` (`W0-108`, a benchmark) and `minuit2_ParallelTest` |
| **all non-tutorial** | **1185 / 2820 (42 %)** | **1429 / 2820 (50.7 %)**: roottest 1241 + gtest 61 + pyunittest 81 + `test/` 9 + `other` 40. The per-kind figures above are each a measurement at the round named beside it; the last whole-suite run is `/tmp/r41/suite-fixed` |
| tutorials | 558 / 1189 committed baseline (47 %) | unchanged (not re-run); a `hist/` smoke run (93 tutorials, 75 PASS) lost no baseline PASS |

The "now" column is two runs at `2ef9c80`: `/tmp/r9/final2b` (roottest, 1141) and
`/tmp/r9/final1` (the other kinds: gtest 52, pyunittest 81, `test/` 6, `other` 21). The
three roottest tests the last round brought in are the last TIMEOUT
(`multicore/testSetAddressLoop`, `P-10`: a read of a translated `std::atomic` hands the
interpreter to the thread it waits for) and the two `treeformula/event/reference` +
`fastcloningeventtree/reference` fixtures (the 40 draws behind each are still FAIL, at
their next wall). `tree/friend/file2` and the four CERN-network tests are environment, not
code (`W0-28`). `other` reads 21
at `--jobs 5` **and still 21 alone**, because `minuit2_ParallelTest` now takes 309 s of the
300 s limit: measured alone at `8af51aa`, `bb19779` and `68a8d07`, so it is this machine, not
a code change (it took 195-236 s when the earlier rounds measured it). It is `P-9`'s
`cppstd/containers.py:_fast_int` hot path. The six roottest tests the last round brought in are
`io/json/PolyMarker`, `io/rootcint/cltest`, `io/stdarray/arrTemplMemberWrite`,
`io/stdarray/colWiseWrite_3`, `meta/tclass/execTmpltD32` and
`treeformula/string/cernbuild`. No PASS was lost; `roottest/main/RootcpReadRemote` and
`io/hadd/test_hadd_regr_20872` time out when their CERN endpoint is slow and pass
when rerun alone (both did, twice), and `minuit2_ParallelTest` does the same under
load.

**The shape of the remaining work**, which decides the order below:

1. Breadth of ROOT's names and methods — 153 roottest tests fail on a *name*
   that does not exist, 218 more on a missing method/attribute. Mostly small,
   independent, and cheap.
2. Three deep engines: **reflection/dictionaries** (`TClass` etc., ~117
   roottest tests), **tree/formula semantics** (~180), **written-file byte
   layout** (18 roottest + 13 tutorial file DIFFs).
3. A performance tail: 10 tutorials time out at 300 s; 14 are >10× ROOT.
4. RooFit/RooStats output parity: 84 tutorial stdout DIFFs.
5. The C++ translator is **not** the bottleneck (20 of 1125 failing roottest
   tests).

**Rules of engagement are in §3 — read them before the first commit.** Two of
them decide whether this document helps or hurts: *never raise a limit or a
timeout*, and *never write a PASS into a baseline that you did not observe in a
real run*.

---

## 1. Environment

This machine (the one the assessment ran on) has: macOS ARM64, 18 cores,
`/opt/homebrew/bin/python3` = 3.15.2 (too new to be the gate interpreter),
`python3.14`, and the ROOT 6.40.04 oracle at `~/.local/root-oracle`
(`root-config --version` → 6.40.04, Python 3.13.18 with pytest/numpy/pandas).
There is **no** `python` on `PATH`. There is **no** ROOT source tree checked
out; clone it (§1.3).

### 1.1 The working virtualenv

```bash
cd /Users/rcurrie/src/xrdroot
python3.14 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev,zstd,fit,docs]"     # ~2 min; xrdclient comes from its git pin
export PYTHONPATH=$PWD/src                            # the harnesses want this
.venv/bin/python -c "import xrdroot, xrdclient; print('ok')"
```

Notes and gotchas:

- `.[dev]` already brings numpy, pytest, pytest-xdist/cov/timeout, ruff, mypy,
  radon/complexipy, iminuit, matplotlib/plotly/bokeh/mplhep, pandas/polars/
  pyarrow/awkward, hist/boost-histogram/uhi, scikit-learn, scipy, xgboost,
  numba, dask.
- **XGBoost does not import without `libomp`**: `brew install libomp`. Until it
  is installed, 4 test files error and `tmva/rbdt.py` sits at 14 % coverage, so
  the 100 % gate reads 99.92 %. Install it once and the gate is honest.
- If you want the *local* client sources instead of the pinned wheel:
  `export PYTHONPATH=$PWD/src:$(ls -d /Users/rcurrie/src/grid_client/*/src | tr '\n' ':')`
  (xrdclient needs `xrdcore`; both live under `src/grid_client/`).
- `.coverage` (9.6 MB) in the repository root is a stale artefact; it is
  gitignored. Delete it if it confuses you.
- Optional extras that the harness environment should also have, or ~30
  tutorials are counted SKIP instead of measured: `torch`, `keras`/`tensorflow`
  (huge — decide deliberately), and the oracle wants an xml-enabled build for
  the 13 `analysis/unfold` tutorials.

### 1.2 The oracle (ROOT 6.40.04)

Already installed here. Elsewhere:

```bash
micromamba create -p ~/.local/root-oracle -c conda-forge --override-channels \
    root=6.40.04 python=3.13 pytest numpy pandas        # ~2.3 GB
```

ROOT's tutorial outputs are cached under `~/.cache/xrdroot-tutorials/oracle/`
(keyed by a hash of the tutorial, its siblings, the settings and ROOT's
version), so the oracle half of a tutorial run is free after the first run.
`--refresh-oracle` forces a rerun. **Do not** delete that cache to "clean up";
re-running 1 189 tutorials under ROOT costs hours.

### 1.3 ROOT's sources

```bash
git clone --filter=blob:none --sparse --depth 1 -b v6-40-04 \
    https://github.com/root-project/root.git rootsrc-6.40
git -C rootsrc-6.40 ls-tree -r --name-only HEAD | grep -oE '^.*/test/' \
    | grep -vE '^(interpreter|roottest|tutorials)/' | sort -u > /tmp/testdirs
git -C rootsrc-6.40 sparse-checkout set tutorials test roottest cmake/modules $(cat /tmp/testdirs)
```

`rootsrc-6.40/` is gitignored. The assessment used `/tmp/rootsrc-6.40` (409 MB
checked out, 167 MB of it `.git`); a checkout inside the repository works too.

### 1.4 The translation cache

`~/.cache/xrdroot/cint` was **771 MB** after one tutorial run and grows without
bound. It is a cache, it is safe to delete (`rm -rf ~/.cache/xrdroot/cint`), but
deleting it makes the next macro run translate again (slower). `XRDROOT_CINT_CACHE`
points it elsewhere.

---

## 2. Measure before you change anything

Run all of this once, on the commit you start from, and keep the output. Every
claim below is reproducible; if you cannot reproduce it, stop and find out why
before you write code.

### 2.1 The gates (a few minutes)

```bash
cd /Users/rcurrie/src/xrdroot && export PYTHONPATH=$PWD/src
.venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --cov=xrdroot   # 11 270 cases, ~80 s
XRDROOT_TEST_OTHER_MACHINE=1 .venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --cov=xrdroot
.venv/bin/ruff check src tests tools          # 0.4 s
.venv/bin/mypy                                # strict; 995 files; ~2 min cold
.venv/bin/python tools/maintainability.py check    # "27 804 functions in 1657 files pass"
.venv/bin/python tools/components.py check         # "994 source modules; 11 519 imports; 0 undeclared"
.venv/bin/mkdocs build --strict
```

Expected today: everything green **except** coverage at 99.92 % (XGBoost/libomp,
§1.1) and one genuine failure:
`tests/test_roofit_branch_chisquare_gamma.py::test_the_chisquare_likelihood_takes_the_kernels_digits`
asserts ROOT's last bit (`13.972970374276391`) without the `ROOTS_MACHINE` guard
of `tests/refmachine.py`, and on arm64 gets `…639`. Fix with the same guard the
other 18 files use (`grep -rl ROOTS_MACHINE tests`) — see **Q-9**.

### 2.2 The test suite (25 minutes)

```bash
# Kinds that need no ROOT. ~9-10 min, 650 tests.
.venv/bin/python -m tools.rootsuite run --root /tmp/rootsrc-6.40 --jobs 6 \
    --out /tmp/xrun/suite1 --kind gtest --kind pyunittest --kind test --kind other

# roottest. ~15 min, 2173 tests.
.venv/bin/python -m tools.rootsuite run --root /tmp/rootsrc-6.40 --jobs 6 \
    --out /tmp/xrun/suite2 --kind roottest
```

Read `RUN/summary.md` (counts, per-area tables, ranked causes) and
`RUN/results.json` (every record with `reason` and log path). Logs are under
`RUN/logs/<kind>/<id>/run.log`; gtest and pyunittest runs also leave a
`junit.xml` per test, which is the fastest way to enumerate failing *cases*.

Expected totals on this commit: `suite1` → gtest 52/350, pyunittest 78/129,
test 4/59, other 14/109; `suite2` → roottest 1038/2173. **If your numbers differ,
you are on a different commit or a different environment; say so in the report
rather than "fixing" the baseline.**

Useful subsets while working (ids are `kind/area/…`, `--only` takes fnmatch
globs, `*` crosses `/`, fixtures are pulled in automatically):

```bash
... --only 'roottest/root/meta/*'
... --only 'pyunittest/**/pyroot_pyz_ttree_*'
... --only 'gtest/hist/*' --kind gtest
```

### 2.3 The tutorials (up to 2 hours wall, ~1.5 h CPU)

```bash
.venv/bin/python -m tools.tutorials run --tutorials /tmp/rootsrc-6.40/tutorials \
    --jobs 4 --oracle-prefix ~/.local/root-oracle --out /tmp/xrun/tut1
.venv/bin/python -m tools.tutorials ratchet /tmp/xrun/tut1/results.json
```

`--only` here takes **path prefixes** (`--only hist/ roofit/roofit/rf7`) — not
globs. The tutorial results file records, per tutorial, the ROOT and xrdroot
exit codes, runtimes, files, and the **first differing line** of stdout
(`details`). That `details` field is the most useful text in the repository:
read it before opening a log.

Expected here: 544 PASS / 118 / 82 / 117 / 323 / 5 (PASS/DIFF/FAIL/UNSUPPORTED/
SKIP/ORACLE-FAIL) and **25 baseline PASSes lost**, of which 4 are the timeouts,
8 are RooFit/RooStats stdout DIFFs that come from the unpinned NumPy/Python, and
2 are `RNTupleProcessor`. The committed baseline (558 PASS) was taken in
`~/xrdroot-parity/venv312-pin806` (Python 3.12, NumPy 1.26), which **does not
exist on this machine**. Resolving that is task **W0-1**.

### 2.4 The API inventory (2 minutes)

```bash
.venv/bin/python -m tools.tutorials api-coverage --oracle-prefix ~/.local/root-oracle \
    --out /tmp/xrun/api
```

Writes `api-coverage.md`/`.json`: per ROOT library, classes/methods *present /
refused / missing* in `xrdroot.pyroot`. Today: `Core` 74/305 classes,
`Hist` 51/149, `RooFitCore` 76/289, `ROOTNTuple` 14/195, `MathMore` 0/52,
`Tree` 8/73. It also counts GUI libraries (`Gui` 0/213, `Eve` 0/197) — see **W0-7**.

### 2.5 The translator survey

```bash
.venv/bin/python tools/cint_survey.py /tmp/rootsrc-6.40/tutorials --min-percent 98
.venv/bin/python tools/cint_survey.py /tmp/rootsrc-6.40/roottest --top 20
```

Last recorded: 897 of 910 macros translate (98.6 %), 10 refused, **3 crashed**
(a crash is worse than a refusal: it must become a refusal or a fix). Re-run it
after any `pycxx` change.

### 2.6 Where the evidence from the assessment lives

`/tmp/xrun/` is disposable but was present at handoff:
`suite1/`, `suite2/` (non-tutorial kinds), `tut1/` (tutorials),
`api/` (API coverage), `pytest1.log`, `pytest-cov.log`, plus `bench*.root`
and the compression/throughput measurements of §2.6.1. If it is gone, §2.1–2.4
regenerate everything.

#### 2.6.1 Throughput baseline (measured, for **P-0**)

| operation | rate |
|---|---|
| tree write, 1 M rows × 2 float64, `compression=None` | 9.1 M rows/s |
| … `compression="lz4"` | 58.5 M rows/s |
| … `compression="zstd"` | 17.7 M rows/s |
| … `compression="zlib"` (today's default, level 6) | 1.8 M rows/s |
| … `compression="zlib", level=1` (**ROOT's default; what Q-6 makes the default**) | **4.0 M rows/s** |
| tree read, same file | 34.7 M rows/s |
| `RDataFrame("t",f).Filter("x>100").Define("z","x*y").Mean("z")` | 32.9 M rows/s |
| `TH1D.FillN` | 38.5 M rows/s |

---

## 3. Rules of engagement

Distilled from `AGENTS.md`; they are not negotiable and they are what keeps the
measurement trustworthy.

1. **Never raise a limit.** Not the 300 s harness timeout, not the maintainability
   limits (CCN ≤ 10, cognitive ≤ 15, NPath ≤ 200, Halstead ≤ 500, nesting ≤ 4),
   not the coverage floor. Split the function, make the code faster.
2. **Never add `# pragma: no cover` to dodge a branch**, and never delete or
   loosen an assertion to make a test pass. Decide whether the code or the test
   was wrong, and say which in the commit message.
3. **A PASS is a promise.** `tools/rootsuite/baseline.json` and
   `tools/tutorials/baseline.json` only ever change in the same commit as the
   improvement they record, and only from a real run's `results.json`
   (`python -m tools.rootsuite baseline RUN/results.json`, same for tutorials).
   Losing a PASS needs a root cause, not a re-baseline.
4. **Bit-exactness only where ROOT's reference was produced** (`tests/refmachine.py`,
   darwin x86_64). Everywhere else use `FIT_REL`/`PROFILE_REL`/`NLL_REL`/
   `roots()` with a comment saying which numerics (glibc vs libm, FMA, `sum()`
   before 3.12) make the difference. Route transcendental functions that claim
   ROOT exactness through `xrdroot.random.libm`.
5. **Refusals are full sentences** that name the class or algorithm and say what
   to do instead: `raise UnsupportedFeatureError("…")`. A refusal is how the
   harness ranks work; a `NotImplementedError("todo")` or a silent `None` costs
   the whole project.
6. **Small commits, one fix group**, `feat:`/`fix:`/`perf:`/`tests:`/`docs:`.
   Gates green, then commit, then push, then watch CI (`gh run watch`).
7. **Do not build process machinery.** No receipt/audit/ledger files, no
   hash-chained acceptance records. The evidence is `results.json` and the
   ratchets. (The 805-line `docs/history/agent-status-2026-10.md` is exactly the
   thing not to do again — see **H-2**.)
8. **Long jobs run in the background with their output under the run
   directory**, outside the repository.
9. **Optional packages are extras, imported lazily**, refused with the
   `pip install` that fixes them (`src/xrdroot/library.py` is the pattern).
10. **Component boundaries hold.** A lower component never imports a higher one;
    the way across is a provider in `xrdroot.plugins`, a callback, or plain
    data, in that order. `python tools/components.py check` must stay at zero
    undeclared.
11. **A new file is owned by a component.** Add it to the owning
    `[tool.xrdroot.components.ownership.<component>].modules` list in
    `pyproject.toml`, or `components.py check` will report it as undeclared. A
    new module under `src/xrdroot/pyroot/` also goes into `SUBMODULES` in
    `src/xrdroot/pyroot/__init__.py`, and a new user-facing class is documented
    in `docs/root.md` (house style) — both are checked by eye in review, not by
    a gate, so they are the two things most often forgotten.

---

## 4. The queue

Every task is written to be independently shippable. `Tests` is the number of
tests the assessment measured as blocked by it; treat it as a forecast, not a
promise. Sizes: **XS** < 1 h, **S** < half a day, **M** < 3 days, **L** > 3 days.

### Wave 0 — make the number trustworthy (do this first; a few days)

Nothing else is worth doing until "PASS" means the same thing twice.

| ID | task | size |
|---|---|---|
| W0-1 | Pin the parity environment (or re-baseline it) | S |
| W0-2 | Run the harnesses in CI, nightly, with the ratchet | M |
| W0-3 | Make `ratchet` judge only the kinds a run contains | S |
| W0-4 | Put the first differing line into every DIFF reason | S |
| W0-5 | Explain the 3 % discovery gap by name | S |
| W0-6 | A timing ratchet (median + slowest 20) | M |
| W0-7 | Scope the API-coverage report to non-GUI libraries | S |
| W0-8 | Fix the two harness blind spots found while triaging | S |

**W0-1 — pin the parity environment.** Today the tutorial numbers depend on
Python/NumPy versions, and the committed baseline came from
`~/xrdroot-parity/venv312-pin806` which does not exist. Choose one:
(a) recreate that venv and keep using the committed baseline; or (b) declare a
new pinned environment (`constraints-parity.txt` with exact `numpy`, `scipy`,
`iminuit`, `matplotlib` versions plus the client pin) and **re-baseline in the
same commit** with a note in `tools/tutorials/baseline.json`'s commit message.
(b) is the honest long-term answer: the ratchet is only meaningful if the
environment is a file in the repository. Record the interpreter version in
`docs/history/` or `README`, not in a file outside the repo.

**W0-2 — CI.** Add `.github/workflows/parity.yml`:
- `schedule: nightly` + `workflow_dispatch`, one job per kind
  (`gtest`, `pyunittest`, `test`, `other`, `roottest`), `--jobs 4`,
  `ubuntu-latest`, ROOT sources by the §1.3 sparse clone (cache it with
  `actions/cache` keyed on the tag), the pinned parity env, then
  `python -m tools.rootsuite ratchet RUN/results.json`. A kind that is not in
  the run must not be judged (that is W0-3).
- A second job for tutorials, `--only` the ~150 cheapest areas on PRs and the
  full 1 189 nightly, with the oracle **cached** (the cache is keyed by ROOT
  version, so the first run pays and later runs do not).
- Upload `results.json` and `summary.md` as artefacts. **No PASS may be lost**:
  a red ratchet fails the job.
Note for the implementer: the rootsuite needs no ROOT *installation* — only
ROOT's sources — so this job is cheap on a Linux runner.

**W0-3 — `ratchet` scoping.** `tools/rootsuite/ratchet.py::regressions` compares
against every baseline PASS, so a `--kind roottest` run reports 148 phantom
"regressions" that are simply kinds not run (verified). Read `doc["meta"]["kinds"]`
and judge only those kinds; `MISSING` stays an error only inside them. Add a
regression test in `tests/test_rootsuite_running.py`.

**W0-4 — DIFF reasons.** `"standard output differs from roottest/cling/const/good1.output"`
is 96 roottest tests' entire reason. `tools/rootsuite/refdiff.py` already knows
how to find the first difference (the tutorial harness uses it); have the judge
append it: `first difference line 2: expected 'with const MyClass4*', got 'with MyClass4*'`.
That single change makes the biggest DIFF group self-triaging.

**W0-5 — discovery gap.** `python -m tools.rootsuite list` reports gtest 350/389,
roottest 2173/2210, test 59/64, pyunittest 129/109, tutorials 894/918 against
`tools/rootsuite/model.py::CI_COUNTS`. Produce a list of the missing ctest names
(they are the ones ROOT's CMake generates at configure time, plus a few feature
gates) and either discover them or document each by name in
`docs/components/`+ the report. A silent 3 % is a hole in "the whole suite".

**W0-6 — timing ratchet.** `tools/tutorials/timing.py` and
`baseline-timings.json` exist; nothing enforces them. Add
`python -m tools.tutorials timing RUN/results.json` (or extend `ratchet`) that
fails when the median ratio or any of the 20 slowest grows by more than, say,
15 % against `baseline-timings.json`, with an explicit "machine is loaded"
escape that requires re-running the same tests alone (the same discipline the
300 s timeout already gets).

**W0-7 — API coverage scope.** `tools/tutorials/api_coverage.py::LIBRARIES`
currently lists every library it found, so the report is dominated by GUI
(`Gui`, `Eve`, `RGL`, `Ged`, `FitPanel`, `WebGui6`, `ROOTBrowserv7`…). Split
into `LIBRARIES` (the ones the suite uses) and `OUT_OF_SCOPE_LIBRARIES`
(reported but not counted), and put the pass/fail line in `summary.md`. The
definition of done says "every non-GUI ROOT class the suite uses"; make the
report say that.

**W0-8 — harness blind spots.** Two found while triaging:
(a) pyunittest FAILs are reported as `exit N, expected N: … ===== 3 failed, 8 passed …`
for 19 of 46 files, because the judge does not read the `junit.xml` it already
wrote — it does for gtest cases (`N of N cases failed; first test_x: …`). Use
the same path for unittests.
(b) `tools/rootsuite/judge.py` treats a nonzero exit as the whole reason even
when the program printed its own diagnosis on the last stdout line (the
`(int) 1` family). Append the last non-empty stdout line, as ROOT's own ctest
comparison would show it.

### Wave 1 — cheap, wide wins (2–3 weeks)

The tasks below account for ~210 tests directly, and for the cascades they
release.

Order inside the wave is by tests-per-hour; they are independent, so several can
be done in parallel in separate worktrees.

| ID | task | tests | size |
|---|---|---|---|
| Q-1 | `TClass` reflection: `GetClassInfo` & friends | 75 | M |
| Q-2 | `TArray*::fArray` (data member) | 16 | XS |
| Q-3 | `char[N]` branch address (`View.get` `.item()`) | 24 cases | S |
| Q-4 | `RNTupleWriter` context manager | 7 cases | XS |
| Q-5 | `TH1::SetContent` | 5 + 2 tutorials | XS |
| Q-6 | zlib default level 1 (ROOT's default) | perf + sizes | XS |
| Q-7 | Missing ROOT names, batch 1 (the 3+-test ones) | ~60 | S |
| Q-8 | `FromCSV` options, `GetEXlowd`, `RooStringVar`, `PyConfig`, `ROOT.__name__` | ~10 cases | S |
| Q-9 | The arm64 exactness test that is not guarded | gate | XS |
| Q-10 | `.Class` / `.typedef` / `.autodict` as `ProcessLine` commands | 9 | S |

**Q-1 — `TClass` reflection (the single biggest roottest blocker).**
45 tests die on `AttributeError: 'TClass' object has no attribute 'GetClassInfo'`,
plus `GetDict` (5), `GetMethod` (4), `GetMissingDictionaries` (4),
`GetListOfEnums` (3), `BuildRealData`, `TInterpreter.GetMapfile` (4), and most
of `roottest/root/meta` (182 tests).

- Everything lives in `src/xrdroot/pyroot/core/reflection.py` (`class TClass`,
  line 138) with the registry in `src/xrdroot/pyroot/core/declared.py`
  (`declared_class`, `foreign`) over `pycxx.runtime.namespace.DECLARED`.
- Semantics to match (read the tests!): `GetClassInfo()` is *not* null once the
  class has a dictionary — the `cling/stl/map/stringMap.cxx` test uses
  `cl->GetClassInfo()==0` to decide whether to `.L stringMapLoad.cxx+`, so a
  class the interpreter has declared must answer non-null and the class must
  then work. `GetDict()` likewise. `GetMethod(name)` returns a `TMethod` (there
  is `GetListOfMethods` and `GetMethodAny` already — build on them).
  `GetMissingDictionaries()` returns a `TList` of the *bases* with no
  dictionary. `GetListOfEnums()` a `TList` of `TEnum`/`TEnumConstant`.
  `TInterpreter.GetMapfile()` is the `.rootmap` path a loaded library would have.
- Verify: `--only 'roottest/cling/stl/map/*' --only 'roottest/root/meta/*'`
  before/after, and add focused tests in `tests/test_pyroot_reflection.py`.
- Pitfall: do not make `GetClassInfo` return a bare `True`; several tests call
  methods on the result and compare pointer identity.

**Q-2 — `TArray*::fArray`.** `src/xrdroot/pyroot/vectors.py` (`_TArray`, line 176;
`GetArray()` already returns the NumPy view). ROOT exposes `fArray` as a
public data member, so add a property returning the same view. 16 tests
(`roottest/root/io/bigevent/*`, `TArray` evolution) read `arr->fArray[i]`.

**Q-3 — the `char[N]` branch address (24 pyunittest cases, 2 files).**
`src/xrdroot/pyroot/trees/addresses.py:74-77`:

```python
def get(self, count=None):
    if count is None:
        return self.view[self.start].item()      # <- explodes
```

Reproduce in one line (from `rootsrc-6.40/bindings/pyroot/pythonizations/test`):

```bash
PYTHONPATH=$PWD/src .venv/bin/python -c \
 'import xrdroot.pyroot as R; R.gInterpreter.Declare("#include \"TreeHelper.h\""); \
  R.CreateTTree("t.root","t",10,4,0,"RECREATE")'
```

Diagnosis (already done): the address for `char s[10]` is a `View` over a
`cppstd.buffers.CharBuffer`, whose `__getitem__` returns a Python `int`; every
other `View` is over a real `ndarray`. Decide, by what the test asserts about
`chararrayb` (it must read back as the string `"onetwo"`), whether a `char[N]`
should be a `Text`/char-array address rather than a `View` — and if it should
stay a `View`, at least `get()` must not assume `.item()` exists. Then check
`tests/test_pyroot_tree_branch*.py` and run
`--only 'pyunittest/**/pyroot_pyz_ttree_*'` (20 cases across 2 files).

**Q-4 — `RNTupleWriter` context manager.** `src/xrdroot/pyroot/ntuple/writing.py:192`.
`test_nested_ctxmanager` and `test_weird_ctxmanager` require ROOT's behaviour:
entering twice raises, exiting closes/flushes, `__enter__` returns the writer.
7 cases in `ntuple_py_basics`.

**Q-5 — `TH1::SetContent`.** `src/xrdroot/pyroot/core/hists.py` (`class TH1:35`).
ROOT's `SetContent(const Double_t* content)` fills **every** bin including
under/overflow from a flat array (`bins + 2` values for a 1-D histogram) and
returns `bool`. Two tutorials (`OneSidedFrequentistUpperLimitWithBands`,
`TwoSidedFrequentistUpperLimitWithBands`, both `.C` and `.py`) and one pyunittest
file (`xroofit_python`) use it. There is no `SetContent` anywhere today —
`grep -rn SetContent src` returns nothing.

**Q-6 — the default write compression.** `src/xrdroot/compression.py:45`:

```python
LEVELS = {"zlib": 6, "lzma": 6, "lz4": 1, "zstd": 3}
```

ROOT's `TFile` default is **101** = zlib **level 1** (verified against the
oracle: `GetCompressionSettings()==101`, `GetCompressionLevel()==1`). Change the
zlib default to 1, keep `level=` overridable, check every test that pins a level
(`grep -rn "level" tests/test_compression*.py tests/test_writer*.py`), and
confirm with the throughput in §2.6.1 (1.8 → 4.0 M rows/s) and with the
`Disk size of … should have been …` tests (they will *not* all pass from this —
see Q-20 — but this removes one source of size difference).

**Q-7 — batch 1 of the missing names.** Add (as real implementations or as
honest refusals; prefer real):

| name | tests | where it belongs |
|---|---|---|
| `TEnum` (+ `TEnumConstant`) | 9 | with Q-1 |
| `TTreeFormula` | 5 | `pyroot/trees/` over the `formula/` engine |
| `std::complex` | 5 | `cppstd/`, then `pyroot/stdlib` |
| `std::forward_list`, `std::multiset`, `std::atomic_int` | 11 | `cppstd/containers.py` |
| `gDebug` | 5 | `pyroot/core/` session |
| `TClassEdit` | 4 | `pyroot/core/reflection.py` (`std::` name munging) |
| `TBufferJSON` | 4 | `io/` (JSON of a ROOT object) |
| `fgets` | 4 (+3 tutorials) | the ROOT dialect's C functions (`pycint/dialect.py`) |
| `TCpu`, `Architecture`, `ProcInfo_t` | 8 | `pyroot/core/system.py` |
| `TSelector` (+ `TTree::MakeSelector`) | 5 | `pyroot/trees/` |
| `ROOT.Detail`, `ROOT.Internal.RDF` | 5 | `pyroot/rdf/` |

A shortcut that is worth considering for the *names* (not the behaviour): the
API inventory (§2.4) lists every class and method ROOT has. A generated skeleton
that answers with the right signature and refuses in one sentence would stop a
missing name from costing a test, leaving only the behaviour to write. If you do
that, generate it from the inventory so it cannot drift.

**Q-8 — small API gaps the pyunittest kind found.** All are XS:
- `ROOT.RDF.FromCSV(leftTrim=…, rightTrim=…, skipFirstNLines=…)` —
  `src/xrdroot/pyroot/rdf/frame.py:337` passes every keyword straight to
  `csv_source`; the options live in `src/xrdroot/rdf/csvsource.py`.
  Test: `--only 'pyunittest/**/datasource_csv'`.
- `TGraphAsymmErrors.GetEXlowd/GetEXhighd/GetEYlowd/GetEYhighd` —
  `src/xrdroot/pyroot/core/graphs.py:525`. ROOT returns the low/high errors as
  double arrays; there are already `GetEXlow()`-family methods to copy.
- `RooStringVar` — `pyroot/roofit/`; needed by `pyunittest/.../pyroot_roofit`.
- `ROOT.PyConfig.IgnoreCommandLineOptions` must default **True** —
  `src/xrdroot/pyroot/core/environ.py:212` says `False`.
- `import ROOT; ROOT.__name__ == "ROOT"` — the alias module the suite imports
  (`xrdroot.cli.shims` / `tools/rootsuite/pyshim.py`); the module object needs
  ROOT's name, not `xrdroot.pyroot`'s. Test `root_module.py::test_import`.
- `gDirectory` must be a *lazy* proxy that reads null inside
  `TDirectory.TContext(nullptr)` — `root_module.py::test_lazy_gdirectory`
  (`src/xrdroot/pyroot/core/files.py` + `pyroot/core/troot.py`).

**Q-9 — the unguarded exactness assertion.** `tests/test_roofit_branch_chisquare_gamma.py:60`
asserts `nll.getVal() == 13.972970374276391`; on arm64 it is `…639`. Guard it
with `tests/refmachine.ROOTS_MACHINE` and a tolerance + comment for elsewhere,
as the other 18 files do. This is the only real unit-test failure in the tree.

**Q-10 — `gROOT->ProcessLine(".Class …")`.** `src/xrdroot/pycint/execute.py:306`
handles `.x`/`.L` and delegates the rest to
`src/xrdroot/pycint/interactive/commands.py`; `.Class`, `.typedef` and
`.autodict` are not there, so 9 roottest tests die with
`ValueError: .Class is not a command gROOT->ProcessLine runs here`. Match
cling's behaviour and messages (`roottest/cling/operator/runEqualTest`,
`roottest/root/meta/autodict/*`, `roottest/cling/typedef/*`).

### Wave 2 — tree, I/O and reflection (6–10 weeks)

~290 tests directly, plus the ~100 that currently fail only because a fixture
these unblock did not set up.

| ID | task | tests | size |
|---|---|---|---|
| Q-11 | `TTreeFormula`: a branch's member calls, and `TTree::Draw` features | 122 | L |
| Q-12 | Tree column typing: `int() argument …`, object columns, `Event` splitting | 55 | L |
| Q-13 | Object ownership, destructors, heap flags, persistence flags | ~12 | M |
| Q-14 | `TGeoManager::Import` (GDML / ROOT-written geometry) | 20 | M |
| Q-15 | `TTree::BranchRef` / `TRefTable` (write and read) | 18 | M |
| Q-16 | `std::array`/`TClonesArray`/`vector<vector<T>>` writing and splitting | 25 | M |
| Q-17 | `TTree::Print`/`TObjString`/dictionary print output as the tests compare it | ~5 | M |
| Q-18 | `classloader`/`DataBlock`-layout refusals in `root/io` | 10 | M |
| Q-19 | `TTree::MakeSelector`, `TSelector` | 5 | M |
| Q-20 | **Byte-exact written-file layout** (`Disk size … should have been N`) | 18 + 13 tutorials | L |
| Q-21 | `hsimple`'s `hprof` bin errors (2·10⁻⁶ relative) | 1 + fixture | S |
| Q-28 | **`RootClassVersion(Class,N)` and `__CLING__`** | 5 + a written-file layout | M |
| Q-29 | `TStreamerInfo::BuildCheck`/`CompareContent` warnings, and which stream an OUTREF test compares | 4 | S |
| Q-30 | The emulated layout: file offsets, `newtype`, `Emulation` titles, `GetStreamerInfos` identity | 40 | L |
| Q-31 | `hadd` of a split branch whose name ends in a dot (`'holder.' is not a name a column can have`) | 4 | S |
| Q-32 | TMVA's DNN layers beyond DENSE (`RESHAPE` done) and `ROOT has Architecture` | 7 | M |
| Q-33 | pycxx brace-init of `unordered_set`/`multiset`/`multimap`/`unordered_multimap`, and `_Set`'s arity | 8 | S |
| Q-34 | `TObjArray` written as ROOT's `TObjArray::Streamer` writes it (with the `TNamed` title) | 44 | M |
| Q-35 | `TObject`'s heap bits and `TRef::SetAction`'s BIT(16) | 4 | XS |
| Q-36 | `TBufferFile::CheckByteCount`, and reading by the declared layout when versions match but checksums differ | 5 | M |
| Q-37 | A `TRef`'s read side: `TFile::ReadProcessID`, `TRef::GetObject`, `TRefTable` resolved per entry | 12 | M |
| Q-38 | `interp.Each` is a `Values`, so every `isinstance(column, Values)` "one streamed object per entry?" test misreads a split member | 12 + 29 clone tests | S |
| Q-39 | `TString*` counted by a `//[n]` comment (`kStreamLoop`): away with `bigevent`'s `fTemplateMember` | 16 | S |
| Q-40 | A macro class deriving from `TObject` cannot be split (`wclasses.declared` refuses any base; `harvested` excludes `TStreamerBase`) | 9 | M |
| Q-41 | `.Class` in `ProcessLine`, and a template template parameter in a dictionary build | 7 | M |
| Q-42 | The treeformula walls after `fMatrix`: `==` with a string, `strstr`, `fValid<<4`, `fValid&0x1`, `TestBitNumber`, `TMath::BreitWigner`, `Alt$` | 40 | L |
| Q-43 | A `dict`-held record's members by attribute (`'dict' object has no attribute 'fA'`) | 4 | S |
| Q-44 | `TGeoManager`'s `Import`/`Export` lines and a `TGeoManager` file reader (GDML is read) | 21 | M |
| Q-45 | A whole-object branch's member letter is not a number (`event.fTriggerBits` names an object) | 42 | M |
| Q-46 | `TFile::MakeProject`, which the RNTuple makeproject tests call after writing | 4 | M |
| Q-47 | `.Class` on a class a library defines (`Class Name::MyClass not found`) | 3 | S |
| Q-48 | The formula reads an object member as an object (`event.fTriggerBits`), the last wall before 42 tests | 42 | L |
| Q-49 | An old element code with a name today's ROOT knows (`Int_t` under code 6) is not a width difference | 2 | S |Notes that will save days:

- **Q-38** is the round-4 worker 4's find and it is a bug, not a gap: `interp.Each` *is* a
  `Values`, so `copying._whole`'s "one streamed object per entry?" says yes for a split
  `TNamed` base's `fName`/`fTitle` columns and the clone refuses with
  `the branch 'TNamed' holds whole objects ...`. Its 64-line prototype is
  `/tmp/r4/w04-prototype-patch.diff` (adds `and not isinstance(column, Each)`, plus a slot
  for the counter leaf `N_` a per-object column is counted by); with it, 29 clone tests
  pass and `CloneTree(0)` gets as far as the writer's `SplitClones`.
- **Q-42** is the whole of `treeformula`'s 40: with `fMatrix`'s shape kept (Q-30), the
  fixture dies at `dt_DrawTest.C:32` on `fType == "type1"` (`fType` is a `char[20]` column),
  then `1 && strstr(fType,"1")`, `fValid<<4`, `fValid&0x1`,
  `fTracks.fTriggerBits.TestBitNumber(5)`, `TMath::BreitWigner(fPx,3,2)` and
  `Alt$(fClosestDistance,0)`, in that order. `TTree.keys()` and `TTree.arrays` do not exist
  on pyroot's `TTree` at all, so a PyROOT-level test of a formula shape has to go through
  `xrdroot.open_root(...)['T']`.
- **P-9** after `f113afb`: `pack_member` is 2x faster, and what is left is the *per-record*
  rebuild of the class description and its offsets (`core/declared.py::_plain_object` ->
  `wdeclared.declared`), which is what the 825 s / 2 430 s figures are made of.

- **Q-28 = `RootClassVersion` + `__CLING__`.** `src/xrdroot/pycint/dialect.py:62` defines
  `__CLING__` for every translation, including an ACLiC library build, where real ROOT does
  not - so `#ifndef __CLING__ RootClassVersion(MyClass,3) #endif` in a test header is dropped
  and `RootClassVersion` does not exist anywhere in `src/`. Consequence on disk: xrdroot's
  `myclass3.root` records `MyClass` version 1 where ROOT records 3. This is the whole of
  `io/evolution/versions/AddVersion-1-3` and `-1-4`.
- **Q-30.** Measured by worker 5 against all 40 `.ref`s: the file's `offset=4` is
  recomputed by ROOT's `TStreamerInfo::Build` (the macOS oracle prints 8 for the same file -
  ROOT's `.ref` was made on x86_64), the in-memory layout is the file's with each member's
  known type substituted (`newtype=` keeps the file's, and the action line prints `type+200`),
  `Emulation` titles mark a built-from-file layout, and `#N` is the index in ROOT's
  `fStreamerInfos`, which appends a layout on every refresh - so the same checksum appears
  twice. `TStreamerElement.GetOffset` always returns 0 here because the reader drops the
  file's offset.
- **Q-34.** ROOT's own 88-byte `TObjArray` stream is `/tmp/r3/oracle-objarray/oa.bin`:
  `40000054 ffffffff "TObjArray" 40000042 0003`, then `TObject`, then **name and title**,
  then `fSize` and `fLowerBound`, then each slot as a tagged object.
- **Q-36.** ROOT reads a record by the *declared* layout when the on-file version equals the
  in-memory version but the checksums differ; xrdroot reads by the file's layout everywhere,
  so `meta/evolution/execMixedBaseClass_v2` silently misreads (44 B where ROOT reads 56).
  `Buffer.resume(end)` also jumps to the recorded end with no `CheckByteCount`.
- **P-9.** Worker 14's numbers, per 10^6 records: `WriteBuffer-1/2` ~825 s,
  `ReadBuffer-1/2` ~2 430 s, both in `core/declared.py::_plain_object` -> `wdeclared.declared`
  -> `wdobjects._basic` -> `wpacking.pack_member` (11.7 of 30.8 s per 10^5 calls); the fix is
  to memoise the class description and its `struct.Struct` per `TClass` rather than per
  `Streamer` call. `ntuple_largefile1` writes 268 435 456 entries one at a time (`Rndm` per
  event). `testBulkApiMultiple` spends 14.2 us per `Fill` in `trees/store.py:209` ->
  `_nbytes` -> `tree.py:184`, and the per-slot byte count can be computed at branch creation.
  `minuit2_ParallelTest` is `cppstd/containers.py:_fast_int` in a translated
  `__getitem__`; it has grown from 195-236 s to **309 s** (measured 2026-11-19 at `8af51aa`,
  `bb19779` and `68a8d07` alike, so the machine and not a commit), which is over the 300 s
  limit and makes it a FAIL-by-timeout until the hot path is fixed.
- **P-10.** `multicore/testSetAddressLoop` used to hang and now completes a run in 53 s
  (`16b3766`): 33 runs of it is 29 minutes. A run is 9 threads reading one entry; the busy
  wait `while (counter < kNThreads) {}` competes with them for the GIL.

- **Q-11.** The message to look for is
  `FormulaError: 'GetNtrack' in 'GetNtrack()' is not a function ROOT's formulas know`
  (`src/xrdroot/formula/`), raised while `TTree::Draw` evaluates a formula whose
  variable is a *tree* whose named objects have methods. ROOT's `TTreeFormula`
  resolves those through the tree's friends/branches and their classes, so this
  lands on the same reflection registry as Q-1 — **do Q-1 first**. The same work
  gives `ROOT.TTreeFormula` (5 more tests).
- **Q-12.** Two different errors with one root cause:
  `TypeError: int() argument must be a string, a bytes-like object or a real
  number, not 'TNamed'` (a `TTree` branch read back as an object when the leaf
  says a number — `interp.py`, `pyroot/trees/*`) and
  `ValueError: 'event' holds float64 values, and these are object, which would
  not go into it without losing what they are` (writing a column whose dtype is
  `object` — `wtree.py:747` `_cast`). Both are the "`Event` class split into
  branches, then written back" path used by ~30 tests. The error messages are
  good; use them.
- **Q-13.** About a dozen tests fail with a self-diagnosis on stdout, e.g.
  `Error: Either the destructor of MyClass was not invoked or the TObjArray lost
  memory of its ownership.` (`roottest/root/collection/AccidentalOwnership`).
  `pycxx/runtime/objects.py` and `cppstd`'s cells hold the destructor
  semantics; `interp.py` holds the heap flags (`kIsOnHeap`).
- **Q-20.** The 16 tests are all in `roottest/root/io/filemerger/*` (their
  programs are named `testSimpleFile` and `execTestMultiMerge`) and they
  compare **exact file sizes** with ROOT's. That needs ROOT's basket sizes,
  `TKey` padding, free-segment records and compression choices reproduced
  byte-for-byte. It is well defined (compare against the oracle) but it is the
  hardest single item in this document; it is also what the 13 tutorial
  `ROOT file differs` results and AGENTS.md's "file round-trip matrix" need.
  Build the round-trip matrix first (write with xrdroot → read with ROOT, and
  the reverse, for every streamer layout) — it is the diagnostic that makes
  the byte differences findable.
- **Q-21.** `hsimple.C`'s `hprof` file differs in 2 of 102 bins:
  `0.110847836279504` vs `0.11084761176289247` (2·10⁻⁶ relative). It is a
  `TProfile` error computation (`src/xrdroot/profile.py`, `errors`/`spread`):
  the accumulation order of `sumw2`/`sumw` in a bin, not the formula. Compare
  against `TProfile::GetBinError` bin by bin with the oracle for the same
  fills. Fixing it removes the canonical fixture's file DIFF, and any dependent
  tutorial that rewrites `hsimple.root` (638 tutorials *depend* on it, though
  the harness gives them ROOT's copy so they are not blocked by it).

### Wave 3 — performance (3 weeks; the exit criterion is "no tutorial over 150 s")

| ID | task | effect | size |
|---|---|---|---|
| P-0 | Write the throughput benchmark down (use §2.6.1) and make it a gate | keeps the tail from growing | S |
| P-1 | Batched/vectorised RDataFrame for **declared C++ functions** | 10 timeouts → passes | L |
| P-2 | Vectorised `ROOT::VecOps` over jagged arrays | the same 10, plus df026/df033/df016 | L |
| P-3 | RooStats toy studies and RooFit integral/caching reuse | 32→ ~15 tutorials >5× | L |
| P-4 | Canvas text and palette speed (`perceptualcolormap`, `schroedinger_hydrogen`, `tmathtext`) | 12–15× → <3×, and the same defect is the worst image DIFFs | M |
| P-5 | zlib level 1 (Q-6) + a reused `compressobj`; consider `python-isal` | 2–3× on every write | S |
| P-9 | The five genuinely O(n) timeouts, measured by worker 14 | 8 tests | L |
| P-10 | `multicore/testSetAddressLoop`: 33 runs of a 53 s run | 1 test | M |

**P-1 is the highest-value engineering task in this document.** The batch
evaluator (`src/xrdroot/rdf/expression.py`, `src/xrdroot/pyroot/rdf/batched.py`)
already vectorises expressions; a string that calls a function declared through
`gInterpreter.Declare` (which is exactly `df101`, `df103`, `df104`, `df105`,
`df107`, `hist102`, `hist103`, `tmva100`) falls back to
`src/xrdroot/pyroot/rdf/entrywise.py` — one Python call per entry, 12–45× slower
than ROOT. The path: translate the declared function body (the translator
already can) into per-batch NumPy when its operations have batch semantics
(the `batched.py` docstring enumerates the rule), and keep the entrywise path
as the fallback. Measure with
`python -m tools.tutorials run --only analysis/dataframe/df103_NanoAODHiggsAnalysis.py`.
Never raise the 300 s timeout; the target is <30 s.

### Wave 4 — RooFit/RooStats output parity (~84 tutorial DIFFs, ongoing)

Work in dependency order, with the tutorial `.ref` as the specification. The
tutorial results' `details` field names the first differing line, so triage is
free — e.g. `math/fit/myfit.C` differs at line 1 because xrdroot prints
`(TFile *) 0x…` at top level in batch mode where ROOT prints nothing (that one
is a `pycint` interactivity bug, worth fixing first: 6 tutorials, **Q-22**).

Groups, in the order the report ranks them:
1. INFO/`PROGRESS`/`WARNING` log **order** and cache keys
   (`RooAbsCachedPdf::getCache … for nset (x,mean) with code 1` vs `(x) … 2` in
   `rf211`, `rf705`, `rf707`).
2. Evaluation counts (`NCalls = N`) — the number of FCN/integral calls.
3. Minimizer status messages (`Minuit2Minimizer : Invalid minimum - status = 3`
   vs `Valid minimum - status = 0` in `rf612`) — the recovery path.
4. `RooFitResult` EDM/status print precision.
5. Missing RooFit classes, by tutorial count: `RooLagrangianMorphFunc` (4),
   `RooJeffreysPrior`, `RooMomentMorphFuncND`, `RooPolyFunc`, `RooSimWSTool`,
   `RooStats::SPlot` (2 each).

**Q-22 (add to Wave 1)** — top-level pointer echo. In batch mode ROOT does not
print the value of a top-level expression; xrdroot's driver prints
`(TFile *) 0x…`, which is line 1 of the difference in 6 tutorials and shifts
every later line. Find it in `src/xrdroot/cli/root.py` / `pycint`'s returned-value
printing (there is a `pycint/returned.py`) and gate it on ROOT's interactive/batch
rule.

### Wave 5 — gtest (the weakest kind; 209 of 350 programs fail)

| ID | task | tests | size |
|---|---|---|---|
| G-1 | `ROOT::TestSupport::FileRaii` and the rest of GoogleTest's ROOT support | 17 | S |
| G-2 | Translator test-body gaps: mutable lambda capture, `std::string::begin()`, `std::accumulate` with an op, `sizeof` of unknown types, container iterators, "changing this in the middle of an expression" | 25 | M |
| G-3 | `ROOT.Internal.*`, `ROOT.Detail`, `ROOT.MultiProcess`, `ROOT.std.true_type` | 12 | S |
| G-4 | `TTree::kOnlyFlushAtCluster` and other missing enum constants | 5 | XS |

G-2's messages are all `… is not something this translator turns into Python` —
each is a real `pycxx` gap with a precise message, and each blocks 2–5 programs.
Take them one at a time, add a case to `tests/pycxx/`, and keep
`tools/cint_survey.py` crash-free.

### Wave 6 — split readiness (parallel; 2 weeks)

| ID | task | why |
|---|---|---|
| S-1 | Transport as a plugin: `xrdroot.errors` owns its base class; `file.py` opens through a provider with a plain-`open` default | `core` currently hard-depends on `xrdclient` (`errors.py:12`, `file.py:19,26,58`); this is the last thing blocking a split |
| S-2 | Per-area `__all__` contract inside `pyroot` + a component rule that one area may not import another's internals | `pyroot` is 48 539 lines in one component bucket |
| S-3 | Generate `src/xrdroot/pyroot/__init__.pyi` from the API inventory | 845 dynamic names, invisible to IDEs and `mypy` users |
| S-4 | Cap the translation cache (LRU + size limit, `xrdroot cache prune`) | 771 MB and unbounded |
| S-5 | Delete `xrdroot.cint` (the 68-line shim) and stop mirroring ROOT's private `_distrdf` modules | roottest imports ROOT's internals; keep only what a test needs |

---

## 5. Housekeeping (do these as you pass; none is urgent)

- **H-1 — AGENTS.md's "What remains" list has drifted.** Verified at handoff
  (`hasattr(xrdroot.pyroot, name)`): `TTask`, `gVirtualX`, `TGeoManager` are
  present; `fopen`/`fgets` exist in `cppstd/cstdio.py` but are **not reachable
  as ROOT names** (4 roottest + 3 tutorial tests want them); `THnSparseD`/
  `THnSparseF` are **not** in the namespace at all (only the `sparse.py` engine
  is, so 3 roottest tests fail on the name). Still genuinely missing:
  `TMultiLayerPerceptron`, `TSpider`, `TAnnotation`, `TGraphTime`, `TPrincipal`,
  `RooPolyFunc`, `RooSimWSTool`, `RooStats::SPlot`, `TF1Convolution`,
  `RooLagrangianMorphFunc`. Regenerate the list from the harness reports and
  this document rather than editing it by hand:

  ```bash
  PYTHONPATH=src .venv/bin/python -c "import xrdroot.pyroot as R; \
      print([n for n in ['TTask','gVirtualX','fopen','fgets','THnSparseD'] if not hasattr(R,n)])"
  ```
- **H-2 — `docs/history/agent-status-2026-10.md` (805 lines) is
  process narrative** — audit hashes, per-prefix gate proofs, resource unions.
  It contradicts `AGENTS.md`'s own "do not build receipt/audit-chain machinery"
  rule and it will mislead the next agent into repeating it. Delete it or reduce
  it to the dozen decisions that still matter.
- **H-3 — dependency hygiene.** `xrdclient` is pinned to a git SHA (which pulls
  `xrdcore`); publish and version it, add dependabot/renovate. No `CHANGELOG`,
  no `CONTRIBUTING.md`, no `pre-commit`, no release workflow yet.
- **H-4 — the monolithic guide.** `docs/root.md` is 3 915 lines of one page.
  Split it by area and add a generated API reference (`mkdocstrings` is already
  configured and currently barely used) when S-3 lands.

---

## 6. Open decisions for a human (do not decide these alone)

1. **Pinned parity environment vs re-baseline** (W0-1) — this rewrites what
   "no PASS may be lost" means from today forward.
2. **100 % line+branch coverage over 109 063 statements** — it is the reason
   every new ROOT class costs a test. Options: keep as is (status quo, proven),
   or move the `pyroot` adapters to diff-coverage (they are covered end-to-end
   by the harnesses). Do not decide by lowering `fail_under` silently.
3. **How far to go on byte-exact file layout** (Q-20). Reproducing ROOT's exact
   file sizes is a large, well-defined project. If the answer is "not now",
   record it as an explicit scope note rather than leaving 18 tests failing for
   an unstated reason.
4. **Which optional Python packages the harness installs** (`torch`, `keras`,
   `tensorflow`) — each unlocks ~5 tutorials and costs gigabytes in CI.
5. **Windows** — nothing here is tested there; decide whether it is in scope.

---

## 7. Progress ledger (fill in as you go)

Keep this table in this file; it is the one piece of project state that is
cheap to maintain and impossible to reconstruct.

| task | status | commit | tests moved | notes |
|---|---|---|---|---|
| Q-9 | done | `da13415` | gate | the chisquare likelihood's last bit is `refmachine.roots()` with `NLL_REL`, so the unit suite has no failing test on arm64 |
| Q-6 | done | `aeab6a7` | perf + file size | zlib level 1 by default: a written file's header says 101, as ROOT's `TFile` default does; 1.8 → 4.0 M rows/s on a two-column tree |
| Q-2, Q-4, Q-5 | done | `7aa93f9`, `8e83d8e5` | +23 pyunittest cases | `TArray::fArray`; `TH1::SetContent` and `GetIntegral` as ROOT keeps `fIntegral`; `RNTupleWriter`/`RNTupleReader` context managers, `entry["f"]`, the cloned model |
| Q-3 | done | `95f4f41` | +20 pyunittest cases | `char[N]` is text; a `TLeafC` of `char[n]` reads as cppyy's char array proxy; the writer keeps `label[10]/C` and `fLen` 10 (ROOT reads the file back, checked against the oracle) |
| Q-12 (part) | done | `95f4f41` | +1 roottest | a macro's `Cell` knows its C++ type, so `float n` is a `Float_t` branch; `SetBranchAddress` answers `-2` for a buffer of another type |
| Q-8 | done | `f9b593a` | +2 pyunittest files (+6 cases) | `ROOT.__name__`, `PyConfig.IgnoreCommandLineOptions`, `TContext(nullptr)`, `FromCSV`'s options, `Take` of strings, `TGraphBentErrors`, `ROOT.vector` |
| Q-1 (part) | done | `867410f` | roottest 1038 → 1043 | `TClass::GetClassInfo` and `ClassInfo`; gained `cling/stl/map/stringMap`, `badstringMap`, `cling/stl/vector/Run`, `io/perf/julius`, `treeformula/scan/FormattingWrite`, none lost |
| Q-1 (part) | done | `7da3a87`, `ddc6039` | other +2, `test/` +2 | `TMath::LnGamma` is `::lgamma` at a negative argument (ROOT's own `mathcore-testSpecFuncGamma` exits 0 now); `gDebug`, `TROOT::GetVersionDate`/`GetVersionTime` |
| Q-7 (part) | done | `ddc6039` | 1 | `gDebug`; `fgets`, `TCpu`/`Architecture`, `TEnum`, `TTreeFormula`, `TSelector` and the TMVA DNN surface are what the next block of `other` tests waits on |
| Q-7 (part) | done | `32f16f4` | roottest +6, pyunittest +1 file | `std::complex` (with `<complex>`'s functions, dispatched by the argument's type), `std::atomic` and its `atomic_int` spellings, `std::forward_list`, `std::multiset`, `std::make_pair`, `std::map::emplace` in all three forms, and `RNTupleWriter::Recreate` taking a list of `make_pair` fields |
| Q-7 (part) | done | `31c7441` | roottest +4 | `TClassEdit::ShortType`/`GetNameForIO` (with `TClass` naming a class by its short spelling), `TInterpreter::GetMapfile` (null - there is no rootmap to walk), `GetInterpreterTypeName`, `TSystem::LoadAllLibraries`, `std::list::sort/unique/reverse/remove` |
| W0-8b | done | `8f53999` | triage | a failure's reason prefers the line that names a diagnosis over cling's `(int) N` echo, which was the whole reason of 37 roottest tests; rerun, they now say what they are: 21 are Q-20's written-file sizes (`Disk size of hsimple101.root should have been 41502`), 2 are `io/bill0*` checksizes, and the rest are one each |
| Q-1 (part) | done | `57604e1` | roottest +1 | `TObject`'s `EStatusBits` as names, an enum constant listed among a class's data members as C++ lists it, and `TClass::GetDataMember` looking in the bases |
| W0-1 … W0-8 | todo | | | the rest of the harness work is untouched. Next: `baseline.json` is still `10e0e58`'s, which is a subset of today's PASSes (so the ratchet cannot regress), and `python -m tools.rootsuite baseline RUN/results.json` wants the four oracle-free kinds in one run |
| P-6 | todo | | -4 | `xrdroot cp` and `hadd` of a `root://` file read it basket by basket through the transport: `roottest/main/RootcpReadRemote` took 174.8 s when the server was quick and over 300 s twice since, which is the whole reason 4 tests are missing from "now". The same file copied *locally* takes 1.0 s, and reading it whole over the network takes 2.1 s, so the cost is ~650 round trips - the fix is to copy a key's record whole (`cp` has no reason to rewrite a tree it is not cutting) or to batch the transport's reads |
| Q-16 (part) | done | `52420ee`, `0ce283c`, `f533e36`..`551c6a6` | roottest +6 | `std::array<T,N>` of a number, of a class and of a `std::array` laid out and written as ROOT writes one (oracle-checked), `std::complex` members, `std::array` brace elision, and the `T a[N]` twin of each |
| Q-1 (part) | done | `084cfd5`, `57604e1` | roottest +2 | a class's `GetState` is ROOT's (`kInterpreted` 3, `kLoaded` 4) and a container's dictionary can be generated, so `meta/method/ReuseMethod` passes; `TObject`'s status bits are data members and `TTreeReaderValue` of a class branch is the object (`stdarray/colWiseRead_11/12/21/22`) |
| Q-13 (part) | done | `5ae0a0f` | roottest +1 | `TObjArray::Delete` runs the destructors of what an owning collection holds, and a reference member is a cell |
| Q-7 (part) | done | `35560bb`, `faba133` | roottest +1 | `fgets` and `sscanf`, `TBufferJSON` (`io/json/Map`), `TFile.SetReadaheadSize`/`GetCacheRead`, `ROOT::TestSupport::FileRaii` (17 gtest programs reach it) |
| Q-17 (part) | done | `0b481bb` | roottest +1 | a doxygen trailing comment (`///<`, `//!<`) documents a member instead of making it transient |
| P-6 | todo | | -0 | unchanged: `cp`/`hadd` of a `root://` file still reads it basket by basket. It passed this run (195 s) and timed out on others |
| Q-1 (TEnum) | **attempted, blocked** | — | 0 | worker 2 found `TClass::GetListOfEnums` needs `ROOT::GetListOfEnums`/`GetListOfGlobals` and the enum table the interpreter keeps, which lives outside `reflection.py`; `roottest/root/meta/enums/fclassVal` also needs a pointer cast of an object |
| Q-12 (double32) | **attempted, blocked** | — | 0 | worker 8: the failing line is `gROOT->GetClass("MyVector<Double32_t>")->GetStreamerInfo()->ls()` — `TClass::GetStreamerInfo` is missing, so the fix is in `reflection.py`, not `interp.py` |
| Q-16 (rest) | done | `68a8d07` | roottest +2 | a member of a *class template* kept by value is written as an instance: `colWiseWrite_3` and `arrTemplMemberWrite` pass, and `colWiseWrite_4`/`colWiseRead_31..42` move on. Found by worker e, one line, in a file outside its scope |
| Q-7 (rest) | done | `e554a93`, `f64894e`, `fa63558`, `270eea2` | roottest +2 | `TBufferJSON`'s constructor bits and name (`io/json/PolyMarker`); `fgets`/`sscanf` writing into a macro's `char[]` (`treeformula/string/cernbuild` passes, and the three `stressgraphics` tests now reach `TPostScript`); `TClass::HasInterpreterInfo`; a tree never written counts its streamed record in `Print`'s Total |
| Q-20 (part) | done | `90f227e` | 0 | `hadd` now carries the streamer infos its inputs describe (merging `tests/data/tntuple.root` lost `TTree v20`, `TBranch v13`, `ROOT::TIOFeatures`, `TBranchRef`, `TRefTable`; now 18/18). It moves nothing in `io/filemerger`, and worker a measured why: the input there is xrdroot's own `hsimple.root`, 412 079 bytes where ROOT's is 414 892 — the gap is in what **the writer** puts in `hsimple.root`, not in the merge |
| Q-11 | **attempted twice, not merged** | branches `w01`, `xc` | 0 | workers 1 and c taught the formula engine to call a branch object's methods - 27 `FAIL`s became refusals naming the reason (`fH.GetMean()` on an object read from a file is a dictionary of members), and `TH1::GetMean` became a method derived from the sums a file's object holds - but `treeformula/sync/readfile-0`, a PASS, becomes a refusal in both, so neither branch is merged. The refusal to relax is `formula/binding.py::_mismatch` for `fTracks.fVertex[][fTracks.fNpoint%3]`, which indexed a loop by a value that changes and the older code handled |
| Q-14 (GDML) | **attempted, blocked** | — | 0 | worker f: all 20 `io/geo` refusals come from `TGeoManager::Import`, which needs a GDML reader and a reader for a `TGeoManager` in a ROOT file; both are absent, and the refusal it already gives names them |
| Q-1 (part) | done | `e0ab124`…(reverted `5e9076e`) | 0 | `TClass::GetStreamerInfo`/`GetStreamerInfos` landed and turned 40 `pairEnumEvo` crashes into output, then was **reverted**: it printed `offset=0` and the `i=` action lines for a layout only a file describes, where ROOT prints `version=N` and `offset=na` with no actions, and `io/evolution/versions/AddVersion-2-1/2/5` went PASS → DIFF. The printer must tell a layout the process has a dictionary for (no `version=`, `offset=na`, actions listed) from one only the file describes (`version=N`, `offset=na`, none). See the queue row below |
| Q-7 (part) | done | `8491022`, `bfd2911`, `c006793` | roottest +6 | genreflex's `--selSyntaxOnly` (`meta/genreflex/XMLParsing/0..4` pass); a tree's object where its column of numbers goes is refused by name; `std::array<T,N>` as an element type (`io/stdarray/aclicModelWrite` passes) |
| Q-13 (part) | done | `038038b`, `cb88a7a` | roottest +1 | a pointer member initialised with `0` is the null pointer (`tree/evolution/WriteClones` passes), and a class's streamer record names the base its member is of |
| Q-15 (part) | done | `8dc4156`, `43a287a` | 0 | an RNTuple field of a nested vector, laid out as ROOT lays it out; a field made from a type name |
| Q-11 (part) | done | `6b16368`, `a3bf9b6` | 0 | the formula engine reads a branch object's members and their getters, and works out an index that loops inside the loop the branch makes; `roottest/root/treeformula/sync/readfile-0` stays PASS and now reaches `sync.C:70` (`Clones` has no `At`) instead of `:37` |
| Q-12 (part) | done | `bbdbec1` | 0 | a reference initialised with the object it already names builds nothing (`typeid(T)`'s `const type_info&`) |
| Q-16 (part) | done | `7a80af7` | 0 | an array of `std::vector` is the STL element ROOT lays out at `kOffsetL`; `io/bigevent`'s refusal moved eleven members down the class (to `fVectorTobject`) |
| P-7 | todo | | 8 | **the seven hangs**, diagnosed by a worker: `io/TFile/assert_tcontext_thread_hoping` (a dead thread; `std::atomic::exchange`), `root/multicore/testSetAddressLoop` (`TBranch::GetAddress` returns ...), and five more in its table at `/tmp/r1/`'s reports; each costs 300 s of wall time and hides its real failure |
| P-8 | todo | | | `std::atomic::exchange` and `TBranch::GetAddress`, the two named causes of hangs above — check names first, the first may already exist on `_Atomic` and be missing on another type |
| G-5 | todo | | 4 | `emit_vars.VariableEmitter._binds_an_object` is right; the *template choice* (`_template_of`/`_template_taking`, `emit_templates.py`) picks the template whose parameters take as many arguments as the call passes without checking that each argument's type binds its parameter, so an instantiation with a `char*` parameter was chosen and `std::numeric_limits<char*>::epsilon()` was refused (`execSourceTypes-build` PASS → UNSUPPORTED, reverted at `7a43bd9`). Add the type check, then retry worker 15's target |
| G-6 | todo | | 2 | a member template constructor of a non-template class (`template <class T> WithTemplate(T*)` in `io/alloc/customConstructor.C`) needs `emit_classes` to instantiate a constructor per call; worker 14's patch did it but regressed `test_a_pointer_member_initialised_with_zero_is_null_not_the_number` |
| Q-23 | todo | | 0 | `TStreamerInfo::ls` and `TStreamerElement::ls` must print what ROOT prints (`roottest/root/io/evolution/versions/*.ref`): a dictionary-less class shows `version=N` and `offset=na` and no `i=` action line; a class with a dictionary in memory shows no `version=`, `offset=na`, and the actions. This is the one change that would let `pairEnumEvo`'s 40 tests compare output, so it is worth a worker of its own (`reflection.py`) |
| W0-9 | todo | | | harness: the `io/hadd` tests write `src1.root`/`dest.root` in one working directory, so at `--jobs 5` they collide — `test_MergeNestedSameName` read a file another test was writing and went PASS → DIFF, then passed alone. Give each test of a directory its own working copy, or serialise that directory |
| Q-27 | **done in part** | `c6c380f` (worker 2) | roottest +5 | `CloneTree(-1,"fast")` never reads the objects: it moves the basket bytes and keeps the source file's class description, which is now what `copying.py` does (`Streamed` slot, bytes from `basket.start_of`..`end_of`, a cut picks rows by `Entry$`). `fastcloningeventtree/clone-{split0,split1,split9,streamed}` and `fastcloning/runabstract-datageneration1` pass, and 20 more `draw-*`/`reference` reach the macro instead of a refusal. A class ROOT streams with its own streamer (`TLorentzVector`) is refused in a sentence: writing its bytes back under a member-by-member description read as another shape (verified) |
| Q-15 | **done in part** | `3683a1d` (worker 4), `16b3766` | roottest +5 | `TRefTable`/`TBranchRef` exist in memory: `BranchRef()` makes the branch (not in `GetListOfBranches`, as ROOT), `Fill` records the row, and `Write` refuses a tree whose table has rows rather than writing it without one. The twelve `io/tree/create-r`/`test-r` tests move two layers down (to splitting a `TClonesArray` of a macro class), and `io/tree/copy` plus the four `treeformula/references/new.split*` now PASS because a `TRef` row can be counted. Worker 4's writer patch - exact, disabled - is `/tmp/r3/w04-patch.diff`; it needs `winfo.INFOS` entries for both classes and a `_ref_branch` in `wtree.py` |
| Q-20 | **measured exactly** | `a5c3163` (worker 8) | 0 | the 2 811-byte gap between xrdroot's `hsimple.root` (412 081 B with the ntuple-title fix; ROOT's is 414 892) is accounted for byte by byte: 120 B of file title (`writer.py` writes `("TFile", label, "", label, "")`), 2 268 B of twelve missing `StreamerInfo` entries, 346 B of the drawn `TH1F`'s `TPaveStats`, 48 + 11 B of a `float` vs `double` `pz` (a *translator* difference: 3 923 of 25 000 entries differ in the last bit), 18 B of `TTree v20`/`TBranch v13`. **Eleven `io/filemerger` FAILs are exactly these bytes.** None of it is in `wpacking.py`/`wtree.py` |
| Q-24 | **done in part** | `63852cf` (worker 5) | 0 | a versioned class's layouts are listed in version order, as ROOT keeps them at their version's index - the `reflection.py` half of `AddVersion-1-4`. The 40 `pairEnumEvo` tests stay DIFF and the diagnosis is now exact: see `Q-28` and `Q-30` |
| Q-14 | **done in part** | `fe937cd` (worker 7) | 0 | `geom/gdml.py` reads a GDML file's tables, solids and placed volumes and passes its own tests; what `io/geo` still needs is `TGeoManager::Import` wired to it, ROOT's eleven `Info in <TGeoManager...>` lines verbatim (they are in `/tmp/r3/w07.log`), `Export`, and a reader for a `TGeoManager` written in a ROOT file. Any partial support only moves the failure, because the compared `.ref` is the GDML one |
| G-2 (part) | done | `34d6a73` (worker 15) | roottest +1 | `roottest/utils-build` passes: a forward `goto` inside one function is the `while True:`/`break` a retry is, and an increment of an increment translates. `VectorSort`'s `v[i](x) = ...` moved to a *place* refusal (it needs `cppstd`), and the six `TSeqTests`/`coreconttestUnit` tests moved to `ROOT has TListIter` / `'TSeqI' object has no attribute 'begin'` |
| Q-32 | done | `80c109c` (worker 13) | other +6 | a RESHAPE layer is implemented rather than refused: the five `other/tmva/tmva/DNN/TMVA-DNN-MethodDL-*-Cpu` and `TMVA-DNN-Regression-Cpu` pass |
| Q-11 (part) | done | `9cb6d31` (worker 1), `d5b0caa` | 0 | a getter whose class names no such member (`Event::GetHistogram` -> `fH`) is the member the rest of the name reads, tried against each object the expression has reached and only when the rest of the name reads something off it; and `Draw("event.GetHistogram().GetXaxis().GetXmax()>>h")` no longer dies in `ast.literal_eval` before the formula engine is asked. No PASS moved: `treeformula/event`'s fixture now gets from `dt_DrawTest.C:17` to line 113, where `'fMatrix' has 1 dimension and is given 2 indices` is the next gap |
| Q-25 (part) | done | w11's patch, `f2977d4` | roottest +4 | a pointer read by the class it was declared with (`BaseABC *fABC` holding a `Derived`) gives back the derived class the stream named: `buffer.py` keeps the name in front of a record and `pyroot/core/declared.py` makes the object of that class. All four `tree/branches/abc-*-read` pass |
| P-7 | **done in part** | `16b3766` | 2 hangs | `std::atomic::exchange`/`compare_exchange_strong` and a `_Whole` that answers with the object it holds end the two deadlocks worker 14 found: `io/TFile/assert_tcontext_thread_hoping` gets past its thread and now FAILs on its own subject (`One of the two file is not deleted`), and `multicore/testSetAddressLoop` completes a run in 53 s where it used to hang - 33 runs of speed work now, not a deadlock (`P-10`) |
| Q-16 (part) | done | `d8f37c2` (worker 9) | roottest +1 | an RNTuple record is a field per member with no column of its own, so `ntuple/atlas-datavector/write` passes and `evolution/write_ntplevolv` fails one step later (`'dict' object has no attribute 'fA'`) |
| W0-10 | **resolved by decision** | - | 0 | worker 16 measured the premise wrong: reverting `27791f5` gives 9 PASS + 4 FAIL + 7 DIFF where HEAD gives 11 PASS + 8 FAIL + 1 UNSUPPORTED, and the seven `tree/stl` DIFFs turn into failures because of the `.typedef`/`.autodict` *command*, not `execute.load`'s `remember_source` line (removing that line alone changes only `cling/typedef/runptrptr`). **Two PASSes beat seven DIFFs**, so the commit stays; the cause is now known |
| Q-34 | attempted, not merged | w12's `/tmp/r3/w12-patch.diff` | 0 | worker 12's `TObjArray` writer patch moved `io/datamodelevolution/00/test1` from `required argument is not an integer` to `'list' object has no attribute 'Print'`, but its `_array` writes ROOT's `TObjArray::Streamer` **without the `TNamed` title string**, so the bytes are wrong; ROOT's own 88-byte stream is `/tmp/r3/oracle-objarray/oa.bin`. Not merged, with the reference bytes kept |
| Q-35 | todo | | 4 | a `TObject` inside a heap object carries `kIsOnHeap|kNotDeleted` = `0x03000000` and `TRef::SetAction` sets BIT(16) = `0x10000`, so ROOT prints `0x3010000` where `TObject._bitword()` starts at 0 (`io/references/actions/test` and its three dependants). Oracle-verified: `0x2000000`, then `0x2010000`, then `0x3000000` |
| W0-12 | done | `db5f021` (worker 13's patch) | roottest +4 | cling's preprocessor pass over a macro finishes before the macro's statements run, so a `#define` it makes is in force for the `.x` it starts itself - and the file's own `#ifndef`s read the pass's before-state. xrdroot kept nothing, so `produceLittleFile.C` re-read itself to the recursion limit; keeping it before the run instead made the file skip its own `.L`. The dialect is now taken before the defines and handed to the run. All four `io/fakeClass/*` tests pass |
| W0-13 | done | `932164a` (worker 16) | roottest +3 | `ROOT.cmdLineUtils` exists as a Python module - the one ROOT's `rootls`, `rootcp`, `rootmv` and `rootrm` are written over - on top of xrdroot's own subcommands: `main/SimplePattern{1,2,3}` pass |
| Q-20 (part) | done | `bcc03ba` (worker 1) | roottest +2 | the file's title is written where ROOT writes it (four times over, 4 bytes per character, so 120 B for `hsimple`), ten more streamer layouts are harvested from ROOT 6.40.04's own `hsimple.root`, and `TBranchRef`/`TRefTable` are described in every file holding a tree - which is why `io/newClassDef/noclassdef/driver-build` and `io/stdarray/modelCheckValues` now pass. The eleven `io/filemerger` FAILs still miss the `TPaveStats` 346 B and the `float`/`double` `pz` 59 B |
| Q-36 (part) | done | `f6374c2` (worker 6) | 0 | the reader half: `streamers.read_layouts` keeps a class's layouts per version (a merged file no longer collapses them), `streamers.reconcile` applies ROOT's rule - same version, different checksum, so the *in-memory* layout reads - and speaks its warning verbatim, and `Buffer.check_count` is `TBufferFile::CheckByteCount`'s consumed-vs-recorded pair. 8 unit tests. The wiring in `interp.py` is the next step |
| Q-15 (ntuple) | done in part | `a5b6dee` (worker 9) | 0 | `std::bitset<N>` is a field type the RNTuple writer declares, laid out as ROOT writes it and checked both directions against the oracle. `makeproject/rntuple` still needs the class-typed field before it |
| P-9 (part) | done | `f113afb` (worker 14) | 0 | `pack_member` writes one number through a `struct.Struct` memoised per streamer type (0.45 us -> 0.24 us per member), with the NumPy path kept for columns and for every value `struct` refuses; tests pin 15 streamer types byte-for-byte against it. The four `io/perf` tests still time out - the remaining 825 s / 2 430 s is the per-record description rebuild, which is `P-9`'s own item |
| Q-28 (part) | done | `6b1eee4` (worker 15) | 0 | `RootClassVersion` exists and a compiled library no longer sees `__CLING__`, so the header's version block compiles as ROOT's does. `AddVersion-1-3`/`-1-4` stay DIFF: the version now reaches the library but not the `Declaration` the writer emits |
| Q-30 (part) | done | `06101b4` (worker 2's patch) | 0 | a fixed array keeps the shape its file declares: `Member.dims` holds `(4, 4)` for `Double32_t fMatrix[4][4]`, and the interp step reshapes a fixed array to it, so a formula indexing `m[1][2]` no longer refuses. The `treeformula` fixture moves from `dt_DrawTest.C:113` to line 32, where `==` between a string and a literal is the next wall |

| Q-28 (part) | done | `cef312f` (worker 13's patch) | roottest +3 | a file written before a class had a version is listed above the library's: ROOT records version 1 for a class whose library had none when the file was written, so that layout belongs above the in-memory one instead of colliding with it (`AddVersion-1-3`, `-1-4`, `-2-3`). A compiled library's translation takes the dialect that does not define `__CLING__`, so `RootClassVersion` is seen. Its `_free_version` split keeps the function under the maintainability limit; its dead `IsForeign` branch removed |
| W0-14 | done | `f583422` | roottest +1 (restored) | the dialect change above cost one PASS - `cling/other/runfileClose`, whose header takes `&x` in `x`'s own initialiser and whose `gSystem->Load("fileClose_C")` had no library to find. A static that takes its own address is now made before it is filled, and `Load` of a name ACLiC gave a library builds the macro it came from, as ROOT does. **The one PASS round 5 lost is back** |
| P-9 (part) | done | `6ab5b1b` (worker 7) | roottest +2 | a declared class is described once, not once per record: `io/perf/WriteBuffer-1` and `-2` pass inside the 300 s limit (they were TIMEOUTs every round until now). `ReadBuffer-1/2` still time out - the read direction rebuilds the layout per record - and that is the rest of `P-9` |
| Q-41 | done in part | `10faa1e` (worker 5) | roottest +1 | `.Class` prints a class's layout, which takes four `cling/operator`/`template` tests from FAIL to DIFF, and a template template parameter is parsed, which is `cling/dict/exectemplatetemplateTest` passing (`Q-47` is what the DIFFs still want) |
| Q-38 | done | `efe41a6` (worker 2) | 0 in this run | a clone writes a split `TClonesArray` as the one branch ROOT writes, on top of worker 4's `Each`-is-a-`Values` bug: `CloneTree(0)` no longer refuses. The twelve `io/tree` tests are one layer further on now (the writer's `SplitClones` for a class the file describes) |
| Q-39 | done | `169c6cf` (worker 3) | 0 | a `TString*` counted by a `//[n]` comment is ROOT's `kStreamLoop` - as many strings as the count says; `io/bigevent` now gets past `fTstringp` to the next refusal (a template member whose spelling `find_class` does not match) |
| Q-43 (part) | done | `cee27a7` (worker 8) | 0 | a ROOT collection read back is the namespace's collection, not the list it read as: `TList`/`THashList`/`TSortedList`/`TObjArray` members come back as objects, a `TObjArray`'s holes as holes |
| Q-35 (part) | done | `f2b74f2` (worker 12) | 0 (1 with the patch) | a `TRef`'s heap bits, its action bit and in-place assignment: `io/references/actions/test` prints `0x3000000` where it printed `0x0` and needs one more line (`TRef::operator=` on assignment, worker 12's `w12-ASSIGNED.patch`) |
| Q-15 (ntuple) | done in part | `f313b4e` (worker 9) | roottest +1 | an RNTuple field of a `std::set` or `std::map` is written as its items: `makeproject/rntuple/write_rntuple` passes; the read side needs `TFile::MakeProject` (`Q-46`) |
| W0-15 | done | `684763f` (worker 16) | 0 | the harness's `WILLFAIL` inversion is one function in `judge.py`, shared with the pyunittest path, and a `WILLFAIL` test that xrdroot *refuses* or that times out is still xrdroot failing - so `cling/bytecode/runarrayhist` is PASS and nothing can pass by not running |
| Q-20 (part) | attempted, not merged | - | 0 | worker 6 measured that neither of the two remaining rows is in the files its brief named: the `TPaveStats` 346 B is the histogram *payload* the writer never builds, and the 59 B of `pz` is the translator's `float`-vs-`double` narrowing. Both need a worker next round with `wobjects.py`/`wdeclared.py` and `pycxx/emit_expr.py` |
| Q-42 (part) | attempted, not merged | - | 0 | worker 1 took the formula engine seven walls further (a `char[N]` member of a whole object is the string it holds, so `fType == "type1"` and `strstr` work) and the fixture now dies at `dt_DrawTest.C:17` on `event.fTriggerBits`, which is an *object* the expression reads as a number (`Q-45`). The 42 targets stay blocked |
| Q-14 (GDML) | attempted, not merged | - | 0 | worker 4's patch for `TGeoManager::Import`/`Export` is at `/tmp/r5/w04.log`; it needs 21 tests' worth of ROOT's `Info in <TGeoManager...>` lines and a `TGeoManager` file reader (`Q-44`) |
| Q-40 (part) | attempted, not merged | - | 0 | worker 10 read `treeproxy/vectorint` down to `Branch()`'s deduction of a `std::vector<std::vector<int>>` and its patch is in `/tmp/r5/w10.log`; blocked on `pyroot/trees/tree.py`'s `_declared` |
| Q-36 (part) | attempted, not merged | - | 0 | worker 11's wiring patch for `interp.whole_object` (using `streamers.reconcile`) is in `/tmp/r5/w11.log`; it needs `rdobjects.py`, which another worker owned |
| W0-16 | attempted, not merged | - | 0 | worker 14's four-line patches (`interp.py`'s version-0 header, `cppstd/bitset.py`'s `set`/`reset`/`test`, `TCollection.Add(obj, opt)`) are in `/tmp/r5/w14-patches.diff`; each unblocks a different `io/json`/`io/abstractclass` test |
| W0-17 | attempted, not merged | - | 0 | worker 15 read `roottest/python/memory/memory`'s three failures; its report is `/tmp/r5/w15.log` and the fix is in ROOT's Python-side memory accounting, which nothing here implements |

| P-9 | **done** | `267e974` (worker 2) | roottest +2 | a declared class is read by a plan made once, not one made per record: `io/perf/ReadBuffer-1` and `-2`, TIMEOUTs in every round until now, pass (221.6 s and 172.1 s of the 300 s limit). With round 5's write half, `io/perf` is four PASSes from four TIMEOUTs |
| Q-44 | **done in part** | `ce4c62a` (worker 8) | roottest +1 | `TGeoManager::Import` reads a GDML geometry as ROOT's `TGDMLParse` does - `io/geo/Parse` passes with stdout byte-identical to `execParse.ref`. The other twenty `io/geo` tests need `Export` and a `TGeoManager` file reader |
| Q-46 | **done in part** | `24a51fb` (worker 9) | roottest +2 | `TFile::MakeProject` writes the classes a file describes as a project: `makeproject/rntuple` and `read_rntuple-build` pass; `read_rntuple` itself fails its own gtest (`RNTupleMakeProject.ReadBackRNTuple`) |
| Q-35 | **done** | `330c55c` (worker 12) | roottest +3 | a `TRef` read inside an object gets the process its record names, and `TRef`'s heap bits and action bit are set: `io/references/actions/{test,test1,test2}` pass, `test3` is one call away |
| Q-16 (part) | done | `43ec833`, `8190689`, `7ba79e1` (workers 3, 6, 10) | 0 | a list may hold a ROOT object this writer has a layout for; a bitset's `set`/`reset`; `TCollection::Add(obj, opt)`; a `TObjArray` comes back with the name ROOT read into it from the bytes |
| Q-41 (part) | done | `00ab886` (worker 14) | roottest +1 | `fscanf` scans a file's fields on from where the call before it stopped: `cling/bytecode/runhenry` passes |
| Q-42 (part) | done | `88bb607` (worker 1) | 0 | a whole `TBits` member drawn a set bit at a time, and `TestBitNumber` over its bits: the `treeformula` fixture moves from `dt_DrawTest.C:17` to `fTriggerBits` read as an object (`Q-48`) |
| Q-29 | **attempted, reverted** | `f156b59`, reverted by `40036da` | 0 | `TStreamerInfo::BuildCheck`/`CompareContent` made `AddVersion-1-6` pass but printed a warning two PASSes' `.ref` files do not have (`io/evolution/ClonesArrayEvo`, `treeformula/clones/nested-2`), because an old file spells `Int_t` under a code today's ROOT has dropped (`Q-49`). Narrowing the gate to the member's *type name* does fix those two - measured, `/tmp/r6/check-lost5` - but leaves `AddVersion-1-6` short by its `offset=na` listing, so the warning is reverted and this is the shape the next attempt starts from |
| Q-24 (part) | attempted, not merged | - | 0 | worker 11's `CompareContent` also needs `offset=na` in `TStreamerElement::ls` for a file-only layout, which is `Q-30`'s printing half |
| W0-18 | attempted, not merged | - | 0 | worker 13's bigevent template member is pinned by its own test (`647f8e9`) but the writing half needs the template spelling fix in `pyroot/core/declared.py` (round 4's worker 3 patch); its measurement is `/tmp/r6/w13.log` |
| W0-19 | done | `267e974` | 0 | worker 15's `tree/stl` seven DIFFs: the `.typedef`/`.autodict` command's effect on a split branch was measured and is unchanged by anything in `pycint/execute.py`; the row stays `W0-10` with its numbers |
| W0-20 | attempted, not merged | - | 0 | worker 16 read `roottest/python/memory/memory`'s three failures (`/tmp/r6/w16.log`); they ask for ROOT's Python-side memory accounting, which nothing here implements |

| Q-29 | **done** | `9375b4e` (worker 2) | roottest +1 | `TStreamerInfo::BuildCheck`/`CompareContent` are back, judged by the member's *width* the way round 6 measured (`Q-49`: an old file spells `Int_t` under a code today's ROOT has dropped, so the element parse normalises it first). `AddVersion-1-6` passes and the two PASSes round 6 lost - `io/evolution/ClonesArrayEvo`, `treeformula/clones/nested-2` - still pass |
| Q-44 | **done in part** | `fe27ac6` (worker 3) | roottest +1 | a `TGeoManager` written in a ROOT file is read back: `io/geo/readGeo` passes. `Parse` (GDML) was round 6's; the twenty `export_*` tests still need `TGeoManager::Export` |
| Q-46 | **done in part** | `95a48b2` (worker 14) | 0 | an RNTuple record is read back as an object of the class its file keeps - the read side of `MakeProject`; `read_rntuple`'s own gtest still fails on what it compares |
| Q-41 (part) | done | `5541b90` (worker 15) | 0 | a selector's reader takes its class and its kind from what ROOT would type it by: `tree/selectorreader/CollectionClasses` moves on, `Classes` stays refused by name |
| Q-40 | attempted, not merged | - | 0 | worker 6 showed `treeproxy/vectorint`'s branch is a collection of collections, and made it refused *by name* rather than given a leaf list (`b6d919a`); the tests it needs are behind a `Branch()` the files it may edit do not hold |
| Q-48 | attempted, not merged | - | 0 | worker 1 measured the formula's next wall (its patch is `/tmp/r7/w01-prototype.patch`) and found it outside `formula/`: a whole-object member drawn as numbers needs `pyroot/trees/methods.py`'s `_called`, which reads a cell's value |
| Q-38 (part) | attempted, not merged | - | 0 | worker 4 needs three files outside its scope for `SplitClones` - `/tmp/r7/w04.log` has the list and the measurement |
| Q-39 | attempted, not merged | - | 0 | worker 5: `io/bigevent`'s blocker is the template spelling in `pyroot/core/declared.py` (its report is `/tmp/r7/w05-report.md`) |
| Q-36 | attempted, not merged | - | 0 | worker 7 measured that neither `interp.py` nor `rdobjects.py` moves `meta/evolution` or `datamodelevolution`, and said where the walk is instead (`/tmp/r7/w07.log`) |
| W0-21 | attempted, not merged | - | 0 | worker 8: `makeproject/ttree`'s `_Set` needs `pycxx/emit_expr.py` and `cppstd/containers.py` before `pyroot/trees/store.py` can bind it |
| W0-22 | attempted, not merged | - | 0 | worker 9: `io/abstractclass` and `io/double32` both need `TClass::GetStreamerInfo`/`DataBlock` answers that live outside `pyroot/core/files.py` |
| W0-23 | attempted, not merged | - | 0 | worker 10: `python/memory` needs a lifetime patch (`/tmp/r7/w10-memory-lifetime.patch`) in ROOT's Python-side accounting |
| W0-24 | attempted, not merged | - | 0 | worker 12: `test/stressgraphics` compares xrdroot's *encoder* output, so no change inside `pyroot/graphics/` can pass them (`/tmp/r7/w12-report.md`) |
| W0-25 | attempted, not merged | - | 0 | worker 13: all six `io/json` targets need `pyroot/core/jsonbuffer.py`'s container writers plus `TMap` (`/tmp/r7/w13.log`) |

| Q-45 (abstract) | **done** | `17c3e9e` (worker 8) | roottest +3 | a class whose version is zero is read as a version unless its record has room for a checksum: the three `io/abstractclass/read_test*` pass |
| Q-46 | **done** | `7d0502e` (worker 15) + `95a48b2` | roottest +1 | an RNTuple record is read back as an object of the class its file keeps, and a whole-object member of a split collection is a row of objects: `makeproject/rntuple/read_rntuple` passes, so all four `makeproject` tests do |
| Q-48 | **attempted, reverted** | `e9cb809`, reverted by `f65e991` | 0 | worker 1 put a method that gives back an object back to the formula, as ROOT draws one - right about ROOT, but the shape it left (`_divergence`'s closure in `interp.py`) has a branch no test without a real file can reach, so it was reverted rather than shipped under the coverage floor |
| Q-48 (formula) | **done in part** | `18af0e9` (worker 1) | 0 | `pyroot/trees/methods.py` now leaves an object a method gives back to the formula, which is what round 7's prototype asked for; the 40 `treeformula`/`fastcloningeventtree` tests move on to their next wall |
| Q-39 | **done in part** | `1ab88cb` (worker 2) | 0 | a template instance is one class under either spelling of its arguments (`pyroot/core/declared.py`); `io/bigevent` reaches the writer's next member |
| Q-36 (part) | **done in part** | `260be7c` (worker 3) | 0 | the version-0 header guard from `/tmp/r5/w14-patches.diff`, with the `reconcile` wiring still to come (its third hunk was reverted with `f65e991`) |
| Q-33 | **done** | `13507ab` (worker 4) | 0 | a braced list fills the hashed and multi- containers, and two containers compare by what they hold: `makeproject/ttree`'s first wall is gone |
| Q-44 (export) | **done in part** | `98f8b65` (worker 6) | 0 | a geometry's polycones and polygons are read as the solids their arrays make, and a sphere is not taken for a tube - the reader side of the twenty `io/geo/export_*` tests |
| W0-25 | **done in part** | `6b1d29d` (worker 7) | 0 | `TBufferJSON` reads an object back from its `_typename` and writes a member of declared lengths; the JSON targets move on |
| Q-41 (bigevent) | **done in part** | `a6c0ffc` (worker 11) | 0 | a member of a `TDatime` is a description this writer carries |
| Q-40 (part) | **done** | `70029ef` (worker 12) | 0 | a collection of collections is refused with the layout that does keep its sizes, instead of being given a leaf list |
| W0-26 | attempted, not merged | - | 0 | worker 9's `ROOT.SetHeuristicMemoryPolicy` (`c9edf96`) landed, but `roottest/python/memory` still needs the lifetime patch (`/tmp/r7/w10-memory-lifetime.patch`) applied with it |
| W0-27 | attempted, not merged | - | 0 | workers 5, 10, 13, 14 and 16 reported without a commit: `SplitClones` still needs `wclasses.py` (5), `stressgraphics` compares encoders (10), `selectorreader/Classes` needs a class the process has a dictionary for (13), `pycint/typedefs.py` did not move `tree/stl` (14), and `testSetAddressLoop`'s 53 s is the GIL's (16) - each report is `/tmp/r8/wNN.log` |

| P-10 | **done** | `4ce84ff` (worker 15) | roottest +1 | the last TIMEOUT is gone: a read of a translation's `std::atomic` hands the interpreter to the thread it waits for, and `multicore/testSetAddressLoop` - 33 runs of a script that used to take 53 s each - passes inside the limit. **No roottest TIMEOUT is left that is xrdroot's own** |
| Q-48 (part) | **done in part** | `4b423b6` (worker 2) + `2ef9c80` (reverted) | roottest +2 | the two reference fixtures pass (`treeformula/event/reference`, `fastcloningeventtree/reference`): the 40 draws behind each are still FAIL, at `fTriggerBits` read as an object. The counted-pointer half was reverted (`2ef9c80`) after it failed to help `io/bigevent` and muddied `tree/friend/file2`'s diagnosis |
| Q-16 (TEnum) | **done in part** | `8bff48c` (worker 16) | 0 | a class answers `GetListOfEnums` with the part of the session's enum table it owns; `meta/enums` moves to its next wall |
| Q-44 | **done in part** | `928fbd8` + `6448cd3` (worker 6) | 0 | `TGeoManager::Export` writes GDML ROOT's own reader reads back; `export_*` now has a writer to run against (its `.xrdlib` was left untracked in the worktree and is part of the landed commit) |
| Q-36 | **attempted, reverted** | `048d717`, reverted by `ce93de5` | 0 | worker 5's read-by-the-declared-layout is right about ROOT but turned three `meta/evolution` tests into failures and `io/evolution` into one, and gained nothing where it was aimed: measured with it reverted (`/tmp/r9/revert-run`), all four pass. The revert stands; the report (`/tmp/r9/w05.log`) names the real wiring |
| W0-28 | todo | | 1 | `tree/friend/file2` is `cmake -E copy Event.root Event2.root` where `Event.root` comes from a fixture *another* test writes - at `--jobs 2` in a fresh copy the file is there (the test passes), at `--jobs 5` it is not. A working-copy race the harness must order, not a code gap |
| W0-29 | attempted, not merged | - | 0 | worker 7 took `io/json/BasicTypes` to PASS in its worktree (`/tmp/r9/w07.log`) with the container writers in `jsonbuffer.py`; its commit was never made and the work is in the worktree, unmerged |
| W0-30 | attempted, not merged | - | 0 | workers 3, 4, 9, 10, 11, 12, 13 and 14 reported without a commit; each report (`/tmp/r9/wNN.log`) names the file it wanted, and together they are round 10's brief |

| Q-42/Q-48 | **done** | `87ac2d8`, `26bc629d`, `fe307d31` (workers 1, 2) | roottest +40 | the 40 `treeformula/event/draw-*` and `tree/fastcloningeventtree/draw-*` tests pass: a `TBits` inside a `TClonesArray` element is the bits it holds, `Alt$` reads its branch at the loop's flat element index (ROOT's rule, mean -0.021071 against xrdroot's old -0.301389), and a tree made only to hold friends reads as far as they do. Four worker-rounds of measurement (`/tmp/r7`…`/tmp/r9/w01.log`) went into these 40 |
| Q-38/Q-40 (writer) | **done in part** | `9955f67`, `e7b2784`, `ecb6457`, `38f45ec` (workers 3, 4, 5, 6) | 0 | a member held by value is one whole object written where it stands; a macro's class with a base and a `TRef` member is described for a split collection; a counted pointer is a loop of objects, laid out and written; the type a split member is held as is read from the file's streamers. The 12 `io/tree` tests are one layer further on |
| Q-41 (part) | **done in part** | `9134a49` (worker 8) | 0 | `TMap` holds its pairs by key and a clones array can be told its class - `io/json`'s `RootClasses` moves on |
| W0-20 | **done** | `a93ce61`, `56df50d` (worker 11) | roottest +1 | a translated object's destructor runs when its Python object is collected, with the memory lifetime patch: `roottest/python/memory/memory` passes (3 assertions) |
| W0-25 | **done in part** | `0a9c9d4` (worker 13) | 0 | `TMarker::DisplayMarkerTypes` draws ROOT's table of marker styles - `test/stressgraphics` moves to its next wall |
| W0-10 | **done** | `01d34f3` (worker 14) | 0 | `.U` forgets the typedefs of the macro it unloads, as cling does: the `tree/stl` seven still DIFF but the mechanism is now measured and pinned |
| Q-36 | **reverted again** | `422ed08`, reverted by `71aa3256` | 0 | worker 15's checksum-agnostic read is right in principle and wrong in this wiring: it turned `execBaseClass_v2` and both `baseClass_53410` tests into failures and fixed nothing. **Round 9 reverted the same shape from worker 5** - the next attempt must not touch `interp.py`'s walk without a like-for-like run of `meta/evolution` |
| W0-31 | todo | | 0 | **The ROOT source tree was damaged** (`/tmp/rootsrc-6.40`: `roottest/root/meta/evolution/` had 3 of 82 files) and every roottest figure up to round 9 came from it. The intact clone is `/tmp/rootsrc-6.40b`; `tools/rootsuite list` shows the same 2820 non-tutorial tests, so counts are comparable but individual statuses were not. **Always run with `/tmp/rootsrc-6.40b` from now on**, and re-derive any figure older than this round before quoting it |
| W0-32 | todo | | 1 | four workers died with `backend unavailable` (2, 7, 10, 12) and two reported without a commit; their briefs are unchanged and are round 11's by default |

| Q-39 | **done** | `d48bd3e1` (worker 1) | roottest +1 | a pointer to a container is written as the container it points at: `tree/evolution/WriteVector` passes. Round 10's worker 9's measured one-liner in `declared.py::_plain`, taken with its `and not found.stars` caveat |
| W0-25/Q-41 | **done in part** | `caf4d6cb` (worker 9) | roottest +1 | `TBufferJSON` writes the containers a class declares and reads back empty ones: `io/json/STL0` passes |
| Q-44 (geo) | **done in part** | `6d99995a` (worker 3) | 0 | `TBrowser` opens nothing in batch, as ROOT's own batch run says - the 17 `io/geo/export_*` tests now get into the geometry macros (worker 2 took the file writer's side: `dd674358`, ROOT's self-streamed collections written back as the file wrote them) |
| Q-16 (part) | done | `90f5c5e8` (worker 5) | 0 | a `TArrayF`/`TArrayI` member of a declared class is described as ROOT describes it |
| Q-33 (part) | done | `c1fca432` (worker 10) | 0 | an unqualified hashed multi-container or forward list is the standard library's |
| W0-10 | **done in part** | `1053acde` (worker 14), `b590e726` (worker 16) | 0 | `.typedef` lists the typedefs of the headers a macro includes, and worker 16 wrote the measurement the next attempt needs: `tests/test_root_split_branch_name.py` pins what a split branch's name and leaf list are with and without the command |
| Q-39 (writer) | **attempted, not merged** | `5b65bffa` | 0 | worker 15's "one of ROOT's arrays of numbers is written as its count and values" collided with worker 5's `TArrayX` work in `wdeclared.py`: merging the two broke three of worker 5's tests, so the pair was resolved in worker 5's favour and 15's is unmerged |
| W0-33 | attempted, not merged | - | 0 | seven workers reported without a commit (4, 6, 7, 8, 11, 12, 13); the refusal each hit has *moved one file further out* - `w04-outside-scope.patch`, `w06`'s `wdobjects.py`, `w07-nested-vector.patch`, `w11-out-of-scope.patch`, and the `storage`/`TBrowser` walls in 12 and 13. Their reports are `/tmp/r11/wNN.log` |

| **Round 12 re-scoped the fan-out**: instead of sixteen disjoint *files*, sixteen disjoint *chains* - every file a refusal walks through - because four rounds of workers had each spent their budget discovering the wall one file further out. The result: **14 roottest tests gained and none lost** (`/tmp/r12/final2` vs `/tmp/r11/final2`), against 2 gained in round 11 |
| Q-38/Q-40 | **done in part** | `6b920c39`, `5bfaccac`, `f599f6ea`, `8078577b`, `9672d02c` (workers 1, 3, 4, 5, 7) | roottest +6 | a split collection of a macro's class is carried rather than named among ROOT's; a class deriving from a container is laid out as ROOT lays that base out; a member-wise collection reads its items as the program's class is; a branch of a `std::vector` of collections is written whole, both counts in its record; a selector's reader is typed by the class it was declared for. `treeproxy/vectorint/createfile` and `selectorreader/{CollectionClasses,Options}` pass |
| Q-45 (part) | done | `e2410c4d` (worker 9) | roottest +5 | an unqualified `bitset` is the standard library's, as a macro writes it: the five `treeformula/schemaEvolution/*` and `treeformula/array/nonSplit` pass with `collection/tbits` and `tree/addresses/twrite` |
| Q-44 (graphics) | **done in part** | `b8f09177` (worker 6) | 0 | a pad saved under a `.C` name is written as the macro ROOT writes - `test/stressgraphics` moves to its next wall |
| Q-16 (TEnum) | **done in part** | `2c3cfbe9` (worker 10) | 0 | the enum table knows the type an enumeration is stored as and lists its global enumerators |
| W0-10 | **done in part** | `67d6b9bf` (worker 11) | 0 | the interpreter's typedef walk lists what `.typedef` lists, and the characterisation test now matches ROOT's line shape (`f7e42087`) |
| Q-36 | **done in part** | `2c3cfbe9`, `f599f6ea` (workers 4, 10) | 0 | a member-wise collection reads its items as the program's class is and says when it did not - `meta/evolution`'s wiring is one layer further on |
| Q-41 (part) | done | `9b7ca7e9` (worker 14) | 0 | TMVA reads an option's number where C++ stops |
| W0-34 | done | `e4708933` (worker 15) | 0 | `gSystem`'s answers as ROOT gives them, checked against the oracle |
| G-2 (part) | done | `3aff3e15` (worker 16) | 0 | `HasFailure()` inside a lambda reads GoogleTest's own state - the gtest kind's largest single translator cause |
| W0-35 | attempted, not merged | - | 0 | worker 2 left five files uncommitted in its worktree (the `TGeoManager` record) and workers 8, 12 and 13 died or reported without a commit; their chains are round 13's by default |

| Q-44 | **done** | `c747849c`, `f2ddea6c` (worker 1) | roottest +19 | the geometry chain, end to end: a `TGeoManager` record is written into the ROOT file the export makes, and all nineteen `io/geo/export_*` tests pass. Five rounds had approached this one file at a time |
| Q-46/Q-41 | **done** | `8f7775b0` (worker 7) | roottest +4 | a `std::bitset` member written, generated and read back: all four `ntuple/makeproject/ttree/*` pass, so the makeproject group is complete |
| Q-39 (list) | **done** | `c262c846` (worker 5) | roottest +1 | `ROOT.list` and the other bare standard templates, as cppyy has them: `tree/evolution/WriteList` passes |
| W0-34 | **done** | `1fc6bff2` (worker 12) | 0 | `gErrorIgnoreLevel` is read where a script set it, on ROOT itself |
| Q-40 (part) | done | `be6f0cd7` (worker 6) | 0 | a `TClonesArray` of a macro's class is split off its declaration |
| Q-38 (compression) | **done in part** | `c0cd9118` (worker 2) | 0 | a compression level past an algorithm's range packs at its top, as ROOT's does - `io/compression` no longer dies in `zlib`, it now refuses by name where the algorithm is missing |
| Q-44 (graphics) | **done in part** | `fb57bfb4` (worker 9) | 0 | a pad's macro writes polylines, arrows, paves, round shapes and its sub-pads |
| Q-16 (TEnum) | **done in part** | `344809f5` (worker 10) | 0 | `TEnum::GetEnum` loads the library a rootmap lists an enumeration in |
| G-2 (part) | done | `85065e8d` (worker 13) | 0 | `ROOT::TestSupport::FilterDiagsRAII`, a test's own handler for a diagnostic |
| Q-32 (TMVA) | **done in part** | `8342d8d6` (worker 14) | other +6 | PyMVA's interpreter, and the ensembles' multiclass fit: six `other/tmva/pymva/*` tests pass |
| Q-38 (math) | **done in part** | `a5d25f67` (worker 16) | other +1 | `ROOT::Math`'s `Polynomial`, `IGenFunction` and `GSLMinimizer1D`: `mathmore-testMinimization1D` passes |
| W0-36 | attempted, not merged | - | 0 | workers 3, 4, 11 and 15 each measured that their chain's wall is one file outside it (`pyroot/trees/core.py`'s `MakeProxy`, `interp.py`'s walk, `files.py`); their reports are `/tmp/r13/wNN.log` and are round 14's chains |

| W0-37 | **done** | `428f2bfd`, `ad59b281`, `5fc12bf5`, `d1c3e1a0`, `327062f5`, `294a40e0`, `3070b310`, `9e2fd3a2`, `c5cf418b`, `c545445e` (ten workers) | roottest +11, other +1 | the round's ten chains: ROOT's drawing classes written where an object is a member (bigevent's next wall), `TBranch::GetClonesName`, ROOT's own `TRef` description, the WILLFAIL verdict, a clone's split-collection count, a pad's pie chart, `TDirectory::rmdir`, a message with no place to name, `ROOT::Math`'s sine and cosine integrals, and `TTree::MakeProxy`. The last is the round: it unblocked five `treeproxy` tests, two `vectorint` selectors and `treeformula/string/MakeProxy` |
| W0-38 | todo | | 0 | **A second agent session is committing to `feat/root-testsuite`.** Mid-round it reset the branch and re-applied its own commits on top; the round's ten cherry-picks survived only on their `r14-NN` branches and were re-applied by hand (all ten were content-identical once rebased). Every later round must check `git log` at its start *and* its end, and quote the sha it actually measured |
| W0-39 | todo | | 0 | the four `io/perf` buffers now take 211-261 s of the 300 s limit (`/tmp/r14/solo`), so they will flip to TIMEOUT on any loaded machine. `P-9` closed them functionally; they need the *next* factor of two, or a per-test timeout the harness can justify |
| W0-40 | attempted, not merged | | 0 | six workers (3, 4, 6, 12, 15, 16) reported without a commit; their reports name the next file each chain needs - `pyroot/trees/core.py`'s proxy (landed by worker 14), `interp.py`'s walk, `tools/rootsuite`'s working copy, `jsonbuffer.py`, `files.py`. Those are round 15's chains |

| W0-41 | **done in part** | `ccfd6aa3`, `4f8856f3`, `601996d1` (worker 3) | roottest +3 | a file's member pairs with the class's own by position as ROOT's `CompareContent` pairs it; a file's layout of a version the class also has stays the class's current one; and a file's pair base and a dictionary's template instance are no missing dictionary. Three `meta/evolution/checkWarn*` tests pass, and the six stderr DIFFs move on |
| Q-39 (arrays) | **done in part** | `c69185f5` (worker 1) | 0 | an array of containers of objects is one record of columns - the bigevent writer's next member |
| W0-42 | attempted, not merged | - | 0 | workers 2, 4 and 8 reported without a commit and each measured that its chain's wall is outside it (`pyroot/trees/core.py`, `interp.py`, `rdobjects.py`); their reports are `/tmp/r15/wNN.log`. Worker 8 took `io/perf` no further: the buffers pass again at `--jobs 3` and still have only a 40-90 s margin (`W0-39` stands) |
| W0-43 | todo | | 4 | the CERN endpoint was slow for the whole round: `main/RootcpReadRemote{,CheckOutput,Clean}` and `io/hadd/test_hadd_regr_20872` timed out at `--jobs 1` with nothing else running (`/tmp/r15/net`). They are not code failures and must never be counted as losses without this rerun |
| W0-44 | todo | | 0 | **the machine is shared**: load averages of 20, then 64, during this round, from another session. Rounds should check `uptime` before choosing `--jobs`, and a round whose load is above ~8 should measure at `--jobs 3` and say so |
| W0-45 | todo | | 0 | **the coverage gate flaps under load**: three `-n 4` runs at a load average above 20 each missed 100% by one branch in a different file (`pyroot/__init__.py` twice, then `interp.py`), and `-n 2` was clean twice. A round on a loaded machine should run the gate with `-n 2` and not add a test to chase the branch |

| W0-40 | **done** | `2dad75a7` | roottest +3 | **a harness defect, not xrdroot's**: ROOT's `ROOTTEST_ADD_TEST` hands the driver `OUTPUT <build>/<testname>.log`, so a test that follows one reads its sibling's output (`schemaRun2check`'s `sed schemaRun2.log`). The harness stored nothing, so five `schemaEvolution` checks died on a missing file; and an `OUTREF` naming a build-tree file was looked for in ROOT's *source* tree, which raised `FileNotFoundError`. Every roottest now writes its output log, and the judge reads a build-tree reference from the working copy. The group goes 9 PASS / 5 FAIL → **12 PASS / 2 DIFF / 4 UNSUPPORTED**, and the two DIFFs are now honest comparisons |
| W0-46 | todo | | 5 | the same `STOREOUT` rule will move **every** group whose tests read a sibling's `.log`; the next round should re-run the six areas with multi-test logs (`treeformula/*`, `io/*`, `tree/*`) and record which of them were failing on this alone |
| W0-47 | attempted, not merged | - | 0 | round 16 ran **one** chain instead of sixteen: the machine's load average stayed between 9 and 36 from another session, and a fan-out would have produced measurements that could not be trusted. The eight chains of round 15's list (`pyroot/trees/core.py`, `interp.py`/`rdobjects.py`, `wdobjects.py`, `reflection.py`, `stressgraphics`, `jsonbuffer.py`, `files.py`) are untouched and are round 17's |

| W0-41 (part) | **done** | `f31a77e8` (worker 3) | roottest +6 | a dictionary that arrives after its file is ROOT's `PostLoadCheck`: `meta/evolution/checkWarn{0,1,10,11,12,13}` pass - the six stderr DIFFs round 15 named now compare equal, and the `meta/evolution` area lost nothing |
| Q-38/Q-40 (writer) | **done in part** | `bb51ccae` (worker 1), `2c65bae4` (worker 2) | 0 | a split `TClonesArray` writes a whole `TRef` member as the `TRef`'s record; an array of `TString`s is the one string element with the array offset |
| Q-44 (graphics) | **done in part** | `928dd29c` (worker 6) | 0 | a `TPaveText` and a `TMathText` written as the macro that draws them again - `test/stressgraphics` moves on |
| W0-25 | **done in part** | `13034109` (worker 7) | 0 | `TBufferJSON` writes a `TString` array at the depth it is held |
| Q-41 (TRealData) | **done in part** | `f632eb6e` (worker 8) | 0 | `TRealData`, `BuildRealData` and `GetListOfRealData`, as ROOT flattens a class's members - `double32`'s `'NoneType'` wall now has the name it wanted |
| W0-46 | **done** | - | 0 | the `STOREOUT` survey round 16 asked for: `grep -rn '\.log' roottest/**/CMakeLists.txt` finds exactly **two** directories reading a sibling's log - `schemaEvolution` (fixed in round 16, 3 tests) and `io/vararyobj`, whose tests are refused for their own reasons. The rule had no wider prize, and that is now measured rather than assumed |
| W0-48 | todo | | -64 measured, 0 real | **the suite's roottest count is not stable under load**: at `--jobs 3` with a load average of 44 the `io/event` fixture's `TestIoEvent` collided with itself and **47 fixtures** failed behind it, plus 13 `ObjectSlot` failures - 64 "losses" that are all load. The same area passes **29/29 at `--jobs 3` run alone** (`/tmp/r17/check-ev`), and `io/event`+`fastcloningeventtree` is now a group to measure on its own. Round 17's full-run JSON is kept at `/tmp/r17/final2` but is **not** quoted anywhere |
| W0-49 | todo | | 0 | a **second session's uncommitted work** keeps appearing in the main worktree (`/tmp/r15/stray-main-worktree.patch`, `/tmp/r17/stray-main-worktree.patch` - `Slot.takes` in `copying.py`/`objects.py`/`store.py`, untested and so a red coverage gate). A round must `git status` before every gate run and set aside what is not its own |

| Q-38/Q-40 (writer) | **done in part** | `28dce7e4` (worker 4), `eb7199b3` (worker 7) | 0 | an array of pointers is written one pointer per element; a branch of a bare `TObject` splits into the members of its class. Both move `io/bigevent`'s and `io/tree`'s refusals one member on, and neither flips a test yet |
| Q-44 (graphics) | **done in part** | `7fd2d377` (worker 6) | 0 | a `TGraph`, a histogram and a `TGaxis` written as the macro that draws them again - `test/stressgraphics` is now three layers into the pad's macro and still refused for the next one |
| W0-50 | todo | | 0 | **round 18 gained nothing, and that is the finding.** Ten areas, 167 tests, three commits, 0 moved: every remaining group in them is one *feature* from passing, not one line - `io/bigevent`'s next member, `pairEnumEvo`'s emulated layout (`Q-30`), the data model's reader (`interp.py`), `double32`'s `TRealData` consumer, the graphics pad's macro, `io/tree`'s `SplitClones` and `jsonbuffer.py`'s last writers. Four of the round's workers stopped at the same sentence: "the wall is outside my chain". **A chain scoped by file path cannot contain a wall that is a missing behaviour**; the next rounds should take ONE group each with the whole component as scope, and expect a round per group |

| W0-51 | **done in part** | `f62ee33e`, `91a598db`, `bdc4a2cc`, `0a01f00c`, `82f0b1e5`, `b9019f08` (six workers) | 0 | six commits, each removing one refusal: an array of containers, an axis's second bit word and a `TDatime` written; a streamer element a file spelled another way names the type in memory; a member of ROOT's read by the getter without a `Get` (`TObjArray::LowerBound`); real data of a template instantiation listing that object's members; a `char[N][M]`'s rows cut where its declaration cuts them; `gInterpreter.LoadText`. **The measured areas gained 0 and lost 0** (`/tmp/r19/areas`, `/tmp/r19/tree`) - `io/tree`'s refusals became 13 FAILs with the same 5 PASS |
| W0-52 | todo | | ~40 | **what each remaining group actually needs, from this round's eight reports.** `io/bigevent` (16): a layout model for a template member and an array of containers of objects - `wdobjects.py`'s element kinds. `pairEnumEvo` (4 + 40): the emulated layout AND ROOT's `fStreamerInfos` index bookkeeping, whose checksum and offset rules can only be calibrated against a running ROOT. `io/tree` (11): `pyroot/core/reflection.py:GetClassVersion` falls back to 1 for a file-only layout - one line, then the writer's `SplitClones`. `double32` (7): four things, none in `pyroot/core` (`W0-53`). The data model (7): a class record read by its own description. Graphics (3): the pad's PostScript **character counts** against ROOT's `.ref` at ±50 - `canvas/`. `jsonbuffer` (6): `MEMBERS` entries for `TObject`. `roottest/python` (30): 24 cases from one ACLiC `NameError` in `pycint/library/loading.py` |
| W0-53 | todo | | 7 | `double32/template32` needs four things outside `pyroot/core`, named by worker 5's report (`/tmp/r19/w05.log`): a `Double32_t` template argument's range comment read from the macro, the writer's range narrowing, `TClass::GetStreamerInfo` for a template instance, and the read side's `SetFloat`-style packing |

| W0-54 | **done in part** | `2ae5d980`, `95014dca`, `8b83e603` (three workers) | 0 | the three "one named change" walls round 19 identified: a class's own statics read in its body by the body's own names (the ACLiC `NameError`); `TClass::GetClassVersion` answering with the version a file gives a class it alone describes; `TBufferJSON` reading back a class its table describes and writing a `TBox`. **`roottest/python` measured 8 PASS / 17 FAIL / 9 UNSUPPORTED and `io/tree` 5 PASS / 12 FAIL** - both unchanged in PASS, so each fix revealed the group's *next* wall |
| W0-55 | todo | | ~50 | **three consecutive rounds at 0 gained**: rounds 18, 19 and 20 landed 12 commits between them and moved no test from FAIL to PASS. The remaining failures in these groups are features with several walls each, and every wall is a real ROOT behaviour - so a 50-minute worker cannot finish one, however precisely the brief names its first wall. **The honest plan for the loop's remaining iterations is one feature per round, chosen from `W0-52`/`W0-53`, with the round's success measured by the wall it removes and the commit it lands, not by a PASS count.** The PASS ledger stays authoritative for what has been achieved (roottest 1239, non-tutorial 1406); it is no longer a good per-round target |

| **Round 21 was one feature, four files, as `W0-55` said it should be**: `Event::fTemplateMember`, a `template2<template1<int>>` member of `io/bigevent`'s class, blocked for eight rounds | `67d08e8e` (winfo), `5b3c58f9` (wdeclared), `d84f3e7f` (wdobjects), `ac2541de` (rdobjects) | 0 PASS, but **the wall is gone**: `io/bigevent` now builds its dictionary (**2 PASS / 0 FAIL / 2 UNSUPPORTED**, `/tmp/r21/ev2`) and the refusal has moved from the template member to `a TArrayI is not something this writer puts in a ROOT file` |
| W0-56 | todo | | 0 | **a harness run started immediately after a cherry-pick can fail with a stale-cache `ImportError`** (`cannot import name 'compare' from 'xrdroot.filling'`, `/tmp/r21/ev`), and the same group passes 2/2 on a rerun (`/tmp/r21/ev2`). The dictionary and library builds cache their translations; when the source changes under them the cache is not always revalidated. Rerun a group once before believing an `ImportError` in a build step |
| W0-57 | todo | | 16 | `io/bigevent`'s next wall, named: `a TArrayI is not something this writer puts in a ROOT file`. It is `TArrayI`/`TArrayF`/`TArrayD` as a *member* (`Event.h` has several), which is the same family that `W0-53`'s `double32` work needs - one feature for the next round |

| **Round 22: the next feature, same shape** - `TArrayF`/`TArrayI` as a *member* (`W0-57`), which served two groups | `d0e3a132` (winfo), `de5f7ef3` (wdobjects), `79c574f3` (rdobjects) | 0 PASS, and **the wall is gone**: `io/bigevent` still measures **2 PASS / 0 FAIL / 2 UNSUPPORTED** (`/tmp/r22/ev`) and its refusal moved from `a TArrayI is not something this writer puts in a ROOT file` to `a TVector3 here is a float rather than the dict of ...`; `io/double32` measures **3 PASS / 1 FAIL** (`/tmp/r22/d32`) with `template32` still at its `'NoneType'` wall |
| W0-58 | todo | | 16 | `io/bigevent`'s next wall, named: `a TVector3 here is a float rather than the dict of …` - a `TVector3` member arrives where the writer expected its members, which is `wdobjects.py`'s member dispatch (a class-typed member of a ROOT class whose layout this writer carries). The next feature, and the third in the same group |
| W0-59 | todo | | 0 | **three of four workers in round 22 left untracked `docs/README.md` / `docs/gotchas/` / `docs/discoveries/` in their worktrees** - a worker writing documentation outside its scope, or another process doing it. They were not cherry-picked; a brief should say plainly that `docs/` is not theirs, and a round should look for stray `docs/` before copying a worktree's files |

| **Round 23: the third wall of `io/bigevent`, and the first PASS since round 17** - a ROOT class's member that is itself one of ROOT's classes (`TLorentzVector::fP` is a `TVector3`) | `4913f410` (declared.py), `2c123241` (wdobjects), `9da8ee08` (rdobjects) | **roottest +1**: `io/bigevent/write` **PASSES** (`3 PASS / 0 FAIL / 1 UNSUPPORTED`, `/tmp/r23/ev`). The `read` side's next wall is named: `'event' holds Event, which holds 'fVectorTobject', which is of a kind that this reader does not decode inside an entry` |
| W0-60 | todo | | 1 | `io/bigevent/read`'s wall, named: a `vector<TObject> *` member (`fVectorTobject`) is a kind `rdobjects.py` does not decode inside an entry. That is the reader's own feature, and the group's last one - the *writer* already lays it out |
| W0-61 | todo | | 0 | **three rounds of stray files in the main worktree from other sessions** (`dot.C`, `h.h`, `mkroot.py`, `test.root` this round; `interp.py`/`buffers.py`/`rdobjects.py` in rounds 19-20). `git status` before every gate run caught them all, but the loop is sharing a worktree with another writer: the round should `git clean -n` its own files before the gate |

| **Round 24: `io/bigevent`'s reader takes one more member** - a container a member *points at* (`vector<TObject> *fVectorTobject`, stype 71) is read as the container it names | `0d4ed89f` (interp.py), `1ac5101d` (rdobjects.py) | 0 PASS: the group stays **3 PASS / 0 FAIL / 1 UNSUPPORTED** (`/tmp/r24/ev3`) and `read`'s refusal moved from `fVectorTobject` to `fVectorTnamed` |
| W0-62 | todo | | 1 | **`fVectorTnamed` is not a stype to teach.** It is `vector<TNamed> *fVectorTnamed[6]` - a pointer to a container *and* an array (`Event.h:100`, stype 91 = `STL_POINTER + 20`) - which ROOT writes into one record for the whole array. This round tried treating 91 as 71 and the harness's refusal moved on, but by **misreading the bytes**: the existing test `test_an_array_of_pointers_to_containers_leaves_the_type_as_the_file_describes_it` failed with `DID NOT RAISE`, which is why the change was reverted. A future round must write the array-of-pointers reading, not relabel the kind |
| W0-63 | todo | | 1 | the harness's own lesson from `W0-62`: a one-line change that moves a refusal *further* is not progress until the full suite agrees. The unit tests caught this in 0.3 s; the harness alone would have called it a gain |

| W0-106 | todo | | 0 | **the whole-suite gate is not runnable on this machine, in three separate ways** (measured this round): (a) `tests/test_tmva_learn_networks.py` (torch, 15 cases) fails 5-9 cases at `-n 4` and `-n 2` and **passes alone in 8 s**, and its worker's death loses the coverage data - `-n 2` reported **24%**, `-n 4` 9 failures; (b) `-n 0` (one process, everything) dies with `Fatal Python error` at 63-91%; (c) `tests/test_pyroot_references.py`'s `test_a_reference_numbers_its_object_in_the_session_process_and_finds_it` fails only when it shares a worker (26 pass alone), the global-`TProcessID` flake the loop has seen before. **The decomposition that works**: `pytest -n 4 --cov --ignore=tests/test_tmva_learn_networks.py --ignore=tests/test_pyroot_references.py` (12 925 pass), then `pytest tests/test_tmva_learn_networks.py tests/test_pyroot_references.py --cov --cov-append` (41 pass) |
| W0-107 | todo | | 0 | **10 pre-existing uncovered lines in another session's `src/xrdroot/tmva/nettrain.py`** (`188`, `199-207`, and the branch at `50`): no test in the tree reaches the trainer's error path, so even the two-command decomposition above ends at **99.99%**. `git log -1 -- tmva/nettrain.py` names the commit; the lines are the "torch is not installed / training failed" path, which needs a test that refuses without torch or a marked skip |

| **Round 41: a qualified static call's out-parameter, and the std algorithms a macro asks for** - `9acf40db` (a qualified static call fills the local handed to it by reference), `e008ab0b` (`std::generate`/`shuffle`/`next_permutation`, the ostream formatting state, and `ROOT::Math::Random<T>`'s engine kinds - worker 02's uncommitted work, taken from its tree) | `9acf40db`, `e008ab0b` | **no harness PASS measured**: the load average reached **144**, so `io/json` and `other/math` were not measured; `roottest/root/math/*` read 0 PASS / 4 FAIL / 1 TIMEOUT under it, which is unproven (`W0-103`) |
| W0-99 | partly | | 2 | **half of `ArrayCompress` landed**: `TBufferJSON::FromJSON(T*& obj, json)` is a *qualified static* call, and `pycxx/program.py`'s `reference_positions` answered out-parameters only for a member call or a single-part name - so `arr2` was never a cell and nothing could fill it. `9acf40db` asks ROOT's table by the name's last part as well, and `pycint/dialect.py`'s `("FromJSON", 2): (0,)` makes `arr2` a cell. The probe now reaches `TBufferJSON::FromJSON cannot read this JSON as any class: a bare JSON array does not say what class it holds` - **the next gap is `FromJSON` reading a `TArrayI` in `jsonbuffer.py`**. `RootClasses` needs three gaps, one of them jsonbuffer's (round 41's worker note, `/tmp/r41/w01/NOTE-r41-01-json.md`) |
| W0-110 | partly | | 0 | **w02's work landed, w03's did not**: `std::generate`/`shuffle`/`next_permutation`, `std::cout.width/precision/fill` and `ROOT::Math::Random<T>`'s engine kinds are in (`e008ab0b`, its own next-permutation test split to stay under CCN 10). w03's `pyroot/linalg/smatrix.py` change is **not taken**: its own tests pass, but `roottest/root/math/*` went from **0/4/1 UNSUPPORTED** to **0/4/1 TIMEOUT** under a load average of 144 - unproven, and the patch is kept at `/tmp/r41/w03-smatrix.patch` |
| W0-111 | todo | | 0 | **two workers' sessions ended without a commit** (w02 with 7 files dirty in `cppstd/`, w03 with `smatrix.py`); the round recovered w02's work by hand from its tree and left w03's as a patch. A round should check each worktree's `git status` at the halfway point, not only at the end (`W0-94` again, now with the recovery that works) |

| **Round 40: `io/tree` is clean of FAIL and DIFF, and the dictionary warning is right** - `82001fb0` (a class ROOT has a dictionary for is no warning, by name as well as by layout), `de1e9cac` (the declared-class accessors' fall-backs) | `82001fb0`, `de1e9cac` | **roottest +1: `io/tree/read` PASSES** - `roottest/root/io/tree/*` is **18 PASS / 0 FAIL / 0 DIFF / 2 UNSUPPORTED** (`/tmp/r40/tree`, from 17 PASS / 0 FAIL / 1 DIFF in `/tmp/r36/tree`) |
| **Round 42: a cell's class, and ROOT's iterator family** - `4e5436ff` (a cell holding an object pointer carries the class its declaration names, so `FromJSON(T*&, json)` can make one), `a2ef7d77` (TIterator/TListIter/TObjArrayIter/THashTableIter/TMapIter, and a TIter that owns and deletes one; a default answered through `self` is put off to the body) | `4e5436ff`, `a2ef7d77` | **roottest +1**: `io/json/ArrayCompress` **PASSES** and `roottest/root/io/json/*` is **11 PASS / 0 FAIL / 0 DIFF / 1 UNSUPPORTED** (`/tmp/r42/json`, from 10/0/0/2) - only `RootClasses` left. **gtest**: `coreconttestUnit`'s two `TIter` cases pass. The load average stayed at 14-31, so `other/math` and the gtest kind were not re-measured |
| W0-99 | **done** | `4e5436ff` | 1 | **closed**: `TBufferJSON::FromJSON(T *&obj, json)` makes an object of the class the *declared* pointer names, and a cell carried only a number's width, so a null pointer cell could not say which class - which is why a bare JSON array (ROOT's uncompressed form) was refused. `pycxx/emit_vars.py::cell_type` now carries an object pointer's class as well; `void *` stays bare, naming no class. `ArrayCompress` reads all six of its arrays back, in ROOT's four compressed forms too |
| W0-112 | todo | | 0 | **`next_permutation`'s successor search had no test** (`cppstd/objects.py:321`, `successor -= 1`): round 41's commit shipped it, and the gate could not be measured then (`W0-103`), so the gap went unnoticed until this round's second coverage command named it. A permutation whose suffix starts above the pivot (`[2, 3, 1]`) reaches it; the test is in `tests/cppstd/test_cppstd_pure_objects.py`. **A round that cannot run the gate should still run the two-command decomposition over the files it touched** |
| W0-113 | todo | | 2 | `gtest/core/cont/testTypedIteration` (`TCollection.RangeCast`, `TCollection.TypedIter`) needs `ROOT::Detail::TRangeStaticCast`/`TRangeDynCast`, and behind them a modelling decision: a ranged-for over a `TList` yields `TObject*` *statically*, so `bcl->Class()` is `TObject::Class()` where ours answers the dynamic class. ROOT's `Class()` is a static member of each class, so this needs an element view whose static type is the range's, not a fix to `Class()` |
| W0-114 | todo | | 1 | `roottest/root/io/json/RootClasses` is the area's last test and needs a `TH1I`'s JSON: 35 members (`TH1`'s `fNcells`/`fXaxis`/`fTsumw*`/`fFunctions`/`fArray`…), three `TAxis` of 26 each, and the read back - none of which the namespace's `TH1I` exposes under those names, since it is an engine object. The committed `RootClasses.ref` makes it verifiable without the harness: `python -m xrdroot run runRootClasses.C` and diff. A round-sized feature, not a one-file fix |
| **Round 43: ROOT::Math's GSL solvers, and the class constants ROOT's tests name** - `0be3733b` (`Roots::Brent`/`Bisection`/`FalsePos`/`Newton`/`Secant`/`Steffenson` over `GSLRootFinder`/`GSLRootFinderDeriv`, siblings under `IRootFinderMethod`), `4e473e1e` (`TTree`/`TAxis`/`TH1`/`TInterpreter` status words and `RootFinder::kMODAB`) | `0be3733b`, `4e473e1e` | **`other` +1: `other/math/mathmore-testGSLRootFinder` PASSES** (`/tmp/r43/gsl`, 1.7 s), `other/math` **22 → 23 PASS**. **The gtest kind measured whole for the first time since round 13**: **53 PASS / 219 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** (`/tmp/r43/gtest`), bucketed at `/tmp/r43/gtest-buckets.md` |
| W0-116 | **done** | `0be3733b` | 1 | **closed**: `ROOT::Math::Roots` names GSL's algorithms one class at a time and each takes a `GSLFuncPointer` with the `void *p` GSL hands it; `GSLRootFinder`/`GSLRootFinderDeriv` are siblings under `IRootFinderMethod` as ROOT has them, because the two are handed different functions. `Name()` answers GSL's spelling and `kMODAB`'s is `ModABRootFinder`, both read off the oracle. `testGSLRootFinder` passes all six of its solvers; ROOT's own answer for that file is the *exit code*, since it defines its own `TStopwatch` and cannot be interpreted at all |
| **Round 44: ROOT::TSeq's random-access iterator** - `250324e3` (`begin()`/`end()`, `+`/`-`/`++`/`--`/`[]`/`*it` over `TSeqI`, and the missing `TSeqU`/`TSeqUL`) | `250324e3` | **gtest +1: `gtest/core/cont/TSeqTests` PASSES** (`/tmp/r44/seq`), the kind 53 → 54; `coreconttestUnit` moves from `TSeqIterators.Containers` to `TCollection.RangeCast` (`W0-113`) |
| W0-118 | **corrected** | | 219 | **round 43's table counted occurrences, not programs** - it read as if `RVec::Pt` blocked 17 programs, and it blocked one, 17 times (`W0-75`'s mistake again: a hand-rolled count). Recounted per program from the same evidence (`/tmp/r44/gtest-buckets.md`): **114 of the 219 FAIL programs ask for no missing name at all** - their first failure is an assertion of their own subject, so each is its own feature - and the 105 that do ask for a name are 1-3 programs each (`TTree.kOnlyFlushAtCluster` 3, `TFormula.CladStorage` 2, `TSeqI.begin` 2, `Cell.GetTitle` 2, `ROOT.Detail` 2, everything else one). **There is no wide name cluster in the gtest kind**: it is a long tail of individual features, and the round-43 ranking must not be used as if there were |
| **Round 45: the `test/` and `other` kinds measured whole, and root's unknown-option message** - `4a98e20d` (`fflush` reads the name a translation writes for a stream, as `fprintf` does; `TBits::Set`/`Get`), `4468d37b` (`root` names an option it cannot read and points at `--help`; the harness matches a `PASSREGEX` across newlines as CMake does) | `4a98e20d`, `4468d37b` | **`other` +1: `other/roottest/root/core/root_exe_errorMessage` PASSES** (`/tmp/r45/errm2`). **Both kinds measured whole for the first time in ~28 rounds**: `other` **40 PASS / 0 FAIL / 67 UNSUPPORTED / 2 TIMEOUT** of 109 (`/tmp/r45/other` for the 39, `errm2` for the one); `test/` **8 PASS / 7 FAIL / 44 UNSUPPORTED** of 59 (`/tmp/r45/testkind`) |
| W0-123 | todo | | 1 | **a `Char_t[]` initialised from a braced list of bytes becomes a `str`, truncated at the first NUL** - `test/testbits.cxx`'s `static const Char_t char_bits[] = {0,1,3,7,15,31,63,127}` is emitted as `char_bits = ''`, so `char_bits[i>>3] & 1 << (i&7)` has nothing to work on and the test's first assertion fails. C++ makes it an array of chars (indexable, arithmetic-able); the text shortcut is right only for a string literal or an array used as text. `emit_vars._char_array`/`_characters` truncate at NUL (`text.split("\0", 1)[0]`), which loses the bytes. The model to add: a braced list of numbers in a `char[]`/`Char_t[]` is a *buffer* of bytes (`cppstd.objects.array`), and the dialect should say which spellings ROOT means as bytes (`Char_t`, `UChar_t`) rather than as characters |
| **Round 46: a char array of bytes, and a TString read out of a stream** - `af1e82fd` (a `char[]` from a braced list is a buffer unless it spells a C string; the dialect names the free functions whose char array is memory; `TBits::Get` fills bytes into any element width), `7e369769` (a word read into a dialect string class is that class) | `af1e82fd`, `7e369769` | **`test/` +1: `test/testbits` PASSES** (`/tmp/r46/bits`, 41 tests), the kind 8 → 9 (`/tmp/r45/testkind` was 8). `test/stressgraphics-interpreted` moves FAIL → UNSUPPORTED (`ROOT has localtime`, `W0-126`) |
| W0-123 | **done** | `af1e82fd` | 1 | **closed**: a `char[]` initialised from a braced list is a **buffer of bytes** unless the list spells a C string (printable characters or C's white-space escapes, then NUL padding) - so `{0,1,3,7,15,31,63,127}` is `array('char', 8, [...])` and not `''`. The dialect gained `buffer_functions` (`memset`, `memcpy`) for the free functions whose first char array is memory, which is how `memset(buf, 0, sizeof(buf))` makes `buf` a buffer from its declaration |
| W0-125 | **done** | `af1e82fd` | 1 | **closed**: `test/testbits` needed only `W0-123`, as it said. `TBits::Get` also had to write `GetNbytes()` *bytes* into the array whatever its element width (an array of `Short_t` takes them two at a time), which its own test found after the array was a buffer |
| **Round 47: a failed read stores nothing, and C's broken-down calendar time** - `35c96338` (`extracted(stream, current, ctype)`: an extraction that fails leaves the variable as C++ does, and a bad stream is not read again), `f0655cf0` (`tm`, `localtime`, `gmtime`, `formatted_time`; `strftime` written as `sprintf` is) | `35c96338`, `f0655cf0` | **the gtest kind re-measured whole: 54 PASS / 218 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** (`/tmp/r47/gtest`, from round 43's 53), `TSeqTests` gained and **nothing lost**. `test/stressgraphics-interpreted` **UNSUPPORTED → FAIL**: it now runs the whole graphics stress suite and fails on image bytes (`Test 1. TLine` 1289 vs 529), which is `canvas/` fidelity |
| W0-127 | **done** | `35c96338` | 0 | **closed**: `while (in >> word) {}` left `word` cleared, because the empty text a failed read gives was stored; C++ sets failbit and assigns nothing, and does not read through a stream that has already failed. `cppstd/istreams.py::extracted(stream, current, ctype)` is the read now, in both places one is written, and it is also where the dialect's string class goes around the word - the class and the failure rule are one function |
| W0-126 | **done** | `f0655cf0` | 0 | **closed**: `tm`, `localtime`, `gmtime`, `formatted_time` and `strftime` exist (`cppstd/chrono.py`), with C's field rules (`tm_year` from 1900, `tm_mon` from 0) and a `time_t` that may be a cell. `strftime`'s `char[]` is written the way `sprintf`'s is - the text is returned and the variable keeps it - through a new statement handler `_strftime`, not `TEXT_TARGETS` (which is for cells) |
| W0-128 | todo | | 1 | `test/stressgraphics-interpreted` is now a **graphics-fidelity** failure rather than a refusal: it runs ROOT's own 32-test stress suite and compares each image's byte count with ROOT's (`Test 1. TLine` 1289 vs 529, `PDF` 1585 vs 12827, `JPG` 41211 vs 23848, `PNG` 4665 vs 3937). That is the `canvas/` work `W0-52` records - PostScript/PDF/PNG encoders and their exact character counts - and it is days-scale, not a name. The whole `test/` kind's remaining seven FAILs are of this shape or a stress program's own subject |
| **Round 48: TH1's normalisation and its statistics-overflow flag** - `4e2ba73b` (`TH1::Normalize` with its three options, `TH1::EStatOverflows` as the enum a macro names a value through, `Get/SetStatOverflows` and the static `StatOverflows`) | `4e2ba73b` | **no harness PASS**, but ROOT's own `hist/hist/test/test_TH1.cxx` goes from **13 failing cases to 11** by its two: `TH1.Normalize` and `TH1.StatOverflows` both **OK** now (measured directly, no harness: `python -m xrdroot run hist/hist/test/test_TH1.cxx`). The program is not one test - eleven cases still name `SmoothArray`, `THLimitsFinder`, `SetBuffer` and three bin-computation differences |
| W0-122 | **done** | `4e2ba73b` | 1 case | **closed**: `TH1::EStatOverflows` is both `TH1.kConsider` and `TH1::EStatOverflows::kConsider`, `GetStatOverflows` reads the file's `fStatOverflows` (`kNeutral` by default), `SetStatOverflows` writes it, and the static `TH1::StatOverflows(on)` is the global flag that changes no histogram's own - which is exactly what ROOT's `TH1.StatOverflows` case asserts |
| W0-130 | todo | | 9 cases | `gtest/hist/hist/testTH1` is **eleven failing cases**, each named: `TH1.SmoothArrayCrossCheck` (`type object 'TH1' has no attribute 'SmoothArray'`), `THLimitsFinder` (`ROOT has THLimitsFinder`), `TH1.SetBuffer` (`'TH1D' object has no attribute 'SetBuffer'`), `TH1.RegistrationToTDirectory_ImplicitOwnershipOff` and two root-finding/bin cases (lines 175, 218, 270, 276-277), plus `TAxis.BinComputation_FPAccuracy`, `TAxis.FindBinApprox`, `TAxis.EqualBinEdges` and `THLimitsFinder.TTreeDraw_AutoBinning`. Each is a feature of its own; **the program is not closable in one round**, and the three `TAxis` cases are bin-edge numerics that must match ROOT's exactly |
| **Round 49: `AddBinContent` takes one bin number per axis** - `3f3dae7f` (`TH2::AddBinContent(binx, biny[, w])` and `TH3`'s are the coordinates a macro names, not a global bin and a weight; `Eval` keeps adding to the global bin it worked out) | `3f3dae7f` | **no harness PASS**, but ROOT's own `hist/hist/test/test_TH1.cxx` goes **11 failing cases → 10**: `TH1.AddBinContent` is **OK** (it read 0 where ROOT reads 1). Measured by running the macro directly |
| W0-129 | todo | | 10 cases | what `testTH1` still needs, case by case (measured at the round's start, each named): **`TH1.SmoothArrayCrossCheck`** (`type object 'TH1' has no attribute 'SmoothArray'` - a static over a raw array, ROOT's 353 routine, which `Smooth` may already have); **`THLimitsFinder.Degenerate`** (`ROOT has THLimitsFinder` - `OptimizeLimits`); **`TH1.SetBufferedSumw2`** (`'TH1D' object has no attribute 'SetBuffer'` - and the case needs ROOT's *buffer*: 200 weighted fills into one bin, then `Sumw2()`, must still give `sqrt(200*4)`, so the buffer has to keep the weights for a later `Sumw2`; `SetBuffer` also raises a small size to at least 100); **`TH1.DumpOutput`** (`Dump` must print `*fArray                       ->0`); the two `RegistrationToTDirectory_ImplicitOwnership*` cases; and three `TAxis` bin-edge numerics (`BinComputation_FPAccuracy`, `FindBinApprox`, `EqualBinEdges` - the last wants `ROOT_EXPECT_ERROR` messages from `TAxis::Set`). **Each is a feature; the program is not closable in one round** |
| **Round 50: `TAxis` reports the binning it cannot use, and takes a vector of edges** - `2e10bb4d` | `2e10bb4d` | **no harness PASS**, but ROOT's own `hist/hist/test/test_TH1.cxx` goes **10 → 9 failing cases**: `TAxis.EqualBinEdges` is **OK**. ROOT's `TAxis::Set` *reports* (`Error in <TAxis::Set>: …`, it does not raise) and leaves the default axis, and `TAxis(edges)` - `TAxis _({1, 1})` - was read as no argument at all, so the check never saw the edges |
| W0-129 | todo | | 9 cases | what `testTH1` still needs, each named from a direct run: **`TH1.SmoothArrayCrossCheck`** (`type object 'TH1' has no attribute 'SmoothArray'` - the static 353 routine over a raw array, which `Smooth` may already have); **`THLimitsFinder.Degenerate`** (`ROOT has THLimitsFinder` - `OptimizeLimits`); **`TH1.SetBufferedSumw2`** (`SetBuffer` missing, *and* the buffer's semantics: 200 weighted fills into one bin then `Sumw2()` must still give `sqrt(200*4)`, so the buffer keeps weights for a later `Sumw2`, and `SetBuffer` raises a size under 100 to 100); **`TH1.DumpOutput`** (ROOT's `Dump` prints the *C++* members as `*name` padded to column 30 then `->value`, and the test wants `*fArray                       ->0`; ours prints the Python object's attributes - a fidelity gap, not a name); **`TH1.SetBuffer`'s own case**, the two `RegistrationToTDirectory_ImplicitOwnership*` cases, and `TAxis.BinComputation_FPAccuracy`/`FindBinApprox` (bin-edge numerics that must match ROOT exactly) |
| **Round 51: `TH1::SmoothArray`, the static 353 smoother** - `4227df45` | `4227df45` | **no harness PASS**, but ROOT's own `hist/hist/test/test_TH1.cxx` goes **9 → 8 failing cases**: `TH1.SmoothArrayCrossCheck` is **OK**. `stats.smooth_array` has been this kit's 353QH-twice since `TH1::Smooth` was written; the *static* name over a raw array a program hands as `v.data()` was never exposed. The write lands where the pointer points, so a `std::vector`'s own values smooth |
| W0-129 | todo | | 8 cases | what `testTH1` still needs, each measured: **`TH1.SetBufferedSumw2`** (`SetBuffer` missing *and* the buffer's semantics - 200 weighted fills into one bin then `Sumw2()` must give `sqrt(200*4)`, so a buffer keeps its weights for a later `Sumw2`, and `SetBuffer` raises a size under 100 to 100; `fill_one` is the hottest path, so every read that must flush is a surface to get right); **`THLimitsFinder`** (2 cases: `OptimizeLimits`, and `TTreeDraw_AutoBinning`'s `GetEntries == GetEffectiveEntries` after `BufferEmpty(1)`); **`TH1.DumpOutput`** (ROOT's `Dump` is a whole-object listing - `==> Dumping object at: 0x…`, then name padded to column 30, the value, a comment, recursing into `fXaxis.fNbins`; the line the test wants is `*fArray                       ->0                 [fN] Array of fN floats`, and `fArray` is not in this writer's member table); **`TAxis.BinComputation_FPAccuracy`/`FindBinApprox`** (bin-edge numerics that must match ROOT exactly); and the two `RegistrationToTDirectory_ImplicitOwnership*` cases (auto-registration off/on around a `TDirectory`) |
| **Round 52: a collection's element has the type C++'s range gives it** - `5f58c997` (`TypedElement`: the object in every respect but `Class`, which answers the range's static type; `iterate` hands a `TCollection`'s objects out as the `TObject *` C++ iterates; `ROOT::Detail::TRangeStaticCast<T>`/`TRangeDynCast<T>`, and `ROOT.Detail`) | `5f58c997` | **gtest +2: `gtest/core/cont/coreconttestUnit` and `gtest/core/cont/testTypedIteration` PASS** (`/tmp/r52/cont`: `gtest/core/cont/*` 2 -> 4 PASS), the kind **54 -> 56**. **No PASS lost**: `roottest/root/io/tree/*` **18 / 0 / 0 / 2** (`/tmp/r52/tree`) and `roottest/root/io/json/*` **11 / 0 / 0 / 1** (`/tmp/r52/json`) are each what they were, and 66 gtest programs of `io`/`hist`/`graf2d` are **26 PASS / 32 FAIL / 8 UNSUPPORTED** - the same set as round 47's `/tmp/r47/gtest` (`/tmp/r52/gtest-iohist`). Gate `TOTAL 119625 0 29998 0 100%` (12 990 + 44 passed, `/tmp/r52/gate1.log`) |
| W0-113 | **done** | `5f58c997` | 2 | **closed**: `for (auto bcl : *cl->GetListOfBases()) EXPECT_EQ(tobjectCl, bcl->Class())` wants `TObject::Class()`, because `Class()` is a *static* member and C++ resolves it through the range's type; `TRangeStaticCast<TBaseClass>`/`TRangeDynCast<TClass>` are how a macro names that type itself. `pyroot/core/rangecast.py` holds `TypedElement` (the object in every way but `Class`, with `__class__` so `isinstance` still sees the object), and `pycint/runtime/adapters.py::iterate` is where a `TCollection` is handed out that way - the one place a translated range-for asks what it visits |
| W0-130 | todo | | 0 | **a wrapper is one more reference a macro can keep**: a range-for's element is a `TypedElement`, so a macro that adds it to another collection (`for (auto o : *a) b->Add(o)`) stores the wrapper, and `type(x)` is the wrapper where `isinstance` is the object. Nothing does that today - nothing was lost in 66 gtest programs, `io/tree` or `io/json` - but the next wall of that shape is this, and the fix is for `TCollection` to unwrap what it is handed (`pyroot/core/collections.py`) |
| W0-131 | todo | | 0 | **round 44's gtest bucket table (`/tmp/r44/gtest-buckets.md`) is eight rounds stale**: it still names `TH1.StatOverflows`, `TH1.EStatOverflows`, `RootFinder.kMODAB`, `TInterpreter.kNoError` and `TSeqI.begin` as a program's first failure, and rounds 43, 44, 48 and 52 fixed all five. Its three-program rows (`TTree.kOnlyFlushAtCluster`, `NoneType.Get`, `_Namespace.MultiProcess`, `RNTupleModel.AddProjectedField`) were not re-checked this round: any round that works from that table must first re-measure the gtest kind whole (~1 h at `--jobs 3`), which only round 47 has done since |
| W0-132 | todo | | 1 | `gtest/hist/hist/test_THBinIterator` runs **20 cases in 64 s** and fails 7 of them (`TH1D`/`TH2D`/`TH3D`/`TH2Poly` `BinIterator`), measured directly this round. Neither the time nor the failure is a missing name: the test walks every bin of every histogram through `TH1`'s bin iterator, so both are that iterator's cost and its fidelity - the speed work `AGENTS.md` lists, and worth a whole measurement before anything is changed for it |
| **Round 53: `ROOT::RangeStaticCast`/`RangeDynCast` and `TDataType::GetType`** - `7f90083d` (`ROOT/RRangeCast.hxx`'s pair: a pointer type, any range, a C array of pointers, and a class the program declared found in the interpreter's table), `f93144fd` (`TDataType::GetType(typeid(T))` over `EDataType`'s codes, with them under the bare names a macro writes - `kInt_t`, `kBool_t`) | `7f90083d`, `f93144fd` | **gtest +2: `gtest/core/foundation/testRRangeCast` and `gtest/core/meta/testTDataType` PASS** (`/tmp/r53/corefm`, the two areas 1 -> 3 PASS with nothing lost), the kind **56 -> 58**. **The gtest kind measured whole at HEAD** (`/tmp/r53/gtest`, 573 s): **56 PASS / 216 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 - the same set as round 47's 54 / 218 but current, and **nothing lost in eighteen rounds of fixes** | 2: `W0-131` **closed** (the ranking is current, and it says there is no name cluster: 216 FAIL programs, 214 with their own first failure - `/tmp/r53/one-case.md` names the **47 that are one failing *case* from a PASS**); `W0-133` (`TInterpreter::Calc` echoes where ROOT is silent) |
| W0-131 | **done** | `6f1b71a9` | 0 | **closed**: the gtest kind re-measured whole at HEAD, `--xrdroot-src` a worktree of it so a commit landing mid-run cannot move the answer - **56 PASS / 216 FAIL / 76 UNSUPPORTED / 2 TIMEOUT**, 573 s at `--jobs 3`, `/tmp/r53/gtest`. It answers round 44's question: the 218 programs that are not PASS (216 FAIL and 2 TIMEOUT) reduce to **216 distinct first failures** - only the two timeouts share one - so **there is no wide name cluster left in the gtest kind**, and no round may plan as if there were. The table a round wants is the other counting: gtest prints a failing case twice, and counted that way **47 programs have exactly one failing case** - one feature each from a PASS - listed with its message in `/tmp/r53/one-case.md` (45 after this round's two) |
| **Round 54: a block-scoped RNTuple writer writes its file, and `Calc` is silent** - `98df6a2e` (a qualified static call `T::f(...)` asks the dialect's `method_returns`, so `auto w = RNTupleWriter::Recreate(...)` has a type; `RNTupleWriter` is a guard and gains the `_destruct` a guard needs), `2ce1f188` (`TInterpreter::Calc` takes the silent path `ProcessLineFast` takes) | `98df6a2e`, `2ce1f188` | **gtest +1: `gtest/tree/ntuple/ntuple_randomaccess` PASSES** (`/tmp/r54/ntuple`: the area 2 / 32 / 14 / 1, unchanged except that gain), the kind **58 -> 59**. Nothing lost: `gtest/core/{foundation,meta,cont,base}/*` **13 / 9 / 4** with the two round-53 gains intact (`/tmp/r54/core`), and the 49 `gtest/tree/ntuple/*` above. Gate `TOTAL 119669 0 30000 0 100%` (13 017 + 44, `/tmp/r54/gate1.log`) |
| W0-133 | **done** | `2ce1f188` | 0 | **closed**: `TInterpreter::Calc` is `_fast_processor`, so it reads a value out without cling's echo - `gInterpreter->Calc("1+2")` prints nothing where `ProcessLine("1+2")` prints `(int) 3`, measured against ROOT 6.40 (`/tmp/r54calc.C`). **`W0-133`'s own row was wrong about the cost**: no existing assertion needed changing, because `_fast_processor` falls back to `process_line` when a translator has only that one - `tests/test_pyroot_groot.py` and `tests/test_cint_run.py` pass untouched (53 passed) |
| W0-134 | todo | | 1 | `gtest/tree/ntuple/rfield_string` moved its wall from a truncated file (fixed by `98df6a2e`) to **`'RNTupleDescriptor' object has no attribute 'GetClusterDescriptor'`**: the test wants a column's *pages* - `descriptor.GetClusterDescriptor(0).GetPageRange(1).GetPageInfos()` - the page-level introspection of an RNTuple descriptor, which is that test's own subject (a `std::string` column overflowing to a second page). A descriptor feature, not a name |
| W0-135 | todo | | 1 | `gtest/tree/tree/testTChainSaveAsCxx` still stops in `SetUp` with `chaintree-fr.root ends at 0, where 100 bytes were expected from 0` - three `TFile`s are declared in one scope and the **third is written empty**. Which directory a `TTree`'s constructor registers with when three files are open is not measured here: ROOT's own run passes, so its answer differs from this kit's, and ROOT's `TTree::TTree` attaches to `gDirectory` (`pyroot/trees/`). The oracle can settle it in one probe; nothing may be changed until it does |
| W0-136 | todo | | 0 | **another session's stopped processes starve a whole-kind run**: a `--kind gtest` run at `--jobs 3` sat at 226/350 for 25 minutes with its three workers in state `Ts` (suspended), while two `xrdroot run` processes from `/private/tmp/suite-9c38` were `Rs` at 84 % CPU. Nothing in this tree stopped them; `kill -9` of this run's own three let it be killed cleanly. **A round must check `ps -o pid,stat` for its own workers before waiting on a whole-kind run**, and measure the areas it touched instead when the machine is shared (`W0-38`, `W0-49`, `W0-111`) |
| **Round 55: the IO-feature bits a ROOT 6.40 basket records** - `d1752d03` (`ROOT.EIOFeatures` and `ROOT::Experimental`'s two, `TBasket::kIOBitCount`/`EIOBits::kSupported`, every number measured against the oracle) | `d1752d03` | **gtest +1: `gtest/tree/tree/testTIOFeatures` PASSES** (`/tmp/r55/areas`: `gtest/hist/*` + `gtest/tree/tree/*`, 73 tests, **30 PASS / 37 FAIL / 5 UNSUPPORTED / 1 TIMEOUT**, that one the only gain and **nothing lost**), the kind **59 -> 60**. Gate `TOTAL 119689 0 30000 0 100%` (13 020 + 44 passed, `/tmp/r55/gate1.log`) |
| W0-137 | todo | | 0 | **the first whole figure for `roottest/root/hist/*` + `roottest/root/tree/*`**: 313 tests, **138 PASS / 115 FAIL / 22 DIFF / 36 UNSUPPORTED / 1 SKIP / 1 TIMEOUT** (`/tmp/r55/rhist`, `--jobs 3`). No round has measured these two areas whole since round 17's suite run, so this is a snapshot and not a delta - the next round that touches them has a *before*. The TIMEOUT is `roottest/root/tree/cloning/runtreeCloneTest2` (known, `W0-39`'s neighbours); the two areas' 115 FAILs are unranked |
| W0-138 | todo | | 0 | **the round's own procedure assumes a `main` this repository does not have**: `TODO`-level step 0 of `.ralph/ROOT.md` opens a round off `origin/main` and step 6d pushes `$ROUND-main:main`. Measured this round, `main` is **238 commits behind `feat/root-testsuite`** and `origin/main` 45 behind, so a round based on either would lose every fix of rounds 1-54; and 54 rounds of work have never been pushed. **The round therefore opened a worktree off `feat/root-testsuite`'s tip `127c20e2`, integrated there, and did not push.** Whoever owns that decision must say whether `main` is meant to become the trunk (one deliberate push of 238 commits) or whether `feat/root-testsuite` is - until then a round keeps to `feat/root-testsuite` and records it |
| **Round 56: an array of objects is destroyed as its scope ends** - `4f884ec7` (`cppstd.objects.ObjectArray`; `_destructible` asks a one-dimensional array's *element*; `std::make_unique<A[]>(n)` builds the array `A a[n]` is) | `4f884ec7` | **gtest +1, and the kind measured whole for the first time since round 53: 61 PASS / 211 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r56/gtest`, `--jobs 3`). The five it gained over round 53's 56 are `testRRangeCast`, `testTDataType` (round 53), `ntuple_randomaccess` (54), `testTIOFeatures` (55) and **`testMake_unique`** (56) - **and none lost**, which is the `W0-63` check the destructor machinery needed. Gate `TOTAL 119714 0 30014 0 100%` (13 026 + 44, `/tmp/r56/gate3.log`) |
| W0-139 | todo | | 2 | **the bulk-fill tests sit near the 300 s wall**: `gtest/tree/ntuple/ntuple_extended` took **133.0 s** and `gtest/tree/tree/testBulkApiMultiple` TIMEOUT at 300 s in round 56's whole-kind run (`/tmp/r56/gtest`), and `ntuple_largefile1` with it; round 55 saw `ntuple_extended` at 255 s and `testBulkApi` at 193 s. All are the same pathology - a per-entry Python loop where ROOT writes a column in one pass - so the fix is the batched fill `P-9` and the `W0-108` decision's neighbours, not a bigger limit. Measure one alone before touching it |
| W0-140 | todo | | 0 | **an array of *arrays* of objects is still not destroyed**: `A a[2][2]` keeps plain Python lists for its rows (`cppstd.objects._object_array` returns them for `len(dims) > 1`), so `pycxx`'s `_owned` gives `None` and runs no destructor rather than raising (`tests/test_cint_scope_destructors.py` pins that). C++ destroys four `A`s. One dimension is done (round 56); the rows want a nested `ObjectArray` each |
| W0-133 | todo | | 0 | **`TInterpreter::Calc` echoes where ROOT does not**: `gInterpreter->Calc("1+2")` prints nothing in ROOT 6.40 and xrdroot writes `(int) 3` (`/tmp/r53/calc.C` measures both, and ROOT's `ProcessLine` *does* echo `(int) 5` for the same shape). `pyroot/core/troot.py`'s `TInterpreter.Calc` is `gROOT.ProcessLine` verbatim, and the silent path already exists (`_fast_processor`, `ProcessLineFast`). Two existing tests pin the echo - `tests/test_cint_run.py:199` and `tests/test_pyroot_groot.py`'s fake `process_line` asserting `seen` holds `"d"` - so the fix corrects both assertions in the same commit, saying ROOT prints nothing there |
| W0-124 | todo | | 5 | `test/`'s **five** `RooUnitTest` refusals are the largest single blocker in that kind: `test/roofit/roofitcore/stressroofit-{cpu,codegen,codegen_no_grad}` and `test/roofit/roostats/stressroostats-cpu{,-minuit2}` all stop at `stressRooFit_tests.h:93` / `stressRooStats_tests.h:115` with `ROOT has RooUnitTest`. `RooUnitTest` is ROOT's base class for a RooFit unit test (a `runTest` per named test, a `RooWorkspace`, an output directory and a pass/fail line), and each stress test is a list of ~40 of them: a framework, not a name. Recorded so the next round knows what the biggest `test/` group costs |
| W0-125 | todo | | 1 | **`test/testbits` needs `TBits::Set`/`Get` (landed, `4a98e20d`) and then `W0-123`'s `Char_t[]` buffer** - measured after the methods: `Assertion (bits.TestBitNumber(j)) failed at test/testbits.cxx:134`, because the array it reads the bits from is the empty string. Nothing else in the test is missing |
| W0-119 | todo | | 26 | **the 26 `CreateTemporary` programs are the Clad derivator family** (`math/mathcore/test/CladDerivatorTests.cxx` and its siblings): each does `gInterpreter->CreateTemporary()`, `gInterpreter->Evaluate("clad::differentiate(pow2, 0).execute(1)", *value)` and reads `value->GetAsDouble()`. `CreateTemporary`/`Evaluate` are ordinary `TInterpreter` API (a temporary `TInterpreterValue`), but `clad::differentiate` is cling's Clad plugin. **This is a scope decision, not a fix**: either the committed scope labels cover "a compiler plugin implementing automatic differentiation" - which they must say in a full sentence - or the derivative is computed numerically for the polynomials the tests use. Do not label it either way without a review |
| W0-120 | todo | | 1 | `std::vector::resize` is missing: `gtest/tree/treeplayer/leafs.cxx:101` says `'list' object has no attribute 'resize'` (the vector is a Python list here). It is `treeplayer_leafs`'s next wall after `4e473e1e`, and `cppstd`'s other size-changing methods (`reserve`, `shrink_to_fit`) should be checked at the same time |
| W0-121 | todo | | ? | **a `.cxx` program with `main()` runs its namesake function instead**, and echoes the returned value as a macro's (`testGSLRootFinder` prints `(int) 0` where a compiled ROOT program prints nothing). `xrdroot run` looks for the file's namesake before `main`. Where the judge compares only the exit code it is invisible; where a `test/` or `other` test has an `OUTREF` it would be a DIFF. The order should be `main()` first for a `.cxx`, as `root -q file.cxx` compiles and runs it |
| W0-122 | todo | | 1 | `TH1::SetStatOverflows(kConsider)` does not exist, which is what `gtest/hist/hist/testTH1`'s `TH1.StatOverflows` case asserts: the *constant* `TH1.kConsider` landed in `4e473e1e`, the behaviour - the statistics counting the under- and overflow bins - did not. `hist.py`'s member table already writes `fStatOverflows: 2` |
| W0-117 | todo | | 1 | **a pre-existing flake the coverage gate hides**: `tests/test_pyroot_distrdf_graph.py::test_a_dataset_spec_is_cut_like_a_chain_of_its_samples` fails on its own at the round's base `4dad0dad` (measured in a clean worktree) and passes under `--cov`, because the warning is a `ResourceWarning` for an unclosed `a.root` raised when a file object is collected - and coverage tracing keeps the object alive past the test, where the file's autouse fixture closes it. The traceback names `rdf/distributed.py:73`, `Window.change`'s `self.inner, self.start, self.stop = inner, start, stop`: the chain it replaces is dropped with its `open_root` files still open, and the fixture can only close what is still in `xrdroot.file.OPENED`. Next step: give a replaced `owned` inner a `close()` where it is provably the last reference - or make the test hold the headnode it drops |
| W0-115 | todo | | 2 | the gtest kind's `tree/dataframe/dataframe_interface` needs `ROOT::Detail::RDF::RActionImpl<T>` (`CounterHelper.h` uses it as a CRTP base for an action), and `core/base/CoreBaseTests` needs `TQObject`'s `Emit`/`Connect` surface. Both are features rather than names; the round-13 gtest ranking is still the widest view of that kind (`/tmp/r13/final1/results.json`), but it is 29 rounds old and should be re-measured whole |

| W0-102 | done | | 0 | **closed after three rounds**: the four `Warning in <TClass::Init>: no dictionary for class …` lines the test wants (`TEmcl`, `TPhysObj`, `TNonEmcl`, `TNonPhysObj`) are emitted, and ROOT's own `TGeoAtt`/`TGeoVoxelFinder`/`TGeoParallelWorld` stay silent because they are named in `ROOT_CLASSES` - "ROOT has a dictionary for it and this kit implements nothing of it" (`pyroot/core/reflection.py`). The reverted round-37 attempt was right except for those three |
| W0-108 | decision | | 0 | **`other/math/mathcore/mathcore-stdsort` is a benchmark, not a correctness test**: `stdsort.cxx` runs `200000/(n/10)` passes of `TMath::Sort` and `std::sort` over vectors from 10 to 100 000 elements, with a comparator struct per pair - 10^7 Python-level calls by design, and ROOT's own run prints microseconds per call. Measured: `xrdroot run stdsort.cxx` ran **4 minutes without finishing the first size**. It cannot pass inside the 300 s limit and the limit is not to be raised, so it stays TIMEOUT and is recorded here rather than chased |
| W0-109 | todo | | 0 | **the gate needs a load average under ~20, again**: at 34-54 this round's two-command gate reported **99.54%** and **99.85%** with all 12 931 tests passing - a worker's coverage data lost, not a gap. The last clean measurement is round 39's `TOTAL 118830 0 29826 0 100%`; round 40's only outstanding gap (2 lines + 1 branch in `pyroot/core/declared.py:551, 582->581, 584`) is closed by `de1e9cac`, whose file reports `declared.py` at 98% alone with those three covered |

| **Round 39: the gate measured again, and the first harness numbers in three rounds** - `4f7ddee5` (both of the trainer's optimisers) | `4f7ddee5` | **the gate is green**: `TOTAL 118830 0 29826 0 100%` by the two-command decomposition (12 925 pass, then 44 at `--cov-append`). **`other` +1**: `other/math` **21 -> 22 PASS** (`/tmp/r35/other` -> `/tmp/r39/other`, `mathcore-testSpecFuncBeta`), 0 FAIL, 43 -> **41 UNSUPPORTED**, `mathcore-stdsort` now *runs* and is the round's one new TIMEOUT. `io/tree/read` is **4 lines** from PASS (`/tmp/r39/read`) |
| W0-107 | done | | 0 | **closed**: `tmva/nettrain.py`'s `_step` has two branches - ADAM and momentum - and **no test reached either** (the deep-network tests train through the trainer without calling a step). One parametrised test (`4f7ddee5`) calls it with each optimiser and checks each rule's first step: momentum's is `lr * gradient`, ADAM's is `lr` once its moments are corrected. With it the gate reads **100%** |
| W0-108 | todo | | 1 | `other/math/mathcore/mathcore-stdsort` - **the test `W0-98`'s feature unblocked** now *runs* and hits the 300 s limit (`--jobs 1`, exit None, timed out at 300.007 s). The translated `std::sort` over its array goes through `ItemRef`s one element at a time; `sort_range` in `cppstd/` needs to sort the container's own buffer in one NumPy pass (`np.sort`/argsort on a view) rather than a Python-level comparator loop. One test, one file - and **the limit is not to be raised** |
| W0-102 | open, sharpened | | 1 | `roottest/root/io/tree/read` is **exactly four lines** from PASS: `missingClass.ref` begins with `Warning in <TClass::Init>: no dictionary for class TEmcl is available` (and `TPhysObj`, `TNonEmcl`, `TNonPhysObj`), and our output has none. Round 37's reverted commit warned for them **and** for `TGeoAtt`/`TGeoVoxelFinder`/`TGeoParallelWorld`, which broke a geom PASS. **The two-part fix**: those three are ROOT's own classes and belong in `winfo.INFOS` (ROOT has streamer info for them); then the warning "a class in the file's streamer list that is in neither `INFOS` nor `DECLARED`" is exactly right, and both tests pass |

| **Round 38: the vector-element feature back with its tests, and three pre-existing breakages fixed** - `bcd98495` (the address of a vector element reaches every branch it added), `2c64e67f` (`wpacking`'s `OverflowError` on Python 3.14 and `np.atan`'s spelling), `db1c1a00` (two float comparisons through their decimal spelling, as a tolerance) | `bcd98495`, `2c64e67f`, `db1c1a00` | **0 harness PASS** (no area was measured - the load average stayed 20-124, `W0-103`); **unit tests: 916 pass** in every file the three commits touch (`tests/test_cint_*`, `tests/cppstd`, `tests/pycxx`, `tests/test_wpacking.py`, `tests/test_random_libm_batches.py`, `tests/test_root_containers.py`) |
| W0-101 | done | | 0 | **closed**: the address of a vector element (`b28bb2c2`'s feature) is back in the tree **with the six branches it needed covered** (`bcd98495`): `cells.ItemRef.__repr__`, `_element_pointer`'s and `_element_call_end`'s `None` fall-backs (a range end taken by `at()`, a member call that names no element and is refused, and a plain name whose address *is* a pointer), and `_element_call_address`. The two forms that failed in round 37 were `cintfake.fake()`'s `std::vector<double> v(3, 1.5)` (no `(size, value)` constructor there) and the `list expected at most 1 argument` of two dereferenced pointers in one function - `push_back` and one pointer per test avoid both |
| W0-104 | done | | 7 | **three pre-existing breakages on the branch, fixed** (`2c64e67f`, `db1c1a00`): (a) `wpacking._whole` caught `struct.error` for a number too wide for a member, but **Python 3.14 raises `OverflowError`** - six `test_wpacking` cases died instead of falling back to NumPy (1 line); (b) `test_random_libm_batches` asked NumPy for `'atan'`, which is **NumPy 2.0's** alias for `arctan` and the venv has 1.26 - a spelling map; (c) `test_root_containers` compared `round(x, 5) == 30.39946` and `round(x, 5) == -5.89043` with `==` - two *different floats* whose repr matches, so it asserted a spelling; a `pytest.approx` with the same five-decimal precision says what it meant. All of them predate round 38 (bisected to the branch, not to this round) |
| W0-105 | todo | | 0 | **the venv runs NumPy 1.26.4 and Python 3.14**, while parts of the tree assume NumPy 2.x aliases (`np.atan`) and Python's older `struct` exceptions. A round that touches numerics or `struct` should check both: `python -c "import sys, numpy; print(sys.version_info[:2], numpy.__version__)"`. CI on 3.10-3.13 will not reproduce either class of failure |

| **Round 37: two commits in, two reverted, and no gate measurable under load 51-124** - `4e0461e6` (a TMap's JSON as the pairs it holds), `aa265151` (a symmetric SMatrix filled from its upper triangle) | `4e0461e6`, `aa265151`; **reverted**: `b28bb2c2` (the address of a vector element) and `03c2c442`+`afd8afdd` (a user class's missing dictionary) | **0 PASS**: `io/json` **10 / 0 / 0 / 2** (`/tmp/r37/json`, unchanged) and `roottest/root/math/*` **0 / 4 / 1** (`/tmp/r37/rmath`, unchanged). `io/tree` not re-measured (its last DIFF needs w03's reverted work) |
| W0-101 | revert | | 0 | **the address of a vector element** (`W0-98`'s top row, 10 of `other/math`'s 43) is implemented and **reverted for coverage**: its six new branches (`cppstd/cells.py:94`, `pycxx/emit_calls.py:389`, `:397-399`, `emit_expr.py:538`) have no test that reaches them, and the 100% floor may not be relaxed. Branch **`r37-01`** keeps it. The two tests it needs: `&v.at(i)`/`&v.front()` as an initialiser (the `ItemRef` form), and a call whose name is neither (`&v.size()`) falling through to the member's own address - both were written and both hit a *translator* limit first (`list expected at most 1 argument, got 2` when two dereferenced pointers coexist, and `a float returned by value cannot be assigned to in place` for `v.front() = x`) |
| W0-102 | revert | | 0 | **a user class's missing dictionary** (`03c2c442`, `afd8afdd`) is **reverted**: the guard that warns for a class with no dictionary also warned for `TGeoAtt`, `TGeoVoxelFinder` and `TGeoParallelWorld` - ROOT's own classes, which ROOT *has* dictionaries for - and `tests/test_pyroot_geom_rootfile.py` pins ROOT's exact message sequence when a geometry is read from a file. An existing PASS outranks `io/tree/read`'s DIFF. To take it later, those three need entries in `winfo.INFOS` (their streamer layouts) or the guard needs to know which classes ROOT carries |
| W0-103 | todo | | 0 | **no gate could be measured this round**: the machine's load average went 18 -> **124** from another session. Under it, `pytest -n 2` failed 7 `tests/test_tmva_learn_networks.py` cases that pass alone (15 in 3.9 s), `--cov` reported **0-27%** with "The following workers failed to return coverage data", and one single-process run died with `Fatal Python error`. **`W0-45` is worse than recorded**: above a load average of ~20 the coverage gate is not measurable at all, and no conclusion may be drawn from it - rerun when it falls |

| **Round 36: the whole TRef family closed, and `other/math`'s refusals bucketed** - `a17cb476` (a reference written after a clone names the file's process record), `b73ea7eb` (ROOT's own array-compression forms in JSON) | `a17cb476`, `b73ea7eb` | **roottest +3: `io/tree` 14 -> 17 PASS**, 4 -> **1 DIFF**, 0 FAIL (`/tmp/r35/tree` -> `/tmp/r36/tree`: `test-rcr2`, `test-rcrr2`, `test-rcrr123`). `io/json` unchanged at **10 / 0 / 0 / 2** (`/tmp/r36/json`) |
| W0-95 | done | | 0 | **closed**: a reference written after a clone names the file's process record (`a17cb476`) - the clone-of-a-clone level, and with it the whole `TRef` family: `test-rcr2`, `test-rcrr2`, `test-rcrr123` PASS. `io/tree` is **17 PASS / 0 FAIL / 1 DIFF** (`read`'s `missingClass`) **/ 2 UNSUPPORTED** (`tref-write`/`tref-read`, `the branch 'Refs' holds a TRef`) |
| W0-96 | partly | | 0 | **bucketed**: round 36's worker read all 43 refusals and reduced them to **22 causes** with a file and first step each (its report is `/tmp/r36/w03/R36-math-buckets.md`; it ran no harness). `W0-98` takes its top row; the rest are listed there |
| W0-98 | todo | | 10 | **`other/math`'s largest refusal, 10 of 43 tests**: the translator refuses **the address of a vector element** - `&v[0]`, `&v.front()`, `&v[i]` - as a pointer or iterator (`src/xrdroot/pycxx/emit_calls.py:354`, `:367`); `std::sort`/`std::copy`/`std::generate` then take it and write back through it. Next by size: `ROOT::Math::Random<T>` over the engine kinds (4, `pyroot/core/randoms.py`), MathMore's GSL wrappers (4, `rmath.py`), `std::generate`/`shuffle`/`next_permutation` (3, `cppstd/`), the k-d tree family (3, a new `kdtree/`), then eleven single-test causes each with its file named in the report |
| W0-99 | todo | | 2 | `roottest/root/io/json/{ArrayCompress,RootClasses}` are still the area's two UNSUPPORTED after `b73ea7eb`, which added ROOT's own array-compression forms to the JSON write/read tests without changing the refusal. Read their two refusals again - the tests now have unit coverage, so the remaining gap is named inside `jsonbuffer.py` |
| W0-100 | todo | | 1 | `roottest/root/io/tree/read` is the family's last DIFF (`missingClass.ref`): the `io/tree` area is otherwise **17 PASS / 0 FAIL / 2 UNSUPPORTED**. Worth one measurement of its diff with `custom_diff` before the next `TRef` work |

| **Round 35: the clone's ProcessID record, at last - and five PASS** - `8d04425c` (AssignSym and the MatRepSym a translated type names as text), `8566c730` (a clone writes the identifiers its objects were numbered with), `0c08a823` (a TObject read member at a time carries the two words its object has), `d481532d` + `2a469727` (ROOT::Math's genetic minimiser, its chi-square under a template name, and its seed) | `8d04425c`, `8566c730`, `0c08a823`, `d481532d`, `2a469727` | **roottest +3**: `io/tree` **11 -> 14 PASS**, **0 FAIL**, 7 -> **4 DIFF** (`/tmp/r34/tree` -> `/tmp/r35/tree`: `test-rcr`, `test-rcrr`, `test-rcrr3`). **`other` +2**: `other/math` **19 -> 21 PASS**, 45 -> 43 UNSUPPORTED (`/tmp/r33/other` -> `/tmp/r35/other`: `genetic-testGAMinimizer`, `genetic-GAMinTutorial`). `roottest/root/math/*` unchanged at 0 / 4 / 1 (`/tmp/r35/rmath`) |
| W0-91 | done | | 3 | **closed after four rounds**: `8566c730` (a clone writes the identifiers its objects were numbered with) + `0c08a823` (a `TObject` read member at a time carries the two words its object has) are round 33's two named edits, and they closed **three** of the six tests - `test-rcr`, `test-rcrr`, `test-rcrr3`. Round 33's note was right in every particular, including that both halves were needed |
| W0-85 | done | | 0 | **closed**: the member-wise column fill now gives a split `TObject` the identifier and bit word its object has (`0c08a823`), which is what the three tests above needed |
| W0-95 | todo | | 3 | `roottest/root/io/tree/{test-rcr2,test-rcrr2,test-rcrr123}` - the same feature one level deeper: the `r2` references resolve to the **base** object (`ref: N_0_8`) where ROOT's resolve to the **first clone** (`ref: r1_N_0_8_8`), 180 lines each, counts identical. So a clone of a *clone* is not numbered into its own `ProcessID` record yet - the same two edits applied where a stage reads a file another stage wrote (`pyroot/trees/copying.py` + `wbranch.py`) |
| W0-87 | done | | 1 | **closed as far as the named blockers go**: `genetic-testGAMinimizer` and `genetic-GAMinTutorial` PASS (`d481532d`, `2a469727` - the genetic minimiser, `Chi2FCN` under its template name, and the seed as ROOT's). `other/math` is **21 PASS / 0 FAIL / 43 UNSUPPORTED / 1 TIMEOUT**. What is left there is named in `W0-96` |
| W0-96 | todo | | 41 | `other/math`'s 43 UNSUPPORTED are mostly **translator** refusals (`mathmore-testDerivation`/`-testIntegrationMultiDim`/`-testMCIntegration` need C++ pointer arithmetic: `WrappedParamFunction(&f, dim, p, p+1)`), plus `minuit2_testADMinim`'s `ROOT.Minuit2.Minuit2Minimizer` and `mathcore-testFit`'s `Fit.BinData.GetErrorType`. Worth one measurement pass that buckets all 43 by reason before any worker takes one |
| W0-97 | todo | | 5 | `roottest/root/math/*` is still **0 PASS / 4 FAIL / 1 UNSUPPORTED** after `8d04425c` and `efa5f946`; the four failures move with each SMatrix addition but none closes. A worker should take the **whole area's five macros** and read all five `.ref`s at once instead of one error at a time |

| **Round 34: two of the three halves, no PASS** - `616ee69f` (ROOT::Math's free functions on SVector/SMatrix and the constructors a macro writes), `b1cd1112` (a split `TObject`'s identifier keeps the process byte the file carries), plus the SMatrix test split to stay inside the CCN limit | `616ee69f`, `b1cd1112` | **0 PASS**: `io/tree` stays **11 PASS / 7 DIFF / 2 UNSUPPORTED** (`/tmp/r34/tree`) and `roottest/root/math/*` stays **0 PASS / 4 FAIL / 1 UNSUPPORTED** (`/tmp/r34/rmath`), though both groups' failures moved |
| W0-91 | partly | | 6 | **two of the three pieces are in**: `b1cd1112` writes the process byte into a split `TObject`'s identifier column, and `a65836dd`/`ca510e97` already gave the object path its two values. What is missing is round 33's **first** edit - `src/xrdroot/wbranch.py`'s `tref_record`/`_ref_bytes` must write the process's **file** index, `file.WriteProcessID(ref._pid)`, not its session index - and the second half of the second: `pyroot/trees/copying.py`'s `ClonedSlot` must read a clone's `fUniqueID`/`fBits` from the **live objects** in `tree._addresses[leaf.branch]`. `test-rcr` still differs in `r1.fUniqueID`/`r1.fBits` only, exactly as round 33's note predicted ("with (2) alone, stage 1 would break"). Round 34's worker for it produced no code at all |
| W0-90 | open | | 5 | `roottest/root/math/*` still 0 PASS / 4 FAIL / 1 UNSUPPORTED after `616ee69f`; re-measure and re-read the five reasons (`testSMatrix.cxx:103`, `testOperations.cxx:95`, `testInversion.cxx:303`, `ROOT.Transpose`, `testLoadLibMathMore`) - each failure moved, none closed |
| W0-93 | todo | | 0 | **a worker's own test can fail the maintainability gate**: round 34's SMatrix commit shipped a test with **CCN 14** against the limit of 10, and `pytest` caught it through `test_maintainability.py`. The round split it into three (kept every assertion). **`W0-74`'s rule is every gate after every cherry-pick, not coverage alone** |
| W0-94 | todo | | 0 | **the fan-out's yield is falling**: round 34's four workers produced one commit each from two (w02, w03) and *nothing* from the other two (w01 wrote no code at all, w04 nothing); the two rounds before lost 2 and 2. A round should start its workers with a **smaller, single-commit target** and check the session files within the first ten minutes rather than waiting an hour for nothing |

| **Round 33: a minimiser that clears, and ROOT::Math's SMatrix** - `88fa5527` (`Minimizer.Clear()`/`GetMinosError()`), `e3838fad` (ROOT::Math's SVector and SMatrix, subscripted as the macro writes them) | `88fa5527`, `e3838fad` | **`other` +1**: `other/math/mathcore-testMinim` PASSES (`/tmp/r32/other` -> `/tmp/r33/other`: **19 PASS / 0 FAIL / 45 UNSUPPORTED / 1 TIMEOUT**). `roottest/root/math/*` is still **0 PASS / 4 FAIL / 1 UNSUPPORTED** (`/tmp/r33/rmath`) but all five now *run* and name their next wall (`W0-90`) |
| W0-87 | partly | | 5 | **one of six taken**: `mathcore-testMinim` (`88fa5527`). Still open: `minuit2_testMinimizer` and `minuit2_testADMinim` (`ROOT.Minuit2.Minuit2Minimizer`), `mathcore-testFit` (`Fit.BinData.GetErrorType`), `genetic-GAMinTutorial` (`_FitNamespace.Chi2FCN`), and the three needing translator pointer arithmetic |
| W0-89 | partly | | 5 | **taken as far as running**: `roottest/root/math/*` is 0 PASS / 4 FAIL / 1 UNSUPPORTED, each with a named wall (`W0-90`). SVector/SMatrix exist (`e3838fad`) |
| W0-90 | todo | | 5 | `roottest/root/math/*`, measured at `/tmp/r33/rmath`: `testLoadLibMathMore` -> `AttributeError: mo…`; `smatrix/Inversion` -> `TypeError: "SMatrix<d…` at `testInversion.cxx:303`; `smatrix/Kalman` -> `AttributeError: ROOT has Transpose; xrdroot.pyroot does not yet`; `smatrix/Operations` -> `ValueError: invalid l…` at `testOperations.cxx:95`; `smatrix/SMatrix` -> `TypeError: an SMatrix o…` at `testSMatrix.cxx:103`. Five named next steps in `src/xrdroot/pyroot/linalg/smatrix.py` + `ROOT.Transpose` |
| W0-91 | todo | | 6 | **`W0-79`'s root cause, measured byte for byte by round 33's worker** (its uncommitted note is `/tmp/r33/w02/W02-tref-note.md`): ROOT numbers a clone into a **new `ProcessID` record** when `addref`'s `TRef` constructor calls `TProcessID::AssignID`, and writes *that* index into the reference **and** into each cloned object's identifier top byte (`r1.fUniqueID = 0x02000009`, `r1.fBits = 50331672`; we write `0` and `50331648`). Two edits, both named: (1) `src/xrdroot/wbranch.py`'s `tref_record`/`_ref_bytes` writes the process's **session** index where `TRef::Streamer` writes its **file** index - `file.WriteProcessID(ref._pid)`, which `pyroot/trees/objects.py::_named_processes` already calls and discards; (2) `src/xrdroot/pyroot/trees/copying.py`'s `ClonedSlot` copies a clone's `fUniqueID`/`fBits` from the source tree's decoded columns, so an identifier `AssignID` changed after `GetEntry` goes out old - its addresses must read the live objects in `tree._addresses[leaf.branch]`, as `_member()` already does for the non-clone path. **With (2) alone, `r1`'s refs would resolve to the clones and stage 1 would break** - both are needed |
| W0-92 | todo | | 0 | a file xrdroot writes carries **no `TRefTable`** at all (`pyroot/trees/tree.py:251` never writes a `fBranchRef`; `wtree.py:1499` writes `0`), and `read_referenced()` compensates by reading *every* split collection of the entry in branch order - which is why a number collision inside one process resolves by branch order instead of by `TRefTable.GetParent(uid, pid)` |

| **Round 32: a container copies what it is given, and ROOT::Math's names** - `8c150a28` (a set or a map keeps a copy of what it is given, as C++ does), `62afe7fe` (ROOT::Math's integration kinds, function interfaces and Util) | `8c150a28`, `62afe7fe` | **roottest +1**: `roottest/root/io/json/STL` PASSES - `io/json` is **10 PASS / 0 FAIL / 0 DIFF / 2 UNSUPPORTED** (`/tmp/r32/json`). **`other` +2**: `other/math/*` **16 -> 18 PASS**, no PASS lost (`/tmp/r32/other`, measured by the worker; `mathcore-testIntegration` and `mathcore-testAnalyticalIntegrals`) |
| W0-86 | done | | 0 | **closed**: a `std::set`/`std::map` keeps a copy of what it is given, as C++ does (`8c150a28`), so `TJsonEx13`'s `set1` is written in full and `STL` PASSES |
| W0-87 | todo | | 6 | `other/math`'s next step, measured by round 32's worker: `pyroot/core/mathtools.py`'s `Minimizer` lacks **`Clear()` and `GetMinosError()`** - `mathcore-testMinim` and `minuit2_testMinimizer` reach the end of their files and stop only there - and `minuit2_testADMinim` needs `ROOT.Minuit2.Minuit2Minimizer`. Then `Fit.BinData.GetErrorType` (`mathcore-testFit`) and `_FitNamespace.Chi2FCN` (`genetic-GAMinTutorial`). `mathmore-testDerivation`/`-testIntegrationMultiDim`/`-testMCIntegration` need C++ pointer arithmetic the translator cannot express (`WrappedParamFunction(&f, dim, p, p+1)`) |
| W0-88 | todo | | 0 | **The round's process finding**: three workers died without a commit in the last two rounds (round 30's 02/03, this round's 01/02), and the *most* useful product of the ones that live is a **discovery record left uncommitted in their worktree** (`/tmp/r32/w04/docs/discoveries/…`) - correctly, since `docs/` is not theirs. **A round must read `<worktree>/docs/discoveries/` before it closes**: that is where `W0-87`'s six blocked tests and their file came from |
| W0-89 | todo | | 5 | `roottest/root/math/*` is a nearly empty area - 5 tests, **PASS 0** (SMatrix/Operations FAIL, 3 UNSUPPORTED) - and nothing there calls `TMath`; cheap, and untouched by any round |
| W0-85 | todo | | 5 | still open: `roottest/root/io/tree/{test-rcr,test-rcr2,test-rcrr,test-rcrr2,test-rcrr3}` differ in `r1.fUniqueID`/`r1.fBits` **only** - the two values `a65836dd`/`ca510e97` restored on the object path, for a **cloned** file's split collection read through the member-wise column path. Round 32's worker for it died without a commit |
| W0-79 | todo | | 1 | still open: `test-rcrr123`'s third-level `TRef` chain (ours names the base object, ROOT's the previous level's clone). Round 32's second dead worker |

| **Round 31: the two values ROOT's reader restores, and FromJSON filling containers** - `a65836dd` (`fBits`' heap bits), `ca510e97` (`fUniqueID`'s process byte), `46dbd65c` (`FromJSON` fills a container a constructor left empty), `6b3e0928` (a value in full where a pointer is a `$ref`), `d07a00cd` (a `Double32_t` template spelling) | `a65836dd`, `ca510e97`, `46dbd65c`, `6b3e0928`, `d07a00cd` | **roottest +3**: `io/tree/test-rc` PASSES (`/tmp/r31/tree` -> `/tmp/r31/tree2`: `io/tree` 10 -> 11 PASS), and `io/json/STL1` + `io/json/StreamerLoop` PASS while `STL` moved UNSUPPORTED -> DIFF (`/tmp/r30/json2` -> `/tmp/r31/json`: `io/json` 7 -> 9 PASS) |
| W0-83 | done | | 0 | **closed**: a `TObject` reads back with `fBits`' `kIsOnHeap|kNotDeleted` (`a65836dd`) and `fUniqueID`'s process byte - `(pid_index + 1) << 24 | uid` (`ca510e97`), both measured against the oracle before they were written (`/tmp/r30/oracle/`). `test-rc` PASSES |
| W0-82 | done | | 0 | **closed**: `FromJSON` fills the container members an object's constructor left empty (`46dbd65c`), and a value is written in full where a pointer is a `$ref` (`6b3e0928`). `STL1` and `StreamerLoop` PASS; `STL` now runs and differs (`W0-86`) |
| W0-85 | todo | | 5 | `roottest/root/io/tree/{test-rcr,test-rcr2,test-rcrr,test-rcrr2,test-rcrr3}` - each now differs in **`r1.fUniqueID` and `r1.fBits` only**: the same two values `W0-83` restored, but for the objects of a **cloned** file read through a split collection's member-wise path (`ClonesSlot`/columns), which is not the object read those two commits fixed. Extend `a65836dd`/`ca510e97` to the column path: `interp.py`'s `_columns`/`_member_columns` and the `TObject` column fills |
| W0-86 | todo | | 1 | `roottest/root/io/json/STL` - our `TJsonEx13`'s `set1` is 2 343 bytes where ROOT's is 5 144: the set of `TJsonEx1` objects is written with fewer members (ROOT writes each element's `_typename` and its members; ours writes...). Measure with `custom_diff` and compare the two `set1` blocks - `src/xrdroot/pyroot/core/jsonbuffer.py` |

| **Round 30: a translator bug found by a worker and taken by another** - a `char[N][M]` is one buffer of its rows (`71dba22d`), a `std::pair` written on its own is `$pair` (`8e797bcb`), the identifier and its process (`ae646b4e`) | `71dba22d`, `8e797bcb`, `ae646b4e` | **roottest +2: `io/json/BasicTypes` and `io/json/Objects` PASS** (`/tmp/r30/json` -> `/tmp/r30/json2`): `io/json` is now **7 PASS / 0 FAIL / 0 DIFF / 5 UNSUPPORTED** - every test of the area that *runs* passes. `io/tree` unchanged at 10 PASS / 8 DIFF / 2 UNSUPPORTED (`/tmp/r30/tree`) |
| W0-81 | done | | 0 | **closed**: the four `io/json` `TypeError: 'NoneType' object is not subscriptable` failures were **not** in `jsonbuffer.py` - the translator emitted `self.fChar[i][j] = with_char(...)` for a `char[N][M]` member, with the runtime's array a flat buffer of rows. `71dba22d` (a different worker, from the finding alone) fixes it: `BasicTypes` and `Objects` become byte-identical to their `.ref`s. `STL`/`STL1` moved FAIL -> UNSUPPORTED on the next wall below |
| W0-82 | todo | | 3 | `roottest/root/io/json/{STL,STL1,ArrayCompress}` - `TBufferJSON::FromJSON does not yet fill a member that holds values, and TJsonEx7::fVectDouble holds some: it reads back an object whose containers are empty, which is what a class's constructor made`. `FromJSON` must fill an existing object's container members, not only make new ones - `src/xrdroot/pyroot/core/jsonbuffer.py` |
| W0-83 | todo | | 6 | **`W0-78`, now with the mechanism measured**: the `fBits`/`fUniqueID` reassembly does **not** belong in `buffer.py::referenced` - a worker measured that doing it there breaks seven round-trip tests. The file holds `fBits & ~0x03000000` and a plain identifier; ROOT's reader puts the heap's two bits back and the process index **+1** on the object (`TBufferIO::GetPidOffset`, kept in `TKey::fSeekPdir`'s high bits). The reassembly belongs in **`interp.py`'s element read**, where the object is made |
| W0-84 | todo | | 2 | round 30's workers 02 and 03 died without a commit (only stray `docs/` in their worktrees), so `W0-79` (`test-rcrr123`'s third-level `TRef` chain) and `W0-77` (`io/bigevent/read`'s class reference at 10787) are still open with the same briefs. Two deaths in two rounds is a pattern: check the session files, not the logs |

| **Round 29: three one-line behaviours, and the first PASS since round 27** - a null `char*` member is `null` in JSON (`44e2d2ed`), `Show` prints a bare record as the reference it is (`d07c8610`), every object carries the two heap bits (`8ffa748c`) | `44e2d2ed`, `d07c8610`, `8ffa748c` | **roottest +1: `roottest/root/io/json/String` PASSES** (`/tmp/r28/json` -> `/tmp/r29/json`: `io/json` 4 -> 5 PASS, 5 FAIL, 3 UNSUPPORTED). `io/tree` stays 10 PASS / 8 DIFF / 2 UNSUPPORTED (`/tmp/r29/tree`) but `test-rc`'s diff fell from **four line kinds to three** (`W0-78`) |
| W0-78 | todo | | 7 | `roottest/root/io/tree/{test-rc,test-rcr,test-rcr2,test-rcrr,test-rcrr2,test-rcrr3}` - **three lines per entry** now, measured with the harness's own `custom_diff`: ` N.fUniqueID = 16777225, ...` (ROOT) vs `9, ...` (ours), `N.fBits = 50331672` vs `24`, `r1.fBits = 50331648` vs `0`. Same count of each (10 per file). The oracle writes the same `TClonesArray("TNamed")` + `TRef` file for a byte-by-byte comparison - that is the whole task |
| W0-79 | todo | | 1 | `roottest/root/io/tree/test-rcrr123` - the only one of the family that needs a **third-level** `TRef` chain: ROOT's refs name the previous level's clones (`r2_r1_0_0_0 ref: r1_N_0_8_8`), ours name the base objects (`ref: N_0_8 GOOD!`) and the `GOOD!` suffixes pile up (`GOOD! GOOD! GOOD!`). Its own feature, after `W0-78` |
| W0-75 | done | | 0 | **corrected and closed**: its item (c), "one trailing blank line", was **my measurement artefact** - the harness's judge (`tools/rootsuite/refdiff.filtered`) drops blank lines and `Processing …` lines, so they never counted. Measured with `custom_diff`: `String` was exactly **one** line from PASS (fixed by `44e2d2ed`) and `test-rc` four line kinds (three now). Item (b), `fRef` printing `TRef`, is `d07c8610`; item (a) is `W0-78` |
| W0-80 | todo | | 1 | worker 04 of round 29 died without a commit, leaving only stray `docs/` (`W0-59` again). `W0-77` (`io/bigevent/read`'s `FormatError: a class reference at 10787 points nowhere`) is therefore still open, with the same brief |

| **Round 28: five features, none of them enough for a PASS** - `Show` prints a split collection's count and unbroken members (`df9caceb`), a clone carries the `ProcessID` record its `TRef`s name (`1d00c3ed`), an array of strings and an array of pointers read every element (`2aee3989`), a member function named through its class is called on the object (`6224b151`), `TBufferJSON` writes a macro class's member (`db8b97ee`) | `df9caceb`, `1d00c3ed`, `2aee3989`, `6224b151`, `db8b97ee` | **0 PASS**: `io/tree` 10 PASS / 8 DIFF / 2 UNSUPPORTED (`/tmp/r28/tree`, unchanged in PASS), `io/json` 4 / 4 / 1 / 3 with **`String` FAIL -> DIFF** (`/tmp/r28/json`), `io/bigevent` 3 / 0 / 1 (`/tmp/r28/ev`) |
| W0-75 | todo | | 7 | `roottest/root/io/tree/{test-rc,test-rcr,test-rcr2,test-rcrr,test-rcrr2,test-rcrr3,test-rcrr123}` - **three lines per entry from PASS**: (a) `fUniqueID`/`fBits` - ROOT writes a referenced object's process in the top byte (`16777225` = `0x01000009`, `fBits` `50331672`), ours has `9`/`24` (`src/xrdroot/writer.py`/`wobjects` + the reader, which must keep the raw value for `Show`); (b) `r1.fRef` prints `->` where ROOT prints `TRef` - `printing.py::_printed` needs the leaf's declared class, and `tests/test_pyroot_trees_print.py` pins `->` today (the ROOT `.ref` says otherwise); (c) one trailing blank line. 7 tests, one commit each |
| W0-76 | todo | | 1 | `roottest/root/io/json/String` - **one line from PASS** (`/tmp/r28/json`): `ex4.fStr2 = null` (ROOT) vs `ex4.fStr2 = ""` for a null `TString*`, plus the same trailing blank line (`jsonbuffer.py::_string`, `~618`) |
| W0-77 | todo | | 1 | `io/bigevent/read` - the walk now reaches a later container and stops at `FormatError: a class reference at 10787 points nowhere` (`/tmp/r28/ev`), i.e. a `TStreamerObjectPointer` tag whose class the file's streamer list lacks, or whose offset is misread. One member after `fMapTNamedp` |

| **Round 27: four features in one chain and three areas** - a split `TClonesArray` read by `SetAddress` is filled (`42898a9a`), a `TRef` member read out of a tree resolves through its `TProcessID` (`212010d0`), a split `TClonesArray` of a file-only class is cloned (`9867dfc9`), a member-wise container reads a base ROOT gives a record to (`5dd0ebba`), plus `ROOT.memset` and collections in JSON (`fd1d360a`, `32cc0d7f`) | `42898a9a`, `9867dfc9`, `fd1d360a`, `5dd0ebba`, `212010d0`, `32cc0d7f`, `23e557d8` (the six branches they left untested) | **roottest +4 in `roottest/root/io/tree/*`: PASS 6 -> 10** (`/tmp/r26/tree` -> `/tmp/r27/tree2`): `create-rc`, `create-rcr`, `create-rcrr` and **`test-r`**; **0 FAIL** there (was 0) and 8 DIFF where 12 were UNSUPPORTED. `io/json` 4 PASS / 5 FAIL / 3 UNSUPPORTED (`/tmp/r27/base-json` -> `/tmp/r27/json2`, `String` moved on), `io/bigevent` still 3 / 0 / 1 (`/tmp/r27/ev`) |
| W0-66 | done | | 0 | **closed**: a `TRef` member read out of a tree resolves through its `TProcessID` (`212010d0`) - `roottest/root/io/tree/test-r` PASSES and its refs print the names they name. `tref-write`/`tref-read` remain UNSUPPORTED (the same feature on the writer side, `roottest/root/io/tree/tref_test.C`) |
| W0-67 | done | | 0 | **closed**: a member-wise container of a class with a base reads every member (`5dd0ebba`); `vector<TLine>`'s 480-byte desync is gone. The walk now stops further on, at `W0-71` |
| W0-68 | done | | 0 | **closed**: a split `TClonesArray` of a class the program declared is cloned by that declaration (`9867dfc9`) - `create-rc`, `create-rcr`, `create-rcrr` PASS |
| W0-71 | todo | | 1 | `io/bigevent/read` - past `fVectorTLine`, the walk reads a length it cannot trust: `FormatError: an array of 134217728 bytes does not fit in the record` (`/tmp/r27/ev/logs/roottest/root/io/bigevent/read/run.log`). One of the containers after `fMapTNamedp` (`map<TString,TList*>`, `map<EventHeader,…>`, `set<TAxis*>`, `fVectAxis`) is read with the wrong shape; the writer lays them all out, so mirror `wdobjects.py::collection` again |
| W0-72 | done | | 0 | **closed, and its diagnosis was wrong in this row until round 28**: the seven `io/tree` DIFFs were **not** `Show`'s formatting alone. Round 28 measured them line by line: after `df9caceb` and `1d00c3ed` each entry differs in exactly **three** lines - `fUniqueID`/`fBits` (ROOT stores the object's process in the top byte) and `fRef` (ROOT prints the class, `TRef`) - plus one trailing blank line. That is `W0-75` |
| W0-73 | todo | | 9+ | `roottest/root/io/json/*` - 5 FAIL / 3 UNSUPPORTED. `runString` now dies on a translator bug, `TypeError: TJsonEx1.Init() missing 1 required positional argument: 'self'` (a method called unbound on a macro class instance), and `BasicTypes`/`Objects`/`STL`/`STL1` still fail with `TypeError: 'NoneType' object is not subscriptable` inside `ConvertToJSON`/`ConvertFromJSON` |
| W0-74 | todo | | 0 | **The coverage gate must run per commit, not per round**: the round's first five commits measured `99.99%` (seven lines, six branches) because a worker's eight-file commit added `_live`/`_live`-paths and two `printing.py` branches with tests for only some of them. `23e557d8` adds the five missing tests. A round that cherry-picks three commits before running `pytest` pays for it once |

| **Round 26: `fName`/`fTitle` are live members**, a container's nested `iterator`, and the `Event` maps (one feature each, three workers) | `7f1663d1` (`pyroot/core/`), `172093d3` (its tests), `33678ae2` (`interp.py`), `a6336c75` (`cppstd/containers.py`) | **roottest +1: `io/tree/create-r` PASSES** (`/tmp/r26/base1` -> `/tmp/r26/tree`: 5 -> 6 PASS in `roottest/root/io/tree/*`); `test-r` FAIL -> **DIFF** once a container's nested `iterator` could be named; `io/bigevent` stayed 3 PASS / 0 FAIL / 1 UNSUPPORTED (`/tmp/r26/ev`) |
| W0-66 | todo | | 11 | `roottest/root/io/tree/{test-r,create-rc,test-rc,create-rcr,test-rcr,test-rcr2,create-rcrr,test-rcrr,test-rcrr2,test-rcrr3,test-rcrr123}` - `test-r` now *runs* and prints `TTP  ref: (!NULL!)`: a `TClonesArray` of the macro class `tref_test_pid` reads back with **empty names and null `fRef`**, so reading a `TRef` (and filling a macro class's `fName` from the file) is the family's real feature. `tref-write`/`tref-read` are refused for the same thing: `the branch 'Refs' holds a TRef` |
| W0-67 | todo | | 1 | `io/bigevent/read` - after round 26's maps read, the walk reaches **`fVectorTLine`** (`vector<TLine>`, member-wise) and desynchronises: `a container of vector<TLine> written field by field ended 480 bytes from where it said it would` = **48 bytes per event**, exactly `TLine`'s four doubles - a class with a base read member-wise drops its own members. `src/xrdroot/interp.py`'s `_columns`/`_embedded` for a described class's own elements |
| W0-68 | todo | | 11 | `roottest/root/io/tree/create-rc` - `a TClonesArray of tref_test_pid is split here only for a class this writer carries a description of; give the branch split level 0`. `tIn->CloneTree()` of a split branch whose element class lives only in the **file's** description (a macro class), not in `wdeclared.DECLARED`. That is the writer's `SplitClones` half of the same chain |
| W0-69 | todo | | 0 | **`W0-59` recurred**: worker 04 committed `572b39e2` touching `docs/README.md` and `docs/discoveries/…`; the round did not take it. A brief's "docs/ is not yours" is not enough - a round checks `git show --stat` of every worker commit for `docs/` before cherry-picking |
| W0-70 | todo | | 0 | worker 03 left scratch files `dbg.py`, `trace.py` untracked in its worktree (harmless, not copied) - the shape of `W0-61`: copy *the files a commit names*, never `git add -A` in a worker's tree |
| **Round 25: an array of pointers to containers** (`W0-62`), the wall round 24 identified and reverted | `23be66d4` (interp.py), `b25b54a4` (rdobjects.py) | 0 PASS: `io/bigevent` stays **3 PASS / 0 FAIL / 1 UNSUPPORTED** (`/tmp/r25/ev`) and `read`'s refusal moved from `fVectorTnamed` to `fMapTNamedp` - two members further than round 24 managed, and this time with the full suite agreeing |
| W0-64 | todo | | 1 | `io/bigevent/read`'s next wall, named: `fMapTNamedp` is `map<TNamed*,int>` (stype 500 = `kSTLp`, extras `(4, 61)`) and the reader refuses it as **a class with a streamer of its own** - so a `std::map` whose *key* is a ROOT class needs the class-stored-map reading, not the plain one. The writer already lays this member out |
| W0-65 | done | - | 0 | round 24's one-line revert (`W0-62`) was the right call and round 25 is the proof: the *feature* (an array of pointers to containers read out of one record, each item named) removed the wall and left every gate green, where relabelling the stype had removed the refusal and broken a unit test |


**Totals, measured.** Four rounds of work took the suite from 1185 to 1231 passing tests:

| | at `10e0e58` | `ddc6039` | `8f53999` | `5ae0a0f` | `68a8d07` | `7a43bd9` | `5c88c8c` | `8af51aa` | `932164a` | `f583422` | `3825443` | `2d0e458` | `f65e991` | `2ef9c80` | `71aa3256` | `90f5c5e8` | `f7e42087` | `a5d25f67` | `19be128b` | `601996d1` | `2dad75a7` | now (`13034109`, intact tree) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| roottest | 1038 | 1044 | 1052 | 1070 | 1076 | 1078 | 1089 | 1105 | 1114 | 1123 | 1132 | 1134 | 1138 | 1141 | 1183 | 1181 | 1195 | 1220 | 1226 | 1230 | 1233 | **1239** |
| pyunittest (files / cases) | 78 / 439 | 80 / 469 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 81 / 470 | 80 / 470 | 80 / 470 | 80 / 470 | 80 / 470 | 80 / 470 | 80 / 470 | 80 / 470 | 80 / 470 |
| gtest (files / cases) | 52 / 507 | 52 / 507 | 52 / 510 | 52 / 512 | 52 / 512 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 | 52 / 511 |
| `test/` | 4 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| `other` | 14 | 16 | 16 | 16 | 16 | 16 | 16 | 21 | 21 | 22 | 22 | 21 | 21 | 21 | 21 | 21 | 21 | 28 | 29 | 29 | 29 | 29 |
| **all non-tutorial** | **1185** | **1198** | **1207** | **1225** | **1231** | **1233** | **1244** | **1265** | **1274** | **1284** | **1293** | **1294** | **1298** | **1301** | **1342** | **1340** | **1354** | **1386** | **1393** | **1397** | **1400** | **1406** |

The last column is round 16's `/tmp/r16/final2` (roottest, 1233) and `/tmp/r16/final1` (the other
kinds) at `2dad75a7` **plus this round's six areas** (`/tmp/r17/areas`, `/tmp/r17/check-ev`), on the
intact tree `/tmp/rootsrc-6.40b`. Round 17's own full-suite run is **invalid and unquoted**: at
`--jobs 3` under a load average of 44 the `io/event` fixture collided with itself and 47 fixtures
failed (`W0-48`), while the same area passes 29/29 at `--jobs 3` when run alone; the column before it, `/tmp/r15/final2` (1230) and `/tmp/r15/final1` at
`601996d1`; the column before it, `/tmp/r14/final2` (1226) and
`/tmp/r14/final1` at `19be128b` (at `--jobs 5`), so the two columns are not strictly
comparable - the four CERN-network tests alone account for part of the difference. the column before it,
`/tmp/r13/final2` (1220) and `/tmp/r13/final1` at `a5d25f67`; the column before it,
`/tmp/r12/final2` (1195) and `/tmp/r12/final1` at `f7e42087`; **every earlier column was measured on
a damaged tree** (`W0-31`), so compare counts only within a column. The column before it is
`/tmp/r9/final2b` (1141) and `/tmp/r9/final1` at `2ef9c80`; the column before it, `/tmp/r8/final2` (1138) and `/tmp/r8/final1` at `f65e991`; the column before it, `/tmp/r7/final2` (1134) and `/tmp/r7/final1` at `2d0e458`; the column before it, `/tmp/r6/final2b` (1132) and `/tmp/r6/final1` at
`3825443`; the column before it, `/tmp/r5/final2b` (1123) and `/tmp/r5/final1` at
`f583422`; the column before it, `/tmp/r4/final2` (1114) and `/tmp/r4/final1` at
`932164a`; the column before it, `/tmp/r3/final2` (roottest, 1105) and
`/tmp/r3/final1` at `8af51aa`; the column before it, `/tmp/r2/final2` (roottest, 1089) and
`/tmp/r2/final1` (the other kinds) at `5c88c8c`; the round before it, `/tmp/r1/final2` (1078) and `/tmp/r1/final1`
at `7a43bd9`. The eleven roottest tests the last round brought in are
`cling/parsing/assertSpaces`, `cling/typedef/runptrptr`,
`io/alloc/customConstructor-build`, `io/alloc/customLoader-build`,
`tree/selectorreader/Collections`, `tree/selectorreader/Simple`, and five that pass
when the machine is quiet (`main/RootcpReadRemote` and its two dependants,
`io/hadd/test_hadd_regr_20872`, `io/hadd/test_MergeNestedSameName`); the round before,
six: `io/json/PolyMarker`, `io/rootcint/cltest`, `io/stdarray/arrTemplMemberWrite`,
`io/stdarray/colWiseWrite_3`, `meta/tclass/execTmpltD32` and
`treeformula/string/cernbuild`; the round before that, eighteen:
`collection/ROOT-7249`, `io/evolution/WriteFixedArray{New,Old}`,
`io/evolution/pragma_read/testv1`, `io/json/Map`, `io/stdarray/colWiseRead_11`,
`colWiseRead_12`, `colWiseRead_21`, `colWiseRead_22`, `colWiseWrite_1`,
`io/transient/execTransientNoctor`, `meta/method/ReuseMethod`,
`meta/runPublicDataMembers-auto`, `treeformula/array/filltree`, and four network
tests (`main/RootcpReadRemote` and its two dependants,
`io/hadd/test_hadd_regr_20872`) that pass when the CERN endpoints are quick.

**No PASS was lost.** Two tests time out when their CERN endpoint is slow -
`main/RootcpReadRemote` and `io/hadd/test_hadd_regr_20872` - and pass when rerun
alone, which has been checked every time it happened; `other`'s
`minuit2_ParallelTest` does the same under load, taking 195-236 s against the
300 s limit.

Rules for the ledger: a row moves to `done` only when its **verification
command** was run and the delta is in a `results.json`; quote the number
(`roottest 1038 → 1044`), never "improved".

---

## Appendix A — the 40 largest cause groups (from the assessment's runs)

`tests` counts tests whose *first* reason is this one; the last column is one
of them, to open (`RUN/logs/<kind>/<rest>/run.log`) and see the real thing.

| tests | status | cause | sample id |
|---|---|---|---|
| 86 | DIFF | `standard output differs from <path>` (needs W0-4) | `roottest/cling/const/run1` |
| 45 | FAIL | `'TClass' object has no attribute 'GetClassInfo'` | `roottest/cling/stl/map/stringMap` |
| 35 | FAIL | `(int) N` — the program's own diagnosis (needs W0-8b) | `other/math/mathcore/mathcore-testSpecFuncGamma` |
| 25 | FAIL | `FormulaError: 'GetNtrack' … not a function ROOT's formulas know` | `roottest/root/treeformula/event/draw-new-0-0` |
| 20 | UNSUP | `TGeoManager::Import reads a geometry from GDML or …` | `roottest/root/io/geo/Parse` |
| 17 | FAIL | `TypeError: int() argument must be …` (tree column typing) | `roottest/root/io/evolution/fixarr3` |
| 16 | FAIL | `ValueError: 'event' holds float64 values, and these are object` | `roottest/root/tree/fastcloningeventtree/reference` |
| 15 | FAIL | `TypeError: int() argument must be …` (no fixture) | `roottest/cling/specialobj/reread` |
| 13 | FAIL | `standard output differs` (cascade) | `roottest/cling/const/run2` |
| 12 | UNSUP | `TTree::BranchRef gives 'T' a TRefTable branch …` | `roottest/root/io/tree/test-r` |
| 10 | DIFF | `standard error output differs from <path>` | `roottest/cling/other/checkMissingSymbolExitCode` |
| 8+8 | UNSUP | `'TArrayI' object has no attribute 'fArray'` | `roottest/root/io/bigevent/{write,read}` |
| 7 | TIMEOUT | `still running after N s` | `other/math/minuit2/minuit2_ParallelTest` |
| 7 | UNSUP | `ROOT has Architecture` | `other/tmva/tmva/DNN/TMVA-DNN-Loss-Functions-Cpu` |
| 7 | FAIL | `'NoneType' object has no attribute 'ls'` | `roottest/root/io/double32/template32` |
| 7 | UNSUP | `ROOT has TEnum` | `roottest/root/io/evolution/enums/WriteVectorOfNonDefaultEnums` |
| 6 | FAIL | `gtest case <path> failed` (needs the case's own message) | `gtest/math/matrix/testMatrixT` |
| 6 | UNSUP | `ROOT has TMVA::PyMethodBase` | `other/tmva/pymva/PyMVA-RandomForest-Classification` |
| 6 | UNSUP | `ROOT has TMVA::Python_Executable` | `other/tmva/pymva/PyMVA-Torch-Classification` |
| 6 | UNSUP | `TMVAError: a RESHAPE layer …` | `other/tmva/tmva/DNN/TMVA-DNN-MethodDL-SGD-Optimization-Cpu` |
| 6 | UNSUP | `TTree::BranchRef … TRefTable` (no fixture) | `roottest/root/io/tree/create-r` |
| 5 | UNSUP | translator: `changing this in the middle of an expression` | `gtest/core/cont/coreconttestUnit` |
| 5 | UNSUP | translator: `an iterator … cannot follow back to its container` | `other/math/mathcore/mathcore-testTMath` |
| 5 | UNSUP | `ROOT has RooUnitTest` | `test/roofit/roofitcore/stressroofit-cpu` |
| 5 | UNSUP | `ROOT has std.complex` | `roottest/cling/bytecode/runcomplex` |
| 5 | UNSUP | `ROOT has vector` | `roottest/cling/stl/vector/Run` |
| 5 | FAIL | `ValueError: the column 'a' is of type 'object'` | `roottest/root/dataframe/regression_snapshot` |
| 5 | FAIL | `'TClass' has no attribute 'GetDict'` | `roottest/root/meta/fwdDecls/fwdDeclarations` |
| 5 | FAIL | `Parsing <path>` | `roottest/root/meta/genreflex/XMLParsing/0` |
| 5 | FAIL | `'int' object has no attribute …` | `roottest/root/tree/evolution/ReadClones` |
| 5 | FAIL | `TTree has MakeSelector, or does not` | `roottest/root/tree/selectorreader/Classes` |
| 5 | UNSUP | `ROOT has TTreeFormula` | `roottest/root/treeformula/array/NestedFail` |
| 4 | UNSUP | `rmath' has no attribute 'IMultiGenFunction'` | `other/math/genetic/genetic-testGAMinimizer` |
| 4 | UNSUP | `ROOT has std.atomic_int` | `gtest/net/net/testParallelMergingFile` |
| 4 | UNSUP | `ROOT has fgets` | `test/stressgraphics` |
| 4 | UNSUP | `ROOT has TCpu` | `other/tmva/tmva/DNN/TMVA-DNN-Activation-Functions-Cpu` |
| 4 | UNSUP | `ROOT has Detail` | `gtest/tree/dataframe/dataframe_interface` |
| 4 | FAIL | `.Class is not a command gROOT->ProcessLine runs here` | `roottest/cling/operator/runEqualTest` |
| 4 | FAIL | `'TClass' has no attribute 'GetMethod'` | `roottest/cling/parsing/execROOT6625` |

The full lists are in the runs' `results.json`; `summary.md` has the same
information ranked, per kind and per area, and `report` can redraw it from the
JSON at any time.

## Appendix B — where things are

| you want | file |
|---|---|
| the C++ translator | `src/xrdroot/pycxx/` (`parse_*`, `emit_*`, `instantiate.py`, `runtime/`) |
| ROOT's dialect of it, macros, `.x`, the prompt | `src/xrdroot/pycint/` (`execute.py`, `interactive/commands.py`, `dialect.py`) |
| the ROOT namespace (845 names) | `src/xrdroot/pyroot/__init__.py` + `SUBMODULES` |
| `TClass`, `TEnum`, dictionaries | `src/xrdroot/pyroot/core/reflection.py`, `declared.py` |
| `TTree`, branches, addresses | `src/xrdroot/pyroot/trees/` (`addresses.py`, `store.py`, `tree.py`) |
| ROOT file reading / writing | `src/xrdroot/file.py`, `interp.py`, `tree.py`; `writer.py`, `wtree.py`, `wpacking.py` |
| tree formulas / `Draw` | `src/xrdroot/formula/`, `treedraw.py` |
| histograms, profiles, fits | `src/xrdroot/hist.py`, `profile.py`, `fit/`, `function/` |
| RooFit / RooStats / HistFactory | `src/xrdroot/{roofit,roostats,histfactory}/` |
| RDataFrame (engine / adapter) | `src/xrdroot/rdf/`, `src/xrdroot/pyroot/rdf/` |
| canvas rasteriser, fonts, palettes | `src/xrdroot/canvas/`, `palette.py` |
| compression, errors, plugins | `compression.py`, `errors.py`, `plugins.py` |
| the harnesses | `tools/rootsuite/`, `tools/tutorials/`, their tests in `tests/test_rootsuite_*.py`, `tests/test_tutorial_*.py` |
| the component contract | `pyproject.toml` `[tool.xrdroot.components]`, `tools/components.py`, `docs/components/` |

## Appendix C — one-shot repros for the small bugs

Each of these was verified at handoff; if the message is different now, the bug
changed and the instruction is stale.

```bash
cd /Users/rcurrie/src/xrdroot && export PYTHONPATH=$PWD/src

# Q-3, Q-4, Q-5, Q-6, Q-9 are done (§7); the checks below now say what they should.
grep -n 'LEVELS = ' src/xrdroot/compression.py        # {"zlib": 1, ...}, ROOT's 101
/Users/rcurrie/src/xrdroot/.venv/bin/python -c \
 'import xrdroot.pyroot as R; h=R.TH1D("h","",3,0,3); print(hasattr(h,"SetContent"))'   # True
/Users/rcurrie/src/xrdroot/.venv/bin/python -c \
 'import xrdroot.pyroot as R; m=R.RNTupleModel.Create(); m.MakeField["int"]("f"); \
  print(hasattr(R.RNTupleWriter.Recreate(m,"n","x.root"),"__enter__"))'   # True

# W0-3: ratchet on a partial run
.venv/bin/python -m tools.rootsuite ratchet /tmp/xrun/suite2/results.json | tail -1
# "148 regressions" — all of them kinds that were simply not run
```
