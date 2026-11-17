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
| roottest | 1038 / 2173 (47.8 %) | **1043 / 2173** |
| gtest | 52 / 350 (14.9 %; 507 / 2020 cases) | 52 / 350 |
| pyunittest | 78 / 129 (60.5 %; 439 / 515 cases) | **80 / 129 (469 / 515 cases)** |
| `test/` | 4 / 59 | 4 / 59 |
| module tests (`other`) | 14 / 109 | **15 / 109** |
| **all non-tutorial** | **1185 / 2820 (42 %)** | **1192 / 2820** |
| tutorials | 558 / 1189 committed baseline (47 %) | unchanged (not re-run) |

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

Notes that will save days:

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
| W0-1 … W0-8 | todo | | | the harness tasks are untouched |
| Q-7, Q-10, Q-11, Q-13 … | todo | | | |

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
