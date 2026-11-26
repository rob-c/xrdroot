# xrdroot: continue the parity work with sixteen workers

This loop advances `/Users/rcurrie/src/xrdroot` — a pure-Python, NumPy-backed re-implementation
of CERN ROOT — one **round** at a time. A round is: read the queue in `TODO.md`, pick sixteen
disjoint scopes, fan sixteen workers out over their own git worktrees, integrate what passes the
gates, measure with ROOT's own test suite, and then **close the round in `TODO.md`** — the numbers,
a ledger row per fix and per attempt, and a queue entry for every piece of work the round
discovered and did not do.

**Every round runs in worktrees of its own**: made from `main`'s tip when the round opens, merged
back onto `main` before it closes, and removed at the close — no worktree and no round branch is
carried from one iteration of this loop to the next (steps 0 and 6d).

Read `AGENTS.md` and `TODO.md` in full before the first round. They are the specification; this
file is the procedure and the record of what has already been learned the hard way.

This is the project's **ultracode** mode: sixteen workers in parallel, one per git worktree, each
on a disjoint slice of the queue, integrated one commit at a time under the rule that no PASS may
be lost. It runs with **no iteration limit**: `0` is the value that means unlimited, since the
plugin stops only when `maxIterations > 0`. Start it with `/ralph start ROOT --max-iterations 0`,
or `ralph_start({ name: "ROOT", taskContent: <this file>, maxIterations: 0 })`, and continue it with
`/ralph resume ROOT` — a resume keeps the state file's `maxIterations`, so
`.ralph/ROOT.state.json` must say `0` for the loop to stay unlimited.

**This file is append-only.** Each round appends its own `## Round NN's close` section at the end
and one row of the round-log table, and nothing before it is rewritten, reordered or trimmed: the
rounds already here are the record of 86 rounds and the only account of what was tried and refused.
`/ralph start ROOT` and `/ralph resume ROOT` only ever *read* this file, and every `ralph_done`
re-reads it before the next iteration, so an appended close is what the next iteration reads — but
the **`ralph_start` tool writes its `taskContent` argument over the file**, so this loop is started
and resumed through the slash commands, never with an inline `taskContent`. The next round is the
one after the last `## Round NN's close` below; `iteration` in `.ralph/ROOT.state.json` is a counter
of prompts and has no relation to a round's number.

## Mission

Make xrdroot run ROOT 6.40's own test suite, in pure Python, with ROOT used only as a validation
oracle. "Done" is defined at the bottom of `AGENTS.md` and nothing here changes it: every test of
all five kinds PASS, or refused with a full-sentence reason for something genuinely out of scope
(GUI/Qt, SYCL/CUDA, cling internals, generated C++ dictionaries).

## Maximise the tests that pass — two counters, both rising

There are two numbers, and every round must move at least one of them without lowering the other:

| counter | where it comes from | how to read it |
|---|---|---|
| **harness PASS** | `python -m tools.rootsuite run --root $ROOTSRC --jobs 4 --xrdroot-src $PWD/src --out $R/run` | `jq '[.tests[]|select(.status=="PASS")]|length'` on `$R/run/results.json` |
| **unit tests** | `pytest -q -n 2 --cov=xrdroot tests` | the passed count in the last line |

`tools/rootsuite/baseline.json` is the floor for the first. It must stay honest: every PASS in it is
one a real run produced, the ratchet must print `0 regressions`, and a PASS that no longer reproduces
is recorded at its measured status with the reason in the commit message — never hand-kept, because a
hand-kept PASS fails the ratchet on every later run (`docs/decisions/bank-the-run-not-the-wish-*`).

### Rank the work by how many tests a cause blocks

Group the failures by *cause*, not by test, and count the tests behind each. A cause that blocks forty
tests at once beats any amount of polish worth one. The grouping command is in "1. Read the queue";
run it every round, because the ranking changes as causes are cleared.

### The levers, in the order they have paid

1. **A refused name is one test.** `UnsupportedFeatureError` on a class or method that ROOT's own test
   calls converts to PASS as soon as the name exists and behaves: `QuasiRandomSobol`,
   `QuasiRandomNiederreiter` and `RandomMT` turned `other/math/mathmore/mathmore-testQuasiRandom` from
   UNSUPPORTED into PASS in one commit. Read the refusal text in the `.ref`/log, implement exactly what
   the test calls, then re-run that one test id.
2. **A TIMEOUT is a PASS you cannot see.** The cap is 300 s and never rises, so a test at 268 s is a
   bug. `execmanydirs` was **162 s → 27 s** once `TFile::Write` stopped checkpointing once per
   subdirectory and stopped replaying every patch (`75478091`), and it had been passing only by luck.
   Profile the program at a *small* size, not the test: `python -m cProfile -o p.out -m xrdroot root -q -b
   'macro.C(20)'`, then `pstats` by `tottime`.
3. **Warning and silence parity is worth whole directories.** A missing `ROOT::TIOFeatures` warning broke
   `treeformula/stl/SparseSelection`; an extra `TVirtualIndex` warning broke `secondindex`; both were one
   line of judgement about which classes ROOT stays silent about (`75478091`).
4. **Formatting-only DIFFs are cheap PASSes.** A `.ref` difference that is spacing, a count word, or the
   order of two lines is a bug in the printer, not in the physics: fix the printer, re-run the id.
5. **A dictionary regression hides a class.** `CheckMissingDictionaries` read a class with no members
   because a lookup cache had settled before the dictionary loaded; the fix was to cache hits and never
   misses (`9b8b9f00`). Any cache over reflection must be invalidated by a load, which imports no module
   (`docs/architecture/a-loaded-dictionary-s-classes-are-in-declared-*`).
6. **Perf work that is not a timeout is still a PASS.** `n=200` of `execmanydirs` went 437 s → 27 s; the
   same shape (a quadratic over keys, a `sys.modules` scan per call) is behind several slow roottests.

### Never buy one PASS with another

- No PASS may be lost: run the ratchet on every harness run and name any id it reports.
- Never raise a limit, the 300 s timeout, or `fail_under`; never add `# pragma: no cover`; never loosen
  an assertion to make a test pass — find out whether the code or the test is wrong.
- A test that passes only because of the machine's order is not a PASS: `ClonesArrayEvo`,
  `forwardDeclAndFake` and `readGeo` differ inside a four-job run and pass alone, which is a harness
  defect still worth fixing.

### Measure each candidate before believing it

- Pin the tree under test with `--xrdroot-src`; the harness otherwise uses whatever the environment
  inherited, and a run against the wrong source looks like a regression.
- Re-run a TIMEOUT and any DIFF alone before calling it either way: `--jobs 1 --only '<id>'`.
- A roottest with a `DEPENDS` fixture is meaningless alone — its `-libgen-build` step creates the class it
  reads. Run the whole glob, or accept that the isolated answer means nothing
  (`docs/gotchas/an-isolated-roottest-failure-*`).
- Five tests need CERN's network (`root://eospublic.cern.ch`, a `TWebFile`): they are environmental and
  must never be counted as regressions.
- To bisect a harness failure, rebuild the fixture at each commit (`--only '<dir>/*'`), or build it once
  and run the macro directly when only the *reader* changes — 3 s per variant instead of 35
  (`docs/gotchas/a-bisect-over-harness-tests-lies-*`).

### Every worker's brief names its delta

A scope is only worth a worker if it can be written as: *this id, from this status, to PASS, by editing
these files, and the harness run that proves it is this command*. "Improve X" is not a scope; "move
`roottest/root/io/json/*` from DIFF to PASS by matching ROOT's key ordering in `writer.py`" is.

## Where it stands

Measured at `986f0d8` (`/tmp/x8/final2b` for roottest, `/tmp/x8/final1b` for the rest):

| kind | PASS / total | note |
|---|---|---|
| roottest | **1076 / 2173** | was 1038 at the assessment's `10e0e58` |
| gtest | 52 / 350 (512 / 2020 cases) | the weakest kind |
| pyunittest | 81 / 129 (470 / 515 cases) | |
| `test/` | 6 / 59 | |
| module tests (`other`) | 16 / 109 | |
| **all non-tutorial** | **1231 / 2820** | |
| tutorials | 558 / 1189 committed baseline | not re-run since; a `hist/` smoke run showed no lost PASS |

`TODO.md` §7 is the ledger: every gain and every attempt, with the commit and the `results.json`
it came from. §0's table and §4's queue are the plan. **Start each round from the ledger**, not
from `AGENTS.md`'s stale "what remains" list.

## Where it stood at `a2ef7d77`, the speed pass's measurement (2026-11-21)

Added rather than written over the table above, which is the record of the assessment at
`986f0d8` and stays as it was. Measured at `a2ef7d77` — each run against a **checkout of that
one commit** (`--xrdroot-src /private/tmp/wt-measure/src`), because this tree is committed to by
more than one session and a commit landing mid-run changes the answer for the tests after it:
the earlier attempt at this measurement had `math/*` and `machine_learning/*` failing for
`ROOT.TIterator`, which landed at `a2ef7d77` while that run was in progress. Five commits have
landed since that measurement; none of them was in it.

| kind | PASS / total | against the committed baseline |
|---|---|---|
| roottest | **1276 / 2173** | 1037 → 1276 (**+239**) |
| gtest | 53 / 350 (530 / 2109 cases) | 52 → 53 (+1); the weakest kind, as at the assessment |
| pyunittest | 81 / 129 (549 / 603 cases) | 78 → 81 (+3) |
| `test/` | 8 / 59 | 4 → 8 (+4) |
| module tests (`other`) | 38 / 109 | 14 → 38 (+24) |
| **all non-tutorial** | **1456 / 2820** | 1185 → 1456 (**+271**, 281 gained and 10 lost) |
| tutorials | **550 / 1189** | 564 → 550 (**−14**: 9 gained, 23 lost) |

`/tmp/r41/suite-fixed` and `/tmp/r41/tutorials-fixed`, both `-j 8`, load average ~49 from the
other session, Python 3.11.18 (the venv was replaced during the day; nothing measured across
that change is comparable, and every number here is an A/B inside one interpreter).

The ten the non-tutorial baseline had and this run did not are the tree's state rather than this
run's: the five I re-ran at `4dad0dad` (`--only` those five, same checkout trick) answer exactly
as they do at `a2ef7d77`. Five of the ten differ only by a line ROOT does not print, all of one
shape — `Warning in <TClass::Init>: no dictionary for class … is available` for `TArray`
(`fakeClass/forwardDeclAndFake`, `evolution/ClonesArrayEvo`), `TVirtualIndex` (`ClonesArrayEvo`,
`tree/addresses/ursula`, `datamodelevolution/stl/MapContent`) and `TGeoPatternFinder`
(`io/webfile/CloseTWebFile`) — which is the class-warning work of rounds 40–42 still in flight,
not a reading of the file. The other five: `distrdf_unit_test_friendinfo` refuses a friend tree
shorter than its parent, and `evolution/execBaseClass_v2` with its two `baseClass_53410_*`
dependents and `tree/friend/file2` are the long-standing ones this file's earlier rounds record.

The twenty-three the tutorial baseline had are twenty `DIFF`s, two `UNSUPPORTED`
(`ntpl012`/`ntpl015`, processor chains) and one `FAIL` (`math/vecops/vo002_VectorCalculations.C`).
Of the `DIFF`s, two are written files whose numbers differ in the last digits (`hsimple.root`'s
`hprof` errors in 2 of 102 places, `tree105.root`'s `vect` in 2626 of 70 000) — the committed
baselines were recorded on Python 3.14.10 and this venv is 3.11.18, where `sum()` of floats and
the libm wrappers this package routes through are not guaranteed the same last bit, so those two
are worth re-measuring on 3.14 before being called a regression; the other eighteen differ in
`stdout` — a `(TFile *) 0x…` echo ROOT does not print, canvas and legend lines, and
RooFit/RooStats `INFO`/`WARNING` order and values — and one of those, `visualisation/rcanvas/
rline.cxx`, also wrote no `line.png`.

Six non-tutorial tests and five tutorials timed out at the 300 s cap under that load and are
counted as not passing, as the harness counts them:
`other/math/mathcore/mathcore-stdsort`, `other/math/minuit2/minuit2_ParallelTest`,
`gtest/tree/ntuple/ntuple_largefile1`, `gtest/tree/tree/testBulkApiMultiple`,
`roottest/root/math/smatrix/Kalman`, `roottest/root/tree/cloning/runtreeCloneTest2`, and
`df101_h1Analysis.C`/`.py`, `df102_NanoAODDimuonAnalysis.py`, `df107_SingleTopAnalysis.py`,
`hist102_NanoAODDimuonAnalysis.C`. On an idle machine the first six do finish — they are slow,
not stuck: the two bulk-API tests write and read 10⁷ entries, `Kalman` runs four million matrix
iterations of `testKalman.cxx`, and `runtreeCloneTest2` clones a large tree; the tutorial
timeouts are the RDF analyses that read NanoAOD over the network.

## The round, in seven steps

### 0. Open the round in worktrees of its own, off `main`

**Every round makes fresh worktrees, and none outlives its round.** A worktree carried over from
an earlier round holds a stale base, that round's branches and half its files; anything measured
in it answers a question about a tree nobody has any more. So the round opens with a directory of
its own, one integration worktree based on `main`, and sixteen worker worktrees branching from
that same commit:

```bash
R=$HOME/.cache/xrdroot-runs/ralph-round-$(date +%Y%m%d%H%M)   # outside the repository, and
                                             # outside /tmp, which is wiped mid-session
ROUND=$(basename $R)                         # the round's name, in its branches
PY=/Users/rcurrie/src/xrdroot/.venv/bin/python   # the venv is the primary checkout's; the
                                                 # worktrees have none of their own
mkdir -p $R
git -C /Users/rcurrie/src/xrdroot fetch origin main
BASE=$(git -C /Users/rcurrie/src/xrdroot rev-parse origin/main)   # what `main` *is* now, not
                                                                  # what a checkout was left at
git -C /Users/rcurrie/src/xrdroot worktree add -q $R/main -B $ROUND-main main
for w in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  git -C /Users/rcurrie/src/xrdroot worktree add -q $R/w$w -B r-$w $BASE
done
cd $R/main && export PYTHONPATH=$PWD/src && git log --oneline -3 && git status --short
```

The **primary checkout is not touched** — do not switch its branch and do not commit in it: it
belongs to whichever session is working there, and it is usually mid-round. The round integrates
in `$R/main` and reaches `main` by pushing that worktree's branch (step 6d), so a round that dies
halfway cannot leave `main` looking as though its work had been measured.

`-B` and not `-b`: a branch name left over from an earlier round is reset onto this round's base
instead of inherited with that round's commits — and if it is still checked out somewhere, the
command fails loudly, which is the check that the previous round's teardown really ran.

`main` is the trunk. Measured at round 65 — `git rev-list --count origin/main..HEAD` and
`HEAD..origin/main` on the primary checkout both `0` — every commit of `feat/root-testsuite` is in
`main`, so a round based on `main` loses nothing. That is what `W0-138` objected to while `main`
was 238 commits behind and a round rightly refused to base on it; the objection is now spent.
**Check it again when the round opens.** If `origin/main` has fallen behind the line the project is
working on, stop and ask the operator rather than quietly basing a round somewhere else.

A worktree gets a branch of the round's own, never one another worktree has: `git worktree add
/path feat/root-testsuite` hands the same branch to two worktrees, and it cost twenty minutes when
a worker's commits landed as a staged revert (`W0-153`). `r-NN` and `$ROUND-main` are the round's
names; every other branch is read-only here.

### 1. Read the queue and rank the work

```bash
cd $R/main && export PYTHONPATH=$PWD/src      # the round's own worktree, which is `main`'s tip
git log --oneline -6; git status --short      # must be clean
sed -n '/## 7. Progress ledger/,/## Appendix A/p' TODO.md
```

Then rank the *current* failures by cause, from the last run's JSON rather than from memory:

```bash
python3 - <<'EOF'
import json, re, collections
tests = [t for f in ("/tmp/x8/final2b/results.json", "/tmp/x8/final1b/results.json")
         for t in json.load(open(f))["tests"]]
c, sample = collections.Counter(), {}
for t in tests:
    if t["status"] == "PASS":
        continue
    r = re.sub(r"^A fixture it requires was not set up: (?:exit -?\d+, expected -?\d+: )?", "",
               t.get("reason") or "")
    r = re.sub(r"/private/[^\s:]+|/tmp/[^\s:]+", "<path>", re.sub(r"\b\d+\b", "N", r))
    c[r[:95]] += 1
    sample.setdefault(r[:95], t["id"])
for k, n in c.most_common(25):
    print(f"{n:4d}  {k}\n        e.g. {sample[k]}")
EOF
```

Pick the sixteen largest groups that a worker could plausibly *finish* in forty minutes, and make
sure no two of them own the same file (see "Choosing the sixteen scopes" below).

### 2. Choose the sixteen scopes

A scope is: a target test (or glob), the files one worker may edit, and the reason it fails today.
Give each worker **one test group and a disjoint file set**; workers cannot see each other's
edits, so two workers in one file means a cherry-pick conflict later.

Good scopes, in the order the last ranking suggests (target → files):

| target | files |
|---|---|
| `roottest/root/treeformula/event/draw-new-*` (25) | `src/xrdroot/formula/`, `src/xrdroot/treedraw.py` — **branch `w01`/`xc` already holds 90 % of this**; a worker may `git cherry-pick w01` in its own worktree |
| `roottest/root/io/filemerger/*` (21 file sizes) | `src/xrdroot/writer.py`, `src/xrdroot/wpacking.py`, `src/xrdroot/wtree.py` — the input `hsimple.root` that xrdroot writes is 412 079 bytes where ROOT's is 414 892; find where |
| `roottest/root/io/stdpair/pairEnumEvo*` (40) | `src/xrdroot/pyroot/core/reflection.py` — needs `GetStreamerInfos`, `HasInterpreterInfo` (done), `GetListOfEnums`; then real streamer evolution |
| `roottest/root/io/tree/create-r`, `test-r` (18) | `src/xrdroot/wbranch.py`, `src/xrdroot/writer.py` — `TTree::BranchRef`/`TRefTable` |
| `roottest/root/tree/fastcloningeventtree/*` (20) | `src/xrdroot/wtree.py`, `src/xrdroot/pyroot/trees/store.py` — writing a column whose dtype is object |
| `roottest/root/io/stdarray/colWiseWrite_4`, `colWiseRead_3*`/`4*` (5) | `src/xrdroot/wbranch.py`, `src/xrdroot/pyroot/trees/objects.py` — a split branch of `std::array<T,N>` |
| `roottest/cling/operator/runEqualTest` and `.Class`/`.typedef`/`.autodict` (9) | `src/xrdroot/pycint/interactive/commands.py` |
| `roottest/root/meta/runTEnum-auto`, `meta/enums/*` (9) | `src/xrdroot/pyroot/core/reflection.py`, `troot.py` |
| `test/stressgraphics*` (3, now at `TPostScript`) | `src/xrdroot/pyroot/graphics/` — `TPostScript`/`TSVG` |
| `roottest/root/io/json/{String,STL,STL0,STL1,ArrayCompress}` (5) | `src/xrdroot/pyroot/core/jsonbuffer.py` |
| `gtest` FileRaii family (17 programs) | `src/xrdroot/pycint/runtime/`, `src/xrdroot/pycxx/` (G-2 translator gaps) |
| `other/math/*` TMath names (5) | `src/xrdroot/pyroot/core/rmath.py`, `tmath.py` |
| `roottest/root/io/geo/*` (20, blocked) | `src/xrdroot/geom/` — needs a GDML reader and a `TGeoManager` file reader |
| `roottest/root/io/double32/*` (7) | `src/xrdroot/pyroot/core/reflection.py` (needs `GetStreamerInfo`) |
| `roottest/root/tree/selectorreader/*` (5) | `src/xrdroot/pyroot/trees/` (`TSelector`, `MakeSelector`) |
| `roottest/root/io/complex/*` (2) | `src/xrdroot/pyroot/core/files.py` — `WriteObjectAny` ignores its `classname` |
| a `hist/`+`roofit/` tutorial area whose stdout DIFFs are formatting, not physics | the tutorial harness's `details` field names the first differing line |

Rebalance freely: the rule is sixteen scopes, disjoint files, each with a *measured* before.

### 3. Fan the sixteen out

Two ways, in this order:

**(a) The `subagent` tool, if it is in your tool list.** Then use it directly — one `worker` child
per scope, `async: true`, each with its worktree as `cwd` and the brief as its task.

**(b) Otherwise, `pi -p`** — the same harness, headless, one process per worktree. This is what
the last two rounds used. It works; do not use any other agent CLI.

The worktrees themselves were made in step 0; this step writes the briefs and launches one agent
in each.

Write one brief per worker into `$R/w<NN>-brief.md`. It must carry: the mission and the hard rules
(below), the environment, the target test id, the files it **may not leave**, the reason the test
fails today, the exact harness command to reproduce it, the timebox, and the report it must end
with. A brief that worked is worth copying:

```markdown
# Task r-NN: <one line>

You are one of sixteen workers on xrdroot, a pure-Python re-implementation of CERN's ROOT.
Read /Users/rcurrie/src/xrdroot/AGENTS.md first, then TODO.md. The rules that bind you:
 - Never widen a refusal to hide a failure, never raise a limit, never delete or loosen an
   assertion. Decide whether the code or the test is wrong and say which in the commit.
 - Refusals are full sentences naming what is unsupported and what to do instead
   (`raise UnsupportedFeatureError("...")`).
 - Match the surrounding idiom; docstrings say why.
 - Keep every gate green: ruff, mypy (strict), `pytest -q --cov=xrdroot` (100 % line AND
   branch), tools/maintainability.py check, tools/components.py check.
 - A lower component never imports a higher one.

Environment:
  cd <your worktree>
  export PYTHONPATH=$PWD/src
  PY=/Users/rcurrie/src/xrdroot/.venv/bin/python
  ROOT's sources: /tmp/rootsrc-6.40     oracle: ~/.local/root-oracle/bin/root -l -b -q

Measure, before and after:
  $PY -m tools.rootsuite run --root /tmp/rootsrc-6.40 --jobs 2 --out <R>/run-rNN --only '<id>'
Read <R>/run-rNN/results.json and <R>/run-rNN/logs/<kind>/.../run.log.

Deliver: the smallest change that makes the target pass, unit tests for it (the coverage gate),
a commit in your worktree (`git -c commit.gpgsign=false commit -am "feat: ..."`), and a report
giving the status and reason text before and after, plus what you found that you did not fix.

Do not touch: TODO.md, docs/, the two baseline.json files, pyproject.toml's component lists,
<any file outside your scope>. Do not push, rebase, or enter another worker's worktree.
If the fix needs a file outside your scope, STOP, leave the tree clean and report the patch —
two of the best fixes so far came from exactly that.
Timebox: about forty minutes and forty tool calls.
```

Launch them staggered (four seconds apart: sixteen `pi` processes contending on the same npm
update check lose each other), each detached with stdin from `/dev/null`:

```bash
for w in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  ( cd $R/w$w && PYTHONPATH=$R/w$w/src nohup pi -p "$(cat $R/w$w-brief.md)" \
      > $R/w$w.log 2>&1 < /dev/null & )
  sleep 4
done
```

Watch progress with the session files, not the logs (a `pi -p` log stays empty until the worker
finishes):

```bash
for w in $(seq -w 1 16); do                  # $BASE is step 0's, the round's base
  f=$(ls -t ~/.pi/agent/sessions/*/2026-*.jsonl 2>/dev/null \
      | xargs grep -l "Task r-$w:" 2>/dev/null | head -1)
  printf "w%s commits=%s dirty=%s events=%s\n" "$w" \
    "$(git -C $R/w$w rev-list --count HEAD ^$BASE)" \
    "$(git -C $R/w$w status --porcelain | wc -l)" "$(wc -l < "$f" 2>/dev/null)"
done
```

A worker's **working tree** can be ahead of its commit — two rounds lost the final state of the
best implementation that way. When a worker committed early and kept working, copy its *files*
(`cp $R/wNN/<path> $R/main/<path>`), not just its commits.

### 4. Integrate, and let no PASS be lost

All of this happens in `$R/main` — the round's own worktree on `main`, with `PYTHONPATH=$PWD/src`
there — never in the primary checkout. For each worker with a commit, in order of expected value:

```bash
git cherry-pick -x <commit>            # -x records whose work it is
export PYTHONPATH=$PWD/src
$PY -m pytest -q -p no:cacheprovider -n 4 --cov=xrdroot      # must be 100 % line+branch
ruff check src tests tools && mypy && $PY tools/maintainability.py check \
  && $PY tools/components.py check
$PY -m tools.rootsuite run --root /tmp/rootsrc-6.40 --jobs 2 --out /tmp/x/check-wNN \
    --only '<the worker's target>'
```

Rules for integrating:

- **A cherry-pick that conflicts** (two workers appended to one test file, or both edited one
  function): do not force it. Take the *code* commit with `git cherry-pick -n`, restore the
  conflicted test file from `HEAD`, re-add that worker's new test by hand, and commit with a
  message that says so.
- **A worker whose branch loses a PASS is not merged.** Check it in *its own* worktree first
  (`--only 'roottest/root/treeformula/*'`, compare to the last run's `results.json`); if it turns
  a PASS into a refusal, leave the branch for a later round and record the exact refusal to
  relax. This has already happened twice with the formula engine (`formula/binding.py::_mismatch`
  refuses `fTracks.fVertex[][fTracks.fNpoint%3]`, which the older code handled).
- **A worker who reports a patch instead of a commit** — because the fix lay outside its scope —
  is a gift: apply the patch, run the gates, and credit the worker in the commit message.
- Keep one commit per fix group, message style `feat:`/`fix:`/`perf:`/`tests:`/`docs:`, saying what
  ROOT does and what the test now shows.

### 5. Measure, record, commit

Two runs, in the background, with their output in the round directory:

```bash
$PY -m tools.rootsuite run --root /tmp/rootsrc-6.40 --jobs 5 --out $R/final1 \
    --kind gtest --kind pyunittest --kind test --kind other
$PY -m tools.rootsuite run --root /tmp/rootsrc-6.40 --jobs 5 --out $R/final2 --kind roottest
```

Then read them against the previous run and *name* every move:

```bash
python3 - <<'EOF'
import json
a = {t["id"]: t for t in json.load(open("<this round>/final2/results.json"))["tests"]}
b = {t["id"]: t for t in json.load(open("<last round>/final2/results.json"))["tests"]}
gained = sorted(k for k in a if b.get(k, {}).get("status") not in (None, "PASS") and a[k]["status"] == "PASS")
lost = [(k, a[k]["status"], (a[k].get("reason") or "")[:70]) for k in a
        if b.get(k, {}).get("status") == "PASS" and a[k]["status"] != "PASS"]
print("gained", len(gained), gained)
print("lost", len(lost), lost)
EOF
```

Rerun any TIMEOUT alone (`--jobs 2`, `--only '<id>'`) before calling it a loss: three tests are
CERN-network copies (`main/RootcpReadRemote` and its two dependants, `io/hadd/test_hadd_regr_20872`)
and `other/math/minuit2/minuit2_ParallelTest` takes 195–236 s of a 300 s limit; all four have
passed when rerun alone, repeatedly.

Then close the round with step 6: the gates are run before the commit, never after, and the round's
commits reach `main` in step 6d — a merge that is only local is a merge the next session's
`git reset --hard origin/main` throws away.

### 6. Close the round in `TODO.md` — every round ends here

A round is not finished until `TODO.md` carries it, in the **same commit as the improvement** it
records. Three things go in, and the third is the one that is easy to forget:

**(a) The measured numbers.** §0's "now" column (per kind), and a new column in §7's cumulative
table for this round, quoting the round's `final1`/`final2` `results.json` by path. Quote
`roottest 1076 → 1084`; never "improved".

**(b) One ledger row per fix group and per attempt.** For a merged fix: task id, commit, tests
moved, and a note saying what ROOT does. For an attempt that is *not* merged: the same row with
`attempted, not merged`, the branch it is on, `0` tests moved, and the exact refusal or missing
name to relax — `formula/binding.py::_mismatch` refuses `fTracks.fVertex[][fTracks.fNpoint%3]` is
the shape of a useful row.

**(c) The extra work the round discovered — as queue entries, not as prose.** Every one of these
becomes a row in §4's queue (or, where it is small, a bullet in §5's housekeeping), with the file,
the observed message and one line of why:

- a worker's "the fix is one line, in a file outside my scope" (that is the highest-value finding
  a round can produce);
- a refusal that a fix turned from a crash into a sentence — the next worker needs to know it is
  now a *named* gap;
- a missing ROOT name, class or method a test asked for and nothing in the queue listed;
- a test that needs an environment package (`jupyter`, `torch`, `libomp`) rather than code;
- a hang or a test near the 300 s limit, with the stack that shows where it spends its time;
- a file layout measured against the oracle (a byte count, a streamer element, a leaf title) that
  differs and is not yet fixed;
- any tool or harness gap found (a wrong test id in a brief, `ratchet` misfiring, a stale
  `baseline.json`), because the next round will otherwise pay for it again.

Write each as a row someone can act on without the round's context:

```markdown
| Q-NN | <the name of the thing> | <tests blocked> | S | <file(s)> | `<message or measurement>` — <what ROOT does, what is missing>. <the first step> |
```

and keep the numbering of the existing queue (extend `Q-`, `P-`, `G-`, `W0-`), so the next round's
ranking lands on the same ids. If a discovered item is one the round's own workers were told to
do, mark the *original* row's status instead of adding a new one.

Finish with:

```bash
cd $R/main && PYTHONPATH=$PWD/src $PY -m pytest -q -p no:cacheprovider -n 4 --cov=xrdroot
git -c commit.gpgsign=false commit -am "docs: the ledger after round <N>"
```

and put the round's own summary — the sixteen scopes, what landed, what was rejected, what is
new — in the loop's task file (`.ralph/<name>.md`), which is where the next iteration reads it.

**(d) Merge onto `main`, then tear the round down — no worktree, no round branch, no
`$ROUND-main`.** This is the last thing a round does, as four acts in this order; the round is not
closed until the third is verified, and cleanup is the fourth:

1. **Merge** — the workers' commits are cherry-picked onto `$R/main`, whose branch was cut from
   `main` at step 0, so `$ROUND-main` *is* `main` plus the round's commits;
2. **Push** — `git push origin $ROUND-main:main` (rebasing onto `origin/main` first if another
   session pushed);
3. **Verify** — `git ls-remote origin main` is the round's HEAD, which is the only proof that the
   fixes and the ledger row are on the trunk;
4. **Tear down** — only now, because a worktree removed before the push, with `main` moved under
   the round, would take the round's only remaining reference to its commits into a directory that
   no longer exists.

```bash
cd $R/main && git status --short         # must be empty: nothing is left uncommitted
git push origin $ROUND-main:main || { git fetch origin main && git rebase origin/main \
  && git push origin $ROUND-main:main; }  # another session pushed: rebase the round onto it
git ls-remote origin main | cut -f1      # the tip, which must be this worktree's HEAD

cd /Users/rcurrie/src/xrdroot
git worktree list --porcelain | awk '/^worktree /{print $2}' | grep "^$R/"   # what will go
for w in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  git worktree remove --force $R/w$w && { git branch -D r-$w 2>/dev/null || true; }
done
git worktree remove --force $R/main && git branch -D $ROUND-main
git worktree prune
git worktree list | grep "^$R/" || echo "every round worktree is gone"
```

A branch the round decided **not** to merge is no exception: `git format-patch $BASE..r-NN -o
$R/rejected/` writes it into the round's directory as the evidence it now is, the ledger row
names the refusal it hit, and the branch goes with the worktree. A branch is a worktree's state;
a patch is evidence.

Keep the rest of the round directory: `final1/`, `final2/` and any `areas/`/`solo/` run the
ledger quotes, the briefs, and the workers' reports. The worktrees and their logs are what goes —
`rm -rf $R/w*.log` once their findings are in `TODO.md`.

## Hard rules (from `AGENTS.md`, do not violate)

1. **Never raise a limit** — not the 300 s harness timeout, not CCN ≤ 10 / cognitive ≤ 15 /
   NPath ≤ 200 / Halstead ≤ 500 / nesting ≤ 4, not the 100 % coverage floor. Split the function.
2. **No `# pragma: no cover` to dodge a branch**, and **never delete or loosen an assertion** to
   make a test pass.
3. **A PASS is a promise.** The two `baseline.json` files change only in the same commit as the
   improvement, only from a real run (`python -m tools.rootsuite baseline RUN/results.json`).
4. **Bit-exact only on ROOT's machine** (`tests/refmachine.ROOTS_MACHINE`); elsewhere a tolerance
   with a comment saying which numerics (glibc vs Apple libm, FMA, `sum()` before 3.12).
5. **Refusals are full sentences** naming the class and the alternative.
6. **Small commits, one fix group.**
7. **No process machinery** — no receipt/audit/ledger files beyond `TODO.md`'s own ledger. The
   evidence is `results.json`.
8. **Long jobs in the background, output outside the repository.**
9. **Optional packages are extras**, imported lazily, refused with the `pip install` that fixes
   them.
10. **Component boundaries hold**; a new module is owned by a component in `pyproject.toml`, and a
    new `pyroot` module also goes in `SUBMODULES`.
11. **A new worktree every round, off `main`, merged to `main` and torn down in that same round**
    (steps 0 and 6d), on disjoint files, `PYTHONPATH=$PWD/src`. The order is merge, push, verify
    (`git ls-remote origin main` is the round's HEAD), then clean up — a round whose push is not
    verified is not closed. The round's integration worktree is the one that pushes `main`; nothing
    else is pushed without being asked. Never edit another worker's worktree, never reuse one
    round's worktree in the next, and never give two worktrees the same branch.

## Traps already paid for

- `setsid` does not exist on macOS; `nohup … < /dev/null &` inside a subshell is what survives.
- Sixteen `pi` processes started at once contend on `pi`'s update check and some exit silently;
  stagger them, and check `pgrep -fl cli.js` plus each worktree's `git log`.
- `pi -p` writes nothing to its log until it finishes; watch the JSONL session files.
- A worker's commit can lag its working tree — copy the files as well as the commits.
- Two workers appending tests to one file (`tests/test_root_declared_objects.py` twice now) always
  conflicts; expect to re-add a test by hand.
- **Global state in tests**: a test that calls `set_type_resolver(None)` wipes the resolver pyroot
  installed for every later test *in the same xdist worker*. Save and restore it (`e9b5a5f`).
- A brief that names a test which does not exist wastes a worker: `roottest/root/tree/tree/Print`
  is not a 6.40 test. Check the id with
  `$PY -m tools.rootsuite list --root /tmp/rootsrc-6.40 | grep <name>` first.
- The 300 s timeout is a hard wall: never raise it, and measure any test near it alone.
- **Never start a measurement while a conflicted file is on disk.** A `git apply -3` that leaves
  `<<<<<<<` markers in `reflection.py` and a harness run started beside it made 114 pyunittest
  files fail with `SyntaxError` and read like a catastrophe; check `git status` for `UU` first.
- Tests in one directory share a working copy: `io/hadd`'s tests write `src1.root`/`dest.root`, so
  at `--jobs 5` they collide and a PASS turns into a DIFF that passes alone (`W0-9`).
- A harness run's PASS count is only comparable at the same `--jobs`: 300 s tests and
  same-directory collisions both move with the parallelism.
- **Another session resets `main` to `origin/main`** (`git reflog show main` says `branch: Reset
  to origin/main`), which discards anything committed to local `main` and not pushed, within
  minutes. So the round pushes `main` itself (step 6d) and then checks `git ls-remote origin
  main`; if the round's commits are missing afterwards they are still in `$R/main`'s log —
  re-apply them there before the worktree goes.

## Round 1's close (recorded here for the next iteration)

- Base `8c1b9f8`, closed at `a4ebcd0`. Sixteen workers, sixteen commits, **seven** roottest tests
  in and **none** lost (`/tmp/r1/final2` vs `/tmp/x8/final2b`).
- Two branches were reverted for losing a PASS: `TClass::GetStreamerInfo`'s printer
  (`5e9076e` reverts `e0ab124`) and the template choice (`7a43bd9` reverts `81b9f75`). Both are
  queue entries now — `Q-23` and `G-5` — with the exact text ROOT prints and the exact check
  that is missing.
- Two patches a round could not take, and why: a member template constructor (`G-6`, it regressed
  a null-pointer member test) and the streamer-info printer (it conflicted with the work above).
- Four workers found their fix outside their own scope and reported a patch instead
  (`w07` genreflex, `w12` the streamer base name, `w13` the null pointer, `w15` the reference) —
  that is the highest-value outcome a brief can ask for, and all four were applied here.
- The next round's obvious first scope: `Q-23` (one worker, `reflection.py`), because it is what
  makes `pairEnumEvo`'s 40 tests comparable; then `P-7`/`P-8` (seven hangs), `G-5`, `G-6`, and
  the two big queues (`Q-20`'s writer bytes, `Q-11`'s formula).

## Round 2's close (recorded here for the next iteration)

- Base `773a63c`, closed at `450231d`. Sixteen workers, twelve cherry-picked commits plus one patch
  applied, **eleven** roottest tests in and **no PASS lost** (`/tmp/r2/final2` vs `/tmp/r1/final2`).
- The round's biggest step is not a PASS: the streamer-info printer (`43686bf`) took **all 40
  `pairEnumEvo` tests from a crash to a DIFF**, and 28 formula tests from a refusal to a failure
  that names its next gap. That is what `Q-24` and the formula rows now carry.
- One regression: `.typedef`/`.autodict` (`27791f5`) turns seven `tree/stl` DIFFs into six failures
  and one refusal, for two PASSes the other way. It is `W0-10`, with the revert that isolates it
  recorded there. Decide next round: find what a prompt command changes about a `.L` load, or
  revert it and lose the two.
- Three workers produced only a diagnosis and a patch (the `CloneTree` column typing, `TBranchRef`'s
  streamer slot, the split `TClonesArray`). Their notes are `Q-25`, `Q-26`, `Q-27`.
- The next round's first scopes: `Q-24` (pairEnumEvo's listing), `W0-10`, `Q-25`, `Q-26`, then the
  writer's bytes (`Q-20`) and the formula (`Q-11`).

## Round 3's close (recorded here for the next iteration)

- Base `bb19779`, closed at `b5fe478`. Sixteen workers, **ten** commits landed plus two reported
  patches applied (worker 11's derived-class pointer and worker 1's `methods.py`), **no PASS lost**
  (`/tmp/r3/final2` vs `/tmp/r2/final2`: 16 in, 0 out; `/tmp/r3/final1`: 6 in `other`, 0 out).
- Two results worth more than a PASS:
  - **Q-20 is now arithmetic**: the 2 811-byte gap in `hsimple.root` is accounted for byte by byte
    (file title 120, twelve missing streamer infos 2 268, `TPaveStats` 346, a `float`-vs-`double`
    `pz` 48 + 11, tree/branch versions 18). Eleven `io/filemerger` FAILs *are* those bytes. Worker 8
    also found the `pz` difference is the **translator**'s, not the writer's: 3 923 of 25 000 entries
    differ in the last bit because `px*px+py*py` is computed in `double` where C++ uses `float`.
  - **Two of the nine timeouts were deadlocks** (worker 14's diagnosis, P-7). `std::atomic::exchange`
    and a `_Whole` that answers with the object it holds ended both: one test now fails on its own
    subject, the other completes a run in 53 s. What is left of the nine is five genuinely O(n)
    tests whose hot functions worker 14 named (`P-9`).
- **`W0-10` is closed as a decision, not a fix.** Worker 16 measured that the `.typedef`/`.autodict`
  commit does turn seven `tree/stl` DIFFs into failures but also gives two PASSes, and that the cause
  is the *command*, not `remember_source`: 11 PASS + 8 FAIL beats 9 PASS + 7 DIFF by the only measure
  that counts. The row now says so with the numbers.
- Four workers stopped at a file outside their scope and left the diagnosis the next round needs:
  worker 12's `TObjArray` writer patch (rejected here — it omits the `TNamed` title, so the bytes are
  wrong; ROOT's own 88-byte stream is kept at `/tmp/r3/oracle-objarray/oa.bin`), worker 3's six-file
  map of splitting a `TClonesArray`, worker 10's four pycxx/`_Set` gaps, worker 6's misread of an
  evolved class (`Q-36`) and worker 14's timing table (`P-9`).
- The next round's first scopes: `Q-20`'s first three rows (2.7 kB of `writer.py` + `winfo.py` unlocks
  eleven PASSes at once), `Q-28` (`RootClassVersion`, worth 5 + a written-file layout), `Q-10`
  (`W0-10`'s seven DIFFs), `Q-24`/`Q-30` (40 tests), and `P-9`'s memoised class description.

## Round 4's close (recorded here for the next iteration)

- Base `9cf98a5`, closed at `16eb8a7`. Sixteen workers, **six** commits landed plus **two** reported
  patches applied by hand (worker 13's cling-preprocessor fix, worker 2's array-shape fix), **nine**
  roottest tests in and **no PASS lost** (`/tmp/r4/final2` vs `/tmp/r3/final2`; `other` 21 either way).
- The round's pattern is worth reading before choosing the next scopes: **five of sixteen workers
  found their fix outside the files their brief named** and reported a diagnosis and a patch instead
  of a commit. That is the briefs being wrong, not the workers - `Q-38`, `Q-39`, `Q-40`, `Q-41` and
  `Q-42` are all "one small change, in a file the brief excluded". Next round: give those five the
  files they asked for, one worker each, and nothing else.
- The best find is a **bug**: `interp.Each` is a `Values`, so `copying._whole`'s "one streamed object
  per entry?" answers yes for a split `TNamed` base's `fName`/`fTitle` columns. Worker 4's 64-line
  prototype (`/tmp/r4/w04-prototype-patch.diff`) makes 29 clone tests pass and reaches the writer's
  `SplitClones`. It is `Q-38`.
- The writer's missing bytes (`Q-20`) lost two of its three rows: the file title is written where
  ROOT writes it (`bcc03ba`) and ten more streamer layouts are emitted, which is what
  `io/newClassDef/noclassdef/driver-build` and `io/stdarray/modelCheckValues` were really waiting
  for. What is left is the `TPaveStats` 346 B and a `float`/`double` `pz` 59 B.
- Worker 14's `pack_member` is 2× faster (`f113afb`) and the measurement says the rest of the
  825 s / 2 430 s is the per-record description rebuild - `P-9`'s own item, now with a number.
- The next round's first scopes: `Q-38` (a bug, 40 tests behind it), `Q-42`'s seven formula walls
  (40 tests, and the file list is known), `Q-39` (`TString*`, 16), `Q-20`'s last two rows (11),
  `Q-40` (9), and `P-9`'s per-record rebuild (4 timeouts).

## Round 5's close (recorded here for the next iteration)

- Base `7e54e3c`, closed at `edbfdfa`. Nine commits landed plus worker 13's and worker 2's verified
  patches applied by hand, **nine** roottest tests in and **no PASS lost** - and one lost in the
  first measurement and recovered by `f583422` before the round closed.
- **P-9's first half is the round's best result**: a declared class is described once instead of
  once per record (`6ab5b1b`), and `io/perf/WriteBuffer-1`/`-2` - TIMEOUTs in every round before
  this one - now pass inside the limit. `ReadBuffer-1`/`-2` still time out, so the read direction
  rebuilds its layout per record; that is the rest of `P-9` and it is one worker's job.
- The round's lesson, again: **six of sixteen workers found their fix outside the files their brief
  named** - `Q-40` (worker 10), `Q-42` (worker 1), `Q-14` (worker 4), `Q-36` (worker 11), `Q-20`
  (worker 6), `W0-16` (worker 14) - each with a patch in `/tmp/r5/wNN.log`. Those patches were
  measured and not merged, and they are the shortest path to the next round's numbers.
- The one PASS lost and recovered is worth reading: the `__CLING__` dialect change (`cef312f`) broke
  `cling/other/runfileClose` twice over - a header taking `&x` in `x`'s own initialiser, and
  `gSystem->Load("fileClose_C")` with no library to find. Both are general fixes now (`f583422`):
  a cell is made before it is filled, and a load of a name ACLiC gave a library builds its macro.
- The next round's first scopes: `Q-45` (42 formula tests, one wall left in the way), `P-9`'s read
  direction (2 timeouts), the six unmerged patches above (`Q-20`'s last 405 B, `Q-40`, `Q-36`,
  `W0-16`), `Q-38`'s `SplitClones` half (12), `Q-44`'s GDML wiring (21) and `Q-46`.

## Round 6's close (recorded here for the next iteration)

- Base `a3a3d51`, closed at `5998baf`. Eleven commits landed, **nine** roottest tests in and **no PASS
  lost** (`/tmp/r6/final2b` vs `/tmp/r5/final2b`).
- **`P-9` is finished.** Round 5 did the write direction, round 6 the read one
  (`267e974`: a declared class is read by a plan made once), and all four `io/perf` tests that had
  timed out in *every* round now pass - `ReadBuffer-1` in 221.6 s of the 300 s limit.
- The round's lesson is about the *revert*: worker 11's `BuildCheck` warning made `AddVersion-1-6`
  pass and cost two PASSes their `.ref` files do not have. Rather than ship it with an exception,
  the gate was narrowed to the member's type name (which does fix the two - measured in
  `/tmp/r6/check-lost5`), found to leave its own target one row short, and reverted with the
  measurement recorded as `Q-29` and the underlying old-code parse gap as `Q-49`. The next attempt
  starts from a known-good shape instead of from a guess.
- `Q-44`'s GDML is one test from twenty: `Parse` passes byte-identically, and `Export` plus a
  `TGeoManager` file reader are what the other twenty want. `Q-46`'s `MakeProject` is two of four.
- The next round's first scopes: `Q-48` (42 tests, one wall), `Q-29`+`Q-49` (the warning, now shaped),
  `Q-44`'s `Export` (20 tests), `Q-38`'s `SplitClones` half (12), `Q-30`'s `offset=na` listing, and
  `Q-39`'s template spelling.

## Round 7's close (recorded here for the next iteration)

- Base `ebf4262`, closed at `f816485`. Six commits landed (one a two-commit worker), **two** roottest
  tests in and **no PASS lost** (`/tmp/r7/final2` vs `/tmp/r6/final2b`).
- The two are cheap and worth having: `AddVersion-1-6` came in because round 6 had already measured
  the exact shape the warning needs (judge by width, normalise the old element code), and
  `io/geo/readGeo` because round 6's GDML reader gave the geometry something to be written back to.
  **A measurement recorded one round becomes a PASS the next** - that is the loop working as intended.
- **Eleven of sixteen workers found their fix outside the files their brief named**, each with a
  measurement and a patch path: `Q-48` (worker 1), `Q-38` (worker 4), `Q-39` (worker 5), `Q-40`
  (worker 6), `Q-36` (worker 7), `Q-46`'s read (worker 14), and `W0-21`…`W0-25`. The briefs are
  still systematically one file short: the pattern is that the file a behaviour *lives* in is one
  component above the file the test's symptom names.
- What that means for round 8: build the sixteen briefs **from the eleven reports' file lists**
  (`/tmp/r7/wNN.log`), not from the test ids. Each report names the file it wanted, which is exactly
  the disjoint scope the next round needs. The three biggest are `Q-48`'s
  `pyroot/trees/methods.py`, `Q-38`'s three-file `SplitClones`, and `W0-25`'s `jsonbuffer.py`.

## Round 8's close (recorded here for the next iteration)

- Base `3f1cb63`, closed at `fdb58c4`. Eleven commits landed (one reverted), **four** roottest tests
  in and **no PASS lost** (`/tmp/r8/final2` vs `/tmp/r7/final2`).
- **Building the briefs from the previous round's reports worked.** Every worker started at the file
  the last round had named, and two of them (`Q-39`'s spelling, `Q-36`'s header guard) landed
  directly on that basis. Keep doing that: the reports under `/tmp/r8/wNN.log` are round 9's scopes.
- The four PASSes are all one cause each: `io/abstractclass`'s version-0 class (three tests) and the
  last `MakeProject` test. With them, all four `ntuple/makeproject` tests pass.
- One commit was **reverted for branch coverage**: worker 1's divergence closure in `interp.py`
  cannot reach its "no warning" branch without a real file. The lesson for the next round: when a
  worker adds a callback, the *callback's* branches need a test that reaches them - ask for it in the
  brief, or the revert eats the round's work.
- Round 9's first scopes, from this round's reports: `Q-38`'s `wclasses.py` half (12, worker 5's
  report), `Q-44`'s `Export` writer (20, worker 6's), `W0-25`'s container writers (6, worker 7's),
  `Q-36`'s `reconcile` wiring (9, worker 3's remaining hunks), `Q-48`'s next formula wall (40), and
  `P-10`'s GIL (worker 16's).

## Round 9's close (recorded here for the next iteration)

- Base `9b37ece`, closed at `a35c8e7`. Five commits landed, **two reverted after measurement**, three
  roottest tests in and **none lost to code** (`/tmp/r9/final2b` vs `/tmp/r8/final2`).
- **The last roottest TIMEOUT is gone.** `P-10` closes with `P-7`/`P-8`/`P-9`: a read of a translated
  `std::atomic` hands the interpreter to the thread it waits for, and `testSetAddressLoop`'s 33 runs
  pass inside the limit. The two remaining roottest TIMEOUTs are the CERN-network copies.
- **Both reverts were verified by rerunning the area.** Worker 5's declared-layout read cost four
  PASSes and gained none (`/tmp/r9/revert-run`); worker 2's counted pointer gained nothing where it
  was aimed. A measurement that says "revert" is as valuable as one that says "merge" - both are in
  the ledger with their numbers.
- `tree/friend/file2` moved for an environmental reason (`W0-28`: a fixture's file is missing at
  `--jobs 5` in a fresh copy), not a code one - chasing it in code would have been a mistake, and the
  bisect that proved it is what saved the round an hour.
- One worker (7) took its target to PASS and left the work **uncommitted in its worktree** (`W0-29`);
  the commit message discipline is worth repeating in the brief.
- Round 10's briefs come from `/tmp/r9/wNN.log`: `SplitClones`' four writer sites (worker 3's
  measurement), `jsonbuffer.py`'s containers (worker 7's unmerged work), `_Set`'s translator half
  (worker 8), `stressgraphics`' encoder (worker 14), `double32` (worker 10), `selectorreader`
  (worker 11), and `Q-48`'s next formula wall (worker 1).

## Round 10's close (recorded here for the next iteration)

- Base `05b43b7`, closed at `d33afbbd`. Eleven commits landed, one reverted after measurement, and
  **roottest went 1141 → 1183** (`/tmp/r10/final2`): the 40 `treeformula`/`fastcloningeventtree`
  draws that four worker-rounds had measured now pass, with `python/memory`, `execCreateAndMerge`,
  `execKeyOrder`, `test_stringfiltercolumn` and the `cmsUnload` pair.
- **The most important finding is about the harness, not the code.** `/tmp/rootsrc-6.40` was
  damaged: `roottest/root/meta/evolution/` held 3 files where the release has 82. Every roottest
  figure up to round 9 came from that tree, so tests with missing inputs both "passed" for nothing
  and failed for nothing. The intact sparse clone is **`/tmp/rootsrc-6.40b`** and every later run
  must use it; `tools/rootsuite list` gives the same 2820 non-tutorial tests there, so counts stay
  comparable. Round 9's base re-measured on the intact tree (`/tmp/r10/base9-run`) fails the same
  three `meta/evolution` tests, which is how the delta was separated from the tree.
- Two lessons for the next round:
  - **A revert is not proof of a cause**: `Q-36` was reverted in round 9 *and* round 10, and the
    second bisect showed the three "lost" tests fail at the old base too. Bisect against the *base*
    before blaming a commit.
  - Four workers died with `backend unavailable` (`W0-32`); their briefs are unchanged and are round
    11's scopes by default.
- Round 11's first scopes: the 12 `io/tree` tests (both `SplitClones` halves landed: `b231145`,
  `e7b2784`, `ecb6457`), `io/json`'s containers (`9134a49` landed `TMap`), the `_Set` cursor
  (worker 12 never committed), `stressgraphics`' encoder, and re-deriving one round-9 figure on the
  intact tree to confirm the comparison.

## Round 11's close (recorded here for the next iteration)

- Base `2a4744f6`, closed at `641748fd`. Nine commits landed, one unmerged for a collision, **two**
  roottest tests in (`tree/evolution/WriteVector`, `io/json/STL0`) against four lost to the
  CERN-network timeout - all four pass alone (`/tmp/r11/net`), so the effective count is 1185.
- **The round's real property: the wall moved one file further out, seven times.** Workers 4, 6, 7,
  8, 11, 12 and 13 each measured that the refusal now lives one file beyond the one their brief
  gave them, and left a patch. That is the fourth round in a row with this shape, and it is now the
  dominant cost of the loop: each worker spends forty minutes to hand over a file name.
- Round 12 should therefore **aim at whole subsystems, not single files**: give one worker the
  *chain* a refusal walks (`tree.py` → `objects.py` → `wbranch.py`, or `geometry`'s writer plus its
  `TBrowser`), and let it edit all of them. The disjointness rule has been protecting against
  cherry-pick conflicts, but the conflicts have been cheap (one this round, resolved in five
  minutes) while the handovers have been expensive.
- One collision resolved by rule: worker 15's `TArrayX` writer and worker 5's `TArrayX` declaration
  both landed in `wdeclared.py`; merging both broke worker 5's tests, so 15's is unmerged and the
  ledger says why.
- Next round's first scopes, from these reports: the `io/geo` export chain (18 tests, two files),
  `treeproxy`'s nested vector (`w07-nested-vector.patch`), `SplitClones`' next layer, bigevent's
  next member, and the `stressgraphics` encoder.

## Round 12's close (recorded here for the next iteration)

- Base `14ef50ad`, closed at `e15028fe`. Twelve commits landed, **14 roottest tests in and none
  lost** (`/tmp/r12/final2` vs `/tmp/r11/final2`) - six times round 11's two, and the difference is
  the scoping rule.
- **The experiment: sixteen chains, not sixteen files.** Every worker got the whole path a refusal
  walks. Round 11's seven handovers became zero this round: `treeproxy`'s nested vector, two
  `selectorreader` tests, five `schemaEvolution` draws, `collection/tbits`, `tree/addresses/twrite`
  and a split collection of a macro's class all came in. **Keep chain scoping.**
- One test was rewritten, not loosened: `.typedef` now lists what the session's macros brought
  (ROOT's behaviour), so the characterisation test asserts CINT's heading and the shape of each
  line, and keeps its substance - that `.typedef` and `.autodict` change neither the branch's name
  nor its leaves. It failed under xdist before the fix because another test had loaded a macro in
  the same worker: **a test that pins global state is a flake waiting for a different `-n`.**
- Round 13's chains, from this round's reports and the queue: the `io/geo` export writer (worker 2
  left five files uncommitted), `SplitClones`' last layer (worker 1 moved it again), bigevent,
  `meta/evolution` (worker 4's member-wise collection is one layer on), `TBufferJSON`'s containers,
  the makeproject translator, and the `stressgraphics` pad macro.

## Round 13's close (recorded here for the next iteration)

- Base `70523896`, closed at `e0097751`. Twelve commits landed, **25 roottest tests and 7 module
  tests in, none lost** (`/tmp/r13/final2`, `/tmp/r13/final1`). Chain scoping is now two rounds old:
  39 tests where file scoping produced 2.
- **The geometry chain is the round**: nineteen `io/geo/export_*` tests, five rounds after the first
  worker measured that `Import` needed a GDML reader. Nothing new was needed this round - only the
  freedom to edit `pyroot/geom/`, `geom/`, `wdobjects.py` and `winfo.py` together.
- The makeproject group completed (`io/ntuple/makeproject` is all PASS), `ROOT.list` and the bare
  standard templates landed, TMVA's PyMVA brought six module tests, and `ROOT::Math`'s `Polynomial`
  one more.
- Four workers still reported a wall one file outside their chain: `pyroot/trees/core.py`'s
  `MakeProxy` (worker 15), `interp.py`'s walk (worker 4), `files.py` (worker 11), the writer's
  `SplitClones` (worker 3). Their reports are `/tmp/r13/wNN.log`, and round 14 should build chains
  that *start* at those files.
- Round 14's first chains: `pyroot/trees/core.py` + `selector.py` + `layout.py` (MakeProxy, 8 tests),
  `interp.py` + `streamers.py` + `rdobjects.py` + `pyroot/core/declared.py` (meta/evolution, 14),
  the compression/`store.py` chain, `io/geo`'s last two, and the `io/tree` writer chain.

## Round 14's close (recorded here for the next iteration)

- Base `12646b58`, closed at `77042aa8`. Ten commits landed, **eleven roottest tests and one
  module test in**. The round's win is `TTree::MakeProxy` (`c545445e`): five `treeproxy` tests, two
  `vectorint` selectors and `treeformula/string/MakeProxy`, which round 13's worker 15 had measured
  but could not reach because the file sat outside its chain.
- **A second agent session is committing to `feat/root-testsuite`.** Mid-round it reset the branch
  and re-applied its own commits; the round's ten cherry-picks survived only on their `r14-NN`
  branches. All ten were re-applied by hand and are content-identical. **Check `git log` at the
  start and the end of every round, and quote the sha actually measured** (`W0-38`).
- **`io/perf` is one bad machine away from red**: the four buffers pass at 211-261 s of a 300 s
  limit (`/tmp/r14/solo`), and this round's load average of 12 flipped all four to TIMEOUT. `P-9`
  closed them functionally; they need the next factor of two (`W0-39`).
- Six workers reported without a commit and named the file each chain still needs: `interp.py`'s
  walk, `tools/rootsuite`'s working copy, `jsonbuffer.py`, `files.py`. Round 15's chains start
  there, and the `schemaEvolution` log bug (`tools/rootsuite`) is a harness defect worth a worker
  of its own.

## Round 15's close (recorded here for the next iteration)

- Base `99525851`, closed at `177f6159`. Four commits landed from eight chains, **eight roottest tests
  in and none lost to code** (`/tmp/r15/final2` at `--jobs 3` vs `/tmp/r14/final2` at `--jobs 5`).
- **The machine is shared and it dominated the round.** Load averages of 20, then 64, from another
  session: five-way measurement was impossible, so the round measured at `--jobs 3` and says so.
  `W0-44` is the rule for next time: check `uptime` first, and record the parallelism a number was
  measured at, because a count is only comparable within one.
- **The CERN endpoint was slow the whole round**: the four network tests timed out at `--jobs 1` with
  nothing else running (`/tmp/r15/net`). `W0-43` says they must never be counted as losses without
  that rerun - this is the third round the distinction has mattered.
- Worker 3's `CompareContent` pair-by-position work is the round's real gain: three `checkWarn*` tests
  and, more importantly, the six stderr DIFFs in `meta/evolution` now fail for a *named* reason - the
  harness compares stdout for an OUTREF test where ROOT merges stderr into it.
- **The coverage gate flaps under load**: three consecutive `-n 4` runs under a load average above 20
  each reported 100% coverage of a different single branch missing, and `-n 2` is clean twice. That
  is a measurement artefact, not a gap - `W0-45` records it so no round adds a test to chase it.
- Round 16's first chains: the `meta/evolution` stderr question (16 tests, harness + `streamers.py`),
  `pyroot/trees/core.py` + `objects.py` (`SplitClones`' last answer), `interp.py` + `rdobjects.py`
  (the data model), `wdobjects.py` (bigevent), and `reflection.py` (`pairEnumEvo`).

## Round 16's close (recorded here for the next iteration)

- Base `df6d9833`, closed at `4509b416`. **One chain ran, not sixteen** - the machine's load average
  stayed between 9 and 36 from another session all round, so a fan-out would have produced
  measurements nobody could trust (`W0-47`). The iteration was spent on the one thing that needed no
  machine: a harness defect.
- **`W0-40` is fixed.** ROOT's `ROOTTEST_ADD_TEST` hands the driver `OUTPUT <build>/<name>.log`, so a
  test that follows one reads its sibling's output (`schemaRun2check`'s `sed schemaRun2.log`). The
  harness stored nothing and looked for a build-tree `OUTREF` in ROOT's *source* tree. Every roottest
  now writes its log and the judge reads a build-tree reference from the working copy:
  `schemaEvolution` went **9 PASS / 5 FAIL → 12 PASS / 2 DIFF / 4 UNSUPPORTED**.
- **The rule probably applies far wider** (`W0-46`): any group whose tests read a sibling's `.log`
  was failing on this alone. Round 17 should re-run the multi-test-log areas and count what moves -
  that may be the cheapest remaining win in the suite.
- Round 17's chains, unchanged from round 15's list: `pyroot/trees/core.py` + `objects.py`
  (`SplitClones`), `interp.py` + `rdobjects.py` (the data model), `wdobjects.py` (bigevent),
  `reflection.py` (`pairEnumEvo`), `stressgraphics`, `jsonbuffer.py`, `files.py`.

## Round 17's close (recorded here for the next iteration)

- Base `ba2985b6`, closed at `5ff715a5`. Six commits landed from eight chains, **six roottest tests in
  and none lost**, measured in the six areas they touch (`/tmp/r17/areas`) plus `io/event`
  (`/tmp/r17/check-ev`).
- **Round 17's own full-suite run is invalid and is not quoted.** At `--jobs 3` with a load average
  of 44 the `io/event` fixture's `TestIoEvent` collided with itself and **47 fixtures failed behind
  it**, plus 13 `ObjectSlot` failures - 64 apparent losses. The same area passes **29/29 at
  `--jobs 3` alone**. `W0-48` records it: the suite's roottest count is not stable under load, and a
  group whose tests share a working copy must be measured on its own.
- **Two assumptions were measured and closed.** `W0-46`: `STOREOUT` had no wider prize - exactly two
  roottest directories read a sibling's log, one of them now fixed. And `W0-49`: a second session's
  uncommitted work keeps appearing in the main worktree (`Slot.takes`, untested, red coverage gate);
  a round must `git status` before every gate run.
- Round 18's chains: `interp.py` + `rdobjects.py`'s data model, `reflection.py`'s `pairEnumEvo`,
  `pyroot/core/files.py`'s `double32`, `wdobjects.py`'s bigevent, `jsonbuffer.py`, and the graphics
  encoder - all untouched this round - **plus** `io/event`+`fastcloningeventtree` measured on its own
  so its 47 fixtures stop moving the count.

## Round 18's close (recorded here for the next iteration)

- Base `cb5c17e4`, closed at `6ae78a7a`. Three commits landed from eight chains, **0 tests gained and
  0 lost** across ten areas (167 tests, `--jobs 1`, `/tmp/r18/areas`).
- **That is the round's finding, not an accident** (`W0-50`). Every remaining group in those areas is
  one *feature* from passing: `io/bigevent`'s next member, `pairEnumEvo`'s emulated layout, the data
  model's reader, `double32`'s `TRealData` consumer, the graphics pad's macro, `io/tree`'s
  `SplitClones`, `jsonbuffer.py`'s last writers. Four of eight workers stopped with the same
  sentence - "the wall is outside my chain" - which is true and no longer useful.
- **The scoping rule has to change with the work.** A chain scoped by *file path* worked while the
  walls were one file away (rounds 12-13 gained 39 tests that way). It cannot contain a wall that is
  a missing *behaviour*: the next rounds take **one group each with the whole component in scope**,
  and expect a round per group. `W0-50` says so.
- The machine's load average stayed at 9-20; every measurement was `--jobs 1`, and `io/event` +
  `fastcloningeventtree` was measured on its own (29/29) so `W0-48`'s collision cannot move the count.
- Round 19's candidates, one per round now: `io/bigevent` (16), `pairEnumEvo` (4 + 40 near-misses),
  the data model (7), `double32` (7), `io/tree` (11), the graphics pad (3), `jsonbuffer` (6).

## Round 19's close (recorded here for the next iteration)

- Base `2c6be318`, closed at `a12bc98b`. Six commits landed from eight whole-component groups,
  **0 tests gained and 0 lost** in the measured areas (`/tmp/r19/areas`, `/tmp/r19/tree`).
- **Two rounds at zero is a measurement of the work, not of the loop.** `W0-52` now records, from the
  eight reports, exactly what each remaining group needs: a layout model for bigevent's template
  member, ROOT's `fStreamerInfos` index bookkeeping for `pairEnumEvo`, one line in
  `pyroot/core/reflection.py:GetClassVersion` for `io/tree`, four things for `double32` (`W0-53`),
  the pad's PostScript character counts at ±50 in `canvas/` for the graphics tests, `MEMBERS` entries
  for `jsonbuffer`, and one ACLiC `NameError` in `pycint/library/loading.py` that blocks 24 cases of
  `roottest/python`.
- **The most valuable next step is the cheapest one already measured**: the `NameError` (24 cases)
  and the `GetClassVersion` line (11 tests) are each one change in a named file. The rest are
  days-scale features, and the ledger now says so instead of hoping a 50-minute worker will find them.
- A conflicted cherry-pick left conflict markers in `pyroot/core/declared.py` for a while - the
  documented catastrophe. It was resolved by hand in `bdc4a2cc`, and a stray `libTD.xrdlib` was kept
  out of the tree. `git status` before every gate run is not optional.

## Round 20's close (recorded here for the next iteration)

- Base `1000afdd`, closed at `f283ae1c`. Three commits landed from four targets, **0 tests gained, 0 lost**.
- **The three walls round 19 named as cheapest are now landed** and none of them flipped a test:
  the ACLiC `NameError` (`2ae5d980`), `TClass::GetClassVersion` for a file-only layout (`95014dca`) and
  `TBufferJSON` reading a class its table describes (`8b83e603`). `roottest/python` measured
  **8 PASS / 17 FAIL** and `io/tree` **5 PASS / 12 FAIL** - unchanged in PASS.
- **Three consecutive rounds at 0 gained (12 commits).** `W0-55` records the conclusion: these groups
  are features with several walls each, every wall a real ROOT behaviour, so no 50-minute worker can
  finish one however precisely its first wall is named. **The next iterations take ONE feature per
  round**, chosen from `W0-52`/`W0-53`, and are judged by the wall they remove and the commit they
  land. The PASS ledger stays authoritative for what has been achieved (**roottest 1239, non-tutorial
  1406**) but is no longer a good per-round target.
- `W0-49` bit twice more this round: another session's uncommitted work (`interp.py`, `buffers.py`,
  `rdobjects.py`, a test) put a failing test and 99.99% coverage in the tree; it was saved to
  `/tmp/r20/stray-main-worktree.patch` and `/tmp/r20/stray2.patch` and the gate re-run clean.

## Round 21's close (recorded here for the next iteration)

- Base `57940f19`, closed at `02f37c6e`. Four commits, **one feature**, as `W0-55` set: `Event::fTemplateMember`
  (`template2<template1<int>>`) is now declared, described, written and read (`67d08e8e`, `5b3c58f9`,
  `d84f3e7f`, `ac2541de`), one file per worker.
- **The wall is gone even though no PASS moved.** `io/bigevent`'s dictionary and library now build and the group
  measures **2 PASS / 0 FAIL / 2 UNSUPPORTED** (`/tmp/r21/ev2`), its refusal moved from the template member to
  `a TArrayI is not something this writer puts in a ROOT file` (`W0-57`). That is the unit `W0-55` asks a round
  to be judged by.
- **`W0-57` is the next feature and it serves two groups**: `TArrayI`/`TArrayF`/`TArrayD` as a *member* is what
  `io/bigevent` needs next and is one of the four things `double32`'s `template32` needs (`W0-53`).
- **`W0-56` is a harness hazard**: a run started immediately after a cherry-pick failed with
  `ImportError: cannot import name 'compare' from 'xrdroot.filling'` and the same group passed on a rerun. The
  translation cache is not always revalidated when the source changes under it - rerun once before believing an
  `ImportError` in a build step.

## Round 22's close (recorded here for the next iteration)

- Base `249dce34`, closed at `356d077f`. Three commits, **one feature**, `TArrayF`/`TArrayI` as a *member*
  (`d0e3a132`, `de5f7ef3`, `79c574f3`), one file per worker - the same shape as round 21.
- **The wall moved again and no PASS did.** `io/bigevent` still measures **2 PASS / 0 FAIL / 2 UNSUPPORTED**
  and its refusal is now `a TVector3 here is a float rather than the dict of ...` (`W0-58`). `io/double32`
  measures **3 PASS / 1 FAIL**; `template32` did *not* follow, so `W0-57`'s second half needs its own work.
- **Two features, two rounds, two walls removed, zero PASS.** That is the rate the remaining work has, and
  the ledger says so; the group is now three walls deep (`fTemplateMember`, `TArrayI`, `TVector3`) and the
  next is named.
- **`W0-59`:** three of four workers left untracked `docs/README.md`, `docs/gotchas/`, `docs/discoveries/` in
  their worktrees. Nothing was copied into the tree, but a brief must say plainly that `docs/` is not
  theirs, and a round should look for stray `docs/` before taking a worktree's files.

## Round 23's close (recorded here for the next iteration)

- Base `21ab15c4`, closed at `4fa05128`. Three commits, **one feature**, and **the first PASS since round
  17**: `io/bigevent/write` passes (`4913f410`, `2c123241`, `9da8ee08`).
- **The feature was the third wall of the same group**: a ROOT class's member that is itself one of ROOT's
  classes - `TLorentzVector::fP` is a `TVector3`, and `root_row` gave **a float** where a dict of members
  was needed. One-line diagnosis in the brief (`root_row(R.TLorentzVector(1,2,3,4), "TLorentzVector")["fP"]`),
  three files, one commit each.
- `io/bigevent` now measures **3 PASS / 0 FAIL / 1 UNSUPPORTED** (`/tmp/r23/ev`). `read`'s last wall is named
  in `W0-60`: a `vector<TObject> *` member is a kind the *reader* does not decode inside an entry - the
  writer already lays it out. That is one feature, and it would close the group.
- **Three features, three rounds, and this one paid**: the first two moved walls without a PASS (rounds 21,
  22), this one moved a wall *and* flipped a test. The difference is that this wall was in a *row*, not in a
  layout - the cheap kind `W0-52` predicted, one file deep.
- **`W0-61`:** the main worktree keeps receiving another session's stray files (`dot.C`, `h.h`, `mkroot.py`,
  `test.root` this round). Every one was caught by `git status` before the gate; a round should also
  `git clean -n` before it commits.

## Round 24's close (recorded here for the next iteration)

- Base `a12e5e43`, closed at `59a2cac3`. Two commits, **one feature**: the reader now takes a container a
  member *points at* (`vector<TObject> *fVectorTobject`, stype 71) - `0d4ed89f`, `1ac5101d`.
- `io/bigevent` stays **3 PASS / 0 FAIL / 1 UNSUPPORTED** (`/tmp/r24/ev3`) and `read`'s refusal moved on to
  `fVectorTnamed`.
- **The round's most useful result is a near-miss, recorded instead of shipped.** `fVectorTnamed` is
  `vector<TNamed> *fVectorTnamed[6]` - a pointer to a container *and* an array (stype 91 = `STL_POINTER + 20`).
  Treating 91 as 71 moved the harness's refusal further, which looked like progress; the existing unit test
  `test_an_array_of_pointers_to_containers_leaves_the_type_as_the_file_describes_it` failed with
  `DID NOT RAISE`, so the change was **reverted** (`W0-62`). A future round writes the array-of-pointers
  reading rather than relabelling the kind.
- **`W0-63` is the reusable lesson**: a one-line change that moves a refusal further is not progress until
  the full suite agrees. The unit tests caught this in 0.3 s; the harness alone would have called it a gain.
  Every round should run `pytest` before believing a harness delta.

## Round 25's close (recorded here for the next iteration)

- Base `3aa0b0db`, closed at `dab9b98d`. Two commits, **one feature**: an array of pointers to containers reads
  out of one record, each item named (`23be66d4`, `b25b54a4`) - `W0-62`, the wall round 24 had found and reverted.
- `io/bigevent` stays **3 PASS / 0 FAIL / 1 UNSUPPORTED** (`/tmp/r25/ev`) and `read`'s refusal moved **two members
  further**, to `fMapTNamedp`.
- **`W0-65` records the contrast that matters**: round 24 removed the refusal by relabelling the stype and a unit
  test caught it (0.3 s); round 25 removed the same refusal by *implementing the reading* and every gate agreed.
  Both moved the harness's refusal; only one was progress. That is `W0-63` demonstrated rather than asserted.
- **`W0-64` is the next feature**: `fMapTNamedp` is `map<TNamed*,int>` (stype 500, extras `(4, 61)`) and the reader
  refuses it as "a class with a streamer of its own" - a `std::map` whose *key* is a ROOT class. The writer already
  lays it out.
- The machine's load average was 10-12, so both workers measured at `--jobs 1` and the round quotes those runs.

## Round 26's close (recorded here for the next iteration)

- Base `a4f10d15`, closed at `85adfa05`. **Three features, three workers, disjoint files** - and for the first time
  a *general* correctness gap rather than a test-specific one.
- **`fName`/`fTitle` are the object's own members, read and written** (`7f1663d1` + tests `172093d3`). A macro class
  deriving from `TNamed` may now do `fName.Form("%s_%d", n, id)` and `fTitle = "x"`, exactly as C++ does:
  `xrdroot run` on the six-line probe prints `name=[hello_7] title=[x]`, and **`roottest/root/io/tree/create-r`
  PASSES** (`/tmp/r26/base1` -> `/tmp/r26/tree`, PASS 5 -> 6).
- **A container template's nested `iterator` is nameable** (`a6336c75`): `std::list<std::pair<A,B>>::iterator` used to
  fail with `AttributeError: '_Template' object has no attribute 'iterator'`; `test-r` now *runs* and moved FAIL ->
  **DIFF**.
- **The `Event`'s maps read** (`33678ae2`): `io/bigevent` stays 3 PASS / 0 FAIL / 1 UNSUPPORTED but the refusal is now
  a *format* error deep in the data (`vector<TLine>` 480 bytes = 48/event off) rather than a refusal at a map - which
  is `W0-67`, a real reader bug the refusal had been hiding.
- **`W0-66` is the next feature and it is 11 tests**: the `io/tree` chain now reads the objects but prints
  `TTP  ref: (!NULL!)` - empty names and null `TRef`s. Reading a `TRef` (PID bookkeeping) is what the family wants,
  and `tref-write`/`tref-read` are refused for the same reason.
- **`W0-69`**: worker 04 committed `docs/README.md` and `docs/discoveries/…` (`572b39e2`), not taken. A brief cannot
  prevent it; **check `git show --stat` of every worker commit for `docs/` before cherry-picking**.

## Round 27's close (recorded here for the next iteration)

- Base `4c82dc31`, closed at `0e7e9fda`. **Seven commits, four features**, two waves of workers (four, then two).
- **`roottest/root/io/tree/*` went PASS 6 -> 10** (`/tmp/r26/tree` -> `/tmp/r27/tree2`): `create-rc`, `create-rcr`,
  `create-rcrr` (a split `TClonesArray` of a class the *program* declared is cloned by that declaration) and
  **`test-r`** (a `TRef` read out of a tree resolves through its `TProcessID`, so the refs print the names they name).
  The other seven of the family now *run* and are DIFF: `TTree::Show`'s formatting is all that is left of them
  (`W0-72`, the cheapest seven tests in the suite today).
- **Three named walls fell without a PASS**: `W0-67` (a member-wise container of a class with a base - `vector<TLine>`'s
  480-byte desync is gone; `io/bigevent` now stops at a garbage length, `W0-71`), `W0-68` (SplitClones of a file-only
  class) and part of `W0-66`.
- **`W0-74` is the round's process lesson**: five commits were cherry-picked before the coverage gate ran, and it
  reported **99.99%** - a worker's eight-file commit had tests for some of its branches and not the other six
  (`ClonedSlot._live`'s three answers, `filed_object`'s `PutObjectWithID`, two `Show` text paths). `23e557d8` adds
  the five missing tests. **Run `pytest --cov` after every cherry-pick**, not once per round.
- `io/json` gained no PASS but `String` moved UNSUPPORTED -> FAIL onto a translator bug (`W0-73`); `io/bigevent` is
  still 3 PASS / 0 FAIL / 1 UNSUPPORTED.
- The machine's load average was 8-14 all round; every measurement was `--jobs 1`, and the `io/tree` family was
  measured alone each time (its tests share one working directory).

## Round 28's close (recorded here for the next iteration)

- Base `d42d8a5c`, closed at `3d69abeb`. **Five commits, five features, 0 PASS** - and the round's value is that it
  turned three groups into *counted* differences.
- **`W0-72`'s row was wrong and round 28 corrected it.** The seven `io/tree` DIFFs were not `Show`'s formatting: after
  `df9caceb` (the count line, unbroken members, ROOT's line shape) and `1d00c3ed` (a clone carries the `ProcessID`
  record its `TRef`s name) each entry differs in **exactly three lines** (`W0-75`): `fUniqueID`/`fBits` (ROOT writes a
  referenced object's process in the top byte - `16777225` where we have `9`), `r1.fRef` (`->` where ROOT prints
  `TRef`), and one trailing blank line. **Seven tests, three one-line behaviours each.**
- **`io/json/String` is one line from PASS** (`W0-76`): `ex4.fStr2 = null` vs `""` for a null `TString*`. It got there
  because `6224b151` fixed the translator - `TJsonEx1.Init()` was being called off the class instead of the object.
- `io/bigevent/read` moved its wall again (`W0-77`: a class reference that points nowhere); `2aee3989` fixed an array
  of strings and an array of pointers reading fewer elements than it has.
- `db8b97ee` writes a macro class's member and its `$ref` in JSON - it did not flip `BasicTypes`/`Objects`/`STL`/`STL1`.
- **The coverage gate ran after every cherry-pick** this round (`W0-74`'s lesson) and was 100% each time;
  `12819 -> 12833 passed`, `TOTAL 117866 0 29554 0 100%`.
- The machine's load average was 9-14; every measurement was `--jobs 1`, the `io/tree` family alone.

## Round 29's close (recorded here for the next iteration)

- Base `9dd616a0`, closed at `21517c9a`. **Three commits, +1 PASS.**
- **`roottest/root/io/json/String` PASSES** (`44e2d2ed`): a null `TString*` member is `null` in JSON, not `""`. It had
  been one line from PASS since round 28.
- **`W0-75` was corrected, not chased**: its third item - "one trailing blank line" - was **my own measurement artefact**.
  The harness's judge (`tools/rootsuite/refdiff.filtered`) drops blank lines and `Processing …`, so they never counted.
  Measured with the harness's own `custom_diff`, `String` was always one line and `test-rc` four line kinds. **Use
  `custom_diff` to measure a diff, never a hand-written one.**
- **`test-rc` is down to three line kinds** (`W0-78`): `N.fUniqueID` (`16777225` vs `9`), `N.fBits` (`50331672` vs `24`),
  `r1.fBits` (`50331648` vs `0`) - ten of each per file, and no other difference. The `fRef`→`TRef` line is
  `d07c8610` and the `Show` shape is round 28's `df9caceb`. **The oracle can write the same file for a byte-by-byte
  comparison; that is the whole of the next round's first task.**
- `test-rcrr123` (`W0-79`) is different in kind: its references name the *previous level's* clones where ours name the
  base objects, and the `GOOD!` suffixes pile up to three - a third-level `TRef` chain, its own feature.
- `8ffa748c` (the heap bits) moved no test; worker 04 died with only stray `docs/` committed nowhere (`W0-80`), so
  `W0-77` (`io/bigevent/read`'s class reference) is still open with the same brief.
- Gates: `12835 passed`, `TOTAL 117876 0 29556 0 100%`, ruff/mypy/maintainability/components/`mkdocs --strict` clean,
  and `XRDROOT_TEST_OTHER_MACHINE=1` the same. The coverage gate ran after **each** cherry-pick (`W0-74`).

## Round 30's close (recorded here for the next iteration)

- Base `4cdb3649`, closed at `9d509b79`. **Three commits, +2 PASS** - and the round's shape is new: the fix that paid came
  from a worker who only *wrote the bug down*.
- **`W0-81` is the round.** The four `io/json` `TypeError: 'NoneType' object is not subscriptable` failures were never in
  `jsonbuffer.py`: the translator emitted `self.fChar[i][j] = with_char(...)` for a `char[N][M]` member whose runtime
  array is one flat buffer of rows. Worker 04 measured it with its own docs commit (`c3045a11`, not taken - `docs/`), the
  round wrote the finding into a brief, and worker 05 - with nothing but the write-up - fixed it in `71dba22d`.
  **`BasicTypes` and `Objects` became byte-identical to their `.ref`s**, `io/json` went 5 -> 7 PASS and **every test of
  that area that runs now passes** (0 FAIL, 0 DIFF; the five left are named refusals).
- **`W0-83` sharpens `W0-78`**: the `fBits`/`fUniqueID` reassembly does **not** belong in `buffer.py::referenced` - worker
  01 measured that doing it there breaks seven round-trip tests. The file holds `fBits & ~0x03000000` and a plain
  identifier, and ROOT's reader puts the heap's two bits back plus the process index **+1** (`TBufferIO::GetPidOffset`,
  in `TKey::fSeekPdir`'s high bits). It belongs in **`interp.py`'s element read**. The oracle probe that measures it is
  `/tmp/r30/oracle/{probe2,show}.C`: ROOT prints `16777216` and `50331672` for a referenced `TNamed`.
- `8e797bcb` (a `std::pair` on its own) and `ae646b4e` (the identifier's tests) moved nothing; `W0-82` (`FromJSON` filling
  an existing object's containers) is `STL`/`STL1`/`ArrayCompress`.
- **`W0-84`**: workers 02 and 03 died without a commit (only stray `docs/`), so `W0-79` and `W0-77` are still open with
  the same briefs. Two deaths in two rounds - check the session files, not the logs.
- Gates: `12845 passed`, `TOTAL 117893 0 29564 0 100%`, ruff/mypy/maintainability/components/`mkdocs --strict` clean,
  `XRDROOT_TEST_OTHER_MACHINE=1` the same; the coverage gate ran after each cherry-pick (`W0-74`).

## Round 31's close (recorded here for the next iteration)

- Base `c862d06a`, closed at `27f23efd`. **Five commits, +3 PASS.**
- **`W0-83` closed**: a `TObject` reads back with the two values ROOT's reader restores, each measured against the oracle
  *before* it was written (`/tmp/r30/oracle/`): `fBits`' `kIsOnHeap|kNotDeleted` (`a65836dd`) and `fUniqueID`'s process
  byte, `(pid_index + 1) << 24 | uid` (`ca510e97`). `io/tree/test-rc` **PASSES**; `io/tree` is **11 PASS / 7 DIFF / 2
  UNSUPPORTED**.
- **`W0-82` closed**: `FromJSON` fills the container members an object's constructor left empty (`46dbd65c`) and a value
  is written in full where a pointer is a `$ref` (`6b3e0928`). `io/json/STL1` and **`StreamerLoop`** PASS - the second
  was a refusal nobody had listed - and `io/json` is **9 PASS / 1 DIFF / 2 UNSUPPORTED**.
- **`W0-85` is the next two-line feature and it is five tests**: the five siblings of `test-rc` now differ in
  `r1.fUniqueID` and `r1.fBits` **only** - the same two values, but for the objects of a *cloned* file read through a
  split collection's member-wise path (`ClonesSlot`/columns), which the object-path commits do not reach.
- `W0-86` is `STL`: our `TJsonEx13`'s `set1` is 2 343 bytes where ROOT's is 5 144.
- `d07a00cd` (a `Double32_t` template spelling) moved nothing: `io/double32` is still 3 PASS / 1 FAIL.
- Gates: `12867 passed`, `TOTAL 118034 0 29632 0 100%`, ruff/mypy/maintainability/components/`mkdocs --strict` clean,
  `XRDROOT_TEST_OTHER_MACHINE=1` the same; the coverage gate ran after each cherry-pick (`W0-74`).

## Round 32's close (recorded here for the next iteration)

- Base `3372a1eb`, closed at `f359f0be`. **Two commits, +3 PASS** - and two of the round's four workers produced nothing,
  which is now a pattern worth its own row.
- **`io/json` is finished as far as running goes**: `roottest/root/io/json/STL` PASSES (`8c150a28` - a `std::set`/`std::map`
  keeps a copy of what it is given, as C++ does), and the area is **10 PASS / 0 FAIL / 0 DIFF / 2 UNSUPPORTED**
  (`/tmp/r32/json`: only `ArrayCompress` and `RootClasses` remain, each a named `TBufferJSON` refusal).
- **`other/math/*` is 16 -> 18 PASS** (`62afe7fe`: ROOT::Math's `IntegrationOneDim`/`IntegrationMultiDim`/
  `IntegratorMultiDim`, `WrappedTF1` and `Integrator::Integral(f, a, b)`), measured by its worker at
  `/tmp/r32/other` with no PASS lost.
- **`W0-88` is the round's process finding.** Four of the last round's workers died without a commit, and *this* round's
  two live ones left a **discovery record uncommitted in their worktree** (`/tmp/r32/w04/docs/discoveries/…`), correctly,
  since `docs/` is not theirs. That record is where `W0-87`'s six blocked tests came from - `mathtools.py`'s
  `Minimizer.Clear()`/`GetMinosError()`, `ROOT.Minuit2.Minuit2Minimizer`, `Fit.BinData.GetErrorType`,
  `_FitNamespace.Chi2FCN` - and it names the three that need translator pointer arithmetic. **A round reads
  `<worktree>/docs/discoveries/` before it closes.**
- `W0-89` records that `roottest/root/math/*` is nearly empty (5 tests, **0 PASS**) and untouched by any round.
- `W0-85` (five `io/tree` tests, two lines each) and `W0-79` (`test-rcrr123`'s third-level chain) are open only because
  their workers died: both briefs are unchanged and are the next iteration's first two.
- Gates: `12888 passed`, `TOTAL 118142 0 29650 0 100%`, ruff/mypy/maintainability/components/`mkdocs --strict` clean,
  `XRDROOT_TEST_OTHER_MACHINE=1` the same; the coverage gate ran after each cherry-pick (`W0-74`).

## Round 33's close (recorded here for the next iteration)

- Base `91bf694c`, closed at `cc6125b7`. **Two commits, +1 PASS** - and one worker produced a diagnosis worth more than
  the commits.
- **`other/math` is 19 PASS / 0 FAIL / 45 UNSUPPORTED / 1 TIMEOUT** (`/tmp/r33/other`): `mathcore-testMinim` passes because
  `pyroot/core/mathtools.py`'s `Minimizer` can now be cleared and asked for its MINOS errors (`88fa5527`).
- **`roottest/root/math/*` now runs**: ROOT::Math's `SVector`/`SMatrix` exist (`e3838fad`, a new
  `pyroot/linalg/smatrix.py`) and all five tests reach their next wall, each named in `W0-90` (`ROOT.Transpose` is
  missing, `testLoadLibMathMore`, `SMatrix<d…` typing, `ValueError: invalid l…` at `testOperations.cxx:95`).
- **`W0-91` is the round's prize: `W0-79`'s root cause, measured byte for byte.** Worker 02 shipped no commit but found it
  (its note is `/tmp/r33/w02/W02-tref-note.md`): ROOT numbers a clone into a **new `ProcessID` record** when `addref`'s
  `TRef` constructor calls `TProcessID::AssignID`, and writes that index into the reference **and** into each cloned
  object's identifier top byte (`r1.fUniqueID = 0x02000009`, `r1.fBits = 50331672`; we write `0` and `50331648`). Two
  edits, both named: `wbranch.py`'s `tref_record` writes the process's **session** index where `TRef::Streamer` writes its
  **file** index (`file.WriteProcessID(ref._pid)` - `pyroot/trees/objects.py::_named_processes` already calls it and
  discards the result), and `copying.py`'s `ClonedSlot` must read a clone's `fUniqueID`/`fBits` from the live objects in
  `tree._addresses[leaf.branch]` rather than from the decoded columns. **With the second alone, stage 1 would break** -
  both are needed before any of the six DIFFs move.
- **`W0-92`**: a file xrdroot writes carries no `TRefTable` (`pyroot/trees/tree.py:251` never writes a `fBranchRef`;
  `wtree.py:1499` writes `0`), so `read_referenced()` reads every split collection of the entry in branch order.
- Worker 01 of this round died with nothing at all; worker 02 died with the note. **Read `<worktree>/*.md` too, not only
  `docs/discoveries/`** (`W0-88`'s rule, extended).
- Gates: `12907 passed`, `TOTAL 118409 0 29722 0 100%`, ruff/mypy (1019 files)/maintainability/components/`mkdocs
  --strict` clean, `XRDROOT_TEST_OTHER_MACHINE=1` the same; the coverage gate ran after each cherry-pick (`W0-74`).

## Round 34's close (recorded here for the next iteration)

- Base `8997cd97`, closed at `84271e6c`. **Three commits, 0 PASS** - and the round's most useful output is a sharper
  `W0-91` plus two process rows.
- **`W0-91` is two-thirds in.** `6bd0d6ba` writes the process byte into a split `TObject`'s identifier **column**, on top of
  rounds 31's object-path `a65836dd`/`ca510e97`. What is still missing is round 33's **first** edit
  (`src/xrdroot/wbranch.py`'s `tref_record`/`_ref_bytes` must write the process's **file** index via
  `file.WriteProcessID(ref._pid)`, not its session index) and the other half of the second
  (`pyroot/trees/copying.py`'s `ClonedSlot` must read a clone's `fUniqueID`/`fBits` from the live objects in
  `tree._addresses[leaf.branch]`). `test-rcr` still differs in exactly `r1.fUniqueID`/`r1.fBits`, which is round 33's note's
  prediction ("with (2) alone, stage 1 would break") - **the next round takes those two files and nothing else**.
- **`W0-93`**: a worker's own test shipped with **CCN 14** against the limit of 10, and `pytest` caught it through
  `test_maintainability.py`. The round split it into three, keeping every assertion. **Run every gate after every
  cherry-pick** (`W0-74` extended): coverage, ruff, mypy, maintainability, components.
- **`W0-94`**: two of this round's four workers produced nothing at all (one wrote no code, one not even a note) - the third
  and fourth such rounds in a row. A round should give a worker **one commit-sized target** and look at the session files
  in the first ten minutes instead of waiting an hour.
- `efa5f946` (ROOT::Math's free functions) moved `roottest/root/math`'s four failures without closing one; `W0-90`'s five
  reasons are re-stated in the queue for a fresh measurement.
- Gates: `12917 passed`, `TOTAL 118534 0 29744 0 100%`, ruff/mypy (1019 files)/maintainability (30 869
  functions)/components/`mkdocs --strict` clean, `XRDROOT_TEST_OTHER_MACHINE=1` the same.

## Round 35's close (recorded here for the next iteration)

- Base `1c25bcff`, closed at `4a9eb5ce`. **Five commits, +5 PASS** - the best round since round 27, and it closed the
  feature four rounds had been chasing.
- **`W0-91` is closed.** `920dfe4c` (a clone writes the identifiers its objects were numbered with) + `dee7d185` (a
  `TObject` read member at a time carries the two words its object has) are **round 33's two named edits**, and both were
  needed exactly as its note said. `io/tree` went **11 -> 14 PASS, 7 -> 4 DIFF, 0 FAIL** (`/tmp/r35/tree`):
  `test-rcr`, `test-rcrr`, `test-rcrr3` closed.
- **`other/math` is 19 -> 21 PASS** (`/tmp/r35/other`): `84e15c0f` (AssignSym/MatRepSym), `3e2588d0` + `e67c9a79` (ROOT::Math's
  genetic minimiser, `Chi2FCN` under its template name, the seed as ROOT's) closed `genetic-testGAMinimizer` and
  `genetic-GAMinTutorial`. It is **21 PASS / 0 FAIL / 43 UNSUPPORTED / 1 TIMEOUT**.
- **`W0-95` is the next feature and it is the same one a level deeper**: `test-rcr2`, `test-rcrr2`, `test-rcrr123` print the
  same 180 `r2` lines as ROOT but resolve them to the **base** object (`ref: N_0_8`) where ROOT resolves to the **first
  clone** (`ref: r1_N_0_8_8`) - a clone of a *clone* is not yet numbered into its own `ProcessID` record.
- `W0-96` and `W0-97` re-state the two maths areas so the next round does not re-derive them: `other/math`'s 43
  UNSUPPORTED want bucketing by reason (most are translator pointer arithmetic), and `roottest/root/math` should be taken
  as five macros read together, not one error at a time.
- The machine's load average reached **37** mid-round (another session) and then settled at 13; every measurement was
  `--jobs 1`, the gates `-n 2` (`W0-44`, `W0-45`).
- Gates: `12942 passed`, `TOTAL 118743 0 29788 0 100%`, ruff/mypy (1019 files)/maintainability (30 946
  functions)/components/`mkdocs --strict` clean, `XRDROOT_TEST_OTHER_MACHINE=1` the same.

## Round 36's close (recorded here for the next iteration)

- Base `59744112`, closed at `630618f8`. **Two commits, +3 PASS** - and the `TRef` family is finished.
- **`W0-95` closed and the family with it.** `a17cb476` (a reference written after a clone names the file's process record) is
  the clone-of-a-clone level, and it closed `test-rcr2`, `test-rcrr2`, `test-rcrr123`: `io/tree` is **17 PASS / 0 FAIL /
  1 DIFF / 2 UNSUPPORTED** (`/tmp/r36/tree`). Four rounds ago the same group was 11 PASS / 7 DIFF; the whole path
  `W0-66` -> `W0-83` -> `W0-91` -> `W0-95` is closed.
- **`W0-96` is bucketed**, and the table is the round's other product: a worker read all 43 `other/math` refusals (its own
  `--jobs 1` run was the only measurement of round 35, so it re-used it) and reduced them to **22 causes** with a file and
  a first step each (`/tmp/r36/w03/R36-math-buckets.md`). **Its top row is `W0-98`: the translator's refusal of the address
  of a vector element (`&v[0]`, `&v.front()`, `&v[i]`) blocks 10 of the 43 by itself** - `pycxx/emit_calls.py:354`, `:367`.
- `W0-99`: `io/json`'s last two refusals did not move with `f0c859a9` (which added ROOT's own array-compression forms as
  unit tests, not a fix). `W0-100`: `io/tree/read` is the group's last DIFF.
- **The round ran under a load average of 15-45** (another session): every harness measurement was `--jobs 1`, the gates
  `-n 2`, and no worker was asked to run more than one measurement (`W0-44`, `W0-45`, `W0-47`).
- Gates: `12949 passed`, `TOTAL 118750 0 29792 0 100%`, ruff/mypy (1019 files)/maintainability (30 962
  functions)/components/`mkdocs --strict` clean, `XRDROOT_TEST_OTHER_MACHINE=1` the same.

## Round 37's close (recorded here for the next iteration)

- Base `d13ce7e8`, closed at `b84bdc87`. **Two commits in, two reverted, 0 PASS** - and the round's real finding is about
  the *gate*, not the code.
- **`W0-103`: no gate could be measured.** The machine's load average went from 18 to **124** (another session). Under it
  `pytest -n 2` failed 7 `tests/test_tmva_learn_networks.py` cases that **pass alone** (15 in 3.9 s), `--cov` reported
  **0-27%** with "The following workers failed to return coverage data", and one `-n 0` run died with `Fatal Python error`
  at 91%. **`W0-45` is worse than recorded: above a load average of ~20 the coverage gate is not measurable, and nothing
  may be concluded from its output.** The tree's own state is sound - ruff, mypy (1019 files), maintainability (30 978
  functions), components and `mkdocs --strict` all pass, and the failing tests pass alone.
- **Two reverts, each with its exact reason.** `W0-101`: the address of a vector element (`W0-98`'s top row, 10 of
  `other/math`'s 43) **works** but ships six branches no test reaches (`cppstd/cells.py:94`, `pycxx/emit_calls.py:389`,
  `:397-399`, `emit_expr.py:538`) and the 100% floor may not be relaxed - branch **`r37-01`** keeps it, and the two tests
  it needs both hit a *translator* limit first (`list expected at most 1 argument, got 2` when two dereferenced pointers
  coexist; `a float returned by value cannot be assigned to in place` for `v.front() = x`). `W0-102`: the dictionary
  warning also warns for `TGeoAtt`, `TGeoVoxelFinder` and `TGeoParallelWorld` - ROOT's own classes - and
  `tests/test_pyroot_geom_rootfile.py` pins ROOT's exact message sequence.
- The two surviving commits moved nothing: `io/json` stays **10 PASS / 0 FAIL / 0 DIFF / 2 UNSUPPORTED**, `roottest/root/math`
  **0 / 4 / 1**. `io/tree` was not re-measured (its last DIFF, `read`, needed the reverted work).
- **Another session committed to the branch mid-round** (`21254ccb`, a harness perf change) and left
  `tools/rootsuite/runner.py` modified in the tree; it was checked out before the gates (`W0-49` again).
- Gates: ruff, mypy (1019 files), maintainability (30 978 functions), components (0 undeclared), `mkdocs --strict` - all
  pass. **The pytest/coverage gate is deferred to the next iteration** with the exact command below and the load caveat.

## Round 38's close (recorded here for the next iteration)

- Base `7216c284`, closed at `4bd75d90`. **Three commits, no harness PASS, and three pre-existing breakages fixed.**
- **`W0-101` is closed**: the address of a vector element (`b28bb2c2`'s feature, reverted in round 37 for six uncovered
  branches) is **back** in `bcd98495` **with each of them covered** - `cells.ItemRef.__repr__`, `_element_pointer`'s and
  `_element_call_end`'s `None` fall-backs, and `_element_call_address`. The two test forms that failed in round 37 failed
  for reasons that had nothing to do with the feature: `cintfake.fake()` has no `std::vector<double> v(3, 1.5)`
  constructor, and two dereferenced pointers in one function hit `list expected at most 1 argument, got 2`. `push_back`
  and one pointer per test avoid both. **`W0-98`'s ten `other/math` tests now have their feature** - they need a
  measurement, not more code.
- **`W0-104`: three breakages that predate this round**, found by running the gate rather than trusting it:
  `wpacking._whole` caught `struct.error` where **Python 3.14 raises `OverflowError`** (six `test_wpacking` cases died
  instead of falling back to NumPy), a test asked NumPy for `'atan'` (**NumPy 2.0's** alias for `arctan`; the venv has
  1.26), and two tests compared `round(x, 5) == 30.39946` with `==` - two *different floats* whose repr matches, so they
  asserted a spelling rather than a number, now a `pytest.approx` with the same five-decimal precision.
- **`W0-105`:** the venv runs **Python 3.14 and NumPy 1.26.4**, and CI runs 3.10-3.13 with NumPy 2.x - so both classes of
  failure are invisible to CI and visible only here. A round touching numerics or `struct` should print
  `sys.version_info[:2]` and `numpy.__version__` first.
- **The gate is still not measurable** (`W0-103`): the load average was 20-124 all round, `pytest -n 2 --cov` reported
  24% with dead workers and 7 `test_tmva_learn_networks.py` failures that pass alone. What *can* be stated: **916 unit
  tests pass** in every file the three commits touch, and ruff, mypy (1019 files), maintainability, components and
  `mkdocs --strict` are clean.
- **The next iteration's first act** should be to re-run the full gate when the load is under ~20, then measure
  `other/math` (the ten tests `W0-98` names) and `io/tree/read` (`W0-100`).

## Round 38's addendum (recorded at its close)

- **The whole-suite gate has three separate obstacles on this machine** (`W0-106`, all measured): the torch file
  `tests/test_tmva_learn_networks.py` fails 5-9 cases under `-n 4`/`-n 2` and passes alone in 8 s while its dead worker
  takes the coverage data with it (24% at `-n 2`); `-n 0` dies with `Fatal Python error` at 63-91%; and
  `tests/test_pyroot_references.py::test_a_reference_numbers_its_object_in_the_session_process_and_finds_it` fails only
  when it shares a worker (26 pass alone) - the global-`TProcessID` flake again.
- **The decomposition that works**: `pytest -n 4 --cov --ignore=tests/test_tmva_learn_networks.py
  --ignore=tests/test_pyroot_references.py` (12 925 pass, 99.73%), then
  `pytest tests/test_tmva_learn_networks.py tests/test_pyroot_references.py --cov --cov-append` (41 pass, 99.99%).
- **`W0-107`:** the last ten lines are `src/xrdroot/tmva/nettrain.py:188, 199-207` and a branch - another session's
  "training failed" path, which no test reaches. Even the decomposition cannot reach 100% until it has one.

## Round 39's close (recorded here for the next iteration)

- Base `ebd7485f`, closed at `297950c4`. **One commit, and the gate is measurable again.**
- **`W0-107` closed and the gate is green.** `tmva/nettrain.py`'s `_step` has two optimiser branches and **no test
  reached either** - the deep-network tests train through the trainer without ever calling a step. One parametrised test
  (`4f7ddee5`) calls it with ADAM and with momentum and checks each rule's first step. The two-command decomposition now
  reads **`TOTAL 118830 0 29826 0 100%`**: `pytest -n 4 --cov --ignore=tests/test_tmva_learn_networks.py
  --ignore=tests/test_pyroot_references.py` (12 925 pass), then `pytest tests/test_tmva_learn_networks.py
  tests/test_pyroot_references.py --cov --cov-append` (44 pass).
- **The first harness numbers in three rounds.** `other/math` is **21 -> 22 PASS** (`/tmp/r39/other`:
  `mathcore-testSpecFuncBeta`), 0 FAIL, 43 -> 41 UNSUPPORTED - and **`mathcore-stdsort` now runs**, which is the test
  `W0-98`'s vector-element feature unblocked. It hits the 300 s limit: `W0-108`, the translated `std::sort` must sort a
  buffer in one NumPy pass instead of one `ItemRef` at a time. **The limit is not to be raised.**
- **`io/tree/read` is exactly four lines from PASS** (`/tmp/r39/read`): `missingClass.ref` begins with four
  `Warning in <TClass::Init>: no dictionary for class …` lines (`TEmcl`, `TPhysObj`, `TNonEmcl`, `TNonPhysObj`) and our
  output has none. **`W0-102` now has its two-part fix named**: `TGeoAtt`, `TGeoVoxelFinder` and `TGeoParallelWorld` are
  ROOT's own classes and belong in `winfo.INFOS`; then "a class in the file's streamer list that is in neither `INFOS`
  nor `DECLARED`" is exactly the right guard, and both tests pass.
- Gates: ruff, mypy (1019 files), maintainability (30 990 functions), components (0 undeclared), `mkdocs --strict` all
  clean; the two-command pytest gate at 100%.

## Round 40's close (recorded here for the next iteration)

- Base `17c9a7fa`, closed at `aabf7f48`. **Two commits, +1 roottest PASS, and the `TRef` group's area is clean.**
- **`W0-102` is closed after three rounds.** `82001fb0`: the four `Warning in <TClass::Init>: no dictionary for class …`
  lines `missingClass.ref` wants (`TEmcl`, `TPhysObj`, `TNonEmcl`, `TNonPhysObj`) are emitted, and **`TGeoAtt`,
  `TGeoVoxelFinder` and `TGeoParallelWorld` stay silent** because a new `ROOT_CLASSES` set in `pyroot/core/reflection.py`
  names "ROOT has a dictionary for it and this kit implements nothing of it". Round 37's reverted attempt was right in
  every other respect. `roottest/root/io/tree/read` **PASSES**; the area is **18 PASS / 0 FAIL / 0 DIFF / 2
  UNSUPPORTED** (`/tmp/r40/tree`) where three rounds ago it was 11 PASS / 7 DIFF.
- **`W0-108` is closed as a decision, not a fix.** `other/math/mathcore/mathcore-stdsort` is a **benchmark**:
  `200000/(n/10)` passes of `TMath::Sort` and `std::sort` over vectors of 10 to 100 000 elements with a comparator
  object per pair. Measured: `xrdroot run stdsort.cxx` ran **4 minutes without finishing the first size**. It cannot pass
  inside the limit, and the limit is not to be raised.
- `de1e9cac` adds the tests for `declared.py`'s accessor fall-backs - the 2 lines and 1 branch that were the only gap at
  a clean load (its own file reports `declared.py` at 98% with those three covered).
- **`W0-109`: the gate needs a load average under ~20.** At 34-54 this round's two-command gate reported **99.54%** and
  **99.85%** with all 12 931 tests passing - a worker's coverage data lost, not a gap. Round 39's `TOTAL 118830 0 29826 0
  100%` remains the last clean measurement.
- ruff, mypy (1019 files), maintainability (31 005 functions), components (0 undeclared) and `mkdocs --strict` all pass.

## Round 41's close (recorded here for the next iteration)

- Base `54ff9563`, closed at `24b40f3b`. **Two commits, no measured PASS** - the machine's load average reached **144**
  again, so the round could measure almost nothing (`W0-103`/`W0-109`).
- **The round's finding became code**: `TBufferJSON::FromJSON(T*& obj, json)` is a **qualified static call**, and
  `pycxx/program.py`'s `reference_positions` answered out-parameters only for a member call or a single-part name, so
  `arr2` was never a cell and nothing could fill it (`9acf40db` + `pycint/dialect.py`'s `("FromJSON", 2): (0,)`). The
  refusal moved to `TBufferJSON::FromJSON cannot read this JSON as any class: a bare JSON array does not say what class
  it holds` - **the next gap is `FromJSON` reading a `TArrayI`, in `jsonbuffer.py`** (`W0-99`).
- **Worker 02's uncommitted work was recovered from its tree** and committed (`e008ab0b`): the three std algorithms, the
  ostream formatting state and `ROOT::Math::Random<T>`'s engine kinds - 11 `other/math` tests' worth of features whose
  measurement the load denied. Its next-permutation test had CCN 11 and was split in two, every assertion kept.
- **Worker 03's `smatrix.py` change is not taken** (`W0-110`): its own 31 tests pass, but `roottest/root/math/*` went from
  **0/4/1 UNSUPPORTED** to **0/4/1 TIMEOUT** under the load - unproven, and the patch is kept at
  `/tmp/r41/w03-smatrix.patch`.
- **`W0-111`:** two of three workers ended without a commit. A round should read each worktree's `git status` at the
  **halfway point** and copy the files of a session that has stopped, which is how w02's work survived this round.
- ruff, mypy (1019 files), maintainability (31 028 functions), components (0 undeclared) and `mkdocs --strict` all pass;
  the six test files the commits touch report **355 passed**.

## Round 42's close (recorded here for the next iteration)

- Base `4dad0dad`, closed at `938d9faf`. **Three commits, +1 roottest PASS**, and the round had no worker fan-out:
  the load average opened at **76** and never went under 14, so `W0-103`/`W0-109` meant code and unit tests only.
- **`W0-99` is closed and `io/json` is one test from finished.** `4e5436ff`: `TBufferJSON::FromJSON(T *&obj, json)`
  makes an object of the class the *declared* pointer names, and a cell carried only a number's width - so a null
  pointer cell could not say which class, and a bare JSON array was refused. `pycxx/emit_vars.py::cell_type` now carries
  an object pointer's class too (`void *` stays bare). **`roottest/root/io/json/ArrayCompress` PASSES** - all six of
  its arrays, in ROOT's four compressed forms - and the area is **11 PASS / 0 FAIL / 0 DIFF / 1 UNSUPPORTED**
  (`/tmp/r42/json`, from 10/0/0/2). `W0-114` is the last one: `RootClasses` needs a `TH1I`'s JSON, which is a
  feature and not a fix.
- **ROOT's iterator family landed** (`a2ef7d77`): `TIterator`, `TCollectionIter`, `TListIter`, `TObjArrayIter`,
  `THashTableIter`, `TMapIter`, each collection making its own, and a `TIter` that owns the one it holds and deletes it
  in `_destruct` - which made `TIter` a dialect *guard*, so a macro's scoped `TIter` is destroyed where its scope ends.
  `gtest/core/cont/coreconttestUnit`'s two `TIter` cases pass.
- **That exposed a translator bug of the general kind**: `Bool_t dir = kIterForward` is answered through
  `inherited(self, ...)`, and Python builds a default list as the class is defined, where there is no `self` - so the
  emitted `def __init__(self, ..., dir=inherited(self, ...))` raised `NameError: name 'self' is not defined`. A default
  that mentions `self` is now put off to the body, as a non-constant default already was (`emit_funcs._put_off`).
- **`W0-112` is the round's process finding, and it is about *my own* verification**: `next_permutation`'s successor
  search (`cppstd/objects.py:321`) and `TCollection.MakeIterator` had **no test**, and round 41 shipped the first with
  the gate unmeasurable. Part 1 of the two-command decomposition reports **99.73%** no matter what the tree does (the
  ignored files' code), so only **part 2's report names the misses**, and it is the command that has to be read.
  Both now have tests (`938d9faf`).
- **`W0-117` is the same lesson from the other side**: without `--cov`,
  `test_pyroot_distrdf_graph.py::test_a_dataset_spec_is_cut_like_a_chain_of_its_samples` fails on its own -
  measured at the round's **base** in a clean worktree - because an unclosed `a.root` is finalised inside the test
  and the suite turns a `ResourceWarning` into an error, while coverage tracing keeps the object alive until the
  fixture closes it. `rdf/distributed.py:73`'s `Window.change` drops an owned chain with its files open.
- **`938d9faf`** also lands `RootFinder::Name`/`Iterations` with ROOT's names, verified against the oracle
  (`BrentRootFinder`, `bisection`, `falsepos`, `brent`, `newton`, `secant`, `steffenson`). It moves no PASS and the
  round says so: `other/math`'s `testGSLRootFinder` prints its four solver blocks and stops at `ROOT::Math::Roots`,
  which is `W0-116`.
- **The gate is green at `fd657322`, even at a load average of 90-139**: `TOTAL 119070 0 29912 0 100%` by the
  two-command decomposition (12 948 passed, then 44 with `--cov-append`). ruff clean, mypy 1019 files,
  maintainability 31 064 functions, components 1018 modules / 0 undeclared, `mkdocs build --strict` built.

## Round 43's close (recorded here for the next iteration)

- Base `0cbe14e9`, closed at **`4e473e1e`**. **Two commits, +1 `other` PASS**, and the round's other product is
  the **first whole measurement of the gtest kind since round 13** - the load average was 11-20, so both the
  harness and the gate could be run.
- **`W0-116` is closed.** `0be3733b`: `ROOT::Math::Roots::Brent`/`Bisection`/`FalsePos`/`Newton`/`Secant`/
  `Steffenson` over `GSLRootFinder` and `GSLRootFinderDeriv` - **siblings** under `IRootFinderMethod`, as ROOT
  has them, because the two are handed different functions (a bracket, or the derivative and the `fdf` that
  gives both). `other/math/mathmore-testGSLRootFinder` **PASSES** (`/tmp/r43/gsl`): all six of its solvers
  bracket `sqrt(5)` and `Name()` answers with GSL's own spelling. `kMODAB`'s is `ModABRootFinder`, read off
  the oracle. `other/math` is **22 → 23 PASS**.
- **`4e473e1e` moves seven programs' walls and flips none**, and says so: `TTree`/`TAxis`/`TH1`/`TInterpreter`
  status words and `RootFinder::kMODAB` are the *first* failure of thirty gtest programs, and every one of
  those programs has several walls. Measured before and after at `/tmp/r43/enums2` (0 PASS / 7 FAIL both
  times); what each hits now is in the queue - `THistRange` for `test_THBinIterator`, `TH1::SetStatOverflows`
  (`W0-122`), `std::vector::resize` (`W0-120`), `BulkApiVariableTest.stdRead`, `TTreeClusterTest.countBaskets`.
- **The gtest ranking is current again** (`/tmp/r43/gtest`, bucketed at `/tmp/r43/gtest-buckets.md`):
  **53 PASS / 219 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350. The 219 first failures reduce to 16 missing
  class names and 16 missing methods; the rest are each program's own subject. The widest are
  `TInterpreter.CreateTemporary` (**26 programs**, `W0-119`), `RVec::Pt` (17), `ostream::rdbuf` (12),
  `ROOT.Detail` (11), `RNTupleModel.AddProjectedField` (10), `RDataFrame.Vary` (9).
- **`W0-119` is a scope question, not a fix**: the 26 `CreateTemporary` programs are the Clad derivator family,
  and `clad::differentiate` is cling's Clad plugin. Either the committed scope labels cover "a compiler plugin
  that differentiates C++" in a full sentence, or the derivative is computed numerically. It needs a review,
  not a quiet label.
- **`W0-121` is a fidelity bug found while proving the round's PASS**: `xrdroot run` looks for a `.cxx`'s
  *namesake* function before `main`, and echoes the returned value as a macro's - so `testGSLRootFinder` prints
  `(int) 0` where a compiled ROOT program prints nothing. Invisible wherever the judge is the exit code (as
  for every `other` test), a DIFF wherever a `test/` or `other` test has an `OUTREF`.
- ruff, mypy (1019 files) clean; the gate was run at the round's final commit - see "Final verification".

## Round 43's result of the gate (recorded in its Final verification)

- At `a57e2d05` (the docs commit on top of `4e473e1e`), the two-command decomposition is **`TOTAL 119167 0 29918
  0 100%`**: 12 954 passed then 44 passed, at a load average of 11-20. ruff clean, mypy 1019 files,
  maintainability 31 089 functions in 1729 files, components 1018 modules and 0 undeclared imports,
  `mkdocs build --strict` built.

## Round 44's close (recorded here for the next iteration)

- Base `9ba7eb49`, closed at `250324e3`. **One commit, +1 gtest PASS**, and a correction of round 43's own ranking.
- **`gtest/core/cont/TSeqTests` PASSES** (`250324e3`, measured at `/tmp/r44/seq`: the four `core/cont` programs are
  2 PASS / 2 FAIL, where they were 1 PASS / 3 FAIL). `ROOT::TSeq`'s `begin()`/`end()` are C++ **random-access**
  iterators: `it + n`, `it - m` for the distance, `--`/`-=` in place, `it[n]`, and `*it` as the `value` it holds.
  A `TSeq` works its value out - `begin + index * step` - rather than looking it up, which is why `c[3]` is the
  fourth value on from `c` even past the last stored one; `TSeqU`/`TSeqUL` were missing too.
  `coreconttestUnit`, which includes the same file, moves on to `TCollection.RangeCast` (`W0-113`'s
  `TRangeStaticCast`). The gtest kind is **54 / 350**.
- **`W0-118` is corrected, and the correction is the round's other product.** Round 43's bucket table counted
  *occurrences in a log*, not programs, so `TInterpreter.CreateTemporary` read as "26 programs" and `RVec::Pt` as
  "17" - it was 1 program, 17 times. Recounted per program from the same evidence (`/tmp/r44/gtest-buckets.md`):
  **114 of the 219 FAIL programs ask for no missing name at all** - each is an assertion of its own subject, so
  each is its own feature - and the 105 that do are 1-3 programs each. **There is no wide name cluster in the
  gtest kind**, and the next round must not plan as if there were. This is `W0-75`'s mistake (a hand-rolled count
  believed over the harness's own) and the rule stands: count with the evidence's own structure.
- **`W0-119` shrinks with it**: the Clad family is **not** 26 programs. Re-counted, `CreateTemporary` is one
  program's first wall (`math/mathcore/CladDerivatorTests`), so the scope question is worth asking but it is not
  the kind's wide win.
- The gate is green at `250324e3`: **`TOTAL 119220 0 29922 0 100%`** by the two-command decomposition (12 955
  passed, then 44), at a load average of 7-14. ruff clean, mypy 1019 files, maintainability 31 108 functions.

## Round 45's close (recorded here for the next iteration)

- Base `2f5d8392`, closed at **`4468d37b`** (ledger `80b9e82e`). **Two commits, +1 `other` PASS**, and the round's
  other product is that **the `test/` and `other` kinds were measured whole for the first time in ~28 rounds**
  (the load average was 11-26, so both the harness and the gate could run).
- **The kinds, measured:** `other` **40 PASS / 0 FAIL / 67 UNSUPPORTED / 2 TIMEOUT** of 109 (`/tmp/r45/other` is
  39 of them; `root_exe_errorMessage` makes 40, `/tmp/r45/errm2`); `test/` **8 PASS / 7 FAIL / 44 UNSUPPORTED**
  of 59 (`/tmp/r45/testkind`). The ledger's 23 for `other` was `other/math` plus a handful - the kind had never
  been counted whole. **All non-tutorial is 1421 / 2820 (50.4 %)** by those five figures.
- **`root_exe_errorMessage` PASSES**, and it took *two* fixes that each hid the other:
  `root` silently dropped a long option it did not know and read `-e invalid` glued into one word as a *macro
  file* - ROOT's parser names every option it cannot read and runs nothing, so `cli/root.py` now does, keeping
  the long options it takes and ignores (`--web=off`, `--notebook`, …) silent; and the message is three lines
  while the check is one expression, because **CMake's `.` in a `PASS_REGULAR_EXPRESSION` is every character
  including a newline** - `tools/rootsuite/judge.py` searched with `re.M` only and now searches with `re.S`.
  That second half is a **harness defect affecting every `PASSREGEX` test**, and it is the kind of thing that
  only shows up when a message is longer than a line.
- **`4a98e20d`** fixes a general translation bug: a translation has no `FILE` object for `stderr`, so
  `fprintf(stderr, …)` is `fprintf('stderr', …)` - which `printf.py` already read as the stream it names, and
  `fflush` did not, so `fprintf(stderr, …); fflush(stderr);` failed with `'str' object has no attribute
  'flush'`. Both read the name now (`cstdio.standard`), and `fflush(NULL)` flushes the two streams a translation
  writes to. `TBits::Set`/`Get` - ROOT's bulk forms over the *bytes* of an array of any width - landed with it.
- **`W0-123` is the next `testbits` wall and it is a translator gap**: `static const Char_t char_bits[] =
  {0,1,3,7,15,31,63,127}` is emitted as `char_bits = ''`, because a `char[]` from a braced list is taken as text
  and truncated at the first NUL. C++ makes it an array of chars, indexable and arithmetic-able. `W0-124` records
  that `test/`'s five `RooUnitTest` tests are a *framework*, not a name - the biggest group in that kind.
- Gates: **`TOTAL 119253 0 29932 0 100%`** by the two-command decomposition (12 959 passed, then 44), at a load
  average of 18-26. ruff clean, mypy 1019 files, maintainability 31 117 functions, components 1018 modules /
  0 undeclared.

## Round 46's close (recorded here for the next iteration)

- Base `56b0a92a`, closed at **`e142f240`** plus a tests commit. **`test/` +1: `test/testbits` PASSES**
  (`/tmp/r46/bits`, 41 tests), the kind **8 → 9**. The load average was 16-20, so the harness and the gate both ran.
- **`W0-123` closed**, and it was the general gap it was recorded as: a `char[]` initialised from a braced list was
  always a Python `str`, cut at its first NUL - so `const char bits[] = {0,1,3,7,15,31,63,127}` was `''` and
  `bits[i>>3] & 1 << (i&7)` had nothing to work on. An array is text now only when its list **spells a C string**
  (printable characters or C's white-space escapes, then NUL padding) and a byte buffer otherwise. The dialect
  gained **`buffer_functions`** (`memset`, `memcpy`) for the free functions whose first `char` array is memory - the
  companion of `buffer_methods` - which is how `memset(buf, 0, sizeof(buf))` makes `buf` a buffer from its
  declaration rather than a `str` no write reaches in place (`W0-125`, closed with it).
- **`TBits::Get` had a second bug its own test found**: ROOT's overloads fill `GetNbytes()` *bytes* whatever the
  array's element width, and writing one element per byte ran off a `Char_t[8]` and put a byte in each of a
  `Short_t[4]`'s slots. It copies bytes now, so an array of `Short_t` takes them two at a time.
- **`7e369769` is a general translator fix**: `while (in >> line) if (line.Contains(...))` - ROOT's own
  `test/stressGraphics.cxx` counting a PostScript file - read into a `str`, because an extraction gives one and a
  `str` has no `Contains`. Both places a read is written (`extraction` in an expression, `_read_statement` for a
  statement) now build the dialect's string class around the word. `test/stressgraphics-interpreted` moves
  **FAIL → UNSUPPORTED**, refused by name: `ROOT has localtime` (`W0-126`).
- **`e142f240` is the gate's own lesson**: the two fixes pushed `_read_statement` to ccn 11 / npath 624 and
  `_buffers` to ccn 14 / npath 386 - the full gate caught it and both were **split**, never limited
  (`W0-93` again: run every gate after every commit).
- Also recorded for the next round: a **failed extraction leaves the target changed** (`in >> word` at the end of a
  stream sets `word` to `''` where C++ leaves it as it was), found while writing the test above and not chased.
- Gates: **`TOTAL 119286 0 29940 0 100%`** by the two-command decomposition (12 962 passed, then 44), at a load
  average of 16-20. ruff clean, mypy 1019 files, maintainability 31 126 functions, components 1018 modules /
  0 undeclared.

## Round 47's close (recorded here for the next iteration)

- Base `44c42d2d`, closed at **`f0655cf0`**. **Two fixes, no harness PASS**, and two measurements that matter more
  than a single test: the whole **gtest kind re-measured** and **`stressgraphics`'s refusal turned into its real
  subject**.
- **`W0-127` closed** (`35c96338`): `while (in >> word) {}` cleared `word` as the loop ended, because the empty text a
  *failed* read gives was stored. C++ sets failbit and assigns nothing, and never reads through a stream that has
  already failed. `cppstd/istreams.py::extracted(stream, current, ctype)` is the read now in both places one is
  written - and it is also where the dialect's string class goes around the word, so the class and the failure rule
  are one function instead of two.
- **`W0-126` closed** (`f0655cf0`): C's broken-down calendar time exists - `tm` (with `tm_year` from 1900 and
  `tm_mon` from January being 0), `localtime`, `gmtime`, `formatted_time` and `strftime`, whose `char[]` is given
  its text the way `sprintf`'s is (a new statement handler `_strftime`; `TEXT_TARGETS` is for cells, and putting it
  there wrote through a `Cell` wrongly).
- **The gtest kind is current: 54 PASS / 218 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r47/gtest`), against
  round 43's 53. `TSeqTests` is the gain (round 44) and **nothing was lost in four rounds of fixes** - that is the
  `W0-63` check the kind had not had since round 43.
- **`test/stressgraphics-interpreted` moved UNSUPPORTED → FAIL, and that is the round's most useful measurement**
  (`/tmp/r47/sg2`): it now *runs ROOT's own 32-test graphics stress suite* and fails on the images' byte counts -
  `Test 1. TLine` 1289 against ROOT's 529, PDF 1585 against 12827, PNG 4665 against 3937. Behind the refusal was the
  `canvas/` fidelity work `W0-52` has recorded since round 19, now with a number per format (`W0-128`).
- **`test/testbits` still PASSES** (`/tmp/r47/sg2`), so `W0-127`'s change to every read did not cost the round 46
  PASS.
- Gates: ruff, mypy (1020 files), maintainability, components all clean and the two-command coverage gate at
  **`TOTAL 119506 … 100%`** before the last two test-only commits, then re-run at `f0655cf0` - see Final
  verification.

## Round 48's close (recorded here for the next iteration)

- Base `c003f165`, closed at **`f16f5aeb`**. **Two commits, no harness PASS** - and the round's real product is a bug my
  own gate caught plus ROOT's own `testTH1` measured case by case.
- **`W0-122` closed** (`4e2ba73b`): `TH1::Normalize` (no option makes the integral over the bins' *widths* one, `"max"`
  the largest bin, `"sum"` the sum of the weights, `GetEntries` untouched) and `TH1::EStatOverflows` as the enum a
  macro names a value through (`TH1::EStatOverflows::kConsider`), with `Get/SetStatOverflows` over the file's
  `fStatOverflows` and the static `TH1::StatOverflows(on)` as the global flag. **ROOT's own
  `hist/hist/test/test_TH1.cxx` went from 13 failing cases to 11**: `TH1.Normalize` and `TH1.StatOverflows` are
  **OK** now, measured by running the macro directly (`W0-63`: no harness delta is believed without the code's own
  test agreeing). The numbers in `Normalize`'s unit test are ROOT's ref's, including the variable-bin case
  (14.399998 and 7.999999).
- **`f16f5aeb` is the gate's catch, and it is the round's most useful finding.** Putting the enum in `TH1`'s body as a
  *class* made `hfnative._attribute` refuse that body - a class object is neither a method nor metadata - so **`TH1D`
  stopped being a native class** and every batched ntuple fill fell back to the scalar loop: **13 failed at the
  gate**. The bisect (78 passed at the round's base in a fresh worktree) settled it in two minutes rather than an
  argument. A class written in a body is an enum's namespace - reading it runs nothing - so `_attribute` now accepts
  a class whose own members are metadata, and a body holding an arbitrary object is still refused.
- **`W0-129` is what `testTH1` still needs, case by case** (recorded, not chased): `SmoothArray`, `THLimitsFinder`,
  `SetBuffer`, the implicit-ownership registration case and rows 175/218/270/276-277, plus three `TAxis` bin-edge
  numerics that must match ROOT exactly. **The program is not closable in one round** - nine cases, each a feature.
- **`W0-121` was re-examined and left as it stands**: whether `xrdroot run` should call a `.cxx`'s namesake or its
  `main` cannot be settled against the oracle, because the only file that shows the difference
  (`testGSLRootFinder.cxx`) defines its own `TStopwatch` and ROOT cannot interpret it at all. ROOT's own
  `TApplication::ProcessFile` calls the *namesake*, which is what this kit does; the echo is cling's. Recorded so the
  next round does not re-derive it.
- Gates: **`TOTAL 119535 0 29984 0 100%`** by the two-command decomposition (12 980 passed, then 44), at a load
  average of 12-26. ruff clean, mypy 1020 files, maintainability 31 211 functions, components 1019 modules /
  0 undeclared, `mkdocs --strict` built.

## Round 49's close (recorded here for the next iteration)

- Base `55318546`, closed at **`3f3dae7f`**. **One fix, no harness PASS**, and ROOT's own `testTH1` is now **10 failing
  cases** where round 48 left it at 11.
- **`TH1.AddBinContent` takes one bin number per axis.** `TH1::AddBinContent(bin[, w])` is one axis's bin and its weight;
  `TH2`'s and `TH3`'s overloads are `(binx, biny[, binz][, w])` - a weight only when one *more* number arrives than the
  histogram has axes. Ours read the first argument as a global bin and the second as a weight, so
  `h2.AddBinContent(1, 1, 1.)` added to global bin 1 and dropped the y bin; ROOT's test read 0 where ROOT reads 1.
  The mapping is the bins' own (`_global`/`GetBin`), and `Eval` - which works out a global bin itself - now goes
  through a new `_add_to_bin` rather than through the coordinate form, which the round's own gate caught as a
  regression in `test_a_function_of_two_or_three_variables_is_set_at_each_bin_s_centre_too` (`W0-63` again).
- **`W0-129` is now a case-by-case account of the ten that remain**, each with its message and what it needs:
  `SmoothArray` (a static over a raw array, ROOT's 353 routine), `THLimitsFinder` (`OptimizeLimits`), `SetBuffer` **and
  the buffer's own semantics** (200 weighted fills into one bin then `Sumw2()` must still give `sqrt(200*4)`, so a
  buffer has to keep the weights for a later `Sumw2`, and `SetBuffer` raises a small size to at least 100), `Dump`'s
  `*fArray                       ->0` line, the two implicit-ownership cases, and three `TAxis` bin-edge numerics.
- Gates: **`TOTAL 119540 0 29984 0 100%`** by the two-command decomposition (12 981 passed, then 44), at a load
  average of 21-23. ruff clean, mypy 1020 files, maintainability 31 213 functions, components 1019 modules /
  0 undeclared, `mkdocs --strict` built.

## Round 50's close (recorded here for the next iteration)

- Base `a21ec264`, closed at **`2e10bb4d`**. **One commit, no harness PASS**, and ROOT's own `testTH1` is now **9 failing
  cases** (round 49 left it at 10).
- **`TAxis` reports the binning it cannot use.** ROOT's `TAxis::Set` *reports* - `Error in <TAxis::Set>: Axis limits need
  to be finite numbers`, `Error in <TAxis::Set>: bin edges must be in increasing order` - and leaves the default one-bin
  axis; it does not raise. Ours said nothing. And `TAxis(edges)` - ROOT's constructor from a vector, which a macro writes
  as `TAxis _({1, 1})` - was read as *no* argument and fell back to the default binning, so the equal edges never reached
  a check. `TAxis.EqualBinEdges` is **OK** now.
- **`W0-129` is the handover item**: `testTH1`'s nine remaining cases, each with its message and its real requirement -
  `SmoothArray` (the static 353 routine), `THLimitsFinder`, `SetBuffer` **and the buffer's semantics** (a buffered fill
  must keep its weights for a later `Sumw2`), `Dump` (ROOT prints the *C++* members as `*name` at column 30; ours prints
  Python attributes - a fidelity gap), the two implicit-ownership cases and two `TAxis` bin-edge numerics. **None is a
  missing name alone; each is a feature, and the program is not closable in one round.**
- Gates: **`TOTAL 119555 0 29992 0 100%`** by the two-command decomposition (12 982 passed, then 44), at a load
  average of 24-39. ruff clean, mypy 1020 files, maintainability 31 215 functions, components 1019 modules /
  0 undeclared, `mkdocs --strict` built.

## Handover: where this loop stands at iteration 50

- **The mission is not done** and cannot be declared so: roottest ≈ **1241 / 2173**, gtest **54 / 350**, pyunittest
  **81 / 129**, `test/` **9 / 59**, `other` **40 / 109** - about **1423 / 2820** non-tutorial. `AGENTS.md`'s "Done means"
  asks for all of them PASS or refused in a full sentence for something genuinely out of scope.
- **The gate is green and reproducible**: `TOTAL 119555 0 29992 0 100%` at `2e10bb4d`, by the two-command
  decomposition of `W0-106` (one process cannot run it on this machine). Run it with a load average under ~20
  (`W0-109`) or read only its failures, never its coverage figure.
- **The most valuable queue entries** for whoever continues, in order of measured cost:
  `W0-129` (`testTH1`'s nine cases, `hist/`), `W0-113` (`TRangeStaticCast` and ROOT's static `Class()`, 2 gtest
  programs), `W0-120` (`std::vector::resize`), `W0-128` (the `canvas/` encoders' character counts, which is what
  `test/stressgraphics` now fails on), `W0-126`/`W0-127`-style translator gaps being closed one at a time, and
  `W0-114`/`W0-115` for the last two big roottest and gtest items.
- **Nothing is uncommitted and nothing is pushed.** The last ten rounds' evidence lives in `/tmp/r41` … `/tmp/r50`.

## Round 51's close (recorded here for the next iteration)

- Base `b9570d8d`, closed at **`4227df45`**. **One commit, no harness PASS**, and ROOT's own `testTH1` is now **8 failing
  cases** (round 50 left it at 9).
- **`TH1::SmoothArray(nn, yy[, ntimes])` exists.** It is HBOOK's `hsmoof` - Friedman's 353QH twice - over an array a
  program hands as a pointer, in place; `stats.smooth_array` has been this kit's version of it since `TH1::Smooth` was
  written, so the *static* name was the whole gap. Its write lands where the pointer points, so a `std::vector`'s own
  values smooth, and ROOT's assertion - a ramp and one repeated value come back unchanged - holds.
- **`W0-129` is down to eight cases with each one measured**, and the two expensive ones are named as such:
  `SetBufferedSumw2` needs `SetBuffer` *and* a buffer that keeps its weights for a later `Sumw2` (with `fill_one` the
  hottest path in the project, every read that must flush first is a surface); `DumpOutput` needs ROOT's whole-object
  listing, whose wanted line is `*fArray                       ->0                 [fN] Array of fN floats` - and `fArray`
  is not in this writer's member table, so it is not a formatting tweak.
- Gates: **`TOTAL 119561 0 29992 0 100%`** by the two-command decomposition (12 983 passed, then 44), at a load
  average of 19-26. ruff clean, mypy 1020 files, maintainability 31 217 functions, components 1019 modules /
  0 undeclared.

## Round 52's close (recorded here for the next iteration)

- Base `0537d989`, closed at **`5f58c997`**. **One commit, and the first harness PASSes in four rounds: gtest 54 -> 56.**
- **A collection's element has the type C++'s range gives it.** `for (auto bcl : *cl->GetListOfBases())` hands its
  body a `TObject *` whatever the list holds, and `Class()` is a *static* member, so the compiler resolves
  `bcl->Class()` to `TObject::Class()` and ROOT prints `TObject` for a `TBaseClass`; Python finds `Class` on the
  object and printed the class it really is. What a translated range visits is now `TypedElement`
  (`pyroot/core/rangecast.py`): the object in every respect but `Class`, which answers the range's static type - with
  a `__class__` property so `isinstance` still sees the object. `pycint/runtime/adapters.py::iterate` is the one place
  that is decided, because it is the one place a translated range-for asks what it visits.
- **`ROOT::Detail`** exists too: `TRangeStaticCast<T>(c)` (every object as a `T *`, unchecked, as `static_cast` does)
  and `TRangeDynCast<T>(c)` (only the objects that really are a `T`, `None` for the rest, as `dynamic_cast`'s
  `nullptr`). The translator already resolved the unqualified names to `ROOT.TRangeStaticCast`, so no `pycxx` change
  was needed.
- **`gtest/core/cont/coreconttestUnit` and `gtest/core/cont/testTypedIteration` both PASS** (`/tmp/r52/cont`, 2 -> 4 PASS
  in the area); **`W0-113` closed**. Nothing lost: `roottest/root/io/tree/*` **18 / 0 / 0 / 2** (`/tmp/r52/tree`),
  `roottest/root/io/json/*` **11 / 0 / 0 / 1** (`/tmp/r52/json`), and 66 gtest programs of `io`/`hist`/`graf2d`
  **26 / 32 / 8** - the same set as round 47's (`/tmp/r52/gtest-iohist`).
- Gates: **`TOTAL 119625 0 29998 0 100%`** (12 990 passed, then 44) at a load average of 24-33; ruff clean, mypy 1021
  files, maintainability 31 245 functions, components 1020 modules / 0 undeclared.
- Queued: **`W0-130`** (a wrapper is one more reference a macro can keep - `TCollection` should unwrap what it is
  handed), **`W0-131`** (round 44's gtest bucket table is eight rounds stale; re-measure the kind whole before working
  from it), **`W0-132`** (`test_THBinIterator`: 20 cases in 64 s, 7 failing - a bin iterator's cost and fidelity).

## Round 53's close (recorded here for the next iteration)

- Base `6f1b71a9`, closed at **`f93144fd`**. **Two commits, +2 gtest PASS (56 -> 58)**, and the round's second
  product is the gtest ranking made current, which answers round 44's open question.
- **`W0-131` is closed and it says "no name cluster"**. The kind measured whole against a worktree of HEAD
  (`--xrdroot-src /tmp/r53/head/src`, so a commit landing mid-run cannot move the answer) at `--jobs 3`: **56 PASS /
  216 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 in **573 s** - far cheaper than the hour it was budgeted at. The 218
  that are not PASS reduce to 216 distinct first failures, so each failing program is its own feature. **The counting
  that matters is the other one**: gtest prints a failing case twice, and counted once **47 programs have exactly one
  failing case** - one feature each from a PASS - listed with its message in `/tmp/r53/one-case.md` (45 after this
  round's two). That is the list the next rounds should work down.
- **`ROOT::RangeStaticCast`/`RangeDynCast`** (`7f90083d`): `ROOT/RRangeCast.hxx`'s pair takes a *pointer* type and works
  on any range - `ROOT::RangeStaticCast<const ClassB *>(arr)` over a C array of pointers - where round 52's `Detail`
  pair takes a class and a collection. `_class_named` now strips a template argument's `const` and `*`, and asks the
  interpreter's own table (`declared.find_class`) for a class the *program* declared, which a namespace lookup cannot
  see. `gtest/core/foundation/testRRangeCast` PASSES, both cases.
- **`TDataType::GetType`** (`f93144fd`) answers the `EDataType` code a `typeid` is, which is how a streamer, a leaf and
  a branch say what they hold. The codes were already `collectionproxy.EDataType`'s; this is the lookup from the
  Itanium name and the codes under the bare global names a macro writes (`kInt_t`, `kBool_t`). The mapping is ROOT's,
  measured (`/tmp/r53/dtype4.C`): `-1` for `const char *` as for every class, `7` for `char *`.
  `gtest/core/meta/testTDataType` PASSES.
- **`W0-133` is a fidelity bug the work surfaced**: `gInterpreter->Calc(...)` prints nothing in ROOT 6.40
  (`/tmp/r53/calc.C` measures it beside `ProcessLine`'s `(int) 5`), and `TInterpreter.Calc` here is
  `gROOT.ProcessLine` verbatim, so `testTDataType` writes `(int) 3` ROOT does not. Two existing tests pin the echo, so
  the fix corrects both in one commit.
- Gates: **`TOTAL 119666 0 30000 0 100%`** by the two-command decomposition (13 009 passed, then 44) at a load
  average of 20-42; ruff clean, mypy 1022 files, maintainability 31 252 functions, components 1021 modules /
  0 undeclared.

## Round 54's close (recorded here for the next iteration)

- Base `2a668e00`, closed at **`2ce1f188`**. **Two commits, +1 gtest PASS (58 -> 59)**, and the round's real
  product is a bug of the general kind: **a local C++ destroys at its scope's end, and one spelling never did.**
- **`98df6a2e`**: `RNTupleWriter::Recreate` gives a `std::unique_ptr` back, so a macro that fills a writer inside
  `{ ... }` writes its file at the closing brace - and ours did not, so the file stayed 0 bytes and the reader
  refused it (`test_ntuple_readstring.root ends at 0`). Three things were missing and only the three together
  move it: `pycxx`'s inference answered `None` for a **qualified static call** `T::f(...)` (its `method_returns`
  table was consulted only for a call on an object; `_library_call` now asks it for `parts[-2], parts[-1]`), the
  ROOT dialect says what the two RNTuple writer factories give back, and `RNTupleWriter` - already a `Guarded` -
  needed the `_destruct` a dialect guard is called through. **`gtest/tree/ntuple/ntuple_randomaccess` PASSES.**
- **`2ce1f188` closes `W0-133`**: `TInterpreter::Calc` takes the silent path `ProcessLineFast` takes, because
  ROOT's `TCling::Calc` prints nothing where `ProcessLine` echoes `(int) 3` (`/tmp/r54calc.C` measures both).
  **The row was wrong about its cost**: it claimed two existing tests pinned the echo and would have to be
  corrected. They did not - `_fast_processor` falls back to `process_line` when a translator has only that one -
  and both files pass untouched. A queue row's *measurement* is worth more than its *prediction*.
- **`W0-134` is the wall behind it**: `gtest/tree/ntuple/rfield_string` now wants
  `RNTupleDescriptor::GetClusterDescriptor`/`GetPageRange`/`GetPageInfos` - the page-level introspection that is
  that test's own subject. `W0-135` records `gtest/tree/tree/testTChainSaveAsCxx`'s three `TFile`s in one scope
  (whose the third `TTree` registers with is unmeasured against the oracle), and `W0-136` the harness hazard this
  round paid for.
- **`W0-136` cost this round a whole-kind measurement**: a `--kind gtest` run at `--jobs 3` sat at 226/350 for
  25 minutes with its own three workers in state `Ts` (suspended) while two stale `xrdroot run` processes from
  `/private/tmp/suite-9c38` burned 84 % CPU. The kind was therefore measured in its two areas instead
  (`/tmp/r54/ntuple`, `/tmp/r54/core`), which is honest but is not the same claim.
- Gates: **`TOTAL 119669 0 30000 0 100%`** by the two-command decomposition (13 017 passed, then 44) at a load
  average of 10-78; ruff clean, mypy 1022 files, maintainability 31 263 functions, components 1021 modules /
  0 undeclared.

## Round 55's close (recorded here for the next iteration)

- Base `127c20e2`, closed at **`d1752d03`**. **One commit, +1 gtest PASS (59 -> 60)**, and the round is a pure-data
  one: five numbers ROOT's own oracle gave, and nothing else.
- **`d1752d03`**: `ROOT/TIOFeatures.hxx` declares `EIOFeatures` twice - `ROOT::EIOFeatures` is the 6.34 spelling that
  6.40 keeps and no longer sets a bit in (`kSupported` 0), `ROOT::Experimental::EIOFeatures` the one a current writer
  sets (`kSupported` 1) - and `ROOT::Experimental::EIOUnsupportedFeatures::kUnsupported` is 0. A basket's header has
  `TBasket::kIOBitCount` (1) bits for them, so `TBasket::EIOBits::kSupported` is 1. **`ROOT::EIOUnsupportedFeatures`
  and `ROOT::ExperimentalEIOFeatures` are deliberately not namespace names**: measured against 6.40, ROOT has neither,
  and the test writes both through `ROOT::Experimental`. `pyroot/histv7._Experimental` is the one object this kit
  models `ROOT::Experimental` with, so it grows the two attributes and its docstring says why. **`testTIOFeatures`
  PASSES.**
- Nothing lost: `gtest/hist/*` + `gtest/tree/tree/*` - 73 tests, **30 PASS / 37 FAIL / 5 UNSUPPORTED / 1 TIMEOUT**
  (`/tmp/r55/areas`), `testTIOFeatures` the only gain. `roottest/root/{hist,tree}/*` measured whole for the first
  time since round 17: 313 tests, **138 PASS / 115 FAIL / 22 DIFF / 36 UNSUPPORTED / 1 SKIP / 1 TIMEOUT**
  (`/tmp/r55/rhist`) - **a snapshot, not a delta**, and `W0-137` says so.
- **`W0-138` is the round's process finding and it needs a human.** The task file's new step 0 opens a round off
  `origin/main` and step 6d pushes `$ROUND-main:main`. Measured here: **`main` is 238 commits behind
  `feat/root-testsuite`, and `origin/main` 45 behind** - a round based on either would lose every fix of rounds
  1-54, and 54 rounds of work have never been pushed. The round therefore opened its worktree off
  `feat/root-testsuite`'s tip in `/tmp/ralph-round-202611220111/main`, integrated and gated there, and did **not**
  push. Whether `main` is meant to become the trunk is a decision for the operator; until it is made, a round keeps
  to `feat/root-testsuite` and records that it did. **Resolved at round 65: `main` is the trunk** — `main` and
  `feat/root-testsuite` are the same commit (`origin/main..HEAD` and `HEAD..origin/main` both `0`), so step 0 bases
  every round on `main` and step 6d merges back to it; this paragraph is history, not procedure.
- The round **did** work in a worktree of its own and tore it down at the close (`rm -rf` of the worktree and the
  `round55-main` branch), which is the half of step 0/6d that holds here.
- Gates: **`TOTAL 119689 0 30000 0 100%`** by the two-command decomposition (13 020 passed, then 44) at a load
  average of 13-17; ruff clean, mypy 1023 files, maintainability 31 266 functions, components 1023 modules /
  0 undeclared.

## Round 56's close (recorded here for the next iteration)

- Base `9f4395db`, closed at **`4f884ec7`**. **One commit, +1 gtest PASS (60 -> 61)**, and - more valuable - **the
  gtest kind measured whole for the first time since round 53: 61 PASS / 211 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of
  350 (`/tmp/r56/gtest`, `--jobs 3`). The five it gained over round 53's 56 are rounds 53-56's, **and none was lost**:
  that is the `W0-63` check across four rounds that touched the destructor machinery and the `RDataFrame`/RNTuple paths.
- **`4f884ec7`: an array of objects is destroyed as its scope ends.** C++ destroys an array's elements where the
  array's scope ends, last first; this kit built `A a[3]` as a plain Python list, so nothing ran their destructors -
  `A::n` stayed 3 after the block. It is `cppstd.objects.ObjectArray` now (a list whose `_destruct` destroys each
  element and takes it out as it goes), `pycxx`'s `_destructible` asks a one-dimensional array's *element* instead of
  refusing every array, and the local lands in the `try`/`finally` a scalar local already used. **`testMake_unique`
  PASSES** - its `std::make_unique<A[]>(3u)` had been emitted as `A(3)`, the array form lost and its size taken as a
  constructor argument; `_make` now builds `array('A', n, make=A)` (and `array('int', n)` for an arithmetic element),
  and `_owned` reads the array back out of the `unique_ptr<A[]>` the dialect infers, so the pointer's scope end
  destroys the elements too.
- **The full gate caught one contract, and that is the round's process lesson.** `ObjectArray` in
  `cppstd.objects.__all__` broke `tests/test_cppstd_objects.py`'s rule that every name there is one the legacy runtime
  re-exports (`xrdroot.cint.runtime.objects` has no `ObjectArray`) - 1 failed, 13 024 passed. It is a runtime data
  type, not a name a translation writes, so it stays out of `__all__`. Then part 2 of the gate named **one uncovered
  line** (`emit_calls.py:265`, the arithmetic `make_unique<int[]>` arm) that no unit test reached although the target
  program uses it - a second test covered it. Both are `W0-63` and `W0-112` in one round.
- Queued: **`W0-139`** (`ntuple_extended` 133 s, `testBulkApiMultiple` TIMEOUT - the bulk-fill pathology near the
  wall) and **`W0-140`** (an array of *arrays* of objects is still not destroyed; one dimension is done).
- Worktree discipline (`W0-138`): the round opened `/tmp/ralph-round-202611220140/main` off `feat/root-testsuite`'s
  tip, integrated and gated there, landed the commits on `feat/root-testsuite` by fast-forward, and removed the
  worktree and its branch. **`W0-138`'s question - whether `main` is meant to be the trunk - is still open**, so
  nothing was pushed.
- Gates: **`TOTAL 119714 0 30014 0 100%`** by the two-command decomposition (13 026 passed, then 44) at a load
  average of 12-31; ruff clean, mypy 1023 files, maintainability 31 274 functions, components 1023 modules /
  0 undeclared.

## Round 57's close (recorded here for the next iteration)

- Base `44eb8a61`, closed at **`b42926d0`**. **One commit, 0 harness PASS**, and two programs' walls removed: the
  round is the *measured-refusal* kind `W0-55` accepts, not the PASS kind.
- **`b42926d0`: a ROOT number type written as a function is the C++ cast.** `TFormula` compiles its expression as
  C++, so `UInt_t(3.7)` is a cast - ROOT answers 3, `UInt_t(-1)` is 4294967295 and `Short_t(70000)` is 4464 -
  and this kit's `TFormula` knew `TMath`, `<cmath>` and the `ROOT::Math` densities and **no type at all**.
  `function/library.py` now builds a `Call` for every name in `formula.functions.CASTS` (the table the tree-draw
  engine already converts by), and `tmva/fdaformula.py` takes the same table because TMVA hands its split
  expression to its own parser rather than to `TFormula`. A conversion has no slope to give, so no `partials` is
  declared and `TF1` differentiates such a formula numerically - which the gate's coverage report is what made me
  notice: the `_level` I had written first was **never reached**, so it came out.
- **Why it matters, and the measurement that shaped it**: TMVA's own default split expression is
  `UInt_t([EventNumber])%UInt_t([NumFolds])`, so *every* cross-validation was refused at booking.
  `gtest/tmva/tmva/crossvalidation/testCrossValidationIntVar` and `...MultiProc` now run **4.6 s and 2.8 s of real
  cross-validation** where they stopped in 19 ms, and both arrive at `ROOT has cv_milli_t` (`W0-141`).
- **`W0-142` records a trap I nearly walked into.** ROOT's standalone `TFormula` answers 3 for `Int_t(7)/2` and
  this kit 3.5, which looks like a wide integer-division gap. It is not: **`TTree::Draw` really does evaluate in
  double** - measured against the oracle (`/tmp/r57/draw3.C`): `Draw("a/b")` on `int` branches gives 3.5, and
  `Draw("Int_t(a)/Int_t(b)")` crashes ROOT's own formula. The formula engine's declaration that "ROOT evaluates
  in double" is right for its domain, and the typing gap belongs to the standalone `TFormula` alone.
- Nothing lost: `gtest/hist/*` + `gtest/tmva/*` - 66 tests, **27 PASS / 33 FAIL / 6 UNSUPPORTED** (`/tmp/r57/areas`),
  the same set as `/tmp/r56/gtest`.
- Worktree discipline (`W0-138`): opened `/tmp/ralph-round-202611220240/main` off `feat/root-testsuite`'s tip,
  gated there, fast-forwarded `feat/root-testsuite`, tore the worktree and branch down. **`W0-138` is still open**
  - whether `main` is meant to be the trunk - so nothing was pushed. Two stray directories (`cv-multiproc/`,
  `dataset/`) that running the TMVA tests wrote into the worktree were removed before the gate (`W0-61`).
- Gates: **`TOTAL 119732 0 30014 0 100%`** by the two-command decomposition (13 036 passed, then 44) at a load
  average of 14-25; ruff clean, mypy 1023 files, maintainability 31 283 functions, components 1023 modules /
  0 undeclared.

## Round 58's close (recorded here for the next iteration)

- Base `1fc32254`, closed at **`ad8c97a8`**. **One commit, 0 harness PASS, 0 lost** - and **the whole gtest kind
  re-measured: 61 PASS / 211 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r58/gtest`, `--jobs 3`), identical
  to round 56's, which is the `W0-63` check across rounds 57 and 58.
- **`ad8c97a8` closes `W0-140`**: round 56's `ObjectArray` was built for one dimension, so `A a[2][3]` kept plain
  lists for its rows and destroyed nothing. `_filled` makes every level an `ObjectArray`, so the outer one destroys
  its rows and each row its elements - four `A`s for `A a[2][2]`, last first. `_owned` therefore has no `None` case
  left, and **the gate's uncovered-line report is what showed the dead branch**, so it came out. The test in
  `tests/test_cint_scope_destructors.py` that pinned the *gap* now measures the four destructions.
- **The round's finding is a queue row, not a PASS**: `gtest/tree/dataframe/dataframe_take`'s `VectorBool` is one
  case from a PASS and its cost is one *type* - `Take<std::vector<bool>>` gives a `Jagged` whose rows are NumPy
  slices, so neither level answers `size()` where ROOT's `std::vector<std::vector<bool>>` answers it twice. **The
  requested C++ type is already in the emitted call** (`df2.Take['std::vector<bool>']('x')`) and `frame.Take`
  drops it, which is where `W0-143` starts. `W0-144` records `testTMultiGraphGetHistogram`'s log-scale range with
  its four numbers, and says the source must be read before anything is changed.
- Worktree discipline (`W0-138`): opened `/tmp/ralph-round-202611220314/main` off `feat/root-testsuite`'s tip,
  gated there, fast-forwarded `feat/root-testsuite`, tore the worktree and branch down. **`W0-138` is still open**,
  so nothing was pushed. Two `.root` files my probes wrote into the worktree were removed before the gate (`W0-61`).
- Gates: **`TOTAL 119730 0 30012 0 100%`** by the two-command decomposition (13 036 passed, then 44) at a load
  average of 11-17; ruff clean, mypy 1023 files, maintainability 31 282 functions, components 1023 modules /
  0 undeclared.

## Round 59's close (recorded here for the next iteration)

- Base `74caab18`, closed at **`0c58dd9e`**. **One commit, +1 gtest PASS (61 -> 62)**, and it is round 58's queued
  row `W0-143` taken exactly as that row described it.
- **`0c58dd9e`: `Take` of a column of vectors is a vector of vectors.** `Take<V>` gives `std::vector<V>`, so
  `Take<std::vector<bool>>("x")` over a column of `std::vector<bool>` is a **vector of vectors** - which is where
  `vec.size()` and `vec[0].size()` both answer. Two things were missing and the second is the interesting one:
  `_Method.__getitem__` returned `self` with "a column's type is read from the data" (true for a scalar column, not
  for one of vectors), **and a row of a `Jagged` is a NumPy array whose `size` is a number and not a call**, so a
  vector of vectors cannot be a `Jagged` at all. `_Method` now carries the arguments, `RResultPtr` keeps them and
  `_vector_of` builds `std::vector<std::vector<T>>` - one `std::vector` per row, the element type from the row's own
  dtype. **`gtest/tree/dataframe/dataframe_take` PASSES, all three of its cases.**
- **The measurement is the safety net, not the idea**: 109 `gtest/tree/{dataframe,ntuple,treeplayer}/*` programs
  measured before and after - **4 PASS / 75 FAIL / 29 UNSUPPORTED / 1 TIMEOUT** (`/tmp/r59/df`), `dataframe_take`
  the only gain and **nothing lost**. The RDF unit tests (664 with `-k "rdf or deframe or dataframe"`) pass except
  the known `ResourceWarning` flake `W0-117`.
- **`W0-145` records the nuance the fix leaves**: `_vector_of` builds its rows from the *row's* dtype, so
  `Take<std::vector<double>>` over a `float` column gives `std::vector<float>` rows where ROOT gives
  `std::vector<double>`. Nothing in the suite asks for that, and the 109 programs say so; it is a fidelity gap to
  close when a test wants it.
- Worktree discipline (`W0-138`): opened `/tmp/ralph-round-202611220350/main` off `feat/root-testsuite`'s tip, gated
  there, fast-forwarded `feat/root-testsuite`, tore the worktree and branch down. **`W0-138` is still open**, so
  nothing was pushed.
- Gates: **`TOTAL 119736 0 30014 0 100%`** by the two-command decomposition (13 037 passed, then 44) at a load
  average of 15-20; ruff clean, mypy 1023 files, maintainability 31 283 functions, components 1023 modules /
  0 undeclared.

## Round 60's close (recorded here for the next iteration)

- Base `0e88415c`, closed at **`6e981a56`**. **One commit, +1 gtest PASS (62 -> 63)**, after two targets were
  measured and **dropped for good reasons** - which is the round's other half.
- **`6e981a56`: `TServerSocket`.** ROOT's `net/net` module lets a program open a port for itself -
  `TServerSocket(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback)` - and this kit
  had none of it, so `gtest/net/net/nettests` was refused at the socket. It is Python's `socket` behind ROOT's names,
  which is all ROOT's is: `TInetAddress` answers `GetHostAddress`/`GetHostName`/`GetPort`/`IsValid`
  (`127.0.0.1`, `localhost`, the port, true), a default bind gives `0.0.0.0`, a negative port binds nothing as ROOT's
  does, and `Close` gives the port back - it is a dialect **guard**, so a local socket is closed where its scope ends.
  The numbers are the oracle's: `kDefaultBacklog` 10, `kInaddrAny` 0, `kInaddrLoopback` 1. **`nettests` PASSES.**
- **Two targets measured and dropped, each with its reason in the queue**:
  `ntuple_largefile2` writes **268 435 456** entries to exceed 2 GB (`W0-146`) - the two-part null fix that would get
  past its first wall was written, **measured, and reverted**, because it turns FAIL into TIMEOUT rather than PASS,
  and its shape is recorded in `W0-149` for whoever needs a nulled value-model local; and
  `ROOT::Math::Delaunay2D::Interpolate` is **not** barycentric interpolation of z (`W0-147`) - scipy reproduces 12 of
  its 15 values exactly and none of the three on the diagonal edge, so ROOT's source must be read first.
- **`W0-148` is a scoping question the round measured**: `gROOT.GetName()` is `PyROOT` here and `root` in a compiled
  ROOT program (an interactive `Rint` on the oracle, `/tmp/r60/g.C`), so `TEntryList::Enter`'s prefix disagrees with
  `entrylist_enter`'s `GetEntryList("root/t1","")`; three unit tests pin `PyROOT`.
- Nothing lost: 71 `gtest/net/*` + `gtest/core/*` + `gtest/tree/tree/*` programs, **21 PASS / 35 FAIL /
  14 UNSUPPORTED / 1 TIMEOUT** (`/tmp/r60/net`), the same set as round 58's.
- Worktree discipline (`W0-138`): opened `/tmp/ralph-round-202611220414/main` off `feat/root-testsuite`'s tip, gated
  there, fast-forwarded `feat/root-testsuite`, tore the worktree and branch down. **`W0-138` is still open**, so
  nothing was pushed.
- Gates: **`TOTAL 119809 0 30022 0 100%`** by the two-command decomposition (13 045 passed, then 44) at a load
  average of 13-15; ruff clean, mypy 1024 files, maintainability 31 307 functions, components 1024 modules /
  0 undeclared.

## Round 61's close (recorded here for the next iteration)

- Base `24cfc1ea`, closed at **`efe8b9de`** (code at `1a7e31ca`). **Two gtest PASSes, the kind 63 -> 65**, and the
  whole gtest kind re-measured at the round's tip: **65 PASS / 206 FAIL / 77 UNSUPPORTED / 2 TIMEOUT** of 350
  (`/tmp/r61/gtest`, `--jobs 3`, `--xrdroot-src /tmp/r61/head/src` frozen at `1a7e31ca`). Four gained over round 58's
  61 - `dataframe_take` (r59),`nettests` (r60), and this round's `testRooExtendedBinding` and `testKNNDensity` - and
  **none lost**.
- **`730bb942` (worker w01): `RooExtendedBinding`.** A `RooAbsReal` holding one density and optionally a
  normalisation set; its value is `pdf.expectedEvents(normSet)`, so 1.0 without a set and the density's integral over
  the observables with one. The numbers were checked in plain Python first (`RooRealSumPdf` of a `RooUniform` answers
  1.0 and 20.0), which is why this was a forty-minute target rather than a round: the engine already had every part
  but the class. `gtest/roofit/roofitcore/testRooExtendedBinding` PASSES (`/tmp/r61/reb`).
- **`618f5a98`: `std::span` and `TMath::KNNDensity`.** ROOT's density takes three spans; here a span holds *what
  `&x` makes in a translated program*, a `Cell`, so `result[i] = ...` writes the caller's own `double` - which is the
  whole point of the signature. The k-nearest-neighbour walk is ROOT's own, transcribed from
  `math/mathcore/src/TMath.cxx` (fetched into the sparse checkout this round), and it agrees with the 6.40 oracle
  **bit for bit** on six numbers measured with `/tmp/r61/work-knn/knn.C`. `gtest/math/mathcore/testKNNDensity` PASSES
  (`/tmp/r61/knn2`).
- **Two translator behaviours came out of the same test, and both were measured against the oracle** (`/tmp/r61/work-knn/tc.C`):
  `std::size_t(3.7)` is 3, `uint8_t(300)` is 44, `std::size_t(-1)` is 18446744073709551615 - a standard typedef
  written as a call is the scalar cast, not a name of the runtime (`9d380d3f`); and a braced list handed to a
  `std::vector` parameter is the container C++'s `initializer_list` constructor makes.
- **The round's own gate caught the second one being in the wrong place**, which is the round's most useful hour:
  writing that conversion into *every* function's prologue broke 53 tests in `tests/test_rdf_masked_call.py`, because
  the batched-call contracts look for a wrapper's single `return` statement. It was narrowed to a **lambda's** body
  (`a006df48`) - a lambda's call site carries no types, a function's does - and the function case is queued as
  `W0-151` with the call-site shape named. The gate also named `_wrap_parameters` at CCN 11 and `KNNDensity` at CCN 14,
  cognitive 25, NPath 2192; both were split rather than excused.
- **`W0-150` is the round's sharpest finding** and it is about measuring, not about ROOT: `tools/rootsuite/baseline.json`
  still says `FAIL` for **eight** gtest programs that pass, because `ratchet.write_baseline` refuses anything but a
  full four-kind run and no round can afford one. The suite's promise list therefore understates it by every gain
  since round 43. A `--merge` mode that records only the PASSes a real run saw is the fix, and the round's own
  `/tmp/r61/gtest` already holds the rows.
- **`W0-152` is the `W0-63` check done on round 60's work**: the two programs whose wall `TServerSocket` removed did
  not pass, they moved on - `RCurlConnection` for `curl_connection`, and a UNIX-domain socket *path* read as a port
  for `testParallelMergingFile` (a small feature of `pyroot/core/net.py`, queued with the measured message).
- **`W0-153` cost twenty minutes**: `git worktree add /tmp/r61/w01 feat/root-testsuite` gives the same branch to two
  worktrees, so the worker's commit moved the branch under the main tree's index, which then showed the worker's patch
  as a *staged revert*. Recovered with `git reset` + `git checkout -- <the worker's paths>`; **use `git worktree add -d`**.
- Worktree discipline (`W0-138`): the round worked off `feat/root-testsuite`'s tip, gated there, and **pushed nothing** -
  `W0-138` is still the operator's decision.
- **The roottest risk was reasoned down to one test and then measured**: a lambda whose parameter is a standard container *by value* is the only shape the conversion can touch, and a grep of `roottest/`, `test/` and `tutorials/` finds exactly three sources - `roottest/root/multicore/`'s `tSequentialExecutor.cpp`, `tThreadExecutor.cpp` and `executorTests.hxx`. The only one with a PASS at stake is `roottest/root/multicore/tSequentialExecutor-build`, and it still PASSES (`/tmp/r61/seq`). The other 30 matches are references, which `_receives_a_list` excludes.
- Evidence kept: `/tmp/r61/{gtest,knn2,reb,net2,pmf,romb-base,base,head,work-knn,seq,gate1.log,gate1b.log,w01*,w01-brief.md}`.

## Round 62's close (recorded here for the next iteration)

- Base `62c38856`, closed at **`78252929`** (code at `9fe62239`). **Two gtest PASSes, the kind 65 -> 67**, and the whole
  gtest kind measured at the tip: **67 PASS / 204 FAIL / 77 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r62/gtest`,
  `--jobs 3`, `--xrdroot-src /tmp/r62/head/src` frozen at `9fe62239`) - the two gained, **none lost and not one other
  row moved**.
- **`a911f526` is a general translator bug found by pricing a target, not by reading a queue row**: `_owned` answered a
  smart pointer's *own* type for a single-object pointee, so `std::unique_ptr<TFile> f{TFile::Open("p1.root","RECREATE")}`
  was never destroyed where its scope ended and the file was written at process exit. A program that writes a file and
  reads it back in the same run therefore read one that was not there yet: `file p1.root is not a ROOT file`, measured
  before (`open=0 get=0`) and after (`open=1 get=1`) in `/tmp/r62/probe/p.C`. `testHistFactoryPlotting` moved 1100 lines
  for it (`W0-155` names what is left), and the round's own gate caught its one casualty - the translator's fake
  `TFile` had no `_destruct` (`9fe62239`, `W0-159`).
- **`0b11a27c` (worker r-62-w01) settles `W0-148`, which round 60 left as a scoping question**: a *translated program* is
  the compiled program's kin, so `xrdroot run FILE` runs under `gROOT` named `root` - which is what `TEntryList::Enter`
  prefixes a sub-list with, and why `entrylist_enter`'s `GetEntryList("root/t1","")` missed - while the prompt's `TRint`
  session keeps `Rint` (the oracle's answer) and an import of `xrdroot.pyroot` keeps `PyROOT`, so the four unit tests
  that pin it still pass. **`gtest/tree/tree/entrylist_enter` passes, all four cases.**
- **`35a387d1` is the cheapest PASS of the recent rounds**: `RooBDecay::DecayType::SingleSided` (0), `DoubleSided` (1),
  `Flipped` (2) - measured on the oracle, which also answers the enumerators bare because the enum is unscoped, and
  `RooAbsAnaConvPdf` has none - was the *only* wall in `testRooTruthModel.IntegrateSubrange`, whose whole subject (a
  `bcpg` integrating to 0 over a range `[2,2]`) already worked. `8652fbcb` moved the declaration into one typed helper
  after mypy objected to assigning through a union of class types.
- **`W0-150` is closed, and it was the round's measuring fix**: `rootsuite baseline RUN/results.json --merge` records the
  rows a real run saw as PASS and leaves every other row exactly as it was, so a round can write down what it gained
  without a many-hour run of all four kinds. `write_baseline` keeps its full-run refusal and `regressions()` is
  untouched. Its first two real uses recorded **13** gtest rows from `/tmp/r61/gtest` and this round's two from
  `/tmp/r62/gtest`, with `ratchet` reporting 0 regressions on each.
- **Two of the round's four "fixes" are measured refusals that moved on**, and the queue says so: `TServerSocket`'s local
  socket (`085bc619`, every answer the oracle's: valid, port -1, `0.0.0.0`, `UnknownHost`, window 0) puts
  `testParallelMergingFile` at `TServerSocket.Accept` (`W0-156`); and `TBuffer::ResetMap` (`72614fe5`) is the first of
  the three lines `testTF1Simple` writes, the read of a `TF1` back out of a buffer being `W0-157`.
- **`W0-158` is the round's measurement trap, and it nearly read as a regression**: round 62's roofit subset showed
  `testHistFactory` FAIL -> UNSUPPORTED. It was the *harness*: a `--only` run's working copy holds only the directories
  it was asked for, so `#include "../../roofitcore/test/gtest_wrapper.h"` - the header that defines
  `ROOFIT_EVAL_BACKENDS` - is missing, and the program refuses at translation. Proven at the base with the same command
  (`/tmp/r62/hf-base`), and the whole-kind run at the tip reads FAIL again. **A subset comparison is a subset
  comparison.**
- **Two harness/process notes**: `tests/test_pyroot_distrdf_dask.py`'s seven errors in the first gate run were dask's
  `LocalCluster` failing to start under a load average above 20 - it passes alone (14 passed) - and w01 was taken from
  its *working tree* after 45 minutes with no commit (`W0-111`), its five files copied and its work credited in
  `0b11a27c`.
- Worktree discipline (`W0-138`): two worktrees, both **detached** (`git worktree add -d`, `W0-153`), integrated by
  cherry-pick or by copying files, and **nothing pushed** - `W0-138` remains the operator's decision.
- Evidence kept: `/tmp/r62/{gtest,truth2,ele,tree65,roofit,areas,hf-base,hfp,pmf2,probe,head,gate1.log,gate1b.log,w01*,w02*,w0?-brief.md}`.

## Round 63's close (recorded here for the next iteration)

- Base `86001091`, closed at **`1300854d`** (code at `564bbb88`). **Two gtest PASSes, the kind 67 -> 69**, and the whole
  kind measured at the tip: **69 PASS / 202 FAIL / 77 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r63/gtest`, `--jobs 3`,
  `--xrdroot-src /tmp/r63/head/src` frozen at `564bbb88`) - the two gained, **none lost and not one other row moved**.
- **`64cf43ae` is the round's general fix, and it came from reading the oracle instead of guessing**: `TFile::Write` in
  ROOT calls `TDirectoryFile::Write`, `WriteStreamerInfo`, `WriteFree` and `WriteHeader` (`io/io/src/TFile.cxx:2489`,
  fetched into the sparse checkout this round), so a file is on disk and readable the moment it is written. Ours wrote
  nothing until the close, so a program that writes a file and reads it back in the same run read one that was not
  there: ROOT's own `w.C` probe answers `open=1 tree=1` and we answered `open=0 tree=0`. **`WritableFile.checkpoint`**
  is that sequence, made repeatable - `_finish_containers` writes each tree's and ntuple's record once, and the streamer
  info only when a class new since the last record has been used. The round's own gate caught a 105-test break on the
  first attempt (`UpdatedFile` has no `_start`-set attributes; a memory directory has no file), the NPath limit caught
  `_assemble` at 256 (split into `_finish_containers`), and part 2 named two uncovered branches (`checkpoint`'s
  `_closed` guard, which no caller can reach, and the ntuple loop, which an RNTuple in a `TFile` now covers).
- **The two worker targets were both priced from ROOT's own source before a brief was written**, and both landed:
  - **`85fb85eb` (w01): GDML names.** `GetMaterial("Iron")` found nothing because a GDML file names its materials
    `Iron0xccc` for XML uniqueness and ROOT strips the trailing `0x…` when it registers them; and ROOT names an element
    by its **formula** (`GetElement(0)->GetName()` is `"FE"`). `TGeoMixture` now keeps its elements, so `GetWmixt()` and
    `GetElement(i)` answer as ROOT does. **`gtest/geom/gdml_material_name_collision` PASSES.**
  - **`d28fcf12` (w02): `TMultiGraph::GetHistogram`'s log frame.** ROOT pads a log axis by *the logarithm of the ratio* -
    `min/(1+0.5*log10(r))`, `max*(1+0.2*log10(r))` (`hist/hist/src/TMultiGraph.cxx:1035`) - and `TGraph::ComputeRange`
    takes the smallest **positive** minimum on a log axis (`TGraph.cxx:759`). The test's numbers (0.65234 and 79.9602)
    are now ours to 1e-3, where the round began at 1.08468 and 65.5672. **`testTMultiGraphGetHistogram` PASSES.**
- **`W0-160` is the moved wall**: with the file written, `testTChainSaveAsCxx` reaches the macro its `SaveAs` wrote and
  stops on our own sentence `'testtree' was read from a file, and WriteObject adds to a tree being filled; CloneTree(0)
  makes one with its branches to fill` - the wall is now a feature the kit names rather than a file that was not there.
- **`W0-161` records what the `Write` checkpoint does to the bytes**, because the file-size work of rounds 3-6 depends
  on it: ROOT's file grows by 274 bytes for one `Write` and 399 for two, ours by 442 and 489 (`/tmp/r63/probe/size1.C`).
  The *shape* is ROOT's - a second cycle per object per write, which ROOT does too (`/tmp/r63/probe/keys.C` lists `h`
  at cycles 3, 2, 1) - but the sizes differ, and the streamer-info record is the first suspect.
- **Measured areas, before and after**: 38 programs of `gtest/geom/*` + `gtest/hist/hist/*` give **11 PASS / 25 FAIL /
  2 UNSUPPORTED** (`/tmp/r63/areas`), the two gains and nothing else moved; 31 programs of `gtest/io/io/*` +
  `gtest/tree/tree/*` give **7 / 18 / 5 / 1 TIMEOUT** (`/tmp/r63/io`), unchanged, so the checkpoint costs no PASS.
- Worktree discipline (`W0-138`): two worktrees, both **detached** (`git worktree add -d`, `W0-153`), integrated by
  cherry-pick `-x`, and **nothing pushed** - `W0-138` remains the operator's decision.
- Evidence kept: `/tmp/r63/{gtest,areas,io,t-gdml,t-mg,chainsave,probe,head,gate1.log,gate2.log,gate3.log,w01*,w02*,w0?-brief.md}`.

## Round 64's close (recorded here for the next iteration)

- Base `fd36b30f`, closed at **`b6a93b06`** (code at `3d1aee19`). **Three gtest PASSes, the kind 69 -> 72**, and the whole
  kind measured at the tip: **72 PASS / 200 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r64/gtest`, `--jobs 3`,
  `--xrdroot-src /tmp/r64/head/src` frozen at `3d1aee19`) - and one program moved `UNSUPPORTED -> FAIL`
  (`testMapCppName`, which translates and runs now and stops on its own subject).
- **Two of the three PASSes are one general fix each, found by reading the test rather than the queue**:
  - **`d192a920`: a `std::ofstream` is closed as its scope ends.** C++'s stream writes through a buffer that its
    destructor closes; ours had neither destructor nor guard, so a file written in one scope was empty until the
    process ended - the oracle prints `read=[hello]` for the probe where this kit printed `read=[]`. The dialect's
    guards name `ofstream`, so the local's scope end calls the destructor.
  - **`c9476ae6`: `#pragma extra_include`.** rootcling reads the named headers into a dictionary and the compiler pass
    never sees them; a library build here is the one pass both share, so the header is translated beside the macro.
    `dictgen_base`'s `f()` is in `myextra.h` and its macro never includes it. **`gtest/core/dictgen/dictgen_base`
    PASSES.**
- **`500efb6a` (worker r-64-w01)**: `std::thread` as a `cppstd` class - `join`/`joinable`/`detach`, the static
  `hardware_concurrency`, and `std.thread` in the namespace - which is what the fixture's
  `ROOT::EnableImplicitMT(std::min(4u, std::thread::hardware_concurrency()))` asked for. **`test_norootextension`
  PASSES, both the Seq and the MT case** (the MT instantiation is live here because the dialect defines `R__USE_IMT`).
- **`2dcebb86` (worker r-64-w02) and the patch it measured but could not commit**: `TObject::Streamer(TBuffer&)` and
  the record read finished in the object it was handed. Its own discovery record (`W0-88`'s rule, read at the close)
  named the last piece exactly - `(char *)f.IsA()->New()` has no known return type, so the cast was `cstr` and
  `p + off` sliced a `str`; it proposed `char_pointer`/`moved_pointer`. **The first shape of that patch broke five
  unit tests** - moving every `char *` offset into a call is not an assignment target, and `sprintf(buf + n, ...)`
  writes through one - so the round settled on `ObjectAddress`: a `char *` of an object holds an address whose
  `p[off:]` is the object moved by that offset, and `p + n` stays a slice everywhere else (`724be159`).
  **`gtest/hist/hist/testTF1Simple` PASSES.**
- **`W0-162` is the round's process finding and it is about evidence**: **the machine rebooted and `/tmp` was wiped** -
  ROOT's sources (`/tmp/rootsrc-6.40b`, which rounds 10-63 measured against) and every round directory (`/tmp/r1` …
  `/tmp/r64`). The round re-cloned `/tmp/rootsrc-6.40` with `AGENTS.md`'s own recipe and confirmed
  `tools.rootsuite list` gives the same 2820 non-tutorial tests, so the counts stay comparable, then re-measured the
  three targets and the whole kind. **A round's `results.json` under `/tmp/rN` is not durable**; the durable record is
  `TODO.md`'s ledger, which quotes the numbers.
- **`W0-163` is a wall the round measured without chasing**: `test_THBinIterator` takes 85-170 s of its 300 s limit
  (the spread is the machine, measured at three commits) and 5 of its 20 cases refuse `THistRange`
  (`test_THBinIterator.cxx:121`). Two runs TIMEOUTed under a load burst and passed the check alone - `W0-45`/`W0-103`
  again, and the reason the A/B was run at three commits before believing a slowdown.
- **Measured areas, before and after**: 46 programs of `gtest/core/*` + `gtest/roottest/*` give the two gains there and
  nothing else moved (`/tmp/r64/areas`, before the wipe); 33 of `gtest/hist/hist/*` give **11 PASS / 19 FAIL / 2
  UNSUPPORTED / 1 TIMEOUT** with `testTF1Simple` the only gain (`/tmp/r64/histareas`).
- Worktree discipline (`W0-138`): two worktrees, both **detached** (`git worktree add -d`, `W0-153`), integrated by
  cherry-pick `-x` plus one patch the worker measured and reported, and **nothing pushed** - `W0-138` remains the
  operator's decision.
- Evidence kept: `/tmp/r64/{gtest,areas,histareas,t-gtest*,gate1.log,gate3.log,gate4.log,w0?-brief.md}` (the earlier
  `/tmp/r64/tf1*`, `thbin*` runs and the workers' trees went with the wipe).

## Checklist (per round)

- [x] `git status` clean; the gates green at the round's base commit
- [x] the queue read and the 35 one-failing-case programs priced from `/tmp/r63/gtest` - **before the reboot that wiped
      `/tmp`**, which is why the ledger's numbers were written down as they were measured (`W0-162`)
- [x] **three targets, disjoint files**: `std::ofstream` + `#pragma extra_include` (found by probing ROOT's own
      `dictgen_base`), `std::thread` (worker w01), `TBufferFile`'s object read (worker w02)
- [x] two worktrees created **detached** (`git worktree add -d`, `W0-153`) and two briefs, each naming the test and
      the wall
- [x] both workers' commits cherry-picked `-x`, plus **the patch w02 measured and could not commit**, applied here and
      refined when five unit tests caught its first shape
- [x] **every branch that loses a PASS rejected** - none lost; the whole gtest kind at the tip (72 / 200 / 76 / 2,
      `/tmp/r64/gtest`), and the areas measured on their own (`/tmp/r64/areas`, `/tmp/r64/histareas`)
- [x] every ROOT number claimed measured on the oracle first (`read=[hello]`, and the ACLiC probe's `f()`)
- [x] `TODO.md` §0 and the ledger row in one commit at the close, and the **baseline rows raised from the round's own
      whole-kind run** with `rootsuite baseline --merge` (`b6a93b06`, `ratchet` 0 regressions on the same run)
- [x] **every extra thing the round found added to §4's queue** (`W0-162`, `W0-163`)
- [x] the round's summary written into the loop's task file (`.ralph/<name>.md`)
- [x] gates re-run at the round's final commit (four runs: the first two named the round's own breakages, the third a
      parallel-run flake that the fourth and a solo run agree is not)
- [x] commits made; nothing pushed unless asked
- [x] the round directory kept - as much of it as survived the wipe; the numbers themselves are in `TODO.md`

## Verification

For the round's work, the evidence is, in this order:

1. `pytest -q -p no:cacheprovider -n 4 --cov=xrdroot` → `TOTAL … 100%` and no failure
2. the same with `XRDROOT_TEST_OTHER_MACHINE=1`
3. `ruff check src tests tools` → `All checks passed!`
4. `mypy` → `Success: no issues found in … source files`
5. `python tools/maintainability.py check` → `… functions in … files pass`
6. `python tools/components.py check` → `… imports; 0 undeclared imports`
7. `mkdocs build --strict` → `Documentation built`
8. `<round>/final2/summary.md` and `<round>/final1/summary.md`: the per-kind counts, compared with
   the previous round's, with every gain and every loss named

## Final verification

- Round 43's result of them, at `a57e2d05`: **`TOTAL 119167 0 29918 0 100%`** - 12 954 passed then 44 passed - and
  ruff clean, mypy 1019 files, maintainability 31 089 functions, components 1018 modules / 0 undeclared,
  `mkdocs build --strict` built. `/tmp/r43/gate1.log` is the first command's output.
- Exact monitor-rerunnable commands (`W0-106`'s decomposition; run them in this order, from
  `/Users/rcurrie/src/xrdroot`, with `export PYTHONPATH=$PWD/src`, after `rm -f .coverage`):
  ```
  .venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --cov=xrdroot --cov-report= \
      --ignore=tests/test_tmva_learn_networks.py --ignore=tests/test_pyroot_references.py
  .venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_tmva_learn_networks.py tests/test_pyroot_references.py --cov=xrdroot --cov-append
  ```
  The first reports ~99.73% by design (the ignored files' code is not covered yet) and the second
  reaches 100%. Then `ruff check src tests tools`, `mypy`, `python tools/maintainability.py check`,
  `python tools/components.py check`, `mkdocs build --strict`.
- Round 51's result of them, at `4227df45`: **`TOTAL 119561 0 29992 0 100%`** - 12 983 passed then 44 passed - at a
  load average of 19-26; ruff clean, mypy 1020 files, maintainability 31 217 functions, components 1019 modules /
  0 undeclared. `/tmp/r51/gate1.log` is the first command's output.
- Round 52's result of them, at `5f58c997`: **`TOTAL 119625 0 29998 0 100%`** - 12 990 passed then 44 passed - at a
  load average of 24-33; ruff clean, mypy 1021 files, maintainability 31 245 functions, components 1020 modules /
  0 undeclared. `/tmp/r52/gate1.log` is the first command's output.
- Round 53's result of them, at `f93144fd`: **`TOTAL 119666 0 30000 0 100%`** - 13 009 passed then 44 passed - at a
  load average of 20-42; ruff clean, mypy 1022 files, maintainability 31 252 functions, components 1021 modules /
  0 undeclared, `mkdocs --strict` built. `/tmp/r53/gate1.log` is the first command's output (the tree was committed
  exactly as measured, in the two commits above).
- Round 54's result of them, at `2ce1f188`: **`TOTAL 119669 0 30000 0 100%`** - 13 017 passed then 44 passed - at a
  load average of 10-78; ruff clean, mypy 1022 files, maintainability 31 263 functions, components 1021 modules /
  0 undeclared. `/tmp/r54/gate1.log` is the first command's output (the tree was committed exactly as measured).
- Round 55's result of them, at `d1752d03`: **`TOTAL 119689 0 30000 0 100%`** - 13 020 passed then 44 passed - at a
  load average of 13-17; ruff clean, mypy 1023 files, maintainability 31 266 functions, components 1023 modules /
  0 undeclared. `/tmp/r55/gate1.log` is the first command's output, run in the round's own worktree
  (`/tmp/ralph-round-202611220111/main`, since removed - the commit is on `feat/root-testsuite`).
- Round 56's result of them, at `4f884ec7`: **`TOTAL 119714 0 30014 0 100%`** - 13 026 passed then 44 passed - at a
  load average of 12-31; ruff clean, mypy 1023 files, maintainability 31 274 functions, components 1023 modules /
  0 undeclared. `/tmp/r56/gate3.log` is the first command's output (an earlier run, `/tmp/r56/gate1.log`, was the one
  that failed on `cppstd.objects.__all__` and named the uncovered line), run in the round's own worktree
  (`/tmp/ralph-round-202611220140/main`, since removed - the commit is on `feat/root-testsuite`).
- Round 57's result of them, at `b42926d0`: **`TOTAL 119732 0 30014 0 100%`** - 13 036 passed then 44 passed - at a
  load average of 14-25; ruff clean, mypy 1023 files, maintainability 31 283 functions, components 1023 modules /
  0 undeclared. `/tmp/r57/gate2.log` is the first command's output (an earlier run, `/tmp/r57/gate1.log`, named the
  one uncovered line - a `_level` partial I had written and never reached), run in the round's own worktree
  (`/tmp/ralph-round-202611220240/main`, since removed - the commit is on `feat/root-testsuite`).
- Round 58's result of them, at `ad8c97a8`: **`TOTAL 119730 0 30012 0 100%`** - 13 036 passed then 44 passed - at a
  load average of 11-17; ruff clean, mypy 1023 files, maintainability 31 282 functions, components 1023 modules /
  0 undeclared. `/tmp/r58/gate2.log` is the first command's output (an earlier run, `/tmp/r58/gate1.log`, named the
  dead `None` branch that `_owned` could no longer answer), run in the round's own worktree
  (`/tmp/ralph-round-202611220314/main`, since removed - the commit is on `feat/root-testsuite`).
- Round 59's result of them, at `0c58dd9e`: **`TOTAL 119736 0 30014 0 100%`** - 13 037 passed then 44 passed - at a
  load average of 15-20; ruff clean, mypy 1023 files, maintainability 31 283 functions, components 1023 modules /
  0 undeclared. `/tmp/r59/gate1.log` is the first command's output, run in the round's own worktree
  (`/tmp/ralph-round-202611220350/main`, since removed - the commit is on `feat/root-testsuite`).
- Round 60's result of them, at `6e981a56`: **`TOTAL 119809 0 30022 0 100%`** - 13 045 passed then 44 passed - at a
  load average of 13-15; ruff clean, mypy 1024 files, maintainability 31 307 functions, components 1024 modules /
  0 undeclared. `/tmp/r60/gate1.log` is the first command's output, run in the round's own worktree
  (`/tmp/ralph-round-202611220414/main`, since removed - the commit is on `feat/root-testsuite`).
- Round 61's result of them, at `1a7e31ca` (the code; `efe8b9de` adds only `TODO.md`): **`TOTAL 119899 0 30046 0 100%`** - 13 073 passed then 44 passed - at a load average of 13-16; ruff clean, mypy 1026 files, maintainability 31 351 functions, components 1025 modules / 0 undeclared, `mkdocs --strict` built. `/tmp/r61/gate1b.log` is the first command's output (an earlier run, `/tmp/r61/gate1.log`, named round 61's own two regressions: `_wrap_parameters` at CCN 11 and `KNNDensity` at CCN 14 / cognitive 25 / NPath 2192, plus the 53 batched-call tests the prologue conversion broke). Run in the main tree, which is `feat/root-testsuite` itself (`W0-138`: nothing is pushed).
- Round 64's result of them, at `3d1aee19` (the code; `b6a93b06` adds the ledger and the baseline rows): **`TOTAL 120057 0 30074 0 100%`** - 13 106 passed then 44 passed - at a load average of 8-22; ruff clean, mypy 1026 files, maintainability 31 418 functions, components 1025 modules / 0 undeclared. `/tmp/r64/gate4.log` is the first command's output (an earlier run, `/tmp/r64/gate3.log`, failed one `tests/test_tutorial_harness.py` case that passes alone and in the rerun - a parallel-run flake). Run in the main tree, which is `feat/root-testsuite` itself (`W0-138`: nothing is pushed).
- Round 63's result of them, at `564bbb88` (the code; the docs commits follow): **`TOTAL 120003 0 30060 0 100%`** - 13 094 passed then 44 passed - at a load average of 16-28; ruff clean, mypy 1026 files, maintainability 31 394 functions, components 1025 modules / 0 undeclared. `/tmp/r63/gate3.log` is the first command's output (earlier runs named round 63's own breakages: `/tmp/r63/gate1.log` the NPath 256 in `_assemble`, `/tmp/r63/gate2.log` the two uncovered branches). Run in the main tree, which is `feat/root-testsuite` itself (`W0-138`: nothing is pushed).
- Round 62's result of them, at `9fe62239` (the code; `78252929` and `14b61d71` add the ledger and the baseline rows): **`TOTAL 119943 0 30054 0 100%`** - 13 085 passed then 44 passed - at a load average of 13-22; ruff clean, mypy 1026 files, maintainability 31 370 functions, components 1025 modules / 0 undeclared, `mkdocs --strict` built. `/tmp/r62/gate1b.log` is the first command's output (an earlier run, `/tmp/r62/gate1.log`, named the translator's fake `TFile` missing `_destruct` - fixed in `9fe62239` - and seven dask `LocalCluster` errors that pass alone). Run in the main tree, which is `feat/root-testsuite` itself (`W0-138`: nothing is pushed).
- Round 50's result of them, at `2e10bb4d`: **`TOTAL 119555 0 29992 0 100%`** - 12 982 passed then 44 passed - at a
  load average of 24-39; ruff clean, mypy 1020 files, maintainability 31 215 functions, components 1019 modules /
  0 undeclared, `mkdocs --strict` built. `/tmp/r50/gate1.log` is the first command's output.
- Round 49's result of them, at `3f3dae7f`: **`TOTAL 119540 0 29984 0 100%`** - 12 981 passed then 44 passed - at a
  load average of 21-23; ruff clean, mypy 1020 files, maintainability 31 213 functions, components 1019 modules /
  0 undeclared, `mkdocs --strict` built. `/tmp/r49/gate1.log` is the first command's output.
- Round 48's result of them, at `f16f5aeb`: **`TOTAL 119535 0 29984 0 100%`** - 12 980 passed then 44 passed - at a
  load average of 12-26; ruff clean, mypy 1020 files, maintainability 31 211 functions, components 1019 modules /
  0 undeclared, `mkdocs build --strict` built. `/tmp/r48/gate2.log` is the first command's output.
- Round 47's result of them, at `f0655cf0`: **`TOTAL 119506 0 29978 0 100%`** - 12 977 passed then 44 passed - at a
  load average of 13-21; ruff clean, mypy 1020 files, maintainability 31 201 functions, components 1019 modules /
  0 undeclared. `/tmp/r47/gate3.log` is the first command's output.
- Round 46's result of them, at `e142f240`: **`TOTAL 119286 0 29940 0 100%`** - 12 962 passed then 44 passed - at a
  load average of 16-20; ruff clean, mypy 1019 files, maintainability 31 126 functions, components 1018 modules /
  0 undeclared. `/tmp/r46/gate3.log` is the first command's output.
- Round 42's result of them, at `fd657322`: **`TOTAL 119070 0 29912 0 100%`** - 12 948 passed then
  44 passed - at a load average of 90-139, which is the highest a 100% has ever been measured at here.
  **The same two commands with `XRDROOT_TEST_OTHER_MACHINE=1` are green too** (12 948 then 44, 100%,
  `/tmp/r42/gate-other2.log`). ruff clean, mypy 1019 files, maintainability 31 064 functions in 1729
  files, components 1018 modules and 0 undeclared imports, `mkdocs build --strict` built.
  `/tmp/r42/gate-final-cov.txt` is the summary; `/tmp/r42/EVIDENCE.md` says where each number came from.
- Exact single command (the same gate in one process, which `W0-106` says fails on this machine):
  `cd /Users/rcurrie/src/xrdroot && PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --cov=xrdroot`
- Round 41's result of it, at `24b40f3b`: **not measurable** - the load average reached 144 (`W0-103`). The six test
  files the two commits touch report 355 passed, and ruff/mypy (1019 files)/maintainability (31 028
  functions)/components/`mkdocs --strict` are clean.
- Round 40's result of it, at `aabf7f48`: **not measurable at this load** (`W0-109`): at a load average of 34-54 the
  two-command gate reported 99.54% and 99.85% with all 12 931 tests passing. Round 39's at `297950c4` remains the last
  clean one: `TOTAL 118830 0 29826 0 100%` by the two-command decomposition (12 925 + 44 passed).
- Round 39's result of it, at `297950c4`: `TOTAL 118830 0 29826 0 100%` **by the two-command decomposition**
  (`pytest -n 4 --cov --ignore=tests/test_tmva_learn_networks.py --ignore=tests/test_pyroot_references.py` →
  12 925 passed, then `pytest tests/test_tmva_learn_networks.py tests/test_pyroot_references.py --cov --cov-append`
  → 44 passed, 100%); ruff/mypy (1019 files)/maintainability (30 990 functions)/components/`mkdocs --strict` clean.
- Round 38's result of it, at `4bd75d90`: **not measured** - the load average was 20-124 all round (`W0-103` again);
  916 unit tests pass in every file the three commits touch, and ruff/mypy (1019 files)/maintainability/components/
  `mkdocs --strict` are clean. **Rerun when the load average is under 20.**
- Round 37's result of it, at `b84bdc87`: **not measured** - the load average was 51-124 and the coverage gate reported
  0-27% with dead workers (`W0-103`); ruff, mypy (1019 files), maintainability (30 978 functions), components 0
  undeclared and `mkdocs build --strict` all pass, and the 7 `test_tmva_learn_networks.py` failures under load pass alone.
  **Rerun when the load average is under 20.**
- Round 36's result of it, at `630618f8`: `TOTAL 118750 0 29792 0 100%`, `12949 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1019 files,
  maintainability 30 962 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 35's result of it, at `4a9eb5ce`: `TOTAL 118743 0 29788 0 100%`, `12942 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1019 files,
  maintainability 30 946 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 34's result of it, at `c5962455`/`84271e6c`: `TOTAL 118534 0 29744 0 100%`, `12917 passed,
  1 skipped, 46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1019 files,
  maintainability 30 869 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 33's result of it, at `cc6125b7`: `TOTAL 118409 0 29722 0 100%`, `12907 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1019 files,
  maintainability 30 820 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 32's result of it, at `f359f0be`: `TOTAL 118142 0 29650 0 100%`, `12888 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1018 files,
  maintainability 30 734 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 31's result of it, at `27f23efd`: `TOTAL 118034 0 29632 0 100%`, `12867 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1018 files,
  maintainability 30 659 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 30's result of it, at `9d509b79`: `TOTAL 117893 0 29564 0 100%`, `12845 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1018 files,
  maintainability 30 613 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 29's result of it, at `21517c9a`: `TOTAL 117876 0 29556 0 100%`, `12835 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1018 files,
  maintainability 30 600 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 28's result of it, at `3d69abeb`: `TOTAL 117866 0 29554 0 100%`, `12833 passed, 1 skipped,
  46 xfailed`; `ruff` clean, `mypy` 1018 files, maintainability 30 596 functions, components 0
  undeclared, `mkdocs build --strict` built.
- Round 27's result of it, at `0e7e9fda`: `TOTAL 117765 0 29514 0 100%`, `12817 passed, 1 skipped,
  46 xfailed`; `ruff` clean, `mypy` 1018 files, maintainability 30 562 functions, components 0
  undeclared, `mkdocs build --strict` built.
- Round 26's result of it, at `85adfa05`: `TOTAL 117554 0 29432 0 100%`, `12777 passed, 1 skipped,
  46 xfailed` (measured at the three feature commits; the docs commits follow); `ruff` clean, `mypy` 1018
  files, maintainability 30 477 functions, components 0 undeclared.
- Round 25's result of it, at `dab9b98d`: `TOTAL 117460 0 29386 0 100%`, `12750 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1018 files,
  maintainability 30 418 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 24's result of it, at `59a2cac3`: `TOTAL 117378 0 29358 0 100%`, `12740 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1018 files,
  maintainability 30 398 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 23's result of it, at `4fa05128`: `TOTAL 117308 0 29330 0 100%`, `12725 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 367 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 22's result of it, at `356d077f`: `TOTAL 117152 0 29290 0 100%`, `12704 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 323 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 21's result of it, at `02f37c6e`: `TOTAL 117057 0 29252 0 100%`, `12685 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 295 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 20's result of it, at `f283ae1c`: `TOTAL 116891 0 29140 0 100%`, `12664 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 270 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 19's result of it, at `a12bc98b`: `TOTAL 116782 0 29124 0 100%`, `12652 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 236 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 18's result of it, at `6ae78a7a`: `TOTAL 116689 0 29096 0 100%`, `12634 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 208 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 17's result of it, at `5ff715a5`: `TOTAL 116559 0 29080 0 100%`, `12620 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 168 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 16's result of it, at `4509b416`: `TOTAL 116525 0 29068 0 100%`, `12588 passed, 1 skipped,
  46 xfailed`; `ruff` clean, `mypy` 1017 files, maintainability 30 094 functions, components 0
  undeclared, `mkdocs build --strict` built. Run with `-n 2` while the machine is loaded (`W0-45`).
- Round 15's result of it, at `177f6159`: `TOTAL 116525 0 29068 0 100%`, `12586 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 090 functions, components 0 undeclared, `mkdocs build --strict` built.
  **Run it with `-n 2` while this machine is loaded**: at `-n 4` under a load average above 20 the
  xdist workers lose a test's coverage now and then and the 100% gate flaps (`W0-45`).
- Round 14's result of it, at `77042aa8`: `TOTAL 116342 0 29012 0 100%`, `12575 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1017 files,
  maintainability 30 055 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 13's result of it, at `e0097751`: `TOTAL 115653 0 28754 0 100%`, `12256 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1016 files,
  maintainability 29 837 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 12's result of it, at `e15028fe`: `TOTAL 115351 0 28678 0 100%`, `12205 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1015 files,
  maintainability 29 728 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 11's result of it, at `641748fd`: `TOTAL 115054 0 28600 0 100%`, `12167 passed, 1 skipped,
  46 xfailed`; `ruff` clean, `mypy` 1014 files, maintainability 29 631 functions, components 0
  undeclared, `mkdocs build --strict` built.
- Round 10's result of it, at `d33afbbd`: `TOTAL 114900 0 28534 0 100%`, `12122 passed, 1 skipped,
  46 xfailed`; `ruff` clean, `mypy` 1014 files, maintainability 29 553 functions, components 0
  undeclared, `mkdocs build --strict` built.
- Round 9's result of it, at `a35c8e7`: `TOTAL 114450 0 28424 0 100%`, `12060 passed, 1 skipped,
  46 xfailed`; `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1012 files,
  maintainability 29 398 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 8's result of it, at `fdb58c4`: `TOTAL 114122 0 28324 0 100%`, `12033 passed, 1 skipped,
  46 xfailed`; `ruff` clean, `mypy` 1010 files, maintainability 29 315 functions, components 0
  undeclared, `mkdocs build --strict` built.
- Round 7's result of it, at `f816485`: `TOTAL 113892 0 28242 0 100%`, `11995 passed, 1 skipped,
  46 xfailed`; `ruff` clean, `mypy` 1010 files,
  `maintainability` 29 235 functions, components 0 undeclared, `mkdocs build --strict` built.
- Round 6's result of it, at `5998baf`: `TOTAL 113596 0 28142 0 100%`, `11962 passed, 1 skipped,
  46 xfailed`; `ruff` clean, `mypy` 1009 files, `maintainability` 29 145 functions, components 0
  undeclared, `mkdocs build --strict` built.
- Round 5's result of it, at `edbfdfa`: `TOTAL 113219 0 28014 0 100%`, `11912 passed, 1 skipped,
  46 xfailed`; with `XRDROOT_TEST_OTHER_MACHINE=1` the same; `ruff` clean, `mypy` 1008 files,
  maintainability 29 031 functions, components 1007 modules and 0 undeclared imports,
  `mkdocs build --strict` built.
- Round 4's result of it, at `9bc439a`: `TOTAL 112760 0 27888 0 100%`, `Required test coverage of
  100.0% reached`, `11863 passed, 1 skipped, 46 xfailed` in 38 s; with
  `XRDROOT_TEST_OTHER_MACHINE=1` the same 11 863 passed. `ruff` clean, `mypy` 1007 files,
  maintainability 28 915 functions, components 1006 modules and **0 undeclared imports**,
  `mkdocs build --strict` built.
- Round 3's result of it, at `e46e28d`: `TOTAL 112531 0 27814 0 100%`, `Required test coverage
  of 100.0% reached`, `11620 passed, 1 skipped, 46 xfailed` in 59 s; with
  `XRDROOT_TEST_OTHER_MACHINE=1` the same 11 620 passed. `ruff` clean, `mypy` 1006 files,
  maintainability 28 831 functions, components 1005 modules and **0 undeclared imports**,
  `mkdocs build --strict` built.
- Working directory: `/Users/rcurrie/src/xrdroot`
- Required preserved artifacts: the venv `.venv` (with `.[dev,zstd,fit,docs]`), ROOT's sources at
  `/tmp/rootsrc-6.40`, the oracle at `~/.local/root-oracle`, the tutorial oracle cache at
  `~/.cache/xrdroot-tutorials/oracle`, and the round directories holding its `results.json`
- Result: recorded in `TODO.md` §7 (the cumulative table's last column and the round's ledger rows)

## Round log

Each iteration appends one row here, and the *detail* of the round goes into `TODO.md` (step 6).

| round | base | scopes | in the round's `results.json` | new queue entries |
|---|---|---|---|---|
| 1 | `8c1b9f8` | sixteen, `/tmp/r1/wNN-brief.md`: formula member calls (from branch `w01`), an array of `std::vector` in a class, object columns in a tree, `BranchRef`/`TRefTable`, `GetStreamerInfo`+`pairEnumEvo`, evolution print DIFFs, genreflex XML, split `TClonesArray`, `std::array` as an element type, `RFieldBase`, nested-collection RNTuple, a text branch read, `ReadClones`, `typeid` of a template parameter, the `generate` template, and the seven timeouts (diagnosis only) | **roottest 1076 → 1078, non-tutorial 1231 → 1233** (`/tmp/r1/final2`, `/tmp/r1/final1` at `7a43bd9`) | 9 rows: `Q-23` (the streamer-info printer), `P-7`/`P-8` (the hangs and their two named causes), `G-5` (the template choice), `G-6` (member template constructors), `W0-9` (hadd tests colliding in one directory), plus the reverts recorded in the ledger |
| 2 | `773a63c` | sixteen, `/tmp/r2/wNN-brief.md`: the streamer-info printer (`Q-23`), formula objects, tree columns of objects, an array of `std::vector`, `BranchRef`, the `a(b) = c` SyntaxError (5 gtest), the template choice (`G-5`), `typeid` (`G-6`), `.Class`/`.typedef`/`.autodict`, TEnum, `TPostScript`, `TBufferJSON`, `TSelector`, a split `TClonesArray`, an RNTuple class field, and `WriteObjectAny` | **roottest 1078 → 1089, non-tutorial 1233 → 1244** (`/tmp/r2/final2`, `/tmp/r2/final1` at `5c88c8c`) | 6: `Q-24`, `Q-25`, `Q-26`, `Q-27`, `W0-10`, and the `pairEnumEvo`/formula notes in the ledger |
| 3 | `bb19779` | sixteen, `/tmp/r3/wNN-brief.md`: the formula's `GetHistogram`, a fast clone's column type (`Q-25`), a split `TClonesArray` (`Q-27`), `BranchRef` (`Q-15`), the streamer listing (`Q-24`), an old file's class (`Q-36`), GDML (`Q-14`), the writer's bytes (`Q-20`), RNTuple records, `_Set` and `treeproxy`, an inherited branch read, `datamodelevolution`, TMVA's RESHAPE, nine timeouts, the translator's `goto`/assignment, and `W0-10` | **roottest 1089 → 1105, non-tutorial 1244 → 1265** (`/tmp/r3/final2`, `/tmp/r3/final1` at `8af51aa`) | 10: `Q-28`, `Q-29`, `Q-30`, `Q-31`, `Q-32`, `Q-33`, `Q-34`, `Q-35`, `Q-36`, `Q-37`, plus `P-9`, `P-10` |
| 4 | `9cf98a5` | sixteen, `/tmp/r4/wNN-brief.md`: the writer's 2 811 bytes (`Q-20`), the formula's `fMatrix` (`Q-42`), a template member of a macro class (`Q-39`), a branch holding a `TNamed` (`Q-38`), `TRef` bits (`Q-35`), an evolved class (`Q-36`), a split `TClonesArray` (`Q-40`), the data-model reader, RNTuple fields, a vector member's branch name, a `TObjArray` the reader can give back, `.Class` (`Q-41`), `fakeClass`'s recursion, the buffer tests (`P-9`), `RootClassVersion` (`Q-28`), and `cmdLineUtils` | **roottest 1105 → 1114, non-tutorial 1265 → 1274** (`/tmp/r4/final2`, `/tmp/r4/final1` at `932164a`) | 6: `Q-38`, `Q-39`, `Q-40`, `Q-41`, `Q-42`, `Q-43` |
| 5 | `7e54e3c` | sixteen, `/tmp/r5/wNN-brief.md`: the formula's seven walls (`Q-42`), the `Each`-is-a-`Values` bug (`Q-38`), a `TString*` counted by a comment (`Q-39`), a split `TClonesArray` of a macro class (`Q-40`), `.Class` and a template template parameter (`Q-41`), the writer's last 405 bytes, the per-record description rebuild (`P-9`), the data-model reader, RNTuple fields, a vector member's branch name, an evolved class, `TRef` bits, a class version the file records (`Q-28`), JSON/`abstractclass`, `python/memory`, and `WILLFAIL` | **roottest 1114 → 1123, non-tutorial 1274 → 1284** (`/tmp/r5/final2b`, `/tmp/r5/final1` at `f583422`) | 4: `Q-44`, `Q-45`, `Q-46`, `Q-47`, plus `W0-14`…`W0-17` in the ledger |
| 6 | `a3a3d51` | sixteen, `/tmp/r6/wNN-brief.md`: the formula's object wall (`Q-48`), the read direction of `P-9`, the writer's last 405 bytes, `treeproxy`, the `reconcile` wiring (`Q-36`), bitset/collections (`W0-16`), `SplitClones` (`Q-38`), GDML (`Q-44`), `MakeProject` (`Q-46`), a `TObjArray` read back, the `BuildCheck` warnings (`Q-29`), `TRef` assignment (`Q-35`), bigevent's template member (`Q-39`), `fscanf`/TMath, `tree/stl` (`W0-10`), and `python/memory` | **roottest 1123 → 1132, non-tutorial 1284 → 1293** (`/tmp/r6/final2b`, `/tmp/r6/final1` at `3825443`) | 2: `Q-48`, `Q-49`, plus `W0-18`…`W0-20` and the `Q-29` revert in the ledger |
| 7 | `ebf4262` | sixteen, `/tmp/r7/wNN-brief.md`: the formula's object wall (`Q-48`), the `BuildCheck` warning and its old element code (`Q-29`/`Q-49`), `TGeoManager::Export` (`Q-44`), `SplitClones` (`Q-38`), bigevent's template member (`Q-39`), `treeproxy` (`Q-40`), the `reconcile` wiring (`Q-36`), `_Set`, `abstractclass`/`double32`, `python/memory`, `TClonesArray` of a base-less class, `stressgraphics`, `io/json`, the RNTuple read, `selectorreader`, and the last TIMEOUT (`P-10`) | **roottest 1132 → 1134, non-tutorial 1293 → 1294** (`/tmp/r7/final2`, `/tmp/r7/final1` at `2d0e458`) | 5: `W0-21`…`W0-25`, plus the `Q-29`/`Q-44`/`Q-46` closures in the ledger |
| 8 | `3f1cb63` | sixteen, `/tmp/r8/wNN-brief.md`, **built from round 7's reports' file lists**: the formula's object wall in `methods.py` (`Q-48`), the template spelling (`Q-39`), the reader's layouts (`Q-36`), the brace-init containers (`Q-33`), `SplitClones` (`Q-38`), `TGeoManager::Export` (`Q-44`), `TBufferJSON` (`W0-25`), a version-0 class, `python/memory`, `stressgraphics`, bigevent, `treeproxy`, `selectorreader`, `.typedef`, and the last TIMEOUT | **roottest 1134 → 1138, non-tutorial 1294 → 1298** (`/tmp/r8/final2`, `/tmp/r8/final1` at `f65e991`) | 1: `W0-26`, `W0-27` in the ledger |
| 9 | `9b37ece` | sixteen, `/tmp/r9/wNN-brief.md`: the formula's object member (`Q-48`), bigevent's template member (`Q-39`), `SplitClones` over four writer files (`Q-38`), `treeproxy` (`Q-40`), the `reconcile` wiring (`Q-36`), `TGeoManager::Export` (`Q-44`), `TBufferJSON` (`W0-25`), `_Set`, splitting in `objects.py`, `double32`, `selectorreader`, `tree/stl` (`W0-10`), `python/memory`, `stressgraphics`, the last TIMEOUT (`P-10`), and `TEnum` | **roottest 1138 → 1141, non-tutorial 1298 → 1301** (`/tmp/r9/final2b`, `/tmp/r9/final1` at `2ef9c80`) | 3: `W0-28`, `W0-29`, `W0-30` in the ledger |
| 10 | `05b43b7` | sixteen, `/tmp/r10/wNN-brief.md`, from round 9's reports: the reference loop's reads (`Q-42`/`Q-48`), `Alt$`'s flat index, `SplitClones` on both sides (`Q-38`/`Q-40`), bigevent's members, `treeproxy`, `selectorreader`, `jsonbuffer`, `objects.py`, `double32`/`TEnum`, the memory lifetime patch, the `_Set` cursor, `stressgraphics`, `tree/stl` (`W0-10`), the read-by-declared-layout (`Q-36`), and the geo exports | **roottest 1141 → 1183, non-tutorial 1301 → 1342** (`/tmp/r10/final2`, `/tmp/r10/final1` at `71aa3256`) - the first roottest figure from an **intact** source tree | 2: `W0-31` (the damaged tree), `W0-32` |
| 11 | `2a4744f6` | sixteen, `/tmp/r11/wNN-brief.md`: a `vector<Track>` pointer (`Q-39`), a `TGeoManager` record, a batch `TBrowser` (`Q-44`), `SplitClones`' next layer, bigevent, an evolved class (`Q-36`), `treeproxy`, `selectorreader`, `TBufferJSON` (`W0-25`), the `_Set` cursor, the last TIMEOUT, `TEnum`, `stressgraphics`, `tree/stl` (`W0-10`), the writer's arrays, and the split-branch measurement | **roottest 1183 → 1181 measured (1185 with the four network tests), non-tutorial 1342 → 1340 (1344)** (`/tmp/r11/final2`, `/tmp/r11/final1`, `/tmp/r11/net` at `90f5c5e8`) | 1: `W0-33`, plus the `W0-10`/`W0-25` rows in the ledger |
| 12 | `14ef50ad` | sixteen **chains**, `/tmp/r12/wNN-brief.md`: the split `TClonesArray` writer chain, the geometry export chain, bigevent's declaration chain, the evolved-class read chain, `treeproxy`'s pyroot chain, the graphics encoders, a selector's reader, `TBufferJSON`, the makeproject translator chain, `TEnum`, `tree/stl`'s `pycint` chain, a 300 s clone, `double32`, TMVA, `roottest/python`, and the gtest runtime | **roottest 1181 → 1195, non-tutorial 1340 → 1354** (`/tmp/r12/final2`, `/tmp/r12/final1` at `f7e42087`), **14 in, 0 lost** | 2: `W0-34`, `W0-35` |
| 13 | `70523896` | sixteen **chains**, `/tmp/r13/wNN-brief.md`: the geometry export chain, the compression-setting chain, the split `TClonesArray`, the evolved-class read, `ROOT.list` and JSON, `lotsRef`, the makeproject _Set, the tests near the limit, the graphics encoders, `TEnum`/enums, `double32`, `roottest/python`, the gtest kind, TMVA, `treeproxy`, and the module tests' missing names | **roottest 1195 → 1220, non-tutorial 1354 → 1386** (`/tmp/r13/final2`, `/tmp/r13/final1` at `a5d25f67`), **25 roottest + 7 `other` in, 0 lost** | 1: `W0-36` |
| 14 | `12646b58` | sixteen **chains**, `/tmp/r14/wNN-brief.md`, each **starting** at the file round 13's reports named: bigevent's drawing classes, `GetClonesName`, the evolved-class read, the data model's records, a `TRef` in a `TClonesArray`, the harness's lost `schemaRun1.log`, the WILLFAIL verdict, the tests near the limit, the graphics encoders, `roottest/python`, the gtest kind, TMVA, `other/math`, `MakeProxy`, `TBufferJSON`, and `double32` | **roottest 1220 → 1226 measured (1231 with the five that pass alone), non-tutorial 1386 → 1393 (1398)** (`/tmp/r14/final2`, `/tmp/r14/final1`, `/tmp/r14/solo` at `19be128b`) | 3: `W0-37`, `W0-38`, `W0-39`, `W0-40` |
| 15 | `99525851` | **eight** chains (the machine's load average reached 64): bigevent's writer, `SplitClones`, the evolved class's warnings and walk, the data model's records, a `TRef` in a `TClonesArray`, `pairEnumEvo`'s listing, the harness's lost working file, and the four buffers' margin | **roottest 1226 → 1230 at `--jobs 3`, non-tutorial 1393 → 1397** (`/tmp/r15/final2`, `/tmp/r15/final1` at `601996d1`), 8 in, 0 lost to code | 4: `W0-41`…`W0-44` |
| 16 | `df6d9833` | **one chain**, not sixteen (the machine stayed at a load average of 9-36): the harness's `STOREOUT` rule and a build-tree `OUTREF` - ROOT stores every roottest's output as `<build>/<name>.log` and the next test reads it, which xrdroot never did | **roottest 1230 → 1233, non-tutorial 1397 → 1400** (`/tmp/r16/final2`, `/tmp/r16/final1` at `2dad75a7`, `--jobs 3`), 3 in, 0 lost | 3: `W0-46`, `W0-47`, plus the `W0-40` closure |
| 17 | `ba2985b6` | **eight** chains (the machine stayed at a load average of 13-44): `SplitClones`' last answer, bigevent's writer, the evolved class's six stderr DIFFs, the data model's records, `pairEnumEvo`'s listing, the graphics encoders, `TBufferJSON`, and `double32` | **roottest 1233 → 1239, non-tutorial 1400 → 1406** measured in six areas (`/tmp/r17/areas`, `/tmp/r17/check-ev` at `13034109`); the round's own full run is invalid and unquoted (`W0-48`) | 4: `W0-46` closed, `W0-48`, `W0-49` |
| 18 | `cb5c17e4` | **eight** chains (load average 9-20): the data model, `pairEnumEvo`, `double32`, bigevent's writer, `TBufferJSON`, the graphics encoders, `SplitClones`, and `roottest/python` | **0 gained, 0 lost**: ten areas, 167 tests, three commits that each move a wall without flipping a test (`/tmp/r18/areas` at `--jobs 1`) | 1: `W0-50` - a file-path chain cannot contain a missing behaviour |
| 19 | `2c6be318` | **eight groups with their whole component in scope** (the rule `W0-50` set): bigevent, `pairEnumEvo`, `io/tree`, the data model, `double32`, the graphics pad, `TBufferJSON`, `roottest/python` | **0 gained, 0 lost again** (`/tmp/r19/areas`, `/tmp/r19/tree`): six commits, each removing one refusal | 3: `W0-51`, `W0-52` (what each group actually needs), `W0-53` |
| 20 | `1000afdd` | **four** targets, each one change round 19 had named: the ACLiC `NameError` (24 cases of `roottest/python`), `TClass::GetClassVersion` for a file-only layout (11 `io/tree` tests), `TBufferJSON`'s `MEMBERS` entries, and the pad's PostScript counts | **0 gained, 0 lost**: three commits, and both groups measured unchanged in PASS (`/tmp/r20/areas`, `/tmp/r20/tree`) - each fix revealed the group's next wall | 2: `W0-54`, `W0-55` |
| 21 | `57940f19` | **one feature in four files** (`W0-55`'s rule): `Event::fTemplateMember`, the `template2<template1<int>>` member of `io/bigevent`'s class - `winfo.py`, `wdeclared.py`, `wdobjects.py`, `rdobjects.py` | **0 PASS, and the wall is gone**: `io/bigevent` builds and measures **2 PASS / 0 FAIL / 2 UNSUPPORTED** (`/tmp/r21/ev2`), its refusal moved to `a TArrayI is not something this writer puts in a ROOT file` | 2: `W0-56` (stale-cache ImportError), `W0-57` (`TArrayI` as a member) |
| 22 | `249dce34` | **one feature in three files** (`W0-55`'s rule): `TArrayF`/`TArrayI` as a *member* of a class (`W0-57`), which serves `io/bigevent` and `double32` - `winfo.py`, `wdobjects.py`, `rdobjects.py` | **0 PASS, the wall gone**: `io/bigevent` still **2 PASS / 0 FAIL / 2 UNSUPPORTED** with its refusal moved to `a TVector3 here is a float rather than the dict of ...`; `io/double32` **3 PASS / 1 FAIL** (`/tmp/r22/ev`, `/tmp/r22/d32`) | 2: `W0-58` (the `TVector3` member), `W0-59` (stray `docs/` in worktrees) |
| 23 | `21ab15c4` | **one feature in three files**: a ROOT class's member that is itself one of ROOT's classes (`TLorentzVector::fP` is a `TVector3`) - `pyroot/core/declared.py`, `wdobjects.py`, `rdobjects.py` | **roottest +1: `io/bigevent/write` PASSES** - the group is **3 PASS / 0 FAIL / 1 UNSUPPORTED** (`/tmp/r23/ev`), its `read` side walled by `a vector<TObject> * member ... that this reader does not decode inside an entry` | 2: `W0-60`, `W0-61` |
| 24 | `a12e5e43` | **one feature in two files**: a container a member *points at* is read as the container it names (`vector<TObject> *fVectorTobject`, stype 71) - `interp.py`, `rdobjects.py` | **0 PASS**: `io/bigevent` stays **3 PASS / 0 FAIL / 1 UNSUPPORTED** (`/tmp/r24/ev3`), `read`'s refusal moved to `fVectorTnamed` | 2: `W0-62` (that wall is `vector<TNamed> *[6]`, an array of pointers - the round's one-line attempt was reverted by its own unit test), `W0-63` (a refusal that moves further is not progress until the full suite agrees) |
| 25 | `3aa0b0db` | **one feature in two files**: an array of pointers to containers reads out of one record, each item named (`W0-62`) - `interp.py`, `rdobjects.py` | **0 PASS**: `io/bigevent` stays **3 PASS / 0 FAIL / 1 UNSUPPORTED** (`/tmp/r25/ev`), `read`'s refusal moved two members on to `fMapTNamedp` - and this time the full suite agreed, where round 24's relabelling had broken a unit test | 2: `W0-64` (a `std::map` whose key is a ROOT class), `W0-65` (the feature that replaced the revert) |
| 26 | `a4f10d15` | **three features, three workers, disjoint files**: `fName`/`fTitle` are the object's own members, read and written (`7f1663d1`, `172093d3`, `pyroot/core/`); a container template's nested `iterator` is a name a macro can declare (`a6336c75`, `cppstd/`); the `Event`'s maps read (`33678ae2`, `interp.py`) | **roottest +1: `io/tree/create-r` PASSES** (`/tmp/r26/base1` -> `/tmp/r26/tree`, 5 -> 6 PASS in `roottest/root/io/tree/*`), `test-r` FAIL -> DIFF (it runs now); `io/bigevent` 3 PASS / 0 FAIL / 1 UNSUPPORTED with a named format error 48 bytes/event into `vector<TLine>` (`/tmp/r26/ev`) | 5: `W0-66` (the chain's real feature: a `TRef` read, 11 tests), `W0-67` (a class with a base read member-wise), `W0-68` (SplitClones of a file-only class), `W0-69` (`docs/` committed by a worker again), `W0-70` (scratch files) |
| 27 | `4c82dc31` | **seven commits, four features, one chain**: a split `TClonesArray` read by `SetAddress` is filled (`42898a9a`); a `TRef` member read out of a tree resolves through its `TProcessID` (`212010d0`); a split `TClonesArray` of a class the program declared is cloned by it (`9867dfc9`); a member-wise container reads a base ROOT gives a record to (`5dd0ebba`); plus `ROOT.memset` and collections in JSON (`fd1d360a`, `32cc0d7f`) and the six branches they left untested (`23e557d8`) | **roottest +4 in `roottest/root/io/tree/*`: PASS 6 -> 10** (`/tmp/r26/tree` -> `/tmp/r27/tree2`: `create-rc`, `create-rcr`, `create-rcrr`, **`test-r`**), 12 UNSUPPORTED -> 8 DIFF, 0 FAIL; `io/json` 4/5/3 (`/tmp/r27/json2`); `io/bigevent` still 3/0/1 (`/tmp/r27/ev`) | 4: `W0-71` (bigevent's next container), `W0-72` (7 `io/tree` DIFFs = `Show`'s formatting), `W0-73` (`io/json`'s unbound-method and `None` shapes), `W0-74` (the coverage gate belongs after every commit) |
| 28 | `d42d8a5c` | **five commits, five features, no PASS**: `Show` prints a split collection's count line and unbroken members (`df9caceb`); a clone carries the `ProcessID` record its `TRef`s name (`1d00c3ed`); an array of strings and an array of pointers read every element (`2aee3989`); a member function named through its class is called on the object (`6224b151`); `TBufferJSON` writes a macro class's member (`db8b97ee`) | **0 PASS**: `io/tree` 10 PASS / 8 DIFF / 2 UNSUPPORTED (`/tmp/r28/tree`), `io/json` 4 / 4 / 1 / 3 with **`String` FAIL -> DIFF** (`/tmp/r28/json`), `io/bigevent` 3 / 0 / 1 (`/tmp/r28/ev`) | 3: `W0-75` (**three lines per entry** from seven PASSes: the process byte in `fUniqueID`/`fBits`, `fRef` printing `TRef`, a trailing blank line), `W0-76` (`io/json/String` one line from PASS), `W0-77` (bigevent's class reference points nowhere); `W0-72` corrected |
| 29 | `9dd616a0` | **three one-line behaviours**: a null `char*` member is `null` in JSON (`44e2d2ed`); `Show` prints a bare record as the reference it is (`d07c8610`); every object carries the two heap bits (`8ffa748c`) | **roottest +1: `roottest/root/io/json/String` PASSES** (`/tmp/r29/json`: `io/json` 4 -> 5 PASS), `io/tree` 10 PASS / 8 DIFF / 2 UNSUPPORTED with `test-rc` down from **four** differing line kinds to **three** (`/tmp/r29/tree`) | 3: `W0-78` (the `fBits`/`fUniqueID` bytes - the last three lines of six tests), `W0-79` (`test-rcrr123`'s third-level `TRef` chain), `W0-80` (a worker died; `W0-77` still open); `W0-75` corrected and closed |
| 30 | `4cdb3649` | **three commits**: a `char[N][M]` is one buffer of its rows (`71dba22d`), a `std::pair` written on its own is `$pair` (`8e797bcb`), the identifier and its process (`ae646b4e`) | **roottest +2: `io/json/BasicTypes` and `io/json/Objects` PASS** (`/tmp/r30/json` -> `/tmp/r30/json2`); `io/json` is **7 PASS / 0 FAIL / 0 DIFF / 5 UNSUPPORTED**, `io/tree` unchanged at 10 / 8 / 2 (`/tmp/r30/tree`) | 3: `W0-82` (`FromJSON` filling an existing object's containers), `W0-83` (`W0-78` with its mechanism measured), `W0-84` (two workers died, `W0-77`/`W0-79` open); `W0-81` closed |
| 31 | `c862d06a` | **five commits, one feature in two halves**: a `TObject` reads back with `fBits`' heap bits (`a65836dd`) and `fUniqueID`'s process byte (`ca510e97`); `FromJSON` fills a container a constructor left empty (`46dbd65c`); a value in full where a pointer is a `$ref` (`6b3e0928`); a `Double32_t` template spelling (`d07a00cd`) | **roottest +3**: `io/tree/test-rc` PASSES (`/tmp/r31/tree2`: `io/tree` 10 -> **11** PASS), and `io/json/STL1` + `StreamerLoop` PASS while `STL` moved UNSUPPORTED -> DIFF (`/tmp/r31/json`: `io/json` 7 -> **9** PASS, 1 DIFF, 2 UNSUPPORTED); `io/double32` unchanged at 3 PASS / 1 FAIL (`/tmp/r31/d32`) | 2: `W0-85` (the same two values for a **cloned** file's split collection - five tests, two lines each), `W0-86` (`STL`'s `set1` is 2 343 bytes where ROOT's is 5 144); `W0-82` and `W0-83` closed |
| 32 | `3372a1eb` | **two commits**: a `std::set`/`std::map` keeps a copy of what it is given, as C++ does (`8c150a28`); ROOT::Math's integration kinds, function interfaces and Util (`62afe7fe`) | **roottest +1**: `io/json/STL` PASSES - `io/json` **10 PASS / 0 FAIL / 0 DIFF / 2 UNSUPPORTED** (`/tmp/r32/json`); **`other` +2**: `other/math/*` **16 -> 18 PASS**, no PASS lost (`/tmp/r32/other`) | 5: `W0-87` (`other/math`'s next six: `Minimizer.Clear()`/`GetMinosError()`, `Minuit2Minimizer`, `BinData.GetErrorType`, `Chi2FCN`), `W0-88` (**read a dead worker's `docs/discoveries/` - that is where `W0-87` came from**), `W0-89` (`roottest/root/math/*`, 5 tests, 0 PASS); `W0-86` closed; `W0-85`/`W0-79` re-queued |
| 33 | `91bf694c` | **two commits**: a `Minimizer` that can be cleared and asked for its MINOS errors (`88fa5527`); ROOT::Math's `SVector`/`SMatrix`, subscripted as the macro writes them (`e3838fad`) | **`other` +1**: `other/math/mathcore-testMinim` PASSES - `other/math` **19 PASS / 0 FAIL / 45 UNSUPPORTED / 1 TIMEOUT** (`/tmp/r33/other`); `roottest/root/math/*` still **0 PASS / 4 FAIL / 1 UNSUPPORTED** (`/tmp/r33/rmath`) but all five now *run* | 4: `W0-90` (`roottest/root/math`'s five named walls), `W0-91` (**`W0-79`'s root cause, measured byte for byte** - a clone's new `ProcessID` record and the file index in both the reference and the identifier), `W0-92` (no `TRefTable` is written, so a collision resolves by branch order); `W0-87`/`W0-89` partly taken |
| 34 | `8997cd97` | **three commits**: ROOT::Math's free functions on `SVector`/`SMatrix` (`efa5f946`); a split `TObject`'s identifier keeps the process byte the file carries (`6bd0d6ba`); the SMatrix test split to stay inside the CCN limit (`c5962455`) | **0 PASS**: `io/tree` stays **11 / 7 / 2** (`/tmp/r34/tree`) and `roottest/root/math/*` stays **0 / 4 / 1** (`/tmp/r34/rmath`); both groups' failures moved but none closed | 4: `W0-93` (**a worker's test with CCN 14 failed the maintainability gate** - every gate after every cherry-pick, not coverage alone), `W0-94` (**two of four workers produced nothing at all**); `W0-91` partly taken (two of three pieces), `W0-90` re-queued |
| 35 | `1c25bcff` | **five commits**: ROOT::Math's `AssignSym` and the `MatRepSym` a translated type names as text (`84e15c0f`); **a clone writes the identifiers its objects were numbered with** (`920dfe4c`, round 33's two edits at last); a `TObject` read member at a time carries the two words its object has (`dee7d185`); ROOT::Math's genetic minimiser and its seed (`3e2588d0`, `e67c9a79`) | **roottest +3**: `io/tree` **11 -> 14 PASS**, 7 -> **4 DIFF**, **0 FAIL** (`/tmp/r35/tree`: `test-rcr`, `test-rcrr`, `test-rcrr3`); **`other` +2**: `other/math` **19 -> 21 PASS** (`/tmp/r35/other`: `genetic-testGAMinimizer`, `genetic-GAMinTutorial`); `roottest/root/math/*` 0 / 4 / 1 (`/tmp/r35/rmath`) | 3: `W0-95` (**the same feature one level deeper** - a clone of a clone still names the base object, three tests, 180 identical-count lines each), `W0-96` (`other/math`'s 43 UNSUPPORTED bucketed), `W0-97` (`roottest/root/math`, take all five macros at once); `W0-91`, `W0-85`, `W0-87` closed |
| 36 | `59744112` | **two commits**: a reference written after a clone names the file's process record (`1b2ae9d5`, `W0-95`); ROOT's own array-compression forms in JSON (`f0c859a9`) | **roottest +3**: `io/tree` **14 -> 17 PASS**, 4 -> **1 DIFF**, **0 FAIL** (`/tmp/r36/tree`: `test-rcr2`, `test-rcrr2`, `test-rcrr123`) - **the whole `TRef` family is closed**; `io/json` unchanged at 10 / 0 / 0 / 2 (`/tmp/r36/json`) | 3: `W0-98` (**the translator's `&v[0]`/`&v.front()` refusal blocks 10 of `other/math`'s 43** - the top of round 36's 22-cause bucket table), `W0-99` (`io/json`'s last two refusals), `W0-100` (`io/tree/read`'s last DIFF); `W0-95` and `W0-96` closed |
| 37 | `d13ce7e8` | **two commits in, two reverted**: a TMap's JSON as the pairs it holds (`4e0461e6`); a symmetric SMatrix filled from its upper triangle (`aa265151`). **Reverted**: the address of a vector element (`b28bb2c2`, `W0-98`'s top row - six branches with no test, branch `r37-01` kept) and a user class's missing dictionary (`03c2c442`, `afd8afdd` - it warns for `TGeoAtt`/`TGeoVoxelFinder`/`TGeoParallelWorld`, which ROOT has dictionaries for, and breaks a PASS) | **0 PASS**: `io/json` **10 / 0 / 0 / 2** (`/tmp/r37/json`) and `roottest/root/math/*` **0 / 4 / 1** (`/tmp/r37/rmath`), both unchanged | 3: `W0-101` (what the vector-element tests must reach), `W0-102` (why the class warning broke a PASS), `W0-103` (**no gate is measurable above a load average of ~20** - `W0-45` is worse than recorded) |
| 38 | `7216c284` | **three commits**: the address of a vector element reaches every branch it added (`bcd98495`, `W0-101` closed); `wpacking`'s `OverflowError` on Python 3.14 and `np.atan`'s NumPy-2 spelling (`2c64e67f`); two float comparisons through their decimal spelling, as a tolerance (`db1c1a00`) | **no harness measurement** (load average 20-124, `W0-103` stands); **916 unit tests pass** in every file the commits touch | 2: `W0-104` (three pre-existing breakages, fixed, with their causes), `W0-105` (**the venv is Python 3.14 + NumPy 1.26**, where the tree assumed NumPy 2 and older `struct` errors); `W0-101` closed |
| 39 | `ebd7485f` | **one commit**: both of the trainer's optimisers, and the coverage gate measurable again (`4f7ddee5`, `W0-107`) | **the gate is green**: `TOTAL 118830 0 29826 0 100%` by the two-command decomposition (12 925 pass, then 44 with `--cov-append`); **`other` +1**: `other/math` **21 -> 22 PASS** (`/tmp/r39/other`, `mathcore-testSpecFuncBeta`), 0 FAIL, 45 -> 41 UNSUPPORTED, one new TIMEOUT (`mathcore-stdsort` runs now); `io/tree/read` **4 lines** from PASS (`/tmp/r39/read`) | 1: `W0-108` (**`mathcore-stdsort` times out at 300 s** - the translated `std::sort` needs one NumPy pass); `W0-102` sharpened to its two-part fix; `W0-107` closed |
| 40 | `17c9a7fa` | **two commits**: a class ROOT has a dictionary for is no warning, by name as well as by layout (`82001fb0`, `W0-102` closed after three rounds); the declared-class accessors' fall-backs (`de1e9cac`) | **roottest +1: `io/tree/read` PASSES** - `roottest/root/io/tree/*` is **18 PASS / 0 FAIL / 0 DIFF / 2 UNSUPPORTED** (`/tmp/r40/tree`) | 2: `W0-108` (**`mathcore-stdsort` is a benchmark** - 10^7 comparator calls, 4 min without finishing the first size: a decision, not a fix), `W0-109` (**the gate needs a load under ~20 again** - 99.5% at 34-54 with every test passing); `W0-102` closed |
| 41 | `54ff9563` | **two commits**: a qualified static call fills the local handed to it by reference (`9acf40db`, half of `W0-99`); `std::generate`/`shuffle`/`next_permutation`, the ostream formatting state and `ROOT::Math::Random<T>`'s engine kinds (`e008ab0b`, worker 02's uncommitted work taken from its tree) | **no harness PASS measured** - the load average reached **144**, so `io/json` and `other/math` were not measured and `roottest/root/math/*`'s 0/4/TIMEOUT is unproven (`W0-103`) | 3: `W0-99` sharpened (**`FromJSON` reading a `TArrayI`** is the next gap), `W0-110` (w02's work landed, w03's patch not taken - UNSUPPORTED → TIMEOUT), `W0-111` (**check the worktrees at the halfway point**) |
| 42 | `4dad0dad` | **no fan-out** (the load average opened at 76 and never went under 14): the cell's class for an object pointer (`4e5436ff`, `W0-99` closed), ROOT's iterator family and a `TIter` that owns one (`a2ef7d77`), `RootFinder::Name`/`Iterations` plus the two tests round 41's unmeasurable gate hid (`938d9faf`) | **roottest +1**: `io/json/ArrayCompress` PASSES, `io/json` **11 PASS / 0 FAIL / 0 DIFF / 1 UNSUPPORTED** (`/tmp/r42/json`); gtest `coreconttestUnit`'s two `TIter` cases pass | 6: `W0-112` (a branch shipped with no test while the gate was unmeasurable), `W0-113` (`TRangeStaticCast` and ROOT's static `Class()`), `W0-114` (`RootClasses` needs a `TH1I`'s JSON), `W0-115` (`RActionImpl`, `TQObject`), `W0-116` (`Roots::Newton`/`Brent` for `testGSLRootFinder`), `W0-117` (the `ResourceWarning` flake only a run without `--cov` sees) |
| 43 | `0cbe14e9` | **two commits**: ROOT::Math's GSL solvers - `Roots::*`, `GSLRootFinder`, `GSLRootFinderDeriv`, `IRootFinderMethod` (`0be3733b`, `W0-116` closed); the class constants ROOT's tests name - `TTree`/`TAxis`/`TH1`/`TInterpreter` status words and `RootFinder::kMODAB` (`4e473e1e`) | **`other` +1**: `other/math/mathmore-testGSLRootFinder` PASSES, `other/math` 22 -> 23 (`/tmp/r43/gsl`); **the gtest kind measured whole for the first time since round 13**: 53 PASS / 219 FAIL / 76 UNSUPPORTED / 2 TIMEOUT (`/tmp/r43/gtest`) | 5: `W0-118` (the gtest ranking's 219 first failures), `W0-119` (the Clad derivator family is a scope decision), `W0-120` (`std::vector::resize`), `W0-121` (a `.cxx` with `main()` runs its namesake), `W0-122` (`TH1::SetStatOverflows`) |
| 44 | `9ba7eb49` | **one commit**: ROOT::TSeq's random-access iterator - `begin()`/`end()`, `+`/`-`/`++`/`--`/`[]`/`*it`, and `TSeqU`/`TSeqUL` (`250324e3`) | **gtest +1**: `gtest/core/cont/TSeqTests` PASSES (`/tmp/r44/seq`), the kind 53 -> 54; `coreconttestUnit` moves to `TCollection.RangeCast` | 0 new: `W0-118` **corrected** (round 43 counted occurrences, not programs - 114 of 219 FAIL programs ask for no missing name at all) |
| 45 | `2f5d8392` | **two commits**: `fflush` reads the name a translation writes for a stream, as `fprintf` does, and `TBits::Set`/`Get` (`4a98e20d`); `root` names an option it cannot read, and the harness matches a `PASSREGEX` across newlines as CMake does (`4468d37b`) | **`other` +1**: `root_exe_errorMessage` PASSES (`/tmp/r45/errm2`), and **both `test/` and `other` measured whole for the first time in ~28 rounds**: `other` 40 / 109 (`/tmp/r45/other`), `test/` 8 / 59 (`/tmp/r45/testkind`) | 3: `W0-123` (a `Char_t[]` of bytes becomes a `str` truncated at the first NUL), `W0-124` (the five `RooUnitTest` tests are a framework), `W0-125` (`testbits` needs only `W0-123`) |
| 46 | `56b0a92a` | **three commits**: a `char[]` from a braced list is a byte buffer unless it spells a C string, the dialect's `buffer_functions`, and `TBits::Get` filling bytes into any element width (`af1e82fd`); a word read into a dialect string class is that class (`7e369769`); the two reads' tests and the split of two functions the limits caught (`e142f240`) | **`test/` +1**: `test/testbits` PASSES (`/tmp/r46/bits`, 41 tests), the kind 8 -> 9; `test/stressgraphics-interpreted` FAIL -> UNSUPPORTED (`ROOT has localtime`) | 1: `W0-126` (C's `localtime`/`struct tm`); `W0-123` and `W0-125` closed |
| 47 | `44c42d2d` | **two commits**: a failed extraction stores nothing, as C++ leaves the variable alone (`35c96338`, `W0-127` closed); C's `tm`/`localtime`/`gmtime`/`strftime` (`f0655cf0`, `W0-126` closed) | **no harness PASS**, but the **gtest kind re-measured whole: 54 PASS / 218 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** (`/tmp/r47/gtest`, one gain, none lost in four rounds), and `test/stressgraphics-interpreted` **UNSUPPORTED -> FAIL**: it runs the graphics stress suite and fails on image byte counts (`/tmp/r47/sg2`) | 1: `W0-128` (the images' character counts - the `canvas/` work `W0-52` records); `W0-126` and `W0-127` closed |
| 48 | `c003f165` | **two commits**: `TH1::Normalize` and `TH1::EStatOverflows`' own name, with `Get/SetStatOverflows` and the static `StatOverflows` (`4e2ba73b`, `W0-122` closed); a class body's enum is metadata, so `TH1D`'s native methods are still called (`f16f5aeb`) | **no harness PASS**, but ROOT's own `test_TH1.cxx` goes **13 -> 11 failing cases** (`TH1.Normalize` and `TH1.StatOverflows` now OK), and the round's own gate caught a regression a class attribute caused in `hfnative` (13 failed, 78 at the base by bisect) | 1: `W0-129` (`testTH1`'s nine remaining cases, each a feature); `W0-122` closed |
| 49 | `55318546` | **one commit**: `AddBinContent` takes one bin number per axis, as `TH2`/`TH3` overload it (`3f3dae7f`) | **no harness PASS**, but ROOT's own `test_TH1.cxx` goes **11 -> 10 failing cases** (`TH1.AddBinContent` now OK); the round's own gate caught the `Eval` regression the change caused | 0 new: `W0-129` **sharpened** to a case-by-case account of `testTH1`'s ten remaining cases |
| 50 | `a21ec264` | **one commit**: `TAxis` reports the binning it cannot use, as ROOT's `TAxis::Set` does, and `TAxis(edges)` is ROOT's constructor from a vector of edges (`2e10bb4d`) | **no harness PASS**, but ROOT's own `test_TH1.cxx` goes **10 -> 9 failing cases** (`TAxis.EqualBinEdges` now OK) | 0 new: `W0-129` re-stated as the handover item, with each of `testTH1`'s nine cases and what it needs |
| 51 | `b9570d8d` | **one commit**: `TH1::SmoothArray`, the static 353QH-twice smoother over a raw array a program hands as a pointer (`4227df45`) | **no harness PASS**, but ROOT's own `test_TH1.cxx` goes **9 -> 8 failing cases** (`TH1.SmoothArrayCrossCheck` now OK) | 0 new: `W0-129` down to eight cases, each measured, with `SetBufferedSumw2` and `DumpOutput` named as the expensive two |
| 52 | `0537d989` | **one commit**: a collection's element carries the type C++'s range gives it (`TypedElement`), `iterate` hands a `TCollection`'s objects out as the `TObject *` C++ iterates, and `ROOT::Detail::TRangeStaticCast<T>`/`TRangeDynCast<T>` (`5f58c997`) | **gtest +2: `gtest/core/cont/coreconttestUnit` and `gtest/core/cont/testTypedIteration` PASS** (`/tmp/r52/cont`), the kind **54 -> 56**; nothing lost in `io/tree` (18/0/0/2), `io/json` (11/0/0/1) or 66 `io`/`hist`/`graf2d` gtest programs (`/tmp/r52/gtest-iohist`) | 3: `W0-130` (a wrapper a macro can keep - `TCollection` should unwrap), `W0-131` (the round-44 gtest bucket table is eight rounds stale), `W0-132` (`test_THBinIterator`: 64 s for 20 cases, 7 failing); `W0-113` closed |
| 53 | `6f1b71a9` | **two commits**: `ROOT::RangeStaticCast`/`ROOT::RangeDynCast` over any range, a C array of pointers included, with a class the program declared found in the interpreter's table (`7f90083d`); `TDataType::GetType(typeid(T))` over `EDataType`'s codes, with them under the bare names a macro writes (`f93144fd`) | **gtest +2: `gtest/core/foundation/testRRangeCast` and `gtest/core/meta/testTDataType` PASS** (`/tmp/r53/corefm`), the kind **56 -> 58**; **the gtest kind measured whole at HEAD** (`/tmp/r53/gtest`): 56 / 216 / 76 / 2 of 350, nothing lost | 2: `W0-131` closed (the ranking is current and has no name cluster; `/tmp/r53/one-case.md` names the 47 programs one failing case from a PASS), `W0-133` (`Calc` echoes where ROOT is silent) |
| 54 | `2a668e00` | **two commits**: a qualified static call `T::f(...)` asks the dialect's `method_returns`, so an `auto` local a factory gives has a type, and `RNTupleWriter` is a guard with the `_destruct` one needs (`98df6a2e`); `TInterpreter::Calc` reads a value out without cling's echo (`2ce1f188`) | **gtest +1: `gtest/tree/ntuple/ntuple_randomaccess` PASSES** (`/tmp/r54/ntuple`), the kind **58 -> 59**; nothing lost in the 49 `gtest/tree/ntuple/*` or the 26 `gtest/core/*` (`/tmp/r54/core`) | 3: `W0-133` closed, `W0-134` (`GetClusterDescriptor`), `W0-135` (three `TFile`s in one scope), `W0-136` (another session's stopped processes starve a whole-kind run) |
| 55 | `127c20e2` | **one commit**: the IO-feature bits a ROOT 6.40 basket records - `ROOT.EIOFeatures`, `ROOT::Experimental`'s two enums and `TBasket::kIOBitCount`/`EIOBits::kSupported`, every number measured against the oracle (`d1752d03`) | **gtest +1: `gtest/tree/tree/testTIOFeatures` PASSES** (`/tmp/r55/areas`: 73 tests of `gtest/hist/*` + `gtest/tree/tree/*`), the kind **59 -> 60**; nothing lost. Gate `TOTAL 119689 0 30000 0 100%` | 2: `W0-137` (the first whole figure for `roottest/root/{hist,tree}/*`: 313 tests, 138 PASS), `W0-138` (**the round's own step 0 assumes a `main` that is 238 commits behind this branch**) |
| 56 | `9f4395db` | **one commit**: an array of objects is destroyed as its scope ends - `cppstd.objects.ObjectArray`, `_destructible` asks the element, and `std::make_unique<A[]>(n)` builds the array `A a[n]` is (`4f884ec7`) | **gtest +1: `gtest/core/foundation/testMake_unique` PASSES**, and **the kind measured whole: 61 PASS / 211 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r56/gtest`) - five gained over round 53's 56, **none lost**. Gate `TOTAL 119714 0 30014 0 100%` | 2: `W0-139` (the bulk-fill tests near the 300 s wall: `ntuple_extended` 133 s, `testBulkApiMultiple` TIMEOUT), `W0-140` (an array of *arrays* of objects is still not destroyed) |
| 57 | `44eb8a61` | **one commit**: a ROOT number type written as a function is the C++ cast - `TFormula` and TMVA's FDA parser both take `formula.functions.CASTS` (`b42926d0`) | **0 harness PASS**, but TMVA's own default split expression `UInt_t([EventNumber])%UInt_t([NumFolds])` now parses: `gtest/tmva/tmva/crossvalidation/*` runs real cross-validation (4.6 s, 2.8 s) and both stop at `ROOT has cv_milli_t`. Nothing lost in 66 `gtest/hist/*` + `gtest/tmva/*` (`/tmp/r57/areas`). Gate `TOTAL 119732 0 30014 0 100%` | 2: `W0-141` (a function-local `using` alias of a template instance is not substituted in a value position - and then TMVA's Reader), `W0-142` (a cast to an integer does not make a following `/` integer, and must not for `TTree::Draw`) |
| 58 | `1fc32254` | **one commit**: an array of *arrays* of objects destroys its rows and their elements - every level of an object array is an `ObjectArray` (`ad8c97a8`) | **0 harness PASS, 0 lost**, and **the whole gtest kind re-measured: 61 PASS / 211 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r58/gtest`), identical to round 56's - the `W0-63` check across rounds 57-58. Gate `TOTAL 119730 0 30012 0 100%` | 2: `W0-143` (`Take<std::vector<bool>>`'s rows are NumPy slices, and the kind the emitted call carries is dropped by `frame.Take`), `W0-144` (`TMultiGraph::GetHistogram`'s log-scale range: 79.9602 and 0.65234 against our 65.5672 and 1.08468); `W0-140` closed |
| 59 | `74caab18` | **one commit**: `Take` of a column of vectors is a vector of vectors - `_Method` carries the template arguments a booking named, `RResultPtr` keeps them, `_vector_of` builds `std::vector<std::vector<T>>` from a `Jagged` (`0c58dd9e`) | **gtest +1: `gtest/tree/dataframe/dataframe_take` PASSES** (all three cases), the kind **61 -> 62**; nothing lost in 109 `gtest/tree/{dataframe,ntuple,treeplayer}/*` programs (`/tmp/r59/df`). Gate `TOTAL 119736 0 30014 0 100%` | 1: `W0-145` (`Take<V>` builds its rows from the row's dtype, not from the element `V` names); `W0-143` closed |
| 60 | `0e88415c` | **one commit**: `TServerSocket`, `TInetAddress` and `ESocketBindOption` over Python's `socket`, with the socket as a dialect guard (`6e981a56`) | **gtest +1: `gtest/net/net/nettests` PASSES**, the kind **62 -> 63**; nothing lost in 71 `gtest/net/*` + `gtest/core/*` + `gtest/tree/tree/*` programs (`/tmp/r60/net`). Gate `TOTAL 119809 0 30022 0 100%` | 4: `W0-146` (`ntuple_largefile2` writes 268 M entries - a large-file test), `W0-147` (`Delaunay2D::Interpolate` is not barycentric interpolation), `W0-148` (`gROOT`'s name is `PyROOT` here and `root` in a compiled program), `W0-149` (a null stored into a class-typed local, with the shape of its two-part fix) |
| 61 | `24cfc1ea` | **two targets, one worker and the round's own hands**: `RooExtendedBinding` in `roofit/pdfs/extend.py`, its registry entry, a test file and a docs line (`730bb942`, worker w01 - a `RooAbsReal` whose value is `pdf.expectedEvents(normSet)`, 1.0 un-normalised and the density's integral with a set); `std::span` and `TMath::KNNDensity` (`618f5a98` - a span holds what `&x` makes here, a `Cell`, so `result[i] = ...` reaches the caller's variable, and the density is ROOT's own walk outwards, bit for bit against the oracle); and two translator behaviours (`9d380d3f`: a std typedef written as a call is the scalar cast, and a braced list becomes its container in a *lambda's* body; `a006df48`: the same conversion in every function's prologue broke 53 batched-call contracts, so it was narrowed) | **gtest +2, the kind 63 -> 65**: `gtest/roofit/roofitcore/testRooExtendedBinding` PASSES (`/tmp/r61/reb`) and `gtest/math/mathcore/testKNNDensity` PASSES (`/tmp/r61/knn2`), and **the whole kind measured at `1a7e31ca`: 65 PASS / 206 FAIL / 77 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r61/gtest`) - four gained over round 58's 61, **none lost**. Gate `TOTAL 119899 0 30046 0 100%` (13 073 + 44, `/tmp/r61/gate1b.log`), ruff, mypy 1026 files, maintainability 31 351 functions, components 1025 modules, mkdocs | 5: `W0-150` (**`baseline.json` understates by every PASS since round 43** - `write_baseline` refuses a partial baseline, so no round can record a gain; the shape of the merge mode is named), `W0-151` (a braced list into a *function's* container parameter - belongs at the call site), `W0-152` (**the two walls `TServerSocket` removed moved on**: `RCurlConnection`, and a UNIX-socket path read as a port), `W0-153` (**a worker must not check out the main tree's branch** - `git worktree add -d`), `W0-154` (`testRooRombergIntegrator` FAIL -> UNSUPPORTED already at the base: a header's refusal now comes first) |
| 62 | `62c38856` | **four targets, two workers and the round's own hands, all provable in a probe first**: `_owned` answers the *pointee* of a single-object smart pointer, so `std::unique_ptr<TFile> f{TFile::Open(...)}` writes its file where its scope ends (`a911f526`); `xrdroot run FILE` names `gROOT` `root` as a compiled program's is while the prompt's `TRint` session keeps `Rint` (`0b11a27c`, worker r-62-w01 - `entrylist_enter` asks for `GetEntryList("root/t1","")`); `RooBDecay::DecayType::SingleSided` and its kin on ROOT's five decay classes (`35a387d1`, `8652fbcb`); `TServerSocket` binds a local socket path with the oracle's own answers (`085bc619`); `TBuffer::ResetMap` (`72614fe5`); and `rootsuite baseline --merge`, which records only the PASSes a real run saw (`2d350cdd`, `4e58cf7f`, `c3e7a67b`, `14b61d71` - worker r-62-w02, and `W0-150` is closed) | **gtest +2, the kind 65 -> 67**: `gtest/roofit/roofitcore/testRooTruthModel` PASSES (`/tmp/r62/truth2`) and `gtest/tree/tree/entrylist_enter` PASSES all four cases (`/tmp/r62/ele`), and **the whole kind at `9fe62239`: 67 PASS / 204 FAIL / 77 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r62/gtest`) - two gained, **none lost, no other row moved**. Gate `TOTAL 119943 0 30054 0 100%` (13 085 + 44, `/tmp/r62/gate1b.log`), ruff, mypy 1026 files, maintainability 31 370 functions, components 1025 modules, mkdocs | 5: `W0-155` (`testHistFactoryPlotting`'s next wall: `RooCurve.GetPoint` and the component-sum invariant), `W0-156` (`Accept`/`TSocket`/`TMonitor` for the local-socket family), `W0-157` (a `TF1` read back out of a buffer), `W0-158` (**a `--only` run is not a whole-kind run**: the working copy lacks a sibling test directory's header), `W0-159` (`cintfake` must answer `_destruct` for every guard) |
| 63 | `86001091` | **three targets, two workers and the round's own hands, each priced against ROOT's own source first**: `TFile::Write` now writes the file - a new `WritableFile.checkpoint` running ROOT's `WriteStreamerInfo`/`WriteFree`/`WriteHeader` sequence, repeatable because each tree's record is written once (`64cf43ae`, `212331b0`, `30897365`, `b92d587e`, `564bbb88`); a GDML name's trailing `0x…` is dropped and an element is named by its formula, with `TGeoMixture.GetWmixt`/`GetElement` (`85fb85eb`, worker r-63-w01); and `TMultiGraph::GetHistogram` pads a log frame by ROOT's own rule, with `TGraph::ComputeRange`'s smallest-positive minimum on a log axis (`d28fcf12`, `7a777e90`, worker r-63-w02) | **gtest +2, the kind 67 -> 69**: `gtest/geom/gdml_material_name_collision` PASSES (`/tmp/r63/t-gdml`) and `gtest/hist/hist/testTMultiGraphGetHistogram` PASSES (`/tmp/r63/t-mg`), and **the whole kind at `564bbb88`: 69 PASS / 202 FAIL / 77 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r63/gtest`) - two gained, **none lost, no other row moved**. Gate `TOTAL 120003 0 30060 0 100%` (13 094 + 44, `/tmp/r63/gate3.log`), ruff, mypy 1026 files, maintainability 31 394 functions, components 1025 modules | 2: `W0-160` (`testTChainSaveAsCxx` now reaches the macro `SaveAs` wrote - our own `WriteObject` sentence is the wall), `W0-161` (**a `Write`'s bytes**: ROOT 3642/3916/4315 against our 3617/4059/4548 for no/one/two writes - the shape is ROOT's, the sizes are not) |
| 64 | `fd36b30f` | **three targets, two workers and a measured patch a worker could not commit**: a `std::ofstream` is a guard with a destructor, so a file written in one scope is readable in the same run (`d192a920`); a `#pragma extra_include` header is translated beside the macro, as rootcling reads it (`c9476ae6`) - together `gtest/core/dictgen/dictgen_base`; `std::thread` with `hardware_concurrency`, reachable as `std.thread` (`500efb6a`, worker r-64-w01); and `TObject.Streamer(TBuffer&)` + a record read finished in the object handed over (`2dcebb86`, worker r-64-w02) + the patch w02 measured but could not commit - `char_pointer`/`ObjectAddress` so `(char *)obj` + offset is the object (`2c8150ef`, refined in `724be159` after five unit tests caught the first shape breaking writable titles) | **gtest +3, the kind 69 -> 72**: `dictgen_base`, `test_norootextension` (Seq and MT) and `testTF1Simple` all PASS, and **the whole kind at `3d1aee19`: 72 PASS / 200 FAIL / 76 UNSUPPORTED / 2 TIMEOUT** of 350 (`/tmp/r64/gtest`), with one program `UNSUPPORTED -> FAIL` (`testMapCppName` now runs). Gate `TOTAL 120057 0 30074 0 100%` (13 106 + 44, `/tmp/r64/gate4.log`), ruff, mypy 1026 files, maintainability 31 418 functions, components 1025 modules | 2: `W0-162` (**a reboot wiped `/tmp`** - ROOT's sources and every round's `results.json`; re-cloned and re-measured), `W0-163` (`test_THBinIterator`: 85-170 s of the limit and `THistRange` missing) |
| 72 | `09f96f39` | **two**: `TKDE` (worker r-72-w01 - dense and adaptive bandwidths, the mirror, 39 unit tests at 100 % line and branch; `gtest/hist/hist/testTKDE` goes from a refusal to **3 of 12 cases**) and the RNG's one-at-a-time path (`Sampler.take_one`; `Gaus` 5.56 -> 4.70 us) | **no harness PASS, nothing lost**: `testTKDE` still FAILS on `TKDEFGT`'s `IsPValid` (`W0-171`), and the speed change is a microbenchmark A/B only because the load average was 38-100 all round - the whole-suite gate is **deferred** with its standing two-command command | 3: `W0-171` (**3 of 12 cases; the wall is `TKDEFGT::IsPValid`**), `W0-181` **partly** (the measured split of a `Gaus`), `W0-184` (the cursor that would take the rest) |
| 73 | `f5ae3e96` | **one**, the round's own hands (both workers died: w02 with `getaddrinfo ENOTFOUND gw.lunaroute.com`, w01 with nothing): a `Cursor` over the draws the sampler holds, so an acceptance algorithm's draws cost an array index - `Gaus` 4.70 -> 1.18 us, `Rndm` 1.74 -> 0.46 us | **no harness PASS, nothing lost**: the program behind `W0-181` does **not** move (`testTProfile2Poly` 96.8 s against the base's 96.0 s at the same load - it is bound by its fills), and the gate is green at `TOTAL 124924 0 31320 0 100%` | 3: `W0-171` (**corrected** - `IsPValid` is the test's own p-value test and the dense and binned densities are wrong, 262/100 and 297/100), `W0-181` **partly** (the draw half is finished, the fills are what is left), `W0-185` (a fractionfitter test that passes alone and fails after its siblings, at the base too) |
| 74 | `d6e4f6da` | **none** - both workers died with no file, no commit and an empty log, so the round's hands went to the two rows that blocked the most: `W0-171` and `W0-185`, both decided by measurement | **0 moved, 0 lost, no code changed**: `git diff --stat a09ae83a d6e4f6da` is the ledger alone, so round 73's `TOTAL 124924 0 31320 0 100%` is this round's gate | 2: `W0-171` (**corrected positively** - `TKDE`'s densities agree with the oracle to 1e-16 in the unbinned *and* relaxed-binned paths, so the wall is the comparison: `Chisquare`, `KolmogorovTest`, `TF1::GetHistogram()`), `W0-185` (**the culprit is `tests/test_hist_compare.py`**, one command, and the fit lands in a different minimum with no RNG anywhere in `fit/`) |
| 75 | `ac4ab519` | **none** - w01 produced nothing and w02 was mid-work on `W0-147` (its `delaunay.py` box patch kept, not taken); the round's hands ran `test_tkde`'s comparison against the oracle step by step | **0 moved, 0 lost, no code changed** (`ac4ab519` adds the ledger alone) | 1: `W0-186` (**a formula that calls the same parameterised function twice loses a term** - `[a]*TMath::Gaus(x,[m],[s])+[b]*TMath::Gaus(x,[m],[s])` = 6.592 against ROOT's 0.7748, with `[a]*sin(x)+[b]*sin(x)` and literal arguments both right); `W0-171` re-pointed at it |
| 76 | `16d23537` | **none** - both workers died silent again, so the round's hands took `W0-186`, the reproduction round 75 had left | **`gtest/hist/hist/testTKDE` 3 -> 8 of its 12 cases** (`63c3236d`): a formula's parameter names are ranked in ROOT's `TFormulaParamOrder` (numbers first, then alphabetical) instead of as written, so `SetParameters` binds as ROOT's does; the four left are the writer's `TKDE` refusal; gate `TOTAL 124930 0 31320 0 100%` | 2: `W0-186` **done** (the diagnosis corrected - not parameter cross-talk, the ordering), `W0-187` (**a `TKDE` written to a file**, the last four cases) |
| 77 | `71c966f0` | **none** - the load average opened at 29, five of six rounds' workers had died, and the round's hands had a bounded target; so it took `W0-185`, the flake round 74 had narrowed | **`W0-185` closed** (`57e1094c`): the fraction fitter's fixture drew its data from `gRandom`, which `Chi2Test`'s `"x"` option leaves somewhere else, so the fit landed in another minimum after any of four sibling tests; the fixture now seeds its own `TRandom3` - the five files report 131 passed, the pair in either order 61. Gate `TOTAL 124930 0 31320 0 100%` at `-n 2`, load average 28 | 1: `W0-185` **done** (`gRandom` was the leak, not `fit/`) |
| 78 | `bcfa1c9d` | **none** - the hands priced `W0-187` first, which is what the round did: the layout off the oracle, then a re-pricing of `W0-147` | **0 moved, 0 lost**: `winfo.INFOS["TKDE"]` is ROOT's own 32 elements (version 3, checksum `0xca2d5094`) with a test pinning every one; `W0-147`'s kept patch was applied, measured and reverted - `ROOT::Math::Delaunay2D` is not even a name here, and ROOT's interpolation of it is CGAL's natural-neighbour, not barycentric. Gate `TOTAL 124930 0 31320 0 100%` at `-n 2`, load average 21-32 | 2: `W0-187` **partly** (the layout is in; the writer dispatch and the read-back factory remain), `W0-147` **re-priced** (the name, the box constructor, then Sibson weights) |
| 79 | `323eb620` | **none** - the hands took `W0-187` by hand: the row of members, the 300/500 `kSTL` spelling, and the write path built, measured and parked | **0 moved, 0 lost**: `TKDE.members` pins all 32 members against the layout, `_kind_of` accepts ROOT 6.40's `kSTL` = 300 as well as 500, and the last wall is named - `fSettedOptions` is a `vector<bool>` and `STL_WRITERS` has none. Gate `TOTAL 124934 0 31320 0 100%` at `-n 2`, load average 11-16 | 2: `W0-187` **partly** (two of three pieces), `W0-188` (**`vector<bool>` has no writer**, with the bytes it needs and how to measure them) |
| 80 | `18d1a7c0` | **none** - the hands finished `W0-187`/`W0-188`: `vector<bool>` in `STL_WRITERS`, `record_payload`, `TKDE.root_payload`, `tkde_from` + `register_members` | **gtest +1, the kind 82 -> 83**: `gtest/hist/hist/testTKDE` **PASSES all 12 cases** (8 at round 79), and the round trip is exact - `GetValue(2.0) = 0.289221398512` either side; **ROOT reads our file** with the right class, name, mean and sigma, but its density differs by 0.5-6 % (`W0-189`: a bandwidth input, not an offset) | 1: `W0-189` (**ROOT's density from a file we wrote is 0.5-6 % off** - the members an adaptive estimate is recomputed from) |
| 65 | `3770efa5` | **two targets, one worker each way**: `RooFit::Detail`'s JSON tree (`661761f8` - `roofit/detail.py`'s `JSONTree`/`JSONNode`, and `TVectorD(3)` as ROOT's three zeros) and `TSpline5` with its `SaveAs` macro (`a69bbae4`, `7fc185ce`, worker r-65-w02 - `pyroot/core/splines.py`, ROOT's quintic spline); the third scope, `TGraphMultiErrors` (worker r-65-w01), came back as **a measured diagnosis and no commit**, because the class cannot be written to a file without `winfo.py`, `writer.py`'s `STL_WRITERS`, `graph.py` and `merging/objects.py` (`W0-164`) | **gtest +2, the kind 72 -> 74**: `gtest/roofit/hs3/testJSONInterface` (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221919/jsonif2`) and `gtest/hist/hist/testSpline` (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221919/t2`), and **the whole kind at `7fc185ce`: 74 PASS / 199 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221919/gtest`) - two gained, **none lost**, one program `UNSUPPORTED -> FAIL` (`W0-165`). Gate `TOTAL 122345 0 30656 0 100%` (13 446 + 44, `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221919/gate1.log`); `baseline --merge` recorded **22** stale gtest rows, `ratchet` 0 regressions | 3: `W0-164` (**`TGraphMultiErrors`'s five files**, measured), `W0-165` (`GradientFittingUnit` runs now), `W0-166` (`testHypoTestInvResult`'s `RooStats::` layouts) |
| 66 | `d53bf743` | **three targets, two workers and the round's own hands, every ROOT number measured on the oracle first**: `THistRange`/`TBinIterator` (`219ddd4d`, `pyroot/core/histrange.py` - the four kinds, the x-fastest global bin, and a poly's flow bins numbered from -9 and walked only to -2 because -1 is ROOT's sentinel); `TKDTreeBinning` + the grid a `TH2Poly` is drawn on (`36d64e1e`, `pyroot/core/tkdtree.py`, `pyroot/th2poly.py` - 100-cell x and y axes, one-cell z, `GetNcells` = bins + 9, all measured); `TGraphMultiErrors` (`06a712aa` + `46659435`, worker r-66-w01 - a layer of y errors per kind, each an attributed `TAttLine`/`TAttFill`, `GetErrorY` summing by ROOT's three modes, and a `Write` that lays out the class's own layout and its three object vectors; its session outran the round, so its **files** were taken, `W0-111`); and a RooStats result read from a file (`45e98b6a` + `b4404ebd`, worker r-66-w02 - `pyroot/roostats/readback.py` registers three `register_members` factories) | **gtest +3, the kind 74 -> 77**: `test_THBinIterator` (all twenty cases), `TGraphMultiErrorsTests` and `testHypoTestInvResult` PASS, and **the whole kind at `06a712aa`: 77 PASS / 196 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221954/gtest`, `--xrdroot-src` a checkout of the tip) - three gained, **none lost, no other row moved**; `gtest/hist/hist/*` measured at the base and the tip is 12 -> 14 PASS with the two named and none lost (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221954/areas-base`, `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221954/areas`). Gate **`TOTAL 122525 … 100%`** (13 462 + 44), ruff, mypy 1039 files, maintainability, components | 5: `W0-167` (`testTH2PolyBinError`: a poly's `GetBinError`/`SetBinError`), `W0-168` (`testTProfile2Poly`: `SetErrorOption`), `W0-169` (`testTH1FindFirstBinAbove`: `TRandomMixMax` + `FindLastBinAbove`), `W0-170` (round 65's `RooFit.Detail` moved `testParamHistFunc`'s wall onto `Detail.compileForNormSet`), `W0-171` (`testTKDE`); and `W0-172` from w01's own record (**`TFile.Get` cannot read back an object written to a file that is still open** - `files.py:380` falls through to a `_reader()` that is `None` until the file closes; a 12-line `_WrittenObject` beside the histograms' `_Written` is the verified shape, and `TGraphMultiErrors.Write` carries a per-class workaround meanwhile) |
| 67 | `9380c498` | **three targets, two workers, and a general fix that flips no program**: `TDirectoryFile.WriteTObject` keeps what every write was made from, so a file still open answers `Get` for **any** class (only histograms and trees were kept) - `4c31ab39`+`f21fd3ba`, and `TGraphMultiErrors`' per-class workaround goes (`W0-172`); a `TH2Poly` keeps ROOT's `fSumw2` and answers `GetBinError`/`SetBinError`, with `TProfile2Poly`'s two error rules (`88fd3c5a`, worker r-67-w01, taken from its tree after 65 minutes with no commit - `W0-111`); `TRandomMixMax` and `TH1::FindFirstBinAbove`/`FindLastBinAbove` in ROOT's search order (`287396aa`+`f4b6eca3`, worker r-67-w02) | **gtest +2, the kind 77 -> 79**: `testTH2PolyBinError` and `test_TH1FindFirstBinAbove` PASS, and **the whole kind at `269ca01a`: 79 PASS / 194 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611222223/gtest3`, `--jobs 3`, `--xrdroot-src` a checkout of the tip) - two gained, **none lost, no other row moved**; the general `Get` fix measured on its own at `f21fd3ba` is 77/196/75/2, round 66's figures exactly. Gate **`TOTAL 123019 0 30814 0 100%`** (13 531 + 44, `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611222223/gate3.log`); `ratchet` 0 regressions and the baseline recorded exactly the two PASSes | 5: `W0-173` (a histogram axis' bin labels live in the pyroot wrapper, so a projection has none - `test_projections`' two cases), `W0-174` (`material_units`: a unit system and element physics), `W0-175` (`TRatioPlot` writes a `TCanvas` to a file), `W0-176` (**`testTProfile2Poly` is one method from a PASS and 161 s of its 300 s limit** - speed first), `W0-177` (`TRandomMixMax` is a documented substitute, not MIXMAX's stream); and `W0-168` is **partly**: four of its five cases pass, the program is not closed 
| 68 | `8e1999a4` | **one target, the round's own hands, and two workers that died with nothing usable**: `Axis.labels` reads the axis record each time and `reshaping` copies a kept axis' labels into a projection and a profile (`b02f04cf`, `W0-173`), and the two projections ROOT's own test also wanted exist - `TH3::Project3DProfile` and `TProfile2D::ProjectionXY`, with ROOT's letters read the other way round (`2c3ca9ba`) | **gtest +1, the kind 79 -> 80**: `gtest/hist/hist/test_projections` PASSES all three cases, and **the whole kind at `2c3ca9ba`: 80 PASS / 193 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230007/gtest`, `--jobs 3`, `--xrdroot-src` a checkout of the tip) - one gained, **none lost, no other row moved**; the three programs the round also ran are unchanged (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230007/mine-check`). Gate **`TOTAL 123091 0 30828 0 100%`** (13 535 + 44, `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230007/gate2.log`) | 3: `W0-178` (**a tree's baskets** - `SetAutoFlush`, `kOnlyFlushAtCluster`, `SetBasketSize`, `GetWriteBasket`, and the 102/2 the test wants), `W0-179` (**ROOT's PDF to 1 %**: 13 763 and 13 927 bytes against our 2101 and 8541), `W0-180` (`TAxis.fLabels` is not written); and `W0-176` now carries round 68's measurement of the fast fill: 161 s -> 77 s but two cases broke, so it was **not taken** |
| 69 | `58f7bfe6` | **one target, no PASS, and the round says so**: an axis' bin labels are written in ROOT's own layout and read back (`d2dd4007`, `W0-180`; the layout measured off a file the 6.40 oracle wrote, now `tests/data/labels-6.40.root`, and `TAxis._labels` answers from `fLabels` when a file brought it) | **0 gtest PASS, none lost**: the whole kind at `d2dd4007` is **80 PASS / 193 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230039/gtest`, `--jobs 3`, `--xrdroot-src` a checkout of the tip) and **not one row moved** from round 68's - which is exactly what a write-path-parity change should do; `gtest/hist/hist/*` measured on its own is 17 PASS / 14 FAIL / 2 UNSUPPORTED (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230039/hist`). Gate **`TOTAL 123099 0 30830 0 100%`** (13 541 + 44, `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230039/gate1.log`) | 2 attempts, no rows: `W0-178` (**a tree's baskets** - r-69-w01 died with `backend unavailable` and wrote nothing; ROOT's own numbers for the target were measured this round: **102 and 2** baskets, `GetBasketSize` 11373 and 106) and `W0-176` (**a second fast profile fill, 161 s -> 65 s, bin contents differ again** - not taken, and the row now names the unit test the next attempt must write first) |
| 70 | `c0e28458` | **a measurement round, with no commit of its own**: the whole gtest kind re-measured at the tip, and ROOT's basket arithmetic read for `W0-178` | **gtest +1, the kind 80 -> 81**: `81 PASS / 192 FAIL / 75 UNSUPPORTED / 2 TIMEOUT` of 350 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230117/gtest`, `--jobs 3`, `--xrdroot-src` the round's worktree) - the gain is `testTProfile2Poly`, whose fast fill came from another session's `wip:` commits already on `main` and **is verified here** (all five cases agree with the `TProfile2D`, 144 s); nothing lost, nothing else moved. Gate **`TOTAL 123212 0 30866 0 100%`** (13 563 + 44, `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230117/gate1.log`), which also makes those `wip:` commits gate-clean | 1: `W0-181` (**`testTProfile2Poly` is 144 s of its 300 s limit**); `W0-176` **closed** by that work; `W0-178` **priced** - ROOT flushes when `lnew + 2*nsize + nbytes >= fBasketSize` with `lnew` the *whole* basket buffer (header included, which is why a 106-byte basket holds 5-6 doubles), `SetBasketSize(n)` is `max(n, 100 + name length)`, and six oracle rows reproduce 102/2/6/2/1/101 |
| 71 | `356e28e6` | **one worker's feature and the round's own hands on a tree's baskets**: worker r-71-w01 wrote ROOT's `TFractionFitter` (`2c7a287d` - the fractions of MC templates a data histogram is made of, over this kit's Minuit2, with `GetPlot`/`GetResult`/`GetMFC`); the round put `TBranch::SetBasketSize`, `TTree::SetAutoFlush` and `kOnlyFlushAtCluster` to work - three ROOT calls that were no-ops - so ROOT's one-basket-per-cluster mode now writes its exact 2 baskets of 106 bytes (`1a843f7a`, `dc095839`, `c455a479`) | **gtest +1, the kind 81 -> 82**: `gtest/hist/hist/test_TFractionFitter` PASSES (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230144/w01-check`), and **the whole kind at `c455a479`: 82 PASS / 191 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230144/gtest`) - one gained, **none lost, no other row moved**. The basket work costs no PASS where trees are written: `gtest/tree/tree/*` 6 PASS either side (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230144/tree-before`, `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230144/tree-after`) and `roottest/root/io/tree/*` 18 / 0 / 0 / 2 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230144/rhist-after`). Gate **`TOTAL 123551 0 30928 0 100%`** (13 602 + 44, `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230144/gate3.log`) | 3: `W0-182` (**ROOT re-optimises basket sizes at every cluster flush** - that is why ROOT's own fixture has 1 x 102, 99 x 4, 1 x 2 and 1 x 500 baskets and this kit 250), `W0-183` (`evolution` needs the LinkDef `#pragma read` schema rule - measured by worker r-71-w02, whose column work was **reverted** for 154 uncovered statements), and `W0-178` **partly** with the count measured |
| 81 | `3e5a19a5` | **none** - no workers: the round opened at a load average of 20-30 after a reboot, and the target priced itself in one probe; the round's own hands wrote it | **gtest +1, the kind 83 -> 84**: `gtest/hist/hist/testTGraphsa` **PASSES all 10 cases** (`$HOME/.cache/xrdroot-runs/ralph-round-202611231323/tgraph`: `{"gtest": {"PASS": 1, "FAIL": 1, ...}}`, the FAIL being `testTGraphSorting`, untouched), because `TGraph::SaveAs` now writes `.csv`/`.tsv`/`.txt` with ROOT's three column sets and its `title`/`asroot`/`native`/`errors`/`asymmerrors`/`noerrors` options, and `TGraph(filename, format, option)` reads one back - and **all ten files are byte for byte what the 6.40 oracle writes** for the same graph (`.../probe/fmt/oracle` against `.../ours`, `cmp` on every one). Gate `TOTAL 125099 0 31362 0 100%` (13 822 + 44, `.../gate3.log`) at `-n 2` under a load average of 24-30 | 2: `W0-190` (`testTH1sa` is a **translator** wall first - a file-scope `struct TestSaveAs` is not a name - and `TH1::SaveAs` second, its bytes measured on the oracle: `fNcells` lines with the flow bins, the file's own delimiter in the header), `W0-191` (ROOT's `%lg` format and delimiter-option reader, refused by name since nothing in the suite asks for it) |
| 82 | `cbf3803f` | **one target, three commits, and the whole gtest kind measured at the base**: ROOT's own `test_TH1_SaveAs` - a class body's statics and enumerators read the body's own name (`cbf3803f`), `std::ios` and the open modes a stream is given (`6dd2a7de`), and `TH1::SaveAs` as `.csv`/`.tsv`/`.txt` with `TH1::GetBinUpEdge` (`24918aa8`) | **`gtest/hist/hist/testTH1sa` goes from a translation refusal to `{'failed': 1, 'passed': 3}`** (`$HOME/.cache/xrdroot-runs/ralph-round-202611231402/th1sa`), and the three files it writes are **byte for byte what the 6.40 oracle writes** for the same histogram; **the whole gtest kind at the round's base** (`…/gtest-base`, `2fa1a7a9`) is **84 PASS / 189 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350, the first whole figure since round 71 and exactly what rounds 72-81's ledger said | 1: `W0-190` **partly** - its translator half is closed, and its one remaining case needs `TH1::SavePrimitive`, the `.C` macro, which is `testMapCppName`'s wall too |
| 83 | `2471ff7f` | **one target, priced from the round-82 ranking**: ROOT's own `test_TH1_SaveAs` - `TObject::SaveAs`'s `.C`/`.cc` macro path with `SavePrimitive`/`SavePrimitiveDraw` (`2471ff7f`), `TH1::SavePrimitive` with `ProvideSaveName`/`MapCppName` and `colors.saved` for `TColor::SavePrimitiveColor`, and the three lines a histogram outside a directory writes (`14bb0ec2`) | **gtest +1, the kind 84 -> 85**: `gtest/hist/hist/testTH1sa` **PASSES all four of its cases** (`$HOME/.cache/xrdroot-runs/ralph-round-202611231440/mine`: `{"gtest": {"PASS": 1, "FAIL": 1, ...}}`, the FAIL being `testMapCppName`), and the macro is **byte for byte what the 6.40 oracle writes** for that histogram (`…/probe`, `diff H1dump.C ours.C` empty). Gate `TOTAL 125285 0 31418 0 100%` (13 850 + 44, `…/gate1.log`) | 2: `W0-193` (ROOT's own `SaveAs` writes a macro for **any** name but a ROOT file - measured: `x.dat` is a macro), `W0-194` (**a pad macro** is what `testMapCppName` needs, not `TH1::SavePrimitive`; `pyroot/graphics/macro.py` has no constructor for `TH2F`/`TH2Poly`/`TProfile`) |
| 84 | `516e979c` | **one target, five names**: ROOT's own `core/clingutils` unit test - `ROOT::TMetaUtils::GetCppName` (the 31-entry encoding every character C++ would not take is spelled as), `GetRealPath`, `ROOT::FoundationUtils::GetCurrentDir`, `gCling` (null, there being no cling here) and POSIX's `symlink` in a new `cppstd/unistd.py` | **gtest +1, the kind 85 -> 86**: `gtest/core/clingutils/coreclingutilstestUnit` goes from two refused cases to both passing (`$HOME/.cache/xrdroot-runs/ralph-round-202611231620/mine`: `{"gtest": {"PASS": 1, "FAIL": 0, ...}}`), and `GetCppName`'s answer is **character for character** the one ROOT's own test pins. Gate `TOTAL 125330 0 31420 0 100%` (13 859 + 44, `…/gate2.log`) | 2: `W0-194` **advanced** (the `MapCppName` encoding `testMapCppName` needs is now here, and ROOT's own file for that canvas was measured: 6646 bytes, 93 lines), `W0-195` (**`gCling` is `None`** - honest, but a dereference of it is a `NoneType` error rather than a full sentence) |
| 85 | `489b800e` | **no workers** - the hands took `W0-155`, priced against the 6.40 oracle first: `RooRealSumPdf.compute` honours ROOT's `isSelectedComp` (a component curve was the whole model), `dependsOn(valueOnly=True)` walks value inputs only as `RooAbsReal::plotOnCompSelect` does, `RooCurve.GetPoint`, and `cppstd.cells.store`/`load` shared with `pyroot.core.refs` | **0 gained, 0 lost**: `gtest/roofit/*` measured at the tip is **9 PASS / 22 FAIL / 16 UNSUPPORTED**, the same 9 as the baseline (`$HOME/.cache/xrdroot-runs/ralph-round-202611231724/roofit`), and `testHistFactoryPlotting` reaches its invariant and stops there (`y2+y3 = 958` against `y1 = 480`). Gate `TOTAL 125363 0 31436 0 100%` (13 863 + 44, `…/gate2.log`) | 2: `W0-196` (**the projection is the wall** - `plotOn`'s component list never reaches `createPlotProjection`, and ROOT's `selectNormalization(&fullNormSet)` is what makes components add up), `W0-197` (`testParamHistFunc` needs the `RooFit::Detail` batch API); `W0-155` **re-priced** with the oracle's 35 points and 2/478/480 |
| 86 | `34c5bb8c` | **no workers** - the hands took `W0-196`/`W0-155`: `plotSamplingHint` (`RooCurve::plotSamplingHintForBinBoundaries`' `delta = (xhi-xlo)*1e-9` either side of every interior bin edge and of both limits) answered by `RooHistFunc`/`RooHistPdf` and merged by `RooProduct`, `RooRealSumPdf`, `RooProdPdf` and the simultaneous projection, the hint being the whole starting grid; and `RooBinWidthFunction`'s `HistFuncForBinWidth` a **shape** server as ROOT declares it | **gtest +1, 86 -> 87**: `testHistFactoryPlotting` FAIL -> PASS, `gtest/roofit/*` **10 PASS / 21 FAIL / 16 UNSUPPORTED**, no other row moved (`…/ralph-round-202611231820/roofit2`); our three curves are the oracle's 35 points, `diff` of the test's body as a macro empty (`…/probe/ours.txt` vs `oracle.txt`). Gate `TOTAL 125406 0 31452 0 100%` (13 877 + 44, `…/gate1.log`, `-n 2`) | 1: `W0-198` (the `-n 4` gate can read 99 % - `commands.py:463->453`, `LOADED` is per worker); `W0-155`/`W0-196` **closed** |
| 87 | `8fb8e0ba` | **a measurement round, no code**: `W0-199` priced before it was believed - ten runs of ROOT's own `rf102_dataimport.py` and nine of ours | **0 gained, 0 lost** (nothing changed): the row was **wrong** - a tutorial that hands RooFit a Python `set` flips in ROOT too (7:3 over ten oracle runs, 4:2 and 2:1 over ours), so the 29 `PASS -> DIFF` rows the round-86 ratchet reported are phantoms. `W0-198` reproduced: the `-n 4` gate reads 99 % on one arc and `-n 2` 100 %, while the same file alone covers that arc - an xdist collection artefact, not a missing test | 1: `W0-200` (the tutorial harness must detect an oracle that varies - two samples find nothing at 1 in 3); `W0-199` closed **as a correction**, `W0-198` sharpened |
| 88 | `aba72d8c` | **two targets, both measured first**: `W0-200` - a tutorial whose own ROOT run varies is `FLAKY`, recorded by `tools/tutorials record-flaky` from repeated oracle runs, and neither a gain nor a loss for the ratchet; and the line-buffering of `xrdroot run` (`cli/run.py::ordered_output`), found while bucketing the rows that stayed | **the roofit tutorial group measured whole**: PASS 124 / DIFF 69 / FAIL 21 / UNSUPPORTED 19 / SKIP 9 / ORACLE-FAIL 4 / FLAKY 9 (`…/ralph-round-202611232004/tut2`), the ratchet's spurious rows **29 -> 20**, and `rf613_global_observables.py` **byte-identical to ROOT's output** where it was the same 14 lines reordered. Nine tutorials recorded as ROOT-nondeterministic. Gate `TOTAL 125429 0 31458 0 100%` (13 885 + 44, `…/gate3.log`) | 2: `W0-201` (`TCanvas::Print`'s one word differs between ROOT's two front ends), `W0-202` (the remaining 20 stdout DIFFs, bucketed); `W0-200` closed |
| 89 | `4b186d80` | **one target, measured before it was believed**: `W0-202`'s 20 rows, bucketed - `tools/tutorials record-order` records ROOT's *own* other draw of a tutorial where only the order of its lines differs, as one more `known_orders` entry, and `record-flaky` keeps the runs whose lines really differ | **14 of the 20 rows are gone**: the 19 listed tutorials measured again are **14 PASS / 5 DIFF** (`…/ralph-round-202611232108/after`) where they were 19 DIFF; `known_orders` grew from 3 tutorials to 22 (492 K), 21 draws recorded. The cause, measured: ROOT prints `rf110_normintegration.py`'s same lines in another order run to run (`affc425ea823` vs `e01d8b4750aa`, ours byte-identical to the second), so the harness's positional comparison failed an output ROOT itself produced | 3: `W0-203` (a degenerate-minimum fit), `W0-204` (Minuit2 `status = 3` where we say 0), `W0-205` (two message lines in the other order); `W0-202` closed, `W0-201` **corrected** (the `TCanvas::Print` word goes to stderr, which is not compared) |
| 90 | `30b2045d` | **one target, priced before it was written**: `W0-204`/`W0-205` probed against the oracle with the tutorial's own data, model and options - and both turned out to be one class, so the round landed the piece that was genuinely missing (`RooRealVar::setVal`'s out-of-range warning) and recorded the rest as a machine class | **0 tutorials flip, and that is the result**: `rf612`'s recovery fit runs on **identical toy data** (mean `3.9831043004248508`, same first five events) to ROOT's `status 3`/`edm 15424.9`/`minNll 29774.18750976386` against our `status 0`/`edm 4.8e-09`/`29650.877364827433` - ours is the better minimum of the same landscape, and Minuit's path down a flat valley is chaotic; `IntervalExamples`' missing warning line is the same thing one ulp down. `84742600` makes `setVal` warn where ROOT warns, word for word, with `enableSilentClipping` as the switch. Gate `TOTAL 125429 0 31458 0 100%` (13 887 + 44) | 2: `W0-206` (ROOT *raises* `invalid_argument` where this kit clips, beyond one epsilon), `W0-207` (five roofit tutorials are pinned to ROOT's machine); `W0-204`/`W0-205` **re-classed** |
| 91 | `c0e181b5` | **one target, then the round's own whole-group run**: `W0-206` - `RooAbsRealLValue.setVal` raises where ROOT raises (`160f81a4`), and the survey it asked for found exactly one internal caller that leaned on silent clipping | **the roofit tutorial group at the tip: PASS 136 / DIFF 57 / FAIL 21 / UNSUPPORTED 19 / SKIP 9 / ORACLE-FAIL 4 / FLAKY 9** (`…/ralph-round-202611232156/tut2`), against round 88's 124 / 69 - rounds 89-90's draws are the first +8, these four the rest. **And the run caught round 90's regression**: its out-of-range warning fired for a NaN and for the `-2.05502e-17` a limited parameter comes back as, so `rf601_intminuit.C`/`.py` and `rf606_nllerrorhandling.C`/`.py` had gone PASS -> DIFF unmeasured. `20460752` keeps a NaN silently and puts an epsilon-out minimiser value back on its limit; the four PASS again. 2 508 roofit/roostats tests pass; gate `TOTAL 125443 0 31464 0 100%` (13 889 + 44) | 1: `W0-208` (`rs101_limitexample` needs `MCMCInterval.GetChainAsDataSet`); `W0-206` closed, `W0-207` re-confirmed |
| 92 | `c1026d40` | **one target, and its wall removed**: `W0-208` - `MarkovChain.GetAsDataSet`, `MCMCInterval.GetChainAsDataSet` and `RooStats::GetAsTTree` (`b3a4106b`), the two bridges `rs101_limitexample` stops at | **0 tutorials flip, and the wall is gone**: `rs101_limitexample.C`/`.py` go from **exit 1** (a missing `GetChainAsDataSet`, then a `RooStats::GetAsTTree` refusal) to **exit 0** printing `plotting the chain data - nentries = 9557` (`…/ralph-round-202611232254/rs101`), and both are DIFF on **one cause** - a `RooArgSet`'s order, which ROOT keeps differently in the minimiser's message and in the result (`W0-209`). Round 91's `_at_limit` returned `Any`, which no gate had asked (`W0-210`). Gate `TOTAL 125497 0 31484 0 100%` (13 894 + 44) | 2: `W0-209` (a `RooArgSet`'s order is not one order in ROOT), `W0-210` (round 91 ran ruff and not mypy); `W0-208` closed |

## Notes

- Write the ledger as you go: it is the only project state that is cheap to maintain and
  impossible to reconstruct.
- Prefer a *measured* claim to a plausible one. Every number in `TODO.md` came from a
  `results.json` named in this file or in its notes.
- When a round yields little, the reason is nearly always that the chosen groups were too big to
  finish in forty minutes: split them further next round rather than widening a worker's scope.
- When a worker reports "the fix is one line, in a file outside my scope", the next round's
  scopes are wrong, not the worker.

## Round 65's close (recorded here for the next iteration)

- Base `3770efa5`, closed at **`7fc185ce`**, and - the first round under the rewritten procedure - **merged to `main` and pushed**, with `git ls-remote origin main` checked.
  The round opened in a worktree of its own under `$HOME/.cache/xrdroot-runs/ralph-round-202611221919` (not `/tmp`, which the last reboot wiped, `W0-162`), integrated
  there, and tore the worktrees down after the push. **Two gtest PASSes, the kind 72 -> 74**, and the whole kind measured at the tip:
  **74 PASS / 199 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`$R/gtest`, `--jobs 3`, `--xrdroot-src $R/head/src` frozen at `7fc185ce`) - two gained,
  **none lost to code**, and one program moved `UNSUPPORTED -> FAIL` (`gtest/math/mathcore/GradientFittingUnit`, which is another session's tree, not this round's work,
  `W0-165`).
- **`661761f8`: `RooFit::Detail`'s JSON tree.** ROOT 6.40's HS3 writers build a document node by node through `RooFit/Detail/JSONInterface.h` - a `JSONTree` that owns it and a
  `JSONNode` for each value - and the test that asks for them is a map of maps of a `TVectorD`. `roofit/detail.py` is those two classes over Python's `dict` and `list`, with
  each child kept as a node so `rootnode["a"]["b"] << 1` and `rootnode["a"]` name the same thing, and `writeJSON` writing ROOT's text. `TVectorD(3)` is ROOT's
  `TVectorD(Int_t n)` (three zeros) now, not the one-element vector `[3]`. **`testJSONInterface` PASSES.**
- **`a69bbae4` + `7fc185ce` (worker r-65-w02): `TSpline5`.** ROOT's quintic spline, transcribed from `hist/hist/src/TSpline.cxx` - the b1/e1/b2/e2 end-point options, the triple
  knot's finite-difference slope - and `SaveAs` writing the C function `splinePrecisionTest` that the test reads back and runs through `ProcessLine`, whose result it casts to
  `double (*)(double)`. **`testSpline` PASSES**, and `ProcessLine` of a function definition followed by its name already worked.
- **The third scope is the round's other product: `TGraphMultiErrors` measured, not built.** Worker r-65-w01 found that the class is not enough - the writer refuses it for a
  missing streamer layout in `winfo.py` and then for three `STL_WRITERS` (`vector<TArrayD>`, `vector<TAttFill>`, `vector<TAttLine>`), `Graph.new` has no path for layers, and
  `mergeable` would silently merge the class as a plain `Graph`. It left the recipe (checksum `0x1a63990a`, the elements, the arithmetic of the three sum modes) and stopped
  rather than edit outside its list, which is the behaviour the brief asks for. **`W0-164`** carries it.
- **`W0-166` is the next cheap-looking read**: `testHypoTestInvResult`'s committed file names `RooStats::HypoTestInverterResult` and friends in its streamer list, our
  `inverterresult.py` already has every member the test calls, and the reader needs the layouts plus a `Source` from a `TFile` before it can harvest them.
- **The gate is on a bigger tree than round 64's**: **`TOTAL 122345 0 30656 0 100%`** (13 446 + 44) against round 64's 120 057 - another session's work has landed in `main`
  since - and ruff, mypy (1035 files), maintainability (32 237 functions) and components (1035 modules) are clean.
- Worktree discipline: three worktrees, each on a branch of the round's own (`r-01`…`r-03`), the round's integration branch `$ROUND-main` cut from `main`, and the four acts in
  the procedure's order - merge, push, verify, tear down. The worker branches that did not land are gone with their worktrees; `W0-164` is their evidence.

## Round 66's close (recorded here for the next iteration)

- Base `d53bf743`, closed at **`d643fe47`** (the code; the ledger, the queue rows and the three
  baseline rows follow in the commit after it). Round dir
  `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221954`. **gtest 74 -> 77**, three gained and **no PASS lost**: the whole kind measured at the tip
  (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221954/gtest`, `--xrdroot-src` a checkout of it) is **77 PASS / 196 FAIL / 75 UNSUPPORTED / 2 TIMEOUT**
  of 350, with `gtest/hist/hist/*` measured at the base *and* the tip (12 -> 14 PASS, the two named,
  none lost - `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221954/areas-base` vs `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221954/areas`) because that is the area the round touched.
- **`W0-163` closed**, and the program is the kind of target worth taking whole: `THistRange`/`TBinIterator`
  (`219ddd4d`) plus `TKDTreeBinning` and a `TH2Poly`'s grid (`36d64e1e`) - all twenty cases pass, ~85 s of CPU
  of the 300 s limit, and it passed a harness run at a **load average of 103**, which is a margin `W0-109`
  would not have predicted. Every number was measured on the 6.40 oracle first: the four range kinds, a
  narrowed axis range, the x-fastest order of `TH2D`/`TH3D`, and - the one that could not be reasoned out -
  a poly's `kUnOfBins`/`kAllBins` walk **-9 .. -2**, stopping before the bin numbered -1, because -1 is
  ROOT's own end sentinel. Read the oracle, not the header.
- **`W0-164` closed** from worker r-66-w01's tree (`06a712aa`, refined by its own last commit `46659435`):
  a pyroot `TGraphMultiErrors`, its layers, its three summation modes, and a `Write` whose closure rule is
  "an array enters a file beside the class that keeps one" - which is also why a `TH1F`'s `TArrayF` base
  stays out. The worker was **still running at 90 minutes and had already made the program pass**; its files
  were copied out (`W0-111`) and its later commit re-taken for `writer.py`. Its own record names the general
  fix it could not make in scope: **`TFile.Get` of an object written to a file still open** (`W0-172`).
- **`W0-166` closed, and its queue row's diagnosis was wrong in a useful way** (`45e98b6a`, `b4404ebd`,
  worker r-66-w02): a `TOther` for a RooStats result is **not** a missing streamer layout - the file describes
  the class and `interp` reads every member - it is a missing `register_members` factory in `pyroot/`.
  Three factories later, `testHypoTestInvResult` answers ROOT's seven numbers from ROOT's own file, which is
  now `tests/data/hypotestinv-6.21.root`. **Check the shape of a `TOther` before writing layouts for it.**
- Two workers, and both outran the timebox: w02 handed back "blocked: timebox" with its work committed and its
  target passing, w01 never committed until after its files were taken (`W0-111`, second round running). Both
  scopes were *whole classes*, which is more than one worker finishes in forty minutes: the round's own two
  targets - a class whose numbers could be read off the oracle - were the ones that landed on time.
- New queue rows, each measured: `W0-167`..`W0-171` are five one-program features in `gtest/hist/hist/*`
  and roofit, each named by its exact missing method; `W0-172` is the general `TFile.Get` gap.
- Gates at `d643fe47`: **`TOTAL 122803 0 30742 0 100%`** by the two-command decomposition - 13 494 passed
  then 44 passed (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611221954/gate5.log` for part 1; the round's earlier runs named its own two gaps first) -
  ruff clean, mypy 1039 files, maintainability 32 513 functions in 1779 files, components 1039 modules /
  0 undeclared, `mkdocs --strict` built. One torch test failed in one part-1 run at a load average of 40
  and passes alone (37 passed in 3.2 s) and in the run quoted; `ratchet` on the round's own whole-kind run
  reports **0 regressions** and the baseline recorded exactly the three PASSes.

## Round 67's close (recorded here for the next iteration)

- Base `9380c498`, closed at **`269ca01a`** (the code; this ledger commit and one tests commit follow).
  Round dir
  `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611222223`. **gtest 77 -> 79**, two gained and **no PASS lost**: the whole kind at the tip
  (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611222223/gtest3`, `--jobs 3`, `--xrdroot-src` a checkout of it) is **79 PASS / 194 FAIL / 75 UNSUPPORTED /
  2 TIMEOUT** of 350, and nothing else moved.
- **`W0-172` closed, and it is the round's general fix**: `TDirectoryFile.WriteTObject` now keeps what
  every write was made from, so a file still open answers `Get` for a `TGraph`, a `TObjString`, anything -
  ROOT appends what it writes and answers for it while its bytes are still in reach. Only histograms and
  trees were kept before, so `Get` fell through to a reader that is `None` until the close
  (`TypeError: 'NoneType' object is not subscriptable`). **It flips no program on its own** (the whole kind
  at `f21fd3ba` is 77/196/75/2, round 66's figures exactly) and that is the `W0-63` check stated up front:
  it is a correctness fix whose value is that the workaround beside it is gone and the next write-then-read
  test works.
- **Two workers, both of whose targets were whole classes, both landed** - `testTH2PolyBinError`
  (`88fd3c5a`, four cases that needed ROOT's `fSumw2`) and `test_TH1FindFirstBinAbove` (`287396aa`+`f4b6eca3`,
  `TRandomMixMax` plus both bin searches in ROOT's search order). w01 took **65 minutes with no commit** and
  its files were copied out and gated here (`W0-111`, the third round running); w02 committed twice and its
  commit says honestly that `TRandomMixMax` is ROOT's name over a stream that is not MIXMAX's (`W0-177`).
- **`W0-168` is partly done and `W0-176` says why it is not closed**: w01's `TProfile2Poly` error option is
  ROOT's and four of the program's five cases now agree with the `TProfile2D` filled beside them, but the
  fifth wants `TProfile2D.GetNumberOfBins` and the program fills 1e6 points **five** times - **161 s alone**,
  and it TIMEOUTed once at `--jobs 3` beside a gate run (`…/gtest2`) and FAILed when measured alone
  (`…/prof-solo`, `…/gtest3`). A program that close to the wall would lose the PASS it earns: **speed first.**
- **A measurement lesson, again mine**: that TIMEOUT was caused by running the coverage gate *beside* the
  kind run (13 500 tests of contention). Every number this round finally quotes was taken alone (`W0-44`).
- New queue rows: `W0-173` (bin labels in the wrapper, not the axis record, so projections lose them -
  `test_projections`), `W0-174` (`test_material_units`: units and element physics), `W0-175` (`TRatioPlot`
  writes a `TCanvas`), `W0-176` (`testTProfile2Poly`, speed then one method), `W0-177` (MIXMAX's real stream).
- Gates at `269ca01a` (the code; the two commits after it are tests and the ledger): **`TOTAL 123019 0 30814 0 100%`** (13 531 + 44), ruff clean, mypy 1039 files,
  maintainability 32 455 functions in 1778 files, components 1039 modules / 0 undeclared, `mkdocs --strict` built.

## Round 68's close (recorded here for the next iteration)

- Base `8e1999a4`, closed at **`2c3ca9ba`** (the ledger and the baseline row follow). Round dir
  `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230007`. **gtest 79 -> 80**, one gained and **no PASS lost**: the whole kind at the tip
  (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230007/gtest`, `--jobs 3`, `--xrdroot-src` a checkout of it) is **80 PASS / 193 FAIL / 75 UNSUPPORTED /
  2 TIMEOUT** of 350, and nothing else moved.
- **`W0-173` closed, and its row was wrong about the size of the job in a useful way**: the labels lived in
  the axis record all along (`TAxis.SetBinLabel` wrote them there) but the engine's `Axis` value read them
  once, when the histogram was booked and before a program had named anything, so a projection answered
  `GetLabels() == None` and `test_projections` died on `'NoneType' object has no attribute 'GetEntries'`.
  `Axis.labels` is a property now, read from the row each time as ROOT's `fLabels` belongs to the axis, and
  `reshaping` copies a kept axis' labels into the projection and the profile. **A property, not a field, is
  the whole difference between a snapshot and ROOT's behaviour** - and the same mistake is worth looking for
  anywhere else this kit reads a member once at construction.
- **The program wanted two more projections than the row knew**: `TH3::Project3DProfile` and
  `TProfile2D::ProjectionXY` (`2c3ca9ba`). Both are ROOT's own arithmetic taken from its source, and the
  first has a trap worth recording: **ROOT reads a projection's letters the other way round** - the first
  along y, the second along x - which `Project3D` here already did and the new profile did not, and the
  labels caught it because they are attached to an axis rather than to a position.
- **Both workers died with nothing usable, and that is the round's other lesson** (`W0-111`, `W0-84`): the two
  `pi -p` processes were gone within half an hour with empty logs. w02 wrote nothing at all (its target,
  `testTTreeCluster`, is `W0-178` with the whole spec). w01 wrote 102 lines - a profile `Fill` that reads its
  target arrays out of the members once - which took `testTProfile2Poly` from **161 s to 77 s** and then broke
  `BinContentCompare` and `BinErrorMeanCompare`, so **it was measured and not taken** and `W0-176` now says
  exactly what is wrong with it (the sums are not `fill_one`'s). **Check the workers' processes early**; a dead
  `pi -p` looks exactly like a busy one until the log is read.
- New queue rows: `W0-178` (a tree's baskets, spec in full), `W0-179` (ROOT's PDF to 1 %, two programs, a
  `TPDF` transcription), `W0-180` (`TAxis.fLabels` written - the half of labelling a projection does not need).
- Gates at `2c3ca9ba`: **`TOTAL 123091 0 30828 0 100%`** (13 535 + 44 passed), ruff clean, mypy 1039 files,
  maintainability 32 467 functions in 1778 files, components 1039 modules / 0 undeclared, and `ratchet` reports 0 regressions
  with the baseline recording exactly the one PASS. **The gate was run alone**: the round's first attempt ran it
  beside the kind run, which is how round 67 got a TIMEOUT that was not there.

## Round 69's close (recorded here for the next iteration)

- Base `58f7bfe6`, closed at **`d2dd4007`** (the ledger follows). Round dir `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230039`.
  **0 gtest programs gained, 0 lost, and not one row moved**: the kind at the tip is **80 PASS / 193 FAIL /
  75 UNSUPPORTED / 2 TIMEOUT** of 350 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230039/gtest`), identical to round 68's, and `gtest/hist/hist/*` on its own is
  17 / 14 / 2 (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230039/hist`). **That is the claim the round makes** - this was write-path parity, and a change there
  that moved a gtest row would have been the surprise.
- **`W0-180` closed, the other half of round 68's labelling**: `TAxis.SetBinLabel` named a bin in a dict only this
  kit could see, so a histogram of named bins lost its names in the file - ROOT streams `TAxis::fLabels` and this
  kit's writer wrote a null pointer for an element its own layout describes. The layout was *measured*, not guessed:
  the oracle wrote a three-bin histogram with bins 1 and 3 named, and its `fLabels` is a `TList` of **the named bins
  only**, each a `TObjString` whose `fUniqueID` is its bin number - no `nbins+2` list of nulls, which is what ROOT's
  in-memory `THashList` would have suggested. That file is now `tests/data/labels-6.40.root`, and one test compares
  this kit's `fLabels` structure against ROOT's member for member. **A file the oracle wrote is the cheapest
  specification there is**, and it settles a question no amount of reading `TAxis.cxx` would have.
- **Both workers came back empty and the row says why.** r-69-w01 died with `backend unavailable` before writing
  anything (`W0-32`), so `W0-178` is untouched - but I measured ROOT's own answers for it first, and they are in the
  row: `102` and `2` baskets and `GetBasketSize` `11373` and `106`, from the oracle running the test's own fixture.
  r-69-w02 did the speed half of `W0-176` a second time - 161 s to 65 s - and broke `BinContentCompare` and
  `BinErrorSpreadCompare` **again**, and the second failure is the informative one: the *contents* differ, not the
  last bit, so the fast fill changes the means rather than only their rounding. **Two independent attempts failing
  the same way means the next attempt must start from a unit test** that fills a profile both ways and compares
  `GetBinContent`/`GetBinEntries` per bin; the row now says so.
- **`origin/main` moved under the round**: another session took the round's branch while it was being pushed
  (`6ffe734b merge: ralph-round-202611230039-main`) and added three `wip:` commits of its own - a fast
  `TProfile2Poly` fill, quasi-random sequences and a collection member, in the very files `W0-176` and
  `W0-178` live in. They are **not this round's** and are **not measured**; the round's own figures stand at
  `d2dd4007`, and every number above quotes that commit, not the tip it was rebased onto.
- Gates at `d2dd4007`: **`TOTAL 123099 0 30830 0 100%`** (13 541 + 44 passed), ruff clean, mypy 1039 files,
  maintainability 32 472 functions, components 1039 modules / 0 undeclared, `mkdocs --strict` built.

## Round 70's close (recorded here for the next iteration)

- Base `c0e28458` (round 69's ledger, on top of another session's three `wip:` commits), closed at the same commit:
  **this round made no code change of its own, and says so**. Round dir `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230117`.
- **The kind at the tip is 81 PASS / 192 FAIL / 75 UNSUPPORTED / 2 TIMEOUT of 350** (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230117/gtest`, `--jobs 3`,
  `--xrdroot-src` the round's own worktree), one more than round 69's 80, **nothing lost and nothing else moved**.
  The gain is `gtest/hist/hist/testTProfile2Poly` - three of this loop's attempts at its fast fill failed
  (`W0-176`), another session's `wip: profile2poly fast fill` + `wip: quasi-random sequences` succeeded, and the
  round's job was to **verify** it: all five cases pass, and `BinContentCompare`/`BinErrorMeanCompare` now *agree*
  with the `TProfile2D` filled beside them, which is exactly what both failed attempts broke. **A round that
  verifies someone else's fix is doing its job** when the alternative is taking it on trust.
- **The gate at the tip is green: `TOTAL 123212 0 30866 0 100%`** (13 563 + 44). That matters twice: the round's
  own claim rests on it, and those `wip:` commits had **never been gated** by anyone - they are gate-clean, and they
  brought 22 tests with them.
- **`W0-176` closed** (by that work, measured here), and **`W0-181`** queued: the program now costs **144 s of its
  300 s limit** at `--jobs 3` (156 s alone), 48 % of the wall, so a slower machine turns the PASS into a TIMEOUT
  (`W0-149`). The next speed work is `TRandom3.Gaus`/`Rndm` - four calls per point, and `tests/test_pyroot_randoms.py`
  pins the stream - not the fills.
- **`W0-178` is priced and left alone, deliberately.** ROOT's rule is `TBranch::FillImpl`'s
  `lnew + 2*nsize + nbytes >= fBasketSize`, where **`lnew` is the basket's whole buffer, header and key included** -
  which is why a `SetBasketSize(100)` branch (106 bytes, `100 + strlen("branch")`) holds **5-6 doubles** and the
  test's tree has 102 baskets rather than the 71 the entry arithmetic suggests - and `Write` then re-optimises to
  11070. Six oracle rows, one variable at a time, are in the row (102 / 2 / 6 / 2 / 1 / 101). Reproducing it means
  the writer's accounting must include the basket header and honour a size set mid-fill, in **every** written tree:
  one round's single feature with the tree area measured before and after, not an hour at the end of this one.
- **Both workers died silently again - the third round running** (`W0-32`, `W0-84`, `W0-94`): no `pi -p` process,
  no file written, no log line. w01 had `TFractionFitter` (one class, one program) and w02 the split collection of
  an evolved struct. **A round should assume the workers will produce nothing** and put its own hands on the item it
  most wants; this round did, and spent them on the measurement instead.

## Round 71's close (recorded here for the next iteration)

- Base `356e28e6`, closed at **`c455a479`** - the round's seven commits, rebased onto an `origin/main` that had
  gained four more of another session's while the round ran (its `wip: RooWorkspace writing` family and a
  provider refactor). **The figures above are the round's own tree**, measured at its pre-rebase tip, which the
  rebase replayed unchanged; the ledger's shas are the rebased ones, and the review of *those* four commits is
  none of this round's business. Round dir `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230144`. **gtest 81 -> 82**, one gained and
  **no PASS lost**: the whole kind at the tip (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230144/gtest`, `--jobs 3`, `--xrdroot-src` a worktree of it) is
  **82 PASS / 191 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350, and **not one other row moved**.
- **A worker produced a real feature for the first time in four rounds.** r-71-w01 wrote ROOT's
  **`TFractionFitter`** (`2c7a287d`, 1 053 lines with its tests): the fractions of MC templates a data histogram is
  made of, fitted with this kit's Minuit2, Poisson likelihood or `-chi2`, answering `GetPlot`, `GetResult`,
  `GetMFC`. **`gtest/hist/hist/test_TFractionFitter` PASSES**, the area is 19 PASS / 12 FAIL / 2 UNSUPPORTED, and
  its own 31 unit tests pin it. A brief that says "read ROOT's source and take the fixture's numbers from the oracle"
  is what made the difference - both earlier worker-rounds on other targets died with nothing.
- **The round's own hands went on `W0-178` and got one of its two halves exactly.** `TBranch::SetBasketSize`,
  `TTree::SetAutoFlush(n)` and `TTree::kOnlyFlushAtCluster` were **no-ops**; they work now, and ROOT's own fixture's
  one-basket-per-cluster tree writes **2 baskets of 106 bytes, as ROOT answers**. The plain tree is **250 baskets
  against ROOT's 102**, and the oracle says exactly why (`…/probe/kl2.C`): ROOT's baskets are 1 x 102 entries,
  99 x 4, 1 x 2 and **1 x 500** - the second cluster went into one basket because **`TTree::FlushBaskets`
  re-optimises every branch's basket size** (`OptimizeBaskets`) at the cluster end, so 106 becomes 11070. That is
  `W0-182`, the last piece, and it is arithmetic in `TTree.cxx` rather than another guess.
- **A worker's commit was reverted and the round says why** (`W0-183`): r-71-w02 added a writer/reader for a split
  `std::vector` of a class (+140 lines) and a **discovery record worth more than the commit** - the test it aimed at
  cannot pass without ROOT's **LinkDef `#pragma read` schema rule**, because `SetBranchAddress` refuses a pointer of
  another element class without one (`TTree.cxx`'s `HasRuleWithSourceClass`). Its column work left **110 statements of
  `objects.py` and 44 of `rebuild.py` that no test reaches**, and its only user is that test, so the gate named it and
  it went. **The record is the valuable half and the queue keeps it.**
- **The basket change was measured against the two areas that write trees before it was kept**: `gtest/tree/tree/*`
  6 PASS either side, `roottest/root/io/tree/*` **18 PASS / 0 FAIL / 0 DIFF / 2 UNSUPPORTED**, and one program that
  used to time out now runs. A change that touches every written tree earns that measurement.
- Gates at `c455a479`: **`TOTAL 123551 0 30928 0 100%`** (13 602 + 44 passed), ruff clean, mypy 1041 files,
  maintainability 32 611 functions, components 1041 modules / 0 undeclared, `mkdocs --strict` built. The gate named
  its own three gaps first (`branches.py`'s new calls, `store.py`'s `cluster_ends`, `wtree.py`'s two
  `only_at_cluster` returns) and each got its test - including an engine-level one for a column of rows, which is the
  only way to reach that branch.

## Round 72's close (recorded here for the next iteration)

- Base `09f96f39`, closed at **`52a67db6`** (the ledger commit follows). **No gtest PASS moved and none was lost** - and the round's honest
  product is a new ROOT class that now runs plus half of a measured performance gap.
- **`52a67db6` (worker r-72-w01): `TKDE`, ROOT's kernel density estimate.** Dense and adaptive (sparse) bandwidths from ROOT's own rules, the
  mirror at the box's edges, the binned grid - 600 lines in `pyroot/core/tkde.py` with **39 unit tests, 100 % line and branch covered by its own
  file**. `gtest/hist/hist/testTKDE` used to refuse at `ROOT has TKDE`; **3 of its 12 cases pass now** (`tkde_fill_empty`, `tkde_hist_adaptive`,
  `tkde_hist_adaptive_mirror`) and the nine that remain stop on one thing: `IsPValid()`/`ArePsValid()` answer false (`test_tkde.cxx:257`, `:262`,
  `:294`), and those are **`TKDEFGT`'s** - the functional-gradient-tree KDE `TKDE` derives from. That is `W0-171`'s next step, named.
- **`aadfed02`+`aefca47d` is half of `W0-181`**: `Sampler.take_one` hands a draw out and moves the cursor in one call, where two Python calls per number was
  most of what an acceptance algorithm costs. `Gaus` **5.56 -> 4.70 us**, `Rndm` **2.11 -> 1.74 us**, A/B'd in one process at one load; the stream
  is untouched and the seed-twice test in `tests/test_pyroot_randoms.py` is the guard.
- **The round's gate was measured at a load average of 15-26, after the first two commits**: part 1 **13 793 passed, 4 skipped, 46 xfailed, 0 failed**
  in 332 s, part 2 **44 passed**, and its coverage report named exactly one gap - **`trandom.py:220-224`, `Sampler.pop`'s five statements, now dead**
  because `take_one` was its last caller. `aefca47d` removes it, and the re-run on the **final** tree is green: 13 793 passed / 0 failed, then 44 with `--cov-append`,
  **`TOTAL 124902 0 31320 0 100%`** (`/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611230829/gate2.log`).
- **The round could otherwise measure almost nothing, and says so.** The load average was **38-100** all round - another session's two 13-hour `busybox sh`
  loops at ~98 % CPU each, plus `mds` and `mediaanalysisd` - so `testTProfile2Poly` read **246.8 s** against round 71's 156 s alone and proves
  nothing (`W0-44`, `W0-103`). **The whole-suite gate could not be measured** and is deferred to the next iteration with the standing
  two-command command (`W0-106`); what can be stated: `ruff` clean, `mypy` 1057 files, `components` 12 579 imports / 0 undeclared,
  `maintainability` 32 947 functions, the new `tkde.py` at **100 %** line and branch from its own test file, and **295 passed** across
  `tests/test_hist_*.py`, `tests/test_fit_histograms.py`, `tests/test_pyroot_randoms.py` and `tests/test_pyroot_tkde.py`.
- **`W0-184` is the measured handover for the rest of the speed work**: `_next` costs 1.25 us a call and `gauss.one` asks it for 2-3 draws per
  value, so a cursor reading `Sampler`'s held run directly (~0.2 us) would make a `Gaus` about 1.5 us - the same numbers in the same order.
- **`W0-182` was priced and dropped again**: `TTree::OptimizeBaskets` is a two-pass optimisation over every leaf (`bmin` 512, `bmax` 256 000, a
  `memFactor` from the memory budget, a cluster-size multiplication in the second pass, `tree/tree/src/TTree.cxx:7253`) - a big, wrong-by-a-byte
  risk to every written tree, not a round's worth under a load average of 100.
- Worker r-72-w02 produced nothing at all (its worktree was clean and its session had exited), the third round running that a worker with a
  narrowly-scoped `register_members` target returns nothing.
- Worktree discipline: `$ROUND-main` plus two **detached** worktrees (`git worktree add -d`, `W0-153`), w01's files taken from its tree after it
  stopped (`W0-111`), merged to `main` and pushed in the same round.

## Round 73's close (recorded here for the next iteration)

- Base `f5ae3e96`, closed at **`a09ae83a`** (the ledger commit follows). **No gtest PASS moved and none was lost**; the products are a 4-7x faster draw, a
  **negative** measurement that redirects the speed work, and a queue row **corrected** where it was wrong.
- **`a09ae83a`: a draw is read straight out of the run the generator holds.** `Gaus`, `Exp` and the other acceptance algorithms ask for two or three numbers
  per value one at a time, and each one went through the sampler's bookkeeping - a method call, a list index, two counters. `Cursor` reads the run the
  `Sampler` already made and hands the draws out itself; every other way in (`_peek`, `_take`, `set_seed`, `state`, `set_state`, `_settle`) folds it back
  first, so the numbers and their order are the sampler's own. **`Gaus` 4.70 -> 1.18 us, `Rndm` 1.74 -> 0.46 us, `_next` 1.335 -> 0.335 us**, A/B'd in one
  process at one load; round 72's `take_one` and the `pop` before it are gone, the cursor being all that is left of that path.
- **The first shape of the cursor was wrong and the gate said so in four minutes**, which is the round's most useful hour: `sync()` advanced the sampler by
  the cursor's *absolute* offset instead of by how many draws it had handed out, so `_held` went negative and `_held.held()`'s `not` was false - the
  sampler was then believed to hold draws it did not, and `genetic_minimizer`/`mathtools`/`hist_compare` failed with `IndexError`. The fix is one
  subtraction and a `_start` slot. **A fast path over mutable bookkeeping needs the gate, not a microbenchmark.**
- **The negative measurement is worth as much as the positive one**: `testTProfile2Poly` takes **96.8 s** on this tree against the base's **96.0 s** at the
  same load (round 71 read 156 s alone, under a load average of 20+). The fixture is bound by its histogram *fills*, not by its draws, so `W0-181` now says
  what is left: a vectorised `TProfile2Poly.Fill`.
- **`W0-171` was wrong and is corrected by measurement.** `IsPValid()` is not a ROOT method: it is the test's own `pval > pcut`
  (`hist/hist/test/test_tkde.cxx:197`). The adaptive cases pass (chi2/ndf 35/100, p 0.95-1.0); the dense and binned ones give **262.185/100, 296.807/100**
  and p-values of 1e-16, so round 72's `tkde.py` has wrong *fixed-bandwidth* and *binned* densities, not a missing flag. That is a one-paragraph brief now.
- **`W0-185` is a flake measured at the base**: `tests/test_pyroot_fractionfitter.py`'s ROOT-fixture test passes alone (31 passed) and fails after four
  sibling files (`1 failed, 130 passed`), on the base tree exactly as on this one.
- **Both workers died**: w02 with `Connection error.: getaddrinfo ENOTFOUND gw.lunaroute.com` after 78 session events and w01 with nothing at all (an empty
  log, no files touched). That is the fourth round in five with no worker product; the round put its own hands on the item it most wanted, as `W0-111` says.
- Gate: part 1 **13 793 passed, 4 skipped, 46 xfailed, 0 failed** in 157 s (99.75 % by design), part 2 **44 passed**,
  **`TOTAL 124924 0 31320 0 100%`** (`$HOME/.cache/xrdroot-runs/ralph-round-202611230931/gate1.log`), at a load average of 8-12.
- Worktree discipline: `$ROUND-main` plus two **detached** worktrees; merged to `main` and pushed in the same round.

## Round 74's close (recorded here for the next iteration)

- Base `d6e4f6da`, closed at **`d6e4f6da`** plus its ledger commit. **A measurement round: no code changed and nothing was lost.** Both workers died -
  w01 and w02 each finished with no file, no commit and an empty log inside the hour - so the round put its own hands on the two queue rows that blocked
  the most, and both are now decided by measurement rather than by a prediction.
- **`W0-171` is corrected, this time *positively*, and the correction saves the next round hours.** Round 73 said `tkde.py`'s dense and binned densities
  were wrong (chi2 262/100 and 297/100). They are not: `/Users/rcurrie/.cache/xrdroot-runs/ralph-round-202611231103/probe/kde.C` (oracle) and `kde2.py`
  (ours) build the same `TKDE` on the same seeded sample and answer **to 1e-16** at five abscissae - `mean -0.0015353127634251749`, `sigma
  0.99314664628994576`, `GetFixedWeight() 0.23003909040934792`, and `GetValue` 0.13166211870684635 / 0.34132448511757563 / 0.37698421120537456 /
  0.35524712964243105 / 0.13134244350073865 - and again on a 20 000-point two-Gaussian mixture in the **relaxed-binned** path (bandwidth 0.5242035562,
  values 0.00031533709248555 / 0.11030333484325 / 0.00034341877452 / 0.19766084316 / 0.05887354477). **The 1.0644-sigmaRob-n^-0.2 bandwidth, the
  kernel normalisation and the binned counts are all ROOT's.** What is left is `test_tkde.cxx`'s *comparison*: `h1->Chisquare(fkde)` (Neyman,
  `pi/histops.py::Chisquare`), `h1->KolmogorovTest(h2)`, or `TF1::GetHistogram()`'s sampling of `kde->GetFunction(100)`.
- **`W0-185` has its culprit, found in one command**: the fraction fitter's ROOT-fixture test passes alone (31 passed) and beside each of its four
  siblings (48/63/52 passed) - and **only `tests/test_hist_compare.py`** makes it fail (`1 failed, 60 passed`). The fit lands in a *different minimum*
  (`[3.7e-05, 0.283, 0.717]` -> `[0.0236, 0.343, 0.634]`), and `fit/` holds no RNG at all, so it is global state a test sets and does not restore.
- **Nothing was gated because nothing changed**: `git diff --stat a09ae83a d6e4f6da` is `TODO.md` and `.ralph/ROOT.md` alone, so round 73's
  **`TOTAL 124924 0 31320 0 100%`** (13 793 + 44, at a load average of 8-12) is this round's gate as well. The load average this round was 10-22.
- **The worker failure is now four rounds in five**: a `pi -p` worker that is given a class to implement and produces nothing, with an empty log, has
  become the norm rather than the exception. The rounds that produced PASSes this fortnight are the ones whose own hands did the work.
- Worktree discipline: `$ROUND-main` plus two **detached** worktrees, torn down at the close; the round directory keeps the two probes (`probe/kde.C`,
  `probe/kde2.py`, `probe/kde.py`) that decide `W0-171` next time.

## Round 75's close (recorded here for the next iteration)

- Base `ac4ab519`, closed at **`ac4ab519`** plus its ledger commit. **A measurement round again - and this one found a live bug in the formula engine**, in
  the exact place the last two rounds only narrowed to.
- **`W0-171`'s wall is `TF1`, not `TKDE`.** `test_tkde`'s own comparison was run step by step against the oracle (`probe/cmp.C` against `probe/cmp.py`):
  the bandwidth **0.7392972631**, `h1`'s integral 17 483.09972, the three `fkde` values 0.03932448956 / 0.1242266055 / 0.04025029895 and
  **`chi2 = 90.00219999`** - all ROOT's. The one thing that differs is the *mixture* the test fills `h1` from:
  `[a1]*ROOT::Math::normal_pdf(x,[s1],10)+[a2]*ROOT::Math::normal_pdf(x,[s2],10)` answers **0.04415934473** at x = 5 where ROOT answers
  **0.03532777287**. So `tkde.py` needs no change at all; the chi2 262 of round 73 came out of a wrong `h1`.
- **`W0-186` is the bug, and the controls name it**: a formula that calls **the same parameterised function twice** loses a term -
  `[a]*TMath::Gaus(x,[m],[s])+[b]*TMath::Gaus(x,[m],[s])` answers **6.592159749** where ROOT answers **0.7748374289**. Each call alone is right
  (`[a]*TMath::Gaus(x,[m],[s])` = 0.1549674858), a call with *literal* arguments twice is right (`[a]*N(x,1,10)+[b]*N(x,7,10)` = 0.03532777287, ROOT's own
  number), and `[a]*sin(x)+[b]*sin(x)` - a call with no `[parameter]` *inside* it - is right. **The `[name]` arguments of a call are compiled once and the
  second call reuses the first's slots**, which is a bug in `formula/parser.py`'s call node and reaches every `TF1` built as a sum of two parameterised
  densities, `TTree::Draw` included.
- **A second `TF1` bug came out of the same probe and is in the row**: `fkde.Eval(5.)` raises `TypeError: only length-1 arrays can be converted to Python
  scalars` (`pyroot/core/funcs.py:403`) for a `TF1` whose code returns an array, so a sampled function cannot be evaluated one point at a time.
- **Both workers produced nothing usable**: w01 no file, no commit, an empty log; w02 was *mid-work* on `W0-147` when the hour ended - its
  `src/xrdroot/delaunay.py` change gives `Delaunay` a `box` argument (the caller's `xmin/xmax/ymin/ymax`, with `window` reporting the unit square's
  meaning) and its own five tests pass, but the target program was never measured, so it is **not taken**; the patch is kept at
  `$HOME/.cache/xrdroot-runs/ralph-round-202611231111/w02-delaunay-box.patch` for the round that takes `W0-147`.
- **No gate was re-run because nothing changed**: `ac4ab519` adds only `TODO.md` and `.ralph/ROOT.md` to the gated `a09ae83a` tree.
- Worktree discipline: `$ROUND-main` plus two **detached** worktrees, torn down at the close; the round directory keeps `probe/cmp.C`, `probe/cmp.py` and
  w02's patch - the three files that decide `W0-186` and `W0-147` next time.

## Round 76's close (recorded here for the next iteration)

- Base `16d23537`, closed at **`63c3236d`** plus its ledger commit. **The first code round in four**, and it closed the bug the last two rounds only
  narrowed: **`gtest/hist/hist/testTKDE` goes from 3 of its 12 cases to 8.**
- **`W0-186` is closed, and its own diagnosis was wrong in a useful way.** The two-call expression was never losing a term. **ROOT numbers a formula's
  parameters in `TFormulaParamOrder`** - the numbers first, then the names *alphabetically* - and this kit numbered them as they are written, so
  `SetParameters(0.2, 10, 7, 0.8, 10, 7)` bound the values to different names. Measured on the 6.40 oracle:
  `TF1('[a]*TMath::Gaus(x,[m],[s])+[b]*TMath::Gaus(x,[p],[q])')` answers `GetParName` 0..5 as **`a, b, m, p, q, s`** and `Eval(5.)` as
  **9.347778517** where this kit had 0.7748374289; `[slope]*x+[offset]` is **`offset, slope`** and `SetParameters(1, 2)` at x = 3 is **7**. A *predefined
  shape* keeps its own order, which the oracle also says: `TF1('xygaus')` is `Constant, MeanX, SigmaX, MeanY, SigmaY`.
- **`function.py` now ranks the names ROOT's way** (`_places`, and the parse map built from that ranking), so `GetParName`, `GetParameter(name)` and
  `SetParameters` all agree with ROOT. `tests/test_function_formula.py`'s ordering test asserted the *old* order and was **corrected to ROOT's**, with the
  oracle measurement in its docstring - the code was wrong, not the test, and the row says which.
- **The gate's own report is what took the dead helper out**: `_ranked` had no caller once the ranking was inlined, and part 2 named `function.py:115`
  as the one uncovered line. Removed, and the two-command gate is green at the final tree: part 1 **13 793 passed, 0 failed**, part 2 **44 passed**,
  **`TOTAL 124930 0 31320 0 100%`** (`$HOME/.cache/xrdroot-runs/ralph-round-202611231122/gate2.log`), at a load average of 17-23.
- **`W0-171` is now 8 of 12**, and the four that remain are not statistics at all: the `_io` cases fail with our own sentence *a TKDE is not something
  this writer puts in a ROOT file*. That is **`W0-187`**, and it is a writer layout plus a read-back factory - the `W0-166` pattern from round 66.
- **Both workers died again** (no file, no commit, an empty log), the fifth round in six; the round's hands did the work, and the 45-minute fix was the
  one both rounds before it had already reduced to a reproduction.

## Round 77's close (recorded here for the next iteration)

- Base `71c966f0`, closed at **`57e1094c`** plus its ledger commit. **One fix, one flake gone, and a clean gate at a load average of 28** (run `-n 2`, as
  `W0-45` says).
- **`W0-185` is closed, and it was `gRandom` all along.** Round 74 found that `tests/test_pyroot_fractionfitter.py`'s ROOT-fixture test passes alone and
  fails after its siblings, and blamed "someone else's global". This round bisected it to `tests/test_hist_compare.py`'s **four option tests**
  (`Chi2Test`'s `"x"` draws its pseudo-experiments from `gRandom`), and then to the real cause: **the fixture drew its data from `gRandom` too**, so
  whatever state the stream was left in moved the data under the numbers ROOT's own `test_TFractionFitter` gives - the fit landed at
  `0.0236 / 0.343 / 0.634` instead of `0.000037 / 0.2832 / 0.7168`.
- **`57e1094c`** gives the fixture a `TRandom3` of its own - seeded with `gRandom`'s default, 4357, which is ROOT's `TRandom3` default - and hands it to
  every `TF1::GetRandom` (whose signature already takes one). The numbers are unchanged, and the five files together now report **131 passed**, the pair in
  either order **61 passed**.
- **A measurement worth recording for the next round**: the fit itself is stable - running it first in a process gives ROOT's numbers - so the fix is in
  the *fixture*, not in `fit/` or `fractionfit.py`. `ROOT::Fit` has no RNG anywhere (`rg 'gaussian|np.random|rng|random_state' src/xrdroot/fit/` is empty),
  which is why the leak could only be upstream of it.
- **No workers this round, and the close says why**: the load average opened at **29** and stayed 28-35, five of the last six rounds' workers died without a
  file, and the round's own hands had a bounded, sub-second-measurable target. Fanning out would have spent the round's time for nothing.
- Gate at the final tree: part 1 **13 793 passed, 0 failed** (`-n 2`, 377 s), part 2 **44 passed**,
  **`TOTAL 124930 0 31320 0 100%`** (`$HOME/.cache/xrdroot-runs/ralph-round-202611231201/gate1.log`).
- **`W0-187` is still the best target in the queue** and it did not move: a `TKDE` needs a writer layout and a read-back factory for `testTKDE`'s last four
  cases - one gtest PASS - and it wants a round with a quieter machine than this one had.

## Round 78's close (recorded here for the next iteration)

- Base `bcfa1c9d`, closed at **`7d671225`** plus its ledger commit. **One commit: `TKDE`'s streamer info, every number of it measured off the oracle** -
  the first of `W0-187`'s three pieces - and a second target **re-priced** in a way that saves the next round hours.
- **`7d671225`**: `winfo.INFOS["TKDE"]` is ROOT 6.40's own layout - **class version 3, checksum `0xca2d5094`, 32 elements** - read off
  `TFile::GetStreamerInfoList` on a file the oracle wrote (`probe/kdeio.C`, `kdeio2.C`, `kdeio3.C`): the `TNamed` base, five `vector<double>`, a
  `vector<bool>` (`(1, 18)`; `TStreamerSTL::GetCtype()` says so), four enum words, eight bools, three `UInt_t` and seven doubles, each with its title,
  streamer type and size. `tests/test_pyroot_tkde.py` pins all 32 and both STL pairs, so a wrong number cannot slip in.
- **What is left of `W0-187` is exactly two pieces, and the row says so**: the writer's dispatch (a `TKDE` is neither a `WRITABLE` nor a `Record`; the
  cheapest route is a `root_payload` on it, the escape hatch `writer.py:1523` already offers) and a `register_members("TKDE", …)` factory that
  rebuilds a working estimate from those members (round 66's pattern). With those, `test_tkde`'s four `_io` cases pass and the program is a PASS.
- **`W0-147` was applied, measured, reverted and re-priced.** Worker r-75-w02's `delaunay.py` box patch went in, its own five tests passed - and the target
  still refused: `gtest/math/mathcore/testDelaunay2D` fails with **`module 'xrdroot.pyroot.core.rmath' has no attribute 'Delaunay2D'`**, i.e. before any
  arithmetic. And ROOT's answer is not barycentric at all: `Delaunay2D.cxx` has two `DoInterpolateNormalized` bodies and the one 6.40 is built with
  calls **`CGAL::natural_neighbor_coordinates_2`** - Sibson's natural-neighbour interpolation, which is why round 60's barycentric reproduction matched 12
  of 15 values and none on the diagonal. The row now names the first step (the name and the box constructor, then Sibson weights).
- Gate at the final tree: part 1 **13 794 passed, 0 failed** (`-n 2`, 316 s at a load average of 21-32), part 2 **44 passed**,
  **`TOTAL 124930 0 31320 0 100%`** (`$HOME/.cache/xrdroot-runs/ralph-round-202611231217/gate1.log`).
- **No workers again**: the round opened at a load average of 18 and the plan was to price `W0-187` first, which the round's own hands did in twenty minutes.

## Round 79's close (recorded here for the next iteration)

- Base `323eb620`, closed at **`f7c113af`** plus its ledger commit. **One commit, two pieces of `W0-187`, and its last wall reduced to a single named member.**
- **`f7c113af`**: `TKDE.members` answers the row a file holds for one - the `TNamed` base and all 32 members of `INFOS["TKDE"]`, by the C++ name each
  has there, from the state the estimate keeps - and a test pins every name against the layout, so a member the writer would look for and not find cannot
  slip in. And **`writer._kind_of` accepts both `kSTL` spellings**: 300, which ROOT 6.40's own streamer infos use (`ROOT::kSTLany`, measured in
  `core/meta/inc/TVirtualStreamerInfo.h:154`), and 500, which every layout written here before says. One `std::` container is written the same way
  whichever a table says, and a table keeps the spelling the oracle gave it.
- **The write path was built, measured, and parked - honestly.** `record_payload(classname, row)` in the writer, `TKDE.root_payload`, a
  `register_members("TKDE", …)` factory and the option-word builder all went in this round and all came out again: with them, `record_payload` lays out
  `fData`, `fEvents` and `fEventWeights` and then **refuses at `fSettedOptions`, a `vector<bool>`**, because `writer.STL_WRITERS` has no writer for one.
  Shipping unreachable code to keep the 100 % coverage floor would be worse than parking it, so the round says what is left and names it: **`W0-188`**.
- **`W0-187` is now one member deep**, and the round records the exact bytes a `vector<bool>` needs (ROOT's `TStreamerSTL` with `fCtype = kBool`: a count
  and one byte per element) and the measurement that settles them (the oracle's own `kde.root`).
- **`W0-147` was applied, measured, reverted and re-priced** at round 78, and round 79 leaves it as that round left it: `ROOT::Math::Delaunay2D` is not a
  name here, and ROOT's answer to it is CGAL's natural-neighbour interpolation.
- Gate at the final tree: part 1 **13 797 passed, 0 failed** (`-n 2`, 294 s at a load average of 11-16), part 2 **44 passed**,
  **`TOTAL 124934 0 31320 0 100%`** (`$HOME/.cache/xrdroot-runs/ralph-round-202611231231/gate1.log`). The maintainability gate caught `tkde_from` at
  CCN 11 / NPath 768 before it was parked - split, never limited.
- **No workers**: the round opened at a load average of 11 and the plan was to take `W0-187` to completion by hand, which the round did as far as the
  `vector<bool>` wall.

## Round 80's close (recorded here for the next iteration)

- Base `18d1a7c0`, closed at the round's TKDE commit. **One feature, and it is a whole gtest program: `gtest/hist/hist/testTKDE` PASSES all 12 of its
  cases** - from 8 at round 79, and from a refusal at the class at round 72. **The gtest kind is 82 -> 83.**
- **`W0-188` closed**: `writer.STL_WRITERS["vector<bool>"]` is `_bools` - a `u32` count and **one byte per element**, which is `TStreamerSTL` with
  `fCtype = kBool` and exactly how this kit reads one back (`buffer.on_disk` makes a `bool` a `u1`, "as C++ has it"). It was the last member between a
  `TKDE` and a file.
- **`W0-187` closed**, the three pieces the last two rounds had measured: `writer.record_payload(classname, row)` lays a row out from `INFOS`'s
  description of the class; `TKDE.root_payload` is the hook the writer already offers for a class the engine does not own; and `tkde_from` +
  `register_members("TKDE", …)` rebuild the estimate from the members, putting every number the file kept back over what the constructor works out
  (`_arguments`, `_restore`, `_option_words`, `_held_numbers` - split when the maintainability gate named a CCN 11 helper).
- **The round trip is exact**: a `TKDE` written to a `TFile` and read back answers `GetValue(2.0) = 0.289221398512` either side, and the harness run
  (`$HOME/.cache/xrdroot-runs/ralph-round-202611231249/tkde`) reads `{"gtest": {"PASS": 1, "FAIL": 0, …}}`.
- **`W0-189` is the honest limit of it**: **ROOT reads a `TKDE` we wrote** - right class, name, title, `GetMean() 2.625`, `GetSigma() 1.70171482139`,
  `GetFixedWeight() -1` so `Adaptive` came back - **but its density differs from ours by 0.5 % at x=1 and 5.7 % at x=5**, which is a *bandwidth*
  difference, not an offset. The members ROOT recomputes an adaptive estimate from are `fCanonicalBandwidths`/`fKernelSigmas2` (the two
  `vector<double>` members after the doubles), `fSigmaRob`, `fRho` and `fData`; one does not hold what ROOT's own record holds. That does not touch the
  test - which writes and reads with this kit - and it is the next step for a file ROOT has to read.
- The gate's own report named the refusal branch of `record_payload` (`writer.py:1338`) as the one uncovered line after the first run, and a test for it
  was added before the second - `W0-112` again, on new code.
- **No workers**: the round opened at a load average of 12 with a target whose three pieces were already measured and named.

## Round 81's close

**One target, and it was priced before it was written.** The gtest ranking was rebuilt from
round 71's `results.json` (`…/ralph-round-202611230144/gtest`, 191 FAIL rows) sorted by how
many cases each program has left: **27 programs are exactly one failing case from a PASS**, and
the round took the one whose failure was a *bug* rather than a missing engine —
`gtest/hist/hist/testTGraphsa` failed all ten of its cases with
`invalid literal for int() with base 10: 'graph.csv'`, which is `TGraph("graph.csv", "")`
being read as `TGraph(n, x, y)`, and behind it `SaveAs("x.csv")` writing a ROOT file.

**What ROOT does, read off its own source and then off the oracle.** `TGraph::SaveAs`
(`hist/hist/src/TGraph.cxx`) takes a delimiter from the name — `.csv` a comma, `.tsv` a tab,
`.txt` a space — and writes one line per point in one of three column sets: the two of a
graph keeping no bars, x/errors/y/errors for a graph with a bar either way, and the six of an
asymmetric one; `noerrors`, `asymmerrors` and `errors` say which whatever the class is,
`asroot`/`native` puts the point before its bars, and `title` writes a header naming a
**column per value** with the axes' titles in the point's own columns. The header is written
with tabs whatever the delimiter is. `TGraph(filename, format, option)` reads the three
extensions back with 2, 4 or 6 columns, skips a line of fewer than two numbers, leaves a
column a line does not have at zero, and answers a missing file with a zombie graph after
ROOT's `Error` line rather than by refusing.

**The evidence that matters.** Ten files were written by the 6.40 oracle and by this kit for
the same graph — `title`, `asroot`, `native`, `noerrors`, `asymmerrors`, the axes' titles, a
plain graph, an even-bar graph and an asymmetric one — and `cmp` says **all ten are byte for
byte identical** (`…/probe/fmt/oracle` against `…/probe/fmt/ours`). That is the whole of the
writer's fidelity: the format is not "close to" ROOT's, it is ROOT's.

**The measurement.** `testTGraphsa` goes from `{'failed': 10}` to `{'passed': 10}`
(`…/tgraph/results.json`, `--only 'gtest/hist/hist/testTGraph*' --xrdroot-src` the round's
tip), `testTGraphSorting` is untouched at 5 failed cases, and the baseline records exactly the
one row (`10f9bb4f`, the diff is `"gtest/hist/hist/testTGraphsa": "FAIL"` → `"PASS"`). The
kind is **84 PASS** by the ledger — 82 whole at round 71, plus round 80's `testTKDE` and this.

- **A refused corner is better than a wrong file.** A `TGraphMultiErrors` keeps a bar per
  layer and ROOT's text file holds one summed bar, so a file read back loses the layers: the
  class refuses the three text formats by name and points at the `noerrors` option or a
  `.root` file. Nothing in the suite exercises it; the alternative was writing `-1` columns
  that *look* like data.
- **The translator, not `SaveAs`, is what blocks `testTH1sa`** (`W0-190`): the file declares
  a file-scope `struct TestSaveAs` and the translation does not expose it. Measured with
  `xrdroot run …/test_TH1_SaveAs.cxx`: `:13: NameError: name 'TestSaveAs' is not defined`.
- **`TH1::SaveAs`'s bytes were measured anyway**, so the next round does not start from a
  guess: a header only with `title` and **the file's own delimiter** in it (not `TGraph`'s
  tab), then `fNcells` lines for `nbins` bins — the underflow and overflow cells are lines of
  their own, with the extrapolated edges `-1` and `6`, not ±inf — and the error column only
  when `fSumw2` is empty.
- **The machine rebooted mid-round and took `/tmp` with it** — ROOT's sources (`W0-162`
  again). They were re-cloned at `v6-40-04` (434 MB) with the same sparse set, grown by
  `hist/hist/test`, `graf2d/graf`, `core/base/src`, `io/io/{inc,src}` and `geom/test`, which is
  what let the round read `TGraph::SaveAs` and `TGraphErrors::TGraphErrors(const char*, …)`
  from ROOT's own source instead of from memory. Round directories live in `~/.cache` and
  survived.
- **The gate takes two commands and the first one's figure is not the answer**: the first
  command reported 99.56% and aborted a `set -e` script before the second ran. The script now
  runs both whatever the first says, and the green run is
  `TOTAL 125099 0 31362 0 100%` (13 822 + 44, `…/gate3.log`).

**The monitor-rerunnable gate for this tip** - two commands, `W0-106`; the first one's figure
is never the answer, and the second is what makes it 100%:

```bash
cd <round>/main && rm -f .coverage
PYTHONPATH=$PWD/src <repo>/.venv/bin/python -m pytest -q -p no:cacheprovider -n 2 --cov=xrdroot \
  --ignore=tests/test_tmva_learn_networks.py --ignore=tests/test_pyroot_references.py --cov-report=
PYTHONPATH=$PWD/src <repo>/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_tmva_learn_networks.py tests/test_pyroot_references.py --cov=xrdroot --cov-append
```

and `ruff check src tests tools`, `mypy`, `python tools/maintainability.py check`,
`python tools/components.py check`, `mkdocs build --strict` - all green at `3e5a19a5`.

## Round 82's close

**One target, priced from the fresh ranking, and three commits.** The round opened with the whole
gtest kind measured at its base for the first time since round 71 —
**84 PASS / 189 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`…/gtest-base`, `2fa1a7a9`, `--jobs 3`,
`--xrdroot-src` a worktree of the base) — which confirms the ledger's 84 exactly: rounds 72-81
gained two programs and cost none. The 27 programs that are one failing case from a PASS were
then read off that run, and the round took `gtest/hist/hist/testTH1sa`, whose case list promised
three cheap cases at `TH1::SaveAs` and one wall.

**What it took, and what the three commits are.**

- **`cbf3803f` — a class body reads its own static and enumerator by the name the body gives
  it.** ROOT's `test_TH1_SaveAs` declares `static constexpr Int_t N = Nbins + 2;` and this kit
  stopped at `:13: NameError: name 'TestSaveAs' is not defined`: the initialiser was written as
  `N = TestSaveAs.Nbins + 2`, in a class body that runs before the class's name is bound. The
  body's names are a scope of their own now, so a static initialiser or an enumerator value that
  names a sibling reads the sibling — which is the order C++ needs and the body already had. The
  reproduction is nine lines (`tests/pycxx/test_pycxx_programs.py`).
- **`6dd2a7de` — `std::ios`.** The test opens its own file with
  `std::ifstream infile(filename.Data(), std::ios::in)`, and `std::ios` was no name at all here.
  The modes are the words the streams already read their mode from, and `in` is a Python keyword,
  so the names are answered by lookup rather than written as attributes; a name C++ has no mode
  for is refused by name.
- **`24918aa8` — `TH1::SaveAs` as text, and `TH1::GetBinUpEdge`.** Measured off the oracle before
  it was written: a line per **cell**, the underflow and overflow cells included, the cell's two
  edges and its content, and its error only when the histogram keeps squared weights to answer one
  with; `title` puts a header in front written with the file's own delimiter, naming the y axis
  where that axis has a title. `GetBinUpEdge` was missing beside `GetBinLowEdge` and answers the
  axis' own edge, so the flow cells report `-1` and `6` rather than infinity.

**The measurement.** `testTH1sa` is `{'failed': 1, 'passed': 3}` through the harness
(`…/th1sa/results.json`) where it was `UNSUPPORTED` at translation, and the three files
(`H1dump.csv`, `.tsv`, `.txt`) are byte for byte the oracle's three for the same histogram
(`…/probe/th1tx/ours` against round 81's oracle output, `cmp` on each).

- **The last wall is a named feature and it is worth two programs.** `IsGood_C` compares the
  written `.C` macro against **31 lines** (`test_TH1_SaveAs.cxx:129-160`) — `TH1::SavePrimitive`
  as ROOT writes it, down to `TColor::GetColor("#000099")` and the six axis font lines. The same
  `SavePrimitive` is what `gtest/hist/hist/testMapCppName` needs for the four histograms its pad
  holds, so the next round's target is one feature and two programs (`W0-190`'s row carries the
  whole 31-line reference).
- **The whole-kind measurement is cheap when the machine is quiet**: 350 programs in about
  twelve minutes at `--jobs 3` under a load average of 6-16, and it is the only thing that says
  whether ten rounds of individual wins cost anything. The ledger had claimed 84 since round 81
  without a whole measurement behind it; now the number has one.

## Round 83's close

**One target, finished: `gtest/hist/hist/testTH1sa` passes all four of its cases** — the program
that round 81 found refusing at translation and round 82 took to three cases. The last one is
`TH1::SavePrimitive`, and the round is the third in a row whose win is a *writer measured against
the oracle before it was written*.

**What ROOT does, read off its own source and then off the oracle.** `TObject::SaveAs` sends a
`.C` (or `.cc`) name to the object's `SavePrimitive`, wrapped in braces and two `//` comment lines
naming the object and ROOT's version; `TH1::SavePrimitive` writes, in ROOT's order, a blank line,
the constructor, a `SetBinContent` per **non-empty** cell and a `SetBinError` per cell that has
one, `SetEntries`, then only the members ROOT's own defaults leave out —
`SetDirectory(nullptr)`/`SetStats(0)`/`SetOption`, the fill, the line colour as the *channels* it
means (`TColor::GetColor("#000099")` for 602), and three axes' fonts and offsets — then `Draw()`.
The variable is the histogram's name and a count (`h__1`), made a C++ identifier.

**The evidence that matters.** The 31-line macro ROOT's own test compares against is **byte for
byte what the 6.40 oracle writes** for that histogram (`diff H1dump.C ours.C` is empty), and the
four cases pass through the harness (`…/mine/results.json`).

- **A prediction was corrected by measurement.** Round 82's ledger row said `TH1::SavePrimitive`
  would unblock `testMapCppName` too. It does not: that program saves a **`TCanvas`**, and
  `pyroot/graphics/macro.py` — 400 lines that already write a pad again for `TArc`, `TH1C`…`TH1S`,
  `TGraph`, `TPave` and twenty more — has no constructor for `TH2F`, `TH2Poly` or `TProfile`. The
  error is measured, the row now says so, and `W0-194` carries the work with the reference the test
  pins (`EXPECT_NEAR(FileSize, 6644, 200)`).
- **`W0-193` is the fidelity gap the work exposed**: ROOT's `SaveAs` writes a macro for *any* name
  that is not a ROOT file — measured, `h.SaveAs("x.dat")` prints `C++ Macro file: x.dat has been
  generated` — where this kit names only `.C`/`.cc` and still writes a ROOT file for anything else.
  Writing that fall-through means every class needs a `SavePrimitive`, so it waits for the pad and
  the graph.
- **The gate named three uncovered lines and the round covered them rather than shipping them**:
  `SetDirectory(nullptr)`, `SetStats(0)` and `SetOption` are each ROOT's own rule over a member, and
  a test now saves a histogram outside a directory, marked and with an option, to reach all three —
  plus the branch where a name holding `_stack_` keeps the name it has. `TOTAL 125285 0 31418 0
  100%` (13 850 + 44, `…/gate1.log`).
- **A test harness detail worth keeping**: `TH1D.AddDirectory(False)` is what makes `GetDirectory()`
  `None`, so the three lines can be reached at all; ROOT's `ProvideSaveName` and `SavePrimitive` are
  both sensitive to that flag, so the test puts it back in a `finally`.

## Round 84's close

**One program, five names, all read off ROOT's own source.** `gtest/core/clingutils/
coreclingutilstestUnit` refused both of its cases at the first name it asked for; it now passes
both.

- **`ROOT::TMetaUtils::GetCppName`** is a 31-entry substitution table: every character C++ would
  not put in a name is spelled as two letters (`+` → `pL`, `<` → `lE`, a space → `sP`, `[` → `oB`),
  and an empty name or one starting with a digit gets an underscore. That is why ROOT's own macros
  hold variables like `lEoBsPnOaTdAp...`, and ROOT's test pins the answer **character for
  character** — which is what made transcribing the table from
  `core/clingutils/src/TClingUtils.cxx` verifiable rather than plausible. The first argument is a
  `std::string &`, so the dialect's out-parameter table marks it a cell and the runtime writes
  through it, as `TGraph::GetPoint`'s `x` and `y` do.
- **`GetRealPath`** is llvm's canonical path (symlinks followed, `~` expanded) — `os.path.realpath`
  of `os.path.expanduser`, and the test's four symlink chains resolve to the files they name.
- **`ROOT::FoundationUtils::GetCurrentDir`** is the working directory *with the separator on the
  end*, which ROOT's own callers paste a file name onto.
- **`gCling`** is `None`: this kit has no cling, its interpreter being `gInterpreter`, and ROOT's
  test asserts `gCling` is null before an interpreter starts — so the truthful answer is also the
  one that passes.
- **POSIX's `symlink`**, which that test's `<unistd.h>` declares, is a new `cppstd/unistd.py`
  gathered by the plain runtime as C's other library calls are — a dialect-free home, because
  `symlink` is C's, not ROOT's.

- **`W0-194` advanced rather than closed**: the *naming* half of `testMapCppName` is now here (the
  encoding above is exactly what its variables need), and this round measured ROOT's own answer
  for that canvas — **6646 bytes**, 93 lines, `…/probe/cppname/C_Output.C` — so the next round
  knows both the shape to match and the `±200` window it must land in. What remains is
  `pyroot/graphics/macro.py`'s 2-D/poly/profile constructors and ROOT's own pad-macro lines
  (`#ifdef __CLING__`, `gStyle->SetOptFit(0)`, `C->Range(0,0,1,1)`, `C->Modified();
  C->SetSelected(C);`, `Draw("alp")`).
- **`W0-195`**: `gCling` being `None` is honest but a *dereference* of it gives a `NoneType`
  attribute error instead of the kit's full-sentence refusals. No test needs that yet.
- **The first gate run was at a load average of 110** when the gate ran, so the coverage figure was
  re-measured once it settled (see the round's `gate*.log`); nothing was concluded from the
  figure taken at that load (`W0-103`, `W0-109`).

## Round 85's close

**A measured-refusal round, and the measurement is the product.** No test changed verdict: the
`gtest/roofit/*` programs are **9 PASS / 22 FAIL / 16 UNSUPPORTED** at the tip, the same nine the
baseline holds, measured on its own (`…/roofit`). What changed is a piece of ROOT's semantics that
was announced and then ignored.

- **`RooRealSumPdf::evaluate` adds a term only where `isSelectedComp()`**. Ours set a component
  selection, logged it, and summed every term anyway - so every component curve of a HistFactory
  model was the whole model. `compute` now skips an unselected term.
- **`dependsOn(..., valueOnly=True)` walks *value* inputs only** - ROOT's
  `RooAbsReal::plotOnCompSelect` walks its nodes that way, and that is what keeps a HistFactory
  sample's bin width, shared by every sample of a channel, from making its neighbours look
  selected: our walk followed a *shape* input through the shared bin width and pulled the sibling
  sample in. `RooAbsArg.value_servers`/`_walk_value` are that walk; `dependsOn` had accepted
  `valueOnly` and ignored it.
- **`RooCurve.GetPoint(i, x, y)`** is the accessor ROOT's own test reads a curve with;
  `cppstd.cells.store`/`load` are the out-parameter write and read that the wrapper's `refs` and
  the engine's curves now share, with `refs.store` keeping the `Float_t` rounding on top.

**The measurement that decides the next round.** With the components finally honoured the test
reaches its invariant and stops there: `y_sig + y_bg = 958` against `y_full = 480` - the `sig`
curve is the whole model, the `bg` curve its own term. Running the test's own body as a macro on
the 6.40 oracle gives ROOT's answer to compare with: **all three curves have 35 points at the same
x**, `y_full = 479.99999999999994`, `y_sig = 2`, `y_bg = 478`, so the components add to the full
curve within **1 ULP** (which is what `EXPECT_DOUBLE_EQ` allows). Two things follow, and both are
queued:

- **`W0-196`**: `plotOn(..., Components(...))` never reaches `createPlotProjection`'s component list
  here, so the projection is the model. ROOT passes `projectionCompList` there and then calls
  `pdf->selectNormalization(&fullNormSet)` on each selected component
  (`roofit/roofitcore/src/RooAbsReal.cxx:1990`), which is why every component shares the model's
  normalisation and the terms add up. The x-grid is a separate question: ROOT's 35 points are the
  bin edges with the `±3e-9` pair at each edge, because a HistFactory model is piecewise constant,
  where ours refines inside a bin.
- **`W0-197`**: `testParamHistFunc` was priced in the same pass and is not a name - it needs
  `RooFit::Detail::compileForNormSet`, `RooFit::Evaluator` and `BatchModeDataHelpers.getDataSpans`,
  a batch-evaluation API this kit does not model.

- **The gate named three gaps in the round's own code and each was closed rather than excused**:
  `refs._indexed` became dead when `store` moved, so it came out; the value walk's dedup and the
  sum's "term not selected" arc each got a test that reaches them (the first sum test had built a
  sum with one coefficient fewer than its funcs, so `zip` never reached the second term). `TOTAL
  125363 0 31436 0 100%` (13 863 + 44, `…/gate2.log`).

## Round 86's close

Round 86 closed `W0-196` and `W0-155` together, and with them
`gtest/roofit/histfactory/testHistFactoryPlotting` passes - the kind's 86th PASS became its 87th.

**What was actually wrong.** Round 85 had made the sum add only the components a plot selected, and
the test then failed on an invariant: at the last sampled point of the first bin the full model gave
478 while its two component curves gave 2 and 478. The cause was not the selection at all but
**where the curve is sampled**: our grid put points exactly on a bin edge, where a binned function's
value depends on which side you ask about, and the three curves asked differently. ROOT never
samples a bin edge. `RooCurve::plotSamplingHintForBinBoundaries` asks for
`delta = (xhi - xlo) * 1e-9` either side of every interior edge and of both limits, and
`RooCurve::addPoints` takes that hint as the **whole** starting grid (`minPoints = hint->size()`,
`xval = hint`) rather than as extra points on the frame's bins - which is also why ROOT's curve has
35 points and not a frame's worth.

**The names that answer it.** `RooHistFunc` and `RooHistPdf` answer the hint (and answer none for an
interpolated bin, which is smooth); `RooProduct`, `RooRealSumPdf`, `RooProdPdf` and the simultaneous
projection merge their children's, as ROOT's do, through one `merge_hints`. `RooHistFunc` and
`RooHistPdf` share it on `_Histogram`, where `bin_boundaries` already lived.

**The one-line discovery that made the components add up.** `RooBinWidthFunction`'s
`HistFuncForBinWidth` proxy is `valueServer=true, shapeServer=true` in ROOT
(`roofit/roofitcore/inc/RooBinWidthFunction.h:35`), and that shape flag is load-bearing: a
value-only walk - which is what `plotOnCompSelect` uses - stops at the bin width, so the bin width
does not drag a sibling sample's histogram into a component's indirect selection. Round 85 had each
component selecting both samples and the selected one masking it; the flag is what makes
`*histoSig*` select `sig` alone and `*histoBG*` select `bg` alone, as ROOT's own INFO lines say.

**How it was proved.** The test's own body was turned into a macro and run twice - once by the
oracle, once by `xrdroot run` - printing `GetPoint(i, x, y)` for all three curves at every point.
`diff ours.txt oracle.txt` is empty: 35 points each, identical x (2896.4482758620688,
2896.5517241379312, 3000, 3199.9999969999999, ... 6103.5517241379312) and identical y
(`y_full = 479.99999999999994`, `y_sig = 2`, `y_bg = 478`). Nothing about this round is argued from
a tolerance; it is the same digits.

**Nothing lost.** `gtest/roofit/*` was re-run whole at the tip: **10 PASS / 21 FAIL / 16
UNSUPPORTED**, and a row-by-row diff against round 85's results names exactly one change,
`testHistFactoryPlotting` FAIL -> PASS. The four local suites that draw curves
(`test_roofit_pdfplot_curves`, `test_roofit_pdfplot_sums`, `test_histfactory_functions`,
`test_roofit_branchcore_plots`) are 95 passed and 15 xfailed, and ten new tests pin the hint itself.

**What the next round should know.** `W0-198` is new: the gate at `-n 4` read 99 % on
`pycint/interactive/commands.py:463->453`, because the test that reaches a *second* class source
requires a library to have been loaded in the same worker (`LOADED` is per process) - the same tree
at `-n 2` is 100 %. Everything else in `gtest/roofit` is where round 85 left it, and `W0-197`
(`testParamHistFunc`'s batch API) is still the biggest single wall in that family.

**A measurement that is not the round's, but had to be explained.** The roofit tutorials were run
whole at the tip (242 tests): 124 PASS / 78 DIFF / 21 FAIL / 19 UNSUPPORTED / 9 SKIP / 4
ORACLE-FAIL, and the ratchet called 29 of them `PASS -> DIFF: stdout`. They are not this round's:
`rf102_dataimport.py`, run eight times in a row, printed its observables in ROOT's order seven times
and in the reverse order once - a hash set's order, where ROOT keeps insertion order - and the
`ratchet` baseline was recorded on a lucky run. Base-vs-tip diffs of two of the 29 contain only that
swap plus a timing line the harness already drops, and `rf205_compplot.py`'s PNG is byte for byte
the same on the base, at the tip, and under the oracle. Queued as `W0-199`.

## Round 87's close

Round 87 wrote no code, and that is its result: the queue row it was given was **wrong**, and
finding that out took ten runs of ROOT itself.

**The row.** `W0-199` said a dataset lists its observables in a hash set's order where ROOT keeps
insertion order, and that this made 29 roofit tutorials' stdout a coin toss. The first half is
true - this kit's `rf102_dataimport.py` printed `1) x = 0.0174204` four times and `1) x =
9.46654` twice over six runs - but the second is not. The tutorial's own line is

```python
ds = ROOT.RooDataSet("ds", "ds", {x, y}, Import=tree)
```

a **Python `set`**, which PyROOT iterates in whatever order the process's memory has laid the two
`RooRealVar`s out in. ROOT's own output therefore flips as well: ten runs of
`~/.local/root-oracle/bin/python rf102_dataimport.py`, each in a working copy of its own, gave `x =
0.0174204` **seven** times and `x = 9.46654` **three** times - and this kit, at four in six and two
in three, is the same coin at the same rate.

**What that means.** The 29 `PASS -> DIFF: stdout` rows the round-86 ratchet reported are not a
regression and no code should chase them. They are the baseline promising something no
implementation can promise for a tutorial that hands RooFit an unordered container.

**The mistake in the measurement, recorded so it is not repeated.** The first sweep ran the oracle
**twice** per tutorial and flagged **0 of 28** (`…/oracle-varies.log`). At a rate of one in three,
two samples agree five times in nine - so a two-sample sweep is blind to exactly the effect it was
looking for. Five runs is the floor; ten is what settled it.

**`W0-198` reproduced, not settled.** The coverage gate at `-n 4` reads `TOTAL 125406 0 31452 1
99%` with the same single arc missing (`src/xrdroot/pycint/interactive/commands.py:463->453`), twice
in a row at the round's base, and 100 % at `-n 2` on the same tree. It is not a missing test: the
file that reaches that arc alone (`pytest tests/test_pycint_interactive_cli.py --cov`) covers it and
misses four other things instead. So the round records it as an xdist **collection** artefact of
`-n 4`, keeps the row open with that evidence, and notes the likely resolution - make `-n 2` the
canonical command and say so, instead of every round rediscovering it.

The round's own gate, since nothing in the tree changed: **`TOTAL 125406 0 31452 0 100%` at
`-n 2`** (13 877 + 44 passed, `…/gate-n2.log`) and **`TOTAL 125406 0 31452 1 99%` at `-n 4`**
(`…/gate-n4b.log`) - the same tree, the same 13 877 tests, one arc of difference.

**What the next round should know.** `W0-200` is the harness change the measurement asks for:
`tools/tutorials run --oracle-repeat N`, marking a tutorial whose *normalised* ROOT outputs
disagree `FLAKY`, with `ratchet` treating `FLAKY` as neither a gain nor a loss. Until that exists,
the roofit tutorial group's 29 stdout rows cannot be quoted either way - and everything else in that
group (the 13 image diffs, the 21 failures, the 19 refusals) is unchanged from round 86.

## Round 88's close

Round 88 made the tutorial harness stop promising what ROOT cannot, and made `xrdroot run` write in
the order a program writes.

**The queued row, done.** `W0-200`: `python -m tools.tutorials record-flaky PATH… --runs N` repeats
ROOT's own run of a tutorial, keeps each *distinct draw* and writes the sentence that tells two draws
apart; a tutorial so recorded is judged `FLAKY`, quoted with its reason in the report, and **neither a
gain nor a loss** for `ratchet`. A draw is compared as the *multiset* of its normalised lines, because
a `.py` tutorial's `print` and RooFit's message stream are two streams into one file and two unchanged
runs interleave them differently - the same lines in another order are the same draw.

**Nine tutorials are recorded**, each measured over four or five oracle runs: `rf102_dataimport.py`
(ROOT prints `1) x = 0.0174204` seven times in ten and `1) x = 9.46654` three), `rf403_weightedevts.py`
(a different PNG every run), `rf404_categories.py` (the category table's order),
`FourBinInstructional.C`/`.py`, `StandardBayesianNumericalDemo.py`,
`StandardHistFactoryPlotsWithCategories.py`, `StandardProfileLikelihoodDemo.py` and
`StandardTestStatDistributionDemo.py` - the last four draw a different picture every run, because they
generate their own toy data from an unseeded `gRandom`. The roofit group's spurious
`PASS -> DIFF` rows fall **29 -> 20**.

**The fix the bucketing found.** Of the 20 that stayed, one was bucketed to its cause and it was not
nondeterminism at all: `rf613_global_observables.py` has the *same lines* as ROOT's run in a different
order, because a tutorial prints through Python's `print` and the kit prints through its own writer,
and under a pipe - how a harness always runs one - Python buffers `print` until the process ends.
`xrdroot run` now line-buffers its stdout (`cli/run.py::ordered_output`), as one `std::cout` does, and
the tutorial is then **byte-identical to ROOT's** where it differed by 14 reordered lines
(`…/probe/on.txt` against `…/probe/r.txt`). The same reordering is behind other rows of the group; a
new test in `tests/test_cint_run_program.py` pins it.

**The gate**, at `e4f1ff53` plus this close's docs commit: **`TOTAL 125429 0 31458 0 100%`** by the
two-command decomposition (13 885 passed, then 44, `…/gate3.log`, `-n 2`). The round's own first
gate run caught its own breakage, as usual: two of the harness's `known_orders` tests had read
`rf404_categories.py`, which this round recorded as `FLAKY`, so they read a tutorial no recording
calls flaky instead (`rf406_cattocatfuncs.py`) - the gate named it and the tests were repointed, not
loosened.

**What the next round should know.** `W0-201` is one word: `TCanvas::Print` says `png file x.png has
been created` under `root -l -b -q` and `file x.png has been created` under the oracle's Python
(measured both ways this round), and our single message writes the `.C` spelling, so every `.py`
tutorial that prints a canvas differs by it. `W0-202` is the remaining 20 rows to bucket - the method
is settled: run both sides, diff with the harness's own `normalise`, and the cause turns out to be one
file, never twenty.

## Round 89's close

Round 89 took the 20 roofit rows round 88 left, and found that **14 of them were not differences at
all**: ROOT prints the same lines in a different order run to run, so the harness's positional
comparison was failing an output ROOT itself produced.

**The measurement that showed it.** The 19 tutorials were re-run with `--refresh-oracle`: **14 PASS
/ 5 DIFF** where the ratchet had 19 DIFF. That alone would only say "the cache was old", so the
entries were read line by line - and for `rf110_normintegration.py` the cache holds **two** draws of
the same run, `affc425ea823` with `[#1] INFO:Eval -- RooRealVar::setRange(x) …` as its first line
and `e01d8b4750aa` with the same line third. **Ours is byte-identical to the second**, and the
harness compared against the first. RooFit's message stream and a script's `print` are two buffers
into one file, and ROOT interleaves them differently every run - the same fact round 88's `FLAKY`
records cannot express, because `FLAKY` compares *multisets* and says "no promise" where this says
"ROOT produced both, accept either".

**The fix, and its shape.** `tools/tutorials/variance.py::record_order` (and
`python -m tools.tutorials record-order PATH… --runs N`) repeats ROOT's own run and keeps each draw
whose *lines* equal the cached draw's in another order as one more `known_orders` entry - the
mechanism the three original roofit entries already used. A run whose lines *differ* is deliberately
left to `record-flaky`: the two findings are never mixed, and the test says so (`same_order` is the
guard, not `same_output`). 21 draws were recorded over the 19 tutorials and `known_orders` grew from
3 tutorials to 22; measured again, **14 PASS / 5 DIFF** (`…/after`).

**What is left is real, and it is three causes** - recorded as `W0-203` (`rf204b_extendedLikelihood_
rangedFit.C` stops on a different degenerate minimum: `Nsig 1.3553e-08` against `4.5698e-08` with the
same error), `W0-204` (`rf612_recoverFromInvalidParameters.C`/`.py`: ROOT's fit fails with
`Minuit2Minimizer : Invalid minimum - status = 3` where ours succeeds with `status = 0`, which loses
the tutorial's whole subject) and `W0-205` (`IntervalExamples.C`/`.py`: an `[#0]
WARNING:InputArguments` line and an `[#1] INFO:Eval` line in the other order).

**`W0-201` is corrected, not closed.** The `TCanvas::Print` message really does differ between ROOT's
front ends - `root -l -b -q` says `png file x.png has been created`, the oracle's Python says
`file x.png has been created` - but that line is on **stderr**, which the tutorial harness does not
compare, so it never explained a DIFF. Four measurements are in the row.

## Round 90's close

Round 90 took the three roofit rows round 89 left, probed each against the oracle with the
tutorial's own data, model and options - and found that **all three are one class**, which is the
round's result.

**`rf612`'s recovery fit, measured side by side** (a probe of the tutorial's own second fit, 10 000
events, `RecoverFromUndefinedRegions=1.0`):

| | ROOT | this kit |
|---|---|---|
| `status` | 3 (`Invalid minimum`) | 0 (`Valid minimum`) |
| `covQual` | 2 | 3 |
| `minNll` | 29774.18750976386 | **29650.877364827433** |
| `edm` | 15424.923309639917 | 4.8e-09 |
| `a1` | -0.024647611021991755 | -0.4981562848757515 |
| `numInvalidNLL` | 79 | 78 |

The toy data are **byte-identical** on both sides (mean `3.9831043004248508`, the same first five
events), so nothing but the arithmetic differs - and ours reaches the **lower** NLL, i.e. the better
minimum of the same landscape. Minuit's path down a valley floor with a NaN penalty in it is chaotic
enough that arm64's fused multiply-adds land it elsewhere, which the kit's own
`test_a_fit_with_no_valid_value_and_a_curve_that_cannot_be_normalised_say_so_as_root` already
documents ("arm64 took 66 invalid values to x86-64's 64").

**`IntervalExamples` is the same thing one ulp down.** Its two missing lines are ROOT's
`[#0] WARNING:InputArguments -- Value 1 is slightly outside the default range [-1, 1] of the
variable "mu"!` and the sentence after it: ROOT's `BayesianCalculator` produces `mu = 1 + 1ulp` and
gets the warning, ours lands exactly on `1.0` and gets nothing. That is why the round landed the
**warning** - `RooRealVar::setVal` now says exactly what ROOT's `printOutOfRangeWarning` says, with
ROOT's own six-digit numbers and its `enableSilentClipping` switch (`84742600`, and a test that pins
the two lines digit for digit) - and not the `invalid_argument` ROOT raises beyond one epsilon,
which would turn every internal step past a limit into an error (`W0-206`).

**So five roofit tutorials are pinned to ROOT's machine** (`W0-207`): `rf204b_extendedLikelihood_
rangedFit.C`, `rf612_…C`/`.py` and `IntervalExamples.C`/`.py`. They are not failures to fix by
inspection; they need an x86-64 run (or `ROOTS_MACHINE`) before anything is claimed about them, and
the harness has no vocabulary for "machine-dependent" yet - that is the next round's question.

**Nothing else moved.** 2 506 roofit and roostats unit tests pass with the new warning, and the gate
reads `TOTAL 125429 0 31458 0 100%` (13 887 + 44, `…/gate1.log`).

## Round 91's close

Round 91 took `W0-206` - the `invalid_argument` ROOT raises where this kit clipped - and its
whole-group run afterwards caught a regression **round 90 had shipped without measuring**.

**The throw, and the survey the row asked for.** `RooAbsRealLValue.setVal` now raises
`CppInvalidArgument` (our `std::invalid_argument`: a `ValueError` with ROOT's `what()`) when the
value is more than `std::numeric_limits<double>::epsilon()` from the clipped one, with ROOT's own
two lines and the value left alone; within an epsilon it warns and clips, as round 90 built. Of
2 508 roofit and roostats tests, **exactly one** leaned on silent clipping: the scalar half of
`test_roofit_parameter_batches.py`, whose *batched* side never looks at a range at all - ROOT's
`RooBatchCompute` writes the values into a buffer - so the comparison now runs `setVal` under
ROOT's own `enableSilentClipping` switch and keeps its bit-for-bit assertion unchanged.

**What the group run found.** The first whole-group run at this round's tip read 132 PASS / 61
DIFF; four of the DIFFs were ours:

- `rf606_nllerrorhandling.C`/`.py`: ours printed `[#0] WARNING:InputArguments -- Value nan is
  slightly outside the default range …` where ROOT prints its own
  `RooAbsMinimizerFcn: Minimized function has error status but is ignored.` and no
  `InputArguments` line at all. **ROOT keeps a NaN without a word** - that is how a minimiser that
  has walked into an undefined region carries on - and round 90's rule compared `abs(clip - nan) >
  eps`, which is false, so it took the *warning* branch.
- `rf601_intminuit.C`/`.py`: ours warned for `Value -2.05502e-17 is slightly outside the default
  range [0, 1] of the variable "frac"`, where ROOT says nothing. A limited parameter transformed
  by Minuit2 comes back **exactly on its limit**; iminuit's comes back one rounding step past it.

`20460752` is both fixes: `setVal` returns a NaN untouched, and the minimiser's own setter
(`_at_limit`) puts a value within an epsilon of the limit back on it - a value genuinely outside
still raises, which is what catches a caller's mistake. The four tutorials PASS again
(`…/nan2`: 4 PASS / 0 DIFF).

**The numbers.** The whole group at the tip is **PASS 136 / DIFF 57 / FAIL 21 / UNSUPPORTED 19 /
SKIP 9 / ORACLE-FAIL 4 / FLAKY 9** (`…/tut2`) against round 88's 124 / 69: rounds 89 and 90's
recorded draws are the first eight, these four the rest, and FAIL and UNSUPPORTED are untouched.

**The lesson for the next round, and it is the round's own.** Round 90 changed a message the whole
suite prints, quoted a green unit gate, and did not run a single tutorial - so it could not see
that it had turned four PASSes into DIFFs. A round that touches anything RooFit prints must run the
tutorial group it affects, even when no queue row asks for it. `rs101_limitexample`'s
`MCMCInterval.GetChainAsDataSet` is the one new gap the run surfaced (`W0-208`).

## Round 92's close

Round 92 took `W0-208`, the row round 91's whole-group run had surfaced, and removed both walls
the tutorial stands behind - so `rs101_limitexample` now runs to its end.

**The two bridges, from ROOT's own source.** `MarkovChain::GetAsDataSet` hands the chain's points
over as a `RooDataSet` of the parameters, the likelihood as a variable of its own
(`nll_MarkovChain_local_`) and the steps it stayed as the dataset's weight
(`roofit/roostats/src/MarkovChain.cxx:128`); `MCMCInterval::GetChainAsDataSet` is a delegation
(`inc/RooStats/MCMCInterval.h:203`); and `RooStats::GetAsTTree`
(`src/RooStatsUtils.cxx:311`) fills a `TTree` with one `"%s/D"` branch a column, in `std::map`'s
alphabetical order, with `_errlo`/`_errhi` or `_err` beside a variable that carries an error and
`-999` for a branch no entry fills. Both are here now: the dataset in
`roostats/markov.py`/`mcmc.py`, the tree in `pyroot/roostats/asdata.py` (it returns a `TTree`, a
`pyroot` object, so that is where it lives).

**Measured.** Round 91's run of `rs101_limitexample.py` stopped at
`'MCMCInterval' object has no attribute 'GetChainAsDataSet'`; then at a full-sentence
`ROOT has RooStats::GetAsTTree`; now it prints `plotting the chain data - nentries = 9557` and
exits 0 (`…/rs101`, both files exit 0).

**What is left is one cause, and it is a `RooArgSet`'s order.** ROOT prints
`... normalize constraints with respect to the parameters (ratioSigEff,ratioBkgEff)` from its
minimiser and `Nuisance Parameters: RooArgSet:: = (ratioBkgEff,ratioSigEff)` from its result -
**the same two parameters in two orders, alphabetical in the result** - where this kit uses one
order for both. So each file differs in exactly one of the two lines, and `.py`'s `box` plot
follows the nuisance order (similarity 0.7972). That is `W0-209`, and it is a real question about
`RooFitResult`, not about this round's code.

**The round's process finding.** Round 91 reported "ruff clean" and its `_at_limit` returned `Any`
from a function declared `-> float` for a whole round; this round's `mypy` named it
(`minimizer.py:83`, fixed in `b3a4106b`). The gate is seven commands and every one of them is the
gate - `W0-210`.

## Round 93's close

Round 93 took `W0-194` and removed its wall: a canvas saved under a `.C` name is now ROOT's own macro,
and `gtest/hist/hist/testMapCppName` PASSES.

**The row's premise was wrong, in the useful direction.** It said the pad macro needed three
constructors - a `TH2F`, a `TH2Poly` and a `TProfile`. Measured first, as `W0-209`'s round should have
been: this kit's file for the test's own canvas was **1545 bytes** against the 6.40 oracle's **6616**,
because *every* class's text differed. A `TGraph` was ten `SetPoint` calls where ROOT writes two
`std::vector<Double_t>` and hands the constructor their `data()`; a histogram's variable was
`th1f2` where ROOT writes `MapCppName`'s encoding of its name (`lEoBsPnOaTdA...__1`); the canvas had
no `#ifdef __CLING__`, no `gStyle` lines, no palette, no `Modified()` and no `SetSelected()`. The
assertion is `EXPECT_NEAR(FileSize, 6644, 200)`, so nothing short of ROOT's own shape could pass it.

**What is in now.** `TObject::SavePrimitiveConstructor` (the type spelled out once per class, as
`gROOT->ClassSaved` has it - a macro's second `TGraph` is `graph = new TGraph(...)`),
`TObject::SavePrimitiveVector` (ROOT's braced list, ten to a line past fifteen, sixteen digits),
`TObject::SavePrimitiveNameTitle`; `TH1::ProvideSaveName`'s `testfdir` and `_identifier` becoming
`ROOT::TMetaUtils::GetCppName` - which round 84 had already added and nobody had wired in;
`TProfile::SavePrimitive` (the error option in the constructor, `SetBinEntries` before
`SetBinContent`); `TH2Poly::SavePrimitive` (a `TGraph` a polygon, `AddBin`, the bins numbered from
one); `TGraph::SavePrimitive` with ROOT's `th2poly` branch; and `macro.write` as
`TCanvas::SaveSource` + `TPad::SavePrimitive`: the header, the comment naming the canvas, the three
`gStyle` sets, `TColor::SetPalette(57, nullptr)`, `Range`, fill and border, then each object's *own*
`SavePrimitive` (so a histogram, a profile, a `TH2Poly` and a graph write ROOT's text and ROOT's
`Draw`), then `Modified()` and `SetSelected()`.

**Measured.** `testMapCppName` FAIL -> PASS (`…/ralph-round-202611232321/mapcpp2`), and the 57
`gtest/hist/*` + `gtest/graf2d/*` programs measured whole at the tip move **one row and no other**
(`…/hist-graf`, 39 PASS / 14 FAIL / 4 UNSUPPORTED; the diff against `baseline.json` is that row,
raised by `rootsuite baseline --merge`). For the test's canvas ROOT writes **6616 bytes, 93 lines**
and this kit **6598, 93**; the 13 lines that differ are the timestamp, `TCanvas(…, 10, 10, …)` for
ROOT's `(…, 0, 0, …)`, `Draw()` for `Draw("alp")` (`W0-212`), the `TH2Poly`'s own fill defaults and
one axis title offset (`W0-213`). **The macro runs**: `xrdroot run C_Output.C` exits 0 and rebuilds
all five objects, which is what a macro is for.

**Two assertions corrected against the oracle, not loosened.** A histogram named in punctuation is
saved under `GetCppName`'s encoding, not an underscore a character - the round's own test had pinned
the wrong spelling, and ROOT's `C_Output.C` is the evidence. And a round trip through ROOT's own
order (every content, then every error, `Sumw2()` taking the contents it finds as the squared
weights) gives `sqrt(content)` as the error of a bin that had none: ROOT does the same, so the test
says so now instead of asserting a losslessness ROOT does not have.

**`W0-209` priced, and it was not the wall.** `rs101_limitexample.C`'s DIFF has six differing lines,
not one: ROOT draws 9588 chain steps at 47.94 % acceptance and reports MCMC limits 20.1859 / 102.808
where this kit draws 9557 at 47.785 % and reports 18.8057 / 103.058. The harness names the *first*
line, and round 92 read that as the only cause. That is `W0-211`; the parameter order is real and
waits behind it. And `rs101_limitexample.py` is ROOT's own coin - 3 distinct draws in 6 oracle runs -
because the tutorial hands RooFit a Python `set`, so it is recorded in `flaky.json` and the two DIFF
rows round 92 left are now **1 DIFF + 1 FLAKY** (`W0-215`).

**Gate.** `TOTAL 125639 0 31526 0 100%` (13 909 + 44, by the two-command decomposition, `…/gate2.log`
for part 1 at `-n 2` and a `--cov-append` part 2); `ruff` clean; `mypy` clean on 1 061 files;
maintainability 33 220 functions in 1 819 files; components 1 061 modules, 0 undeclared imports;
`mkdocs build --strict` clean. The gate named the round's own dead code - `macro._hist` and its
`HISTS` table, unreachable once every `TH1*` writes itself - which was deleted, and a profile's
`if GetSumw2N()` branch, which no object can reach (a `TProfile` keeps squared weights from its
constructor: measured, an unfilled one answers 4 for 4 cells), so it went too.
| 93 | ``0c6490d9`` | **one target, measured before it was believed**: `W0-194` - the pad macro as ROOT writes it (`TCanvas::SaveSource`'s frame, `TPad::SavePrimitive`'s fill and border, and each primitive's own `SavePrimitive`: `TH1`'s `ProvideSaveName`/`MapCppName`, `TProfile`'s entries before its means, `TH2Poly`'s a-`TGraph`-a-polygon, `TGraph`'s two `std::vector<Double_t>`) | ``ff2ec418`` | **gtest 87 -> 88**: `gtest/hist/hist/testMapCppName` FAIL -> PASS, the 57 `gtest/hist/*` + `gtest/graf2d/*` programs moving one row and no other (`…/hist-graf`); our macro is 6598 bytes and 93 lines against ROOT's 6616 and 93 (`…/probe/pad/full`), and it runs under this kit | 5: `W0-194` closed; `W0-211` (**`rs101` is the MCMC stream, not the parameter order** - six differing lines, not the one the harness names), `W0-212`, `W0-213`, `W0-214`, `W0-215` |

## Round 94's close

Round 94 took the smallest one-case program in the hist area that looked like a name, and found three
general gaps behind it: **`std::is_sorted`, a pointer-typed return from a dialect, and `TGraph::Sort`**.
`gtest/hist/hist/testTGraphSorting` PASSES all five of its cases.

**The measurement first.** The round's own words were `ROOT has std.is_sorted` - a name, which is the
kind of row `W0-55` says a round may spend an hour on. Pricing it took four probes: the program refused
at `std::is_sorted` (all five cases), the *translator* then refused the range (`an iterator this
translator cannot follow back to its container`, because `typeof(graph.GetX())` was unknown), and after
that `TGraphMultiErrors` had no `GetEXlow`. Each wall was one layer of the same question - *can a program
hand a graph's own array to an algorithm* - and each answer is general rather than about that test.

**What is in now.** `cppstd.objects.sorted_range` is C++'s `is_sorted`: no element less than the one
before it, with the predicate a program gives or C++'s own `<`; the translator answers
`std::is_sorted(begin, end[, less])` over the range `iterator_range` already follows back to its
container; `pycxx.infer._named_type` reads a dialect's `method_returns` entry as a *declaration*, so
`"Double_t*"` is a pointer and `Graph.GetX() + n` is a range end rather than an element-wise add; the
whole graph family answers `GetX`/`GetY`/`GetEX`/`GetEY` and the asymmetric and bent bars with
`Double_t*`; `TGraph::Sort` takes ROOT's signature, sorts a stable permutation of the point numbers and
moves every array by it (`UpdateArrays`), sets `kIsSortedX`, and refuses a comparator, a descending
order or a sub-range **by name**; and `TGraphMultiErrors::Sort` carries its layers of y errors and its
own x errors with the points, with `GetEXlow`/`GetEXhigh` as ROOT's overrides of the
`TGraphAsymmErrors` accessors.

**Measured.** The program goes 0 of 5 cases to 5 of 5 (`…/sort5`), and the 57 `gtest/hist/*` +
`gtest/graf2d/*` programs measured whole at the tip move **one row and no other** (`…/hist-graf`: 40
PASS / 13 FAIL / 4 UNSUPPORTED, the diff against `baseline.json` being that row, raised by
`rootsuite baseline --merge`). The kind is **89 / 350**.

**Two rows queued rather than guessed at.** `W0-216`: ROOT's `Sort` also takes a comparator, a
descending order and a sub-range, and the three are now *named* refusals instead of silent wrong
sorts. `W0-217`: ROOT sets `kIsSortedX` because `TGraph::Eval` then binary-searches the points; the bit
is set here and nothing reads it, so `Eval` is still linear - nobody measures it yet.

**Gate.** `TOTAL 125670 0 31532 0 100%` (13 917 + 44, by the two-command decomposition, `…/gate1.log`
at `-n 2` and a `--cov-append` part 2); `ruff` clean; `mypy` clean on 1 061 files; maintainability
33 235 functions in 1 820 files; components 1 061 modules, 0 undeclared imports; `mkdocs build
--strict` clean.
| 94 | `3a99abad` | **one one-case program, priced four times before it was believed**: `gtest/hist/hist/testTGraphSorting` - `std::is_sorted` as a `cppstd` range check and a translator name, a dialect's `method_returns` allowed to name a **pointer** so a graph's arrays are ranges, `TGraph::Sort` with ROOT's signature and `kIsSortedX`, and `TGraphMultiErrors::Sort` carrying its layers and x errors | `127f2320` | **gtest 88 -> 89**: all five cases of `testTGraphSorting` PASS (`…/sort5`), and the 57 `gtest/hist/*` + `gtest/graf2d/*` programs move one row and no other (`…/hist-graf`, 40 / 13 / 4) | 2: `W0-216` (`Sort`'s comparator, descending order and sub-range are named refusals), `W0-217` (`kIsSortedX` is set and `Eval` does not read it) |

## Round 95's close

Round 95 took the gtest kind's one-case list - which it first had to *measure*, because the last one
went with `/tmp` - and closed three queued rows behind two of its entries: **a function's own `using`
names a type, `CrossValidation` is a `TNamed`, and `TGraph::Sort` takes ROOT's whole signature**.
Two programs came in, and the kind's whole figure is current again.

**The measurement came first, at the round's base, against a frozen copy of the tree**
(`…/head-src`, so an edit mid-run cannot move the answer): **90 PASS / 183 FAIL / 75 UNSUPPORTED /
2 TIMEOUT** of 350 (`…/gtest`, `--jobs 3`). Two things came out of it: `baseline.json` has **one**
row wrong - `testRooRombergIntegrator` is recorded FAIL and measures UNSUPPORTED, which is `W0-154`
arriving at the file - and **24 of the 183 FAIL programs are exactly one failing case from a PASS**.
That list is now `W0-219`, with each program's own words, so the next rounds do not have to pay for
it again.

**`W0-141` closed, and it is the round's general fix.** ROOT's own cross-validation test times itself
with `using cv_milli_t = std::chrono::duration<double, std::milli>;` inside `runCrossValidation`,
then writes `cv_milli_t(EndTime - StartTime)`. C++ makes one of the type the alias stands for, and a
template instance's constructor is this kit's runtime call of that template - so the alias has to
reach a *value* position. `Program.local_aliases` holds a function's aliases, kept apart from the
parser's type table so that no inference about a name changes (`cv_clock_t::now()` is untouched), and
`emit_calls._alias_call` answers the call in `library_call`. **The first shape of it lived in
`named_call`, and that was wrong**: `_alias_call` emits its arguments, and inference asks
`named_call` the same question, so every `auto x = f()` in the program came back a `tuple`. The gate's
own run named it in one line, and the split is the fix.

**`TMVA::CrossValidation` is a `TNamed`** through its `Envelope` base, so `GetName()` is the
`jobName` its constructor was given - which is how the test builds the path of the weights its folds
wrote. One method, and the wall moved from `cv_milli_t` to it, and then the two programs passed.

**`W0-216` closed**: `TGraph::Sort` takes `greaterfunc` (`TGraph::CompareX`, `CompareY`, or any
callable of `(graph, left, right)`), `ascending`, and a `low`/`high` window; it sorts a stable
permutation with C++'s own predicate `greaterfunc(this, l, r) != ascending` and hands it to
`UpdateArrays`, which `TGraph` answers for x and y and `TGraphMultiErrors` overrides for its layers
and its x errors. `kIsSortedX` is set only by the default whole-graph ascending sort, as ROOT sets it.

**Measured.** `testCrossValidationIntVar` and `testCrossValidationMultiProc` PASS (`…/cv3`), and the
72 programs of `gtest/tmva/*` + `gtest/hist/*` + `gtest/graf2d/*` move **those two rows and no
other** (`…/areas`: 44 PASS / 21 FAIL / 7 UNSUPPORTED), the baseline raised by `rootsuite baseline
--merge`. The kind is **92 / 350**.

**Two traps paid for, both worth writing down.** A slice-based rewrite whose two indices can be in
the wrong order (`text[:start] + new + text[end:]` with `start` *after* `end`) silently **duplicates**
the chunk between them - 403 lines here - and only `git diff --stat` after every scripted edit shows
it. And a predicate that emits into the macro (`self.arguments(node)`) must never sit on a path
inference also takes.

**Gate.** `TOTAL 125738 0 31556 0 100%` (13 922 + 44, by the two-command decomposition,
`…/gate1.log` at `-n 2` and a `--cov-append` part 2); `ruff` clean; `mypy` clean on 1 061 files;
maintainability 33 251 functions in 1 820 files; components 1 061 modules, 0 undeclared imports;
`mkdocs build --strict` clean. The first part-1 run read 99.76% at a load average of 24 - the
`W0-109` artefact, and the second read was part 2's own report, which named nothing.
| 95 | `a88a6128` | **the one-case menu measured first, then three queued rows closed**: the gtest kind whole at the base (90 / 183 / 75 / 2, `…/gtest`) with its 24 one-case programs queued as `W0-219`; `W0-141` (a function's `using` of a template instance in a value position, via `Program.local_aliases` and `emit_calls._alias_call`); `TMVA::CrossValidation.GetName`; `W0-216` (`TGraph::Sort`'s comparator, descending order and window, with `UpdateArrays`) | `5a454e74` | **gtest 90 -> 92**: `testCrossValidationIntVar` and `testCrossValidationMultiProc` PASS (`…/cv3`), and 72 programs of `gtest/tmva` + `gtest/hist` + `gtest/graf2d` move two rows and no other (`…/areas`, 44 / 21 / 7) | 2: `W0-218` (a duration *argument* to such an alias still loses its type), `W0-219` (the 24 one-case programs, each its own feature); `W0-141` and `W0-216` closed |

## Round 96's close

Round 96 closed two queued rows - one of them **by measurement, not by code** - and measured the
gtest kind whole again at the tip. No program flipped, and the round says so; what it bought is a
measured **89x** on a hot path, a wrong row removed from the queue, and a current menu.

**`W0-217` closed.** ROOT's `TGraph::Eval` reads `kIsSortedX` and calls `TMath::BinarySearch` for the
two points around `x`, where an unordered graph is walked point by point. This kit's engine found the
order itself - `np.all(np.diff(xs) > 0)` on *every* call - which for a graph a fit evaluates again and
again is the whole cost: on 200 000 descending points one `Eval` costs **18 467 us before `Sort` and
208 us after it**, the same answer either way (`24691.0`). `graphmath.evaluate` now takes the bit as
`sorted_x`, `TGraph.Eval` passes it, and a graph nobody sorted keeps the engine's own look, so nothing
changes for it.

**`W0-218` is closed as a correction, and that is the round's most useful finding.** The row said a
duration *argument* to a function's `using` alias loses its type (`'tuple' object has no attribute
'dims'`). It does not: both `cv_milli_t(ms)` and `cv_milli_t(end - start)` translate and run at this
round's base. The probe that named it was taken **while round 95's own first shape of `_alias_call`
sat inside `named_call`**, where it emitted arguments during inference - the bug was the round's
intermediate state, not ROOT's or the kit's behaviour. A probe written against a half-changed tree can
name a bug that does not exist, and a queue row is only as good as the tree it was measured on.

**The kind, whole, at the tip**: **92 PASS / 181 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350
(`…/gtest`, `--jobs 3`, `--xrdroot-src` a frozen copy at `…/head-src`) - round 95's two gains still
there, **none lost**, and no other row moved. `baseline.json` still has the one row `W0-154` names
wrong (`testRooRombergIntegrator` measures UNSUPPORTED where the baseline says FAIL). And **`W0-219` is
refreshed to 22** one-case programs - two fewer than round 95's list, because its own two gains were on
it - each with its own words, so the next round starts from the current menu rather than a stale one.

**Why no program came in, priced honestly.** The round measured three of the menu's entries before
choosing: `test_boolean_extrusion` wants `LockDefaultUnits` *and* a GDML geometry whose boolean
composites have **zero** overlaps (a real overlap algorithm), `test_material_units` wants `TGeoUnit`
plus material radiation/interaction lengths inside PDG tolerances for two unit systems, and
`testTTreeCluster` asserts ROOT's own basket re-optimisation (`W0-182`). None is a name; each is a
feature, and the round did not pretend otherwise.

**The round shipped a red gate for one commit, and says so.** `e1feb58f`'s nav entry named a slug
`docs_record` did not choose (`…-names-a-bug-that-is-not-there.md` for the `…-names-a-bug.md` it
wrote), so `mkdocs build --strict` **aborted with one warning** - and the seven-command gate is
seven commands, not six and a note for later. `2c447529` has the real name and a strict build that
is clean at the committed tree. `W0-210`'s lesson has a second half: **read each gate's exit
status, not only the ones you expect to pass**. **Gate.** `TOTAL 125738 0 31556 0 100%` (13 924 + 44, by the two-command decomposition,
`…/gate1.log` at `-n 2` and a `--cov-append` part 2); `ruff` clean; `mypy` clean on 1 061 files;
maintainability 33 253 functions in 1 820 files; components 1 061 modules, 0 undeclared imports;
`mkdocs build --strict` clean.
| 96 | `1a3c7e1d` | **two queued rows closed, one by measurement**: `W0-217` - `TGraph::Eval` reads `kIsSortedX` and one `np.searchsorted` finds the points around `x` (18 467 us -> 208 us on 200 000 points, same answer); `W0-218` - **does not reproduce**, the probe that named it was taken while round 95's own `named_call` wrapper corrupted inference | `b93a1f7f` | **0 gtest PASS, 0 lost, and the kind measured whole again at the tip: 92 PASS / 181 FAIL / 75 UNSUPPORTED / 2 TIMEOUT** of 350 (`…/gtest`), round 95's two gains still there; three menu entries priced and refused as features (`boolean_extrusion`'s overlap check, `material_units`' `TGeoUnit`, `testTTreeCluster`'s basket re-optimisation) | 1: `W0-219` **refreshed to 22** one-case programs, with the list; `W0-217` and `W0-218` closed |

## Round 97's close

Round 97 took the one-case program with the most bounded feature behind it and closed it:
`gtest/roofit/roofitmore/testRooHypatia2` PASSES, the kind **92 -> 93**.

**What the shape is.** ROOT's Hypatia2 - the line shape a heavy-tailed peak is fitted with - is a
`(1 + d^2/sigma^2)^(lambda - 1/2)` core between two cut points, `a*sigma` on the left and `a2*sigma`
on the right, and beyond each a power law `A (B -+ d)^-n` whose `A` and `B` are chosen so the tail
meets the core's value and its slope there. That is `roofit/roofitmore/src/RooHypatia2.cxx:242`
transcribed for the `zeta = 0`, `lambda < 0` branch - the branch ROOT's own test writes - into
`roofit/pdfs/hypatia.py`, and it went in with nothing else: the test's `RooNumIntConfig`,
`defaultIntegratorConfig`, `setIntegratorConfig` and `forceNumInt` were all already here from the
earlier RooFit work, which is why a feature that looked a round deep took one file.

**Measured against the oracle before anything was claimed.** A macro asking ROOT itself
(`…/probe/hyp.C`) gives the density at seven abscissae across the core and both tails, and this kit
answers **the same 17 digits at every one**: `x = -10 -> 0.00098518533684157348`, `-2 ->
0.089442719099991588`, `-0.5 -> 0.71554175279993271`, `0 -> 1`, `0.5 -> 0.71554175279993271`, `2 ->
0.26794336563407328`, `10 -> 0.21632221225408768`. The integral over `[-10, 10]` is 3.8552011036701885
against ROOT's 3.8552008214678435 - **7.3e-8**, the integrator's own tolerance.

**Two refusals, named, and one row for what is not here.** `zeta > 0` needs the ratio of two modified
Bessel functions ROOT computes with `TMath::BesselK` plus its `LogEval`/`diff_eval` pair, and
`lambda >= 0` is a shape ROOT itself answers 0 for while logging; both are refused in a full sentence
with what to build instead. And `createIntegral` is **numeric**: ROOT's
`RooHypatia2::analyticalIntegral` is not written here, so `forceNumInt` changes nothing for this
class and the test's two roads are the same numeric integral - the values agree, the *closed form* is
the gap. That is `W0-220`, and the class's own docstring says it too.

**The area, measured whole at the tip**: 47 `gtest/roofit/*` programs are **11 PASS / 20 FAIL / 16
UNSUPPORTED** (`…/roofit`), from round 95's 10 / 21 / 16 - this round's row and the known `W0-154`
row (`testRooRombergIntegrator` measures UNSUPPORTED where the baseline says FAIL) - so nothing was
lost. The baseline was raised by `rootsuite baseline --merge` from that same run.

**Gate.** `TOTAL 125799 0 31560 0 100%` (13 928 + 44, by the two-command decomposition,
`…/gate1.log` at `-n 2` and a `--cov-append` part 2); `ruff` clean; `mypy` clean on 1 062 files;
maintainability 33 263 functions in 1 822 files; components 1 062 modules, 0 undeclared imports;
`mkdocs build --strict` clean.
| 97 | `d9859f31` | **one one-case program, its feature the smallest in the menu**: ROOT's `RooHypatia2` line shape (`roofit/pdfs/hypatia.py`, `RooHypatia2.cxx:242` transcribed for the `zeta = 0`, `lambda < 0` branch) and its registry entry - the `RooNumIntConfig`/`forceNumInt` surface the test needs was already here | `f0f6813d` | **gtest 92 -> 93**: `gtest/roofit/roofitmore/testRooHypatia2` FAIL -> PASS, and the 47 `gtest/roofit/*` programs measured whole at the tip move two rows - that one and the known `W0-154` - so nothing lost (`…/roofit`: 11 / 20 / 16). The density is **the oracle's to 17 digits** at seven abscissae and its integral differs by 7.3e-8 | 1: `W0-220` (`RooHypatia2::analyticalIntegral` is not written, so both roads are numeric and `forceNumInt` changes nothing); `W0-219` down to 21 |

## Round 98's close

Round 98 went to the two kinds no round had measured in forty, found where their failures live,
and took the two cheapest: **the pyunittest kind goes 81 -> 83**, with both kinds measured whole
before and after.

**What the measurement said first.** `pyunittest` at the round's base is **81 PASS / 45 FAIL /
2 UNSUPPORTED / 1 SKIP** of 129 and `test/` is **10 PASS / 6 FAIL / 43 UNSUPPORTED** of 59
(`…/python-test`). Two things came out of it that the ledger did not have: `test/` measures
**ten**, not the nine the ledger said (the two `stressIOPlugins` programs), and the 45
pyunittest failures have **no wide cluster** - 15 are the `cppyy/*` files that need compiled
dictionaries (a scope question, queued), and the rest are one file each. The biggest group is
the four TTree pythonization files (`W0-222`).

**What landed, each read off ROOT's own code.**

- **`_Vector.__array_interface__`**: a `std::vector` of numbers answers NumPy's array interface,
  transcribed from `bindings/pyroot/pythonizations/python/ROOT/_pythonization/_rvec.py`'s
  `get_array_interface` - the shape, the endian-tagged `typestr`, `version 3`, and, for an empty
  vector, **the address 1**, which ROOT writes for a reason its own comment gives: *"Numpy
  breaks for data pointer of 0 even though the array is empty"*. A vector of anything but a
  number answers none, exactly as ROOT's map of fundamental types decides.
  `pyroot_pyz_array_interface` 2 of 4 cases -> 4 of 4.
- **`TGraphMultiErrors`'s `(name, title, ...)` constructors** (`TGraphMultiErrors.h:63`, every
  overload declared again with them in front, and `(name, title)` alone a graph of no points),
  and **`GetEYlow`/`GetEYhigh` both ways ROOT declares them** (`:143`): summed over the layers
  by the graph's sum mode - which is `CalcYErrorsSum` - or one layer's own bars, a layer the
  graph has not being a null pointer. `pyroot_pyz_tgraph_getters` 1 of 5 -> 5 of 5.

**The gate caught the round's own breakage, and that is the useful hour of it.** Once a vector
answers the interface, `numpy.asarray` never reaches `__array__`, so `_Vector.__array__` was
dead - and the RDF **mask kernel's guard**, which refuses the batched fast path when a program
has replaced an RVec's conversion, was asking about exactly that dead method. Two symptoms in
one gate run: `test_a_replaced_rvec_array_conversion_keeps_its_calls` failing, and
`containers.py:642-643` uncovered. `2fbca5d3` deletes the dead method, asks the guard about the
interface property, and re-points the test at the property it now replaces - the assertion's
substance (the optimised path converts no fewer times than the entrywise one) unchanged.

**Measured at the tip, both weak kinds whole** (`…/final`, `--jobs 4`, `--xrdroot-src` a frozen
copy): **pyunittest 83 / 43 / 2 / 1** and **`test/` 10 / 6 / 43**, against the base's 81 / 45 /
2 / 1 and 10 / 6 / 43 - **exactly the two rows gained and none lost**, and the baseline records
them with `ratchet` reading 0 regressions.

**The other four things the round priced and did not do are queue rows with their numbers**:
`W0-221` is the PDF pair, taken apart byte by byte (**8 009 of ROOT's 13 765 bytes are the 25
fill-pattern streams**, each page's content 125, the assertion ±1 %); `W0-222` is the 45
pyunittest failures bucketed, with the TTree four named as the next group and a
`std::vector<float>` branch read back with **0 of the 4 elements** it was filled with;
`W0-223` is the six `test/` `*-interpreted` stress programs, each with the name it stops on.

**Gate.** `TOTAL 125823 0 31568 0 100%` (13 933 + 44, by the two-command decomposition,
`…/gate2.log` at `-n 2` and a `--cov-append` part 2); `ruff` clean; `mypy` clean on 1 062 files;
maintainability 33 272 functions in 1 822 files; components 1 062 modules, 0 undeclared imports;
`mkdocs build --strict` clean.
| 98 | `13f2a9d0` | **the two weak kinds measured whole, then the two cheapest fixes in them**: a `std::vector` of numbers answers NumPy's `__array_interface__` as ROOT's `_rvec.py` builds it (an empty one reporting the address **1**), and `TGraphMultiErrors` takes ROOT's `(name, title, ...)` constructors and answers `GetEYlow`/`GetEYhigh` both summed and per layer | `d97fd305`, `2fbca5d3` | **pyunittest 81 -> 83**: `pyroot_pyz_array_interface` (2 of 4 cases -> 4 of 4) and `pyroot_pyz_tgraph_getters` (1 of 5 -> 5 of 5) PASS, and both kinds measured whole at the tip move **exactly those two rows and no other** (`…/final`: 83 / 43 / 2 / 1 and 10 / 6 / 43 against the base's 81 / 45 / 2 / 1 and 10 / 6 / 43). The gate caught the round's own breakage - `__array__` was dead once the interface existed and the RDF mask kernel's guard was asking about it - which `2fbca5d3` fixes. `test/` measures **10**, not the ledger's 9 | 3: `W0-221` (the PDF pair priced: 8 009 of 13 765 bytes are 25 fill-pattern streams), `W0-222` (the 45 pyunittest failures bucketed), `W0-223` (the six `test/` interpreted stress programs); `W0-179` and `W0-124` restated |

## Round 99's close

Round 99 took the pyunittest kind's largest remaining group - the four `pyz_ttree_*` files, ten
failing cases between them - and found that the attribute lookup ROOT pythonizes was missing
three of its four rules. **0 programs gained, 0 lost, and the group went from 10 failing cases to
7**, each of the seven measured as a *writer* gap and queued.

**What `GetBranchAttr` does, read off ROOT's own source.** `TTree.__getattr__` is
`bindings/pyroot/pythonizations/python/ROOT/_pythonization/_ttree.py`'s `_TTree__getattr__`, which
asks the C++ helper `GetBranchAttr` (`src/TTreePyz.cxx:187`) for the name, and that helper reads
it in a fixed order: as an **alias** (`TTree::GetAlias`), then as a **branch** - `GetBranch(name)`,
or `GetBranch(name + '.')`, *"for benefit of naming of sub-branches, the actual name may have a
trailing '.'"* - and only then as a **leaf** (`GetLeaf(name)`, then `branch->GetLeaf(name)`, then
the branch's single leaf *"if unambiguously only this one"*). A branch whose address is an object
answers `*(void **)branch->GetAddress()` cast to the branch's class, and a leaf titled with more
than one bracket is shaped by the dimensions `getMultiDims` parses out of its title.

**What was here before** was one of those rules: a leaf, found by column or by name. So
`entry.structb` was a dict of members rather than a `MyStruct`, `ds.myalias` did not exist, and a
2-D array leaf could not be shaped. All three are in now - `_asked_branch`, `_asked_leaf` and
`_branch_value`/`row_object` in `pyroot/trees/reading.py`, `array_dims` beside `python_value` -
and they fix `test08_read_struct_single_branch`, `test_alias_branch` and
`test_struct_branch` (`…/ralph-round-202611241030/a1`).

**The remaining seven are all on the write side, and each was measured against the oracle before
it was queued.**

- `W0-224`: a leaflist's leaf is named after the branch here and its title is flattened. ROOT
  writes `branch [myint1b] title=[myint1balias/I]` holding `leaf [myint1balias]`, and
  `branch [array2Db] title=[array2Db[10][2]/F]` with `leaf [array2Db] title=[array2Db[10][2]]
  len=20` (`…/probe/l2.C`, the 6.40 oracle). Ours writes `array2Db[20]/F`, which is also why the
  reader has no dimensions to shape.
- `W0-225`: a leaflist branch of several leaves is written as one branch per leaf, so
  `SetBranchAddress("structleaflistb", ms)` answers `unknown branch` - the last case of two
  *programs*, the biggest single thing left in this group.
- `W0-226`: `Branch(TCollection)` and `ROOT.<namespace>.<class>` are names this kit has not.

**The gate caught two things in the round's own code, and both were fixed rather than excused**:
`__getattr__` hit **NPath 384** against the limit of 200, so it was split (`_asked_branch`,
`_asked_leaf`) rather than limited; and five branches were unreachable by the suite - a leaf whose
title does not count its values, an object branch with more than one leaf, a class this session
never declared, a class whose constructor needs arguments, a private member in a row - each got a
test that reaches it. `TOTAL 125869 0 31592 0 100%` (13 944 + 44, `…/gate3.log`, `-n 2`), ruff,
mypy 1 062 files, maintainability 33 291 functions, components 1 062 modules / 0 undeclared,
`mkdocs --strict`.

**Measured at the tip, both weak kinds whole**: pyunittest **83 / 43 / 2 / 1** and `test/`
**10 / 6 / 43** (`…/after`) - the same figures as round 98's, so this round cost nothing; the
`gained 96` the diff printed is the round's *base* run being a `--only` of four files, not a gain.
`W0-158` again: a subset run is a subset run.
| 99 | `40339ab8` | **the pyunittest kind's largest group, priced from ROOT's own source**: `TTree.__getattr__` as `GetBranchAttr` - an alias first, then a branch (`n` or `n + '.'`), then a leaf (its own, its branch's, or the branch's only one) - a branch holding an object answered as that object (`row_object`), and a leaf's title's dimensions shaping it (`array_dims` = `getMultiDims`) | `f2d0065e` | **0 programs gained, 0 lost, the wall moved 10 failing cases -> 7** in the four `pyz_ttree_*` files (`…/a1`), with `test08_read_struct_single_branch`, `test_alias_branch` and `test_struct_branch` passing; both weak kinds measured whole at the tip - pyunittest 83 / 43 / 2 / 1, `test/` 10 / 6 / 43 (`…/after`) - nothing lost. The seven left are writer gaps, each measured on the oracle | 3: `W0-224` (a leaflist's leaf name and bracket text), `W0-225` (a multi-leaf leaflist branch written as one branch per leaf - two programs), `W0-226` (`Branch(TCollection)`, `ROOT.<ns>.<class>`); `W0-222` partly |

## Round 100's close

Round 100 closed two of the three rows round 99 had left in the TTree pythonization group, and
one of them was a whole program: **`pyroot_pyz_ttree_branch_attr` PASSES, the pyunittest kind
83 -> 84**, with the four `pyz_ttree_*` files going from 7 failing cases to 4.

**What the two fixes are, and how each was measured first.**

- **A namespace declared in an `#include`d header was not an object.** `ROOT.gInterpreter.Declare`
  runs a text of declarations through `pycint.execute.run_source`, which decided whether to make
  namespaces out of the result by looking for the word in the text it was handed - and
  `'#include "TreeHelper.h"'` has none, so the namespaces the include's *translation* declares
  were never registered (`MyStruct` and `MyTree` were, because classes come through another
  path). `declares_namespaces` now looks through an `#include` as well, which is where most
  namespaces come from: ROOT's own pythonization test asks for `ROOT.Foo.MyTree()` with
  `namespace Foo` in `TreeHelper.h`.
- **A leaf list names the leaf, not the branch, and keeps its brackets.** The 6.40 oracle's own
  answer (`…/probe/l2.C`) is `branch [myint1b] title=[myint1balias/I]` holding
  `leaf [myint1balias] title=[myint1balias]`, and `branch [array2Db] title=[array2Db[10][2]/F]`
  with `leaf [array2Db] title=[array2Db[10][2]] len=20 lenStatic=20`; this kit wrote the branch's
  name as the leaf's and rebuilt the title from the column (`array2Db[20]`), so a static array's
  dimensions were gone from the file and nothing could shape the value read back.
  `leaflist.parse` already knew the token and its brackets, so `Slot.own` carries them and
  `Store.leaf_parts()` hands them to the writer beside `basket_sizes` - `WritableDirectory.tree`,
  `WTree._declare`, `_declared`, `_Column.part`, where `leaf_name`/`leaf_title` answer it. The
  round's files now read back as the oracle's four lines, to the character.

**Measured.** `pyz_ttree_branch_attr` PASSES (its `test_array_2D_branch` and
`test_ttree_subclass_in_namespace` were the two failing cases) and `pyz_ttree_branch` is down to
its two `Branch(TCollection)` cases, `test10_read_struct_separate_branches` now passing. Both
weak kinds measured whole at the tip: pyunittest **84 / 42 / 2 / 1** and `test/` **10 / 6 / 43**
(`…/after`), against round 98's 83 / 43 / 2 / 1 - the one gained and nothing lost, and the
baseline records it with `ratchet` reading 0 regressions.

**One assertion corrected against the oracle, not loosened.** Round 99's own test asserted that a
2-D leaflist branch read back flat (`(20,)`) - which was the *symptom* - and it now asserts
`(10, 2)`, which is what the leaf's title means and what ROOT gives.

**What is left in the group, priced rather than guessed.** `W0-225` is two programs - the last
case of `pyz_ttree_iterable` and of `pyz_ttree_setbranchaddress` - and the round took its
specification off the oracle: `t.Branch("pairb", &s, "first/I:second/I")` is **one `TBranch`**
(not a `TBranchElement`) with two leaves at `fOffset` 0 and 4 and **one basket** holding a row per
entry, which is why `SetBranchAddress` with the struct's address fills it. The writer's unit is
one `_Column` per branch, so it needs a column whose entry is its leaves' bytes in offset order,
a reader that splits that entry back, and a `SetBranchAddress` that fills an object leaf by leaf.
`W0-226` is one case left, `Branch(TCollection)`.

**And a stray-file row the round found**: `main` carries six probe `.root` files at its root
(`kde79.root`, `probe79*.root`, `probe80.root`, `test_tkde.root`), committed by round 79's
`f7c113af`. Nothing reads them; `W0-227` says so, and this round left them alone rather than ship
a deletion with an unrelated fix.

**Gate.** `TOTAL 125882 0 31594 0 100%` (13 946 + 44, by the two-command decomposition,
`…/gate2.log` at `-n 2` and a `--cov-append` part 2); `ruff` clean; `mypy` clean on 1 062 files;
maintainability 33 296 functions in 1 823 files; components 1 062 modules, 0 undeclared imports;
`mkdocs build --strict` clean. The gate's first run named the round's own breakage in 368 errors -
`ClonedSlot` has no `own` either, so `leaf_parts` asks with `getattr` and says why in its
docstring - which is `W0-63` doing its job on new code.
| 100 | `4c2eab48` | **two of round 99's three TTree-pythonization rows, one of them a whole program**: a namespace declared in an `#include`d header is an object (`declares_namespaces` looks through `#include`), and a leaf list's token and brackets are the leaf's own (`Slot.own` -> `Store.leaf_parts()` -> `writer.tree(leaf_parts=)` -> `_Column.part`) | `ec52b4c8` | **pyunittest 83 -> 84**: `pyroot_pyz_ttree_branch_attr` PASSES where 2 of its 10 cases failed, and the four `pyz_ttree_*` files go 7 -> 4 failing cases (`test10_read_struct_separate_branches` too); the written leaves are the oracle's four lines to the character (`branch [myint1b] title=[myint1balias/I]`, `leaf [array2Db] title=[array2Db[10][2]]`). Both kinds measured whole at the tip - 84 / 42 / 2 / 1 and 10 / 6 / 43 (`…/after`) - nothing lost | 3: `W0-224` **closed**, `W0-226` **partly** (the `Branch(TCollection)` case is left), `W0-225` **priced** (one `TBranch`, two leaves at `fOffset` 0/4, one basket - two programs), and `W0-227` (six probe `.root` files `main` carries from round 79) |

## Round 101's close

Round 101 closed the row round 100 had priced - **`W0-225`, the two programs behind a leaf list of
several leaves** - and the kind went **84 -> 86**: `pyroot_pyz_ttree_iterable` and
`pyroot_pyz_ttree_setbranchaddress` both PASS.

**Why it was two programs for one small change.** ROOT writes `Branch("structleaflistb", &ms,
"myintll1/I:myintll2/I")` as **one** `TBranch` - the oracle's own answer is `branch
[structleaflistb] title=[myintll1/I:myintll2/I]`, two leaves at `fOffset` 0 and 4, **one basket**
holding a row per entry - and this kit wrote a branch per leaf. Both failing cases ended at
`SetBranchAddress("structleaflistb", ms)`, which answered `Error in <TTree::SetBranchAddress>:
unknown branch` in a file this kit had written itself.

**What was already there, and what was not.** The *reader* takes a leaf list of one branch from a
ROOT-written file already - round 100's `…/probe/multi.root` reads as `pairb` with leaves `first`
and `second`, and `SetBranchAddress("pairb", obj)` fills the object - and `_Record` already lays a
column out as *"a leaf list, `a/F:b/F:c/F`: one value per leaf each entry"* with the whole leaf list
as the branch's title. So the gap was the store's presentation of a leaf list: it held one slot per
leaf and handed the writer one column per slot. `Store.joined()` groups them - a leaf list's slots
are the only ones whose `branch` is not their own name - and gives the writer one column of rows;
`layout.py`'s record lookup now asks what was written for the **branch** rather than for a leaf,
because one branch is one set of baskets.

**Measured.** The written file answers the oracle's layout to the character, `SetBranchAddress` +
`GetEntry` fills both members for every entry, and both programs pass (`…/a1`: 3 of 4 PASS in the
group, the fourth being `pyz_ttree_branch`). Both weak kinds measured whole at the tip: pyunittest
**86 / 40 / 2 / 1** and `test/` **10 / 6 / 43** (`…/after`) against round 100's 84 / 42 / 2 / 1 -
two gained, nothing lost, and the baseline records them with `ratchet` reading 0 regressions.

**One characterisation test corrected, and it was the round's own specification.** 
`tests/test_pyroot_trees_write.py`'s leaf-list test already asserted ROOT's shape - its last line is
`branch.GetTitle() == "a/I:b/D" and len(branch.GetListOfLeaves()) == 2` - but read its values by
top-level leaf name, which was only true while a leaf list was a branch per leaf. A leaf list of one
type is now one branch, so its leaves are read as the branch's own columns (`xyz.u`); the values,
the title and the leaf count are untouched.

**What is left in the group is one case and one fidelity gap.** `W0-226` is `Branch(TCollection)`,
the last case of `pyz_ttree_branch`. And **`W0-228`** is new, found while pricing this one: a leaf
list of *mixed* types (`mi/I:mf/F`) is still written a leaf to a branch, because `_Record`'s row
needs one element type and `joined()` therefore leaves it out - and the *in-memory* layout groups
those slots by branch, so `t.GetBranch("mixb")` answers before the write and is `None` in the file.
Nothing in the suite asks for it; the row names the first step.

**Gate.** `TOTAL 125901 0 31602 0 100%` (13 949 + 44, by the two-command decomposition,
`…/gate2.log` at `-n 2` and a `--cov-append` part 2); `ruff` clean; `mypy` clean on 1 062 files;
maintainability 33 301 functions in 1 823 files; components 1 062 modules, 0 undeclared imports;
`mkdocs build --strict` clean. The round's own first gate found the characterisation test above,
which is `W0-63` doing its job on a change that touches every written tree.
| 101 | `b14b9fb3` | **the row round 100 priced, closed**: a leaf list of several leaves as **one** `TBranch` - `Store.joined()` groups a leaf list's slots into one column of rows for `_Record`'s leaves at their offsets, and `layout.py`'s record lookup goes by the branch name | `09f7aaf5` | **pyunittest 84 -> 86**: `pyroot_pyz_ttree_iterable` and `pyroot_pyz_ttree_setbranchaddress` PASS where their last case was `SetBranchAddress` on a leaflist of a struct in a file this kit had written; the file's layout is the oracle's, `branch [structleaflistb] title=[myintll1/I:myintll2/I]` with both leaves, and both weak kinds measured whole at the tip - 86 / 40 / 2 / 1 and 10 / 6 / 43 (`…/after`) - nothing lost | 1: `W0-225` **closed**, `W0-228` (**a leaf list of mixed types** is still a leaf to a branch, and the in-memory layout disagrees with the file about it) |

## Round 102's close

Round 102 took `W0-226`, the one row the last three rounds had left standing in the TTree pythonization
group, and finished the group: **`pyunittest/bindings/pyroot/pythonizations/pyroot_pyz_ttree_branch`
PASSES all 14 of its cases**, so all four `pyz_ttree_*` files pass.

**What ROOT does, read off its own source first.** `TTree::Branch(TCollection*, Int_t bufsize, Int_t
splitlevel, const char *name)` (`tree/tree/src/TTree.cxx:1864`) is a recursion with three naming rules,
each transcribed: an object that is itself a collection (but not a `TClonesArray`) is walked as
`<name>_<collection>_` with the split level one lower; an object's branch is `<name>_<object>`, no second
underscore when the name already ends in one; and a split level above 99 ends the name with a dot, which
is how ROOT marks a branch it will not split further. A `TClonesArray` is refused in ROOT's own words.
That is why the test's `TList` holding a `TObjArray` called `myobjarray` of `one` and `two` gives
`myobjarray_one` and `myobjarray_two`, and why `entry.myobjarray_one.GetName()` is `one`.

**The wall behind it was a class, not the naming.** The objects are `TObjString`s, and
`wclasses.harvested()` accepts only `TStreamerBasicType`/`TStreamerObjectAny` members, so a class with a
`TObject` base has no split layout here at all. It is written **whole** instead - `Whole("TObjString",
split=0)`, one `TBranchElement` whose entry is `_record(...)[RECORD_HEAD:]`, the members the class's
description names without the byte count and version its streamer puts round them - which is exactly what
`splitlevel` 0 asks ROOT for. **ROOT reads the file we write**: on the 6.40 oracle,
`SetBranchAddress("myobjarray_one", &one)` and `GetEntry(0)` fill a `TObjString` that answers `one`. And
a row read back makes the object again through a new `BY_ROW` factory, because `entry.b` of a whole branch
is a dict and ROOT hands back the object.

**Measured.** pyunittest **86 -> 87**, and the whole kind measured either side moves that one row and no
other: **87 PASS / 39 FAIL / 2 UNSUPPORTED / 1 SKIP** (`…/ralph-round-202611241201/after`, `--jobs 3`),
with `test/` unchanged at **10 / 6 / 43**. The baseline row was raised by `rootsuite baseline --merge`
from that same run and `ratchet` prints **0 regressions**.

**The gate caught the round's own complexity, twice.** The new form's early return pushed
`TTree.Branch` to **NPath 256** against the limit of 200, so it was split - `_collection_form` for the
dispatch, `_branch_from` for the object-or-leaf-list decision, `_leaf_branch` for the leaves - rather
than excused. Gate: **`TOTAL 125967 0 31620 0 100%`** (13 961 + 44, by the two-command decomposition at
`-n 4`); `ruff` clean; `mypy` clean on 1 062 files; maintainability 33 328 functions in 1 824 files;
`mkdocs --strict` clean.

**What the round queued rather than did.** **`W0-229`** is the fidelity gap the work measured: a class
with a base is written whole where ROOT splits it, and the three pieces that would split it - flatten
`TStreamerBase`, allow `TStreamerString` in `SPLITTABLE`, give `Split._member` a `StringRows` column - are
a change to every class `harvested` describes, so they want their own round with the tree area measured
either side. `docs/gotchas/a-branch-of-a-class-with-a-base-is-written-whole-not-split.md` is the record.
`W0-222` is refreshed: the four TTree pythonization files all pass, and the kind's biggest group left is
`cppyy/*`, which needs compiled dictionaries.
| 102 | `a9d794d8` | **`W0-226` closed, and the TTree pythonization group finished**: `TTree::Branch(TCollection*)` as ROOT's own recursion (`_collection_form`/`_collection_branches`, the three naming rules transcribed from `TTree.cxx:1864`, a `TClonesArray` refused in ROOT's words) plus a `Named` kind that writes a `TObjString` branch **whole** (`Whole(split=0)`, one `TBranchElement`, which the 6.40 oracle reads back into a `TObjString`) and a `BY_ROW` factory so a row makes the object again | `c51f9582` | **pyunittest 86 -> 87**: `pyroot_pyz_ttree_branch` PASSES all 14 cases, so all four `pyz_ttree_*` files pass; the whole kind measures **87 / 39 / 2 / 1** with no other row moved (`…/ralph-round-202611241201/after`) and `test/` unchanged at 10 / 6 / 43. Gate `TOTAL 125967 0 31620 0 100%` (13 961 + 44), the round's own NPath 256 split rather than excused | 1: `W0-229` (**a class with a base has no split layout** - written whole where ROOT splits, with the three pieces and the doc that names them); `W0-226` **closed**, `W0-222` refreshed |

## Round 103's close

Round 103 wrote no code, and says so. It spent itself on the measurement the loop's own rule 1 asks for -
*rank the current failures by cause, from the last run's JSON rather than from memory* - because every
ranking the queue carried was 40 or more rounds old, and two of the last three rounds' targets had been
priced only to discover they were features.

**The measurement: the whole roottest kind, for the first time since round 17.**

**1318 PASS / 449 FAIL / 157 DIFF / 237 UNSUPPORTED / 10 SKIP / 2 TIMEOUT** of 2173
(`$HOME/.cache/xrdroot-runs/ralph-round-202611241234/roottest`, `--jobs 5`, `--xrdroot-src` a worktree of
`57440c96`). That is **60.6 %** where the ledger's own estimate said 1241 (57 %) - it was an estimate built
from in-area runs, and 77 PASS were missing from it. The run took 27 minutes. **The load average was 60-98
throughout** (another session), so the two TIMEOUTs are the CERN-network pair (`main/RootcpReadRemote*`,
`W0-43` - rerun alone at the end, see below) and **the timings are unquotable while the reasons are**;
`W0-103`/`W0-109` say what that means and nothing here is argued from a wall clock.

**The three biggest buckets, each one cause, each priced.**

- **`io/stdpair`: 40 DIFFs, one printer** - the largest bucket in the kind (`W0-230`). Measured line by
  line against `pairEnumEvo_0_0_0_1.ref`: a member's `offset=` is **0** where ROOT prints **4**; our line
  pads after `Emulation` and ROOT's does not; and **a whole `#2 StreamerInfo:` block is missing** (ROOT
  prints the older emulated info too, `checksum=0xb15d1241`, `first type= 3 newtype=13`). `pairEnumEvo.C`
  already runs in all 16 combinations, so the work is in the streamer-info printer and the emulated layout -
  not in I/O.
- **`W0-77`: 11 tests behind one `class reference at N points nowhere`** - seven `io/compression/read-*`
  and `io/bigevent/read`, all reading the `Event.root` `io/bigevent/write` produced, plus
  `io/datamodelevolution/00/test1` and its two dependants. **This is a file this kit wrote itself that it
  cannot read back**, and the round took the trace: eleven `any()` calls in a row return `None` (null
  pointers), then at 10787 a `CLASS_MASK`-tagged reference whose key was never mapped - and the words just
  before it are ASCII, so the reader has desynced ~100 bytes earlier, inside whichever member holds the
  `TNamed` containers. The writer only emits back-references in the streamer-info record (`backrefs`), so
  the tag is one the reader read out of alignment. The trace's `(Named) (null) (null) (Named) …` cadence is
  `vector<TNamed> *fVectorTnamed[6]`.
- **`W0-231`: `__PRETTY_FUNCTION__`, 5 tests** - and the row records why it is not a one-line name: the
  tests **search** the string (`strstr(PF,"TemplateArgument")`, `find("Parent")`), so it has to carry the
  *instantiated* template argument, which the translator has erased by the time it emits the body. The
  route is a runtime helper over the arguments' `ClassName()`.

**The gtest menu is not a menu of cheap entries.** `W0-219`'s 21 remaining one-case programs were re-priced
and **four of them are features**: `testRooHypatia2` already passes (round 97); `test_RooFitZMQ_mkstemp`
needs ZeroMQ itself, not `mkstemp`; `dataframe_splitcoll_arrayview` and `tree/evolution` both need a split
`std::vector<MyStruct>` read back - the feature round 71 wrote, measured and reverted (`W0-183`); and
`testTFoam` needs a whole Foam sampler reproducing ROOT's five draws to 1e-9. Recorded so the next round
does not pay for that pricing again.

**What else the map says, for whoever takes the next rounds.** `io/evolution` 139 PASS of 173 (34 not);
`tree/addresses` 24 of 52 (28 not, 11 of them UNSUPPORTED); `meta/tclass` 21 of 46 (25); `datamodelevolution`
22 of 44 (22, and `W0-233` is the one `Generator.h` `TypeError` behind 21 of them); `tree/friend` 8 of 27;
`meta/genreflex` 70 of 88; `io/filemerger` 31 of 48 (`Q-20`'s byte counts); `meta/evolution` 20 of 37
(`W0-183`'s `#pragma read`); `io/newstl` 19 of 32; `io/stdarray` 11 of 25. `roottest/cling` contributes a
long tail of translator gaps, three of which are named here (`W0-231`, `W0-232`, and `ConversionOp`'s
`'(' was expected`).

**The gate.** Nothing in the tree changed (`git diff` against `57440c96` is `TODO.md` and `.ralph/ROOT.md`),
so the seven commands were run once at the final tree and read: **`TOTAL 125967 0 31620 0 100%`** (13 961 +
44 passed, `…/gate1.log` + `…/gate2b.log`), ruff clean, mypy 1 062 files, maintainability 33 328 functions,
components 1 062 modules / 0 undeclared, `mkdocs build --strict` exit 0.

**A housekeeping finding.** `TODO.md` carries its §7 ledger and its appendices **twice** - a second
`## Appendix A` at the end and a second `W0-77` row - an earlier scripted rewrite duplicating the tail
(`W0-94`'s trap). Rounds append to the first copy; the row is in §5 so a reader does not edit the wrong one.
| 103 | `57440c96` | **no code: the roottest kind measured whole for the first time since round 17**, and its three biggest buckets priced - `io/stdpair`'s 40 DIFFs are one printer (`offset` 0 vs 4, trailing spaces, a missing `#2 StreamerInfo:` block), 11 tests behind one `class reference at N points nowhere` in a file this kit wrote itself (with the desync trace), and `__PRETTY_FUNCTION__`'s 5 tests (which need a runtime helper, not a literal); the gtest one-case menu re-priced with four of its entries shown to be features | no code | **0 PASS moved, 0 lost, and the kind's whole figure is the round's product: 1318 PASS / 449 FAIL / 157 DIFF / 237 UNSUPPORTED / 10 SKIP / 2 TIMEOUT** of 2173 (`…/ralph-round-202611241234/roottest`, `--jobs 5`, load average 60-98) - 77 more than the ledger's 1241 estimate, which was an estimate. Gate `TOTAL 125967 0 31620 0 100%` unchanged | 4 new: `W0-230` (40 tests, one printer), `W0-231` (5, `__PRETTY_FUNCTION__`), `W0-232` (3, multiple-inheritance MRO), `W0-233` (22, one `Generator.h` `TypeError`); `W0-77` **refreshed to 11 tests** with the trace, `W0-219` re-priced |

**Round 103's addendum, recorded after the close's measurement.** The two TIMEOUTs were rerun **alone** at
`--jobs 1` once the load average had fallen to 3.7 (`…/ralph-round-202611241234/net`, 180 s of wall clock):
`roottest/main/RootcpReadRemote` is still `still running after 300 s` and
`RootcpReadRemoteCheckOutput` is a FAIL for the same reason (`A fixture it requires was not set up`). They are
the CERN-network copies `W0-43` has recorded since round 9 - the endpoint, not the code - and the kind's
environmental floor is therefore 1318 PASS with those two never counted as regressions.

## Round 104's close

Round 104 wrote no code. It took the two rows round 103's whole-kind map had put at the top and
**measured which side of each is wrong** - which is what the last two rounds had only predicted.

**`W0-77` is a WRITER bug, and the oracle says so in its own words.** The eleven tests behind
`a class reference at N points nowhere` all read the `Event.root` that `io/bigevent/write` produced,
so the question was whether our writer or our reader is wrong. The answer needed a dictionary ROOT
could load, and the 6.40 oracle can build one from the roottest's own `Event.h`:
`rootcling -f bigdict.cxx -s libbig.so Event.h EventLinkDef.h`, then `clang++ -shared`. With that
loaded, `GetEntry(0)` on **our** file says

```
Error in <TBufferFile::CheckByteCount>: object of class vector<TLine> read too few bytes: 1928 instead of 2408
Warning in <TBufferFile::CheckByteCount>: list<string*>::Streamer() not in sync with data on file …
Error in <TBufferFile::ReadClassBuffer>: class: TAttAxis, attempting to access a wrong version: 510, …
 *** Break *** segmentation violation        (TBufferFile::ReadArray <- TAxis::Streamer)
```

**480 = 10 `TLine` objects × 48 bytes** - the same 48-per-event round 26 measured from the reader's
side - so a **member-wise container of a class with a base** (`TLine : TObject, TAttLine`) is laid
out 48 bytes an object short. Everything after it is the damage. The other direction is wrong too
and *separately*: our reader on **ROOT's own** `Event.root` (the oracle's `IoBigEventGenerator`
compiled against the same dictionary) fails with `FormatError: asked for 50273048 bytes with 30441
left`, so the same member kind is misread as well. Both probes are three commands and are kept at
`…/ralph-round-202611241318/probe/README.md` - the next round starts from a judged file rather than
a trace.

**`W0-230`'s 40 DIFFs, measured with the harness's own `custom_diff`** (round 29's rule: never a
hand-rolled diff). `io/stdpair` is **4 PASS / 40 DIFF** of 44 at this round's base, and across the
40 the whole difference is **556 `offset=` lines, 146 `StreamerInfo` block lines and 32
`Warning in <TClass::Init>: no dictionary for class pair<…>`/`edm::Ref<…>` lines** - with **no test
offsets-only**, so the bucket needs all three:

- **offsets**: ROOT prints each element's accumulated `fOffset` (the enumerated `first` is 4 bytes,
  so `second` is at 4); ours prints 0 in the element *and* the action lines.
- **the info list**: ROOT's is `fStreamerInfo`, a `TObjArray` **indexed by class version**
  (`core/meta/src/TClass.cxx:4657` - `GetStreamerInfo(version)` is `At(version)` with a fall-back to
  the current one), and the macro walks `GetStreamerInfo(i)` for `i` in `1..size-1`, so the `#2`
  block is the *other version's* info at index 2. The 16 combinations differ only in which versions
  exist when - file read before the library load or after, and which library - so one
  version-indexed model answers all 40.
- **the warnings**: ROOT warns about `pair<…>` and `edm::Ref<…>` in the read-before-load and
  interpreted/`kNothing` families; we warn in none.

**A correction to round 103's own row, and it is the round's second lesson.** Round 103 recorded
that our lines *padded* with trailing spaces, quoting a hand-rolled `difflib` diff. The harness's
judge compares with **all spaces removed** (`refdiff.filtered(..., ignore_spaces=True)`), so padding
never counted and that item was never part of the difference. `W0-75`'s rule ("use `custom_diff`,
never a hand-written one") had to be re-learned here, in the very round that quoted round 29's other
half - *a measurement tool that is not the judge's is not the judge*.

**Also measured**: `io/datamodelevolution/00/test1`'s first failure is `W0-77`'s (`FormatError: a
class reference at 120 points nowhere`), not the `Generator.h` `TypeError` `W0-233` names - so
`W0-77` blocks 11 tests and `W0-233`'s 22 are a different member of that family.

**The gate.** Nothing in the tree changed (`git diff` against `ca3c372f` is `TODO.md` and
`.ralph/ROOT.md`), so the seven commands were run once at the final tree and read:
**`TOTAL 125967 0 31620 0 100%`** (13 961 + 44 passed, `…/gate1.log` + `…/gate2b.log`); ruff clean,
mypy 1 062 files, maintainability 33 328 functions, components 1 062 modules / 0 undeclared,
`mkdocs build --strict` exit 0.
| 104 | `ca3c372f` | **no code: the two rows round 103's map put on top, measured to the side they are wrong** - `W0-77` is proved a **writer** bug by the oracle (a `rootcling`+`clang++` dictionary, then `GetEntry(0)` on our `Event.root`: `CheckByteCount: object of class vector<TLine> read too few bytes: 1928 instead of 2408` = 480 = 10 x 48, then a segfault in `TAxis::Streamer`; our reader also fails on ROOT's own file), and `W0-230`'s 40 DIFFs are re-measured with the harness's own `custom_diff` - 556 offsets + 146 block lines + 32 dictionary warnings, no test offsets-only, and round 103's "trailing padding" **corrected** because the judge strips spaces | no code | **0 PASS moved, 0 lost**: `io/stdpair` measured whole at the base is **4 PASS / 40 DIFF** of 44 (`…/ralph-round-202611241318/base`); the round's product is the two rerunnable probes (`…/probe/README.md`) and two corrected, fully priced rows. Gate `TOTAL 125967 0 31620 0 100%` unchanged | 0 new: `W0-77` and `W0-230` **re-stated with both directions measured**; `W0-233` noted as a different member of the same family |

## Round 105's close

Round 105 wrote no code, and the round's result is that it **disproved its own starting hypothesis** in twenty
minutes with a five-file probe - which is worth more than the hour it would have cost to build on it.

**The hypothesis round 104 left.** `W0-77` was recorded as *the writer lays a member-wise container of a class
with a base out 48 bytes an object short*, on the strength of ROOT saying `Error in <TBufferFile::CheckByteCount>:
object of class vector<TLine> read too few bytes: 1928 instead of 2408` on our own `Event.root` (= 480 = 10 × 48).

**The probe.** `struct H { std::vector<TLine> v; }` with ten lines, written by the 6.40 oracle
(`woracle.C`, ACLiC) and by this kit (`wours.C`), then read by each of them and by the other:
- our file **reads back cleanly through our own reader** - `v` has 10 lines and every member of the first is
  right (`TObject`, `TAttLine`, `fX1..fY2`);
- the oracle reads **its own** file cleanly (`v.size()=10 first=(0,0,1,1)`);
- the oracle on our file **segfaults** in `TBufferIO::MapObject` with no byte-count message at all.

So the member-wise container is not 48 bytes short, and the bytes of the two records differ by 58 in a way that
is not per-object.

**What the probe does show, and it is structural.** `TTree::Print` on the two files:
ROOT's `h` branch has a **sub-branch `v`** (`*Br 0 :v : vector<TLine>`, Total 1220) - it splits a declared
class's branch at the default splitlevel 99 - and ours is one branch `h` of Total 1162, written **whole**. Both
are `TBranchElement` with `fType=0 fStreamerType=-1`; the only other difference ROOT names is `fClassVersion`
**1 against our -1**. ROOT therefore reads our whole record with its member-wise streamer, and the
`vector<TLine>` complaint is that reading - a symptom, not the cause.

**And our own reader's failure is a second, separate one, late in the class.** With no instrumentation the
traceback is unambiguous: `interp.py:1000`'s `buf.objarray(held)` for `Event::fObjArray`, reached with a garbage
count (`FormatError: a class reference at 10787 points nowhere`) - so the offsets of the members *before*
`fObjArray` have already drifted. Those are `TString fNames[12]`, `TAxis fXaxis`/`fYaxis[3]`/`fVaxis[3]`/`fPaxis`/
`fQaxis`, `TDatime fDatime` and `EventHeader fEvtHdr`.

**One instrumented run named the wrong member, and the close records it as mine.** Wrapping `interp._steps` to
log each member made the run stop earlier, at `Event::fClosestDistance` with a `TypeError` - a half-changed-tree
artifact of exactly `W0-218`'s shape (round 96's "a probe written against a half-changed tree can name a bug that
does not exist"). Its trace was discarded and only the uninstrumented traceback is quoted.

**So `W0-77` is re-scoped to two rows**, each with its own side of the file:
- **`W0-234`** (8 tests): a declared class's branch is written whole where ROOT splits it - `object_slot` picks
  `MacroWhole()` because a class with containers among its members is not `_splittable`, while ROOT splits at
  splitlevel 99. `W0-183`'s neighbourhood, and the piece that makes an `Event` file ROOT can read.
- **`W0-235`** (3 tests): our reader's per-member offsets inside a 50-member declared class - first step a
  per-member offset log (a `Buffer` position before and after each member against the member's recorded byte
  size), which is the tool `W0-230`'s `offset=` work needs as well.

**The gate.** Nothing in the tree changed (`git diff` against `f488105c` is `TODO.md` and `.ralph/ROOT.md`), so
the seven commands were run once at the final tree and read: **`TOTAL 125967 0 31620 0 100%`** (13 961 + 44
passed); ruff clean, mypy 1 062 files, maintainability 33 328 functions, components 1 062 modules / 0
undeclared, `mkdocs build --strict` exit 0.
| 105 | `f488105c` | **no code: round 104's 48-byte hypothesis disproved in a five-file probe, and `W0-77` split into the two failures it really is** - `struct H { std::vector<TLine> v; }` written by the oracle and by this kit shows our own round trip clean and ROOT's own file clean, so the container is not 48 bytes an object short; what differs is that ROOT writes a declared class's branch **split** (a sub-branch `v`) and we write it whole (`fClassVersion` -1 against 1), which is why ROOT complains of the first container it meets; and our reader dies separately and later, at `buf.objarray` for `Event::fObjArray` with the offsets before it already drifted | no code | **0 PASS moved, 0 lost**: nothing in the tree changed, so the family was not re-measured; the round's product is the probe (`…/ralph-round-202611241332/probe/`) and two named rows. Gate `TOTAL 125967 0 31620 0 100%` unchanged | 2 new: `W0-234` (**a declared class's branch is whole where ROOT splits it**, 8 tests), `W0-235` (**the reader's per-member offsets in a 50-member class**, 3 tests); `W0-77` re-scoped to both |

## Round 106's close

Round 106 wrote no code and **withdrew its own previous hypothesis**. Round 105 recorded `W0-235` as "the reader's
per-member offsets drift inside a 50-member declared class". They do not.

**The per-member log, and why round 105's attempt failed.** `interp._plan` gathers a run of consecutive plain
numbers into **one** struct read ("the fewest reads that read them in the order they come"), so wrapping each step
*before* `_plan` - which is what round 105 did - makes `type(step) is _Plain` false, disables the gathering, and
changes the bytes read. The round-105 trace was an artefact of exactly that, and it was discarded then; this round
rebuilt `_members` from the same `_steps`/`_plan` with logging added *after* the plan is made
(`…/ralph-round-202611241344/trace3.py`), which preserves the read path exactly - the same failure at the same
offset, 10787, and now 39 named members with their byte counts.

**What the log says.** `Event` reads sensibly member by member: `TObject` 10, `fType` 20, `fEventName` 17, `fBoolA`
1, the six gathered plain numbers 120, `fClosestDistance` 37, `fTemperature` 4, `fTracksInVertex` 10, the four
scalar `vector`s 10/10/58/250, `fVectorD` 22, `fVectorTLine` 2412, `fVectorTobject` 36, `fVectorTnamed` 32,
`fLachaud` 112, `fDeque` 36, `fVectorTobjectp` 10, `fListString` 50, `fListStringp` 114, `fMapTNamedp` 385,
`fTemplateMember` 10, `fMapList` 16, `fMapTAxisp` 1150, `fSetTAxis` 10, `fSetTAxisp` 1108, `fMultiMapTNamedp` 385,
`fMultiSetTAxisp` 1108, `fString` 27, `fStringp` 19, `fTstringp` 96, `fNames` 12, `fXaxis` 108, `fYaxis` 324,
`fVaxis` 366, `fPaxis` 978, `fQaxis` 1104, `fDatime` 4, `fEvtHdr` 18 - ending at 10668, with **`fObjArray` the
first step that runs away** (`buf.objarray` reads a garbage count and walks to 10787).

**The six-line reproduction, and it is decisive.** `struct O { TObjArray a; }` filled with two `TNamed`s
(`…/probe/{o.h,owrite_oracle.C,owrite_ours.C}`):
- **the 6.40 oracle's file reads through this kit** - `{'a': [{'TObject': …, 'fName': 'one', 'fTitle': 'for
  file'}, {'TObject': …, 'fName': 'two', 'fTitle': 'for memory'}]}`;
- **ours does not** - `FormatError: the record ends 4 bytes before it said it would`.

**The cause, read off ROOT's own source.** ROOT's `TObjArray::Streamer` (`core/cont/src/TObjArray.cxx:448-497`,
write side) is `WriteVersion(TObjArray::IsA(), kTRUE)`, `TObject::Streamer`, `fName.Streamer`, `nobjects` (Int_t),
`fLowerBound` (Int_t), then each `b << fCont[i]`; a `TList` writes an option string after each entry. Ours routes an
in-memory collection to `wdobjects.record`'s **description** path (`catalogue.written_as("TObjArray")`), which lays
out `INFOS["TObjArray"]`'s three elements - the `TSeqCollection` base, `fLowerBound`, `fLast` - with **no name, no
count and no entries**. The reader is already right: `buffer.objarray`/`tlist` read exactly ROOT's shape, which is
why the oracle's file decodes and ours does not.

**So `W0-235` is rewritten and the fix is a new row.** `W0-235` now carries the log and the repro; **`W0-236`** is
the writer's gap - 8 tests (`io/bigevent/read` and the seven `io/compression/read-*`), with both layouts, the
existing precedent for the fix (`pyroot/graphics/snapshot.py:112` and `pyroot/core/stacks.py:101` already hand the
engine a `Written`/`Listed` of entries), and the two places that must change (`streamed`'s
`(held == "TList") != listed` guard has to learn the in-memory case, and `record` must route every
`STREAMED_COLLECTIONS` class to `streamed`).

**Measured at the base, unchanged**: `io/bigevent` is **3 PASS / 0 FAIL / 1 UNSUPPORTED** of 4
(`…/ralph-round-202611241344/base`, `--jobs 2`).

**The gate.** Nothing in the tree changed (`git diff` against `3355726f` is `TODO.md` and `.ralph/ROOT.md`), so the
seven commands were run once at the final tree and read: **`TOTAL 125967 0 31620 0 100%`** (13 961 + 44 passed);
ruff clean, mypy 1 062 files, maintainability 33 328 functions, components 1 062 modules / 0 undeclared,
`mkdocs build --strict` exit 0.
| 106 | `3355726f` | **no code: round 105's `W0-235` hypothesis withdrawn, and its cause nailed with a per-member log and a six-line reproduction** - `Event`'s reader consumes all 39 members sensibly and `fObjArray` is the first that runs away; `struct O { TObjArray a; }` with two `TNamed`s reads correctly when the 6.40 oracle writes it and fails (`the record ends 4 bytes before it said it would`) when this kit does, because an in-memory collection is written as its *description* (base plus two ints) where ROOT writes `TCollection::Streamer`'s name + count + entries | no code | **0 PASS moved, 0 lost**: `io/bigevent` measured at the base is 3 PASS / 0 FAIL / 1 UNSUPPORTED of 4 (`…/ralph-round-202611241344/base`); the round's product is the log (`…/trace3.py`), the repro (`…/probe/`) and two rows. Gate `TOTAL 125967 0 31620 0 100%` unchanged | 1 new: `W0-236` (**an in-memory `TObjArray`/`TList` member is written as a description where ROOT writes its streamer's shape**, 8 tests); `W0-235` **rewritten** with the evidence |
| 107 | `4aa28416` | **the bigevent `Event` file reads back: an in-memory `TObjArray` is written by hand, not by its description** - `records.MEMBERS` hands the writer a `Record` of a `TObjArray`'s entries and its name, `wdobjects._streams_itself` routes an in-memory array to `streamed`, which writes `TObjArray::Streamer`'s shape (version, `TObject`, `fName`, count, `fLowerBound`, each entry naming its class) instead of `INFOS["TObjArray"]`'s three elements that name no entry; and `_refilled`'s appender is `_appender`, because `Event`'s `UShortVector` derives from `std::vector<UShort_t>` and leaves a plain list where a C++ container's `insert` was asked for | `4aa28416` | **8 tests gained, none lost**: `io/bigevent/read` and the seven `io/compression/read-*` UNSUPPORTED->PASS, the `roottest/root/io/*` subtree measured either side at **770 tests, +8 PASS, 0 lost**; the three other kinds measured whole at the tip unchanged (pyunittest 87/39/2/1, `test/` 10/6/43, gtest 93/180/75/2); and the 6.40 oracle reads this kit's six-line `struct O { TObjArray a; }` file now. Gate `TOTAL 125976 0 31622 0 100%` (13 964 + 44) | 1 new: `W0-237` (a `TObjArray` as a file **key** is refused, not written); `W0-236` and `W0-235` **closed**, `W0-77` **re-scoped**, `W0-139` refreshed |

## Round 107's close

Round 107 is the first round since 97 to move a harness number, and it moved **eight tests** by fixing one file's
writer and one member's reader.

**What was wrong, and it was two things.** Round 106 had proved with a six-line reproduction that this kit writes an
in-memory `TObjArray` as its *description* - the `TSeqCollection` base, `fLowerBound`, `fLast` - where ROOT writes
`TObjArray::Streamer` by hand: `core/cont/src/TObjArray.cxx:448-497` is `WriteVersion(..., kTRUE)`, `TObject::Streamer`,
`fName.Streamer`, `nobjects`, `fLowerBound`, then each `b << fCont[i]`. The fix is the one the row named: the pyroot
layer hands the engine a `Record` of the entries and the array's name (`pyroot/core/records.py`'s
`MEMBERS["TObjArray"] = _held`, `NAMED_COLLECTIONS`), and `wdobjects` routes an in-memory array to the same `streamed`
writer a `TObjArray` read out of a file goes back through. The dispatch is `_streams_itself(classname, value)`, a
function of its own, because folding the `or ... and ...` into `record()` took its **NPath to 576** - the
maintainability gate caught that in the first gate run of the round, before anything was pushed.

**And reading the file then stopped one wall further on.** `Event`'s `UShortVector` derives from
`std::vector<UShort_t>`, so a read leaves a plain Python list where the elements go - and `_refilled` asked for
`push_back`, then `insert`, and a list's `insert` takes a place and an item: `TypeError: insert expected 2 arguments,
got 1`. `_appender` appends to a list and inserts into a container of C++'s own. With both fixes the generator reads
the ten events this kit wrote and `io/bigevent/read` PASSES.

**The oracle decides, and it agrees both ways.** The six-line probe (`.../probe/`: `o.h`, `owrite_oracle.C`,
`owrite_ours.C`, and an `oLinkDef.h` with a `rootcling`-built dictionary against the 6.40 oracle) now prints
`entries=2 [0] one / for file [1] two / for memory` **reading this kit's file**; before the fix the oracle segfaulted in
`TAxis::Streamer` on it. This kit reads ROOT's own array file too, both entries right.

**Measured, not assumed.** `io/bigevent` is **3 PASS / 0 FAIL / 1 UNSUPPORTED -> 4 / 0 / 0 / 0** and `io/compression`
**9 / 0 / 0 / 7 -> 16 / 0 / 0 / 0** (`.../base-run` and `.../base-bigevent` at `8872b7f9` against `.../a2` and `.../a3`
at the tip). The whole `roottest/root/io/*` subtree - 770 tests - was then measured either side and diffed id by id:
**+8 PASS, 0 lost**, and the eight are its only moves (`.../loss-io`, its base round 103's whole-kind run at the same
tree). The other three kinds were measured whole at the tip and are unchanged: pyunittest **87 / 39 / 2 / 1**, `test/`
**10 / 6 / 43**, gtest **93 / 180 / 75 / 2** - whose diff against round 96's whole-kind run is `testRooHypatia2` alone,
round 97's gain, and whose two TIMEOUTs are `ntuple_largefile1` and `testBulkApiMultiple`, the latter re-run **alone**
at `--jobs 1` under a load average of 14 and still `still running after 300 s` (`.../bulk`).

**What is left, and it is honest.** A `TObjArray` handed over as a file **key** (`Write("mine", kSingleKey)`) is now
*refused* - `a TObjArray here is a list rather than the dict of members writing one starts from` - because
`writer._record_payload` has no array shape; before this round the same call wrote bytes with no entry in them, so the
refusal is the better half of the same bug. That is `W0-237`, with the shape it needs named in the row. `W0-77` is
re-scoped: eight of its tests were this writer bug and pass now, and what remains is `W0-234`'s split branch and
`W0-233`'s `io/datamodelevolution/00/test1`.

**The gate.** Seven commands at `4aa28416`, each read for its own exit status: **`TOTAL 125976 0 31622 0 100%`**
(13 964 + 44 passed, `.../gate2.log`), ruff clean, mypy 1 062 files, maintainability 33 333 functions, components 1 062
modules / 0 undeclared, `mkdocs build --strict` exit 0 (`.../mkdocs.log`).

**Still open, and recorded rather than quietly fixed.** Two earlier rows quote a gate total that cannot both be
right: round 95's ledger row in `TODO.md` and round 95's close here both say `125738 0 31556` with **13 922 + 44**
passed, while round 96's rows say the same total with **13 924 + 44**. One total is wrong and only a re-run at round
95's own tree could say which; this file is append-only, so the correction is this sentence and not an edit.
| 108 | `7ed2e7e1` | **two translator gaps the `datamodelevolution` generator walked into**: a name written from `::` is the file's own (`Scope.global_lookup` - `::Print(x)` in `Holder::Print` called the method and recursed), and a dialect's `method_returns` entry is a declaration (`_named_type` canonicalises the typedef, so `Int_t` is `int` and `((char*)obj) + cl->GetBaseClassOffset(b)` is `byte_offset` instead of a string concatenation) | `7ed2e7e1` | **0 harness PASS, 0 lost**: `io/datamodelevolution` measured either side is 22 / 20 / 2 of 44 (`…/dml`, `…/a1`); the 153 programs of `gtest/{hist,graf2d,tree/ntuple,roofit}/*` that use the `method_returns` table are 53 / 65 / 34 / 1 TIMEOUT at the tip (`…/areas`) with none of round 94's 57 PASS rows moved. Gate `TOTAL 125983 0 31624 0 100%` (13 966 + 44) | 1 new: `W0-238` (overload selection by Python kind, with a silent first-candidate fallback); `W0-233` **rewritten** as the four-wall chain it is |

## Round 108's close

Round 108 took the `datamodelevolution` family - the one row (`W0-233`) that promised twenty-two tests behind a
single `TypeError` - and came back with **no PASS and two real fixes**, which is the honest shape of that row.

**What the row said, and what the first look said.** `W0-233` read "`Generator.h` is the same `TypeError` in 21 of
the family's 22 non-PASS tests: one operand's type in a generator expression is the whole area". Round 107's work
had moved the family's first failure, so the round measured it again first: the whole glob is **22 PASS / 20 FAIL /
2 DIFF** of 44 (`…/dml`, and `…/a1` after the round's fixes - the same rows), and the `TypeError` is at
`common/Generator.h:223`, `void* basePtr = ((char*)obj) + cl->GetBaseClassOffset(baseCl);` - emitted as
`obj + cstr(cl.GetBaseClassOffset(baseCl))`, an object plus a string.

**Why it was a string, and it is a general bug of the table.** `infer._named_type` built a `CType` straight from a
dialect `method_returns` entry, and `CType("Int_t").integral` is false - `INTEGRAL` holds C's canonical names only.
So `char* + Int_t` was not pointer arithmetic and fell through to the string-concatenation rule. **A table entry is
a declaration** - `CType`'s own docstring says a `CType` is "with its base name looked through typedefs" - so
`_named_type` now canonicalises it, and `("TClass", "GetBaseClassOffset"): "Int_t"` in the dialect's table makes the
sum `byte_offset(obj, (cl.GetBaseClassOffset(baseCl)))`, which is what ROOT's `(char*)obj + offset` means.

**The second fix came from the next wall, and it is wider than the family.** `stl/WriteFile` died with
`Holder.Print() takes 1 positional argument but 2 were given`, and the generated line was
`self.Print(self.fSingle)` for C++'s `::Print(fSingle)` - **a name written from `::` was resolved to the enclosing
class's member**, because `Scope.lookup` walks block, function and class before the file. A leading `::` is C++'s
global namespace: `Scope.global_lookup` answers from the module's own scope and skips every class and block. Any
macro that writes `::f(x)` inside a method was calling the wrong function; `Holder::Print` was calling itself.

**And the row is a chain, which is the round's measurement.** With both fixes in, the family stops on
`'TDataMember' object has no attribute 'GetDataType'` - and behind it the generator needs `GetOffset()`,
`TClass::New()`, **a write through `(char*)obj + offset`** (which `cppstd.cells.byte_offset` refuses by design:
"only an offset of 0 - a base at the object's start - is followed"), and `stl/WriteFile`'s wall is now
**free-function overload selection**: `Overloaded.__call__` matches on Python kinds and, when none match, silently
calls its *first* candidate - and `std::vector<T>` here is a `_Vector`, not a `list`, so
`Print(const std::vector<Values>&)` never matched and `Print(const Values&)` ran. `W0-233` now carries those four
walls in order, and `W0-238` carries the overload one.

**Nothing was lost, and the areas were measured.** The `method_returns` table is read by the graph getters, the
matrix decompositions, the RNTuple writer factories and `HijackMessageStream`, so the 153 programs of
`gtest/{hist,graf2d,tree/ntuple,roofit}/*` were run at the tip (`…/areas`: 53 PASS / 65 FAIL / 34 UNSUPPORTED /
1 TIMEOUT) and diffed against round 94's 57-row `hist-graf` run: **not one PASS moved**. The family itself is
unchanged at 22 / 20 / 2.

**The gate.** Seven commands at `7ed2e7e1`, each read for its own exit status: **`TOTAL 125983 0 31624 0 100%`**
(13 966 + 44, `…/gate1.log`), ruff clean, mypy 1 062 files, maintainability 33 337 functions, components 1 062
modules / 0 undeclared, `mkdocs build --strict` exit 0 (`…/mkdocs.log`).
| 109 | `971651c5` | **a versioned class's layouts indexed by class version, and `io/stdpair`'s 40 DIFFs priced from its own refs**: `GetStreamerInfos` is ROOT's `fStreamerInfo` now (`AddAt(entry, version)`, holes kept) and `GetStreamerInfo(v)` falls back to the class's current layout as `TClass.cxx` does; the bucket's two remaining differences follow the *configuration* (blocks `#1`..`#4` by lib/read-before-load/reported-order; `second`'s `offset=` 4 except `kNothing` and interpreted-read-first, which say 8) | `971651c5` | **0 PASS moved, 0 lost**: `io/stdpair` 4 / 40 / 0 / 0 of 44 at the tip, the same four as its base, because no ref is offsets-only; 237 tests of `io/{evolution,tree,stdpair}/*` are 161 / 13 / 47 / 16 and the two `AddVersion-2-*` DIFFs are DIFF at the base too (`…/basever`). Gate `TOTAL 125988 0 31626 0 100%` (13 967 + 44) | 1 new: `W0-239` (the dense version numbering of an emulated class's layouts); `W0-230` **rewritten** with both tables |

## Round 109's close

Round 109 landed the bookkeeping half of `io/stdpair`'s 40 DIFFs and measured the rest off the refs themselves.

**What landed.** `TClass.GetStreamerInfos` now builds what ROOT keeps: `fStreamerInfo` is a `TObjArray` **indexed by
class version**, so `At(n)` is version `n`'s layout and a version the class has no layout of is an empty slot - the
class whose files recorded 2 and 3 has `GetSize()` 4 and a hole at 1. `GetStreamerInfo(v)` answers the class's
current layout for a version it does not have, which is `TClass::GetStreamerInfo`'s own `At(version)`-with-a-fall-back
(`core/meta/src/TClass.cxx`), and `TObjArray`'s walk leaves its empty slots out as ROOT's iterator does. Two unit
tests pinned the *old* answer (`GetStreamerInfo` of an absent version is `None`); they were **corrected against ROOT's
source**, with the citation in the test, not loosened - and one new test pins the gapped array (versions 2 and 3,
`At(1)` empty, `GetSize()` 4, the fall-back).

**The measurement, and it is why no test flipped.** The 40 `.ref`s are the oracle's own output, so the bucket was
priced from them instead of from a diff: **no ref is offsets-only or blocks-only**, and both differences follow the
*configuration* - `pairEnumEvo(libtype, fixed, readbeforeload, reportedorder)`:

- blocks: one (`#2`) for `libtype` 0/1 with `readbeforeload=0`; `#2,#3` (`#2,#3,#4` when `reportedorder=0`) for 0/1
  with `readbeforeload=1`; `#2,#3` (`#2`) for 2-with-`0` and for 4; `#1,#2,#3` (`#1,#2`) for `libtype=3`;
- `second`'s `offset=`: `4` everywhere but `libtype=4` (`kNothing`) and `libtype=2` with `readbeforeload=1`, which
  say `8` - `TStreamerInfo::BuildOld`'s `AlignUp(offset, element->GetAlignment())` then
  `offset += element->GetSize()` (`io/io/src/TStreamerInfo.cxx:2723`), the alignment being 8 only where that
  configuration knows the `edm::Ref` class.

**Nothing lost, and it was checked both ways.** `io/stdpair` is 4 PASS / 40 DIFF at the tip, the same four as its
base (`…/base` against `…/areas`); 237 tests of `io/{evolution,tree,stdpair}/*` measured at the tip are **161 PASS /
13 FAIL / 47 DIFF / 16 UNSUPPORTED**, and the two `AddVersion-2-*` DIFFs in it are DIFF at the **base** too
(`…/basever`, measured against the primary checkout's tree) - so the version bookkeeping cost nothing.

**What is left is `W0-239` and `W0-230`'s offsets.** Our reader numbers an emulated class's layouts **densely from
1** (`note_file_infos`'s `_free_version`, whose docstring claims ROOT "numbers its layouts as they come"), and the
pair's files put them at 1 and 2 where the refs print `#1`..`#4` - a dense counter cannot produce the hole at index 1
that makes ROOT print `#2` rather than `#1`. That is the next piece, with `TClass::GetStreamerInfo`'s bookkeeping to
read first, and then the two-line offsets table.

**The gate.** Seven commands at `971651c5`, each read for its own exit status: **`TOTAL 125988 0 31626 0 100%`**
(13 967 + 44 passed, `…/gate1.log`), ruff clean, mypy 1 062 files, maintainability 33 339 functions, components
1 062 modules / 0 undeclared, `mkdocs build --strict` exit 0 (`…/mkdocs.log`).
| 110 | `6249ef1b` | **the `io/stdpair` bucket's two unknowns measured on the 6.40 oracle, and a first fix reverted**: `GetStreamerInfos()->GetSize()` is ROOT's `TObjArray` room (13, not the 2 entries) - the walk a test makes only reaches `#2` because of it; an entry must be the *same object* `GetStreamerInfo()` answers; and `second`'s `offset=` 4-vs-8 is ROOT's emulated `AlignUp(offset, align) + size` run again when the map's proxy is created (`0,0` before, `0,8` after) | no code | **0 PASS moved, 0 lost**: `io/stdpair` 4 / 40 / 0 / 0 either side; the first fix (Expand + per-class cache) broke four existing tests and was reverted - those four are the next attempt's specification. Gate unchanged (`TOTAL 125988 0 31626 0 100%`, round 109's) | 0 new: `W0-239` **rewritten** with the three measured pieces, `W0-230` **carries the offsets rule** |
| 111 | `cee9a211` | **`io/stdpair`'s fix measured to five rules, the fifth tried and the whole attempt reverted**: the room and the entry identity are one change (the room alone prints `#1` as well as `#2`); a class in memory lists as met and only an emulated one is indexed by version (`Evo`: 2 blocks against the ref's 3); building follows the element (an enum carries `newtype=13`, a string element stays `Emulation`); the current entry keeps `from_file` (`offset=99999 type= 3 newtype=-1`); and the pair's dictionary warning needs a known-type predicate first - `stlNoLibTest.ref` warns of `pair<float,int>` where this kit warns of no pair at all | no code | **0 PASS moved, 0 lost**: `io/stdpair` 4 / 40 / 0 / 0 of 44 and `io/emulated` 4 / 1 / 1 / 2 of 8 at the tip, both their predecessors' figures; four attempts reverted, each after the reflection suite named what it broke. Gate unchanged (`TOTAL 125988 0 31626 0 100%`, round 109's) | 1 new: `W0-240` (the dictionary-warning predicate); `W0-239` **rewritten** with the five rules and the seven pinning tests |
| 112 | `46c791dc` | **the layouts array as ROOT keeps it, and the offsets ROOT accumulates - `io/stdpair` 4 -> 8 PASS**: one array per class (entries are the objects `GetStreamerInfo()` answers), ROOT's **thirteen** slots, indexed by class version for a versioned or emulated class, every emulated layout **built** (`built=1` in every slot of the 6.40 oracle), an older layout compared against the current one (`newtype=`, actions two hundred above the file's type) and `offset = AlignUp(offset, GetAlignment()) + GetSize()` with 8 for a container and 4 otherwise - with three unit tests corrected against that measurement | `3b1d6e89` | **roottest +4**: `io/stdpair` 4 / 40 -> 8 / 36 of 44, the four `pairEnumEvo_{0,1}_{0,1}_0_1` and none lost; `io/emulated` 4 / 1 / 1 / 2 unchanged; baseline raised by `--merge`, its only diff those four rows. Gate `TOTAL 126034 0 31642 0 100%` (13 968 + 44), ruff, mypy 1 062 files, maintainability, components, mkdocs | 3 new: `W0-241` (the eight `kNothing` refs' `offset= 8`), `W0-242` (which file's layout is current), `W0-243` (the `Emulation` title and an enum's `type=13`); `W0-239` and `W0-240` partly closed |
| 113 | `47e0b444` | **no code: `io/stdpair`'s remaining 36 DIFFs counted to the line with the judge's own list comparison**, and the one state they wait on named - every test is 5-19 lines from PASS; the six nearest differ only by the pair's dictionary warning and `second`'s `offset= 8`; and the 8-vs-4 split is the *loaded-dictionary* state (`kNothing` and interpreted say 8, a loaded dictionary 4), the same state that decides which file layout is current | no code | **0 PASS moved, 0 lost**: `io/stdpair` 8 / 36 of 44 at the base, `io/datamodelevolution` 22 / 20 / 2 of 44 re-measured (`…/base-dml`), its chain still four features. Gate unchanged (`TOTAL 126034 0 31642 0 100%`, round 112's) | 1 new: `W0-244` (**model the dictionary a macro loads** - alignment, current layout and the warning at once); `W0-241`/`W0-242` marked as its halves |
| 114 | `291a9f05` | **a member's alignment follows the dictionaries the session loaded, and a synthetic pair of unknown classes is warned of** - `TStreamerInfo::BuildOld` aligns at 8 for a class no dictionary has (ROOT's emulated guess) and 4 for one the session knows, a template instance through its template; and `_undictionaried` warns of a `pair<A,B>` nothing answers for | `a60dc3d9` | **roottest +4**: `io/stdpair` 8 / 36 -> **12 / 32** of 44, the four `pairEnumEvo_4_{0,1}_{0,1}_1` and none lost; baseline raised by `--merge`, its only diff those four rows. Gate `TOTAL 126053 0 31650 0 100%` (13 971 + 44), ruff, mypy 1 062 files, maintainability, components, mkdocs | 0 new: `W0-244` **two thirds landed**, `W0-241` **closed**; what is left is the duplicate `StreamerInfo` block and the current-layout choice (`W0-242`) |
| 115 | `e52042af` | **a load re-emulates an emulated class, so its current layout is listed twice** - `loading.load_library` calls `reflection.reload_emulated()`, the array keeps a second copy of the current layout, a taken slot is the next free one, and `GetStreamerInfo(v)` is the array's `At(v)` | `54a1412b` | **roottest +4**: `io/stdpair` 12 / 32 -> **16 / 28** of 44, the four `pairEnumEvo_{0,1}_{0,1}_1_1` and none lost; `io/tree` 18/0/0/2 and `meta/evolution` 20/7/8/2 unchanged; baseline raised by `--merge`. Gate `TOTAL 126072 0 31660 0 100%` (13 972 + 44), ruff, mypy 1 062 files, maintainability, components, mkdocs | 0 new: `W0-244` **three quarters landed**; what is left is `2_*_1_1`'s current-layout choice and the four `4_*_*_0` `kNothing` families |
| 116 | `81364c2b` + `12ec0b60` | **`io/stdpair` 16 -> 20 PASS - a file's layout is matched to a held one by checksum only where the held one was recorded unversioned** - `TStreamerInfo::BuildCheck` looks a layout read from a file up by **checksum** only against an entry whose `GetOnFileClassVersion()` is 0 or 1 and by class version otherwise, so the same bytes can be held twice, at two slots: `io/stdpair`'s `ab02.root` (the pair at version 1, `0xb15d1241`, and at version 2, `0xb5fb752`) followed by `aa02.root` (`0xb5fb752` as version 1) lists `At(1) 0xb15d1241`, `At(2) 0xb5fb752`, `At(3) 0xb5fb752` - measured with two oracle probes (`probe/one.C` per file, `probe/steps.C` in both orders), not deduced. The version each layout recorded is kept beside the slot it is listed at, because `_free_version` renumbers a layout that lands above one already taken; the slot-based first shape **broke** `test_files_layouts_are_numbered_as_met_while_no_class_is_in_memory` and the recorded version is what the second shape reads | `81364c2b` | **roottest +4**: `io/stdpair` **20 / 24 of 44** (`…/run3`), the four `4_{0,1}_{0,1}_0` `kNothing` families; `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2, `meta/evolution` 20/7/8/2, `io/datamodelevolution` 22/20/2 re-measured whole and unchanged (`…/guards`, 64/28/11/6 of 109). Gate **`TOTAL 126083 0 31666 0 100%`** (13 971 + 44 passing, one pre-existing failure), ruff, mypy 1 062 files, maintainability 33 354 functions, components, `mkdocs --strict` | 2 new: `W0-245` (a synthetic pair's in-memory layout is the **emulated** one - the `0_*`/`1_*` refs are byte-identical in either file order, which corrects round 113's "the order decides the current layout"; 8 tests) and `W0-246` (a file layout's offsets are laid out once, as the file is read, so an `#include` afterwards does not move `second` from 8 to 4; 4 tests); `W0-243` gains the evidence, `W0-244` is four fifths landed; a rule that skipped `_current_layout`'s file-over-memory swap for a synthetic pair **moved no test** and was reverted |
| 117 | `e8cdedd5` + `499031e4` | **`io/stdpair` 20 -> 24 PASS - a file layout's member offsets are laid out once, as the file is read** - `TStreamerInfo::BuildOld` accumulates them then (`AlignUp(offset, align)`, then `offset += GetSize()`) and a dictionary or an `#include` arriving later does not move them; the one event that lays them out again is a **library load**, which re-emulates the class so `BuildOld` runs a second time. `io/stdpair` states both halves with the same inputs: `2_*_1_*` read before the include and print `second offset= 8`, `2_*_0_*` read after it and print 4, `0_*_1_1`/`1_*_1_1` read before the load and print 4 - and the freeze-with-no-thaw first shape **lost the four `0_*_1_1`** before `_thaw` in `reload_emulated` supplied the other half (`…/run1` vs `…/run2`). `W0-247` closed as a **test** corrected against the oracle: ROOT finds the object a `TRef` names when the session itself wrote the file (`…/probe/tref.py`), so `GetObject() is None` was the bug; its first correction assumed the session's object count, which the gate's `-n 4` caught (`W0-117`), and the member read now carries `ref.GetUniqueID()` | `e8cdedd5`, `499031e4`, `f87feed3` | **roottest +4**: `io/stdpair` **24 / 20 of 44** (`…/run2`), the four `2_{0,1}_1_{0,1}` families; `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2, `meta/evolution` 20/7/8/2, `io/datamodelevolution` 22/20/2 re-measured whole and unchanged (`…/guards`, 64/28/11/6 of 109). Gate **`TOTAL 126095 0 31672 0 100%`** - part 1 **13 975 passed, 0 failed** (99.75 % by design, so its exit status 1 is the floor, not a test), part 2 44 passed - ruff, mypy 1 062 files, maintainability 33 360 functions, components, `mkdocs --strict` | 2 new: `W0-248` (with the pair's dictionary loaded ROOT lists the mapped layout at `#1` beside the current one and stays silent about `edm::Ref`; 8 tests) and `W0-249` (`2_*_0_*` wants `TStreamerInfo::Build`'s per-member warning, not `TClass::Init`'s per-class one; 4 tests); `W0-245` sharpened with a `rootcling`-built dictionary probe that answers both file orders; `W0-246` and `W0-247` **closed** |
| 118 | `c37ffb3a` + `a32733b1` | **`io/stdpair` 24 -> 28 PASS - a pair a library's dictionary reaches is emulated before a file's record** - ROOT builds the emulated layout for a class a loaded library's dictionary lays out the members of, registers it at the class's own version before the file's records, and drops a file's copy as the same layout, so the emulation's spelling is current in either file order; with nothing at all the class is made from the file's first record instead. Four shapes measured (18/26, 22/22, 24/20, **28/20**) before the predicate was right: `GENERATED`, a load's LinkDef selection under the **plain** name (`Ref`), not an `#include`'s declaration and not the type's bare name | `c37ffb3a`, `a32733b1` | **roottest +4**: `io/stdpair` **28 / 16 of 44** (`…/run5`), the four `pairEnumEvo_{0,1}_{0,1}_0_0`; `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2, `meta/evolution` 20/7/8/2, `io/datamodelevolution` 22/20/2 re-measured whole and unchanged (`…/guards`, 64/28/11/6 of 109). Gate **`TOTAL 126112 0 31680 0 100%`** - part 1 13 978 passed, 0 failed (99.75 % by design), part 2 44 passed - ruff, mypy 1 062 files, maintainability 33 366 functions, components, `mkdocs --strict`; the gate named one uncovered statement, which a third test now reaches | 0 new: `W0-245` **three quarters landed** (what is left is `0_*_1_0`/`1_*_1_0`'s four blocks), `W0-248`/`W0-249` re-priced from the refs |
| 119 | `d8899a87` + `c4ac0fef` | **`io/stdpair`'s four `2_*_0_*` families priced to two lines, and the state a class's current layout is decided in** - `DESCRIBED` keeps what the session knew **when the file was read** ("dictionary" -> emulated own layout, "declaration" -> the file's mapped spelling, `None` -> the first record), and `_undictionaried` is silent for a type `_has_dictionary` answers. 0 PASS moved, 0 lost - the wall is what the round bought, and `_could_be_emulated` came out as dead code | `d8899a87`, `c4ac0fef` | **0 PASS moved, 0 lost**: `io/stdpair` **28 / 16 of 44** either side (`…/run1`, `…/run2`), the four families now two lines from a PASS; guards re-measured whole with `io/newstl` added - 83 / 39 / 13 / 6 of 141, every figure the ledger's. **+4 unit tests.** Gate **`TOTAL 126133 0 31692 0 100%`** (part 1 13 982 passed / 0 failed, part 2 44), ruff, mypy 1 062 files, maintainability 33 372 functions, components, `mkdocs --strict` | 0 new: `W0-249` two thirds landed, `W0-245` sharpened (its remainder is the synthesised own layout, and the refs' checksum is not one this kit can compute) |

## Round 110's close

Round 110 wrote no code, and the round says so: it spent itself turning `W0-239`'s two guesses into measurements, and
then **reverted the fix those measurements made possible** when four existing tests said its shape was wrong.

**What the oracle says, in four probes.** With the roottest's own `aa02.root`/`ab02.root` (`…/probe/slots.C`,
`elems.C`, `testlike.C`, `sizes.C` in `…/ralph-round-202611241641`):

- **`GetStreamerInfos()->GetSize()` is the `TObjArray`'s room, not its entries** - `13` for `TNamed`, `TObject` and
  `pair<int,int>`, `15` for `TH1F`, `GetLast() -2` for a class with no layouts - and the pair's array holds
  `At(1) = v1 0xb5fb752` (the current) and `At(2) = v2 0xb15d1241`, `size=13 last=2 entries=2`, in **either** file
  order. So `pairEnumEvo`'s `GetSize()-1` walk reaches `#2` only because the room is past the highest version; this
  kit answers `GetSize() 2`, the loop runs zero times, and no `#N` block is printed at all.
- **An entry has to be the same object** `GetStreamerInfo()` answers, because the test prints only what
  `info != currentInfo`; `GetStreamerInfo(i)` here builds a fresh `TStreamerInfo` per call, so a first attempt printed
  `#1` as well as `#2`.
- **`_free_version`'s dense numbering is right** - the collision takes v2 - so round 109's version-indexed array was
  the correct half.
- **The `4`-against-`8` split of `second`'s `offset=` is ROOT's emulated accumulation**: the elements answer `offset`
  **0 and 0** (`align` 4 and 8, `size` 4 and 24) with only the files open, and **0 and 8** after the test's own
  `c->GetClassInfo(); c->GetCollectionProxy(); p->GetValueClass()` - `TStreamerInfo::BuildOld`'s
  `offset = AlignUp(offset, element->GetAlignment()); SetOffset(offset); offset += element->GetSize()`, run a second
  time by the proxy. That is why the 40 refs split: a family with a **library loaded** prints the compiled class's
  real offsets (`4`), one without it the emulated accumulation (`8`).

**And the fix was reverted.** `Expand(13)` plus a per-class array cache made the blocks right and broke four existing
tests - `built` for an emulated class, `FindStreamerInfo`'s unbuilt answer, and a version collision that must bump
rather than replace. Shipping it with four red tests is not an option and relaxing them is not either, so the tree is
exactly its base's and **those four tests are the next attempt's specification** - which is worth more than a
half-finished cache.

**Nothing moved, and that is consistent.** `io/stdpair` is 4 PASS / 40 DIFF at the tip and at its base: the room and
the offsets are both needed, and neither alone flips a single file.

**The gate.** Nothing in the tree changed (`git diff` against `6249ef1b` is `TODO.md` and `.ralph/ROOT.md`), so round
109's seven-command gate stands: **`TOTAL 125988 0 31626 0 100%`** (13 967 + 44 passed, `…/ralph-round-202611241641`'s
base is that tip), ruff clean, mypy 1 062 files, maintainability 33 339 functions, components 1 062 modules / 0
undeclared, `mkdocs build --strict` exit 0.

## Round 111's close

**The bucket was opened one line at a time.** `pairEnumEvo_4_0_0_1` run under this kit against its own ref is **2 lines
only ours, 11 only the ref**: the `edm::Ref` element's `offset= 0` where the ref says `8` (twice), one `#2 StreamerInfo:`
block of six lines, and three `TClass::Init` dictionary warnings. That is a small, complete list - the round priced
exactly it.

**The fix is five rules, and only one of them is bookkeeping.** (1) The 13-slot room and the entry identity are **one**
change: the room alone makes the walk print `#1` beside `#2`, which is worse than the base, so a half of it is not
shippable. (2) `by_version` is `_layout() is None and any(version)`, not `any(version)` - indexing `Evo` by version
collapses its memory layout and its first file layout into one slot and prints two blocks where the ref has three, so a
class in memory lists as met and only an emulated one is indexed by version. (3) Building follows the **element**, not
the layout: the ref's enum element is built (`newtype=13`) and `pair<int,string>`'s string element
(`type=300 stl=365 ctype=365 Emulation`) never is, so `test_a_layout_a_file_holds_is_listed_beside_the_one_in_memory`'s
`"   i=" not in` stays true. (4) The array's current entry keeps `from_file` when the current layout is the file's own,
because `test_a_layout_the_class_in_memory_disowns_is_listed_as_roots_file_layout` needs `offset=99999 type= 3
newtype=-1`. (5) The pair's missing warning is a **separate predicate** and needs a known-type test first.

**And it was reverted, four times, at the point the suite named what it broke.** The fifth rule was tried on its own and
measured: the stdpair line appears, and `pair<int,string>` warns too, because nothing here models `string` - so the
predicate has to exist before the rule can. Each attempt ended in `git checkout -- src tests`; the tree is its base's
exactly, and the seven pinning tests are the next attempt's specification. That is the same trade as round 110 and a
better one than a half-finished cache.

**Nothing moved, and that is consistent.** `io/stdpair` is **4 PASS / 40 DIFF** of 44 at the tip and `io/emulated`
**4 / 1 / 1 / 2** of 8, both exactly their predecessors' figures: no ref is warning-only, offsets-only or blocks-only, so
no single rule flips a file. What the round bought is the list of rules and the five `.ref`-backed facts behind them
(`io/newstl/stlNoLibTest.ref` warns of `pair<float,int>` and `pair<short,SameAsShort>`, where this kit warns of no pair at
all).

**The gate.** No code: `git diff` against `cee9a211` is `TODO.md` and `.ralph/ROOT.md` alone, so round 109's seven-command
gate stands - **`TOTAL 125988 0 31626 0 100%`** (13 967 + 44), ruff clean, mypy 1 062 files, maintainability 33 339
functions in 1 824 files, components 1 062 modules / 0 undeclared, `mkdocs build --strict` exit 0. It was re-measured on the
reverted tree anyway, all seven commands, at a load average of 6.3: **13 967 + 44 passed, `TOTAL 125988 0 31626 0 100%`**,
ruff clean, `mypy` clean, maintainability and components clean, `mkdocs --strict` exit 0 - the same figures round 109
measured, which is what a round that ships no code should read.

## Round 112's close

Round 112 is the first round since 107 to move the harness number, and it moved **four tests** on the
loop's biggest bucket: `roottest/root/io/stdpair` goes **4 PASS / 40 DIFF -> 8 PASS / 36 DIFF**, with
nothing lost and `io/emulated` unchanged at 4 / 1 / 1 / 2.

**What was already measured, and what the oracle settled this round.** Three rounds had priced this
bucket; round 112 opened by asking the 6.40 oracle the one question they had left open -
*is an emulated class's old layout built too?* - with two files the roottest itself ships
(`probe/lists.C`): `GetStreamerInfos()->GetSize()` is **13**, `GetLast()` 2; every occupied slot has
`IsBuilt() 1` **immediately**; the current entry is the **same object** `GetStreamerInfo()` answers
(`same=1`); and a ROOT-written `pair<int,string>` lists `i= 0, first type= 3, offset= 0` and
`i= 1, second type=300, offset= 8` - so **the action lines are printed for an emulated class's
layout**, which one of our own unit tests had asserted the opposite of.

**What landed** (`3b1d6e89`, one commit): `TClass` keeps **one** layouts array per class, so an entry
is the object `GetStreamerInfo()` answers; the array has ROOT's **thirteen** slots and is indexed by
class version for a versioned or an emulated class (as-met for a class in memory); every layout of an
emulated class is built, `from_file` survives on a current entry that came from a file, and an older
layout is compared against the class's current one, so its elements carry `newtype=13` and read with
an action two hundred above the file's type. And `TStreamerElement`'s `offset=` is ROOT's
`AlignUp(offset, GetAlignment())` then `offset += GetSize()`, with an alignment of 8 for a container
and 4 otherwise - measured, not guessed: the oracle's `pair<int,string>` puts its string at 8 and its
emulated pair puts the 24-byte `edm::Ref` at 4.

**Three unit tests corrected against the measurement, not loosened.** `GetSize() == 4` became 13 (the
room); a synthetic two-int layout's second action line `offset= 0` became 4 (the accumulation); and
`"   i=" not in printed` became the two action lines the oracle prints. Each carries the oracle's
own output or ROOT's source line in its comment.

**The four gained** are `pairEnumEvo_{0,1}_{0,1}_0_1` - the `reportedorder=1` families of `libtype` 0
and 1 - and the baseline recorded exactly those four rows (`rootsuite baseline --merge`; `ratchet` is
not applicable to a `--only` run, as `W0-158` says). The gate is green: **`TOTAL 126034 0 31642 0
100%`** (13 968 + 44), ruff, mypy 1 062 files, maintainability 33 3xx functions, components 1 062
modules / 0 undeclared, `mkdocs --strict`.

**The round also cost itself two gate cycles, and both are the gate doing its job**: `GetStreamerInfos`
pushed `TClass._held` to **NPath 360** and a line past 100 characters, and the first green-ish run
named two uncovered lines (`_version_of`, now dead, and the array cache's return). The functions were
split (`_held_layouts`, `_indexed_by_version`) and a test now pins that asking a class for its array
twice hands back the same one.

**What is left is three named rows, each measured this round**: `W0-241` (the eight `kNothing` refs
want `offset= 8` where the 32 others want 4 - ROOT's alignment knows the class only when a dictionary
is loaded, and this kit cannot tell the configurations apart), `W0-242` (which file's layout is
*current*: the `libtype=3` ref keeps the last, we keep the first, which reorders their blocks) and
`W0-243` (the `Emulation` element title and an enum's `type=13`). With those, the remaining 36 DIFFs
of this bucket are arithmetic and ordering rather than a missing feature.

## Round 113's close

Round 113 wrote no code, and its product is the map the last three rounds had been circling: **the
remaining 36 `io/stdpair` DIFFs, counted to the line with the judge's own comparison, and the one
state they all wait on.**

**The counting was redone with the judge's rule.** Round 112's bucket table was built by comparing
the ref and our output as *multisets* of space-stripped lines, which under-reports a **swapped pair of
blocks** (the same lines, another order) - and that is exactly what the failing families do. The
judge is `refdiff.filtered(ref, True) == filtered(out, True)`, a **list** comparison, so the round
recounted with `difflib.SequenceMatcher` over those same lists. The true figures: every one of the 36
is **5 to 19 lines** from PASS (0-8 ours-only against 5-19 ref-only), and the 322 ref-only lines are
**132 element lines, 120 action lines, 34 `#N` block lines and 72 others** (warnings, block headers,
and one `TStreamerInfo::Build` warning that only two refs carry).

**The six nearest are five lines from a PASS**, and they need exactly two rules:
`pairEnumEvo_{4_0_0_1, 4_0_1_1, 4_1_0_1, 4_1_1_1, 2_0_1_1, 2_1_1_1}` differ by the missing
`Warning in <TClass::Init>: no dictionary for class pair<...>` (1 line, `W0-240`) and `second`'s
`offset=` being **8** rather than our 4 (4 lines). Nothing else.

**And the round proved one cause behind that 8-versus-4**, from ROOT's refs alone: the families whose
run **loaded a dictionary** print `offset= 4` and the run that loaded nothing (`libtype=4`,
`kNothing`, and the interpreted `libtype=2` reading first) prints **8**. Our kit prints 4 everywhere
because it models no load - which is why round 112's four gains are exactly the four tests whose ref
also says 4. The same state decides **which file layout is current**: `pairEnumEvo_0_0_0_0`'s two
blocks are ours with the roles swapped, and `pairEnumEvo_2_0_0_1`'s current layout is v2 where
`0_0_0_1`'s is v1 - one state, two symptoms. That is **`W0-244`**, and `W0-241`/`W0-242` are its two
halves; the three rows now say so instead of describing the symptoms separately.

**What the round also re-measured, so the next one does not pay for it.** `io/datamodelevolution` is
**22 PASS / 20 FAIL / 2 DIFF** of 44 at this round's base (`…/ralph-round-202611241744/base-dml`),
unchanged from round 108, and its failures still want the four features `W0-233` lists
(`GetDataType`/`GetOffset`/`TClass::New`, a write through `(char*)obj + offset`, free-function
overload selection, `FindConversions`) - not a round's worth of one-line work.

**No code, so the gate stands**: `git diff` against `47e0b444` is `TODO.md` and `.ralph/ROOT.md`
alone, and round 112's seven-command gate - **`TOTAL 126034 0 31642 0 100%`** (13 968 + 44), ruff,
mypy 1 062 files, maintainability, components, `mkdocs --strict` - is this round's.

**The round's own lesson, recorded because it bit twice.** A multiset comparison is not the judge
(`W0-75`, round 104), and a *list* comparison is - so a bucket table built from counts must be built
from the list diff, or it will call a swapped pair "one line" when the judge sees a different test.

## Round 114's close

Round 114 is the first round since 107 to move the harness number by **four**, and it moved them on
the loop's biggest bucket: `roottest/root/io/stdpair` goes **8 PASS / 36 DIFF -> 12 PASS / 32 DIFF**,
nothing lost.

**What round 113 said was missing, and it was already in the kit.** Round 113 priced the last 36
DIFFs to the line and proved they wait on one state - *the dictionary a macro loads* - which decides
a member's `offset=` (8 or 4), which file layout is the class's current one, and whether the pair's
`TClass::Init` warning is printed. This round found that state already exists: `gSystem->Load` of the
`.xrdlib` a fixture built reads its manifest and runs the classes it declares, so
`declared.find_class("edm::Ref")` answers in exactly the families whose run loads a dictionary and
answers nothing in the twelve that load nothing (`kNothing`, and the interpreted run that reads
first). A one-line probe over the harness's own working copy showed the split.

**What landed** (`a60dc3d9`). `TStreamerInfo::BuildOld`'s alignment is now the session's own:
**eight for a class no dictionary has declared** - ROOT's emulated guess, which is the twelve
families' `offset= 8` - and **four for a class the session knows**: a number, a class `winfo`
describes, or one a library it **loaded** declared, a template instance through its template
(`_has_dictionary` / `_answers` / `_bare_type`; a container still aligns at 8, which is where
`pair<int,string>`'s string sat in round 112's oracle probe). And `_undictionaried` no longer
silences every synthetic `pair<A,B>`: it warns when nothing in the session answers for it, which is
ROOT's own `Warning in <TClass::Init>: no dictionary for class pair<...> is available` - silent for
`pair<int,string>`, printed in the twelve dictionary-less families.

**The four gained** are exactly the `kNothing` quartet, `pairEnumEvo_4_{0,1}_{0,1}_1`: round 113 had
counted them at **one line** from a PASS - the missing warning - and the alignment finished the rest.
Three unit tests pin the new rules (the alignment's four cases, a type through its template, and the
pair warning including a template that is no pair), and the baseline recorded the four rows with
`rootsuite baseline --merge`. Gate: **`TOTAL 126053 0 31650 0 100%`** (13 971 + 44), ruff, mypy 1 062
files, maintainability, components, `mkdocs --strict`.

**Two gate cycles, both the gate doing its job.** The first run named `_has_dictionary` at **NPath
288** against the limit of 200 - split into `_bare_type`/`_has_dictionary`/`_answers` - and ten
uncovered branches, the last of them the `_template` arm that my first unit test asserted through the
wrong path (the warning it saw came from the generic fall-through, not the pair rule). Both are fixed
in the tree that ships.

**A revert, recorded.** The alignment rule's first shape asked only `find_class(bare)`, which does not
answer a template instance, so every family printed 8 and the four *loaded* tests that had passed
since round 112 were **lost**; the run that showed it (`…/run1`) is kept, and the rule that answers an
instance through its template is what replaced it.

**What is left is two causes, both measured.** Six of the remaining families (`0_*_1_1`, `1_*_1_1`,
`4_*_*_0`) want **a third `StreamerInfo` block** - the duplicate the macro's own comments describe -
and `2_*_1_1`'s four-line difference is the current-layout choice (`W0-242`). `W0-244` carries both,
`W0-241` is closed by the alignment rule, and `io/stdpair` is now 12 of 44 with the remaining 32 each
6 lines or fewer from a PASS.

## Round 115's close

Round 115 is the third round in five to move the harness number, and it moved **four tests** on the
bucket that has carried the loop since round 103: `roottest/root/io/stdpair` goes **12 PASS / 32
DIFF -> 16 PASS / 28 DIFF**, nothing lost.

**What round 114 left, and what ROOT does.** Round 114's close named the two remaining causes of that
bucket - a **missing duplicate `StreamerInfo` block** in six families, and `2_*_1_1`'s current-layout
choice. The duplicate is ROOT's own documented behaviour: `pairEnumEvo.C`'s comment says *"'missing'
StreamerInfo and after update (use ForceReload in GenCollectionProxy) it now has duplicated
StreamerInfo"*, and a macro that prints its `fStreamerInfo` for a class **read before a library load**
shows `At(1) = v1`, `At(2) = **v1 again**`, `At(3) = v2` - the class was emulated from the file, then
emulated **again** when the library arrived, and both copies are kept.

**What landed** (`54a1412b`). `pycint/library/loading.py` calls a new
`reflection.reload_emulated()` after it declares what a library defines; that marks every class
**nothing in memory describes**, and `_held_layouts` lists such a class's current layout **twice** -
a second object, not the same one, which is why the macro prints it at all. Two supporting pieces
were needed and the first attempt failed without them: a version slot already taken is now the next
free one (`while version in taken: version += 1`), and **`GetStreamerInfo(version)` answers the entry
the array holds at that index** instead of a separate version map - which is exactly the bug the
round's own probe found (`At(2)` held the duplicate while `GetStreamerInfo(2)` answered the older
layout, so the printer showed the wrong block and only one of them).

**Measured.** The four gained are `pairEnumEvo_{0,1}_{0,1}_1_1` - every `readbeforeload=1` family of
the two shared/ACLiC libraries, which is precisely the configuration the duplicate belongs to. Two
areas that touch the same reflection were re-measured at the tip and are unchanged: `io/tree`
**18 / 0 / 0 / 2** and `meta/evolution` **20 / 7 / 8 / 2** of 37. The baseline recorded the four rows
with `rootsuite baseline --merge`. Gate: **`TOTAL 126072 0 31660 0 100%`** (13 972 + 44), ruff, mypy
1 062 files, maintainability, components, `mkdocs --strict`.

**The round's own gate caught two things and both were fixed rather than excused**: `mypy` named the
slot map's `Any` (`Returning Any from function declared to return "TStreamerInfo | None"`), and part 2
named two uncovered lines - the duplicate append and `reload_emulated`'s loop - which a new unit test
now reaches (`test_a_load_registers_an_emulated_class_s_layout_a_second_time`).

**One accident, recorded because it nearly shipped**: the round's first commit deleted `mkdocs.yml`
(120 lines) - the file was missing from the worktree when the gate ran, so the gate could not see it,
and only `git diff --stat` before the amend did. It was restored from the base and the commit amended;
`mkdocs build --strict` is green at the shipped tree. **`git diff --stat` before every commit**, not
only before the gate.

**What is left is two causes no load explains.** `2_*_1_1`'s four lines are the **current-layout
choice** for the interpreted run (`W0-242`), and the four `4_*_*_0` families are `kNothing` - no
library is loaded at all, so the duplicate cannot come from one. Both are in `W0-244`, which is now
three parts done of four; `io/stdpair` is 16 of 44.

## Round 116's close

**One rule, four tests, and the reason the four `kNothing` families were six lines short.** `io/stdpair`
was 16 PASS / 28 DIFF of 44 at the base; the round's oracle probes and ROOT's own source found the
rule the remaining `#3` block needed, and it is not about loads, files or offsets at all:
**`TStreamerInfo::BuildCheck` matches a file's layout to one the class already holds by checksum
only against an entry whose `GetOnFileClassVersion()` is 0 or 1**, and by class version otherwise.
The same bytes can therefore be held twice, at two slots - which `pairEnumEvo.C`'s own comment
calls "after update it now has duplicated StreamerInfo" and prints as its `#2` and `#3` blocks.

**Measured, not deduced.** A probe against the 6.40 oracle (`probe/steps.C`, `probe/one.C` in the
round's directory) read each input file on its own and then in both orders: `ab02.root` holds the
pair at **version 1** (checksum `0xb15d1241`, `first type= 3`) *and* at **version 2** (`0xb5fb752`,
`first type=13`), `aa02.root` holds `0xb5fb752` as **version 1**, and read `ab02` then `aa02` the
array is `At(1) = 0xb15d1241`, `At(2) = 0xb5fb752`, `At(3) = 0xb5fb752` - the version-2 entry is
**not** retired by the version-1 record that repeats it, because the entry it would be matched
against was recorded versioned. Read the other way round the version-2 record *is* matched (against
`At(1)`, which was recorded unversioned) and dropped.

**The first shape of the rule was wrong, and an existing test said so.** Comparing the *slot* a held
layout sits at - `held[1] <= 1` - is not the same question: `_free_version` renumbers a layout that
lands above one already taken, so a layout a file recorded as version 1 can sit at slot 2.
`test_files_layouts_are_numbered_as_met_while_no_class_is_in_memory` opens one file twice and must
hold one layout; the slot-based rule held two. The fix keeps the **recorded** version beside the
slot (`RECORDED`, filled where the layout is noted, read by `_same_layout`) and the test is green
again - it was the second attempt's specification, as `R110`/`R111`/`R114` were before it.

**Measured.** `io/stdpair` **16 PASS / 28 DIFF -> 20 PASS / 24 DIFF** (`…/run3`, `--jobs 3`): the
four gained are `pairEnumEvo_4_{0,1}_{0,1}_0`, the `kNothing` families that read `ab02.root` first.
None lost. The four areas that share this reflection code were re-measured **whole** at the tip and
are **the ledger's figures to the test**: `io/tree` 18 / 0 / 0 / 2, `io/emulated` 4 / 1 / 1 / 2,
`meta/evolution` 20 / 7 / 8 / 2 of 37, `io/datamodelevolution` 22 / 20 / 2 of 44 - 64 / 28 / 11 / 6
of 109 (`…/guards`). The baseline recorded the four rows with `rootsuite baseline --merge`. Gate: **`TOTAL 126083 0 31666 0 100%`** (13 971 + 44 passing; part 1 also reports the one
**pre-existing** failure below), ruff, mypy 1 062 files, maintainability 33 354 functions, components
1 062 modules / 0 undeclared, `mkdocs --strict` - each read for its own exit status.

**What the other twenty-four now wait on, measured and written down rather than guessed.** (a) *Eight
of them* - `0_{0,1}_{0,1}_{0,1}` and `1_{0,1}_{0,1}_{0,1}` - need ROOT's in-memory layout of a
**synthetic pair** to be the **emulated** one (`first type=13`, title `Emulation`), so a file's mapped
layout of the same version takes a *slot beside it* instead of becoming the current one. The refs
prove it without any modelling: `pairEnumEvo_0_0_0_0.ref` and `pairEnumEvo_0_0_0_1.ref` are
**byte-identical**, so for a library loaded without a pair dictionary the file order does not matter
at all - which **corrects round 113's** reading that the order decides the current layout (`W0-245`,
which is `W0-243`'s shape at the class level rather than the element's). (b) *Four more* -
`2_{0,1}_1_{0,1}` - need a file layout's offsets to be laid out **once, as the file is read**: those
families say `second offset= 8` and `2_*_0_*` says `4` with the same `#include`, and the difference
is that the former reads the files *before* the include, while this kit asks `_has_dictionary` at
print time (`W0-246`). (c) The eight `3_*` families need both, plus `W0-242`. `W0-244` is four
fifths landed.

**A rule that was tried and reverted this round**: skipping `_current_layout`'s file-over-memory
replacement for a synthetic pair. It moved **no test** (the area read 20/24 either way), so it is
not in the tree; the round's run 2 is the evidence for that.

**A gate that was already red, measured rather than assumed, and a row in this file corrected.** Part 1
of the two-command gate fails **one** test -
`tests/test_pyroot_trees_refs.py::test_the_reference_of_a_member_read_back_names_the_process_of_the_file_it_came_from`
(`assert made.GetObject() is None` gets the `TNamed` this session made) - and it fails **at the round's
base `e6ea6664` and at round 115's own tip `dee8fd28`** in fresh worktrees (`…/base`, `…/base2`), so
this round neither caused it nor can it honestly claim it gone. What the round *did* find is that
round 115's `gate2.log` records **`part1=1`**: its close quoted `TOTAL 126072 0 31660 0 100%` from
`coverage report` without reading part 1's **exit status**, so the row above it in this file
overstates that gate by one failing test. The lesson is the one `R96` already paid for - *the gate is
seven commands and every one of them is the gate, and each one's exit status is read* - and it is
now written into the run's own gate script as `echo "part1=$?"` at the point it can still be seen.
The failure itself is queued as **`W0-247`**, not patched: it is in the process-id/`TRef` filing that
another round is working in, and a fix guessed from one assertion would break the two tests beside it.

## Round 117's close

**One rule with two halves, four tests, and the gate's part 1 green for the first time in four rounds.**
`io/stdpair` was 20 PASS / 24 DIFF of 44 at the base and is **24 / 20** at the tip.

**The rule.** `TStreamerInfo::BuildOld` accumulates a file layout's member offsets **when the file
is read** - `AlignUp(offset, element->GetAlignment())`, then `offset += GetSize()` - and what the
session learns afterwards does not move them. The one event that lays them out again is a **library
load**, which re-emulates the class (`TClass::Update` through the collection proxy) so `BuildOld`
runs a second time. `io/stdpair` states both halves with the same two input files: the `2_*_1_*`
families read them before `#include "cmspair.h"` and print `second offset= 8` - the emulated guess,
an unknown class aligning at 8 - while the `2_*_0_*` families read after it and print **4**
(`edm::RefCoreWithIndex` is `ushort, ushort, unsigned int`, so 4 is its own alignment); and the
`0_*_1_1`/`1_*_1_1` families read before the **load** print 4 for the same reason in reverse.

**The first shape was half a rule, and the measurement said so in one run.** A freeze with no thaw
(`FROZEN_OFFSETS`, filled where the layout is noted) took the four `2_*_1_*` families and **lost the
four `0_*_1_1`/`1_*_1_1`** - 20 PASS / 24 DIFF with four in and four out, `…/run1`. `_thaw`, called
from `reload_emulated`, is the missing half: a load drops the frozen offsets so the live computation
- which now knows `edm::Ref` - lays them out again. With it, `…/run2` is **24 PASS / 20 DIFF** and
nothing is lost. The four areas that share this reflection code were re-measured whole and are the
ledger's figures to the test: `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2, `meta/evolution` 20/7/8/2
of 37, `io/datamodelevolution` 22/20/2 of 44 - 64 / 28 / 11 / 6 of 109 (`…/guards`).

**`W0-247` closed, and it was the test that was wrong.** The one failure that made the gate's part 1
exit 1 at rounds 115 and 116 asserted `made.GetObject() is None` for a `TRef` read back in the
session that wrote it. The 6.40 oracle says otherwise (`…/probe/tref.py`): write a `TRef` to a file,
close it, read the reference back - `GetPID()` is the same `ProcessID0` record and `GetObject()` is
the `TNamed` it was made from, because that process record is this session's own. The test now pins
the process's identity and the object's. Its **first correction was itself order-dependent**
(`fUniqueID: 1` assumed the session had numbered nothing else) and the gate's `-n 4` caught it: the
member read now carries `ref.GetUniqueID()`, which is `W0-117`'s lesson - a test that reads global
state must not assume it is the only one that has touched it.

**Measured.** Gate at `f87feed3`: part 1 **13 975 passed, 0 failed**, 4 skipped, 46 xfailed - and
99.75 % by design, which is what its exit status 1 means here (`W0-112`: the two ignored files'
code is not covered by part 1); part 2's report, `TOTAL 126095 0 31672 0 100%`, is the coverage
answer. ruff clean after wrapping one docstring it named at 112 characters, mypy 1 062 files,
maintainability 33 360 functions in 1 824 files, components 1 062 modules / 0 undeclared,
`mkdocs --strict` exit 0.

**What the remaining 20 are, each measured this round.** Eight are `W0-245` - ROOT's in-memory
layout of a **synthetic pair** is the **emulated** one, so a file's mapped layout of the same version
sits beside it and not in its place. A probe with a real ROOT dictionary for the map and the `Ref`
but **not** the pair (`…/probe/probe0.C`, built with `rootcling` + `clang++` this round) answers it
in both orders: after `ab02.root` the array is `At(1) version 1 onfile 1 0xb5fb752` (the emulated
spelling) and `At(2) version 2 onfile 1 0xb15d1241` (the mapped one), and `aa02.root` adds nothing -
the class's own emulated layout comes **first**. This kit has **no** layout of the pair in memory
(`mine=None`), so it lists the file's first layout as current. Four more are `W0-249`: `2_*_0_*`
wants `TStreamerInfo::Build`'s per-member warning (`… has no streamer or dictionary, data member
"second" will not be saved`) where this kit prints `TClass::Init`'s per-class one. Eight are
`W0-248`: with the pair's dictionary loaded, ROOT lists the mapped layout at `#1` beside the current
one and stays silent about `edm::Ref`, and this kit lists a block fewer and warns.

**A rule considered and not tried**: giving a synthetic pair an in-memory layout of its own. The
probe above says what it would have to be (the emulated spelling, checksum `0xb5fb752`), and nothing
in this kit builds one; that is `W0-245`'s first step and not a forty-minute change, so the round
stopped at the offsets and recorded the shape instead.

## Round 118's close

**Four measured shapes of one rule, and four tests: `io/stdpair` 24 -> 28 PASS of 44.**

**What the refs say.** Every failing family listed the *mapped* pair layout as the class's current
streamer info where ROOT lists the **emulated** one - `checksum=0xb5fb752`, `type=13 Emulation`,
action `13` with **no** `newtype`, because nothing compared it against another layout. ROOT builds
that layout itself when a **library the session loaded** carries a dictionary of one of the pair's
members (`cmspair.h`'s pragmas dictionary `edm::Ref`), registers it at the class's own version
**before** the file's records, and then drops a file's copy of it as the same layout - which is why
the file's mapped layout is printed as `#2` and not `#1`. With **no** dictionary at all the class is
made *from the file's first record* instead, so that record's layout is current and the emulated one
sits below it: which is exactly the `kNothing` and interpreted families' three blocks.

**The four shapes, each measured.** (1) "the emulation-spelled layout is the current one", with no
condition at all - **18 PASS / 26 DIFF**, six lost, including the four `kNothing` families that had
passed since round 114 (`…/run1`); the refs say why, and the condition is the point. (2) conditioned
on `_has_dictionary` (which counts a **declaration** the interpreter holds) - **22 / 22**, the four
`{0,1}_{0,1}_0_0` in and the two `2_*_1_*` interpreted families out. (3) conditioned on
`GENERATED` under the type's bare name and its head - **24 / 20**, no gain at all, because a load
records its LinkDef's selection under the *plain* name (`Ref`, not `edm::Ref<...>`). (4) the same
predicate asking the head's **last `::` component**, and requiring that the class itself has no
dictionary - **28 / 20**, the four gained and nothing lost (`…/run5`).

**What landed** (`c37ffb3a`): `TClass._file_layout` adopts the emulation-spelled layout - the one
whose elements are all titled `Emulation`, the title `R__CreateEmulatedElement` writes and
`pairEnumEvo`'s own `printElements` keeps - **at the class's own version**, the version the first
record a file holds gave the class, and not at the version the copy met named; if the class's own
dictionary is what the load selected, or none of its members' types is, the first record stays
current. `_emulated_spelling`, `_could_be_emulated` and `_dictionaried` are the three pieces, and
`tests/test_pyroot_reflection.py` pins all three answers.

**Measured.** The four areas that share this reflection were re-measured **whole** at the tip and are
the ledger's figures to the test: `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2, `meta/evolution`
20/7/8/2 of 37, `io/datamodelevolution` 22/20/2 of 44 - 64 / 28 / 11 / 6 of 109 (`…/guards`). The
baseline recorded the four rows with `rootsuite baseline --merge`.

**The gate, seven commands, each read for its own exit status.** Part 1 **13 978 passed, 0 failed**
(99.75 % by design: the two ignored files' code is not covered there); part 2 44 passed and
`coverage report` **`TOTAL 126112 0 31680 0 100%`**; ruff clean; mypy 1 062 files; maintainability
33 366 functions in 1 824 files; components 1 062 modules / 0 undeclared; `mkdocs --strict` exit 0.
The round's first green-ish run named **one uncovered statement** - `_could_be_emulated`'s
"the class itself has a dictionary" return - which a third unit test now reaches; that branch is
also what keeps `3_*` unchanged.

**What the remaining 16 are, each priced from the refs this round.** Four (`0_*_1_0`, `1_*_1_0`) keep
**four** blocks: the files read before the load, then the load's re-emulation - a shape this round did
not reproduce (ROOT's array there is current + mapped + two copies of the emulation). Four
(`2_*_0_*`) want `TStreamerInfo::Build`'s per-member warning - `pair<...>: edm::Ref<...> has no
streamer or dictionary, data member "second" will not be saved` (`W0-249`) - and every `2_*` family
also wants the `TClass::Init` line *gone*, because the include declares the pair's elements. Eight
are `3_*`: with the **pair's** own dictionary loaded the current is the *file's* mapped layout with
`newtype=13`, and ROOT lists an emulated block at `#1` that this kit does not (`W0-248`).

**A doc, for the next reader**: `docs/gotchas/a-pair-a-library-s-dictionary-reaches-is-emulated-before-a-file.md`,
with the four-way table of what `io/stdpair` prints and the two predicates that decide it.

## Round 119's close

**No PASS moved, none lost, three measured rules landed where one wrong one had been - and the four
`2_*_0_*` families of `io/stdpair` are now two lines each from a PASS.**

**What the refs said, and what was wrong.** The sixteen remaining DIFFs of `io/stdpair` split into
three shapes; this round took the four `2_*_0_*` families - the interpreted ones, whose files are read
**after** `#include "cmspair.h"` - and two things were ours alone: a `TClass::Init` warning ROOT does
not print in the **eight** families that read their files after the include or the load (only in the
eight that read them before, when nothing was declared yet), and a current layout whose blocks were
in the wrong roles. Both came from asking the session's state **at print time**, by which point the
include has answered too - which is exactly why round 118's own measurement of that predicate
(shape 2, 22 / 22) failed.

**What landed** (`d8899a87`). `DESCRIBED` keeps, beside a class's layouts, **how the session
described it when the first was read**: `"dictionary"` for a **library** whose LinkDef selected a
member's type - ROOT emulates the class, so its own layout is the emulated spelling (round 118's rule,
kept); `"declaration"` for an `#include` - ROOT lays the class out from the declaration, so its own
layout is the file's **mapped** spelling, which is what the `2_*_0_*` refs' current layout is; and
`None` for nothing at all, where the class is made from the file's first record (the `kNothing` and
read-before-load families, which is what keeps the eight `*_*_1_*` families passing). And
`_undictionaried` is silent for a type `_has_dictionary` answers: `edm::Ref<...>` is no missing
dictionary once `cmspair.h`'s `edm::Ref` is declared, however the instance is spelled.

**Measured: a wall removed rather than a PASS won.** `io/stdpair` is **28 PASS / 16 DIFF** before and
after (`…/run1` for the two warning rules, `…/run2` with the third), and the four `2_*_0_*` diffs fell
from *warning + block roles + comparison* to **two lines each**: the missing `TStreamerInfo::Build`
warning (`W0-249`'s tail) and the **`newtype=13` / action `203` on the current layout**, which needs
the class's own layout to exist so the file's can be compared against it - the synthesised emulation
`W0-245` has carried since round 118. The guards were re-measured **whole** with `io/newstl` added to
them: `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2, `meta/evolution` 20/7/8/2 of 37,
`io/datamodelevolution` 22/20/2 of 44 and `io/newstl` 19/11/2 of 32 - **83 / 39 / 13 / 6 of 141**,
every one the ledger's figure (`…/guards`).

**Three shapes were tried and the first was reverted.** Handing back a *rebuilt* tuple for the
emulated case broke `test_a_layout_a_file_holds_is_listed_beside_the_one_in_memory`: the array's entry
must be **the** object the file holds (round 112's rule), so the entry is returned unchanged where it
already names the class's version and rebuilt only where it does not - `_file_layout` split into
`_file_layout` + `_pair_own_layout` when the maintainability gate named it at ccn 13 / cognitive 18 /
npath 900, and the gate's own coverage report took out `_could_be_emulated`, dead once `_describable`
replaced it.

**A scripted-slice slip, caught by `git diff --stat` and nothing else.** Cutting the dead function out
with `s[:start] + s[end:]` whose indices were in the wrong order **duplicated** 802 characters -
`_describable` was defined twice and the tests still passed - and the `git diff --stat` after the edit
is what showed it (`17 insertions` where a deletion was due). That is `W0-94`'s trap exactly; the file
was restored from `HEAD` and the cut redone with the order asserted.

**+4 unit tests** (101 in the reflection file), which is what the round's second counter moved by:
`test_a_template_a_bare_declaration_answers_for_is_no_missing_dictionary`,
`test_a_class_s_layouts_remember_whether_the_session_described_it_at_the_read`,
`test_a_pair_the_interpreter_only_declares_is_laid_out_from_the_declaration` and
`test_a_pair_a_dictionary_describes_still_reads_a_first_record_its_own_spelling_lacks`.

**The gate, seven commands, each read for its own exit status.** Part 1 **13 982 passed, 0 failed**
(99.75 % by design: the two ignored files' code is not covered there); part 2 44 passed and
`coverage report` **`TOTAL 126133 0 31692 0 100%`**; ruff clean; mypy 1 062 files; maintainability
33 372 functions in 1 824 files; components 1 062 modules / 0 undeclared; `mkdocs --strict` exit 0.
The baseline needed no row (no PASS moved), and `docs/gotchas/a-pair-a-library-s-dictionary-reaches-is-emulated-before-a-file.md`
gained the sentence this round proved.

## Round 120's close

**`io/stdpair` 28 → 32 PASS, none lost: a declared class's own layout is the emulated spelling, and
`TStreamerInfo::Build` is what a *missing record* makes ROOT do.**

**The round's own choice was measured, not assumed.** The queue's two best-priced targets were the
last sixteen `io/stdpair` DIFFs and `W0-233`'s twenty `io/datamodelevolution` FAILs. The second was
measured first (`--only 'roottest/root/io/datamodelevolution/*'`, `…/dme0`): **22 PASS / 20 FAIL / 2
DIFF** of 44, and its first wall is `TDataMember.GetDataType` in a `Generator.h` that walks a class's
members and writes their values **through raw pointers** (`*(int*)(((char*)obj) + member->GetOffset())
= genInt()`), needing `GetDataType`, `GetTrueTypeName`, `GetOffset`, `TClass.New`,
`GetListOfDataMembers`, `GetListOfBases` and a write through `char*` arithmetic - `W0-233`'s four
walls, a programme and not a round. Twelve of its twenty FAILs carry no log at all because a fixture
did not pass, so most of them sit behind other walls. The six `__PRETTY_FUNCTION__` tests (`W0-231`)
were measured too and are **five** (three stop at the name, two are fixture dependants), and the
string must carry the *instantiated* argument, which a template body here has already erased - so
that is `W0-250` and a design decision, not an afternoon. `io/stdpair`'s quartet was the one target
whose shape was already written down: round 119 had priced it at **two lines each**.

**What landed** (`8d76246a`). Two rules, both read off ROOT's source and then measured on the oracle.

*A class the session only declares has a layout in memory from the first read*, so **every** file
layout is set against it - the current one included - and the emulated copies, which *are* that
spelling, against nothing. That is what makes the four `2_*_0_*` families print the file's mapped
layout as `type= 3 newtype=13` read with action `203`: `TStreamerInfo::Compile` adds `kConv` (200) to
a converted member's type (`io/io/src/TStreamerInfoActions.cxx:4317`), and `TStreamerInfoElement::ls`
prints `newtype=` when `fType != fNewType && fNewType != kUnset` (`core/meta/src/TStreamerElement.cxx:501`).
`_own_spelling` reads that layout **off the file's own emulated copy** rather than rebuilding it -
round 119 had measured that the checksum of those same elements is `0xca84d2aa` where the file says
`0xb5fb752`, so a rebuilt one would be a different layout.

*The warning is a different event, and the read order decides it.* `probe/slots.C` under the 6.40
oracle, `#include "cmspair.h"` first, both orders:

| read order | slot 1 | `fOnFileClassVersion` | warning |
|---|---|---|---|
| `ab02.root` then `aa02.root` | `0xb15d1241` | 1 — a file's copy | none |
| `aa02.root` then `ab02.root` | `0xb15d1241` | **0** — built | yes |

Same checksum either way; only the provenance differs, and `TClass::GetStreamerInfo` is where it
comes from (`core/meta/src/TClass.cxx:4716`: no entry at the class's version → `NewInfo` +
`Build`, and `Build` is what warns per member whose class it has no streamer for). The **slot** is
not the version a layout recorded - `_free_version` renumbers - so `_recorded_at` asks the *records*:
"did any file record this checksum at the version the class gives it?". A member spelled with a
number's streamer code (`< 61`, `kObject`) is no class and is passed over; the pair's enumeration is
`3` and its `edm::Ref` member `62`.

**First shape, and what it cost.** `_note_built` was called inside the "a layout is new" branch, so a
file whose records were **all met** never reached it, and the warning never printed (`…/run2`, still
30). The call moved to a second pass over the names a file describes - a record already met still
moves where the class's own layout sits.

**The guards, all five, whole, unchanged** (`…/guard*`): `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2,
`meta/evolution` 20/7/8/2 of 37, `io/datamodelevolution` 22/20/2 of 44, `io/newstl` 19/11/2 of 32 -
**83 / 39 / 13 / 6 of 141**, every figure the ledger's.

**The gate, seven commands, each read for its own exit status.** The maintainability gate named
`note_file_infos` at **ccn 11 / npath 648** after the change, so the per-layout step became
`_note_layout` (no limit touched). Part 1 **13 985 passed, 0 failed** (99.75 % by design); part 2 44
passed and `coverage report` **`TOTAL 126164 0 31706 0 100%`**; ruff clean; mypy 1 062 files;
maintainability 33 381 functions in 1 824 files; components 1 062 modules / 0 undeclared;
`mkdocs --strict` exit 0. **+3 unit tests** (104 in the reflection file), and the fixture now restores
`BUILT` beside the other module state - a state added to the module without the fixture is a state
that leaks between tests.

**The baseline** was raised from the run at this round's tip (`…/tip`, `--jobs 4`), whose only
baseline movement is those four rows; `rootsuite ratchet` on that same `--only` run reports 1503
`MISSING` rows, which is `W0-158` and not a regression.

**A doc, for the next reader**:
`docs/gotchas/a-declared-class-s-own-layout-is-the-emulated-spelling-and-only.md`, with the two
orders' table, ROOT's own print and `Compile` conditions, and the number-versus-class rule.

**Still open, re-scoped**: `W0-245`'s last four (`0_*_1_0`/`1_*_1_0`) - there the *current* layout is
the emulation and the load's re-emulation duplicates **the emulation**, so the array keeps four blocks
where this kit prints five; the fix is which block the duplicate is, not the spelling. `W0-248`'s eight
`3_*` unchanged. New: `W0-250` (five `cling/*` tests, `__PRETTY_FUNCTION__` with its instantiation,
two shapes measured and neither free).

## Round 120's close, continued: CI had been red since round 118 and is now green

`gh run list` after the push showed **failure** on 3.11, 3.12, 3.13 and 3.10 green, for the pushes of
rounds 118 and 119 alike - a red nobody had recorded, because the local gate runs Python 3.11 with
NumPy 1.26 while the runners have NumPy 2:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
src/xrdroot/pyroot/core/funcs.py:72
FAILED tests/test_pyroot_tkde.py::test_the_estimate_as_a_function_is_sampled_and_drawable
```

The cause is a deprecation this machine can show on demand: `float(np.array([2.5]))` is
`DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error
in future` (NumPy 1.25) and NumPy 2 made it that error. Code written for arrays - the normal way to
write a `TF1` here, since the same code takes thousands of rows at once - answers a one-row input
with a one-element array, so the model's point-at-a-time path was relying on removed behaviour.

`_scalar` (`6211a6ef`) extracts the single element, which is what the warning asks for, and refuses
several values for one point rather than truncating to the first; it is used where one point's answer
is taken as a number - the point-at-a-time model and both `EvalPar` branches. `TF1::Eval`'s
`float(code(np.array([x, y, z, t]), params))` was left alone: it is handed a four-vector, a
different question, and no test reaches it.

Gate after it: part 1 **13 986 passed, 0 failed**, part 2 44, **`TOTAL 126169 0 31708 0 100%`**,
ruff, mypy 1 063 files, maintainability 33 383 functions, components, `mkdocs --strict` exit 0. CI
on `6211a6ef`: **all eight jobs success** - `lint`, `docs`, `distribution`, and `test` on
ubuntu 3.10/3.11/3.12/3.13 and macos 3.12 (run 45494267359), the first green run since round 117.

## Round 120's close, continued again: one CI arc lost on a docs-only run, and the load

Two more things the round's last hour taught, both recorded rather than papered over.

**A docs-only push failed CI's coverage gate.** `8fb70a90` changes `mkdocs.yml` and one doc file,
and the macos-latest 3.12 job reported `Coverage failure: total of 99 is less than fail-under=100`
with `TOTAL 126169 0 31708 1 99%` - the single partial branch being `reflection.py 285->284`, the
back edge of `for name in FILE_INFOS:` in `reload_emulated`. `gh run rerun --failed` of that very run
was **green on all eight jobs**, so one `pytest-xdist` worker's arc went missing rather than a defect
appearing. The round's answer is
`test_a_load_re_emulates_every_class_a_file_described_not_only_the_first`, which asserts that *both*
classes of a two-class `FILE_INFOS` are reloaded - so the edge is taken whenever that test runs - and
the fixture now restores `RELOADED` beside the other module state. The residual risk no test can pin
is a worker whose data is dropped; it is `W0-251`, with the first step written down.

**The gate is only measurable under load, and the load was not this round's.** Another session's
`clang` builds and four stray `xrdroot run` processes took the load average to **78**, and at that
load part 1 reported `test_tmva_learn_pytorch.py::test_the_real_pytorch_trains_the_users_model_to_tell_signal_from_background`
failing and the two-command gate did not finish in 1800 s. Waiting five minutes for the load to fall
to **7.4** and re-running the same two commands gave **13 987 passed, 0 failed** and
`TOTAL 126169 0 31708 0 100%` with nothing changed in between - `W0-103`/`W0-109` again, and the
pytorch training test is a load artefact and not a regression.

**Where round 120 ends**: `main` = `52dc57b4`, pushed and verified; CI green on every one of the
eight jobs (run 45646132861). `io/stdpair` is 32 of 44, the roottest kind 1354 of 2173, all
non-tutorial 1584 of 2820, the reflection unit tests 105, and the suite's own gate 126 169 statements
with 0 missed and 0 partial branches.

## Round 121's close

**`io/stdpair` 32 → 36 PASS, none lost: a class's own layout has three answers, and a library load
*after* the files keeps every layout they recorded.**

**What the refs said, and the round's own first shape was wrong.** The eight DIFFs left in the bucket
split into two groups: the four `0_*_1_0`/`1_*_1_0` families, whose files are read **before** a
library load, and the eight `3_*`, where the **pair's own** dictionary is loaded. The first group's
refs list **four** blocks — the emulation as current, the file's mapped layout with `newtype=13` and
action `203`, then the emulation twice — where this kit printed the mapped layout as current and a
synthetic duplicate beside it. The rule: a load that re-emulates a class **nothing described**
(ROOT's `TClass::Update` through the collection proxy) gives it the **emulation's** spelling as its
own layout, at its own version, and leaves **every** layout its files recorded at the slot it was
met at.

The three answers, which is what the round had to keep apart:

| when the session learned the class | the class's own layout | a file's copy of that spelling |
|---|---|---|
| a **library**, loaded **before** the reads (`DESCRIBED` `"dictionary"`) | the emulation's | the same layout — dropped |
| an **`#include`** (`"declaration"`) | the file's **mapped** spelling | the same layout — dropped |
| **nothing**, then a library loaded **after** the reads (`_reemulated()`) | the emulation's | **kept**, at every slot |

**The first shape cost one run and the suite named it in 2 s.** Round 119's declaration rule and this
round's third case are one predicate each, and folding them into one branch — "search for the emulated
spelling whenever a load is involved" — made the four `2_*_0_*` families and the round-119 pinning
test fail (`…/r121a`, `4 failed, 101 passed`), because a declaration's own layout is the *mapped*
spelling. `_reemulated()` is that third answer and nothing else: `DESCRIBED is None and name in
RELOADED`.

**The guards, all five, whole, unchanged** (`…/guard*`): `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2,
`meta/evolution` 20/7/8/2 of 37, `io/datamodelevolution` 22/20/2 of 44, `io/newstl` 19/11/2 of 32 —
**83 / 39 / 13 / 6 of 141**, every figure the ledger's.

**The gate found two things in the round's own work.** The coverage gate named `_held_layouts`'
"duplicate the current layout" line as **dead** once the re-emulated case took the other path — the
rule is still real for a class the session *described*, so a test now pins it rather than the line
being deleted (`test_a_load_duplicates_the_current_layout_of_a_class_the_session_described`). And
ruff named four docstrings written at 101–104 columns, each rewrapped.

Gate: **`TOTAL 126172 0 31708 0 100%`** — part 1 **13 989 passed, 0 failed**, part 2 44 passed —
ruff, mypy 1 062 files, maintainability 33 387 functions in 1 824 files, components 1 062 /
0 undeclared, `mkdocs --strict` exit 0. **+2 unit tests** (107 in the reflection file).

**The baseline** was raised from the run at this round's tip (`…/tip`, `--jobs 4`, 36 PASS / 8 DIFF
of 44); its only movement is those four rows.

**A doc, for the next reader**:
`docs/gotchas/a-class-s-own-layout-has-three-answers-and-a-library-load-after.md`, with the table
above, the ROOT mechanism and the three predicates.

**What the last eight need, measured this round.** `pairEnumEvo_3_0_0_1` at the tip differs by
exactly **one block and one current layout**: ROOT's current is the file's **mapped** spelling with
an emulated block at `#1` beside it, so the order is `current, emulation, mapped` and the predicate
is the class's **own** dictionary — `_dictionaried(name)`, true in no other `libtype`, so the
`0_*`/`1_*`/`2_*`/`4_*` families cannot be disturbed by it. `W0-248` now carries that as its first
step. `W0-245` is closed; `W0-250` (five `cling/*` `__PRETTY_FUNCTION__` tests) and `W0-251` (the
`xdist` arc) are untouched.

## Round 122's close

**`io/stdpair` 36 → 44 PASS — every test of the bucket, and it took a class's *fourth* own layout.**

The eight DIFFs round 121 left were the families whose run loads the library whose LinkDef selects
the **pair itself** (`libCmsPair`). Round 121 had already measured the difference to one block and
one current layout and predicted the predicate (`_dictionaried(name)`); this round ran the change and
found the prediction right about the *state* and wrong about the *slot*.

**What ROOT does, and the slot is the part nobody had measured.** With a dictionary of the class's
own, ROOT answers for its layout with that compiled layout — the file's **mapped** spelling — and
files it at **slot 0** of `fStreamerInfo`, because an unversioned class has no class version to file
it under; every record a file holds then stands from slot 1 on. That is why `pairEnumEvo`'s walk
(`GetStreamerInfo(i)`, `i` from 1) prints `#1` (the emulation) and `#2` (the mapped record) beside a
current layout it never reaches, and why that current layout prints `newtype=13` / action `203`: it
is set against the emulation the class built beside it.

**The first shape measured one block number short, and the diff said so in one run.** With
`_own_is_mapped()` in place and the records kept, the current layout was right — `type= 3
newtype=13`, action `203` — and the emulated block printed as `#2` where the ref says `#1`: our array
put the current at slot 1. `…/run1` is that run (36 PASS / 8 DIFF, the same eight, their diffs
smaller); the slot-0 arm in `GetStreamerInfos` (`…/run2`) is the fix, and it is 44 / 0.

**Three predicates, each a state of its own.** `_own_dictionary()` (the class's own LinkDef selection
in `GENERATED`), `_own_is_mapped()` — that state, or a declaration, gives the mapped spelling, while
a *member's* dictionary or a re-emulation gives the emulation's — and `_keeps_records()`, which is
true of both states that keep every record a file made. **The class's own dictionary outranks a
re-emulation**: `3_*_1_*` reads its files first and loads afterwards, so the class is in `RELOADED`
as well, and ROOT still lists the dictionary's layout first. That is the second unit test, because it
is the kind of ordering a later refactor breaks silently.

**The maintainability gate named the round's own complexity and it was split.** `_pair_own_layout`
reached ccn 11 and `_held_layouts` ccn 11 / npath 272; the flags became `_own_is_mapped` and
`_keeps_records`, and the per-entry appends `_current_entries`. No limit was touched, and ruff named
three docstrings at 101–102 columns which were rewrapped.

**The guards, all five, whole, unchanged** (`…/guard*`): `io/tree` 18/0/0/2, `io/emulated` 4/1/1/2,
`meta/evolution` 20/7/8/2 of 37, `io/datamodelevolution` 22/20/2 of 44, `io/newstl` 19/11/2 of 32 —
**83 / 39 / 13 / 6 of 141**, every figure the ledger's.

**The gate, seven commands, each read for its own exit status.** Part 1 **13 996 passed, 0 failed**
(99.75 % by design); part 2 44 passed and `coverage report` **`TOTAL 126251 0 31716 0 100%`** - measured
at the **pushed** tip. Another session pushed twice while this round was pushing (`03ac9b82`
`std::integral_constant`, then `7e4d761b` `ProcInfo_t` and a docs commit), and each rebase replayed the
four commits unchanged; `io/stdpair` was re-run after the first (44 of 44, `…/tip2`) and the whole gate
re-read after the second, so the shas and the figure above are the same tree's. ruff
clean; mypy 1 062 files; maintainability 33 393 functions in 1 824 files; components 1 062 modules /
0 undeclared; `mkdocs --strict` exit 0. **+2 unit tests** (109 in the reflection file).

**The baseline** was raised from the run at this round's tip (`…/tip`, `--jobs 4`, 44 PASS / 0 of 44);
its only movement is the eight rows.

**A doc, for the next reader**:
`docs/gotchas/a-class-with-a-dictionary-of-its-own-is-laid-out-from-it-at.md`, with all four states
in one table, the `GENERATED` tell and the code path.

**Where this leaves the bucket, and what is left in the run's shape.** `io/stdpair` is finished: the
four `libtype`s, both `fixed` values, both file orders and both read-before-load orders are 44 of 44,
so the streamer-info model now answers every configuration `pairEnumEvo` can put it in. `W0-248` is
closed. What the loop has left on this axis is `W0-250` (five `cling/*` tests that need
`__PRETTY_FUNCTION__` with its instantiation) and `W0-251` (the `xdist` coverage arc), untouched.

**A measurement note for the next round.** The machine opened at a load average of **118** — another
project's `ffmpeg`, a geant4 benchmark and Python 3.14 builds — so the round wrote the code first and
measured the harness **statuses** under it (`W0-103`'s "reasons are quotable, timings are not"), then
waited for the load to fall to **11.8** before the coverage gate, which is the one command a loaded
machine makes meaningless.

### Round 122's CI, recorded rather than assumed

Two things happened to this round's CI, and neither was a defect of its own.

**Two runs were cancelled, not failed.** Another session pushed while this round was pushing (`03ac9b82`
then `7e4d761b` `ProcInfo_t`, then `9d82ce95`, then `a74f4bf0`), and each push cancels the previous
run's in-flight jobs through the workflow's concurrency group. The round's commits were rebased over
each in turn - every rebase replayed the four commits unchanged, `io/stdpair` was re-run at the first
rebased tip (44 of 44) and the two-command gate at the second and the last:

| tree | the round's gate |
|---|---|
| `03ac9b82` + round (13 991 + 44) | `TOTAL 126184 0 31708 0 100%` |
| `7e4d761b` + round (13 996 + 44) | `TOTAL 126251 0 31716 0 100%` |

**The one completed run failed on coverage, and it is `W0-251`'s known flake.** On `a74f4bf0` - which
contains all of the round's commits - the macos-latest 3.12 job reported
`Coverage failure: total of 99 is less than fail-under=100`, `TOTAL 126251 0 31716 1 99%`, the single
partial arc being `reflection.py 285->284`: the back edge of `for name in FILE_INFOS:` in
`reload_emulated`, which is nowhere near this round's change (lines 1400-1700) and which round 120
already recorded as an xdist worker losing one arc. `gh run rerun --failed` of that same run was
**green on every job**, so the trunk is certified green at `a74f4bf0` with the round inside it.
## Round 123's close

**What landed.** A function's `using` ends with its block, so the translator now keeps one alias
table per function instead of refusing a program whose functions disagree. C++ scopes an alias to
the block it is written in; this kit kept **one** table for a program's functions and raised
`the name contPair_t stands for two types in one program` on the second definition. ROOT's own
`roottest/root/io/cpp11Containers/commonUtils.h` writes `contPair_t` twice with different `pair`s -
once for the nested container's key/mapped types, once for the associative one's - and
`roottest/root/math/smatrix/testOperations.cxx` gives `MnMatrix` three meanings in three functions.
That refusal blocked all six `io/cpp11Containers` tests and both `smatrix` ones.

Two tables answer the two questions C++ asks separately, and they were already separate questions in
the code: a **type position** is answered while the declaration is read (`parse_types._alias`
replaces the alias with the `CType` and the node keeps it), so `Cursor.alias_scopes` is a stack laid
on the `variables` scopes - `push`/`pop` open and close a layer, `alias_of` asks the innermost first,
and `parse_stmt.alias` writes into the layer it is standing in. A **value position**, the call
`X(...)`, is answered at emit time, so `Program.function_aliases[id(func)]` keeps each function's own
and `Program.alias_in(self.scope.function, name)` is what `emit_calls._alias_call` asks; the names a
program's functions agree on stay in `local_aliases` for a call emitted with no function in view. The
refusal survives for the one case C++ also rejects: one body giving one name two types.

**The gain, and the loss that was not one.** `io/cpp11Containers` 1 PASS / 6 UNSUPPORTED ->
**3 PASS / 4 UNSUPPORTED**, the two `-libgen-build` rows being baseline promises back, and **no PASS
lost** across the whole kind run before and after (`…/roottest` 1360 / 445 / 127 / 230 / 10 / 1,
`…/roottest2` 1362 / 446 / 127 / 227 / 10 / 1). The only other move is `math/smatrix/Operations`
UNSUPPORTED -> FAIL: the lifted refusal surfacing at its next wall, which `W0-63` says is not
progress in itself and `W0-254` records. The four `io/cpp11Containers` runtime tests now stop at a
**template template parameter** (`W0-252`), which is a translator feature, not a test's size.

**The round's other half is a measurement the project did not have.** A whole-kind run of roottest
had not been made since round 103, and a whole run of all five kinds not since round 107. Doing it
found that `main`'s `tools/rootsuite/baseline.json` claimed PASS for eleven rows that no commit has
produced since round 111: the nine `io/evolution/versions/AddVersion-*`, whose committed refs say
`offset=na` where ROOT 6.40 prints `offset=  0`. That is **not** a regression from rounds 112-122 and
not a code bug — `TStreamerInfo::ls` and `TStreamerElement::ls` both print `offset=%3d`
(`io/io/src/TStreamerInfo.cxx`, `core/meta/src/TStreamerElement.cxx`), `%3d` cannot render `na`, and
the 6.40 oracle on this machine prints `offset=  0` (`…/probe/pf.C`). Round 112 laid a file layout's
members out at ROOT's own offsets, which is what moved this kit's output from the stale form to the
true one, and nothing wrote the change down. The bisect that establishes it: `46c791dc` (round 111)
30 PASS / 2 DIFF, `47e0b444` (round 112) 21 PASS / 11 DIFF, and the same at every commit since. So
the round **corrects** those nine rows to DIFF and raises fifteen more to PASS from these runs —
`ratchet` now reads **0 regressions** on each of the five kind runs where it read 9 on roottest.
`W0-253` asks the question the correction cannot answer: what a stale upstream ref should be *called*
in roottest's vocabulary.

All five kinds measured whole at this round's tip:
roottest **1362** / 446 / 127 / 227 / 10 / 1 of 2173, gtest **93** / 182 / 73 / 2 of 350,
pyunittest **88** / 38 / 2 / 1 of 129, `test/` **10** / 6 / 43 of 59, `other` **49** / 0 / 59 / 1 of
109 — **1602 of 2820 (56.8 %)** non-tutorial.

**The gate, seven commands, each read for its own exit status.** Part 1 **14 000 passed, 0 failed**
(99.75 % by design, `W0-112`); part 2 44 passed and `coverage report` **`TOTAL 126275 0 31728 0
100%`**. ruff clean (it named a `TYPE_CHECKING` import for `cursor.py`'s new annotations); mypy 1 062
files; maintainability 33 424 functions in 1 824 files; components 1 062 modules / 0 undeclared;
`mkdocs --strict` exit 0. **+3 unit tests**: each function gets its own type, an alias does not
outlive its function, one body still refuses a second type - and one more pins the
`_type_scope` guard the round's own change stopped reaching (a function-local `TypeParam` no longer
leaks into the generated GoogleTest class; the emitted GoogleTest program reads the same).

**A gotcha, for the next reader**:
`docs/gotchas/a-functions-using-ends-with-its-block-so-the-alias-table-is-per.md`, with the two
tables, the mermaid of the two questions, and what the four remaining `cpp11Containers` tests need.

**What this says about the next round.** The queue's shape has changed: the biggest measurable thing
left is no longer a code wall but a **measurement discipline** — a whole run of all five kinds is
what finds a stale baseline row, and it takes about fifty minutes of wall time across the five. The
code walls the round named: `W0-252` (template template parameters, 4 tests), `W0-253` (stale refs,
9 tests, needs a decision not a fix), `W0-254` (`SMatrix` dimensions that are named constants, 2),
`W0-233` (the `datamodelevolution` chain), `W0-250` (`__PRETTY_FUNCTION__`), `W0-251` (the `xdist`
arc).

## Round 124's close

**What landed.** Two facts C++ states plainly and this translator had lost.

`template <template <class T> class Cont>` gives `Cont` a **class template**, not a type, so
`Cont<double>` in the body is that template written with `double`. ROOT's own
`roottest/root/io/cpp11Containers/commonUtils.h` writes exactly that - `contPair_t` as a pair of the
nested container's key/mapped types in one function template and of the associative container's
eighty lines later - and passes `std::forward_list`, `std::list`, `std::unordered_set`,
`std::unordered_map`. The parser read the parameter but recorded it as one more *type* parameter, and
`substitute.substituted` applies a binding only to a `CType` with **no** template arguments, so
`Cont<double>` stayed whole and the emitter wrote `ROOT.Cont['double']` - the harness's
`AttributeError: ROOT has Cont`. `substitute.TemplateBinding(head)` now says what such a parameter's
argument is: a template's name. `substituted` replaces the head and keeps the arguments;
`parse_decl._class_template_parameter` records the name in `Function.class_templates` and
`ClassDecl.class_templates`; `instantiate.Templates.templated` wraps the binding before the body is
copied; and `substitute._named` gives an *expression* position (`C<int> make() { return C<int>(); }`)
the same rule as a type position.

The second fact: `template <unsigned Rows, unsigned Cols>` makes `Rows` a **value** inside its own
body, so `SMatrix<double, Rows, Cols>` names a matrix by two numbers the call gives and not by two
names a library is asked about. `_template` declares each value parameter as a variable while the
body is read; the maintainability gate named `_template` at npath 256, so that is
`_body_with_values`.

**The round gains no PASS and loses none, and says so.** The four `io/cpp11Containers` runtime tests
went UNSUPPORTED -> FAIL because they now *run*: they stop at the **writer**, which refuses a
`std::list<double>` handed to `WriteObject` (`a list<double> is not something this writer puts in a
ROOT file`) - a real missing capability, `W0-255`. `math/smatrix/Operations` went from
`testOperations.cxx:95` to `matrix_op.h:202`, `'SMatrix_5_5_0' object has no attribute 'Inverse'` -
ROOT's `SMatrix::Inverse(ifail)`, `W0-256`. `W0-63` asks that a refusal which merely moves further be
called that, and both walls are now library work with names instead of a translator refusal.
`W0-252` is closed by the round; `W0-254`'s fold landed and what remains of it is `Inverse`.

**The measurement.** A whole `roottest` run at `87552801` + the round's rules
(`…/ralph-round-202611261038/roottest`, `--jobs 5`, a machine rising through load 17 -> 45):
**1362 PASS / 450 FAIL / 127 DIFF / 222 UNSUPPORTED / 10 SKIP / 2 TIMEOUT** of 2173 - exactly the
round-123 figures with the four `cpp11Containers` moved from UNSUPPORTED to FAIL. `ratchet` reads
**0 regressions**, so `tools/rootsuite/baseline.json` needs no row touched: an UNSUPPORTED row
becoming FAIL loses no promise. The two TIMEOUTs are the load and were re-measured alone:
`math/smatrix/Kalman` reads UNSUPPORTED at `testKalman.cxx:403` in 219 s (it is near the 300 s cap -
recorded in `W0-255`'s row), and `tree/cloning/runtreeCloneTest2` times out at the base too.

**The gate, seven commands, each read for its own exit status.** Part 1 **14 029 passed, 0 failed**
(99.75 % by design, `W0-112`); part 2 44 passed and `coverage report` **`TOTAL 126519 0 31770 0
100%`**. ruff (after two docstring and test-string rewraps it named), mypy 1 062 files (it named an
unnamed value parameter's `None` name, which the parser names `_` everywhere else), maintainability
33 432 functions in 1 824 files, components 1 062 / 0 undeclared, `mkdocs --strict` exit 0. **+3 unit
tests** in `tests/test_cint_templates.py` (10 there now).

**A gotcha, for the next reader**:
`docs/gotchas/a-template-parameter-that-stands-for-a-template-keeps-the.md`, with the two rules, a
mermaid of each, and the two walls the six tests now sit at.

**What this says about the next round.** The translator's template story is now complete for both
shapes ROOT's own tests use; what the round exposed is that the *writer* is where the next several
tests stop - an STL container cannot be written as a file key at all (`W0-255`), and the reader
already has the layouts. `W0-256` is the small library addition `SMatrix::Inverse`/`InverseFast`.
The queue's shape after this round: four writer tests, two smatrix tests, `W0-233`'s
`datamodelevolution` chain, `W0-250`'s `__PRETTY_FUNCTION__`, `W0-251`'s `xdist` arc, `W0-253`'s
stale refs (a decision, not a fix).

## Round 125's close

**What landed.** An integral constant's name in a template argument is its number. C++ folds it
before anything is instantiated, so `SMatrix<double, N, N>` with `constexpr int N = 5` is
`SMatrix<double, 5, 5>`; three of the four spellings were read as *names* instead. `#define N 5`
already worked, because the preprocessor had replaced it. `constexpr int N = 5;` and
`const int N = 5;` emitted an undefined Python name. `enum { N = 5 };` emitted `'N'` — the parser
lists an enumerator's name as a *type*, so `_typed_arg` won and the emitter spelled a string — and
`roottest/root/math/smatrix/testInversion` died with `TypeError: "SMatrix<double,Name(...)`.

`Cursor.integer_constants` is the table. `parse_stmt.declarators` and `parse_decl._variables` both
register a `const`/`constexpr` declaration whose initialiser is an integer `Literal` — **both paths,
because a statement's declaration and a file-scope one are read by different functions**, which the
first shape got wrong. `parse_decl._enumerators` registers each enumerator through
`_enumerator_values`: a literal is its number, one without a value steps from the one before, and an
expression leaves itself and the ones after it unnamed. `parse_types._template_arg` asks the table
**first** — before `_typed_arg`, because that reading wins otherwise.

The second, unrelated piece: `TClonesArray::New(index)` is ROOT's own spelling of
`ConstructedAt(index)` — `TClonesArray::New` *is* `ConstructedAt` unless `fgAlwaysReorder` was asked
for, which no ROOT test does.

**The crash the round shipped and the whole run caught.** The first shape read
`value = None if written is not None else value + 1`, so an enumerator written as an *expression*
(`kB = kSomeOther`) left `None`, the next implicit one did `None + 1`, and
`meta/autoloading/headerParsingOnDemand/scopedEnums-libgen-build` went **PASS -> UNSUPPORTED** —
a lost PASS that only a whole run of the kind shows, exactly what the run is for. `_enumerator_values`
keeps the `None` state explicit and the area was re-measured to PASS before the round closed.

**The measurement.** A whole `roottest` run at `440e03d1` + the round's changes (`…/roottest2`,
`--jobs 5`, load average 13-39): **1363 PASS / 450 FAIL / 128 DIFF / 221 UNSUPPORTED / 10 SKIP /
1 TIMEOUT** of 2173. Against round 124's run the moves are: `io/references/execWriteRefArrayCompress`
**FAIL -> PASS** (the round's one gain, from `TClonesArray::New`), `treeformula/references/clones`
FAIL -> DIFF (a smaller diff), `math/smatrix/Kalman` TIMEOUT -> UNSUPPORTED (the load), and
`python/cling/api` and `python/tpython/execscript` UNSUPPORTED -> FAIL — **another session's**
TPython/`std::any`/`std::filesystem` work between the two runs lifting their refusals, not this
round's. `ratchet` names no lost PASS. `math/smatrix/Inversion` moved off the `Name(...)` TypeError
and onto `SMatrix<double,5,5,MatRepSym<double,5>>` refused by the pythonization: the constants are
right and the wall is the library's, `W0-256`.

**The gate, seven commands, each read for its own exit status.** Part 1 **14 048 passed, 0 failed**
(99.75 % by design, `W0-112`); part 2 44 passed and `coverage report` **`TOTAL 126753 0 31832 0 100%`**. ruff
(after one docstring rewrap), mypy 1 062 files, maintainability 33 572 functions in 1 831 files,
components 1 062 / 0 undeclared, `mkdocs --strict` exit 0. **+3 unit tests**: the constexpr, the
const, a plain variable (not folded), both enumerator shapes, and `New` answering `ConstructedAt`.

**A gotcha, for the next reader**:
`docs/gotchas/a-template-argument-that-names-an-integral-constant-is-its.md` — the four spellings in
one table, the two declaration paths, and the `None` that must not reach the next enumerator.

**What the run cost and what the queue says.** This round measured six candidate targets before
committing to one, and every big one is a chain: `TDataMember.GetDataType` blocks **10** tests but
needs byte offsets *and* a write through `(char*)obj + offset` (`byte_offset` refuses any non-zero
offset today); `TXMLEngine`'s three tests need a whole XML DOM; `io/cpp11Containers`' four need an
STL container written as a file key (`W0-255`); `io/references`' last two need `TTree::Write()` of a
tree read from a file (`W0-258`); the `datamodelevolution` `00` family differs by two streamed values
and `TObjArray::ls`'s `size=` — the array's *room*, which ROOT prints and this kit does not
(`W0-259`). The flat bucket table is itself the finding: the largest single cause in the kind now
blocks **1-3** tests except the enumerated chains, so the next rounds should take one chain each and
say which test it buys.

## Round 126's close

**What landed.** Two ROOT classes were missing from the namespace in ways only a *file* reveals.

`TArray` is ROOT's abstract base of the fixed arrays, and a file's streamer list names it.
`pyroot/core/reflection.py::_undictionaried` decides whether to print ROOT's
`Warning in <TClass::Init>: no dictionary for class TArray is available` by asking
`TClass(name)._class()` - the namespace's own answer - and the array family lived under a
**private** `_TArray`, so the namespace answered no. Exporting the base as `TArray` (what ROOT calls
it, and what `TArrayD` and its kin derive from) is the whole fix: `io/evolution/ClonesArrayEvo`,
`io/fakeClass/forwardDeclAndFake` and `treeformula/formulaio/old-formula` each differed from their ref
by that one line and now pass. It is round 119's rule (a declared class) and round 122's (a class with
its own dictionary) once more: **the namespace's answer decides, not a list of names**.

`roottest/root/tree/cache/runmultiTree.C` writes ROOT's own idiom
`tree->Branch(lname, &var[i], lname, 512)` with a `TString`, which the C++ layer converts to
`const char*`; this kit's writer takes a `str` and refused. `pyroot/trees/tree.py::_text` answers a
`TString`'s characters at `TTree::Branch` and leaves anything else for the layer that refuses it to
name - a convenience belongs in the wrapper, not in the engine.

**The measurement.** A whole `roottest` run at `72f2f66c` + the round's fixes (`…/roottest`,
`--jobs 5`): **1366 PASS / 450 FAIL / 125 DIFF / 220 UNSUPPORTED / 10 SKIP / 2 TIMEOUT** of 2173,
against round 125's 1363 / 450 / 128 / 221 / 10 / 1. The moves: three DIFF -> PASS
(`io/evolution/ClonesArrayEvo`, `io/fakeClass/forwardDeclAndFake`, `treeformula/formulaio/old-formula`),
`tree/cache/multiTree` FAIL -> UNSUPPORTED (the `TString` fix carried it to `TTreeCache`),
`meta/runInterpClassDef-auto` UNSUPPORTED -> FAIL (a wall moved, not a lost PASS), and
`smatrix/Kalman` UNSUPPORTED -> TIMEOUT, which is the load: it reads UNSUPPORTED alone at 219 s and
the whole run was five jobs on a machine at 12-39. `ratchet` names no lost PASS.

**The gate, seven commands, each read for its own exit status.** Part 1 **14 050 passed, 0 failed**
(99.75 % by design, `W0-112`); part 2 44 passed and `coverage report` **`TOTAL 126757 0 31832 0 100%`**. ruff, mypy,
maintainability, components, `mkdocs --strict` - each exit 0. **+2 unit tests**: the fixed arrays
share `TArray` and `TClass("TArray")` answers for it, and a `TString` name at `Branch` is its
characters.

**A gotcha, for the next reader**:
`docs/gotchas/a-file-names-a-class-the-namespace-must-answer-for-tarray-and-md.md`.

**The pricing that chose this round, and what it left.** The flat tail is the finding: the largest
single own-refusal now blocks **6** tests (`io/vararyobj`'s `B*[fM]` of varying-length arrays) and
every other is 2-5 with a chain behind it. Four walls were priced and recorded instead of chased:
`TTreeCache` (3 tests, whose refs pin cache sizes, buffer sizes and pointer identity),
`Snapshot` of a class column (5), `TBufferFile` through a custom `Streamer(TBuffer&)` (3), and
`TGeoPatternFinder` (1). A round that prices four walls and lands three passes is worth more than one
that starts four chains.
