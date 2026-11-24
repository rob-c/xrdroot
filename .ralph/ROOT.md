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
