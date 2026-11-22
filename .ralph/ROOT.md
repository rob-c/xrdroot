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
be lost. Start it with `/ralph start ROOT`, or `ralph_start({ name: "ROOT", taskContent: <this
file>, maxIterations: 50 })`.

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
R=/tmp/ralph-round-$(date +%Y%m%d%H%M)      # one directory per round, outside the repository
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

**(d) Tear the round down — no worktree, no round branch, and the round's `main` branch with it.**
This is the last thing a round does, and the order matters: the push comes *first*, because a
worktree removed before it — with `main` moved under the round — would take the round's only
remaining reference to its commits into a directory that no longer exists.

```bash
cd $R/main && git status --short         # must be empty: nothing is left uncommitted
git push origin $ROUND-main:main || { git fetch origin main && git rebase origin/main \
  && git push origin $ROUND-main:main; }  # another session pushed: rebase the round onto it
git ls-remote origin main | cut -f1      # the tip, which must be this worktree's HEAD

cd /Users/rcurrie/src/xrdroot
git worktree list --porcelain | awk '/^worktree /{print $2}' | grep "^$R/"   # what will go
for w in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  git worktree remove --force $R/w$w && git branch -D r-$w
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
11. **A new worktree every round, merged to `main` and torn down in that same round** (steps 0
    and 6d), on disjoint files, `PYTHONPATH=$PWD/src`. The round's integration worktree is the one
    that pushes `main`; nothing else is pushed without being asked. Never edit another worker's
    worktree, and never reuse one round's worktree in the next.

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
  to `feat/root-testsuite` and records that it did.
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

## Checklist (per round)

- [x] `git status` clean; the gates green at the round's base commit
- [x] the queue read and the failures ranked by cause, from the last `results.json`
- [ ] sixteen scopes chosen, disjoint files, each with a measured "before" - **not this round**: the load
      average opened at 76 (`W0-103`), so there was no fan-out
- [ ] sixteen worktrees created from `HEAD`, one branch each - **not this round**
- [ ] sixteen briefs written and launched, staggered, detached - **not this round**
- [ ] workers watched; a worker that blocked reports, a worker that left a patch is honoured - n/a
- [x] every commit that applies cleanly cherry-picked `-x`, one fix group each, gates green
- [x] **every branch that loses a PASS rejected**, with the refusal to relax recorded - none lost
- [x] the areas the round touched were measured before and after (`/tmp/r42/ac`, `/tmp/r42/json`)
- [x] `TODO.md` §0 and the ledger rows updated in the same commits (§7's cumulative table has no
      column for this round: no whole-suite run was possible, as in rounds 37-41)
- [x] **every extra thing the round found added to §4's queue or §5's housekeeping**, with file,
      message and first step (see step 6c) — a round that discovers work and does not write it
      down has wasted it
- [x] the round's summary written into the loop's task file (`.ralph/<name>.md`)
- [x] gates re-run at the round's final commit
- [x] commits made; nothing pushed unless asked
- [x] the round directory kept (its `results.json` files are the evidence; `/tmp/r42/EVIDENCE.md`)

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

## Notes

- Write the ledger as you go: it is the only project state that is cheap to maintain and
  impossible to reconstruct.
- Prefer a *measured* claim to a plausible one. Every number in `TODO.md` came from a
  `results.json` named in this file or in its notes.
- When a round yields little, the reason is nearly always that the chosen groups were too big to
  finish in forty minutes: split them further next round rather than widening a worker's scope.
- When a worker reports "the fix is one line, in a file outside my scope", the next round's
  scopes are wrong, not the worker.
