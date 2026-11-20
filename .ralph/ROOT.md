# xrdroot: continue the parity work with sixteen workers

This loop advances `/Users/rcurrie/src/xrdroot` — a pure-Python, NumPy-backed re-implementation
of CERN ROOT — one **round** at a time. A round is: read the queue in `TODO.md`, pick sixteen
disjoint scopes, fan sixteen workers out over their own git worktrees, integrate what passes the
gates, measure with ROOT's own test suite, and then **close the round in `TODO.md`** — the numbers,
a ledger row per fix and per attempt, and a queue entry for every piece of work the round
discovered and did not do.

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

## The round, in six steps

### 1. Read the queue and rank the work

```bash
cd /Users/rcurrie/src/xrdroot && export PYTHONPATH=$PWD/src
git log --oneline -6; git status --short          # must be clean
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

```bash
R=/tmp/ralph-round-$(date +%Y%m%d%H%M)        # outside the repository, per workspace rules
mkdir -p $R && cd /Users/rcurrie/src/xrdroot
for w in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  git worktree add -q $R/w$w -b r-$w HEAD
done
```

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
BASE=$(git -C /Users/rcurrie/src/xrdroot rev-parse HEAD)   # the round's base
for w in $(seq -w 1 16); do
  f=$(ls -t ~/.pi/agent/sessions/*/2026-*.jsonl 2>/dev/null \
      | xargs grep -l "Task r-$w:" 2>/dev/null | head -1)
  printf "w%s commits=%s dirty=%s events=%s\n" "$w" \
    "$(git -C $R/w$w rev-list --count HEAD ^$BASE)" \
    "$(git -C $R/w$w status --porcelain | wc -l)" "$(wc -l < "$f" 2>/dev/null)"
done
```

A worker's **working tree** can be ahead of its commit — two rounds lost the final state of the
best implementation that way. When a worker committed early and kept working, copy its *files*
(`cp $R/wNN/<path> <repo>/<path>`), not just its commits.

### 4. Integrate, and let no PASS be lost

For each worker with a commit, in order of expected value:

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

Then close the round with step 6. Push only if the operator asked; the gates are run before the
commit, never after.

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
cd /Users/rcurrie/src/xrdroot && PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --cov=xrdroot
git -c commit.gpgsign=false commit -am "docs: the ledger after round <N>"
```

and put the round's own summary — the sixteen scopes, what landed, what was rejected, what is
new — in the loop's task file (`.ralph/<name>.md`), which is where the next iteration reads it.

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
11. **Work in worktrees on disjoint files**, `PYTHONPATH=$PWD/src`. Never edit another worker's
    worktree, never push without being asked.

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

## Checklist (per round)

- [ ] `git status` clean; the gates green at the round's base commit
- [ ] the queue read and the failures ranked by cause, from the last `results.json`
- [ ] sixteen scopes chosen, disjoint files, each with a measured "before"
- [ ] sixteen worktrees created from `HEAD`, one branch each
- [ ] sixteen briefs written and launched, staggered, detached
- [ ] workers watched; a worker that blocked reports, a worker that left a patch is honoured
- [ ] every commit that applies cleanly cherry-picked `-x`, one fix group each, gates green
- [ ] **every branch that loses a PASS rejected**, with the refusal to relax recorded
- [ ] both harness runs made; every TIMEOUT rerun alone before being called a loss
- [ ] `TODO.md` §0, §7's cumulative table and the ledger rows updated in the same commits
- [ ] **every extra thing the round found added to §4's queue or §5's housekeeping**, with file,
      message and first step (see step 6c) — a round that discovers work and does not write it
      down has wasted it
- [ ] the round's summary written into the loop's task file (`.ralph/<name>.md`)
- [ ] gates re-run at the round's final commit
- [ ] commits made; nothing pushed unless asked
- [ ] the round directory kept (its `results.json` files are the evidence)

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

- Exact monitor-rerunnable command:
  `cd /Users/rcurrie/src/xrdroot && PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --cov=xrdroot`
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

## Notes

- Write the ledger as you go: it is the only project state that is cheap to maintain and
  impossible to reconstruct.
- Prefer a *measured* claim to a plausible one. Every number in `TODO.md` came from a
  `results.json` named in this file or in its notes.
- When a round yields little, the reason is nearly always that the chosen groups were too big to
  finish in forty minutes: split them further next round rather than widening a worker's scope.
- When a worker reports "the fix is one line, in a file outside my scope", the next round's
  scopes are wrong, not the worker.
