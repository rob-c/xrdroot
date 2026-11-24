---
title: "Where to read xrdroot's current pass counts (and which number is which)"
kind: "instruction"
summary: "Quote pass counts from origin/main: tools/rootsuite/baseline.json (ROOT's suite), tools/tutorials/baseline.json, and the last TOTAL line in .ralph/ROOT.md for xrdroot's own pytest gate; TODO.md's table lags."
tags: [tests, baseline, status, gate, reporting]
created: "2026-11-24T02:32:23.120Z"
updated: "2026-11-24T02:32:23.120Z"
---
Three different numbers answer "how many tests pass", and each has one committed source of truth. Read them from `origin/main` — a checkout can be dozens of rounds behind (152 commits was typical), and `TODO.md`'s §0 table lags, because it is the assessment plus the ledger's in-area gains rather than a fresh run.

```text
tools/rootsuite/baseline.json   ROOT's own ctest suite        -> 1549 / 2820 PASS
tools/tutorials/baseline.json   ROOT's tutorials              ->  564 / 1189 PASS
.ralph/ROOT.md, last "TOTAL"   xrdroot's own pytest gate     -> 13 968 passed, 100 %
TODO.md §0 table               provenance + per-kind history -> lags; not a measurement
```

```bash
cd /Users/rcurrie/src/xrdroot && git fetch origin main -q
git show origin/main:tools/rootsuite/baseline.json | jq -r 'to_entries | group_by(.value) | .[] | "\(.[0].value) \(length)"'
git show origin/main:.ralph/ROOT.md | grep "TOTAL .*100%" | tail -1
```

At `origin/main` = `1a4d97b9` (round 96, 2026-11-24): **rootsuite 1549/2820** (roottest 1317/2173, gtest 92/350, pyunittest 81/129, `test/` 10/59, other 49/109), **tutorials 564/1189** (DIFF 87, SKIP 326, FAIL 63, UNSUPPORTED 134, ORACLE-FAIL 15), and `TOTAL 125738 0 31556 0 100%` — **13 968 unit tests**, 13 924 + 44 by the two-command decomposition.

## Which number is which

- `tools/rootsuite/baseline.json` is written only by `python -m tools.rootsuite baseline RUN/results.json`, so its PASS count is the ratchet — the only figure a "did we lose anything" question can use.
- `tools/tutorials/baseline.json` is the same for tutorials; it has stood at 564 for many rounds because tutorials are expensive to re-run whole.
- The gate line in `.ralph/ROOT.md` is xrdroot's own pytest suite — quote it for "the project's tests", not the ROOT-suite number.
