---
title: "The baseline covers all five kinds: 2820 rows, 1619 PASS"
kind: "architecture"
summary: "tools/rootsuite/baseline.json holds 2820 rows for all five kinds — 1619 PASS, 667 FAIL, 127 DIFF, 392 UNSUPPORTED, 11 SKIP, 4 TIMEOUT; a pyunittest run found eight stale FAIL rows to bank."
tags: [baseline, rootsuite, measurement, roottest, pyunittest, gtest]
created: "2026-11-27T00:07:31.442Z"
updated: "2026-11-27T00:07:31.442Z"
---
`tools/rootsuite/baseline.json` is the project's own scoreboard and it covers **all five kinds**, not one: 2820 rows — gtest 350, roottest 2173, pyunittest 129, other 109, test 59 — reading **1619 PASS, 667 FAIL, 127 DIFF, 392 UNSUPPORTED, 4 TIMEOUT, 11 SKIP**. Read it before claiming anything about "all the tests": it is the honest denominator, and a kind-specific run can be compared to it with `tools.rootsuite ratchet`.

```bash
python - <<'PY'
import json, collections, pathlib
b = json.loads(pathlib.Path("tools/rootsuite/baseline.json").read_text())
print(len(b), dict(collections.Counter(b.values())))
print(dict(collections.Counter(k.split("/")[0] for k in b)))
PY
# 2820 {'PASS': 1619, 'FAIL': 667, 'DIFF': 127, 'UNSUPPORTED': 392, 'TIMEOUT': 4, 'SKIP': 11}
# {'gtest': 350, 'other': 109, 'pyunittest': 129, 'roottest': 2173, 'test': 59}
```

## Why it matters

A `--kind pyunittest` run I made was compared against this file and **eight rows moved FAIL -> PASS** — `pyroot_array_numpy_views`, `pyroot_pyz_stl_set`, `pyroot_pyz_stl_vector`, `pyroot_root_no_implicit_cppyy`, `tpython/test_tpython`, `core/metacling/MetaClingTests`, `tmva/rbdt_xgboost`, `dataframe_snapshot_py` — work that had landed in earlier sessions and never been banked. Banking them is +8 PASS, and it costs one run and one `ratchet`, which read `0 regressions`.

```mermaid
flowchart LR
    A["baseline.json: 2820 rows, all five kinds"] --> B["1619 PASS (the number to defend)"]
    A --> C["667 FAIL + 127 DIFF"]
    A --> D["392 UNSUPPORTED + 11 SKIP + 4 TIMEOUT"]
    E["a per-kind run"] --> F["ratchet: 0 regressions?"]
    F -->|stale row found| G["bank the better status, in that commit"]
```

## How to use it

- Quote the kind you measured, never "the suite": 102/350 gtest and 96/129 pyunittest are both true at once.
- Bank a gain from the run that showed it, in the same commit as whatever caused it.
- Never rewrite the file wholesale — replace only the rows that moved (a round-trip through `json.dumps(..., indent=2, sort_keys=True)` reformats all 2820 lines and buries the real change).
