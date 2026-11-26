---
title: "A half-wired container branch reads back the last entry for every row"
kind: "gotcha"
summary: "Adding only an address class for a std::set branch writes vector<int> and every GetEntry returns the last entry; put() is never called because the branch binds a leaf address, so the reader path is part of the feature."
tags: [trees, branches, std-set, reading, silent-wrongness]
created: "2026-11-26T10:44:20.379Z"
updated: "2026-11-26T10:44:20.379Z"
---
A `std::set` handed to `tree.Branch(name, obj)` with only an address class added **writes `vector<int>` and reads the last entry back for every row** — silently wrong data rather than a refusal:

```text
Branch("set", ROOT.std.set("int")())   ->  leaf type "vector<int>"
Fill with {}, {1}, {1,2}               ->  GetEntry(0|1|2) all give {1, 2}
```

The cause is that `put` is never reached: `GetEntry` fills through an address the *branch* binds (a leaf's `View`, via `SetBranchAddress`/`_columns_bound`), not the object the caller handed over, so a new `Address` subclass alone changes nothing about what a read-back does. Tracing `addresses.Collection.put` showed zero calls while the reads still returned values — the values came from the store's own last-filled slot.

## What to check before believing a container branch works

- The branch's leaf type must name the container (`std::set<int>`), not the collection shape it was inferred as (`vector<int>`).
- `GetEntry(i)` must *reach* the address: instrument the address's `put` and require one call per entry, then assert the first entry's value, not just the last.
- Write three entries that differ (empty, one element, two), as `bindings/pyroot/pythonizations/test/stl_set.py::test_stl_set_tree` does — a single-element test cannot tell a working read-back from a stale one.

The attempt was reverted rather than committed: a container that reads back the last row for every entry would poison any test that then compares physics.
