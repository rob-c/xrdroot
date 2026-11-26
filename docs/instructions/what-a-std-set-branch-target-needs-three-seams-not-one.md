---
title: "What a std::set branch target needs: three seams, not one"
kind: "instruction"
summary: "A container becomes a branch target through three seams — address_of, the leaf spelling _declared derives, and the count-then-elements stream — so std::set needs all three, not just an address class."
tags: [trees, branches, std-set, addresses, feature-pricing]
created: "2026-11-26T10:41:15.179Z"
updated: "2026-11-26T10:41:15.179Z"
---
Adding a container as a tree branch target takes three seams, not one. `tree.Branch(name, obj)` asks `address_of` what it was handed (`src/xrdroot/pyroot/trees/addresses.py`), derives a leaf spec from that address (`_declared`, called from `_leaf_branch` in `pyroot/trees/tree.py`), and then streams the value each `Fill`. A `std::vector` is wired through all three — `_is_vector(given)` → `Vector(given)` (`addresses.py:233`), whose `get`/`put` read `vector.data()` and `vector.assign(value)`.

```text
Branch(name, obj)
  └─ address_of(obj)      addresses.py: _is_vector → Vector, string → Text, ndarray → View,
                          else _refused(obj, what) with the list of things a tree can fill
  └─ _declared(address)   tree.py: the leaf spec - the branch's C++ type and its leaves
  └─ writer/reader        a collection per entry: its count, then its elements
```

## What `std::set<int>` therefore needs

1. `is_set(obj)` beside `is_vector` (`src/xrdroot/pyroot/stl.py`) and a `Set(Address)` in `addresses.py` whose `get` iterates the set and whose `put` clears then inserts — the set's own order holds, so no ordering decision is needed.
2. A leaf spelling for `std::set<T>` where `_declared` switches on the address, matching what a writer names the branch (`std::set<int>`), so a reader can find it again.
3. Writer and reader support for a set-valued branch: ROOT streams a `std::set` as a collection, so the count-then-elements path exists — the question is whether the branch's leaf type names `set<T>` rather than `vector<T>`.
4. A regression in `tests/` modelled on ROOT's `bindings/pyroot/pythonizations/test/stl_set.py::test_stl_set_tree`, which fills three entries (empty, one, two) and reads each back as a Python `set`.

`pyroot/trees/addresses.py`'s `_refused` message enumerates what a tree can fill; that list is the guard rail — a container added to it without the leaf spelling and the streaming behind it fails later, at `Write`, not at `Branch`.
