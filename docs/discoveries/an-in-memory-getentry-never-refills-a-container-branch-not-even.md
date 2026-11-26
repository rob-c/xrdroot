---
title: "An in-memory GetEntry never refills a container branch - not even std::vector"
kind: "discovery"
summary: "In-memory GetEntry never refills a container address — Vector.put is not called either — so a container read-back is a file round-trip and any test of one must write a file and read it back."
tags: [trees, branches, containers, reading, file-roundtrip]
created: "2026-11-26T10:51:44.112Z"
updated: "2026-11-26T10:51:44.112Z"
---
Reading an **in-memory** tree does not refill a container address at all — and that is true of `std::vector` too, not only of a new `std::set`:

```text
tree = TTree("t","t"); v = std.set/vector("int")(); tree.Branch("v", v)
fill [], [1], [1,2]            ->  slot.get() sees [], [1], [1 2]  (the write side is right)
GetEntry(0|1|2); list(v)       ->  [1, 2]  [1, 2]  [1, 2]         (nothing refilled it)
```

Tracing `Vector.get`/`Vector.put` over that run shows three `get` calls (one per `Fill`) and **no `put`** — so `GetEntry` on a tree that was never written to a file has no per-entry value to hand back; the object simply keeps whatever the last `Fill` left in it. A container read-back is therefore a *file* round-trip (`Write`, then read the file), and the reader's column path is what puts values into the address.

## Where that leaves a container branch

- For `std::set<int>` this makes the feature: an address class beside `Vector`, a leaf type that names the set, **and** writer/reader support for a `std::set` column — then a test that writes a file and reads it back.
- A test that only fills in memory and calls `GetEntry` (as ROOT's own `stl_set.py::test_stl_set_tree` does, because ROOT fills in place) cannot pass here whatever the address class, so the regression has to be written as a file round-trip.
- Before adding a container, verify the same scenario for `std::vector` first: if the vector does not refill in memory, a set will not either, and the failure is not the new code's.
