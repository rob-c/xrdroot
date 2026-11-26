---
title: "An enumerator reaches Python as a signed 64-bit int, even from an unsigned enum"
kind: "discovery"
summary: "PyROOT exposes an enumerator as a signed 64-bit int, so `enum E : unsigned long long { BIG = 1ULL << 63 }` reads -9223372036854775808; a kit folding it unsigned fails MetaClingTests' test_GH_20925."
tags: [pyroot, enums, int64, metacling, translator]
created: "2026-11-26T00:40:59.264Z"
updated: "2026-11-26T00:40:59.264Z"
---
PyROOT hands an enumerator to Python as a **signed 64-bit int**, whatever the enum's underlying type, so an unsigned value with the top bit set comes back negative:

```text
C++                                   ROOT 6.40 / PyROOT        our kit (wrong)
enum E : unsigned long long {
   BIG = ((unsigned long long)1)<<63  -9223372036854775808    +9223372036854775808
};
```

`core/metacling/test/MetaClingTests.py::test_GH_20925` asserts exactly that reading, under the docstring "Do not assert when casting huge enum uint64_t into int64_t".

## Where that lands here

The enumerators are folded and emitted by `enum_values` in `src/xrdroot/pycxx/emit_classes.py` (`_constant(value)` for the literal, `self.value(value)` for an expression, each later one counted on with `+1`). The fix belongs at that seam: when the enum's underlying type is a 64-bit *unsigned* integer and a folded value is ≥ 2^63, expose `value - 2**64` — the signed reading PyROOT gives. Anything below 2^63 is unchanged, so ordinary enums are untouched.

A regression test belongs beside the existing enum tests (`tests/test_cint_*.py`): one enumerator that wraps to `-2**63`, and one ordinary value next to it that keeps its own.

## Why it matters beyond the one test

The same seam decides how every `enum : <type>` reaches Python, so a wrong reading there is wrong for any class exposing a flags enum built from a shifted bit — a pattern ROOT's own headers use.
