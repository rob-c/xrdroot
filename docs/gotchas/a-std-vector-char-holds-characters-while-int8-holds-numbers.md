---
title: "A std::vector<char> holds characters while int8 holds numbers"
kind: "gotcha"
summary: "A `char` vector reads characters and `data()` gives text, `int8`/`uint8` keep numbers; key the class by spelling, keep `chars` off number classes, and never let `data()`/`_view()` call each other."
tags: [cppstd, vector, char, pyroot, rdf, coverage]
created: "2026-11-26T12:32:27.975Z"
updated: "2026-11-26T12:32:27.975Z"
---
PyROOT hands a `char` element out as a **one-character string** and `data()` as the **text** of the whole vector (C++'s `data()` is a `char*`). Ours gave numbers, so ROOT's own `pyroot_pyz_stl_vector` failed at the first assertion of its iadd case and its `vec_char_data` case could not even build the vector from `['a', 'b', 'c']`. Both pass at `e2c7ee27`.

```mermaid
flowchart TB
  S["std.vector['char']"] --> K{"one-byte element?"}
  K -->|yes| SP{"spelling in<br/>char / signed char / unsigned char?"}
  SP -->|yes| C["class with chars = True"]
  SP -->|no (int8, uint8)| N["class with no chars attribute"]
  K -->|no| N
  C --> R["v[i] -> chr(byte)<br/>list(v) -> chars<br/>v.data() -> 'ab'<br/>str(v) -> { a, b }"]
  N --> R2["v[i] -> number<br/>v.data() -> numpy"]
  W["v[i] = 'z', push_back(c),<br/>resize(n, c), assign('hey')"] --> ST["_stored: ord(c)"]
  ST --> C
```

## Where the decision lives

`_Template._general` in `src/xrdroot/cppstd/containers.py` keys the class on `_template_key(given, kinds)`: for a one-byte element the key becomes `"char:<cpp name>"` or `"number:<cpp name>"`, so `int8_t` and `char` — one C++ type — are two classes here. `signed char` shares the `char` class; `unsigned char` has its own because its storage is a different dtype. Only the character key sets `made.chars = True`.

`_Vector` then reads and writes through `_shown`/`_stored` (`chr(byte & 0xFF)` and `ord(char)`, with a fill of `0` for `''`), `data()` returns the text, and `__str__` prints the characters.

## Three traps

- **Do not put `chars` on a number's class.** `pyroot/rdf/vectorrows.py::_vector_fields` compares `set(vars(kind))` against an exact set (`_VECTOR_FIELDS`); an extra attribute there makes `_storage` return `None` and every RDF native row of that dtype silently falls back (77 test failures).
- **`data()` must not call `_view()`, and `_view()` must ask a replaced `data()`.** A test (or a pythonization) that replaces `data` is writing down a contract: `_view()` compares against `_NUMERIC_READERS["data"]` — the identity captured at class creation — not against `_Vector.data`, because a test that monkeypatches the *base* makes those the same object and the check passes while the override is bypassed. The two-way version recurses.
- **`vector['int8']` and `vector['uint8']` are object vectors on this kit** (list-backed, `vector<int8>`), not numeric ones: only `char`/`signed char`/`unsigned char` reach the numeric path for a one-byte element. A test that builds one and calls `.data().tolist()` fails with `'list' object has no attribute 'tolist'` — use `list(...)`.

`src/xrdroot/pyroot/core/lfsr.py`'s register is a `vector<uint8>`: it counts bits, it does not hold characters, so it must not use the `unsigned char` spelling.

## How to check

```bash
python -m tools.rootsuite run --root ~/src/.xrdroot-scratch/rootsrc -j 3 \
  --xrdroot-src <worktree>/src --out ~/.cache/xrdroot-runs/goal-char \
  --only 'pyunittest/bindings/pyroot/pythonizations/pyroot_pyz_stl*'
```

Both `pyroot_pyz_stl_vector` (six cases) and `pyroot_pyz_stl_set` PASS. Repository side: `tests/test_cppstd_containers.py::test_a_char_vector_holds_characters_where_a_number_vector_of_that_width_does_not` and `::test_an_unsigned_char_is_its_own_character_class_and_a_float_is_no_character` (split in two, since one function with all of it broke the CCN ≤ 10 gate).
