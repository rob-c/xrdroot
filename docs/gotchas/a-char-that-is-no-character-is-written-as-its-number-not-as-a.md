---
title: "A char that is no character is written as its number, not as a chr()"
kind: "gotcha"
summary: "Writing a char value back as C++ must use its number outside printable ASCII — `-128` round-trips where `'\\x80'` becomes +128 in a signed char — or chr() raises before the line even runs."
tags: [pycint, char, int8, echo, values]
created: "2026-11-26T09:53:47.290Z"
updated: "2026-11-26T09:53:47.290Z"
---
A `char` value outside printable ASCII has no character to quote, so the text that rewrites a value as C++ must be its **number**:

```text
value   written as      reads back as
65      'A'             65
-128    -128            -128      <- a signed char's own value
-128    '\x80'          +128      <- what chr()/escape forms do to a signed char
10      '\n'            10        (control characters keep their escape)
```

`_character` in `src/xrdroot/pycint/interactive/values.py` did `chr(int(value))`, which raised `ValueError: chr() arg not in range(0x110000)` for `-128` **before the line was run** — a probe that echoes a value back as object-like text died on an assignment rather than on anything the program did.

## Where it bites

ROOT's own `array_conversions.py` builds `struct Foo { <type> bar[11][2]; }` per type and assigns the type's minimum and maximum, `char` among them at `-128`, through `gInterpreter.ProcessLine`. Anything that names a `char`'s minimum — a table of bounds, a struct's initialiser, a `TTree` fill of every primitive — hits the same seam.

The rule now: a printable character keeps its quotes, a control character keeps its escape (`'\n'`, `'\t'`, `'\0'`, `'\\'`, `'\''`), and every other byte is written as its integer, which is what a C++ compiler accepts for a `char` and what this kit reads back. `tests/test_pycint_interactive.py::test_a_byte_that_is_no_character_is_echoed_as_its_number` pins all three cases.
