---
title: "A std::string is a Python str, so a member that changes it is an assignment"
kind: "architecture"
summary: "A std::string is a Python str, so each mutating member is emitted as an assignment to its variable (a table in emit_stmt.py), and only a whole-string range can be replaced."
tags: [pycxx, translator, string, algorithms, coverage]
created: "2026-11-26T16:10:44.261Z"
updated: "2026-11-26T16:10:44.261Z"
---
A `std::string` is a Python `str`, which nothing writes in place, so a member that changes the string is translated to an **assignment back to its variable** — and that means every write path has to be a statement. `append`, `push_back`, `clear`, `assign`, `resize`, `pop_back`, `erase(...)`, `insert(...)` and `replace(...)` each have an entry in `_STRING_STATEMENTS` (`src/xrdroot/pycxx/emit_stmt.py`), reached only when the call *is* a statement; the same call in the middle of an expression is still refused as "changing this in the middle of an expression".

## The consequence for an algorithm over a range

`std::replace(s.begin(), s.end(), a, b)` cannot write through the iterators the way it does over a `std::vector`, so `CallEmitter._replace` emits `(s := s.replace(a, b))` instead of `replace_range(...)`. Only the **whole** of a string can be done that way — a partial range (`s.begin() + 1, s.end()`) is refused by name rather than left to fail as `TypeError: 'str' object does not support item assignment`.

```mermaid
flowchart LR
  C[".method(...) as a statement"] --> T{"owner is a std::string?"}
  T -->|yes| S["target = new text<br/>(_STRING_STATEMENTS)"]
  T -->|no| P["the call as written"]
  A["std::replace/sort over s.begin()..s.end()"] --> W{"the whole of one string?"}
  W -->|yes| X["(s := ...)"]
  W -->|no| R["refused by name"]
```

## What to do instead of guessing

Add the member to `_STRING_STATEMENTS` with a test in `tests/test_cint_calls.py` that runs the snippet and prints the text; an arity the kit does not write should `raise refuse(...)` so the failure names itself.
