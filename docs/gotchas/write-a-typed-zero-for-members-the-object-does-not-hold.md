---
title: "Write a typed zero for members the object does not hold"
kind: "gotcha"
summary: "A record must carry every declared member, so missing ones get a typed zero (int/float by FORMS[stype]) — and a ninth payload shape needs _shaped_payload split first, since it sits at npath 256."
tags: [writer, tlatex, winfo, maintainability, coercion]
created: "2026-11-27T12:45:22.257Z"
updated: "2026-11-27T12:45:22.257Z"
---
A record must write **every** member its layout declares, including ones the object does not hold — `TLatex`'s engine has no `fLimitFactorSize`, and passing `None` into the packer raises `struct.error: required argument is not an integer`. And a ninth payload shape no longer fits the dispatch: `_shaped_payload` sits at the maintainability limit, so it has to be split *before* the class is added, not after.

```text
$ python tools/maintainability.py check
  src/xrdroot/writer.py::_shaped_payload: npath=256 > limit 200
```

## The two rules

1. **Fill missing members with a typed zero** while casting:
   ```python
   held = row.get(name)
   row[name] = int(held or 0) if integer else float(held or 0.0)
   ```
   with `integer = FORMS[stype] in "bBhHiIlLqQ"`. This is what made `TLatex` cross ROOT both ways (`io_matrix --classes TLatex` → 1 of 1).
2. **Members are flat**: a base's members (`fTextColor`, `fLineColor`, …) sit beside the class's own in the engine's dict, so the cast walks the whole description closure (`_closure({classname: None})`), not just the class's own elements.

```mermaid
flowchart TD
    A["add a 9th payload shape"] --> B{"_shaped_payload at npath limit?"}
    B -->|"yes (256 > 200)"| C["split it first: value-shaped vs drawing-shaped, with its own tests"]
    B -->|no| D["add the shape"]
    C --> D
    D --> E{"does the object hold every declared member?"}
    E -->|no| F["cast with a typed zero for missing members"]
    E -->|yes| G["_record -> file"]
    F --> G
    G --> H["io_matrix: PASS both ways"]
```

## Why the order matters

Splitting `_shaped_payload` *while* adding the class produced 13 failing tests in this round, and the class had to be reverted with the split. Split first, run `pytest tests/test_writer_tarray.py`, and only then add the payload — otherwise a working class is thrown away with a broken refactor.

`TObjArray` is not this problem at all: `unwrap(ROOT.TObjArray())` has **no `members`**, so it needs a collection payload — a count, then each child through `_record`.
