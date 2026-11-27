---
title: "A nested base row is not coerced by a top-level cast"
kind: "gotcha"
summary: "TLatex fails struct packing because base classes keep members in nested dicts, so coercion must walk the whole description closure; TObjArray's engine has no members at all and needs a collection payload."
tags: [writer, tlatex, tobjarray, winfo, coercion]
created: "2026-11-27T12:37:17.209Z"
updated: "2026-11-27T12:37:17.209Z"
---
`TLatex`'s members reach the packer with the wrong Python types for two different reasons, and only the first is fixed by casting the top-level row: **base classes keep their members in nested dicts** (`{"TAttLine": {"fLineColor": 1.0}}`), so a coercion that walks only `layout_of("TLatex")` leaves the bases' numbers untouched and `struct.pack` raises `required argument is not an integer`. `TObjArray` is further away again: its engine has **no `members` at all**.

```python
# what the engine of a TLatex hands over
unwrap(ROOT.TLatex(0.1, 0.2, "hi")).classname   # 'TLatex'
unwrap(ROOT.TLatex(0.1, 0.2, "hi")).members     # dict, with nested base dicts
unwrap(ROOT.TObjArray()).members                # AttributeError: no members
```

## What a fix has to do

1. Coerce **every** class in the description closure, not just the class being written: walk `_closure({"TLatex": None})` and cast each owner's row — the class's own at the top, a base's inside its nested dict.
2. Cast by what the element declares, not by what Python holds: `FORMS[stype] in "bBhHiIlLqQ"` means `int`, anything else `float`. The engine keeps some integer flags as floats, which is why the plain `_object_payload` path fails where `TGraph2D` succeeded.
3. `TObjArray` needs a real collection payload — a count plus each child written through `_record` — because the engine carries no members to normalise.

```mermaid
flowchart TD
    A["TLatex engine"] --> B["members: {own..., 'TAttLine': {...}, 'TText': {...}}"]
    B --> C{"coerced every owner?"}
    C -->|"no: only the top row"| D["struct.error: required argument is not an integer"]
    C -->|yes| E["_record -> file"]
    F["TObjArray engine"] --> G{"has members?"}
    G -->|no| H["needs a collection payload: count + children"]
```

## Why this is worth recording

The pattern "cast numpy scalars to int/float" worked for `TGraph2D` and is *not enough* one class later — the difference is that `TGraph2D`'s row is flat and `TLatex`'s is nested by base. Anyone repeating the `TGraph2D` recipe on a class with bases will hit this, and the error message points at the packer rather than at the nesting.
