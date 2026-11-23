---
title: "Cache hits, never misses, when a dictionary can supply the answer later"
kind: "discovery"
summary: "Cache hits but never misses when a later dictionary load can supply the answer, and bisect inside one commit by applying its hunks to the parent's tree with `patch --batch`, reusing one built fixture."
tags: [cache, reflection, bisect, harness, fixture]
created: "2026-11-23T18:18:44.019Z"
updated: "2026-11-23T18:18:44.019Z"
---
A cache over a lookup that a later action can change must never remember a *miss*. `_class_named` answered from an index of every loaded module's namespace, and building that index walks all modules at the first ask — before a dictionary has declared anything — so `roottest`'s `CheckMissingDictionaries` saw its class with no members and stayed wrong for every later ask. Caching only what was found, and asking again for what was not, fixed it while keeping the speed.

## The rule

```python
_FOUND_CLASSES: dict[str, type] = {}

def _class_named(name):
    known = _FOUND_CLASSES.get(name)
    if known is not None:
        return known                       # a hit cannot be invalidated by a later load
    for module in sys.modules.values():    # a miss is asked again, every time
        ...
        _FOUND_CLASSES[name] = found
        return found
    return None
```

and `forget_dictionaries()` clears it, so a session reset drops even the hits.

```mermaid
flowchart LR
    a["first ask, before any load"] --> b{"in the cache?"}
    b -->|"no"| c["scan modules: not found"]
    c --> d["a miss is NOT stored"]
    d --> e["dictionary loads, declares the class"]
    e --> f["ask again"] --> g["found, and now cached"]
    b -->|"yes"| h["answered from the cache<br/>(hits only, so never stale)"]
```

## Bisecting inside one commit

The regression was in one commit with four hunks. `git diff GOOD BAD -- <file> > full.patch`, split it on `\n@@` into `hunk1.patch`…`hunk4.patch`, then build each variant from the **parent's whole tree** (not from HEAD's tree with pieces removed — that reconstruction is where two earlier attempts went wrong):

```bash
cp -a PARENT_SRC variant && cd variant && patch --batch --forward -s -p2 < hunk2.patch
```

`--batch` matters: plain `patch` waits for an answer at a prompt and looks like a hang.

## Reusing one built fixture across variants

The harness test costs ~35 s because it rebuilds a dictionary. When the variants only change the *reader*, build the fixture once and run the macro directly in that work directory:

```bash
cd RUN_DIR/work/roottest/<path> && PYTHONPATH=variant/src python -m xrdroot root -q -b \
    -e 'gSystem->Load("libIOWithoutDictionaries_dictrflx.xrdlib")' checkMissingDictionaries.C
```

3 s per variant instead of 35, and the answer is the same one the harness gives — it prints `(int) 0` for pass. Pass the whole glob when the fixture is what creates the class, or the isolated answer is meaningless.
