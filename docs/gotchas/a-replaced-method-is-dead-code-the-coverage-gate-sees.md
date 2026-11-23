---
title: "Replacing a method with a faster one leaves the old one as uncovered dead code"
kind: "gotcha"
summary: "When a hot method is replaced by a faster path, the old method's body fails the 100% coverage gate as unreachable lines; delete it, don't write a test for an unused method."
tags: [coverage, gate, perf, cleanup]
created: "2026-11-23T09:29:37.291Z"
updated: "2026-11-23T09:29:37.291Z"
---
Making one hot path faster by giving it its own inlined implementation leaves the method it replaced with no caller, and the project's 100 % line-and-branch coverage gate then reports that body as unreachable lines. The answer is to delete the old method, not to test a method nothing calls.

## What happened

`Sampler.pop` handed out a held draw and called `advance(1)`; `TRandom._next` used it for every number an acceptance algorithm asked for one at a time, which was 1.25 us a call. A new `Sampler.take_one` does both steps itself, and `_next` asks for that instead.

Both parts of the two-command coverage gate passed every test and then named the gap exactly:

```
src/xrdroot/random/trandom.py     359      5     96      0    98%   220-224
```

Lines 220-224 were `pop`'s body - five statements reachable only by a test nobody would want.

## The rule

When a refactor removes the last caller of a method, remove the method in the same commit that removes the caller. A test written to reach an unused method is coverage theatre: it makes the number green while the code stays dead.
