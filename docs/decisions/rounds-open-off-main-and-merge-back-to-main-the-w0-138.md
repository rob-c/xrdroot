---
title: "Rounds open off main and merge back to main: the W0-138 objection is spent"
kind: "decision"
summary: "main and feat/root-testsuite are the same commit as of round 65, so the loop's rounds now open every worktree off origin/main and merge back to main, verifying the push, before tearing down."
tags: [ralph, worktree, main, rounds, procedure]
created: "2026-11-22T19:07:33.798Z"
updated: "2026-11-22T19:07:33.798Z"
---
The ralph loop's rounds used to have to guess where to start: `main` was 238 commits behind the working line, so round 55 based itself on `feat/root-testsuite` and did not push (`W0-138`), while the task file's step 0 said to open off `main`. That is settled — at round 65 `main` and `feat/root-testsuite` are the **same commit**, measured on the primary checkout:

```text
git rev-list --count origin/main..HEAD   -> 0
git rev-list --count HEAD..origin/main   -> 0
```

## What the loop does now

`.ralph/ROOT.md` step 0 fetches and bases **every** worktree on `origin/main` (`BASE=$(git rev-parse origin/main)`, one integration worktree on a round branch cut from it, sixteen worker worktrees on `r-NN` branches at that commit), and step 6d runs four acts in order: **merge** the workers' commits into the round branch, **push** `$ROUND-main:main`, **verify** that `git ls-remote origin main` is the round's HEAD, then **tear down** every worktree and round branch. The round is not closed until the verification passes. Hard rule 11 carries the same order.

## Two consequences worth knowing

- **Re-check the divergence when a round opens.** If `origin/main` ever falls behind the line the project works on, a round based on `main` would silently drop those fixes: stop and ask the operator instead of basing the round elsewhere.
- **Round directories live under `$HOME/.cache/xrdroot-runs/`**, not `/tmp`, which is wiped mid-session (see the gotcha about `/tmp`); the worktrees, their logs and the round's `results.json` are what a ledger row quotes afterwards.
