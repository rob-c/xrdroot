---
title: "Landing work on main here: a worktree off origin/main, pushed at once, CI read without gh"
kind: "instruction"
summary: "More than one session commits in `/Users/rcurrie/src/xrdroot` at the same time, and they do not share a branch: the primary checkout is usually on `feat/root-testsuite` (someone else's line), `main` is what CI runs, and…"
tags: [worktree, main, concurrent-sessions, ci, gh]
created: "2026-11-22T13:44:31.541Z"
updated: "2026-11-22T13:44:31.541Z"
---
More than one session commits in `/Users/rcurrie/src/xrdroot` at the same time, and they do not share a branch: the primary checkout is usually on `feat/root-testsuite` (someone else's line), `main` is what CI runs, and another session periodically resets `main` to the remote — `git reflog show main` shows `branch: Reset to origin/main`, which silently discards any local-only commit on `main`.

```text
primary checkout (feat/root-testsuite)  -- never switch it, never commit here by accident
        |
        +-- git worktree add ~/.cache/xrdroot-runs/wt-x -B x origin/main
        |        cherry-pick <commits>, run the gates there
        +-- git push origin HEAD:main        (from the worktree, not from the checkout)
        +-- git worktree remove --force …    then: git branch -D x
```

## The rules that follow

- Work in a worktree off **`origin/main`** after a `git fetch`, never in the primary checkout — a commit made there lands on `feat/root-testsuite` and has to be cherry-picked out again.
- **Push as soon as a commit is on `main`**, then check `git ls-remote origin main`; a commit that is only local will be reset away within minutes. Never remove the worktree before the push: it holds the only reference to the round's commits.
- Record gates on the worktree, not on the checkout, and treat `tools/rootsuite/baseline.json` as somebody else's file — it is often dirty from another session's run.
- To read CI here without `gh`: `curl -s "https://api.github.com/repos/rob-c/xrdroot/actions/runs?branch=main&per_page=3"`, then `/actions/runs/<id>/jobs` for per-job conclusions. `gh` has no token on this machine (`gh auth status` prints nothing, `~/.config/gh/hosts.yml` has no `oauth_token`), so `gh run view --log-failed` returns nothing and job logs need admin rights; the jobs endpoint names which interpreter failed, which is usually enough to recognise the known worker-crash flake.
