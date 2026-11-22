---
title: "Local main drifts behind when you push from a worktree; sync it before reporting"
kind: "instruction"
summary: "A push from a worktree leaves local main behind, and git push then proves nothing: fast-forward it with git fetch origin main:main and check both hashes before reporting anything landed."
tags: [git, main, worktree, sync, commit]
created: "2026-11-22T14:15:46.853Z"
updated: "2026-11-22T14:15:46.853Z"
---
Pushing `HEAD:main` from a worktree updates the remote and leaves the repository's own `main` ref behind, where it drifts quietly — five commits behind after two pushes was typical. `git push origin main` then reports "Everything up-to-date" and proves nothing, because it compares the *remote* with that stale local ref, not the other way round.

```text
worktree  --git push origin HEAD:main-->  origin/main   (moves)
primary   --(nothing)-->                  local main     (stays behind)
                                          ^
      git fetch origin main:main ---------+               (fast-forwards it)
```

```bash
cd /Users/rcurrie/src/xrdroot
git status --porcelain                 # clean, or only another session's files
git fetch origin main:main             # fast-forwards local main
git rev-parse main; git rev-parse origin/main   # same hash = in sync
git log --oneline -8 main              # the work really is there
```

`git fetch origin main:main` is safe only because no worktree has `main` checked out; if one did, git refuses with "refusing to fetch into branch". Ask for the two hashes in separate `git rev-parse` calls — `git rev-parse main origin/main` intermittently dies with "fatal: Needed a single revision" while another session's fetch rewrites the ref.

## Before committing anything

Check `git status --porcelain` first: the primary checkout usually holds another session's modified files (`tools/rootsuite/baseline.json` and the sources of a round in flight are common), and an unqualified `git commit -a` would sweep them into your commit. Commit path-scoped with `git add <your files>`, and when the tree is clean and your work is already on `origin/main`, the honest answer is that there is nothing to commit — say so rather than manufacturing one.

Verify the session landed rather than trusting the push:

```bash
for c in <your commit shas>; do git merge-base --is-ancestor $c origin/main && echo "$c on main"; done
```
