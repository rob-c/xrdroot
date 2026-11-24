---
title: "Auditing for unpushed work: detached worktrees are invisible to git branch"
kind: "instruction"
summary: "Audit for unpushed work by walking git worktree list, not branches — the loop commits on detached HEADs that no branch shows, and git push alone proves nothing when local main is stale."
tags: [git, worktree, push, audit, detached-head]
created: "2026-11-24T18:44:43.276Z"
updated: "2026-11-24T18:44:43.276Z"
---
Answering "is everything pushed?" with `git branch` is wrong in this repository: the loop's rounds commit on **detached** worktrees, so their commits appear on no branch at all. This run found the primary checkout 9 commits *behind* `origin/main` with nothing ahead, and had to walk every worktree to be sure nothing was stranded.

```text
$ git branch / for-each-ref      -> misses detached worktrees entirely
$ git worktree list --porcelain  -> the only list that includes them
$ git ls-remote --heads origin   -> what a remote-only branch would hide
```

```bash
cd /Users/rcurrie/src/xrdroot
git fetch origin main -q
echo "ahead:  $(git rev-list --count origin/main..HEAD)"
echo "behind: $(git rev-list --count HEAD..origin/main)"
for w in $(git worktree list --porcelain | awk '/^worktree /{print $2}'); do
  printf "%-52s ahead=%s\n" "$w" "$(git -C "$w" rev-list --count origin/main..HEAD)"
done
git for-each-ref --format='%(refname:short)' refs/heads | while read -r b; do
  a=$(git rev-list --count "origin/main..$b"); [ "$a" -gt 0 ] && echo "branch $b +$a"
done
git ls-remote --heads origin
```

## Reading the answer

- `ahead: 0` on the checkout plus `ahead=0` in every worktree means there is nothing to push; `git push origin main` then answers "Everything up-to-date", which by itself proves nothing (it compares the remote against a possibly stale local `main`).
- A worktree with `ahead>0` is unpushed work even though no branch shows it — integrate and push it before removing the worktree, or the commits die with it.
- A local branch ahead of `origin/main` is not automatically work to push: `archive/*` refs are deliberate (`archive/unjudged-salvage`, `archive/branches-20261123`) and stay local or are pushed only as archives.
- When the checkout is *behind*, sync it with `git merge --ff-only origin/main` while `main` is checked out — `git fetch origin main:main` refuses in that case ("refusing to fetch into branch").
