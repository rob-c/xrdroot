---
title: "Resuming the ROOT ralph loop: /ralph resume ROOT, after checking who owns it"
kind: "instruction"
summary: "The ralph loop in this repository is called **`ROOT`**: its state is `.ralph/ROOT.state.json` (`taskFile: .ralph/ROOT.md`) and the task file is both the procedure and the record of every round."
tags: [ralph, loop, resume, state, workflow]
created: "2026-11-22T19:11:43.091Z"
updated: "2026-11-22T19:11:43.091Z"
---
The ralph loop in this repository is called **`ROOT`**: its state is `.ralph/ROOT.state.json` (`taskFile: .ralph/ROOT.md`) and the task file is both the procedure and the record of every round. Resume it with the slash command, not the tool — `ralph_start` refuses an active loop outright ("Loop \"ROOT\" is already active. Use /ralph resume ROOT").

```text
/ralph resume ROOT        # re-reads .ralph/ROOT.md, iteration += 1, takes ownership
/ralph status             # every loop, its status, iteration and task file
/ralph-stop               # stop when idle, before handing the loop to another session
```

`resume` sets `ownerSessionId` to the resuming session, so it is also how ownership moves. It
refuses a loop whose `status` is `completed` ("Use /ralph start ROOT to restart") — for that case
start it again with the limit you want, `/ralph start ROOT --max-iterations 0`, which resets the
iteration counter and re-reads the same task file. Before resuming, check the state file: `jq -r '.status, .ownerSessionId, .startedAt' .ralph/ROOT.state.json` and compare with `echo $PI_SESSION_ID`. If `status` is `active`, the `ownerSessionId` is a different session, and the file's mtime is within minutes, another session is driving the loop — resuming then gives two drivers over the same `main`. Stop it there first.

## Where "resume" actually starts from

`iteration` in the state file is only a counter. The *round* to continue is the last close in `.ralph/ROOT.md` (`## Round NN's close`, and the round-log table near its end), with `TODO.md`'s §7 ledger and §4 queue as the work itself. A resumed iteration reads step 0 first, which now bases every worktree on `origin/main` and re-checks that `main` has not fallen behind, and closes with step 6d: merge, push, verify `git ls-remote origin main`, then tear down.

The file is 236 KB by round 64; if that is too much prompt, archive the older round closes into `docs/history/` before resuming rather than starting a second, thinner loop.
