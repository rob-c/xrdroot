---
title: "Unlimited ralph iterations: maxIterations 0, and which place wins"
kind: "instruction"
summary: "maxIterations: 0 means no limit (the plugin stops only when it is > 0); the /ralph start flag sets it and overwrites the state file, while resume keeps whatever the state file says."
tags: [ralph, loop, iterations, maxiterations, unlimited]
created: "2026-11-22T19:13:13.550Z"
updated: "2026-11-22T19:13:13.550Z"
---
`maxIterations: 0` is how this loop asks for **no limit**: the plugin's stop checks are `state.maxIterations > 0 && iteration >= maxIterations`, so `0` (or any non-positive number) never stops, and the status line shows the bare iteration with no `/N`. The default when the flag is omitted is 50.

```text
/ralph start ROOT --max-iterations 0     # a fresh start; the flag is what sets the value
/ralph resume ROOT                       # keeps the state file's maxIterations, flag or not
/ralph status                            # "iteration 3" with no "/50" means unlimited
```

`parseArgs` reads `--max-iterations N` with `parseInt(next, 10) || 0`, so a value that does not parse also lands on 0. `ralph_start`'s tool parameter is the same field: `ralph_start({ name: "ROOT", taskContent: <this file>, maxIterations: 0 })`.

## Where the value lives, and which place wins

`.ralph/ROOT.state.json` holds `maxIterations` and `/ralph start` **overwrites the whole state file** from its parsed arguments — so a state file edited by hand is only durable for `resume`, not for a restart. To make it stick: pass `--max-iterations 0` when starting, and check `.ralph/ROOT.state.json` afterwards. A loop whose `status` is `completed` cannot be resumed at all (the plugin answers "Use /ralph start ROOT to restart"), which is the case the ROOT loop reached on 2026-11-22 after round 64; `.ralph/ROOT.md` records the same in its opening paragraph.
