---
title: "The harness reports the first differing line, not the only one"
kind: "gotcha"
summary: "Diff a DIFF's whole normalised stream before pricing it: the harness names one differing line, and rs101 had six."
tags: [harness, diff, tutorials, roottest, triage]
created: "2026-11-24T00:26:47.384Z"
updated: "2026-11-24T00:26:47.384Z"
---
A harness result reports the *first* line a tutorial differs in, not the only one. So a DIFF row is never a one-line fix until the whole normalised stream has been diffed against the oracle's own output: `tools.rootsuite`/`tools.tutorials` names one line because naming one line is what a reader can hold, and three rounds in a row read that line as the cause.

Round 92 left `roofit/roostats/rs101_limitexample.C` DIFF with the detail `stdout line 17: ... normalize constraints with respect to the parameters (ratioSigEff,ratioBkgEff)`, and priced it as a `RooArgSet` ordering question (`W0-209`). Round 93 diffed the whole stream instead and found **six** differing lines, of which the parameter order is one:

```text
line  22  (ratioSigEff,ratioBkgEff)  vs  (ratioBkgEff,ratioSigEff)     the order
line 293  this test stat = 1.64105    vs  1.64097                       the MCMC
line 475  Proposal acceptance rate: 47.94%  vs  47.785%
line 476  Number of steps in chain: 9588    vs  9557
line 481  MCMC lower limit on s = 20.1859   vs  18.8057
line 482  MCMC upper limit on s = 102.808   vs  103.058
```

The real wall is `MCMCCalculator`'s proposal function consuming a different random stream, which no amount of set ordering will fix (`W0-211`).

## What to do

1. Before writing anything, run the tutorial by hand and diff the **normalised** streams: `tools.tutorials.compare.normalise` on the oracle's cached `stdout` (`~/.cache/xrdroot-tutorials/oracle/<version>/<hash>/result.json`) against this tree's run.
2. Count the differences and group them by cause. A DIFF whose groups are one is a fix; a DIFF whose groups are three is a queue row per group.
3. Quote the whole diff, not the harness's one line, when a row's price is written down.

The same trap runs the other way for the suite: `.ref` comparisons in `tools.rootsuite` also report one line, so a `roottest` DIFF needs the same treatment.
