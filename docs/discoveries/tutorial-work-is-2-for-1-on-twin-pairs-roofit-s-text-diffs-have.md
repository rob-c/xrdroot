---
title: "Tutorial work is 2-for-1 on twin pairs; roofit's text DIFFs have three causes, one unreachable"
kind: "discovery"
summary: "310 non-passing tutorials are ~240 problems because 56 twin pairs fail on both sides; size tutorial work by problem, and expect roofit's 57 text DIFFs to split into integral caching, missing classes, and unreachable Minuit2 output."
tags: [tutorials, roofit, measurement, planning, twin-pairs]
created: "2026-11-22T13:43:29.533Z"
updated: "2026-11-22T13:43:29.533Z"
---
The tutorial suite's 310 non-passing files are **only ~240 distinct problems**: 56 problems fail on *both* sides of a `.C`/`.py` twin pair, so one fix moves 112 tutorials (214 twin pairs exist among the 1189). Sizing work by tutorial rather than by problem overstates a campaign by a third and picks the wrong order — prefer any fix whose twin is failing too.

```text
1189 tutorials
 ├─ 247 SKIP permanent (GUI 153, helpers/not-in-CMake 94)
 ├─  76 SKIP oracle-side (xml 13, r 9, root7 4, ML packages 30)
 ├─   5 ORACLE-FAIL
 └─ 861 obtainable   <- 750 is 87% of this, not of 1189
     └─ 310 non-passing = ~240 problems
         └─ 56 of them fail on BOTH sides of a .C/.py pair  -> one fix, two tutorials
```

## Where the roofit non-passes actually go

Sorting the 57 text-only consumer tutorials by *what* differs, three causes cover almost all, and only two are reachable by printing work:

- **integral objects made per use, and projected pdfs named differently** — `rf614` ×2, `rf315` ×2, `rf316` ×2, `rf211` ×2, `rf705` ×2, `rf106` ×2. Ours announces `RooRealIntegral::init(powerlaw_Int[x|bin]_Norm[x])` eleven times where ROOT makes one integral and says so once; our projection is `model_Proj[y]` where ROOT has `model_Int[y]_Norm[x,y]_wrapped_pdf` and `SPECINT[…]`. +10 tutorials for a per-(function, variables, range, normalisation-set) cache plus ROOT's names.
- **missing classes** — `RooStats::SPlot`, `RooJeffreysPrior`, `RooPolyFunc`, `RooSimWSTool`, `RooMomentMorphFuncND`, `RooLagrangianMorphFunc`: 14 tutorials, each a self-contained class.
- **Minuit2's own voice** — `Info in <Minuit2>: MnSeedGenerator …`, `Initial state: FCN … NCalls = 15`, and the EDM/NCalls/Chi2 beside them (`rf506`, `rf607`, `rf611`, `rf204a/b`, `rf612`, `rf619`, `rf8xx`, the RooStats demos). ~20 tutorials not to be had by printing: the harness tolerates EDM within 1 %, but `NCalls 53 → 51` and `Chi2 7.28 → 11.67` are different minima — this is reproducing Minuit2's stepping.

## The measurement discipline that goes with it

A full run is 1–2 h at `--jobs 8`; one area (`--only 'roofit/roofit/'`, 171 tutorials) is ten minutes and shows whether a fix moved anything. Diffing both sides' **whole stdout** locally (`root -l -b -q X` vs `python -m xrdroot run -n X`, then `diff`) finds every difference at once, far quicker than the harness, which reports only the first. Confirm with the harness afterwards: only it applies the comparison's rewrites (addresses, dates, `RT=… s`, dropped "took N ms" lines) and compares images and files.
