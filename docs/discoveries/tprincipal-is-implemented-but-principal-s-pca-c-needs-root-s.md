---
title: "TPrincipal is implemented, but principal's pca.C needs ROOT's LAPACK eigen-basis"
kind: "discovery"
summary: "TPrincipal matches ROOT's table and prologue, but math/principal's pca.C cannot: ROOT's LAPACK picks a different (arbitrary) eigenvector basis for the rank-deficient directions than numpy or scipy do."
tags: [tutorials, tprincipal, eigen, lapack, pairs]
created: "2026-11-22T14:05:24.275Z"
updated: "2026-11-22T14:05:24.275Z"
---
`TPrincipal` is implemented (`src/xrdroot/pyroot/core/principal.py`), and both `math/principal` tutorials moved from UNSUPPORTED/FAIL to **DIFF** — but they cannot become PASS, because the tutorial writes `pca.C` through `MakeCode()` and the harness compares that file. Its eigenvectors come from ROOT's own LAPACK call.

```text
principal.py  ->  principal.MakeCode()  ->  pca.C  (199 lines, compared byte for byte)
                                              |
                    ROOT: dsyev basis        ours: numpy eigh basis
                    signs per vector: mixed  signs: opposite for 9 of 10 vectors
                    3 zero eigenvalues: one  arbitrary basis, a different one
                    specific null-space basis
```

## What matches, and what cannot

The arithmetic matches: means, population sigmas and the seven non-zero eigenvalues agree to the digits `Print` shows, and our table differs from ROOT's only in the three numerically-zero eigenvalues (`7.791e-15` against `1.179e-16`), which the comparison's 1e-12 absolute tolerance accepts. `MakeCode`'s prologue, both functions and all four arrays are ROOT's own bytes.

What cannot match is the eigen**vectors**: for a symmetric matrix with repeated (here exactly zero) eigenvalues the basis is not unique, and ROOT's LAPACK basis is not numpy's — `eigh` (dsyevd) and `scipy.linalg.eigh(driver="ev")` (dsyev) both returned a different one, and neither reproduces ROOT's signs. So `pca.C` stays DIFF however the maths is arranged.

## What this means for the pair

`math/principal.C` and `math/principal.py` are 2-for-1 tutorials that are *not* reachable by implementing `TPrincipal`; they need ROOT's LAPACK eigen-basis, which is a decision (link a LAPACK with the same routine and pin its version, or accept DIFF and say why), not a fix. The other nine twin pairs are different: most compare images as well, so they need the class *and* canvas fidelity.
