---
title: "ROOT::Round decides digits and factoring from the error, and truncates toward zero"
kind: "discovery"
summary: "ROOT::Round picks 1-2 error digits and a power-of-three factor; the factor's cast truncates toward zero (int(), not floor), and a value rounded to nothing keeps its minus."
tags: [stringutils, round, gtest, formatting, pyroot]
created: "2026-11-26T16:11:05.140Z"
updated: "2026-11-26T16:11:05.140Z"
---
`ROOT::Round(value, error, cutoff, delim)` is a *printing* rule, not a rounding rule: it decides how many significant digits of the error to keep and whether to write the pair as `A#pmB` or `(A#pmB)*1eN`. The source is `core/foundation/src/StringUtils.cxx`, not the installed header (`ROOT/StringUtils.hxx` only declares it), so read it from ROOT's GitHub at the matching tag rather than from `$ROOTSYS`.

## The decisions, and the two that bite

```text
error = 0.24, cutoff = 1     first digit 2 > cutoff 1  -> one digit:  110.0#pm0.2
error = 0.24, cutoff = 2     first digit 2 <= cutoff   -> two digits: 120.00#pm0.24
error = 0.24                 factor: int(log10(.24)/3) * 3 = 0        -> no (a#pmb)*1eN
error = .24 with math.floor  factor = -3 (wrong)                      -> (110000#pm200)*1e-3
```

- `cutoff` (default 1) widens the error to two significant digits when its first digit is at most `cutoff`, **and** the two-digit form (rounded to one digit) is still at most `cutoff`; then the error's exponent drops by one.
- The factoring out comes from the **error**: `floor(log10(error)/3)*3` below `1e-3`, and a C **truncating** cast to int above it. `0.24` truncates to `0e0`, so nothing is factored - Python's `int(x)` is the faithful translation and `math.floor` is *not*. Ruff's RUF046 asks for the `int()` to be dropped; it is not redundant here.
- A value that rounds to nothing keeps its sign: `std::round(-0.4)` is `-0.0` and `std::ios` prints `-0`; Python's `f"{0.0:.0f}"` loses it, so `_rounded` returns `-0.0` for a negative input whose magnitude rounds away.

All 36 expectations of `core/foundation/test/testStringUtils.cxx` are the matrix in `tests/test_pyroot_stringutils.py`.
