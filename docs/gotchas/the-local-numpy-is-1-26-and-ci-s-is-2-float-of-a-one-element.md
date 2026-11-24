---
title: "The local NumPy is 1.26 and CI's is 2: float() of a one-element array is green here and red there"
kind: "gotcha"
summary: "The local gate runs NumPy 1.26 and CI runs NumPy 2, where float() of a one-element array raises; extract the element with reshape(()) and check `python -W error::DeprecationWarning` locally."
tags: [numpy, ci, gate, funcs, deprecation]
created: "2026-11-24T22:02:19.544Z"
updated: "2026-11-24T22:02:19.544Z"
---
The local dev venv runs NumPy 1.26 while the CI runners run NumPy 2, so a `float()` of a one-element array passes here and fails there — check any new NumPy-to-Python scalar conversion against that rule before pushing, and treat `python -W error::DeprecationWarning` as the local stand-in for NumPy 2.

## What breaks and how to see it locally

Code written for arrays — the normal way to write a `TF1` here, since one call may take thousands of
rows at once — answers a **one-row** input with a one-element array. Extracting the number with
`float(answer)` was accepted with a warning until NumPy 2 made it an error:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

which is exactly what CI reported on Python 3.11/3.12/3.13 while 3.10 (NumPy 1.x) stayed green, from
round 118 to round 120 (`tests/test_pyroot_tkde.py::test_the_estimate_as_a_function_is_sampled_and_drawable`,
`src/xrdroot/pyroot/core/funcs.py`). Locally the same call only warns, so a green gate proves nothing
about it:

```bash
.venv/bin/python -W error::DeprecationWarning -c 'import numpy as np; float(np.array([2.5]))'
# DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated,
# and will error in future. ... (Deprecated NumPy 1.25.)
```

## The rule

Take the element out explicitly: `float(np.asarray(answer, dtype=np.float64).reshape(()))` after
checking `size == 1`, and refuse several values rather than truncating to the first
(`funcs._scalar`). Do the same for `int()`, `bool()` and comparisons against a size-1 array.

```mermaid
flowchart LR
  A[numpy answer] --> B{size == 1?}
  B -- no --> C[ValueError: not one point's value]
  B -- yes --> D[float after reshape to 0-d]
  D --> E[works on NumPy 1.x and 2.x]
```
