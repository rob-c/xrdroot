---
title: "TLinearFitter must match ROOT's hypN, weights and P+1 answers"
kind: "discovery"
summary: "For fitLinear2.C to pass, TLinearFitter needs hypN with the constant first, 1/e^2 weights, P+1 answers, fixed parameters with error 1, and Eval() fitting; TF1Convolution needs GetNpar and __call__."
tags: [tlinearfitter, tutorials, fitlinear2, tf1convolution, weights]
created: "2026-11-28T14:27:32.285Z"
updated: "2026-11-28T14:27:32.285Z"
---
`TLinearFitter` only looks finished when `math/fit/fitLinear2.C` passes, and ROOT that tutorial's answers depend on are not the obvious ones: `hyp5` is a hyperplane **with its constant first**, a point's error is a **weight** of 1/sigma², `GetParameters`/`GetErrors` answer **one more** number than the formula names, and a fixed parameter comes back with an **error of exactly 1**.

```bash
python -m tools.tutorials run --tutorials ~/src/.xrdroot-scratch/rootsrc/tutorials --jobs 1 \
  --oracle-prefix ~/.local/root-oracle --xrdroot-src $PWD/src \
  --out ~/.cache/xrdroot-runs/tut-fl2 --only math/fit/fitLinear2.C
# then: python -c "import json;d=json.load(open('.../results.json'));print(d['tutorials'][0]['status'])"
```

```mermaid
flowchart LR
    A["SetFormula('hyp5')"] --> B["terms = 1, x0, x1, x2, x3, x4"]
    B --> C["AssignData(n, 5, x, y, e)"]
    C --> D["weighted lstsq: weight = 1/e**2"]
    D --> E["GetParameters: P+1 numbers, constant last"]
    E --> F["FixParameter keeps h[1] and reports its error as 1"]
```

## The four facts, each one a diff away from failure

| what ROOT does | what it looked like without it |
|---|---|
| `hypN` = `1 + x0..xN-1` | terms `x0..x4`: every parameter shifted by one, `par[0]=3.999934` against ROOT's `0.000069` |
| error is a weight | right coefficients, errors 100x too large (`0.101142` against `0.001011` for `e = 0.01`) |
| `P+1` answers | `IndexError: 5 is outside this TVectorD of 0..4` in the tutorial's own loop |
| fixed parameter's error is `1.0` | `par[1]=4.000000+-0.000121` against ROOT's `+-1.000000` |

Also needed, and easy to miss: `GetParameters(TVectorD&)`/`GetErrors(TVectorD&)` must size the vector they are given — here `into.Use(n, data)` — and `Eval()` with no argument **fits** and returns the chi-square, which is how the tutorial drives it.

`TF1Convolution` needs three things before `TF1("f", conv, ...)` will take it: `GetNpar` (the two functions' parameters summed), `SetNofPointsFFT`, and `__call__(x, par)` so it is callable, since `TF1` treats a callable source as a function and anything else as a formula string — a bare object makes the formula parser report `'<TF1Convolution object…>' has '.' at character 8`.
