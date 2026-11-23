---
title: "A binned RooFit curve is sampled either side of its bin edges, and the hint is the whole grid"
kind: "gotcha"
summary: "ROOT samples a binned RooFit curve at plotSamplingHint's points - delta = (xhi-xlo)*1e-9 either side of every bin edge, as the whole seed grid - and a RooBinWidthFunction's histogram is a shape server, or components will not add up."
tags: [roofit, plotting, histfactory, sampling, root]
created: "2026-11-23T19:38:11.111Z"
updated: "2026-11-23T19:38:11.111Z"
---
A curve of a binned RooFit function is not sampled on a frame's own bins: ROOT asks the function where to sample (`plotSamplingHint`), and that answer *is* the whole starting grid. Getting this wrong makes a HistFactory component plot disagree with the model it belongs to, and it is what `gtest/roofit/histfactory/testHistFactoryPlotting` checks.

## The rule, from ROOT's source

`RooCurve::plotSamplingHintForBinBoundaries` (`roofit/roofitcore/src/RooCurve.cxx:897`) returns, for
limits `xlo, xhi` and `delta = (xhi - xlo) * RooCurve::relativeXEpsilon()` with
`relativeXEpsilon() == 1e-9` (`RooCurve.h:104`): `xlo + delta`, `xhi - delta`, and `x - delta`,
`x + delta` for every bin edge `x` strictly inside both limits; sorted. `RooCurve::addPoints`
(`RooCurve.cxx:283`) then uses it as the **entire** seed list — `minPoints = hint->size()`,
`xval = hint` — so a step curve has as many starting points as it has edges, not as many as the
frame has bins.

Every class above the histogram merges its children's hints: `RooRealSumPdf::plotSamplingHint`
(`RooRealSumPdf.cxx:615`, sorted, consecutive duplicates removed), and `RooProduct`, `RooProdPdf`.
A `RooHistPdf`/`RooHistFunc` answers none when its bins are interpolated (`intOrder > 0`), because
an interpolation has no jump to catch.

## The second half: which components a plot selects

`RooAbsReal::plotOnCompSelect` walks in **value** inputs only (`dependsOn(..., valueOnly=true)`),
and a `RooBinWidthFunction`'s `HistFuncForBinWidth` proxy is declared
`valueServer=true, shapeServer=true` (`roofit/roofitcore/inc/RooBinWidthFunction.h:35`). That shape
flag is load-bearing: without it a value-only walk follows the bin width into a *sibling* sample's
histogram, so selecting `*histoSig*` also selects `histoBG` and the two curves cannot add up.

## How to check it here

`/tmp/rootsrc-6.40/roofit/histfactory/test/testHistFactoryPlotting.cxx` run twice — once by
`~/.local/root-oracle/bin/root -l -b -q` and once by `python -m xrdroot run` — printing
`GetPoint(i, x, y)` for all three curves. `diff` of the two must be empty: 35 points each, first x
`2896.4482758620688`, and `y_full 479.99999999999994`, `y_sig 2`, `y_bg 478`. In the kit the code
is `roofit/pdfs/histpdf.py::_Histogram.plot_sampling_hint`, `roofit/real.py::merge_hints`,
`roofit/plot/curve.py::sample(..., hint=...)` and `roofit/plot/curves.py::_sampling_hint`.

```text
plotOn(frame, Components("*sig*"))
  -> curves._add_curve -> _sampling_hint(func, name, low, high)
       RooRealSumPdf.merge  (RooProduct.merge  RooHistFunc.plot_sampling_hint)
  -> curve.sample(..., hint=...)      # the hint is the whole seed grid
  -> RooCurve points == ROOT's 35
```
