# Analysis engines

The engines return plain Python/NumPy-backed analysis objects. ROOT spelling,
wrapper construction and session behavior belong in `pyroot`. Public ROOT APIs
remain covered by the existing [ROOT guide](../root.md).

## hist

Hist owns histograms, profiles, efficiencies, graphs and graph arithmetic,
stacks, sparse data, formulas, functions and fitting. The implementation remains
in existing modules such as `hist`, `graph`, `function`, `formula` and `fit`.
Its only direct component dependency is core.

Shared primitive-array descriptions and jagged operations are imported from
core, not the tree reader. Plain scalar distribution functions belong in the
engine; formula registration must work before importing `xrdroot.pyroot`.
The ROOT `Math` namespace re-exports or wraps those engine functions.

Fitting uses the existing `fit` extra where required. Missing optional packages
retain a full-sentence refusal identifying the existing extra. Numeric moves
preserve integer widths, float32 stores, platform tolerances and ROOT libm
choices; an architectural change grants no new numeric tolerance.

## spectrum

Spectrum owns peak search, background estimation and related spectral analysis
under `xrdroot.spectrum`. It depends directly on core. ROOT `TSpectrum` names
and graphical objects stay in the corresponding `pyroot` wrappers. Spectrum
algorithms must not load the canvas or ROOT session merely to process arrays.

## fits

Fits owns FITS-format data handling under `xrdroot.fitsio`, including the engine
support used by ROOT-facing FITS wrappers. It depends directly on core.
The component name `fits` refers to the astronomical file format; statistical
curve fitting belongs to hist's `fit` package. Public ROOT FITS objects are
constructed by `pyroot`, not by the FITS reader.

## roofit

RooFit owns model expressions, PDFs, datasets, integration, likelihoods,
minimization and workspace decoding under `xrdroot.roofit`. Its direct
component dependencies are core, hist and cppstd.

ASCII dataset output uses the shared C++ stream implementation. Scalar maths
comes from the lower engine. When public operations require a ROOT graph,
marker or binding, pass engine results or explicit callables to the wrapper
instead of importing `xrdroot.pyroot` from the model engine.

Workspace decoding keeps its maker registry and precedence. The existing
ModelConfig maker path remains a registered adapter; its construction body is
owned by RooStats and reached through the declared `workspace.maker` provider.
This preserves maker state without importing the higher component during a
plain RooFit import.

## roostats

RooStats owns hypothesis tests, intervals, limit scans, toy studies and MCMC
under `xrdroot.roostats`. It depends directly on core, hist and RooFit.
Engine scans and fitting consume engine graphs/functions or return numeric data;
public ROOT plotting objects are made in the ROOT wrappers. Persistence-specific
ModelConfig reconstruction belongs in `roostats.workspaceio`.

Refactoring these paths must preserve fitted results, random-number consumption,
return values and exception behavior. A faster or more isolated import is not
numerical acceptance; native comparisons and platform checks retain their scope.

## histfactory

HistFactory owns channels, samples, systematic variations, model construction
and its on-disk layouts under `xrdroot.histfactory`. Its direct dependencies are
core, hist, I/O, RooFit and RooStats.

The generic file interpreter uses the declared `io.layouts` provider for the
known HistFactory class keys. A layout carried by the file takes precedence over
the built-in fallback. The provider declaration is a narrow dispatch permission;
it does not permit the I/O interpreter to import HistFactory directly.

## tmva

TMVA owns training, prediction, evaluation and model persistence under
`xrdroot.tmva`. It depends directly on core, hist and I/O. Existing optional
learners are selected through the project's `tmva` extra and loaded lazily.

ROC calculation returns engine data. ROOT-facing `GetROCGraph`/`GetROCCurve`
and canvas/legend construction live in the wrappers, preserving their public
return contracts. File access inside the engine uses the owning I/O modules
rather than importing the app facade. Training and evaluation fixtures remain
explicit integration tests when they use several components.
