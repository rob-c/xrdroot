# ROOT's interpreter: pycint

`xrdroot.pycint` is the CINT-compatible face of xrdroot: ROOT's macros, read in
ROOT's dialect of C++ by [pycxx](pycxx.md) and run against the ROOT namespace
of [pyroot](applications.md#pyroot). It depends on pycxx and on pyroot and
everything below it; nothing below it imports it. pyroot's
`gROOT.ProcessLine`, `Macro`, `LoadMacro`, `gInterpreter.Declare`, the
session's `.x` and RDataFrame's C++ expressions reach it through the
`interpreter.*` providers of `xrdroot.plugins` (see [providers](providers.md)).

## What it owns

| Module | What it is |
|---|---|
| `dialect` | `ROOT`, the dialect: ROOT's typedefs (`Int_t`, `Double32_t`), what its class names look like, `ROOT/TestSupport.hxx`, the `R__` and `ClassDef` macros and `ROOT_VERSION`, `TMath`, `TString`, `TTreeReaderValue`/`Array`, `RVec` and its aliases, the methods ROOT writes through |
| `emit` | the hooks: `TString::Format` and `s.Form()`, loops over `gRandom` draws filled all at once, and ntuple loops filled in a batch |
| `fill_loops`, `tree_loops` | the analyses those loops are recognised by |
| `returned` | what `root -q` prints for an entry function's value, and the exit status it gives |
| `execute` | `run`, `run_source`, `run_program`, `run_and_quit`, `load`, `process_line`, `split_call`, `arguments`: `.x`, `.L` and `ProcessLine` |
| `runtime` | what a ROOT translation imports: pycxx's runtime, `ROOT` (`runtime.root`), ROOT's meanings of `delete`, placement `new`, setters, `typeid`, `Printf` and kin (`runtime.adapters`), `TestSupport` and `ROOT_EXPECT_ERROR` (`runtime.diags`) and the ntuple batch fill (`runtime.ntuplefill`) |
| `interactive` | the home of the CINT/cling-compatible prompt (`xrdroot root`), built on `execute` |

```python
from xrdroot.pycint import process_line, translate

print(translate('void hello() { TH1F h("h", "h", 10, 0, 1); printf("%d\\n", 7 / 2); }'))
process_line(".x hsimple.C")
process_line("gRandom->Gaus(0, 1)")
```

## The former name

`xrdroot.cint` is kept as a compatibility shim in this component: `translate`,
`translate_file`, `translation`, `Translation`, `Refusal`, `MacroError`,
`CintError`, `Where`, `run`, `load` and `process_line`, and
`xrdroot.cint.execute`'s public functions, are the very objects of
`xrdroot.pycint`. `xrdroot.cint.runtime` is what translations kept from before
the split import, and `xrdroot.cint.runtime.objects` keeps old pickles loading.
Nothing new goes there.

## What remains ROOT-coupled here

Everything ROOT-specific about translating and running a macro lives here and
nowhere in pycxx: the dialect's data, the hooks for ROOT's loops and strings,
cling's printing of a returned value, the null-pointer convention RooFit's
printing follows while a macro runs, and ROOT's test-support diagnostics.
