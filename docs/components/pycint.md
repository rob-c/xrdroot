# ROOT's interpreter: pycint

`xrdroot.pycint` is the CINT-compatible face of xrdroot: ROOT's macros, read in
ROOT's dialect of C++ by [pycxx](pycxx.md) and run against the ROOT namespace
of [pyroot](applications.md#pyroot). It depends on pycxx and on pyroot and
everything below it; nothing below it imports it. pyroot's
`gROOT.ProcessLine`, `Macro`, `LoadMacro`, `gInterpreter.Declare`, the
session's `.x` and RDataFrame's C++ expressions reach it through the
`interpreter.*` providers of `xrdroot.plugins` (see [providers](providers.md)),
and so do `gSystem->Load`, `GetLibraries` and `CompileMacro`
(`interpreter.load_library`, `interpreter.loaded_libraries`,
`interpreter.compile_macro`) and the autoloading of a name `ROOT.<name>` does
not otherwise find (`interpreter.autoload`).

## Libraries

`xrdroot build` (`dictionary`, `library`, `executable`, `aclic`) writes a
library as a `.xrdlib` file: JSON holding the Python its sources translate
into, the source map, what it declares (top-level names, namespaces and their
members, qualified classes - its rootmap), the files it was built from
(relative to the library's directory when under it, so a build directory can
move) and a digest of their text and of the translator, which is what ACLiC's
"up to date" checks. Loading runs the Python once in a module of its own and
keeps its functions, classes and namespace objects in pycxx's `DECLARED`,
where every later translation's host looks. The ROOT dialect's `loaded` hook
(`Dialect.loaded`, called by pycxx's `run_source` and `run_program` once a
program's Python has run) binds a program's copy of a library's header back to
the library's objects, now and when a library is loaded later.

## What it owns

| Module | What it is |
|---|---|
| `dialect` | `ROOT`, the dialect: ROOT's typedefs (`Int_t`, `Double32_t`), what its class names look like, `ROOT/TestSupport.hxx`, the `R__` and `ClassDef` macros and `ROOT_VERSION`, `TMath`, `TString`, `TTreeReaderValue`/`Array`, `RVec` and its aliases, the methods ROOT writes through |
| `layouts` | the size, alignment, data size and polymorphism of each ROOT class xrdroot provides, on macOS ARM64, read from the 6.40.04 oracle: what `sizeof(TH1F)` is, and how a macro's class derived from `TObject` is laid out |
| `emit` | the hooks: `TString::Format` and `s.Form()`, loops over `gRandom` draws filled all at once, and ntuple loops filled in a batch |
| `fill_loops`, `tree_loops` | the analyses those loops are recognised by |
| `returned` | what `root -q` prints for an entry function's value, and the exit status it gives |
| `execute` | `run`, `run_source`, `run_program`, `run_and_quit`, `load`, `process_line`, `split_call`, `called`, `arguments`, `preload`: `.x`, `.L` and `ProcessLine`, ACLiC's `+`, `.L libFoo.so`, and the libraries `R__LOAD_LIBRARY` and `#pragma cling load` name |
| `library` | ROOT's build model as translations: a `.xrdlib` file (`manifest`), rootcling's and genreflex's dictionaries and libraries from sources (`build`, `selection`, `translate`), what those generators refuse, refused with their words and exit status (`checks`), ACLiC (`aclic`), and `gSystem->Load`, rootmap autoloading and binding a macro's copy of a library's header back to the library (`loading`) |
| `runtime` | what a ROOT translation imports: pycxx's runtime, `ROOT` (`runtime.root`), ROOT's meanings of `delete`, placement `new`, setters, `typeid`, `Printf` and kin (`runtime.adapters`), `TestSupport` and `ROOT_EXPECT_ERROR` (`runtime.diags`), the ntuple batch fill (`runtime.ntuplefill`) and `gApplication`, the `TRint` a `root` session is (`runtime.application`) |
| `interactive` | the home of the CINT/cling-compatible prompt (`xrdroot root`), built on `execute`: one translation unit for the session (`unit`), which `gROOT->ProcessLine` of C++ runs in too (`lines`), so what one line declares the next sees, and a macro run with `.x` sees the prompt's variables; `.undo` and `.I` |

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
