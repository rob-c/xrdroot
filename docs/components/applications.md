---
title: "applications"
kind: "architecture"
summary: "PyROOT owns `xrdroot.pyroot` and `session`."
tags: [architecture, docs]
created: "2026-11-23T14:08:15.000Z"
updated: "2026-11-23T14:08:15.000Z"
---
# ROOT adapters and the application

## pyroot

PyROOT owns `xrdroot.pyroot` and `session`. It may import all thirteen engine
components and the C++ translator, [pycxx](pycxx.md). It supplies ROOT's class
names, namespaces, globals and public return objects over the engine
implementation. Running C++ - `ProcessLine`, `.x`, `gInterpreter.Declare`,
RDataFrame's C++ expressions - is [pycint](pycint.md)'s, which is built on this
namespace and reached through the `interpreter.*` providers.

```python
from xrdroot import pyroot as ROOT

values = ROOT.std.vector["float"]([1., 2.])
assert list(values) == [1., 2.]
```

Generic standard-library behavior belongs in `cppstd`. ROOT typedefs, opaque
ROOT class resolution, RVec factories, ROOT units and wrapper construction stay
in explicit adapters. A generic container receives a resolver or registered
factory; it never imports the ROOT namespace to discover its environment.
Compatibility imports must retain one implementation and shared registries,
stream objects, manipulators and random state.

The translator is ROOT-agnostic: ROOT's dialect of it is pycint's. It is not a
C++ compiler, and its native fixtures establish only their recorded semantics
rather than a C++ ABI guarantee. New PyROOT areas must remain present in
`pyroot.SUBMODULES` so the public namespace includes them.

Wrappers import `file`, `writer` and `wupdate` directly for I/O. Returning an
engine object or plain arrays to a wrapper is preferred to making a lower
engine import the ROOT namespace. Existing public ROOT classes are documented
in the [ROOT API guide](../root.md).

## cppyy

cppyy owns `xrdroot.cppyy`: cppyy's interface - `cppdef`, `include`,
`load_library`, `gbl`, `addressof`, `bind_object`, `nullptr`, `ll`, `py` -
over [pycint](pycint.md), ROOT's dialect of the translator, so that
`cppyy.gbl.X` is the object `ROOT.X` is. It depends on cppstd, pyroot and
pycint; only app depends on it. `xrdroot run` and the suite's shim make
`import cppyy` this module through `xrdroot.cli.shims` when no real cppyy is
installed, and only when a script imports it: `import ROOT` never does.

```python
import xrdroot.cppyy as cppyy

cppyy.cppdef("struct Point { int x = 2; };")
assert cppyy.gbl.Point().x == 2
```

## app

App owns `xrdroot/__init__.py`, `cli`, `__main__` and the public application
namespace builder `apinamespace`. It may import every other
component. Session preloading reaches that builder through the declared `session.preloaded`
provider, preserving the live public namespace without a hidden module lookup.
The distribution and command-line entry point remain unchanged:

```bash
xrdroot --help
xrdroot run analysis.C
```

The top-level facade resolves public exports lazily while preserving public
names, object identity, `__all__`, `dir()` and extension-loading behavior. Its
initial bootstrap must be inert: Python runs it before importing any
`xrdroot.cppstd` submodule. Engine, ROOT-namespace and transport imports belong
in requested export resolution, not that bootstrap.

Isolation permits this unavoidable parent-package bootstrap only. It does not
permit other app modules or resolving a higher export from a lower component.
Applications and users can keep public imports such as `from xrdroot import
Histogram`; internal engines use the owning module so their dependencies remain
visible to the contract.
