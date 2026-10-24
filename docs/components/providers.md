# Built-in providers and adapters

Prefer plain engine results, downward imports or an explicit callable before
adding an extension point. A provider is appropriate when an established lower
API must dispatch to a higher implementation while retaining its public
behavior.

## Registry and lifetime

Core's `xrdroot.plugins.PROVIDERS` is a literal table:

```python
PROVIDERS = {
    ("display.plot", "default"): ("graphics", "xrdroot.plot", "plot"),
}
```

Each entry records an extension point and key, the owning component, module and
attribute. `lookup(point, key="default")` returns a declared callable or `None`
for an absent entry. `require(...)` refuses an absent registration with a clear
full sentence. `keys(point)` reads declared keys without loading targets.
`lookup_loaded(...)` reads a target only if its declared module is already in
`sys.modules`; it never imports a module for fallback discovery.

Resolution does not cache callable objects, so replacing a module or mutable
registry remains observable at the existing lifetime boundary. This is an
internal built-in table: there is no distribution lookup, entry-point scan or
component installation prompt. Optional third-party libraries still use their
existing extras and helpful refusals.

## Prepared dispatch sites

| Point | Lower caller | Provider and behavior |
|---|---|---|
| `display.plot` | core `display` | graphics `plot.plot` |
| `display.html` | core `display` | graphics `plot.notebook.figure`; absent entry falls back to escaped repr |
| `io.workspace` | I/O `file` | RooFit `roofit.io.workspace.read_workspace` for `RooWorkspace` |
| `canvas.dress` | I/O `kinds` | graphics canvas classes for the declared canvas/pad/primitive keys |
| `canvas.streamer` | I/O `interp` | graphics `canvas.streamer.streamer_for` for `TCanvas` and `TQObject` |
| `canvas.streamer.custom` | I/O `interp` | already-loaded streamer registry, preserving custom additions and deletions |
| `canvas.classes` | I/O `kinds` | already-loaded `canvas.class_registry` for public registry mutations |
| `io.layouts` | I/O `interp` | HistFactory `histfactory.layouts.described` for declared layout keys |
| `workspace.maker` | RooFit workspace reader | RooStats `workspaceio.make_model_config` |
| `session.preloaded` | ROOT-facing `session` | app `apinamespace.preloaded`, preserving live public facade overrides |

The registry's literal keys are authoritative; this table groups class families
rather than granting wildcard registration. File layouts precede built-in
layouts. Public registry access still yields actual shared objects. After
`CLASSES` materialization its dictionary alone governs later mutations, so a
deleted canvas entry is not silently resurrected by fallback dispatch.

## Contract rules

Each allowed dispatch has a separate extension record containing the exact
caller module, point, provider component and `module:attribute` target. The
component checker validates both declarations and calls, including imported
function aliases. A dynamic key is checked against every possible target at
that literal point. Dynamic points or unlisted targets remain violations.

An allowed provider does not add the higher component to the lower caller's
ordinary direct-import graph or isolation closure. Fresh isolated engine tests
use a fake/downward provider where needed; integration tests exercise the real
higher provider separately. Loading graphics through a provider during a core
isolation check should still be blocked.

Explicit C++ type resolvers and RVec/value registrations follow the same
ownership principle: the higher adapter installs its behavior; generic code
accepts that dependency without discovering or importing its caller. Source
review and tests must cover old/new object identity, mutation, deletion and
cold imports, not just successful lookup.
