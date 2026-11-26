---
title: "The 42 non-passing pyunittest programs, by first failing case"
kind: "discovery"
summary: "The 42 non-passing pyunittest programs triaged by first failing case: eight are one behaviour each, the cppyy block is feature work, and one program's first error hid three behaviours."
tags: [pyunittest, triage, goal, cppyy, pyroot]
created: "2026-11-26T00:15:57.546Z"
updated: "2026-11-26T00:15:57.546Z"
---
The 42 non-passing `pyunittest` programs at `c5aabb22` are each one test id, so triage is per program: the list below is the first failing case of each, which is what a worker needs to start. Two of them are no longer failing (`pyroot_pyz_pretty_printing` and, after the alias fix, nothing else), so treat it as a starting inventory rather than a live count.

```text
cheap, one behaviour each      test_tpython (std::any), pyroot_array_numpy_views (chr() ValueError),
                               rbdt_xgboost (int() of a version string), pyroot_pyz_stl_set (a set
                               into a branch), dataframe_snapshot_py (a missing RDataFrame method),
                               MetaClingTests (INT64_MIN sign), pyroot_numbadeclare (a signature not
                               refused), root_no_implicit_cppyy (cppyy not in sys.modules)
API surface                   test_api (6 cases), pyroot_memory (4 of 9), rentry_getptr,
                               template_instantiation, rfile_py, dataframe_datasetspec,
                               dataframe_merge_results (collection failures)
cppyy's own suite             test_conversions, test_cpp11features, test_crossinheritance,
                               test_doc_features, test_fragile, test_operators, test_overloads,
                               test_pythonization, test_stltypes, test_templates, test_pythonify,
                               test_advancedcpp, test_lowlevel, test_aclassloader, test_datatypes,
                               test_numba, test_regression
ML / RDF / RooFit bindings    ml_dataloader (13 of 91), pyroot_pyz_rdataframe_asnumpy
                               (Internal.RDF), pyroot_pyz_rdataframe_makenumpy, pyroot_roofit,
                               pyroot_pyz_sofie_keras_parser (TMVA::Experimental::SOFIE)
```

## How to read it

`cheap, one behaviour each` is where a single fix flips a whole program — `std::any`, a `chr()` range bug, an `int()` parse, a `set` handed to a branch, one missing `RDataFrame` method, an overflow sign, one refusal message, one lazy import. The `cppyy` block is a *different* kind of work: those programs exercise cppyy's semantics (templates, cross-inheritance, operators, overloads) and each will take a feature, not a line.

The first failing case is only the entry point: `pyroot_pyz_pretty_printing` looked like one `ProcInfo_t` gap and needed three behaviours (process figures, a container's brace text, a typed null) before the program passed. Read the program's whole log (`logs/pyunittest/<id>/run.log`) before pricing it.
