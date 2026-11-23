---
title: "A ROOT macro file is judged byte for byte: read SavePrimitive and GetCppName, then cmp the oracle"
kind: "instruction"
summary: "To write an object as the C macro ROOT writes, take the member order and the variable-name encoding from ROOT's own SavePrimitive and GetCppName, then cmp against the 6.40 oracle before trusting it."
tags: [saveas, saveprimitive, macros, root, oracle]
created: "2026-11-23T18:11:10.552Z"
updated: "2026-11-23T18:11:10.552Z"
---
ROOT writes an object as a C macro through `SaveAs`, and the three facts that decide whether your copy of the file matches its are: which class has a `SavePrimitive`, what names the variables get, and which members ROOT's own defaults leave out. All three are read off ROOT's source and then confirmed against the 6.40 oracle before anything is written, because the tests that judge them pin byte counts and `EXPECT_DOUBLE_EQ` invariants rather than shapes.

```text
SaveAs("x.C")
  └── TObject::SaveAs            core/base/src/TObject.cxx
        ├── "{"  +  "//========= Macro generated from object: name/title"
        │           "//========= by ROOT version6.40.04"
        ├── obj->SavePrimitive(out, option)      <- per class
        │     TH1::SavePrimitive  hist/hist/src/TH1.cxx
        │       blank line, constructor, SetBinContent per non-empty cell,
        │       SetBinError per cell that has one, SetEntries, then only what
        │       ROOT's defaults leave out (directory/stats/option/fill/line/axes)
        │       variable = GetCppName(name + "__<count>")
        └── "}"
```

## The three pieces, and where each one lives

- **`TObject::SaveAs`** (`core/base/src/TObject.cxx`) sends a `.C`/`.cc` name to `SavePrimitive` wrapped in braces and two `//` comment lines; *any other name but a ROOT file* is a macro too — measured: `h.SaveAs("x.dat")` prints `C++ Macro file: x.dat has been generated`. This kit names only `.C`/`.cc` in `pyroot/core/objects.py::MACRO_SUFFIXES`; the rest of the fall-through is queue row `W0-193`.
- **`TH1::SavePrimitive`** (`hist/hist/src/TH1.cxx`) writes a blank line, the constructor, a `SetBinContent` per **non-empty** cell, a `SetBinError` per cell that has one, `SetEntries`, then only what ROOT's defaults leave out — `SetDirectory(nullptr)`/`SetStats(0)`/`SetOption`, fill, line colour as *channels* (`TColor::GetColor("#000099")` for 602, from `colors.saved`), three axes' fonts and offsets — then `Draw()`. Numbers are `f"{value:.16g}"` (`%.16g`, C++'s `setprecision(digits10+1)`), not Python's shortest repr.
- **Variable names** are `ROOT::TMetaUtils::GetCppName(histogram name + "__<count>")`: a 31-entry table that spells every character C++ would not take as two letters (`+`→`pL`, `<`→`lE`, space→`sP`, `[`→`oB`). That is why ROOT's macro variables look like `lEoBsPnOaTdAp...`; `pyroot/core/metautils.py` transcribes the table and `pyroot/core/hists.py::_save_name`/`_identifier` use it.

## How to check a change without guessing

Run the oracle on the same object and `cmp` the two files — that is what made round 83's histogram macro exact: `diff H1dump.C ours.C` empty for ROOT's own `test_TH1_SaveAs` fixture. For a *pad* macro the same check says the work is not done: `pyroot/graphics/macro.py` writes 1541 bytes where ROOT writes 2351 for the same canvas, because our line shapes differ (no `#ifdef __CLING__` header, `SetPoint` per point instead of one `std::vector`, every attribute written even at its default, `Draw("")` instead of `Draw("alp")`) — `testMapCppName`'s reference is `EXPECT_NEAR(FileSize, 6644, 200)` and ROOT's own file for that canvas is 6646 bytes and 93 lines, measured.

## What this costs if you skip it

The three ROOT tests that read these files (`gtest/hist/hist/testTH1sa`, `gtest/hist/hist/testMapCppName`, `gtest/tree/tree/testTChainSaveAsCxx`) judge a file's size or its text, so a plausible-looking macro fails: `testTH1sa` needed the exact 31 lines, and the missing `SavePrimitive` path was the wall for two rounds after the text formats worked.
