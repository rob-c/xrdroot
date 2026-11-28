---
title: "TFoam::Write is three missing pieces, in order"
kind: "architecture"
summary: "TFoam::Write needs gDirectory (pyroot has none), object-reference writing (writer.py:876 emits only null), and FOAM's cell grid for the value-by-value file comparison — in that order."
tags: [tfoam, gdirectory, writer, pointers, persistency]
created: "2026-11-28T15:13:06.665Z"
updated: "2026-11-28T15:13:06.665Z"
---
`TFoam::Write` looks like one method and is three missing pieces, in this order: there is no `gDirectory` for it to write into, the writer can only emit a **null** pointer, and the tutorial's file is compared value by value against ROOT's own FOAM grid. Each was checked against the code, so the ordering is not a guess.

```bash
rg -rn "gDirectory" src/xrdroot/pyroot          # no output: no current-directory concept
rg -n "64" src/xrdroot/writer.py | rg -i pointer   # writer.py:876 - "a pointer that may be null"
rg -n "compare_root_files" tools/tutorials/compare.py   # key by key, value by value
```

```mermaid
flowchart LR
    A["foam_demo.C: FoamX->Write(\"FoamX\")"] --> B{"gDirectory exists?"}
    B -->|no| C["step 1: opening a TFile must make it ROOT's current directory"]
    C --> D["step 2: writer must write a referenced object, not only a null tag"]
    D --> E["step 3: FOAM's cell grid, so values match ROOT's file"]
    E --> F["foam_demo.C can PASS"]
```

## Step 1: `gDirectory`

`TObject::Write` writes into the current directory. xrdroot has no such thing - `rg "gDirectory" src/xrdroot/pyroot` finds nothing - so a tutorial that opens `TFile RootFile("foam_demo.root","RECREATE")` and then calls `FoamX->Write("FoamX")` has no route from the foam to that file. This is a pyroot-core change (a `TFile` becoming current on construction and on `cd()`), and it benefits every class that writes itself, not only `TFoam`.

## Step 2: a referenced object

The harvested `TFoam` layout is 40 elements, five of them pointers - `fMCMonit: TFoamMaxwt*`, `fHistEdg`/`fHistDbg: TObjArray*`, `fHistWt: TH1D*`, `fPseRan: TRandom*` - plus `TFoamCell** fCells` and `std::vector<Long_t> fCellsAct`. `src/xrdroot/writer.py:876` covers stype 64/69 as *"a pointer that may be null - the only kind left"*: it can write the null tag and stop there.

## Step 3: the values

`tools/tutorials/compare.py`'s `compare_root_files` compares **key by key and value by value**, so a correct layout still fails while the sampler is a binned density rather than FOAM's cells. Skipping ahead to step 3 without 1 and 2 produces a file ROOT decodes wrongly - worse than the full-sentence refusal `TFoam::Write` gives today.
