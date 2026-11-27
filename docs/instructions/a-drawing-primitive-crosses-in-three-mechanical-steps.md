---
title: "A drawing primitive crosses in three mechanical steps"
kind: "instruction"
summary: "Harvest the layout, append the generated INFOS block to winfo.py, add the name to writer.MEMBERED, then prove it with io_matrix — but remember the harvest dump carries no element size."
tags: [writer, winfo, harvest, primitives, io-matrix]
created: "2026-11-27T15:35:09.083Z"
updated: "2026-11-27T15:35:09.083Z"
---
A drawing primitive joins the file boundary in three mechanical steps — no new payload code: harvest its layout from a donor file, append the generated `INFOS.update(...)` block to `src/xrdroot/winfo.py`, and add its name to `MEMBERED` in `src/xrdroot/writer.py`. `TLine`, `TMarker`, `TEllipse` and `TArrow` went from "not something this writer puts in a ROOT file" to crossing both ways in one iteration that way.

```bash
# 1. a donor holding one of each, written so every class keeps its own description
python -c 'import ROOT; f=ROOT.TFile.Open("drawing.root","RECREATE"); f.WriteObject(ROOT.TArrow(),"o"); f.Close()'
# 2. harvest it
python -m tools.harvest_infos --root-prefix ~/.local/root-oracle drawing.root > infos.json
# 3. render the INFOS block (the generator lives in the iteration notes: kind by stype, size by TDataType)
# 4. add the names to writer.MEMBERED, then prove it
python -m tools.io_matrix --oracle-prefix ~/.local/root-oracle --classes TLine,TMarker,TEllipse,TArrow
```

## Two things the generator has to get right

- **`size` is not in the harvest dump.** It gives `stype`, `name`, `title`, lengths and max indices, but not the byte size, so the generator maps stype → `TDataType` size (`3`→4, `8`→8, `11`→1, `16`→8, …) and gives a `TString` 24.
- **A base element's extra field is the base's class version**, not its checksum: `('TStreamerBase', 'TAttLine', …, 0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,))` — version 2 from `INFOS['TAttLine'][1]`.

```mermaid
flowchart LR
    A["donor .root written by ROOT"] --> B["tools/harvest_infos"] --> C["infos.json"]
    C --> D["generator: element tuples"] --> E["INFOS.update(...) in winfo.py"]
    E --> F["name in writer.MEMBERED"] --> G["io_matrix: PASS both ways"]
```

## Which classes still need what

- `TLegend` has a layout in the donor but a **pointer member** (`fPrimitives`, stype 64), so it needs a payload that writes its list of primitives — not just a member dict.
- `TCanvas` and `TDatime` were **not in the donor's streamer info at all**, so neither has a layout yet.
- Read-back members nest per base (`{"TAttMarker": {"fMarkerStyle": 1}}`), and a test that looks for `fLineColor` on every class fails on `TMarker`, which has no line attributes — parametrise the key per class, as `tests/test_writer_tarray.py` now does.
