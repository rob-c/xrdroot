---
title: "TRandom3 is four classes and needs a state accessor first"
kind: "gotcha"
summary: "TRandom3 needs four layouts (TRandom3, TRandom, TRandomEngine, TNamed) and a 624-word Mersenne state the pyroot object does not expose; add the accessor before the layout."
tags: [winfo, trandom3, writer, harvest, state]
created: "2026-11-27T02:40:53.609Z"
updated: "2026-11-27T02:40:53.609Z"
---
`TRandom3` looks like one class to add and is four: `TRandom3` (checksum 2185949371, v2), its base `TRandom` (235880363, v3), which itself has the bases `TNamed` and `ROOT::Math::TRandomEngine` (3688732995, v1). Only `TNamed` is already in `winfo.INFOS`. Worse, a correct record needs the engine's **624-element Mersenne state** (`fMt`, `unsigned int[624]`, plus `fCount624`), which xrdroot's `TRandom3` object does not expose — so adding the layout alone would write a file that reads back a *different stream*.

```bash
~/.local/root-oracle/bin/python -c "
import ROOT
f = ROOT.TFile.Open('/tmp/tr.root', 'RECREATE')
r = ROOT.TRandom3(7); r.Rndm(); r.Write('r'); f.Close()"
python -m tools.harvest_infos --out harvest-tr.json /tmp/tr.root
```

```text
TRandom3                  2185949371  v2  TRandom BASE, fMt uint[624], fCount624 int
TRandom                   235880363   v3  TNamed BASE, ROOT::Math::TRandomEngine BASE, fSeed uint
TNamed                    3753331260  v1  TObject BASE, fName TString, fTitle TString   (in INFOS)
ROOT::Math::TRandomEngine 3688732995  v1  (no members)
```

## The prerequisite, in order

1. Give xrdroot's `TRandom3` a way to hand out its state — the 624 words and the count — as ROOT streams them. Without it there is no row to build.
2. Then three `INFOS` entries, with `fMt` as a counted array element (streamer type 33, `kOffsetL + kUInt`, array length 624) and the two `TStreamerBase` elements carrying each base's checksum in max-index slot 1.
3. Accept with `python -m tools.io_matrix --oracle-prefix ~/.local/root-oracle --classes TRandom3` reading PASS in both columns.

```mermaid
flowchart LR
    A["TRandom3 object"] -->|"needs state"| B["fMt: 624 uint words + count"]
    B --> C["row: fMt, fCount624, fSeed, fName, fTitle"]
    C --> D["INFOS: TRandom3 + TRandom + TRandomEngine"]
    D --> E["ROOT reads the key: io_matrix PASS"]
    A -.->|"today: no state accessor"| F["blocked"]
```

A general note for the batch ahead: some classes are like `TMatrixD` — layout plus a row from public accessors — and some are like `TRandom3`, where the *object* must first expose the state the file holds. Check which kind it is before writing an entry.
