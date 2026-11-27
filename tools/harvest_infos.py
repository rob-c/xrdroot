"""Dump every class description a ROOT file carries, in the shape a layout table needs.

`winfo.INFOS` is harvested from donor files, but PyROOT has no getter for a file's
`TStreamerInfo` list - `f.GetStreamerInfo()`, `f.GetListOfStreamerInfo()` and
`TStreamerInfo.BuildFromFile()` are all absent, and `f.Get("StreamerInfo")` returns null.
`TFile::GetStreamerInfoList()` does work, called from C++ through `gInterpreter`, so this
tool runs a small C++ block and prints one JSON object per class.

    python -m tools.harvest_infos --root-prefix ~/.local/root-oracle donor1.root donor2.root

Each class gives its name, checksum and class version; each element gives the streamer
type, member name, title, array length and dimension, the five max indices, and the type
name. That is every field `winfo`'s ``Element`` tuple is built from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

#: The dump itself. Written as C++ because the list is only reachable from there; it prints
#: one JSON object per line, which Python collects. ``%lld`` is used for the checksum because
#: it is a signed 32-bit value widened, and ``%d`` would print a negative one oddly.
DUMP = r"""
{
  TList *list = f->GetStreamerInfoList();
  if (!list) { printf("{\"error\": \"no StreamerInfo list\"}\n"); return; }
  TIter next(list);
  while (TStreamerInfo *si = (TStreamerInfo*)next()) {
    printf("{\"name\": \"%s\", \"checksum\": %lld, \"version\": %d, \"elements\": [",
           si->GetName(), (long long)si->GetCheckSum(), si->GetClassVersion());
    TIter nit(si->GetElements());
    int first = 1;
    while (TObject *o = nit()) {
      TStreamerElement *e = (TStreamerElement*)o;
      if (!first) printf(", ");
      first = 0;
      printf("{\"stype\": %d, \"name\": \"%s\", \"title\": \"%s\", \"array_length\": %d, "
             "\"array_dim\": %d, \"max_index\": [%d, %d, %d, %d, %d], \"type_name\": \"%s\"}",
             e->GetType(), e->GetName(), e->GetTitle(), e->GetArrayLength(),
             e->GetArrayDim(), e->GetMaxIndex(0), e->GetMaxIndex(1), e->GetMaxIndex(2),
             e->GetMaxIndex(3), e->GetMaxIndex(4), e->GetTypeName());
    }
    printf("]}\n");
  }
}
"""

#: What a child does: run the block with the file named in it. The block is a *statement*
#: block rather than a function because cling defines a new function fine but PyROOT's
#: facade does not see it, so ``ROOT.harvest(...)`` would not resolve.
CHILD = r"""
import sys

import ROOT

ROOT.gInterpreter.ProcessLine(sys.argv[1])
"""


def dump(path: Path, root_prefix: Path) -> list[dict[str, object]]:
    """Every class description in one donor file, as the objects this tool reports."""
    python = str(root_prefix / "bin" / "python")
    with tempfile.TemporaryDirectory(prefix="harvest-") as scratch:
        child = Path(scratch) / "harvest_child.py"
        child.write_text(CHILD)
        opened = f'TFile *f = TFile::Open("{path}"); if (!f) return; '
        block = DUMP
        done = subprocess.run(
            [python, str(child), f"{{ {opened}{block.strip().strip('{}').strip()}; f->Close(); }}"],
            capture_output=True,
            text=True,
            timeout=600,
        )
    found = []
    for line in done.stdout.splitlines():
        if line.startswith("{"):
            found.append(json.loads(line))
    if not found:
        raise SystemExit(f"nothing read from {path}:\n{done.stderr[-2000:]}")
    return found


def report(files: list[Path], root_prefix: Path) -> dict[str, object]:
    """The classes of every donor, merged: a class seen twice keeps the first description."""
    classes: dict[str, object] = {}
    for path in files:
        for entry in dump(path, root_prefix):
            classes.setdefault(str(entry["name"]), {**entry, "donor": path.name})
    return {"classes": classes}


def main(argv: list[str] | None = None) -> int:
    """Read each donor, print the report, and write it beside the first one when asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("donors", nargs="+", help="ROOT files whose descriptions to read")
    parser.add_argument(
        "--root-prefix",
        default=str(Path.home() / ".local" / "root-oracle"),
        help="an installation with ROOT (default: the project's oracle)",
    )
    parser.add_argument("--out", help="where to write the JSON; default is stdout")
    arguments = parser.parse_args(argv)
    found = report([Path(name) for name in arguments.donors], Path(arguments.root_prefix))
    text = json.dumps(found, indent=1) + "\n"
    if arguments.out:
        Path(arguments.out).write_text(text)
        print(f"report: {arguments.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
