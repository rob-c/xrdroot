"""Does a ROOT data object cross the file boundary in both directions?

The question is not "is the class implemented" but "does an object of it survive being
written by one side and read by the other". For each class, two directions are tried in
separate processes:

* ``write`` - xrdroot writes a default-constructed object, then ROOT opens the file and
  says which class the key holds;
* ``read``  - ROOT writes the same thing, then xrdroot opens the file and says the same.

A class is supported only when both directions agree on the class name, because a reader
that silently produces nothing passes a laxer check.

    python -m tools.io_matrix --oracle-prefix ~/.local/root-oracle --classes TH1F,TGraph

The report is JSON plus a table, written even when every direction fails: the record is the
point. The exit status is 1 when a class does not cross, so a harness can gate on it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: What a child process is asked to do, under whichever interpreter is under test.
#: ``sys.argv`` is ``mode name path`` and the answer is one JSON object on stdout.
CHILD = r"""
import json
import sys

mode, name, path = sys.argv[1], sys.argv[2], sys.argv[3]
answer = {"ok": False, "class": None, "why": None}
try:
    import xrdroot.pyroot as ROOT
except ImportError:  # under ROOT itself the namespace is plain ROOT
    import ROOT  # type: ignore[no-redef]

try:
    if mode == "write":
        obj = getattr(ROOT, name)()
        handle = ROOT.TFile.Open(path, "RECREATE")
        obj.Write("obj")
        handle.Close()
    else:
        handle = ROOT.TFile.Open(path, "READ")
        obj = handle.Get("obj")
        answer["class"] = obj.ClassName() if obj is not None else None
        handle.Close()
    answer["ok"] = True
except Exception as reason:  # noqa: BLE001 - the refusal is the finding, not a crash
    answer["why"] = f"{type(reason).__name__}: {reason}"
print(json.dumps(answer))
"""

#: The classes asked for when none are given: what every ROOT file leans on, from the base
#: object upward, so that a failure at the bottom explains the ones above it.
DEFAULT_CLASSES = (
    "TNamed",
    "TObjString",
    "TArrayI",
    "TArrayD",
    "TH1F",
    "TH2F",
    "TProfile",
    "TGraph",
    "TGraphErrors",
    "TLorentzVector",
    "TMatrixD",
    "TRandom3",
)


#: Classes ROOT reports under a template spelling: ``TMatrixD`` is a typedef of
#: ``TMatrixT<double>``, so a key holding one is the same class by another name.
ALIASES = {
    "TMatrixD": "TMatrixT<double>",
    "TMatrixF": "TMatrixT<float>",
    "TMatrixDSym": "TMatrixTSym<double>",
    "TMatrixDSparse": "TMatrixTSparse<double>",
}


@dataclass
class Direction:
    """One attempt: what it was, whether it worked, and why not when it did not."""

    #: ``write`` (xrdroot writes, ROOT reads) or ``read`` (ROOT writes, xrdroot reads).
    mode: str
    ok: bool
    seen: str | None = None
    why: str | None = None


@dataclass
class Row:
    """One class's two directions."""

    name: str
    directions: dict[str, Direction] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        """Both directions agreed on the class - the only thing that counts."""
        return bool(self.directions) and all(
            direction.ok and direction.seen == self.expected
            for direction in self.directions.values()
        )

    @property
    def expected(self) -> str:
        """The name ROOT itself reports for this class, which a typedef changes."""
        return ALIASES.get(self.name, self.name)


def _interpreter(side: str, oracle_prefix: str | None) -> str:
    """The interpreter that runs one side: this one, or the oracle's PyROOT."""
    if side == "xrdroot":
        return sys.executable
    if oracle_prefix is None:
        raise SystemExit("the ROOT side needs --oracle-prefix (an installation with ROOT)")
    return str(Path(oracle_prefix) / "bin" / "python")


def _ask(python: str, cwd: Path, mode: str, name: str, path: Path) -> Direction:
    """Run one attempt in its own process, so that neither side's imports touch the other."""
    child = cwd / "io_matrix_child.py"
    child.write_text(CHILD)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if python == sys.executable:
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    try:
        done = subprocess.run(
            [python, str(child), mode, name, str(path)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return Direction(mode, False, why="timeout after 120 s")
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("{"):
            answer = json.loads(line)
            return Direction(mode, bool(answer["ok"]), answer["class"], answer["why"])
    return Direction(mode, False, why=(done.stderr.strip().splitlines() or ["no output"])[-1])


def measure(names: list[str], oracle_prefix: str | None, out: Path) -> list[Row]:
    """Every class's two directions, the object files left in ``out`` for inspection."""
    out.mkdir(parents=True, exist_ok=True)
    here = _interpreter("xrdroot", oracle_prefix)
    there = _interpreter("root", oracle_prefix)
    rows = []
    for name in names:
        row = Row(name)
        written = out / f"xrdroot-{name}.root"
        made = _ask(here, out, "write", name, written)
        row.directions["write"] = (
            _ask(there, out, "read", name, written)
            if made.ok
            else Direction("write", False, why=made.why)
        )
        elsewhere = out / f"root-{name}.root"
        _ask(there, out, "write", name, elsewhere)
        row.directions["read"] = _ask(here, out, "read", name, elsewhere)
        rows.append(row)
    return rows


def table(rows: list[Row]) -> str:
    """The report a person reads: one line per class, then the count."""
    lines = [f"{'class':18s} {'xrdroot->ROOT':>14s} {'ROOT->xrdroot':>14s}  supported"]
    for row in rows:
        cells = []
        for mode in ("write", "read"):
            direction = row.directions[mode]
            if not direction.ok:
                cells.append("FAIL")
            elif direction.seen != row.expected:
                cells.append(f"saw {direction.seen}")
            else:
                cells.append("PASS")
        lines.append(f"{row.name:18s} {cells[0]:>14s} {cells[1]:>14s}  {row.supported}")
    supported = sum(1 for row in rows if row.supported)
    lines.append(f"\n{supported} of {len(rows)} classes cross the boundary both ways")
    return "\n".join(lines)


def report(rows: list[Row]) -> dict[str, object]:
    """The same finding as a machine reads it, so a run directory has the record."""
    return {
        "classes": {
            row.name: {
                "supported": row.supported,
                "directions": {
                    mode: {"ok": direction.ok, "seen": direction.seen, "why": direction.why}
                    for mode, direction in row.directions.items()
                },
            }
            for row in rows
        }
    }


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, measure, print the table, write the JSON, return the status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oracle-prefix", help="a ROOT installation to compare against")
    parser.add_argument("--classes", help="comma-separated names; default is the base set")
    parser.add_argument("--out", help="where the files and the report go")
    arguments = parser.parse_args(argv)
    names = (
        [name.strip() for name in arguments.classes.split(",") if name.strip()]
        if arguments.classes
        else list(DEFAULT_CLASSES)
    )
    out = Path(arguments.out) if arguments.out else Path(tempfile.mkdtemp(prefix="io-matrix-"))
    rows = measure(names, arguments.oracle_prefix, out)
    print(table(rows))
    written = out / "io-matrix.json"
    written.write_text(json.dumps(report(rows), indent=1) + "\n")
    print(f"report: {written}")
    return 0 if all(row.supported for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
