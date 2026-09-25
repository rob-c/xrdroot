"""How much of ROOT's tutorials the C++ translator turns into Python.

    $ python tools/cint_survey.py ROOT/tutorials
    $ python tools/cint_survey.py ROOT/tutorials --top 40 --list refused

Every ``.C``, ``.cxx`` and ``.cpp`` under the directory is translated with
:func:`xrdroot.cint.translate` and the Python it gives is compiled. Each file
is one of: *translated* (the Python compiles), *refused* (the translator
named a construct it will not translate - the refusals are grouped by what
they name, with the construct's wording kept and the specifics dropped), or
*crashed* (anything else, including Python that does not compile: a bug
here, never the macro's fault).
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

from xrdroot.cint import Refusal, translate

#: The file extensions ROOT's tutorials write macros with.
EXTENSIONS = (".C", ".cxx", ".cpp")

#: What is dropped from a refusal to group it with the others like it.
SPECIFICS = re.compile(r"'[^']*'|\"[^\"]*\"|\b\d+\b|\b[A-Za-z_]\w*(?:::\w+)*\(\)")


def reason(why: str) -> str:
    """A refusal's sentence with its particulars (names, numbers, quoted text) taken out."""
    return SPECIFICS.sub("…", why)[:110]


def survey(root: Path) -> tuple[list[Path], dict[str, list[tuple[Path, str]]]]:
    """Every macro under ``root``, and which of translated/refused/crashed each is."""
    files = sorted(p for p in root.rglob("*") if p.suffix in EXTENSIONS and p.is_file())
    outcome: dict[str, list[tuple[Path, str]]] = {"translated": [], "refused": [], "crashed": []}
    for path in files:
        kind, detail = _one(path)
        outcome[kind].append((path, detail))
    return files, outcome


def _one(path: Path) -> tuple[str, str]:
    text = path.read_bytes().decode("utf-8", "replace")
    try:
        python = translate(text, str(path))
        compile(python, str(path), "exec")
    except Refusal as why:
        return "refused", why.why
    except Exception as why:
        frame = traceback.extract_tb(why.__traceback__)[-1]
        return "crashed", f"{type(why).__name__}: {why} ({frame.name}:{frame.lineno})"
    return "translated", ""


def report(root: Path, top: int, listing: str | None) -> str:
    files, outcome = survey(root)
    total = len(files) or 1
    lines = [f"{len(files)} macros under {root}"]
    for kind in ("translated", "refused", "crashed"):
        count = len(outcome[kind])
        lines.append(f"  {kind:<11} {count:>4}  {100 * count / total:5.1f}%")
    for kind in ("refused", "crashed"):
        grouped = Counter(reason(detail) for _, detail in outcome[kind])
        if grouped:
            lines.append(f"\n{kind}, by reason:")
            lines.extend(f"  {n:>4}  {why}" for why, n in grouped.most_common(top))
    if listing:
        lines.append(f"\n{listing}:")
        lines.extend(f"  {path.relative_to(root)}: {detail}" for path, detail in outcome[listing])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tutorials", type=Path, help="ROOT's tutorials directory")
    parser.add_argument("--top", type=int, default=25, help="reasons to show of each kind")
    parser.add_argument(
        "--list",
        choices=("translated", "refused", "crashed"),
        help="also list every file of one kind",
    )
    args = parser.parse_args(argv)
    sys.setrecursionlimit(10_000)
    print(report(args.tutorials, args.top, args.list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
