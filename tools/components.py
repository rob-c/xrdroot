"""Check internal boundaries while keeping xrdroot one installed Python project."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.component_contract.cli import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
