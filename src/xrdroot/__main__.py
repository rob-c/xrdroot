"""``python -m xrdroot``: the same as the ``xrdroot`` command."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
