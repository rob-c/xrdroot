"""The C++ translator's former home, kept so that existing imports keep working.

The translator is now two packages: :mod:`xrdroot.pycxx`, which translates
and runs C++ on its own, and :mod:`xrdroot.pycint`, which runs ROOT's macros
in ROOT's dialect of it. Every name here is :mod:`xrdroot.pycint`'s own
object, re-exported; new code imports from there.
"""

from __future__ import annotations

from ..pycint import (
    CintError,
    MacroError,
    Refusal,
    Translation,
    Where,
    load,
    process_line,
    run,
    translate,
    translate_file,
    translation,
)

__all__ = [
    "run",
    "load",
    "process_line",
    "translate",
    "translate_file",
    "translation",
    "Translation",
    "CintError",
    "Refusal",
    "MacroError",
    "Where",
]
