"""ROOT macros - C++ as Cling reads it - translated into Python, and run.

    >>> print(translate("void hello() { printf(\\"%d\\\\n\\", 7 / 2); }"))  # doctest: +SKIP
    >>> run("hsimple.C")                          # doctest: +SKIP
    >>> run("fit.C", args=(1000,))                # .x fit.C(1000)

A macro is translated, not interpreted: :func:`translate` gives Python that
reads like the C++ it came from, uses ROOT's names from
:mod:`xrdroot.pyroot` as ``ROOT.TH1F``, ``ROOT.gRandom``..., and gets the
C++ semantics Python lacks - integer division, ``printf``, ``cout``, cells
for variables whose address is taken - from :mod:`xrdroot.cint.runtime`.
The Python is meant to be read and kept.

What the translator cannot turn into Python that does the same thing it
refuses by name, at the C++ line, with a :class:`Refusal`; it never writes
a plausible guess. A translated macro that fails as it runs fails with a
:class:`MacroError` placed at the C++ line, through the translation's
source map.
"""

from __future__ import annotations

from pathlib import Path

from .errors import CintError, MacroError, Refusal, Where
from .translation import Translation, translation

__all__ = [
    "translate",
    "translate_file",
    "translation",
    "Translation",
    "CintError",
    "Refusal",
    "MacroError",
    "Where",
]


def translate(source: str, file: str = "<macro>") -> str:
    """The Python that the C++ macro ``source`` becomes; ``file`` names it in errors."""
    directories = [Path(file).parent] if file != "<macro>" else []
    return translation(source, file, directories).python


def translate_file(path: str | Path) -> str:
    """The Python the macro at ``path`` becomes, its local headers read in."""
    where = Path(path)
    return translate(where.read_text(encoding="utf-8", errors="replace"), str(where))
