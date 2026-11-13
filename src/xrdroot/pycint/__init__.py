"""ROOT macros - C++ as Cling reads it - translated into Python, and run: the CINT of xrdroot.

    >>> print(translate("void hello() { printf(\\"%d\\\\n\\", 7 / 2); }"))  # doctest: +SKIP
    >>> run("hsimple.C")                          # doctest: +SKIP
    >>> run("fit.C", args=(1000,))                # .x fit.C(1000)

This is :mod:`xrdroot.pycxx`, the C++ translator, in ROOT's dialect
(:data:`~.dialect.ROOT`): a macro's undeclared names are ROOT's, from
:mod:`xrdroot.pyroot` as ``ROOT.TH1F``, ``ROOT.gRandom``...; ROOT's typedefs,
headers and ``R__`` macros are known; and its loops over trees and random
draws are written as ROOT's objects run them fastest. A macro is translated,
not interpreted: :func:`translate` gives Python that reads like the C++ it
came from, and the C++ semantics Python lacks - integer division, ``printf``,
``cout``, cells for variables whose address is taken - come from
:mod:`xrdroot.pycint.runtime`. The Python is meant to be read and kept.

What the translator cannot turn into Python that does the same thing it
refuses by name, at the C++ line, with a :class:`Refusal`; it never writes
a plausible guess. A translated macro that fails as it runs fails with a
:class:`MacroError` placed at the C++ line, through the translation's
source map. ``.x``, ``.L`` and ``gROOT->ProcessLine`` are :mod:`.execute`'s.
"""

from __future__ import annotations

from pathlib import Path

from ..pycxx.errors import CxxError, MacroError, Refusal, Where
from ..pycxx.translation import Translation
from ..pycxx.translation import translation as _translation
from .dialect import ROOT
from .execute import load, process_line, run

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
    "ROOT",
]

#: The base of what translating or running a macro raises: the C++ translator's own.
CintError = CxxError


def translation(
    source: str, file: str = "<macro>", directories: list[Path] | None = None
) -> Translation:
    """``source`` - the text of the macro ``file`` - translated into Python in ROOT's dialect."""
    return _translation(source, file, directories, ROOT)


def translate(source: str, file: str = "<macro>") -> str:
    """The Python that the C++ macro ``source`` becomes; ``file`` names it in errors."""
    directories = [Path(file).parent] if file != "<macro>" else []
    return translation(source, file, directories).python


def translate_file(path: str | Path) -> str:
    """The Python the macro at ``path`` becomes, its local headers read in."""
    where = Path(path)
    return translate(where.read_text(encoding="utf-8", errors="replace"), str(where))
