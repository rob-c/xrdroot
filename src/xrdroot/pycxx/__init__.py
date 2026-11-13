"""C++ translated into Python that reads like it, and run: a translator that stands alone.

    >>> print(translate("int main() { printf(\\"%d\\\\n\\", 7 / 2); }"))  # doctest: +SKIP
    >>> run_source(open("hello.cxx").read(), "hello.cxx", call=False)["main"]()  # doctest: +SKIP

A program is translated, not interpreted: :func:`translate` gives Python
that reads like the C++ it came from and gets the C++ semantics Python lacks
- integer division, ``printf``, ``cout``, cells for variables whose address
is taken, GoogleTest - from :mod:`xrdroot.pycxx.runtime`, built on
:mod:`xrdroot.cppstd`. The Python is meant to be read and kept.

What the translator cannot turn into Python that does the same thing it
refuses by name, at the C++ line, with a :class:`Refusal`; it never writes a
plausible guess. A translated program that fails as it runs fails with a
:class:`MacroError` placed at the C++ line, through the translation's source
map (:mod:`.execute`).

A program written against a library larger than the standard one is C++ in
that library's :class:`~.dialect.Dialect`: its typedefs, its headers'
macros, the classes it names without declaring and the namespace they come
from at run time. Plain C++ is :data:`~.dialect.CXX`, the default.
"""

from __future__ import annotations

from pathlib import Path

from .dialect import CXX, Dialect, Prelude, active, using
from .errors import CxxError, MacroError, Refusal, Unsupported, Where
from .execute import run_line, run_program, run_source
from .translation import Translation, translation

__all__ = [
    "translate",
    "translate_file",
    "translation",
    "Translation",
    "run_source",
    "run_program",
    "run_line",
    "Dialect",
    "Prelude",
    "CXX",
    "active",
    "using",
    "CxxError",
    "Refusal",
    "MacroError",
    "Unsupported",
    "Where",
]


def translate(source: str, file: str = "<macro>", dialect: Dialect | None = None) -> str:
    """The Python that the C++ ``source`` becomes; ``file`` names it in errors."""
    directories = [Path(file).parent] if file != "<macro>" else []
    return translation(source, file, directories, dialect).python


def translate_file(path: str | Path, dialect: Dialect | None = None) -> str:
    """The Python the C++ file at ``path`` becomes, its local headers read in."""
    where = Path(path)
    return translate(where.read_text(encoding="utf-8", errors="replace"), str(where), dialect)
