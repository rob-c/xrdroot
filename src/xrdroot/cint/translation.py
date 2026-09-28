"""One macro, from C++ text to Python text: preprocess, parse, look over, emit.

:func:`translation` is the pipeline, and a :class:`Translation` what comes
out of it - the Python, the map from its lines back to the C++ ones, the
name of the function that runs the macro, and the local headers it read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .emit_module import Translator
from .parse_decl import parse
from .preprocessor import preprocess
from .program import Program

__all__ = ["Translation", "translation"]


@dataclass
class Translation:
    """A translated macro: its Python, and what is needed to run it and report against it."""

    python: str
    file: str
    source_map: dict[int, tuple[str, int]] = field(default_factory=dict)
    entry: str | None = None
    included: list[str] = field(default_factory=list)
    #: Whether the macro is an unnamed one, ``{ ... }``, which takes no arguments.
    unnamed: bool = False
    #: The type the macro's function gives back, as cling names it; None for ``void``.
    returns: str | None = None


def translation(
    source: str, file: str = "<macro>", directories: list[Path] | None = None
) -> Translation:
    """``source`` - the text of the macro ``file`` - translated into Python."""
    tokens, included = preprocess(source, file, directories)
    unit = parse(tokens)
    program = Program(unit)
    stem = Path(file).stem if file != "<macro>" else "macro"
    translator = Translator(program, file, stem)
    out = translator.translate()
    return Translation(
        out.text(),
        file,
        out.source_map(),
        translator.entry,
        [str(path) for path in included],
        unit.unnamed is not None,
        translator.entry_returns,
    )
