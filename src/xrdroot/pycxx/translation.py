"""One program, from C++ text to Python text: preprocess, parse, look over, emit.

:func:`translation` is the pipeline, and a :class:`Translation` what comes
out of it - the Python, the map from its lines back to the C++ ones, the
name of the function that runs the program, and the local headers it read.
The whole pipeline runs in one :class:`~.dialect.Dialect`: the one given, or
the one active - plain C++ unless :func:`~.dialect.using` said otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dialect import Dialect, active, using
from .emit_module import Translator
from .parse_decl import parse
from .preprocessor import preprocess
from .program import Program

__all__ = ["Translation", "translation"]


@dataclass
class Translation:
    """A translated program: its Python, and what is needed to run it and report against it."""

    python: str
    file: str
    source_map: dict[int, tuple[str, int]] = field(default_factory=dict)
    entry: str | None = None
    included: list[str] = field(default_factory=list)
    #: Whether the program is an unnamed block, ``{ ... }``, which takes no arguments.
    unnamed: bool = False
    #: The entry's return type as the dialect spells it, or None when it prints none.
    returns: str | None = None
    #: What the dialect's ``parsed`` hook made of the unit - plain JSON, kept beside the
    #: translation so that a run from the cache tells the host what parsing would have.
    notes: Any = None


def translation(
    source: str,
    file: str = "<macro>",
    directories: list[Path] | None = None,
    dialect: Dialect | None = None,
) -> Translation:
    """``source`` - the text of the program ``file`` - translated into Python in ``dialect``,
    or else in the dialect active now."""
    with using(dialect or active()):
        tokens, included = preprocess(source, file, directories)
        unit = parse(tokens)
        notes = active().parsed(unit)
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
        notes,
    )
