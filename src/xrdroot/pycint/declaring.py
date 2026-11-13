"""What a unit of C++ declares at its top, typed as cling prints the types.

An interactive prompt translates everything typed so far as one unit, and
prints an expression's value as cling does - ``(int) 4``, ``(TH1F *)
0x...`` - which needs the type the translator worked out, not the Python
value's: an ``int`` and a ``long long`` are one Python ``int``. The
translation runs in ROOT's dialect, as every macro here does, so ROOT's
typedefs and names mean at the prompt what they mean in a macro.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..pycxx.dialect import using
from ..pycxx.emit_module import Translator
from ..pycxx.parse_decl import parse
from ..pycxx.preprocessor import preprocess
from ..pycxx.program import Program
from ..pycxx.runtime.namespace import DECLARED
from ..pycxx.translation import Translation
from .dialect import ROOT as DIALECT

__all__ = ["translate_declaring", "forget"]


def translate_declaring(
    source: str, file: str = "<macro>", directories: list[Path] | None = None
) -> tuple[Translation, dict[str, tuple[str, str | None]]]:
    """``source`` translated, with each variable it declares at its top: Python name and C++ type.

    Each type is spelled as cling prints it, ``const char *`` or
    ``std::vector<int>``, or is ``None`` where the translator does not know it
    (``auto`` of a call into ROOT).
    """
    with using(DIALECT):
        tokens, included = preprocess(source, file, directories)
        unit = parse(tokens)
        program = Program(unit)
        translator = Translator(program, file, Path(file).stem)
        out = translator.translate()
        declared: dict[str, tuple[str, str | None]] = {}
        for name in program.globals:
            symbol = translator.lookup(name)
            assert symbol is not None  # every global is declared before any is translated
            declared[name] = symbol.py, _spelled(symbol.ctype)
    made = Translation(out.text(), file, out.source_map(), translator.entry,
                       [str(path) for path in included], unit.unnamed is not None,
                       translator.entry_returns)  # fmt: skip
    return made, declared


def _spelled(ctype: Any) -> str | None:
    """A C++ type as cling prints it: ``const char *``, ``std::vector<double>``, ``int[3]``."""
    if ctype is None or ctype.is_auto:
        return None
    const = "const " if ctype.const else ""
    stars = f" {'*' * ctype.pointer}" if ctype.pointer else ""
    dims = "".join(f"[{'' if dim is None else dim}]" for dim in ctype.dims)
    return f"{const}{_named(ctype)}{stars}{dims}"


def _named(ctype: Any) -> str:
    """A type's name with its template arguments: ``std::map<int, std::string>``."""
    # A non-type argument, ``std::array<int, 3>``'s ``3``, is a literal: its value is its spelling.
    args = [_spelled(arg) if hasattr(arg, "pointer") else str(getattr(arg, "value", arg))
            for arg in ctype.args]  # fmt: skip
    name: str = ctype.name
    return name + (f"<{', '.join(map(str, args))}>" if args else "")


def forget(names: Sequence[str]) -> None:
    """``.U file``: what an unloaded macro declared is no longer found by lines run later."""
    for name in names:
        DECLARED.pop(name, None)
