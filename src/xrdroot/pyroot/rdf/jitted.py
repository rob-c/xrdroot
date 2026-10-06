"""A frame's string expression that is more C++ than an expression: translated, then called.

xrdroot's frame evaluates a string - ``"pt * pt"``, ``"Sum(jet_pt > 30)"`` -
over whole batches; ROOT compiles it as the body of a C++ function of the
columns it names, so a string may also call ``gRandom->Gaus()``, a
function a script declared, or be statements ending in a ``return``. What
the batch evaluator refuses is made that function here: the columns the
text names become its parameters, of their C++ types, the text its body,
and :mod:`xrdroot.cint` translates it, to be called once per entry, as
ROOT calls it.
"""

from __future__ import annotations

import re
from typing import Any

from ...errors import UnsupportedFeatureError

__all__ = ["JIT_METHODS", "jitted", "retried", "translatable"]

#: The frame methods whose string a translation stands in for: where the string is.
JIT_METHODS = {"Filter": 0, "Define": 1, "Redefine": 1}

#: A name a C++ expression uses: a word not after a ``.``, ``->`` or ``::``.
NAME = re.compile(r"(?<![\w.:>])([A-Za-z_]\w*)")


def translatable(why: Exception) -> bool:
    """Is ``why`` the batch evaluator refusing C++, which a translation could still run?"""
    from ...formula.errors import FormulaError

    return isinstance(why, (FormulaError, UnsupportedFeatureError))


def _body(text: str) -> str:
    """The function's body: statements as they are - the last one ended, as ROOT ends it -
    or an expression, returned."""
    stated = text.strip()
    if not re.search(r"\breturn\b", stated):
        return f"return {stated};"
    return stated if stated.endswith((";", "}")) else stated + ";"


def jitted(frame: Any, text: str) -> tuple[Any, list[str]]:
    """The C++ ``text`` as a function of the frame's columns it names, and those columns."""
    from ...cint.execute import run_source

    used = list(dict.fromkeys(n for n in NAME.findall(text) if frame.HasColumn(n)))
    parameters = ", ".join(f"{frame.GetColumnType(name)} {name}" for name in used)
    source = f"auto rdf_jitted({parameters}) {{\n{_body(text)}\n}}\n"
    try:
        made = run_source(source, "<RDataFrame>", call=False)
    except Exception as why:
        raise UnsupportedFeatureError(
            f"the RDataFrame expression {text!r} is not supported: xrdroot's frame could not "
            f"evaluate it a batch at a time, and as a C++ function of its columns it failed "
            f"too - {why}") from why
    return made["rdf_jitted"], used


def retried(name: str, frame: Any, given: list[Any], why: Exception) -> list[Any] | None:
    """The method's arguments with its string made a translated function of its columns -
    or ``None`` when the method, the string or the refusal is not one a translation helps."""
    from .entrywise import ENTRY, entrywise

    at = JIT_METHODS.get(name)
    if at is None or at >= len(given) or not isinstance(given[at], str) or not translatable(why):
        return None
    function, used = jitted(frame, given[at])
    made = entrywise(function, counted=not used)
    if name == "Filter":  # Filter(expression, name): a function's columns come before its name
        return [made, used or [ENTRY], *given[at + 1:at + 2]]
    return [*given[:at], made, used or [ENTRY]]
