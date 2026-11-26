"""``ROOT/StringUtils.hxx``: ROOT's own text helpers, which the suite's tests call by name.

``ROOT::Split`` and ``ROOT::Join`` are Python's ``str.split`` and ``str.join`` written as
C++ functions, and ``ROOT::Round`` prints a value and its uncertainty with the significant
digits the error carries (``110.0#pm0.2``, and ``(1#pm2)*1e3`` once the value is large
enough to be factored). All three are ROOT's, not the standard library's, so they live
here beside the namespace rather than in :mod:`xrdroot.cppstd`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__ = ["Split", "Join", "Round", "CanConvertEnvValueToBool", "ConvertEnvValueToBool"]


def Split(text: Any, delimiters: Any, skipEmpty: bool = False) -> list[str]:
    """``ROOT::Split(str, delims[, skipEmpty])``: the text cut at every delimiter.

    The behaviour is ``str.split``'s, except that the delimiters are a set of characters
    rather than one string: ``"a,,c"`` split at ``","`` is ``["a", "", "c"]`` unless the
    empty pieces are skipped.
    """
    text, marks = str(text), str(delimiters)
    found: list[str] = []
    begin = 0
    while (end := _first_of(text, marks, begin)) is not None:
        if not skipEmpty or end > begin:
            found.append(text[begin:end])
        begin = end + 1
    if not skipEmpty or len(text) > begin:
        found.append(text[begin:])
    return found


def _first_of(text: str, marks: str, begin: int) -> int | None:
    """The first index at or after ``begin`` of one of ``marks``, or ``None``."""
    places = [text.find(mark, begin) for mark in marks]
    places = [place for place in places if place >= 0]
    return min(places) if places else None


def Join(separator: Any, strings: Sequence[Any]) -> str:
    """``ROOT::Join(sep, strings)``: the pieces with ``sep`` between them, as ``str.join``."""
    return str(separator).join(str(each) for each in strings)


#: ``round``'s halves go away from zero, as C++'s ``std::round`` does, not to even as
#: Python's own ``round`` does - ``ROOT::Round(24., 19.4)`` is ``24#pm19``.
def _rounded(value: float) -> float:
    """``std::round(value)``: the integer nearest ``value``, a half rounded away from zero.

    A value that rounds to nothing keeps its sign, as ``std::round(-0.4)`` is ``-0.0`` and
    ``std::ios`` then prints the minus.
    """
    nearest = float(math.floor(value + 0.5)) if value >= 0 else -float(math.floor(-value + 0.5))
    return -0.0 if nearest == 0 and math.copysign(1.0, value) < 0 else nearest


def Round(value: float, error: float, cutoff: int = 1, delim: str = "#pm") -> str:
    """``ROOT::Round(value, error[, cutoff[, delim]])``: the pair as ROOT prints it.

    The error is rounded to one or two significant digits - two when its first digit is at
    most ``cutoff``, and the number is then quoted to that digit - and a value large or
    small enough to be written as ``(a#pmb)*1e3`` is factored by a multiple of three.
    """
    if error <= 0.0:
        return _to_string(value)
    exponent = math.floor(math.log10(error))
    magnitude = math.pow(10.0, exponent)
    first = int(error / magnitude)
    if first <= cutoff:
        rescaled = error * math.pow(10.0, -exponent)
        if int(_rounded(rescaled * 10) / 10) <= cutoff:
            exponent -= 1
    elif cutoff == 0 and first == 9:
        exponent = _nines(exponent, error)
    factored = _factored(error)
    digits = 0 if exponent - factored >= 0 else factored - exponent
    scale = math.pow(10.0, -exponent)
    shown = math.pow(10.0, -exponent + factored)
    pair = f"{_rounded(value * scale) / shown:.{digits}f}"
    pair += f"{delim}{_rounded(error * scale) / shown:.{digits}f}"
    return f"({pair})*1e{factored}" if factored != 0 else pair


def _nines(exponent: int, error: float) -> int:
    """A single digit ``9`` that rounds up to ``10`` moves the exponent by one."""
    rounded = _rounded(error * math.pow(10.0, -exponent))
    up = math.floor(math.log10(rounded)) if rounded > 0 else 0
    return exponent + 1 if int(rounded / math.pow(10.0, up)) == 1 else exponent


def _factored(error: float) -> int:
    """The multiple of three the printed pair is scaled by: ``1e-9``, ``1e3``, or none."""
    if error <= 1e-3:
        return math.floor(math.log10(error) / 3) * 3
    # ``static_cast<int>`` truncates toward zero, so -0.2 is 0 and not -1: 0.24 is not
    # factored at all, as ROOT prints ``110.0#pm0.2``.
    return int(math.log10(error) / 3) * 3


def _to_string(value: float) -> str:
    """``std::to_string(double)``: six decimals, as C++ writes one."""
    return f"{value:.6f}"


def CanConvertEnvValueToBool(value: Any) -> bool:
    """``ROOT::FoundationUtils::CanConvertEnvValueToBool``: is this text a boolean?

    ROOT reads an environment variable as a flag when it says ``1``/``0``, ``on``/``off``
    or ``true``/``false`` in any case; anything else is not a flag at all.
    """
    return _flag(value) is not None


def ConvertEnvValueToBool(value: Any) -> bool:
    """``ROOT::FoundationUtils::ConvertEnvValueToBool``: the flag the text names."""
    flag = _flag(value)
    if flag is None:
        raise ValueError(f"{value!r} is not a boolean: it is not 1, 0, on, off, true or false")
    return flag


def _flag(value: Any) -> bool | None:
    """The boolean a flag's text names, or ``None`` when it names none of them."""
    text = str(value).lower()
    if text in ("1", "on", "true"):
        return True
    if text in ("0", "off", "false"):
        return False
    return None
