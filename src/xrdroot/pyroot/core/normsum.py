"""``TF1NormSum``: functions normalised over their range, summed with coefficients.

``TF1NormSum(f1, f2, n1, n2)`` is ``n1 f1/∫f1 + n2 f2/∫f2``, times a scale:
a ``TF1`` made of it fits the coefficients - the yields - and the shapes'
parameters together. As ROOT's, it keeps copies of the functions, fixes the
parameter called ``Constant`` of each to one and leaves it out of its own,
normalises each over the first's range (``TF1::SetNormalized``), and numbers
its parameters the coefficients first, then each function's others, names
that repeat suffixed by ``FixDuplicateNames``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["TF1NormSum"]


def _arguments(args: tuple[Any, ...]) -> tuple[list[Any], list[float], float]:
    """The functions, coefficients and scale of any constructor: two or three functions and
    their coefficients, or lists of them."""
    if isinstance(args[0], (list, tuple)) or hasattr(args[0], "size"):
        rest = [float(c) for c in args[1]] + [float(a) for a in args[2:3]]
        functions = list(args[0])
    else:
        functions = [a for a in args if hasattr(a, "GetNpar")]
        rest = [float(a) for a in args if not hasattr(a, "GetNpar")]
    count = len(functions)
    coeffs = (rest[:count] + [1.0] * count)[:count]
    return functions, coeffs, [*rest[count:], 1.0][0]


def _unique(names: list[str]) -> list[str]:
    """``FixDuplicateNames``: a name met again is suffixed ``1``, ``2``... as often as it is."""
    seen: dict[str, int] = {}
    found = []
    for name in names:
        if names.count(name) > 1:
            seen[name] = seen.get(name, 0) + 1
            found.append(f"{name}{seen[name]}")
        else:
            found.append(name)
    return found


def _normalised(f: Any, constant: int, low: float, high: float) -> None:
    """A copy made a shape: its ``Constant`` held at one, over the sum's range, normalised."""
    if constant >= 0:
        f.FixParameter(constant, 1.0)
    f.SetRange(low, high)
    f.SetNormalized(True)
    f.Update()


class TF1NormSum:
    """A normalised sum of functions, to make a ``TF1`` of: ``TF1("f", sum, xmin, xmax, n)``."""

    def __init__(self, *args: Any) -> None:
        functions, self._coeffs, self._scale = _arguments(args)
        self._functions = [f.Clone() for f in functions]
        self._constant = [f.GetParNumber("Constant") for f in self._functions]
        low, high = self._functions[0].GetXmin(), self._functions[0].GetXmax()
        for f, at in zip(self._functions, self._constant, strict=True):
            _normalised(f, at, low, high)
        names = [f"Coeff{i}" for i in range(len(self._functions))]
        self._names = names + _unique([name for _, _, name in self._shapes()])

    def _shapes(self) -> list[tuple[Any, int, str]]:
        """Each function's parameters but its ``Constant``: the function, the number, the name."""
        return [(f, i, f.GetParName(i))
                for f, at in zip(self._functions, self._constant, strict=True)
                for i in range(f.GetNpar()) if i != at]  # fmt: skip

    def GetNpar(self) -> int:
        return len(self._names)

    def GetParName(self, i: int) -> str:
        return self._names[int(i)]

    def GetParameters(self) -> Any:
        """The coefficients, then each function's parameters but its ``Constant``."""
        from ..stl import std

        values = list(self._coeffs) + [f.GetParameter(i) for f, i, _ in self._shapes()]
        made = std.vector["double"]()
        for value in values:
            made.push_back(value)
        return made

    def SetParameters(self, params: Any) -> None:
        """The coefficients and the functions' parameters from ``params``, each function's
        integral redone if its parameters changed."""
        count = len(self._functions)
        self._coeffs = [float(params[i]) for i in range(count)]
        at = count
        for f, constant in zip(self._functions, self._constant, strict=True):
            changed = False
            for i in range(f.GetNpar()):
                if i != constant:
                    changed |= f.GetParameter(i) != float(params[at])
                    f.SetParameter(i, float(params[at]))
                    at += 1
            if changed:
                f.Update()

    def __call__(self, x: Any, params: Any = None) -> Any:
        """The scaled sum of the normalised functions at ``x`` - each point, for arrays."""
        if params is not None:
            self.SetParameters(params)
        points = np.asarray(x[0], dtype=np.float64)
        total: Any = 0.0
        for coeff, f in zip(self._coeffs, self._functions, strict=True):
            total = total + coeff * f._xrd.evaluate(np.atleast_1d(points))
        found = self._scale * total
        return found if points.ndim else float(found[0])
