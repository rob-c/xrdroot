"""``ROOT.Numba.Declare``: a Python function made callable from C++ as ``Numba::name``.

ROOT compiles the function with numba and has cling compile a C++ wrapper
calling it, so that a macro, ``ProcessLine`` or an RDataFrame expression
calls ``Numba::f(x)``. C++ here is translated into Python, so the wrapper
is a Python callable put in the ``Numba`` namespace: it turns its arguments
into the C++ types declared (a ``float`` is ``float32``, an ``RVec<float>``
a ``float32`` array), calls what numba compiled with that signature - so
the arithmetic is the one ROOT's runs - and gives the result back as the
declared C++ type: a number, a ``bool``, an ``RVec``, a ``std::vector`` or
a ``std::array``. The function decorated comes back unchanged, carrying
``numba_func`` and the two wrappers' text, as ROOT's does.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError

__all__ = ["Numba"]

#: Each fundamental C++ type numba is given, by its NumPy type.
NUMBERS = {
    "float": np.float32,
    "double": np.float64,
    "int": np.int32,
    "unsigned int": np.uint32,
    "long": np.int64,
    "unsigned long": np.uint64,
    "bool": np.bool_,
}

#: ``RVecF`` and its kin: the element type of each short name of an ``RVec``.
RVEC_ALIASES = {
    "F": "float",
    "D": "double",
    "I": "int",
    "U": "unsigned int",
    "L": "long",
    "UL": "unsigned long",
    "B": "bool",
}

#: A container type, by its kind: ``RVec<float>``, ``std::vector<int>``, ``std::array<int, 3>``.
CONTAINER = re.compile(r"(?:ROOT::)?(?:VecOps::)?(RVec|vector|array)(\w*)(?:<(.+)>)?$")


class CxxType:
    """One C++ type of a declared signature: a number's, or a container of numbers'."""

    def __init__(self, spelled: str) -> None:
        text = re.sub(r"\bconst\b|&", "", spelled).strip().removeprefix("std::")
        found = CONTAINER.match(text)
        self.spelled = spelled
        self.container: str | None = found.group(1) if found else None
        self.element = self._element(text, found)
        if self.element not in NUMBERS:
            raise UnsupportedFeatureError(
                f"ROOT.Numba.Declare is given the type {spelled!r}; xrdroot's takes the "
                f"fundamental types {', '.join(NUMBERS)} and RVec, std::vector and std::array "
                f"of them - declare the function with those"
            )
        self.dtype = NUMBERS[self.element]

    @staticmethod
    def _element(text: str, found: re.Match[str] | None) -> str:
        if found is None:
            return text
        if found.group(3) is not None:
            return found.group(3).split(",")[0].strip()
        return RVEC_ALIASES.get(found.group(2), found.group(2))

    def numba(self, nb: Any) -> Any:
        """The type numba compiles the function for: an array of a container's elements."""
        scalar = nb.from_dtype(np.dtype(self.dtype))
        return scalar[:] if self.container else scalar

    def argument(self, value: Any) -> Any:
        """``value`` as C++ hands it over: an array of the elements, or one number."""
        if self.container:
            return np.ascontiguousarray(np.asarray(value), dtype=self.dtype)
        return self.dtype(value)

    def result(self, value: Any) -> Any:
        """``value`` as the C++ type, seen from Python: ``float``, ``int``, ``bool``, a vector."""
        if self.container:
            return self._made(np.asarray(value, dtype=self.dtype))
        number = self.dtype(value).item()
        return bool(number) if self.element == "bool" else number

    def _made(self, values: np.ndarray[Any, Any]) -> Any:
        from . import rdf, stl

        if self.container == "RVec":
            return rdf.RVec(self.element)(values)
        if self.container == "vector":
            return stl.std.vector[self.element](values.tolist())
        return stl.std.array[self.element, len(values)](values.tolist())


def _numba() -> Any:
    try:
        import numba
    except ImportError:
        raise UnsupportedFeatureError(
            "ROOT.Numba.Declare compiles the function with numba, which is not installed: "
            "pip install numba"
        ) from None
    return numba


class Wrapper:
    """``Numba::name``: the compiled function, called with C++'s types and answering in them."""

    def __init__(self, compiled: Any, inputs: list[CxxType], output: CxxType | None) -> None:
        self.compiled = compiled
        self.inputs = inputs
        self.output = output

    def __call__(self, *args: Any) -> Any:
        given = [kind.argument(value) for kind, value in zip(self.inputs, args, strict=True)]
        value = self.compiled(*given)
        return value if self.output is None else self.output.result(value)


def _wrapper_text(name: str, inputs: list[CxxType], output: CxxType | None) -> tuple[str, str]:
    """The text of the two wrappers ROOT writes, the C++ one and the Python one."""
    returned = output.spelled if output is not None else "auto"
    params = ", ".join(f"{kind.spelled} x_{i}" for i, kind in enumerate(inputs))
    names = ", ".join(f"x_{i}" for i in range(len(inputs)))
    body = f"{returned} {name}({params}) {{\n  return {name}_py({names});\n}}"
    cpp = f"namespace Numba {{\n{body}\n}}"
    python = f"def {name}_py({names}):\n    return {name}({names})\n"
    return cpp, python


class _Numba:
    """``ROOT.Numba``: the namespace the declared functions are put in."""

    def Declare(
        self, input_types: Sequence[str], return_type: str | None = None, name: str | None = None
    ) -> Callable[[Any], Any]:
        """``@ROOT.Numba.Declare(["float"], "float")``: the decorator that declares one."""
        inputs = [CxxType(str(each)) for each in input_types]
        output = CxxType(str(return_type)) if return_type is not None else None

        def declare(func: Any) -> Any:
            nb = _numba()
            arguments = [kind.numba(nb) for kind in inputs]
            if output is not None:
                compiled = nb.njit(output.numba(nb)(*arguments))(func)
            else:
                compiled = nb.njit(tuple(arguments))(func)
            called = str(name or func.__name__)
            func.numba_func = compiled
            func.__cpp_wrapper__, func.__py_wrapper__ = _wrapper_text(called, inputs, output)
            setattr(self, called, Wrapper(compiled, inputs, output))
            return func

        return declare


#: ``ROOT.Numba``.
Numba = _Numba()
