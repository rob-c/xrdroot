"""``RooWorkspace::factory``: RooFit's little language for building a model in one string.

    Gaussian::g(x[-10,10], m[0,-1,1], 1.5)      a class, by name, with its arguments
    x[-10,10]  m[0,-1,1]  c[5]                  a variable: range, value and range, constant
    1.5                                          a constant
    {a0, a1[0,1]}                                a list
    SUM::m(f[0.5,0,1]*sig, bkg)                  a sum; PROD::p(a, b|x) a product
    EXPR::e('x*y', x, y)   expr::f('a+b', a, b)  a density or a function of a formula
    prod::p(a,b)  sum::s(a,b)                    a product or sum of values

Arguments are split at the commas outside brackets, braces, parentheses and
quotes; each is built - recursively - into an object, created in the
workspace or found there by name. Every object made is imported into the
workspace silently, as the factory does.
"""

from __future__ import annotations

import re
from typing import Any

from ..errors import UnsupportedFeatureError as refuse
from .collections import RooArgList

__all__ = ["Factory", "split"]

#: ``name[...]``: a variable with its numbers.
VARIABLE = re.compile(r"^([A-Za-z_]\w*)\[([^\]]*)\]$")
#: ``Class::name(args)`` or ``Class(args)``.
CALL = re.compile(r"^([A-Za-z_$][\w:]*?)(?:::([A-Za-z_]\w*))?\((.*)\)$", re.S)
#: ``globCounter``: how many top-level objects have been named ``gobj<n>`` this session.
_GLOBAL = [0]
#: A number, as the factory reads one.
NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def split(text: str, separator: str = ",") -> list[str]:
    """``text`` split at each ``separator`` that is outside brackets and quotes."""
    parts: list[str] = []
    depth, quote, current = 0, "", ""
    for char in text:
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())
    return parts


class Factory:
    """What builds a factory expression into a workspace's objects."""

    def __init__(self, workspace: Any) -> None:
        self.w = workspace
        #: ``_autoNamePrefix``: what an object made without a name inside another is called -
        #: ``<outer>_<n>`` for the n-th argument of ``outer``, one more digit inside a list.
        self.prefix: list[str] = []

    def build_arg(self, owner: str, index: int, text: str) -> Any:
        """The ``index``-th argument of ``owner``, built under the name RooFit gives it."""
        self.prefix.append(f"{owner}_{index + 1}")
        try:
            return self.build(text)
        finally:
            self.prefix.pop()

    def keep(self, obj: Any) -> Any:
        """``obj`` imported, and the workspace's copy of it - what the factory hands back."""
        self.w.Import(obj, Silence=True)
        found = self.w.arg(obj.GetName()) if hasattr(obj, "servers") else None
        return obj if found is None else found

    def build(self, text: str) -> Any:
        text = text.strip()
        for method in (
            self._number,
            self._string,
            self._list,
            self._variable,
            self._call,
            self._existing,
        ):
            found = method(text)
            if found is not None:
                return found
        raise refuse(
            f"the factory cannot make sense of {text!r}: it is not a number, a variable, "
            "a list, or a class with arguments, and the workspace has nothing of that name"
        )

    def _number(self, text: str) -> Any:
        if not NUMBER.match(text):
            return None
        from .pdfs.basic import ref

        return ref(float(text))

    @staticmethod
    def _string(text: str) -> Any:
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            return text[1:-1]
        return None

    def _list(self, text: str) -> Any:
        if not (text.startswith("{") and text.endswith("}")):
            return None
        made = []
        for index, part in enumerate(split(text[1:-1])):
            if self.prefix:
                self.prefix.append(f"{self.prefix[-1]}{index + 1}")
            try:
                made.append(self.build(part))
            finally:
                if self.prefix:
                    self.prefix.pop()
        return RooArgList(made)

    def _variable(self, text: str) -> Any:
        found = VARIABLE.match(text)
        if found is None:
            return None
        name, numbers = found.group(1), found.group(2)
        existing = self.w.arg(name)
        if existing is not None:
            return existing
        if "=" in numbers or not numbers or not NUMBER.match(split(numbers)[0]):
            return self.keep(_category(name, numbers))
        from .variables import RooRealVar

        values = [float(v) for v in split(numbers)]
        return self.keep(RooRealVar(name, name, *values))

    def _existing(self, text: str) -> Any:
        return self.w.obj(text) if re.match(r"^[A-Za-z_]\w*$", text) else None

    def _call(self, text: str) -> Any:
        found = CALL.match(text)
        if found is None:
            return None
        kind, name, inner = found.group(1), found.group(2), found.group(3)
        from .factoryops import OPERATORS

        args = split(inner)
        if kind == "$Typedef":  # $Typedef(Gaussian, Gaus): another name for a class
            self.w._aliases[args[1]] = args[0]
            return True
        kind = self.w._aliases.get(kind, kind)
        operator = OPERATORS.get(kind)
        if operator is not None:
            return self.keep(operator(self, name or self._auto(), args))
        return self.keep(self._instance(kind, name or self._auto(), args))

    def _auto(self) -> str:
        """The name of an object made without one: the argument it is of, or ``gobj<n>``."""
        if self.prefix:
            return self.prefix[-1]
        while True:
            found = f"gobj{_GLOBAL[0]}"
            _GLOBAL[0] += 1
            if self.w.arg(found) is None:
                return found

    def _instance(self, kind: str, name: str, args: list[str]) -> Any:
        from .registry import find

        cls = find(kind)
        if cls is None:
            raise refuse(
                f"the factory has no class {kind}: RooFit's classes here are those of "
                "xrdroot.roofit.registry"
            )
        values = [self._argument(cls, name, index, arg) for index, arg in enumerate(args)]
        return cls(name, name, *values)

    def _argument(self, cls: Any, owner: str, index: int, text: str) -> Any:
        """An argument: an object - or an enumerator of the class, such as ``NoMirror``."""
        if re.match(r"^[A-Za-z_]\w*$", text) and self.w.obj(text) is None and hasattr(cls, text):
            return getattr(cls, text)
        return self.build_arg(owner, index, text)


def _category(name: str, text: str) -> Any:
    """``c[A=0,B=1]`` or ``c[A,B]``: a category and its states."""
    from .categories import RooCategory

    made = RooCategory(name, name)
    for part in split(text):
        label, _, index = part.partition("=")
        made.defineType(label.strip(), int(index) if index else None)
    return made
