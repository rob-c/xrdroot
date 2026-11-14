"""``@ROOT.pythonization``: Python run on a C++ class when a script first looks it up.

A pythonizor is registered for class names (or prefixes of them) in a
namespace, and is called with the class - and, if it takes two parameters,
its qualified name - the first time each matching class is looked up:
``ROOT.MyClass``, ``ROOT.NS.MyClass``, ``cppyy.gbl.MyClass`` or a template
instantiated, ``ROOT.MyTemplate['int']``. A class looked up before the
pythonizor was registered is pythonized at once, as cppyy does for the
classes it has bound already. A class template is no class of its own until
it is instantiated, so only its instances are pythonized.

Classes a C++ translation declares are given ``__cpp_name__`` - their name
with its namespaces - and, being templates or not, ``ROOT.X['int']``: the
class of the instance, made once, ``X<int>``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["pythonization"]

#: Where the classes a translation declares are defined.
TRANSLATED = "__cint__"


@dataclass(eq=False)
class Pythonizor:
    """One registered pythonizor: its namespace, names, and the user's function."""

    space: str
    targets: list[str]
    prefix: bool
    function: Callable[..., Any]
    parameters: int
    #: The classes it ran on, so it runs once on each.
    done: set[int] = field(default_factory=set)

    def matches(self, name: str) -> bool:
        if self.prefix:
            return any(name.startswith(target) for target in self.targets)
        return name in self.targets

    def apply(self, klass: type, name: str) -> None:
        if id(klass) in self.done or not self.matches(name):
            return
        self.done.add(id(klass))
        if self.parameters == 1:
            self.function(klass)
        else:
            self.function(klass, klass.__cpp_name__)  # type: ignore[attr-defined]


#: The pythonizors, in the order they were registered.
PYTHONIZORS: list[Pythonizor] = []
#: The classes looked up so far, by namespace and name, as cppyy keeps those it bound.
SEEN: dict[tuple[str, str], type] = {}


def pythonization(class_name: Any, ns: str = "::", is_prefix: bool = False) -> Any:
    """``@pythonization('MyClass', ns='NS', is_prefix=False)``: the decorator registering a
    pythonizor, ``f(klass)`` or ``f(klass, name)``."""
    targets = _targets(class_name)
    space = ns[:-2] if ns.endswith("::") else ns

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        made = Pythonizor(space, targets, is_prefix, function, _parameters(function))
        PYTHONIZORS.append(made)
        for (where, name), klass in list(SEEN.items()):
            if where == space:
                made.apply(klass, name)
        return function

    return decorate


def _targets(class_name: Any) -> list[str]:
    """The class names or prefixes, each without a namespace, said once each."""
    given: Iterable[Any] = [class_name] if isinstance(class_name, str) else class_name
    names: list[str] = []
    for name in given:
        if not isinstance(name, str):
            raise TypeError(
                'Invalid type of "target" argument in @pythonization: must be string or '
                "iterable of strings"
            )
        if "::" in name:
            raise ValueError(
                f'Invalid value of "class_name" argument in @pythonization: namespace '
                f'definition found ("{name}"). Please use the "ns" parameter to specify the '
                f"namespace"
            )
        if name not in names:
            names.append(name)
    return names


def _parameters(function: Callable[..., Any]) -> int:
    count = len(inspect.getfullargspec(function).args)
    if count not in (1, 2):
        raise TypeError(
            f"Pythonizor function {function.__name__} has a wrong number of parameters "
            f"({count}). Allowed parameters are the class to be pythonized and (optionally) "
            f"its name."
        )
    return count


def looked_up(klass: Any, name: str, space: str = "") -> Any:
    """``klass``, looked up by ``name`` in namespace ``space``: given its C++ name, and the
    template brackets if a translation declared it, then pythonized."""
    if not isinstance(klass, type) or name.startswith("_"):
        return klass
    if klass.__module__ == TRANSLATED:
        if "__cpp_name__" not in vars(klass):
            klass.__cpp_name__ = f"{space}::{name}" if space else name  # type: ignore[attr-defined]
        if "__class_getitem__" not in vars(klass):
            klass.__class_getitem__ = classmethod(_instance)  # type: ignore[attr-defined]
    SEEN[(space, name)] = klass
    for each in PYTHONIZORS:
        if each.space == space:
            each.apply(klass, name)
    return klass


def _spelled(argument: Any) -> str:
    """A template argument as C++ spells it: a Python ``float`` is ``double``, as cppyy has it."""
    if isinstance(argument, str):
        return argument
    if argument is float:
        return "double"
    return str(getattr(argument, "__cpp_name__", getattr(argument, "__name__", argument)))


def _instance(template: type, arguments: Any) -> type:
    """``Template[args]``: the class of that instance, a ``Template`` by its C++ name - which,
    being instantiated, is pythonized while the template itself is no longer."""
    given = arguments if isinstance(arguments, tuple) else (arguments,)
    cpp = template.__cpp_name__  # type: ignore[attr-defined]
    space, _, bare = cpp.rpartition("::")
    name = f"{bare}<{','.join(_spelled(each) for each in given)}>"
    made: dict[str, type] = template.__dict__.get("_instances_", {})
    if name not in made:
        members = {"__cpp_name__": f"{space}::{name}" if space else name, "__module__": TRANSLATED}
        made[name] = type(name, (template,), members)
        template._instances_ = made  # type: ignore[attr-defined]
    SEEN.pop((space, bare), None)
    return looked_up(made[name], name, space)  # type: ignore[no-any-return]
