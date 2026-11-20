"""What a library built on C++ adds to the language: the dialect a program is translated in.

The translator knows C++ and its standard library. A program written against
a larger library - one whose headers define typedefs, macros and classes the
program uses without declaring - is C++ in that library's *dialect*: names
spelled like its classes are types, its typedefs are the types they stand
for, its headers define its macros, and every name the program does not
declare is the library's, reached at run time through one namespace object
the generated Python calls the *host*.

A :class:`Dialect` says all of that as data, with a few small callables for
the patterns that need code - a loop the library can run all at once, a call
whose Python is not a call. :data:`CXX` is the default: plain C++, whose host
has the standard library and nothing else. A translation runs with one dialect
:func:`active`; :func:`using` makes another one active for a block, which is
how :func:`~xrdroot.pycxx.translation.translation` applies the one it is given
and how code that parses C++ itself reads it in a library's dialect.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import gtest_macros
from .runtime import CXX as _CXX_HOST
from .runtime import __all__ as _RUNTIME_NAMES

if TYPE_CHECKING:
    from .ctype import CType

__all__ = ["Dialect", "Prelude", "CXX", "active", "using", "NEVER"]

#: A pattern that matches nothing: what a dialect without such names has.
NEVER = re.compile(r"(?!)")

#: The emitter, as the hooks see it: every layer of :class:`~.emit_module.Translator`.
Emitter = Any

#: ``(emitter, function name, call)`` to the Python of a call and its precedence - or ``None``
#: to let the translator write it as it would.
CallHook = Callable[[Emitter, Any, Any], "tuple[str, int] | None"]


@dataclass(frozen=True)
class Prelude:
    """What including one of a library's headers defines: macros, and the ones refused.

    ``guard`` is the macro whose definition says the prelude was already run,
    so including a second header of the same library defines nothing twice.
    """

    text: str
    guard: str
    refused: Mapping[str, str] = field(default_factory=dict)
    #: The file the prelude's tokens are said to come from.
    file: str = "<prelude>"


#: GoogleTest, which is C++ rather than any library's: every dialect has it.
GTEST = Prelude(gtest_macros.PRELUDE, "TEST", gtest_macros.REFUSED, "<gtest>")


def _no_spelling(returns: Any) -> str | None:
    """A host that prints nothing for a returned value spells no return type."""
    return None


def _status(value: Any, spelled: str | None) -> int:
    """A program's entry function's value as its exit status: an integer's, else nothing."""
    return int(value) % 256 if isinstance(value, int) else 0


def _nothing_shown(value: Any, spelled: str | None) -> str | None:
    """A host that spells no return type prints no line for a returned value."""
    return None


def _no_session() -> AbstractContextManager[Any]:
    """A host with no settings of its own for a run: nothing to enter."""
    return nullcontext()


def _nothing_loaded(module: dict[str, Any], made: Any) -> None:
    """A host with no libraries of its own leaves a program's names as it defined them."""


def _nothing_parsed(unit: Any) -> Any:
    """Plain C++ has no host behind it, so a declaration parsed asks nothing of one."""
    return None


def _nothing_noted(notes: Any) -> None:
    """Plain C++ has no host keeping a table, so nothing a parse noted is given to one."""

@dataclass(frozen=True, eq=False)
class Dialect:
    """A library's additions to C++, by name - empty for plain C++ (:data:`CXX`)."""

    #: A short name, part of a cached translation's key.
    name: str = "c++"
    #: What a translation's first line says it was translated by.
    translator: str = "xrdroot.pycxx"

    # -- types -------------------------------------------------------------------------

    #: The library's typedefs, as the types they stand for: ``Real`` is ``double``.
    typedefs: Mapping[str, str] = field(default_factory=dict)
    #: Typedefs that are pointers to what a program never sees inside.
    pointer_typedefs: frozenset[str] = frozenset()
    #: What the name of one of the library's classes looks like, when nothing declared it.
    class_pattern: re.Pattern[str] = NEVER
    #: What one of the library's typedefs looks like, when nothing declared it.
    typedef_pattern: re.Pattern[str] = NEVER
    #: The library's namespaces whose members named with a capital are types.
    type_namespaces: frozenset[str] = frozenset()
    #: The library's templates, which begin a template argument list after them.
    templates: frozenset[str] = frozenset()
    #: The library's user-defined literal suffixes, which the runtime's ``user_literal`` makes.
    literal_suffixes: frozenset[str] = frozenset()
    #: How the library's classes are laid out, for ``sizeof`` and for a class derived from
    #: one: each class's size, alignment, data size (where a derived class's members may
    #: start; ``None`` if none may derive from it) and whether it has a virtual table.
    layouts: Mapping[str, tuple[int, int, int | None, bool]] = field(default_factory=dict)

    # -- the preprocessor --------------------------------------------------------------

    #: ``#define`` lines in force before a program's first line.
    predefined: str = ""
    #: The headers that define macros when included, and what each defines.
    preludes: Mapping[str, Prelude] = field(default_factory=dict)
    #: Headers that make a program one the translation cannot run, and what each makes it.
    out_of_scope: tuple[tuple[re.Pattern[str], str], ...] = ()

    # -- names -------------------------------------------------------------------------

    #: The Python name of the host: where a translation finds every name it did not declare.
    host: str = "CXX"
    #: The C++ namespace that is the host itself: ``Lib::Math::X`` as ``LIB.Math.X``.
    host_namespace: str | None = None
    #: The module a translation imports everything from, and the names it exports.
    runtime: str = "xrdroot.pycxx.runtime"
    runtime_names: frozenset[str] = frozenset(_RUNTIME_NAMES)
    #: The host object of the runtime, which a run binds to another namespace on request.
    host_object: Any = _CXX_HOST
    #: The library's constants that are Python's own: ``YES`` is ``True``.
    constants: Mapping[str, str] = field(default_factory=dict)
    #: The library's functions that are the runtime's under another name.
    functions: Mapping[str, str] = field(default_factory=dict)
    #: What a member of one of the library's classes looks like, used in a derived class.
    member_pattern: re.Pattern[str] = NEVER

    # -- calls and values --------------------------------------------------------------

    #: Calls of the library's functions written as other Python, by the function's name.
    calls: Mapping[str, CallHook] = field(default_factory=dict)
    #: The classes whose static functions :attr:`calls` also covers: ``X::name(...)``.
    call_classes: frozenset[str] = frozenset()
    #: A member call that changes its object, as the assignment it is in Python - or ``None``:
    #: ``(emitter, type of the object, its Python, method, arguments' Python)``.
    member_statement: Callable[[Emitter, Any, str, str, list[str]], str | None] | None = None
    #: The library's methods that write through references, by name and argument count.
    out_parameters: Mapping[tuple[str, int], tuple[int, ...]] = field(default_factory=dict)
    #: The library's free functions' return types: ``Form`` gives a ``char *``.
    returns: Mapping[str, CType] = field(default_factory=dict)
    #: Namespaces of numeric functions, each with its integer and its overloaded functions.
    numeric_namespaces: Mapping[str, tuple[frozenset[str], frozenset[str]]] = field(
        default_factory=dict
    )
    #: What the library's classes give back by value, by class and method.
    method_returns: Mapping[tuple[str, str], str] = field(default_factory=dict)
    #: Methods ``m<T>()`` that give a ``shared_ptr<T>`` a value is written through.
    field_makers: frozenset[str] = frozenset()
    #: Methods that leave their object as it was, so a loop bound using them is fixed.
    read_only: frozenset[str] = frozenset()
    #: The string classes a C string stored into one is converted to.
    string_classes: frozenset[str] = frozenset()
    #: The library's methods a ``char`` array handed to is memory to write into, or read from
    #: as it changes: such an array is a buffer (``cppstd.buffers``), not text.
    buffer_methods: frozenset[str] = frozenset()
    #: The library's members of a ``std::string`` - a string class's that one converts to.
    string_methods: Mapping[str, Callable[[str, list[str]], str]] = field(default_factory=dict)
    #: The classes whose objects do something as their scope ends.
    guards: frozenset[str] = frozenset()
    #: Classes ``R<T>`` whose ``*r`` is a ``T``, read as ``r.Get()``: value readers.
    value_readers: frozenset[str] = frozenset()
    #: Classes ``R<T>`` whose ``r[i]`` and ``r.At(i)`` are a ``T``: array readers.
    array_readers: frozenset[str] = frozenset()
    #: Classes whose ``*r`` is what they hold, read through the runtime's ``deref``.
    held_results: frozenset[str] = frozenset()
    #: Classes a variable of is set by ``=`` in place rather than bound anew, by last name.
    assigned_pattern: re.Pattern[str] = NEVER

    # -- vectors -----------------------------------------------------------------------

    #: The library's containers a braced list fills element by element.
    containers: frozenset[str] = frozenset()
    #: Those of them whose name gives their element type, built from a list as they are.
    typed_containers: frozenset[str] = frozenset()
    #: The library's vector class whose operators work element by element, by last name.
    vector: str | None = None
    #: The library's names for vectors of one element type: ``VecD`` is of ``double``.
    vector_aliases: Mapping[str, str] = field(default_factory=dict)
    #: The type comparing two such vectors makes.
    vector_bool: str | None = None
    #: The namespaces the vector and its aliases may be qualified with.
    vector_namespaces: frozenset[str] = frozenset()
    #: The spellings of the vector a returned local is elided for, as it is a ``std::vector``.
    vector_spellings: frozenset[str] = frozenset()

    # -- statements --------------------------------------------------------------------

    #: ``(emitter, for statement, counter, turns' Python)``: write a counted loop all at once.
    counted_loop: Callable[[Emitter, Any, Any, str], bool] | None = None
    #: ``(emitter, range for statement)``: write a range ``for`` the library's way.
    range_loop: Callable[[Emitter, Any], bool] | None = None

    # -- running -----------------------------------------------------------------------

    #: How the host names a function's return type when it prints the value - or ``None``.
    spelling: Callable[[Any], str | None] = _no_spelling
    #: The line printed for an entry function's value as spelled, and its exit status.
    shown: Callable[[Any, str | None], str | None] = _nothing_shown
    status: Callable[[Any, str | None], int] = _status
    #: What every run happens inside: the host's own settings for running a program.
    session: Callable[[], AbstractContextManager[Any]] = _no_session
    #: ``(module, translation)``: called once a program's Python has run and before anything
    #: in it is called - where a host binds the names a library it loaded already defines.
    loaded: Callable[[dict[str, Any], Any], None] = _nothing_loaded
    #: ``(unit)``: called with what a program declares, as it is parsed and before it runs -
    #: where a host keeping a table of declarations, as an interpreter does, notes them. What
    #: it returns is kept with the translation - plain JSON - and given to :attr:`noted`.
    parsed: Callable[[Any], Any] = _nothing_parsed
    #: ``(notes)``: what :attr:`parsed` returned, given to the host whether the translation was
    #: just made or read back from the cache - a cached translation is parsed not at all, so
    #: without this a program run twice would declare its enumerations only the first time.
    noted: Callable[[Any], None] = _nothing_noted
    #: Directories whose code shapes translations, hashed into a cached translation's key.
    sources: tuple[Path, ...] = ()

    @cached_property
    def all_typedefs(self) -> Mapping[str, str]:
        """The standard's typedefs and the library's, the library's last."""
        from .ctype import STD_TYPEDEFS

        return {**STD_TYPEDEFS, **self.typedefs}

    @cached_property
    def taken(self) -> frozenset[str]:
        """Names a program's own identifiers must not take: the runtime's."""
        return frozenset(self.runtime_names)

    def prelude(self, header: str) -> Prelude | None:
        """What including ``header`` defines: GoogleTest's for its headers, else the library's."""
        found = self.preludes.get(header)
        if found is None and header in gtest_macros.HEADERS:
            return GTEST
        return found


#: Plain C++: the standard library, and nothing past it.
CXX = Dialect()

_ACTIVE: ContextVar[Dialect] = ContextVar("pycxx_dialect", default=CXX)


def active() -> Dialect:
    """The dialect being translated in now: :data:`CXX` unless :func:`using` said otherwise."""
    return _ACTIVE.get()


@contextmanager
def using(dialect: Dialect) -> Iterator[Dialect]:
    """Translate in ``dialect`` until the ``with`` block ends."""
    token = _ACTIVE.set(dialect)
    try:
        yield dialect
    finally:
        _ACTIVE.reset(token)
