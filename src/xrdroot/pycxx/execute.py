"""Running translated C++: a program from its entry point, a source's declarations, a line.

A program is translated (or its translation found in the cache), the Python
run, and then started as its first file says: a function named like the
file, called with the arguments given - an unnamed ``{ ... }`` block is that
function - else ``main(argc, argv)``, else the GoogleTest tests the files
define. What the program raises as it runs is raised again as a
:class:`MacroError` at the C++ line it happened on.

Each function takes the :class:`~.dialect.Dialect` to translate and run in
(the active one when none is given): the dialect's host is what a run binds
another namespace to, its ``session`` what every run happens inside, and its
``shown`` and ``status`` what an entry function's value is printed as and
exits with.
"""

from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from . import cache
from .dialect import Dialect, active
from .errors import MacroError, Refusal, Where
from .runtime import cerr, clog, cout
from .runtime.namespace import DECLARED
from .runtime.streams import FORMATTERS, ostream
from .translation import Translation, translation

__all__ = [
    "run_source",
    "run_program",
    "run_line",
    "translated",
    "placed",
    "located",
    "entry",
    "label",
    "MODULE",
]

#: The name of the module a translation runs as: how a function it defined is told apart.
MODULE = "__cint__"


def _reset_streams() -> None:
    """Every run starts with ``cout`` as C++ starts it: six digits, no ``fixed``, no writers."""
    FORMATTERS.clear()
    fresh = ostream()
    for stream in (cout, cerr, clog):
        for name in ("floatfield", "adjust", "alpha", "base", "plus", "width", "digits", "fill"):
            setattr(stream, name, getattr(fresh, name))


def translated(source: str, file: str, dialect: Dialect, use_cache: bool) -> Translation:
    """The translation of ``source`` in ``dialect``: from the cache if allowed and kept there."""
    found = cache.load(source, file, dialect) if use_cache else None
    if found is not None:
        return found
    made = translation(source, file, [Path(file).parent], dialect)
    if use_cache:
        cache.save(source, file, made, dialect)
    return made


@contextmanager
def placed(made: list[Translation]) -> Iterator[None]:
    """Errors raised in the translated code of any of ``made``, re-raised at their C++ line."""
    try:
        yield
    except (MacroError, Refusal, SystemExit, KeyboardInterrupt):
        raise
    except Exception as why:
        where = located(why, made)
        raise MacroError(f"{type(why).__name__}: {why}", Where(*where) if where else None) from why


def label(made: Translation) -> str:
    """The file name a translation's Python is compiled under: how :func:`located` finds it."""
    return f"<translation of {made.file}>"


def located(why: BaseException, made: list[Translation]) -> tuple[str, int] | None:
    """The C++ line ``why`` was raised at: of the innermost translated line it passed through."""
    labels = {label(each): each for each in made}
    deepest: tuple[Translation, int] | None = None
    trace: types.TracebackType | None = why.__traceback__
    while trace is not None:
        each = labels.get(trace.tb_frame.f_code.co_filename)
        if each is not None:
            deepest = each, trace.tb_lineno
        trace = trace.tb_next
    return None if deepest is None else deepest[0].source_map.get(deepest[1])


def run_source(
    source: str,
    file: str = "<macro>",
    args: tuple[Any, ...] = (),
    *,
    dialect: Dialect | None = None,
    namespace: Any = None,
    use_cache: bool = False,
    call: bool = True,
) -> Any:
    """Translate and run a program's text; call its function (or unnamed block) with ``args``.

    With ``namespace``, the host looks the names the program did not declare up
    there rather than in its own; with ``call`` false, nothing is called and
    the module's names are what is handed back.
    """
    chosen = dialect or active()
    made = translated(source, file, chosen, use_cache)
    module: dict[str, Any] = {"__name__": MODULE, "__file__": file}
    binding = chosen.host_object.bind(namespace) if namespace is not None else nullcontext()
    _reset_streams()
    with chosen.session(), binding:
        with placed([made]):
            exec(compile(made.python, label(made), "exec"), module)
        chosen.loaded(module, made)
        _declare(module)
        if not call:
            return module
        function = entry(made, module, file, args)
        if function is None:
            return None
        with placed([made]):
            result = function(*args)
    sys.stdout.flush()
    return result


def _declare(module: dict[str, Any]) -> None:
    """Keep what a program declared at its top where later lines of C++ look for it."""
    DECLARED.update({name: value for name, value in module.items()
                     if getattr(value, "__module__", None) == MODULE})  # fmt: skip


def entry(made: Translation, module: dict[str, Any], file: str, args: tuple[Any, ...]) -> Any:
    """The function running the program means - refusing arguments nothing would take."""
    function = module.get(made.entry) if made.entry else None
    if args and made.unnamed:
        raise TypeError(f"{Path(file).name} is an unnamed macro, which takes no arguments")
    if args and function is None:
        raise TypeError(
            f"{Path(file).name} defines no function {Path(file).stem}() "
            f"to hand {', '.join(map(repr, args))} to"
        )
    return function


def run_program(
    paths: Sequence[str | Path],
    args: tuple[Any, ...] = (),
    argv: Sequence[str] = (),
    *,
    dialect: Dialect | None = None,
    use_cache: bool = True,
) -> int:
    """The files as one program, run from its entry point; its exit status.

    Each file is translated and run, in order, in one namespace; then the
    program is started as the first file says. A function named like the
    first file is called with ``args``, its value printed as the dialect
    prints it and exited with - unless it needs arguments none were given and
    there is a ``main``; else ``main(argc, argv)``, its value the exit
    status, as a compiled program's; else the GoogleTest tests the files
    define, as ``gtest_main`` runs them; else - nothing to run - a refusal.
    ``argv`` is the program's whole command line, its name first.
    """
    from .runtime import gtest

    chosen = dialect or active()
    wheres = [Path(path) for path in paths]
    module: dict[str, Any] = {"__name__": MODULE, "__file__": str(wheres[0])}
    gtest.reset()
    _reset_streams()
    try:
        with chosen.session():
            made = [_loaded(where, module, chosen, use_cache) for where in wheres]
            _declare(module)
            gtest.REGISTRY.locate = lambda why: located(why, made)
            return _started(made, module, chosen, tuple(args), list(argv) or [wheres[0].stem])
    finally:
        sys.stdout.flush()


def _loaded(where: Path, module: dict[str, Any], dialect: Dialect, use_cache: bool) -> Translation:
    """Translate the file at ``where`` and run what it declares into ``module``."""
    source = where.read_text(encoding="utf-8", errors="replace")
    made = translated(source, str(where), dialect, use_cache)
    with placed([made]):
        exec(compile(made.python, label(made), "exec"), module)
    dialect.loaded(module, made)
    return made


def _started(
    made: list[Translation],
    module: dict[str, Any],
    dialect: Dialect,
    args: tuple[Any, ...],
    argv: list[str],
) -> int:
    """Start the program its first file names: its function, ``main``, or its tests."""
    from .runtime import gtest

    first = made[0]
    main = _own_main(module)
    function = _starting(entry(first, module, first.file, args), main, args)
    if function is not None:
        with placed(made):
            value = function(*args)
        line = dialect.shown(value, first.returns)
        if line is not None:
            sys.stdout.write(line + "\n")
        return dialect.status(value, first.returns)
    if main is not None:
        return _main(main, made, argv)
    if gtest.registered():
        with placed(made):
            return gtest.run_all(argv)
    raise MacroError(
        f"{Path(first.file).name} has nothing to run: it defines no function "
        f"{Path(first.file).stem}(), no main(), no GoogleTest TEST, and it is no unnamed "
        "macro { ... }, which are what xrdroot run looks for, in that order"
    )


def _own_main(module: dict[str, Any]) -> Any:
    """The program's own ``main``, or ``None``."""
    main = module.get("main")
    return main if callable(main) and getattr(main, "__module__", None) == MODULE else None


def _starting(function: Any, main: Any, args: tuple[Any, ...]) -> Any:
    """The function named like the file, unless it needs arguments none were given and the
    program has a ``main``: then it is a compiled program's helper, and ``main`` starts it."""
    if function is not None and not args and main is not None and _needs_arguments(function):
        return None
    return function


def _needs_arguments(function: Any) -> bool:
    """Whether ``function`` has a parameter with no default, which a call with none lacks."""
    parameters = inspect.signature(function).parameters.values()
    return any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
               for p in parameters)  # fmt: skip


def _main(main: Any, made: list[Translation], argv: list[str]) -> int:
    """``main(argc, argv)`` - or ``main()`` - called as a program's: its value the exit status."""
    given = (len(argv), argv)[: len(inspect.signature(main).parameters)]
    with placed(made):
        value = main(*given)
    return 0 if value is None else int(value) % 256


def run_line(
    line: str, *, file: str = "<line>", dialect: Dialect | None = None, namespace: Any = None
) -> Any:
    """C++ statements run at once - and the value of one expression handed back, as an
    interpreter's prompt hands it back."""
    chosen = dialect or active()
    text = line.strip()
    body = text if text.endswith((";", "}")) else text + ";"
    source = _valued(body, file, chosen) or "{\n" + body + "\n}\n"
    return run_source(source, file, dialect=chosen, namespace=namespace)


def _valued(body: str, file: str, dialect: Dialect) -> str | None:
    """A block handing back the value of ``body``, if it is one expression - ``f(1)`` hands
    back what the call returns - or ``None``."""
    if body.count(";") != 1 or not body.endswith(";") or "{" in body:
        return None
    source = "{\n" + "return (" + body[:-1] + ");\n}\n"
    try:
        translation(source, file, [Path(".")], dialect)
    except Refusal:  # a declaration, a statement: no value
        return None
    return source
